"""检查时间 OE IRL 的订单分母、专家可表示性及跨阶段信息隔离。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from run_snapshot_irl import main
from src.snapshot_irl import (evaluate_frozen_weights, freeze_time_irl, learn_calibration_reward,
    pooled_policy_statistics, replay_stage, run_time_irl, validate_irl_config)
from src.session_experiment import fit_streaming_ridge
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import prepare_snapshot_dataset
import test_session_experiment as session_fixture


def config_for_test():
    """数值容差、有限策略及现金消歧在校准前固定，最少一单仅保证可计算。"""
    return dict(schema_version=1, purpose='development_time_OE_IRL', session_protocol_file='protocol.json',
        age_candidates_ms=[60000], threshold_multipliers=[1, 2, 4],
        weight_constraints=['simplex', 'signed_box'], minimum_matured_orders=1,
        solver_tolerance=1e-8, solver_max_iter=50, tie_tolerance_usd=1e-9, tie_tolerance_reward=1e-9)


def policy_row(name, pnl, vector=None, cash=False, count=1):
    """手算有限策略特征，现金均值保持 None，不伪造零成熟订单。"""
    return dict(policy_id=name, cash=cash, threshold=.01, pnl_aggregation_defined=True,
                net_pnl_usd=pnl, matured_order_count=0 if cash else count,
                order_feature_expectation=vector, order_feature_expectation_defined=vector is not None)


class CalibrationRewardTests(unittest.TestCase):
    """数学与失败语义独立于模型预测，避免只测试输出字段。"""
    def test_order_weighted_pooling_and_cash_mean_undefined(self):
        """两日有不同成熟订单数，正确均值为 4.2/5.2，而非日均值 3/4。"""
        daily = []
        for day, count, mean in [('d1', 2, [1., 2.]), ('d2', 8, [5., 6.])]:
            def row(name, number, vector):
                return dict(strategy=name, total_trades=number, total_fills=number*2,
                    terminal_position_liquidated=True, gross_pnl_usd=0., friction_usd=0.,
                    net_pnl_usd=0., matured_order_count=number, unmatured_fill_rewards_at_end=number,
                    order_feature_expectation=vector)
            daily.append(dict(session_id=day, results=[row('trade', count, mean), row('cash', 0, [0., 0.])]))
        specs = [dict(policy_id='trade', cash=False, threshold=.01), dict(policy_id='cash', cash=True, threshold=.01)]
        statistics = pooled_policy_statistics(daily, specs, [1000, 3000])
        np.testing.assert_allclose(statistics[0]['order_feature_expectation'], [4.2, 5.2])
        self.assertEqual(statistics[0]['matured_fraction_of_fills'], .5)
        self.assertIsNone(statistics[1]['order_feature_expectation'])

    def test_profitable_expert_fit_and_weight_sum(self):
        """有可表示专家时，其奖励不低于有限库其他策略；权重和满足 Eq.(3)。"""
        rows = [policy_row('cash', 0., cash=True), policy_row('a', 100., [2., 0.]),
                policy_row('b', 80., [0., 2.])]
        fit = learn_calibration_reward(rows, [1000, 3000], config_for_test(), 'simplex')
        self.assertEqual(fit['status'], 'fitted')
        self.assertTrue(fit['profitable_trading_expert'])
        self.assertAlmostEqual(sum(fit['weights']), 1.)
        self.assertTrue(all(w >= 0 for w in fit['weights']))
        self.assertGreaterEqual(fit['calibration_policy_scores']['a']+1e-8, fit['calibration_policy_scores']['b'])
        self.assertFalse(fit['reward_uniquely_identified'])

    def test_cash_expert_nonrepresentability_and_signed_sensitivity(self):
        """净利专家是现金，但正价格差在非负权下可能解释不了现金；不得称成功。"""
        rows = [policy_row('cash', 0., cash=True), policy_row('a', -10., [1., 1., 10.]),
                policy_row('b', -20., [2., 2., 20.])]
        positive = learn_calibration_reward(rows, [1000, 3000, 9000], config_for_test(), 'simplex')
        self.assertEqual(positive['status'], 'fitted_expert_not_representable')
        self.assertTrue(positive['diagnostics']['converged'])
        self.assertFalse(positive['diagnostics']['expert_representable'])
        self.assertIsNone(positive['selected_policy'])
        signed = learn_calibration_reward(rows, [1000, 3000, 9000], config_for_test(), 'signed_box')
        self.assertEqual(signed['status'], 'fitted')
        self.assertTrue(any(w < 0 for w in signed['weights']))
        self.assertAlmostEqual(sum(signed['weights']), 1.)
        self.assertEqual(signed['selected_policy']['selected_policy_id'], 'cash')
        measured = evaluate_frozen_weights([rows[0]], {'signed': signed})[0]
        self.assertIsNone(measured['frozen_weight_order_rewards']['signed'])

    def test_zero_orders_unobserved_expert_and_zero_information_fail_explicitly(self):
        """没有成熟交易不可由现金零向量冒充学习；不可观测的最佳专家不换次优。"""
        config = config_for_test()
        cash = policy_row('cash', 0., cash=True)
        empty = policy_row('empty', -10., count=0)
        fit = learn_calibration_reward([cash, empty], [1000, 3000], config, 'simplex')
        self.assertEqual(fit['status'], 'blocked_no_matured_trading_policy')
        self.assertIsNone(fit['weights'])
        empty['net_pnl_usd'] = 100.
        fit = learn_calibration_reward([cash, empty, policy_row('observed', 50., [1., 2.])],
                                      [1000, 3000], config, 'simplex')
        self.assertEqual(fit['expert']['selected_policy_id'], 'empty')
        self.assertEqual(fit['status'], 'blocked_expert_reward_unobserved')
        fit = learn_calibration_reward([cash, policy_row('a', -10., [0., 0.])], [1000, 3000], config, 'simplex')
        self.assertEqual(fit['status'], 'blocked_uninformative_expectations')

    def test_ties_failures_and_iteration_exhaustion_are_preserved(self):
        """净利并列现金优先；求解失败及未收敛保留诊断，不发布可用策略。"""
        rows = [policy_row('cash', 0., cash=True), policy_row('a', 0., [1., -1.]),
                policy_row('b', -1., [-1., 1.])]
        fit = learn_calibration_reward(rows, [1000, 3000], config_for_test(), 'simplex')
        self.assertEqual(fit['expert']['tied_best_policy_ids'], ['cash', 'a'])
        self.assertEqual(fit['expert']['selected_policy_id'], 'cash')
        with patch('src.snapshot_irl.IRLRewardLearner.fit_reward_weights', side_effect=RuntimeError('solver unavailable')):
            failed = learn_calibration_reward(rows, [1000, 3000], config_for_test(), 'simplex')
        self.assertEqual(failed['status'], 'solver_failed')
        self.assertIn('unavailable', failed['error'])
        short = learn_calibration_reward(rows, [1000, 3000], config_for_test() | {'solver_max_iter': 1}, 'simplex')
        self.assertEqual(short['status'], 'solver_not_converged')
        self.assertIsNone(short['selected_policy'])


class SnapshotIRLIntegrationTests(unittest.TestCase):
    """复用正式 prepared 合成夹具，但不继承其测试，避免重复执行无关用例。"""
    prepare = session_fixture.SessionExperimentTests.prepare

    def setUp(self):
        session_fixture.SessionExperimentTests.setUp(self)
        self.irl_config = self.root / 'irl.json'
        self.irl_config.write_text(json.dumps(config_for_test()))

    def frozen(self, dataset=None, name='irl-plan'):
        plan = main(['freeze', '--config', str(self.irl_config), '--dataset-dirs', str(dataset or self.dataset),
                     '--output-dir', str(self.root / name)])
        return plan, self.root / name / 'plan.json'

    def test_cli_stage_isolation_frozen_selection_and_reproducibility(self):
        """验证/测试只使用校准选择，全部行情和现金零订单状态如实保留。"""
        plan, path = self.frozen()
        result = main(['run', '--plan', str(path), '--output-dir', str(self.root / 'result')])
        self.assertEqual(result, json.loads((self.root / 'result/result.json').read_text()))
        self.assertEqual(result, run_time_irl(path))
        self.assertIsNone(result['selected_age_ms'])
        case = result['age_cases'][0]
        for phase in ('calibration', 'validation', 'test'):
            data = case['phases'][phase]
            self.assertEqual(len(data['statistics']), len(plan['policies']))
            for row in data['daily'][0]['results']:
                self.assertEqual(row['replay_rows'], data['horizon_audits'][0]['observed_rows'])
            if phase != 'calibration':
                for name, selection in data['calibration_frozen_selections'].items():
                    fit = case['reward_fits'][name]
                    self.assertEqual(selection['selected_policy_id'], fit['selected_policy']['selected_policy_id']
                                     if fit['selected_policy'] else None)
                cash = data['statistics'][0]
                self.assertIsNone(cash['order_feature_expectation'])
                self.assertTrue(all(v is None for v in cash['frozen_weight_order_rewards'].values()))
        self.assertIn('时间尺度 OE IRL', (self.root / 'result/report.md').read_text())
        with self.assertRaises(FileExistsError):
            main(['run', '--plan', str(path), '--output-dir', str(self.root / 'result')])

    def test_future_prices_do_not_change_fitted_model_or_calibration_weights(self):
        """改变验证/测试价格可改变其盈亏，却不能改变模型、专家、权重或冻结选择。"""
        _, first = self.frozen()
        original = run_time_irl(first)['age_cases'][0]
        rows = self.raw_rows.copy()
        mask = rows.ts_event >= pd.Timestamp('2025-10-01T22:00:00Z')
        for name in rows:
            if name.startswith(('bid_px_', 'ask_px_')):
                rows.loc[mask, name] += 100.
        changed = self.prepare(rows, 'future-prices')
        _, second = self.frozen(changed, 'changed-plan')
        modified = run_time_irl(second)['age_cases'][0]
        self.assertEqual(original['model'], modified['model'])
        self.assertEqual(original['reward_fits'], modified['reward_fits'])
        self.assertEqual(original['phases']['calibration'], modified['phases']['calibration'])

    def test_future_labels_do_not_gate_orders_or_reward_learning(self):
        """离线标签缓存被抹去不能删决策、改变成交或抹去根据实价成熟的 OE。"""
        _, first = self.frozen()
        original = run_time_irl(first)['age_cases'][0]
        other = self.prepare(self.raw_rows, 'censored-labels')
        path = other / 'snapshots.parquet'
        table = pq.read_table(path)
        frame = table.to_pandas()
        mask = frame.session_id >= self.days[2]
        for name in frame:
            if name.startswith('label_valid_'):
                frame.loc[mask, name] = False
            elif name.startswith(('future_return_', 'future_price_diff_')):
                frame.loc[mask, name] = np.nan
        frame.loc[mask, 'learning_ready'] = False
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(table.schema.metadata), path)
        _, second = self.frozen(other, 'censored-plan')
        self.assertEqual(original, run_time_irl(second)['age_cases'][0])

    def test_linear_rescore_matches_replay_and_does_not_change_static_fills(self):
        """静态策略的权重重评满足线性均值恒等式，与用新权重直接回放一致。"""
        from src.irl_reward import IRLRewardLearner
        from src.model_selector import SingleModelSelector
        from src.session_experiment import read_day
        from src.time_execution import TimeExecutionEngine
        from src.timed_snapshots import SNAPSHOT_SCHEMA
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        scaler, model, _ = fit_streaming_ridge(reader, self.days[:2], 60000, 3660000)
        specs = [dict(policy_id='ridge', cash=False, threshold=.000015)]
        daily, _ = replay_stage(reader, self.days[2:3], scaler, model, specs, self.protocol, detail=True)
        base = daily[0]['results'][0]
        frame = read_day(reader, self.days[2])
        valid = frame.feature_valid.to_numpy()
        preds = np.full((len(frame), 1), np.nan)
        from src.snapshot_backtest import FEATURE_COLUMNS
        preds[valid, 0] = model.predict(scaler.transform(frame.loc[valid, FEATURE_COLUMNS].to_numpy(float)))
        weights = [.2, .3, .5]
        reward = IRLRewardLearner(horizons=self.protocol['reward_horizons_ms'], weights=weights,
                                 definition='paper_price_difference')
        selector = SingleModelSelector('ridge', [model]); selector.reward_type = 'OE'
        direct = TimeExecutionEngine(reward, interval_ms=60000, latency_ms=60000,
            holding_review_ms=180000, force_replay_end=False).run_backtest(selector,
                frame[SNAPSHOT_SCHEMA.names+['session_id', 'segment_id', 'mid_price', 'feature_valid']], preds, detail=True)
        self.assertEqual(base['fills'], direct['fills'])
        self.assertEqual(base['net_pnl_usd'], direct['net_pnl_usd'])
        statistics = pooled_policy_statistics(daily, specs, reward.horizons)
        score = evaluate_frozen_weights(statistics, {'new': {'weights': weights}})[0]['frozen_weight_order_rewards']['new']
        self.assertAlmostEqual(score, np.mean([r['reward'] for r in direct['reward_observations']]))

    def test_frozen_plan_guards_and_undeclared_or_mixed_inputs(self):
        """绑定候选数量、真实年龄、原始来源与源码；数据变更必须重冻结。"""
        for override in [dict(threshold_multipliers=[2, 4]), dict(minimum_matured_orders=0),
                         dict(weight_constraints=['signed_box']), dict(tie_tolerance_reward=float('nan'))]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_irl_config(config_for_test() | override)
        with self.assertRaisesRegex(ValueError, 'every declared age'):
            freeze_time_irl(self.irl_config, [self.dataset, self.dataset])
        plan, path = self.frozen()
        changed = deepcopy(plan); changed['policies'][1]['threshold'] *= 2
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            run_time_irl(path)
        path.write_text(json.dumps(plan))
        with patch('src.snapshot_irl.irl_code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                run_time_irl(path)
        quality = self.dataset / 'quality.json';quality.write_text(quality.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            run_time_irl(path)

    def test_age_comparison_requires_identical_raw_sources(self):
        """年龄变化可用相同原始消息；伪装另一批原始消息的哈希必须被拒绝。"""
        config = config_for_test() | {'age_candidates_ms': [60000, 120000]}
        self.irl_config.write_text(json.dumps(config))

        def prepare_age(name, source_hash):
            directory = self.root / name
            directory.mkdir()
            paths = []
            for path in sorted((self.root / 'original').glob('*.parquet')):
                table = pq.read_table(path)
                metadata = table.schema.metadata | {
                    b'max_age_ms': b'120000', b'source_sha256': source_hash.encode()}
                target = directory / path.name
                pq.write_table(table.replace_schema_metadata(metadata), target)
                paths.append(target)
            output = directory / 'prepared'
            prepare_snapshot_dataset(paths, output, horizons_ms=(60000, 180000))
            return output

        same = prepare_age('same-source', 'a' * 64)
        plan = freeze_time_irl(self.irl_config, [same, self.dataset])
        self.assertEqual([r['max_age_ms'] for r in plan['cases']], [60000, 120000])
        different = prepare_age('different-source', 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'raw-source provenance'):
            freeze_time_irl(self.irl_config, [self.dataset, different])


if __name__ == '__main__':
    unittest.main()
