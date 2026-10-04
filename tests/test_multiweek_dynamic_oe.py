"""真实多周动态联动的门控、冻结、未来隔离、完整期间和版本性质测试。"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_multiweek_dynamic_oe import main
from src.ars_experiment import replay_ars_day
from src.multiweek_dynamic_oe import (ReplayCheckpoints, audit_dynamic_result, fit_groups, freeze_dynamic,
    replay_scope, run_dynamic, strategy_names, summarize_phase)
from src.session_experiment import fingerprint
import test_history_ars as ars_fixture
import test_learned_oe as learned_fixture
from test_expert_scope import policies
from test_snapshot_irl import config_for_test, policy_row
from test_time_execution import BASE, prepared


class DynamicScopeTests(unittest.TestCase):
    """负净利库内专家能进入回放，而现金或缺成熟专家仍保持原阻断。"""
    def setUp(self):
        self.fixture = learned_fixture.LearnedReplayTests(); self.fixture.setUp()
        self.rows = [policy_row('cash', 0., cash=True), policy_row('a', -10., [2., 0.]),
                     policy_row('b', -20., [0., 2.])]
        self.groups = fit_groups(self.rows, [1000, 3000], config_for_test(), policies())

    def replay(self, scope, frame=None, group=None):
        f = self.fixture
        with patch('src.learned_oe_experiment.read_day', return_value=prepared() if frame is None else frame):
            return replay_scope(f.reader, '2025-09-22', f.case, f.protocol, f.plan,
                                self.groups[scope] if group is None else group, detail=True)

    def test_two_scopes_preserve_blocking_and_negative_expert_activation(self):
        """全候选现金不自动加入动作；库内亏损专家可执行，不给阻断组伪造0收益。"""
        blocked = self.replay('all_candidates')
        self.assertTrue(all(r['net_pnl_usd'] is None and r['run_status'] == 'blocked' for r in blocked))
        with patch('src.multiweek_dynamic_oe.replay_learned_day', side_effect=AssertionError('must not trade')):
            self.replay('all_candidates')
        rows = self.replay('online_library')
        self.assertEqual([r['strategy'] for r in rows], strategy_names('online_library'))
        for r in rows:
            self.assertEqual(r['run_status'], 'executed')
            self.assertEqual(r['reward_weights'], self.groups['online_library']['active_reward_weights'])
            audit_dynamic_result(r, self.fixture.versions, 3000, r['reward_weights'])
        self.rows[1].update(matured_order_count=0, order_feature_expectation=None,
                            order_feature_expectation_defined=False)
        g = fit_groups(self.rows, [1000, 3000], config_for_test(), policies())['online_library']
        self.assertIsNone(g['active_reward_weights'])
        self.assertTrue(all(r['run_status'] == 'blocked' for r in self.replay('online_library', group=g)))

    def test_future_prices_and_labels_cannot_change_earlier_dynamic_choices(self):
        """通过新组接口回放；40秒后价格和离线标签变化不得改变更早决策/反馈。"""
        stamps = list(pd.date_range(BASE, periods=150, freq='500ms'))
        prices = 100. + .25 * np.arange(150)
        a = prepared(stamps, prices); prices[81:] += 100.; b = prepared(stamps, prices)
        for col in b:
            if col.startswith('future_'): b[col] = np.nan
        first = self.replay('online_library', a); second = self.replay('online_library', b)
        cutoff = BASE.value + 40_000_000_000
        for x, y in zip(first, second):
            for key, clock in [('period_selections', 'selected_at_ns'), ('period_feedback', 'observed_at_ns')]:
                self.assertEqual([p for p in x[key] if p[clock] <= cutoff],
                                 [p for p in y[key] if p[clock] <= cutoff])

    def test_audit_rejects_stale_versions_early_rewards_and_lost_orders(self):
        """独立审计必须发现提前成熟、错版、订单丢失及偷偷替换学习权重。"""
        result = self.replay('online_library')[0]
        audit_dynamic_result(result, self.fixture.versions, 3000, result['reward_weights'])
        changed = deepcopy(result); changed['total_fills'] += 1
        with self.assertRaisesRegex(ValueError, 'conserve fills'):
            audit_dynamic_result(changed, self.fixture.versions, 3000)
        changed = deepcopy(result); changed['period_selections'][0]['model_version_id'] = 'future'
        with self.assertRaisesRegex(ValueError, 'future or stale'):
            audit_dynamic_result(changed, self.fixture.versions, 3000)
        changed = deepcopy(result)
        p = next(p for p in changed['period_feedback'] if p['updated_selector'])
        p['observed_at_ns'] = p['last_origin_ns']
        with self.assertRaisesRegex(ValueError, 'maturity'):
            audit_dynamic_result(changed, self.fixture.versions, 3000)
        with self.assertRaisesRegex(ValueError, 'frozen reward'):
            audit_dynamic_result(result, self.fixture.versions, 3000, [3., -2.])

    def test_history_audit_checks_visible_envelope_and_weighted_mean(self):
        """ARS历史均值必须来自完整成熟订单并使用同一冻结权重。"""
        result = self.replay('online_library')[2]
        audited = audit_dynamic_result(result, self.fixture.versions, 3000)
        self.assertGreater(audited['scored_windows'], 0)
        changed = deepcopy(result)
        e = next(e for p in changed['period_selections'] for e in p['evaluations']
                 if e['source'] == 'independent_history_replay' and e['score'] is not None)
        e['score'] += 10.
        with self.assertRaisesRegex(ValueError, 'other weights'):
            audit_dynamic_result(changed, self.fixture.versions, 3000)
        e['reward_observation_latest_required_ns'] = e['known_through_ns'] + 1
        with self.assertRaises(ValueError):
            audit_dynamic_result(changed, self.fixture.versions, 3000)

    def test_blocked_summary_is_none_and_mixed_gate_is_rejected(self):
        """汇总不能把整段阻断变为现金0，也不能在后段看到结果后改变校准门控。"""
        blocked = self.replay('all_candidates')[0]
        daily = [dict(results=[blocked]), dict(results=[deepcopy(blocked)])]
        self.assertIsNone(summarize_phase(daily, [blocked['strategy']])[0]['net_pnl_usd'])
        daily[1]['results'][0]['run_status'] = 'executed'
        with self.assertRaisesRegex(ValueError, 'gate changed'):
            summarize_phase(daily, [blocked['strategy']])


class DynamicExperimentTests(unittest.TestCase):
    """七session真实建库夹具验证两个评价周、冻结CLI和全部原控制一致。"""
    def setUp(self):
        self.fixture = ars_fixture.ARSExperimentTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups); self.f = self.fixture.f
        self.library, _, _ = self.f.freeze()
        self.config = dict(schema_version=1, purpose='development_multiweek_dynamic_OE',
            ars_protocol_file='ars.json', expert_scopes=['all_candidates', 'online_library'],
            active_constraint='sum_only', minimum_evaluation_weekly_versions=2)
        self.path = self.f.f.root / 'dynamic.json'; self.path.write_text(json.dumps(self.config))

    def test_complete_cli_controls_calibration_freeze_and_actual_two_week_execution(self):
        """先拟合一次再评价所有日，两组权重不重学，原18个控制逐字段不变。"""
        frozen = self.f.f.root / 'dynamic-frozen'; output = self.f.f.root / 'dynamic-result'
        checkpoints = self.f.f.root / 'checkpoints'
        plan = main(['freeze', '--config', str(self.path), '--library', str(self.f.libpath), '--output-dir', str(frozen)])
        self.assertEqual(len(plan['strategies']), 24)
        with patch('src.multiweek_dynamic_oe.fit_groups', wraps=fit_groups) as fit:
            result = main(['run', '--plan', str(frozen / 'plan.json'), '--output-dir', str(output),
                           '--checkpoint-dir', str(checkpoints)])
            self.assertEqual(fit.call_count, 1)
        self.assertEqual(result, json.loads((output / 'result.json').read_text()))
        self.assertIn('实际动态周版本', (output / 'report.md').read_text())
        self.assertFalse(result['formal_replication_ready'])
        # 已完成的日账本必须能真正跳过昂贵回放，同时重算审计，且最终结果逐字段相同。
        with patch('src.multiweek_dynamic_oe.replay_candidates', side_effect=AssertionError('unexpected calibration replay')), \
                patch('src.multiweek_dynamic_oe.replay_ars_day', side_effect=AssertionError('unexpected dynamic replay')):
            restored = run_dynamic(frozen / 'plan.json', checkpoint_dir=checkpoints)
        self.assertEqual(result, restored)
        case = result['age_cases'][0]
        for name, coverage in case['dynamic_version_coverage'].items():
            self.assertTrue(coverage['all_evaluation_versions_selected'], name)
            self.assertEqual(len(coverage['selected_version_ids']), 2)
        for phase in ('validation', 'test'):
            self.assertEqual(case['phases'][phase]['calibration_sha256'], case['calibration_sha256'])
            for day in case['phases'][phase]['daily']:
                self.assertEqual(len(day['results']), 24)
                for r in day['results'][18:]:
                    self.assertEqual(r['reward_weights'], case['calibration']['scopes'][r['expert_scope']]['active_reward_weights'])
        # 原控制单独执行一天，确保编排接入不改变其实际账本、奖励或选择。
        from src.library_oe import load_library
        from src.snapshot_backtest import PreparedDatasetReader
        lib, calendar = load_library(self.f.libpath)
        ars = plan['base_ars_plan']; binding = ars['base_period_plan']['cases'][0]['session_binding']
        reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar)
        day = binding['protocol']['sessions']['validation'][0]
        old = replay_ars_day(reader, day, lib['age_cases'][0], binding['protocol'], ars)
        self.assertEqual(old['results'], case['phases']['validation']['daily'][0]['results'][:18])
        with self.assertRaises(FileExistsError):
            main(['run', '--plan', str(frozen / 'plan.json'), '--output-dir', str(output)])

    def test_freeze_rejects_missing_scopes_wrong_constraint_or_missing_evaluation_week(self):
        """不能只选成功来源/权重；只有校准跨周而评价不跨周也不符合本任务。"""
        for key, value in [('expert_scopes', ['online_library']), ('active_constraint', 'simplex'),
                           ('minimum_evaluation_weekly_versions', 3), ('minimum_evaluation_weekly_versions', True)]:
            config = self.config | {key: value}; self.path.write_text(json.dumps(config))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                freeze_dynamic(self.path, self.f.libpath)

    def test_source_parent_schedule_and_library_tampering_are_rejected(self):
        """源码、父计划、实际评价周和模型字节都须与冻结身份一致。"""
        original = freeze_dynamic(self.path, self.f.libpath); p = self.f.f.root / 'dynamic-plan.json'
        p.write_text(json.dumps(original))
        with patch('src.multiweek_dynamic_oe.code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'integrity'):
                run_dynamic(p)
        changed = deepcopy(original); changed['base_ars_plan']['config']['history_window_ms'] += 1000
        changed['plan_sha256'] = fingerprint(changed); p.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            run_dynamic(p)
        changed = deepcopy(original); changed['expected_evaluation_versions'] = {}
        changed['plan_sha256'] = fingerprint(changed); p.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'schedule'):
            run_dynamic(p)
        p.write_text(json.dumps(original)); self.f.libpath.write_text(self.f.libpath.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            run_dynamic(p)


class CheckpointTests(unittest.TestCase):
    """长实验恢复不能混用计划/明细/环境，损坏结果也不能伪装为可用缓存。"""
    def test_identity_payload_and_no_overwrite(self):
        """完整单位可恢复且不重复计算；错误身份、损坏内容和重复发布明确拒绝。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checks'; cache = ReplayCheckpoints(root, 'plan-a', False)
            self.assertIsNone(cache.load('day'))
            payload = dict(profit=-10., weights=[2., -1.])
            cache.save('day', payload)
            self.assertEqual(ReplayCheckpoints(root, 'plan-a', False).load('day'), payload)
            with self.assertRaises(FileExistsError): cache.save('day', payload)
            for plan, detail in [('plan-b', False), ('plan-a', True)]:
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    ReplayCheckpoints(root, plan, detail)
            with patch('src.multiweek_dynamic_oe.environment_versions', return_value={}):
                with self.assertRaisesRegex(ValueError, 'binding differs'):
                    ReplayCheckpoints(root, 'plan-a', False)
            path = root / 'day' / 'checkpoint.json'; row = json.loads(path.read_text())
            row['payload']['profit'] = 10.; path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'integrity differs'): cache.load('day')

    def test_incomplete_publication_is_not_a_valid_day(self):
        """半份或非本入口的目录不能当成已有结果；禁用检查点仍保持原内存行为。"""
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checks'; cache = ReplayCheckpoints(root, 'plan', False)
            (root / 'day').mkdir()
            with self.assertRaisesRegex(ValueError, 'Incomplete checkpoint'): cache.load('day')
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                ReplayCheckpoints(Path(tmp), 'plan', False)
            memory = ReplayCheckpoints(None, 'plan', False)
            memory.save('day', {'value': 1})
            self.assertIsNone(memory.load('day'))


if __name__ == '__main__':
    unittest.main()
