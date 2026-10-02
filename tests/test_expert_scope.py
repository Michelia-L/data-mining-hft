"""专家来源对照的因果隔离、原控制一致性与失败保留测试。"""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from run_expert_scope import main
from src.config import BASE_DIR
from src.expert_scope import calibration_payload, fit_scope, freeze_comparison, run_comparison
from src.session_experiment import fingerprint
from src.snapshot_irl import learn_calibration_reward, pooled_policy_statistics
from test_snapshot_irl import config_for_test, policy_row


def policies():
    """现金是库外对照，a/b为原在线动作；顺序同时定义并列消歧。"""
    return [dict(policy_id=n, cash=n == 'cash', threshold=.01,
                 source='fixed_reference' if n == 'cash' else 'weekly_library')
            for n in ('cash', 'a', 'b')]


class ExpertScopeTests(unittest.TestCase):
    """小矩阵直接验证方法性质，不以真实数据盈利作为测试通过标准。"""
    def setUp(self):
        self.rows = [policy_row('cash', 0., cash=True), policy_row('a', -10., [2., 0.]),
                     policy_row('b', -20., [0., 2.])]

    def fit(self, scope='online_library'):
        return fit_scope(self.rows, [1000, 3000], config_for_test(), policies(), scope)

    def test_default_and_explicit_all_preserve_exact_control(self):
        """新增参数默认不改变老规则；全范围显式声明与原 API 的所有数值完全相同。"""
        group = self.fit('all_candidates')
        for c in ('simplex', 'signed_box'):
            self.assertEqual(group['reward_fits'][c],
                learn_calibration_reward(self.rows, [1000, 3000], config_for_test(), c))
        self.assertEqual(group['reward_fits']['sum_only']['expert']['selected_policy_id'], 'cash')
        self.assertIn('expert_outside_online_library', group['execution_gates']['sum_only']['reasons'])

    def test_negative_best_library_expert_keeps_full_oracle(self):
        """库内亏损最小者可以作为工程专家，但不叫盈利；现金仍留在同一比较集合。"""
        group = self.fit()
        for fit in group['reward_fits'].values():
            self.assertEqual(fit['expert']['selected_policy_id'], 'a')
            self.assertEqual(fit['expert']['best_score'], -10.)
            self.assertFalse(fit['profitable_trading_expert'])
            self.assertEqual(fit['eligible_policy_ids'], ['cash', 'a', 'b'])
            self.assertEqual(fit['status'], 'fitted')
            self.assertAlmostEqual(sum(fit['weights']), 1.)
        self.assertTrue(group['execution_gates']['sum_only']['usable_for_execution'])
        self.assertIsNone(group['active_reward_weights'])

    def test_external_competitor_can_still_make_library_expert_unrepresentable(self):
        """a=[-1,-1]在sum(w)=1下永远低于现金0；不能删除现金制造成功。"""
        self.rows[1]['order_feature_expectation'] = [-1., -1.]
        self.rows[2]['order_feature_expectation'] = [-2., -2.]
        fit = self.fit()['reward_fits']['sum_only']
        self.assertEqual(fit['status'], 'fitted_expert_not_representable')
        self.assertAlmostEqual(fit['diagnostics']['optimal_margin'], -1.)
        self.assertIsNone(fit['selected_policy'])

    def test_unobserved_best_expert_never_falls_back_to_observed_runner_up(self):
        """先选最佳净利a再检查OE，a无成熟订单时不得暗换成有成熟订单的b。"""
        self.rows[1].update(matured_order_count=0, order_feature_expectation=None,
                            order_feature_expectation_defined=False)
        group = self.fit()
        for fit in group['reward_fits'].values():
            self.assertEqual(fit['expert']['selected_policy_id'], 'a')
            self.assertEqual(fit['status'], 'blocked_expert_reward_unobserved')
            self.assertIsNone(fit['weights'])
        self.assertEqual(group['execution_gates']['sum_only']['missing_online_policy_ids'], ['a'])

    def test_declared_order_ties_and_unsettled_library_experts(self):
        """范围列表顺序不重排候选，全部并列保留；没有结算专家时不回退现金。"""
        self.rows[2]['net_pnl_usd'] = -10.
        fit = learn_calibration_reward(self.rows, [1000, 3000], config_for_test(), 'simplex',
                                       expert_policy_ids=['b', 'a'])
        self.assertEqual(fit['expert']['tied_best_policy_ids'], ['a', 'b'])
        self.assertEqual(fit['expert']['selected_policy_id'], 'a')
        self.rows[1]['pnl_aggregation_defined'] = self.rows[2]['pnl_aggregation_defined'] = False
        for fit in self.fit()['reward_fits'].values():
            self.assertEqual(fit['status'], 'blocked_no_settled_expert')
            self.assertIsNone(fit['expert'])

    def test_malformed_expert_scope_rejected(self):
        """未知、重复或空范围直接拒绝，不能悄悄解释成默认全范围。"""
        for ids in ([], ['missing'], ['a', 'a']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                learn_calibration_reward(self.rows, [1000, 3000], config_for_test(), 'simplex',
                                         expert_policy_ids=ids)
        for scope in ('best_observed', ''):
            with self.assertRaises(ValueError): self.fit(scope)


class ExpertArchiveTests(unittest.TestCase):
    """两日不同成熟订单数的归档经真实CLI冻结/拟合，无需行情才能重算校准对照。"""
    def setUp(self):
        temp = TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.config = BASE_DIR / 'config/esz5_expert_scope_comparison.json'
        specs = policies(); days = ['2025-10-06', '2025-10-07']; daily = []
        for day, count in zip(days, (2, 3)):
            rows = []
            for spec, mu, pnl in zip(specs, ([0., 0.], [2., 0.], [0., 2.]), (0., -10., -20.)):
                n = 0 if spec['cash'] else count
                rows.append(dict(strategy=spec['policy_id'], total_trades=n, total_fills=n*2,
                    terminal_position_liquidated=True, gross_pnl_usd=pnl+2*n, friction_usd=2*n,
                    net_pnl_usd=pnl, matured_order_count=n, unmatured_fill_rewards_at_end=n,
                    order_feature_expectation=mu))
            daily.append(dict(session_id=day, results=rows))
        config = config_for_test(); horizons = [1000, 3000]
        stats = pooled_policy_statistics(daily, specs, horizons)
        group = fit_scope(stats, horizons, config, specs, 'all_candidates')
        binding = dict(protocol=dict(sessions=dict(calibration=days), reward_horizons_ms=horizons))
        binding['plan_sha256'] = fingerprint(binding)
        base = dict(plan_kind='frozen_library_time_OE_IRL', irl_config=config, policies=specs,
                    cases=[dict(max_age_ms=60000, session_binding=binding)])
        base['plan_sha256'] = fingerprint(base)
        plan = dict(plan_kind='frozen_multiweek_OE_readiness', base_oe_plan=base, code_sha256={})
        plan['plan_sha256'] = fingerprint(plan)
        case = dict(max_age_ms=60000, calibration=dict(daily=daily, statistics=stats),
            reward_fits=group['reward_fits'], execution_gate=group['execution_gates']['sum_only'],
            phases=dict(validation=dict(net_pnl_usd=123.), test=dict(net_pnl_usd=456.)))
        case['calibration_sha256'] = fingerprint(dict(statistics=stats,
            reward_fits=case['reward_fits'], execution_gate=case['execution_gate']))
        self.artifact = dict(schema_version=1, result_kind='multiweek_OE_readiness_development',
            plan=plan, plan_sha256=plan['plan_sha256'], age_cases=[case])
        self.archive = self.root / 'archive.json'; self.write_archive()

    def write_archive(self):
        self.archive.write_text(json.dumps(self.artifact), encoding='utf-8')

    def freeze(self):
        plan = freeze_comparison(self.config, self.archive)
        path = self.root / 'plan.json'; path.write_text(json.dumps(plan), encoding='utf-8')
        return path

    def test_cli_reproducibility_both_scopes_and_no_overwrite(self):
        """冻结/运行另存产物，原控制完全复现、两组均保留且不激活权重。"""
        main(['freeze', '--config', str(self.config), '--input', str(self.archive),
              '--output-dir', str(self.root / 'frozen')])
        args = ['run', '--plan', str(self.root / 'frozen/plan.json'), '--output-dir', str(self.root / 'result')]
        result = main(args)
        self.assertEqual(result, json.loads((self.root / 'result/result.json').read_text()))
        self.assertEqual(result, run_comparison(self.root / 'frozen/plan.json'))
        self.assertIn('online_library', (self.root / 'result/report.md').read_text())
        case = result['age_cases'][0]
        self.assertTrue(case['full_candidate_control_reproduced'])
        self.assertFalse(case['dynamic_backtest_run'])
        self.assertFalse(result['formal_replication_ready'])
        self.assertIsNone(result['selected_expert_scope'])
        for scope in case['scopes'].values(): self.assertIsNone(scope['active_reward_weights'])
        with self.assertRaises(FileExistsError): main(args)

    def test_future_metrics_never_affect_calibration_payload_or_fits(self):
        """任意改后段收益和质量不改校准载荷与拟合；仍需另冻全文件新身份。"""
        original = calibration_payload(self.archive)
        first = run_comparison(self.freeze())
        self.artifact['age_cases'][0]['phases'] = dict(validation='discarded', test=dict(net_pnl_usd=-1e99))
        self.artifact['age_cases'][0]['readiness'] = dict(ready_for_development_online_run=True)
        self.write_archive()
        self.assertEqual(original, calibration_payload(self.archive))
        second = run_comparison(self.freeze())
        self.assertNotEqual(first['plan_sha256'], second['plan_sha256'])
        self.assertEqual(first['age_cases'], second['age_cases'])

    def test_wrong_summary_missing_duplicate_candidate_and_wrong_day_rejected(self):
        """日账本不完整、混入后段日期或手改聚合均值均不能用于新专家。"""
        original = deepcopy(self.artifact)
        for mode in ('summary', 'missing', 'duplicate', 'day'):
            self.artifact = deepcopy(original); cal = self.artifact['age_cases'][0]['calibration']
            if mode == 'summary': cal['statistics'][1]['order_feature_expectation'][0] += 1
            if mode == 'missing': cal['daily'][0]['results'].pop()
            if mode == 'duplicate': cal['daily'][0]['results'].append(cal['daily'][0]['results'][0])
            if mode == 'day': cal['daily'][0]['session_id'] = '2025-10-20'
            self.write_archive()
            with self.subTest(mode=mode), self.assertRaises(ValueError): calibration_payload(self.archive)

    def test_archive_source_and_nested_plan_tampering_rejected(self):
        """冻结后源码、输入字节或嵌套计划变化必须另存新计划，不能复用旧身份。"""
        path = self.freeze()
        with patch('src.expert_scope.code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'plan/source'): run_comparison(path)
        self.archive.write_text(self.archive.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'input changed'): run_comparison(path)
        self.artifact['plan']['base_oe_plan']['irl_config']['minimum_matured_orders'] += 1
        self.artifact['plan']['plan_sha256'] = fingerprint(self.artifact['plan'])
        self.artifact['plan_sha256'] = self.artifact['plan']['plan_sha256']; self.write_archive()
        with self.assertRaisesRegex(ValueError, 'plan integrity'): calibration_payload(self.archive)

    def test_archived_fit_must_reproduce_even_with_recomputed_identity(self):
        """只修改旧专家并重算摘要哈希仍被实际重新拟合发现，不能以假控制作对照。"""
        case = self.artifact['age_cases'][0]
        case['reward_fits']['simplex']['expert']['best_score'] = 999.
        case['calibration_sha256'] = fingerprint(dict(statistics=case['calibration']['statistics'],
            reward_fits=case['reward_fits'], execution_gate=case['execution_gate']))
        self.write_archive()
        with self.assertRaisesRegex(ValueError, 'control differs'): run_comparison(self.freeze())


if __name__ == '__main__':
    unittest.main()
