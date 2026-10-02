"""可表示性审计的数学见证、失败证书、归档身份与校准隔离性质。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from run_reward_audit import main
from src.reward_audit import CONSTRAINTS, audit_calibration, inequality_system, separation_audit
from src.reward_audit_experiment import calibration_payload, freeze_reward_audit, run_reward_audit, validate_config
from src.session_experiment import fingerprint
from src.snapshot_irl import pooled_policy_statistics
from test_snapshot_irl import config_for_test, policy_row


def audit_config():
    """精度在拟合前声明，两种额外约束与仅等式集合全部保留。"""
    return dict(schema_version=1, purpose='development_reward_representability_audit',
                constraints=list(CONSTRAINTS), verification_tolerance=1e-9, solver_feasibility_tolerance=1e-9)


def verify_evidence(test, row):
    """从输出矩阵/身份重新构造不等式，不能仅断言求解器返回成功。"""
    mu = np.array(row['expectations']);expert = row['policy_ids'].index(row['expert_policy_id'])
    others = [i for i in range(len(mu)) if i != expert]
    A, b, labels = inequality_system(mu[expert]-mu[others], [row['policy_ids'][i] for i in others],
                                    row['constraint'], row['reward_tolerance_price'])
    test.assertEqual(labels, row['inequality_rows'])
    if row['representable']:
        w = np.array(row['witness']['weights'])
        test.assertAlmostEqual(w.sum(), 1.)
        test.assertTrue((A@w-b <= row['verification_tolerance']).all())
        test.assertTrue(row['witness']['verified'])
    elif row['representable'] is False:
        proof = row['infeasibility_certificate'];y = np.array(proof['multipliers']);z = proof['equality_multiplier']
        test.assertTrue((y >= -row['verification_tolerance']).all())
        np.testing.assert_allclose(A.T@y+z, 0., atol=row['verification_tolerance'], rtol=0)
        test.assertAlmostEqual(b@y+z, -1.)
        test.assertTrue(proof['verified'])


class SeparationTests(unittest.TestCase):
    """用手算矩阵区分约束失效与原始 Eq.(3) 下的结构性失败。"""
    def test_box_failure_but_sum_only_has_minimum_L1_witness(self):
        """现金0与候选[2,1]：需要 w1<=-1、w2>=2，盒不允许，仅等式允许。"""
        mu = [[0., 0.], [2., 1.]]
        for name in CONSTRAINTS:
            row = separation_audit(mu, 0, ['cash', 'trade'], name)
            verify_evidence(self, row)
            self.assertEqual(row['representable'], name == 'sum_only')
            self.assertFalse(row['usable_for_execution'])
            if name == 'sum_only':
                np.testing.assert_allclose(row['witness']['weights'], [-1., 2.], atol=3e-8)
                self.assertAlmostEqual(row['witness']['L1'], 3., places=6)
                self.assertGreater(row['witness']['max_abs_weight'], 1.)

    def test_constant_positive_competitor_impossible_under_any_sum_one_weight(self):
        """现金0对[1,1]，无论正负权得分恒1，三种约束都有可核验矛盾。"""
        for name in CONSTRAINTS:
            row = separation_audit([[0., 0.], [1., 1.]], 0, ['cash', 'trade'], name)
            self.assertFalse(row['representable']);verify_evidence(self, row)

    def test_nonnegative_constraint_is_distinct_from_box_constraint(self):
        """现金0对[1,1,10]，负权可以解释；非负约束不能。"""
        rows = [policy_row('cash', 0., cash=True), policy_row('trade', -10., [1., 1., 10.])]
        r = audit_calibration(rows, [1000, 3000, 9000], config_for_test(), audit_config())
        self.assertEqual(r['diagnosis'], 'nonnegative_constraint_excludes_witness')
        self.assertEqual(r['legacy_fits']['simplex']['status'], 'fitted_expert_not_representable')
        for row in r['audits'].values():verify_evidence(self, row)

    def test_convex_mixture_requires_ties_and_all_identical_features_remain_uninformative(self):
        """专家[1,1]处于[2,0]/[0,2]之间可并列，不要求伪造严格正间隔。"""
        row = separation_audit([[1., 1.], [2., 0.], [0., 2.]], 0, ['expert', 'a', 'b'], 'sum_only')
        self.assertTrue(row['representable']);verify_evidence(self, row)
        rows = [policy_row('cash', 0., cash=True), policy_row('trade', -1., [0., 0.])]
        r = audit_calibration(rows, [1000, 3000], config_for_test(), audit_config())
        self.assertEqual(r['diagnosis'], 'uninformative_expectations')
        self.assertTrue(all(a['representable'] for a in r['audits'].values()))
        self.assertEqual(r['legacy_fits']['simplex']['status'], 'blocked_uninformative_expectations')
        self.assertFalse(r['usable_for_execution'])

    def test_unobserved_profitable_expert_is_not_replaced(self):
        """最佳净利专家没成熟单就阻断；不能把较差但有观察的模型换成专家。"""
        rows = [policy_row('cash', 0., cash=True), policy_row('best', 20., count=0),
                policy_row('observed', -1., [1., -1.])]
        r = audit_calibration(rows, [1000, 3000], config_for_test(), audit_config())
        self.assertEqual(r['expert']['selected_policy_id'], 'best')
        self.assertEqual(r['status'], 'blocked_expert_reward_unobserved');self.assertEqual(r['audits'], {})

    def test_solver_status_and_bad_certificate_do_not_count_as_proofs(self):
        """虚假可行解、未验证的不可行证书和数值失败均保持未定义。"""
        args = ([[0., 0.], [1., 1.]], 0, ['cash', 'trade'], 'sum_only')
        fake = SimpleNamespace(success=True, status=0, message='fake', x=np.array([.5, .5, .5, .5]))
        with patch('src.reward_audit.linprog', return_value=fake):r = separation_audit(*args)
        self.assertEqual(r['status'], 'numerically_unverified');self.assertIsNone(r['representable'])
        fake.x[:] = np.nan
        with patch('src.reward_audit.linprog', return_value=fake):r = separation_audit(*args)
        self.assertEqual(r['status'], 'numerically_unverified');self.assertIsNone(r['witness'])
        failure = SimpleNamespace(success=False, status=2, message='infeasible')
        bad_proof = SimpleNamespace(success=True, status=0, message='fake', x=np.array([1., 0.]))
        with patch('src.reward_audit.linprog', side_effect=[failure, bad_proof]):r = separation_audit(*args)
        self.assertEqual(r['status'], 'infeasibility_unverified');self.assertIsNone(r['representable'])
        with patch('src.reward_audit.linprog', return_value=SimpleNamespace(success=False, status=4, message='numeric')):
            r = separation_audit(*args)
        self.assertEqual(r['status'], 'solver_failed');self.assertIsNone(r['representable'])

    def test_invalid_units_dimensions_identity_and_tolerances_are_rejected(self):
        """拒绝非有限矩阵、重复候选身份、错误专家与 HiGHS 不支持的精度。"""
        valid = ([[0., 0.], [2., 1.]], 0, ['cash', 'trade'], 'sum_only')
        invalid = [([[0., float('nan')]], 0, ['cash'], 'sum_only'),
                   ([[0., 0.]], 1, ['cash'], 'sum_only'),
                   ([[0.], [1.]], 0, ['same', 'same'], 'sum_only'),
                   ([[0.]], 0, ['cash'], 'unbounded_margin')]
        for args in invalid:
            with self.assertRaises(ValueError):separation_audit(*args)
        with self.assertRaises(ValueError):separation_audit(*valid, solver_feasibility_tolerance=1e-12)
        with self.assertRaises(ValueError):validate_config(audit_config() | {'constraints': ['sum_only']})


class AuditEntryTests(unittest.TestCase):
    """归档合成账本不需要大行情，验证重算摘要、旧身份、冻结输入与后段隔离。"""
    def setUp(self):
        self.tmp = TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root = Path(self.tmp.name)
        self.config = self.root/'audit.json';self.config.write_text(json.dumps(audit_config()))
        specs = [dict(policy_id='cash', cash=True, threshold=.01, source='fixed_reference'),
                 dict(policy_id='trade', cash=False, threshold=.01, source='weekly_library')]
        bindings, cases = [], []
        for age, mean, count in [(500, [0., 0.], 0), (1000, [1., 1.], 2), (2000, [2., 1.], 2)]:
            def account(name, matured, expectation, cost):
                return dict(strategy=name, total_trades=1 if cost else 0, total_fills=2 if cost else 0,
                    terminal_position_liquidated=True, gross_pnl_usd=0., friction_usd=cost,
                    net_pnl_usd=-cost, matured_order_count=matured,
                    unmatured_fill_rewards_at_end=2-matured if cost else 0, order_feature_expectation=expectation)
            daily = [dict(session_id='2025-10-01', results=[account('cash', 0, [0., 0.], 0.), account('trade', count, mean, 10.)])]
            binding = dict(protocol=dict(sessions=dict(calibration=['2025-10-01']), reward_horizons_ms=[1000, 3000]))
            binding['plan_sha256'] = fingerprint(binding)
            bindings.append(dict(max_age_ms=age, session_binding=binding))
            cases.append(dict(max_age_ms=age, phases=dict(calibration=dict(daily=daily,
                statistics=pooled_policy_statistics(daily, specs, [1000, 3000])), validation={}, test={})))
        plan = dict(irl_config=config_for_test() | {'age_candidates_ms': [500, 1000, 2000]}, policies=specs,
                    cases=bindings, code_sha256={'archived_source.py': 'historical_source_identity'})
        plan['plan_sha256'] = fingerprint(plan)
        self.archive = dict(schema_version=1, result_kind='library_time_OE_IRL', plan=plan,
                            plan_sha256=plan['plan_sha256'], age_cases=cases)
        self.path = self.root/'archive.json';self.save()

    def save(self):self.path.write_text(json.dumps(self.archive))

    def freeze(self):
        return main(['freeze', '--config', str(self.config), '--input', str(self.path),
                     '--output-dir', str(self.root/'frozen')])

    def test_cli_preserves_all_cases_proofs_and_archived_identity(self):
        """校准500阻断，1000仅等式亦失败，2000仅等式可行；旧专家和输入不改。"""
        before = hashlib.sha256(self.path.read_bytes()).hexdigest();plan = self.freeze()
        result = main(['run', '--plan', str(self.root/'frozen/plan.json'), '--output-dir', str(self.root/'result')])
        self.assertEqual(result, json.loads((self.root/'result/result.json').read_text()))
        self.assertEqual(result, run_reward_audit(self.root/'frozen/plan.json'))
        self.assertEqual(before, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(plan['calibration']['archive_code_sha256'], self.archive['plan']['code_sha256'])
        expected = ['insufficient_observation_not_geometric_failure', 'not_representable_even_with_sum_only', 'signed_box_bound_excludes_witness']
        self.assertEqual([c['audit']['diagnosis'] for c in result['age_cases']], expected)
        for case in result['age_cases']:
            self.assertFalse(case['audit']['usable_for_execution'])
            self.assertEqual(case['audit']['expert']['selected_policy_id'], 'cash')
            self.assertFalse(case['audit']['expert_identity_in_library_policy_set'])
            for row in case['audit']['audits'].values():verify_evidence(self, row)
        self.assertIn('最小 L1', (self.root/'result/report.md').read_text())
        with self.assertRaises(FileExistsError):self.freeze()

    def test_future_validation_and_test_values_cannot_affect_calibration_evidence(self):
        """后两段任意换价格/策略表现不改变诊断；输入字节身份当然随文件改变。"""
        plan = self.freeze();a = run_reward_audit(self.root/'frozen/plan.json')
        for case in self.archive['age_cases']:
            case['phases']['validation'] = {'prices': [1e10], 'winner': 'future_perfect'}
            case['phases']['test'] = {'cash': 1e30, 'expectations': [[-1e30, 1e30]]}
        self.save();p = freeze_reward_audit(self.config, self.path)
        second = self.root/'second-plan.json';second.write_text(json.dumps(p));b = run_reward_audit(second)
        self.assertNotEqual(plan['archived_result_source']['sha256'], p['archived_result_source']['sha256'])
        self.assertEqual(a['age_cases'], b['age_cases'])
        self.assertEqual(plan['calibration'], p['calibration'])

    def test_summary_archive_plan_and_age_tampering_is_rejected(self):
        """摘要必须与校准每日订单统计吻合，源计划/日期/完整年龄也不能伪造。"""
        original = deepcopy(self.archive)
        mutations = [lambda a:a['age_cases'][1]['phases']['calibration']['statistics'][1].update(order_feature_expectation=[0., 0.]),
                     lambda a:a['plan'].update(plan_sha256='bad'),
                     lambda a:a['age_cases'].pop(),
                     lambda a:a['age_cases'][0]['phases']['calibration']['daily'][0].update(session_id='2025-10-03')]
        for mutate in mutations:
            self.archive = deepcopy(original);mutate(self.archive);self.save()
            with self.assertRaises(ValueError):calibration_payload(self.path)

    def test_frozen_source_input_and_payload_integrity_guards(self):
        """冻结后改代码、输入字节或校准载荷都拒绝；不可复用旧身份算新数据。"""
        plan = self.freeze();path = self.root/'frozen/plan.json'
        with patch('src.reward_audit_experiment.code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'integrity'):run_reward_audit(path)
        self.path.write_text(self.path.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'input changed'):run_reward_audit(path)
        self.save();plan['calibration']['cases'][0]['statistics'][0]['net_pnl_usd'] = 100.
        plan['plan_sha256'] = fingerprint(plan);path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, 'payload changed'):run_reward_audit(path)
