"""学习目标、动作空间、成熟反馈及冻结时序的性质测试，不用盈利判断复现。"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_learned_oe import main
from src.learned_oe_experiment import (LEARNED_STRATEGIES, blocked_result, execution_gate,
    freeze_learned_experiment, replay_learned_day, run_learned_experiment)
from src.snapshot_dataset import SessionCalendar
from src.snapshot_irl import learn_calibration_reward
from src.sum_only_reward import FrozenSumOnlyReward, fit_sum_only, learn_sum_only
import test_history_ars as ars_fixture
from test_history_ars import versions_for_test
from test_snapshot_irl import config_for_test, policy_row
from test_time_execution import BASE, prepared


def learned(rows):
    """手工校准矩阵同时走原资格/专家规则和新仅等式求解，完全不传后续阶段。"""
    cfg = config_for_test()
    reference = learn_calibration_reward(rows, [1000, 3000], cfg, 'signed_box')
    return learn_sum_only(rows, [1000, 3000], cfg, reference)


class SumOnlyLearningTests(unittest.TestCase):
    """完整 oracle 含专家，将间隔封顶0；不靠盒边界或事后范数预算防无界。"""
    def test_large_signed_weights_preserve_optimal_margin_without_box(self):
        """现金对[1,2]、[1.1,2.1]要求w1>=2.1，最小L1解[2.1,-1.1]。"""
        w, d = fit_sum_only([[0.,0.], [1.,2.], [1.1,2.1]], 0)
        np.testing.assert_allclose(w, [2.1,-1.1], atol=1e-10)
        self.assertAlmostEqual(d['optimal_margin'], 0.)
        self.assertAlmostEqual(d['weight_l1'], 3.2)
        self.assertTrue(d['expert_representable'])
        self.assertLessEqual(max(d['primary_verification'].values()), 1e-8)
        self.assertLessEqual(max(d['secondary_verification'].values()), 1e-8)

    def test_infeasible_expert_has_negative_optimum_not_false_success(self):
        """对手常向量0.25在sum=1下恒得0.25，现金最优间隔只能是-0.25。"""
        w, d = fit_sum_only([[0.,0.], [.25,.25]], 0)
        self.assertAlmostEqual(sum(w), 1.)
        self.assertAlmostEqual(d['optimal_margin'], -.25)
        self.assertFalse(d['expert_representable'])
        fit = learned([policy_row('cash',0.,cash=True), policy_row('a',-10.,[.25,.25])])
        self.assertEqual(fit['status'], 'fitted_expert_not_representable')
        self.assertIsNone(fit['selected_policy'])

    def test_oracle_includes_expert_and_preserves_ties(self):
        """即使允许无限正负权，含专家自身的完整目标不会无界或伪造严格领先。"""
        rows = [policy_row('cash',0.,cash=True), policy_row('a',100.,[2.,0.]),
                policy_row('b',50.,[2.,0.])]
        fit = learned(rows)
        self.assertEqual(fit['status'], 'fitted')
        self.assertAlmostEqual(fit['diagnostics']['optimal_margin'],0.)
        self.assertIn('a',fit['candidate_response']['tied_best_policy_ids'])
        self.assertIn('b',fit['candidate_response']['tied_best_policy_ids'])
        self.assertFalse(fit['reward_uniquely_identified'])

    def test_missing_expert_or_all_trading_observations_remain_blocked(self):
        """盈利专家缺长期标签不能被现金或次优交易策略替换。"""
        rows = [policy_row('cash',0.,cash=True), policy_row('a',100.,None,count=0),
                policy_row('b',50.,[1.,2.])]
        self.assertEqual(learned(rows)['status'],'blocked_expert_reward_unobserved')
        self.assertEqual(learned(rows[:2])['status'],'blocked_no_matured_trading_policy')

    def test_invalid_matrix_and_frozen_weights_are_rejected(self):
        """权重不静默归一化/裁剪，冻结后不能重学；坏维度和非有限值直接拒绝。"""
        for mu, i in [([[np.nan,0.]],0), ([[0.,0.]],2), ([],0)]:
            with self.assertRaises(ValueError):fit_sum_only(mu,i)
        for w in ([2.,0.], [np.nan,1.], [1.]):
            with self.assertRaises(ValueError):FrozenSumOnlyReward([1000,3000],w)
        reward = FrozenSumOnlyReward([1000,3000],[3.,-2.])
        self.assertEqual(reward.score(np.array([2.,4.])), -2.)
        with self.assertRaises(RuntimeError):reward.fit_reward_weights([[0.,0.]],0)


class LearnedReplayTests(unittest.TestCase):
    """手工可解释校准→真实执行器/历史回放连通；合成模型不冒充正式训练收益。"""
    def setUp(self):
        self.calendar = SessionCalendar(); self.versions = versions_for_test(self.calendar)
        self.protocol = dict(reward_horizons_ms=[1000,3000],latency_ms=500,
                             holding_review_ms=1000,threshold=.000015)
        self.plan = dict(base_ars_plan=dict(config=dict(history_window_ms=40000,tie_tolerance_reward=1e-9),
            base_period_plan=dict(model_ids=['long','short'],config=dict(selection_period_ms=5000,exploration_c_price=1.))))
        self.reader = SimpleNamespace(calendar=self.calendar, report=dict(interval_ms=500))
        self.case = dict(versions=self.versions)
        self.policies = [dict(policy_id='long',source='weekly_library'),dict(policy_id='short',source='weekly_library')]
        rows = [policy_row('cash',0.,cash=True),policy_row('long',100.,[2.,0.]),
                policy_row('short',50.,[0.,2.])]
        self.fit = learned(rows)

    def replay(self,frame,weights=None):
        with patch('src.learned_oe_experiment.read_day',return_value=frame):
            return replay_learned_day(self.reader,'2025-09-22',self.case,self.protocol,
                self.plan,self.fit['weights'] if weights is None else weights,detail=True)

    def test_successful_learning_enters_all_three_selectors_and_mature_feedback(self):
        """手工净利专家与完整模型动作一致；权重确实进入UCB和双ARS真实反馈。"""
        self.assertTrue(execution_gate(self.fit,self.policies)['usable_for_execution'])
        rows = self.replay(prepared())
        self.assertEqual([r['strategy'] for r in rows],LEARNED_STRATEGIES)
        for r in rows:
            self.assertEqual(r['reward_weights'],self.fit['weights'])
            self.assertGreater(r['observed_rewards'],0)
            self.assertEqual(sum(p['orders'] for p in r['period_feedback']),r['total_fills'])
            for p in r['period_feedback']:
                if p['updated_selector']:
                    self.assertGreaterEqual(p['observed_at_ns'],p['end_ns'])
                    self.assertGreaterEqual(p['observed_at_ns'],p['last_origin_ns']+3000000000)

    def test_cash_and_missing_library_observations_block_without_fake_zero_profit(self):
        """可表示现金专家也不属于模型库；成熟资格不足不靠增加动作或次优专家修补。"""
        cash = learned([policy_row('cash',0.,cash=True),policy_row('long',-10.,[-1.,-1.]),
                        policy_row('short',-20.,[-2.,-2.])])
        gate = execution_gate(cash,self.policies)
        self.assertIn('expert_outside_online_library',gate['reasons'])
        r = blocked_result('Learned-OE-UCB',gate)
        self.assertIsNone(r['net_pnl_usd']);self.assertIsNone(r['total_fills'])
        fit = deepcopy(self.fit);fit['eligible_policy_ids'].remove('short')
        self.assertIn('online_library_calibration_OE_incomplete',execution_gate(fit,self.policies)['reasons'])

    def test_signed_weights_change_feedback_and_history_score_not_static_prices(self):
        """相同成交条件下非等权评分改变；ARS历史评分也必须用同一新权重。"""
        frame = prepared(prices=[100.+.25*(i%9) for i in range(100)])
        a = self.replay(frame,[.5,.5]);b = self.replay(frame,[3.,-2.])
        pa = [p['reward'] for p in a[0]['period_feedback'] if p['updated_selector']]
        pb = [p['reward'] for p in b[0]['period_feedback'] if p['updated_selector']]
        self.assertNotEqual(pa,pb)
        for r in b[2:]:
            histories = [e for p in r['period_selections'] for e in p['evaluations']
                if e['source']=='independent_history_replay' and e['score'] is not None]
            self.assertTrue(histories)
            for e in histories:
                self.assertLessEqual(e['reward_observation_latest_required_ns'],e['known_through_ns'])

    def test_future_perturbation_cannot_change_past_learned_selections(self):
        """40秒以后价格与未来标签改变，40秒之前学习策略决策/反馈完全相同。"""
        prices = 100.+.25*np.arange(150);frame=prepared(list(pd.date_range(BASE,periods=150,freq='500ms')),prices)
        changed_prices=prices.copy();changed_prices[81:]+=100.
        changed=prepared(list(frame.ts_event),changed_prices)
        for c in changed:
            if c.startswith('future_'):changed[c]=np.nan
        a=self.replay(frame,[3.,-2.]);b=self.replay(changed,[3.,-2.]);cutoff=BASE.value+40000000000
        for x,y in zip(a,b):
            for key,field in [('period_selections','selected_at_ns'),('period_feedback','observed_at_ns')]:
                self.assertEqual([r for r in x[key] if r[field]<=cutoff],[r for r in y[key] if r[field]<=cutoff])


class LearnedExperimentTests(unittest.TestCase):
    """真实冻结/建库/校准CLI，缺观察时阻断，不把合成实验说成正式复现。"""
    def setUp(self):
        self.fixture=ars_fixture.ARSExperimentTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture.f;self.f.freeze()
        config=dict(schema_version=1,purpose='development_learned_OE_selection',ars_protocol_file='ars.json',
            learner='sum_only_full_finite_max_margin_min_L1_tie_break',execution_gate='expert_in_library_all_library_OE_observed')
        self.cfg=self.f.f.root/'learned.json';self.cfg.write_text(json.dumps(config))

    def test_cli_preserves_controls_execution_gate_and_calibration_identity(self):
        """合成七session实际建库/校准成功进入学习策略；控制逐日结果完全一致。"""
        from src.ars_experiment import replay_ars_day
        frozen=self.f.f.root/'learned-frozen';output=self.f.f.root/'learned-result'
        plan=main(['freeze','--config',str(self.cfg),'--library',str(self.f.libpath),'--output-dir',str(frozen)])
        result=main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output)])
        self.assertEqual(len(plan['strategies']),21)
        self.assertEqual(result,json.loads((output/'result.json').read_text()))
        self.assertTrue((output/'report.md').exists())
        for case in result['age_cases']:
            self.assertTrue(case['execution_gate']['usable_for_execution'])
            self.assertEqual(case['active_reward_weights'],case['reward_fits']['sum_only']['weights'])
            for phase in ('validation','test'):
                self.assertEqual(case['phases'][phase]['calibration_sha256'],case['calibration_sha256'])
                for day in case['phases'][phase]['daily']:
                    self.assertEqual(len(day['results']),21)
                    for r in day['results'][18:]:
                        self.assertEqual(r['run_status'],'executed')
                        self.assertEqual(r['reward_weights'],case['active_reward_weights'])
        # 另跑一个旧控制日期，避免只验证名称/字段而漏掉执行路径的变化。
        from src.library_oe import load_library
        from src.snapshot_backtest import PreparedDatasetReader
        lib,calendar=load_library(self.f.libpath);base=plan['base_ars_plan']['base_period_plan']
        binding=base['cases'][0]['session_binding'];reader=PreparedDatasetReader(binding['dataset']['path'],calendar=calendar)
        day=binding['protocol']['sessions']['validation'][0]
        old=replay_ars_day(reader,day,lib['age_cases'][0],binding['protocol'],plan['base_ars_plan'])
        self.assertEqual(result['age_cases'][0]['phases']['validation']['daily'][0]['results'][:18],old['results'])
        with self.assertRaises(FileExistsError):main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output)])

    def test_source_and_nested_plan_changes_rejected(self):
        """新学习源码与父协议均绑定身份，不能静默套用旧冻结参数。"""
        plan=freeze_learned_experiment(self.cfg,self.f.libpath);path=self.f.f.root/'learned-plan.json'
        path.write_text(json.dumps(plan))
        with patch('src.learned_oe_experiment.code_hashes',return_value={}):
            with self.assertRaises(ValueError):run_learned_experiment(path)
        plan['base_ars_plan']['config']['history_window_ms']+=1000
        from src.session_experiment import fingerprint
        plan['plan_sha256']=fingerprint(plan);path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):run_learned_experiment(path)


if __name__=='__main__':unittest.main()
