"""ME的研究性质测试：预测起点、完整信号分母、成熟时序与同成本执行。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_four_combinations import main
from src.four_combinations import (audit_me, calibration_day, compact_evidence, freeze_four,
    me_fit_groups, pool_me, replay_additions, run_four)
from src.history_ars import HistoryARSSelector
from src.model_selector import SingleModelSelector
from src.multiweek_dynamic_oe import freeze_dynamic, run_dynamic
from src.period_ucb import PeriodOESelector
from src.sum_only_reward import FrozenSumOnlyReward
from src.time_execution import TimeExecutionEngine
from src.time_me import PredictionHistoryBacktester, calibration_me, replay_me, signal_features
import test_history_ars as ars_fixture
from test_history_ars import versions_for_test
from test_snapshot_irl import config_for_test, policy_row
from test_time_execution import BASE, prepared


class TimeMETests(unittest.TestCase):
    """手工冻结模型与合成行情隔离训练随机性；通过盈利与否不判断正确性。"""
    def setUp(self):
        self.frame = prepared()
        self.calendar = ars_fixture.SessionCalendar()
        self.versions = versions_for_test(self.calendar)
        self.models = [SimpleNamespace(name=n) for n in ('long', 'short')]
        self.reward = FrozenSumOnlyReward([1000, 3000], [.5, .5])
        self.matrix = np.tile([.01, -.01], (len(self.frame), 1))
        self.ids = np.full(len(self.frame), self.versions[0]['version_sha256'], dtype=object)

    def replay(self, frame=None, matrix=None, mode='ucb'):
        """完整期间5秒，奖励1/3秒；每次重启选择器和账户，明确为合成单位。"""
        frame = self.frame if frame is None else frame
        matrix = self.matrix if matrix is None else matrix
        selector = PeriodOESelector('ME-'+mode, self.models, period_ms=5000, mode=mode)
        return replay_me(selector, frame, matrix, self.ids, self.reward, self.calendar,
                         holding_review_ms=1000, detail=True)

    def test_origin_sign_and_all_scale_visibility(self):
        """从预测时刻计算价差，不从延迟成交起算；短尺度成熟不足以反馈。"""
        f, s = signal_features(self.frame, [30, 30], [1, -1], self.reward, 500, BASE.value+18_000_000_000)
        np.testing.assert_allclose(f, [[.5, 1.5], [-.5, -1.5]])
        self.assertEqual(s.tolist(), ['matured', 'matured'])
        f, s = signal_features(self.frame, [30], [1], self.reward, 500, BASE.value+16_000_000_000)
        self.assertEqual(s.tolist(), ['pending_signals']); self.assertTrue(np.isnan(f).all())
        result = self.replay()
        audit_me(result, self.versions, 3000, [.5, .5])
        first = next(p for p in result['period_feedback'] if p['updated_selector'])
        self.assertEqual(first['signals'], 10)
        self.assertAlmostEqual(first['reward'], 1.)
        self.assertGreaterEqual(first['observed_at_ns'], first['last_origin_ns']+3_000_000_000)
        self.assertGreater(result['matured_signal_count'], result['matured_order_count'])

    def test_no_signal_is_not_a_zero_reward(self):
        """全部预测在门槛内仍可选择，但无信号期间不学习，现金美元0另解释。"""
        result = self.replay(matrix=np.zeros_like(self.matrix))
        self.assertEqual(result['prediction_signal_count'], 0)
        self.assertEqual(result['observed_rewards'], 0)
        self.assertIsNone(result['prediction_feature_expectation'])
        self.assertTrue(all(p['reward'] is None for p in result['period_feedback']))
        self.assertEqual(result['net_pnl_usd'], 0.)

    def test_future_price_and_label_changes_leave_earlier_choices_unchanged(self):
        """30秒以后的行情/离线标签变化不得改变此前成熟反馈和模型选择。"""
        changed = self.frame.copy(); tail = changed.ts_event > BASE+pd.Timedelta(seconds=30)
        for col in ('mid_price', 'bid_px_00', 'ask_px_00'):
            changed.loc[tail, col] += 50.
        for col in changed:
            if col.startswith(('future_', 'label_', 'learning_ready')):
                changed[col] = False if changed[col].dtype == bool else np.nan
        a = self.replay(); b = self.replay(frame=changed)
        cutoff = BASE.value+30_000_000_000
        for key, clock in [('period_selections', 'selected_at_ns'), ('period_feedback', 'observed_at_ns')]:
            self.assertEqual([r for r in a[key] if r[clock] <= cutoff], [r for r in b[key] if r[clock] <= cutoff])

    def test_gap_invalidates_whole_period_and_does_not_shrink_denominator(self):
        """丢失一个目标或中间格使完整信号期间不可学习；不改成成熟子集均值。"""
        gap = self.frame.drop(index=38).reset_index(drop=True)
        result = replay_me(PeriodOESelector('gap', self.models, period_ms=5000), gap,
            np.tile([.01, -.01], (len(gap), 1)), self.ids[:len(gap)], self.reward, self.calendar, detail=True)
        audit_me(result, self.versions, 3000)
        bad = next(p for p in result['period_feedback'] if p['statuses'].get('missing_target',0))
        self.assertGreater(bad['signals'], bad['statuses'].get('matured',0))
        self.assertIsNone(bad['reward']); self.assertFalse(bad['updated_selector'])

    def test_execution_matches_original_engine_for_same_choices(self):
        """同期间预声明轮换的美元账本逐单等价，说明ME复用原延迟和成本条件。"""
        result = self.replay(mode='round_robin')
        choices = {p['period_index']: p['owner'] for p in result['period_selections']}
        start = self.calendar.sessions['2025-09-22']['open'].value
        from src.snapshot_dataset import utc_ns
        signals = np.array([self.matrix[i, choices.get((int(t)-start)//5_000_000_000, 0)]
                           for i,t in enumerate(utc_ns(self.frame.ts_event))])[:,None]
        static = SingleModelSelector('same_signals', [self.models[0]]); static.reward_type='OE'
        old = TimeExecutionEngine(self.reward, calendar=self.calendar, holding_review_ms=1000,
            force_replay_end=False).run_backtest(static, self.frame, signals, detail=True, model_version_ids=self.ids)
        for key in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd', 'total_fills', 'total_trades'):
            self.assertEqual(old[key], result[key])
        self.assertEqual([f['timestamp_ns'] for f in old['fills']], [f['timestamp_ns'] for f in result['fills']])

    def test_recent_and_matured_history_keep_missing_feedback(self):
        """最近窗口保留未成熟，平移窗口手算有方向ME；不凭未来标签挑信号。"""
        history = PredictionHistoryBacktester(self.frame, self.versions, self.calendar, self.reward)
        now = BASE.value+49_000_000_000; version = self.ids[0]
        recent = history.evaluate(now,'2025-09-22',version,40000,'recent',[0,1])
        mature = history.evaluate(now,'2025-09-22',version,40000,'matured',[0,1])
        self.assertTrue(all(r['score'] is None and r['status']=='pending_signals' for r in recent.values()))
        self.assertAlmostEqual(mature[0]['score'], 1.); self.assertAlmostEqual(mature[1]['score'], -1.)
        for row in mature.values():
            self.assertEqual(row['matured_signal_count'],row['total_signals'])
            self.assertLessEqual(row['latest_signal_required_ns'],now)
        selector = HistoryARSSelector('ME-ARS',self.models,history,period_ms=5000,window_ms=40000,alignment='matured')
        result = replay_me(selector,self.frame,self.matrix,self.ids,self.reward,self.calendar,detail=True)
        self.assertGreater(audit_me(result,self.versions,3000)['scored_windows'],0)

    def test_audit_rejects_early_feedback_wrong_version_and_lost_signal(self):
        """独立审计拒绝提前成熟、丢分母和错版；正收益不能掩盖这些问题。"""
        original = self.replay()
        for kind in ('early','lost','version','score'):
            row=deepcopy(original)
            if kind=='early':
                p=next(p for p in row['period_feedback'] if p['updated_selector']);p['observed_at_ns']=p['last_origin_ns']
            elif kind=='lost':row['prediction_signal_count']+=1
            elif kind=='version':row['period_selections'][0]['model_version_id']='future'
            else:
                p=next(p for p in row['period_feedback'] if p['updated_selector']);p['signal_feature_sum'][0]+=10.
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit_me(row,self.versions,3000)

    def test_calibration_uses_matured_signals_and_preserves_blocked_expert(self):
        """校准不读离线标签；亏损专家可拟合，缺成熟最佳专家仍阻断而不换人。"""
        cal=calibration_me(self.frame,self.matrix[:,0],self.calendar,self.reward,500,.000015)
        self.assertGreater(cal['matured_signal_count'],0)
        np.testing.assert_allclose(cal['prediction_feature_expectation'],[.5,1.5])
        rows=[policy_row('cash',0.,cash=True),policy_row('a',-10.,[2.,0.]),policy_row('b',-20.,[0.,2.])]
        stats=[r|dict(matured_signal_count=r['matured_order_count'],prediction_feature_expectation=r['order_feature_expectation']) for r in rows]
        policies=[dict(policy_id='cash',source='fixed_reference'),dict(policy_id='a',source='weekly_library'),dict(policy_id='b',source='weekly_library')]
        groups=me_fit_groups(stats,[1000,3000],config_for_test(),policies)
        self.assertIsNone(groups['all_candidates']['active_reward_weights'])
        self.assertIsNotNone(groups['online_library']['active_reward_weights'])
        stats[1].update(matured_signal_count=0,prediction_feature_expectation=None)
        groups=me_fit_groups(stats,[1000,3000],config_for_test(),policies)
        self.assertIsNone(groups['online_library']['active_reward_weights'])


class FourCombinationTests(unittest.TestCase):
    """七session实建库夹具验证CLI/冻结/恢复，所有组合执行或阻断都必须可审计。"""
    def test_complete_cli_inheritance_new_me_freeze_and_restore(self):
        """旧OE数值逐字段保留，ME仅校准一次；不允许损坏身份或覆盖正式产物。"""
        fixture=ars_fixture.ARSExperimentTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        f=fixture.f;f.freeze();root=f.f.root
        dynamic=dict(schema_version=1,purpose='development_multiweek_dynamic_OE',ars_protocol_file='ars.json',
            expert_scopes=['all_candidates','online_library'],active_constraint='sum_only',minimum_evaluation_weekly_versions=2)
        config=root/'dynamic.json';config.write_text(json.dumps(dynamic))
        plan=freeze_dynamic(config,f.libpath);p=root/'oe-plan.json';p.write_text(json.dumps(plan))
        old=run_dynamic(p);source=root/'oe-result.json';source.write_text(json.dumps(old))
        config=root/'four.json';config.write_text(Path('config/esz5_four_combinations_development.json').read_text())
        frozen=root/'four-frozen';output=root/'four-results';checks=root/'four-checks'
        newplan=main(['freeze','--config',str(config),'--oe-result',str(source),'--output-dir',str(frozen)])
        result=main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output),'--checkpoint-dir',str(checks)])
        self.assertEqual(result,json.loads((output/'result.json').read_text()))
        self.assertEqual(len(newplan['strategies']),len(plan['strategies'])+10)
        for oc,nc in zip(old['age_cases'],result['age_cases']):
            for phase in ('validation','test'):
                for od,nd in zip(oc['phases'][phase]['daily'],nc['phases'][phase]['daily']):
                    self.assertEqual(od['results'],nd['results'][:len(plan['strategies'])])
        evidence=json.loads((output/'evidence.json').read_text())
        self.assertEqual(evidence['age_cases'][0]['phases']['test']['statistics'],result['age_cases'][0]['phases']['test']['statistics'])
        with patch('src.four_combinations.calibration_day',side_effect=AssertionError('unexpected ME fit replay')), \
             patch('src.four_combinations.replay_additions',side_effect=AssertionError('unexpected trading replay')):
            restored=run_four(frozen/'plan.json',checkpoint_dir=checks)
        self.assertEqual(restored,result)
        with self.assertRaises(FileExistsError):main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output)])
        with patch('src.four_combinations.code_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'integrity'):run_four(frozen/'plan.json')
        source.write_text(source.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'integrity'):run_four(frozen/'plan.json')


if __name__=='__main__':
    unittest.main()
