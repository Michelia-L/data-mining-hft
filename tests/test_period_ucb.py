"""固定期间 UCB 的手算、成熟因果性和端到端验证；盈利不作为正确性标准。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_period_ucb import main
from src.irl_reward import IRLRewardLearner
from src.period_experiment import freeze_period_experiment, replay_day, run_period_experiment
from src.period_ucb import PeriodOESelector, PeriodOrderBook
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import SessionCalendar
from src.time_execution import TimeExecutionEngine
from src.weekly_library import build_library, freeze_library
import test_library_oe as library_fixture
from test_time_execution import BASE, Model, prepared


class PeriodFormulaTests(unittest.TestCase):
    """明确 Eq.(5) 分母和 Eq.(6) 计数；不能把逐订单学习伪装成期间评价。"""
    def setUp(self):
        self.selector=PeriodOESelector('period',[Model(),Model()],period_ms=5000,c=2.)
        self.calendar=SessionCalendar();self.book=PeriodOrderBook(self.selector,self.calendar,3000000000)
        self.session='2025-09-22';self.start=BASE.value

    def choose(self,offset,version='v'):
        return self.book.choose(self.session,version,self.start+int(offset*1e9))

    def test_hand_calculated_order_average_waits_for_period_end_and_last_order(self):
        """三单 2/4/6 的期间均值为 4；结束前或最后一单未成熟都不得反馈。"""
        self.choose(0)
        keys=[self.book.add_order(self.session,'v',0,self.start+int(t*1e9)) for t in (1,2,4)]
        self.book.resolve(keys[0],'matured',2.);self.book.resolve(keys[1],'matured',4.)
        self.book.advance(self.start+int(5e9));self.assertEqual(self.selector.reward_history,[])
        self.book.resolve(keys[2],'matured',6.);self.book.advance(self.start+int(7e9))
        self.assertEqual(self.selector.reward_history,[4.])
        self.assertEqual(self.book.completed[0]['orders'],3)
        self.assertEqual(self.book.completed[0]['observed_at_ns'],self.start+int(7e9))

    def test_mature_order_alone_cannot_update_before_period_end(self):
        """早订单的长期标签即使已可见，固定期间未结束仍不能更新。"""
        self.choose(0);key=self.book.add_order(self.session,'v',0,self.start)
        self.book.resolve(key,'matured',8.);self.book.advance(self.start+int(3e9))
        self.assertEqual(self.selector.reward_history,[])
        self.book.advance(self.start+int(5e9));self.assertEqual(self.selector.reward_history,[8.])

    def test_one_bad_order_rejects_full_period_not_subset_or_zero(self):
        """一单成熟为 10，一单缺格，整桶不可用；不按一单平均，也不把坏单补零。"""
        self.choose(0);key=self.book.add_order(self.session,'v',0,self.start)
        self.book.add_order(self.session,'v',0,self.start+int(1e9))
        self.book.resolve(key,'matured',10.);self.book.resolve(key,'missing_target')
        self.book.advance(self.start+int(5e9))
        self.assertEqual(self.book.completed[0]['status'],'incomplete_orders')
        self.assertIsNone(self.book.completed[0]['reward']);self.assertEqual(self.selector.reward_history,[])

    def test_no_orders_and_pending_periods_do_not_fabricate_feedback(self):
        """访问计数包含无订单期间，反馈计数不包含它；期末待成熟也保持未定义。"""
        self.choose(0);self.book.advance(self.start+int(5e9))
        self.choose(5);self.book.add_order(self.session,'v',1,self.start+int(9e9))
        rows=self.book.finish(self.start+int(10e9))
        self.assertEqual([r['status'] for r in rows],['no_orders','pending_orders'])
        self.assertEqual(self.selector.summary()['v']['visits'],[1,1])
        self.assertEqual(self.selector.summary()['v']['feedback_counts'],[0,0])
        self.assertEqual(self.selector.reward_history,[])

    def test_eq6_exact_log_N_and_period_visit_count_with_ties(self):
        """每个期间多次 tick 仅一次访问；手算 W=[1,0],n=[1,1],C=2。"""
        self.assertEqual(self.choose(0),0);self.assertEqual(self.choose(1),0)
        self.assertEqual(self.choose(5),1)
        self.selector.observe_period('v',0,1.)
        self.assertEqual(self.choose(10),0)
        trace=self.selector.period_selections[-1]
        np.testing.assert_allclose(trace['scores_before'],[1.+2*np.sqrt(2*np.log(2)),2*np.sqrt(2*np.log(2))])
        self.assertEqual(trace['visits_before'],[1,1])
        self.assertEqual(trace['feedback_counts_before'],[1,0])
        fresh=PeriodOESelector('tie',[Model(),Model()],period_ms=5000,c=0.)
        for p in range(3):fresh.select_period(self.session,'v',p,self.start+p*int(5e9))
        self.assertEqual(fresh.period_selections[-1]['tied_best_owners'],[0,1])
        self.assertEqual(fresh.period_selections[-1]['owner'],0)
        with self.assertRaisesRegex(ValueError,'complete period'):fresh.observe(0,1.,0)

    def test_W_is_equal_period_mean_not_pooled_orders(self):
        """两个期间均值 2 与 10 的 W=6，不受期间订单数差异改变。"""
        self.selector.observe_period('v',0,2.);self.selector.observe_period('v',0,10.)
        self.assertEqual(self.selector.summary()['v']['observed_period_means'][0],6.)

    def test_old_version_feedback_cannot_change_new_statistics(self):
        """新版本从零开始，旧反馈到达后只改旧统计，不重标为新参数奖励。"""
        self.choose(0,'old');self.choose(5,'new');self.selector.observe_period('old',0,100.)
        self.assertEqual(self.selector.summary()['new']['feedback_counts'],[0,0])
        self.assertEqual(self.choose(10,'new'),1)
        self.assertEqual(self.selector.period_selections[-1]['means_before'],[0.,0.])

    def test_clock_gaps_and_close_boundary_do_not_compress_periods(self):
        """20 秒空档跳到第四期间，只记实际选择；收盘不得新发决策。"""
        self.choose(0);self.choose(20)
        trace=self.selector.period_selections
        self.assertEqual(trace[1]['period_index']-trace[0]['period_index'],4)
        self.assertEqual(sum(self.selector.summary()['v']['visits']),2)
        with self.assertRaises(ValueError):self.book.bounds(self.session,self.calendar.sessions[self.session]['close'].value)
        for settings in ({'period_ms':0},{'c':float('nan')},{'c':-1.},{'mode':'event'}):
            with self.assertRaises(ValueError):PeriodOESelector('bad',[Model()],**settings)


class PeriodReplayTests(unittest.TestCase):
    """使用真实时间账本，核对反馈、成交、未来扰动和换版因果性。"""
    def replay(self,frame,*,versions=None,mode='ucb',preds=None,**settings):
        selector=PeriodOESelector('period',[SimpleNamespace(name='long'),SimpleNamespace(name='short')],
            period_ms=5000,c=1.,mode=mode)
        reward=IRLRewardLearner(horizons=[1000,3000],definition='paper_price_difference')
        return TimeExecutionEngine(reward,**settings).run_backtest(selector,frame,
            np.tile([.01,-.01],(len(frame),1)) if preds is None else preds,
            detail=True,model_version_ids=versions)

    def test_engine_feedback_conserves_all_fills_and_uses_complete_period_only(self):
        """完整期间只更新一次，单笔成熟事件不更新 UCB，所有成交在桶内守恒。"""
        result=self.replay(prepared(),holding_review_ms=1000)
        rows=result['period_feedback']
        self.assertEqual(sum(p['orders'] for p in rows),result['total_fills'])
        self.assertEqual(sum(p['pending'] for p in rows),result['unmatured_fill_rewards_at_end'])
        self.assertEqual(sum(p['statuses'].get('matured',0) for p in rows),result['matured_order_count'])
        self.assertEqual(sum(p['updated_selector'] for p in rows),result['observed_rewards'])
        self.assertTrue(all(not o['updates_selector'] for o in result['reward_observations']))
        for p in rows:
            if p['updated_selector']:
                self.assertGreaterEqual(p['observed_at_ns'],p['end_ns'])
                self.assertGreaterEqual(p['observed_at_ns'],p['last_origin_ns']+3000000000)
                self.assertEqual(p['statuses']['matured'],p['orders'])
                self.assertAlmostEqual(p['reward'],p['reward_sum']/p['orders'])
        self.assertAlmostEqual(sum(t['net_pnl_usd'] for t in result['trades']),result['net_pnl_usd'])

    def test_future_price_perturbation_preserves_past_decisions_and_period_feedback(self):
        """改变 30 秒后行情，之前的 tick/期间选择、成交和完整期间反馈完全相同。"""
        prices=np.arange(100)*.25+100.;a=self.replay(prepared(prices=prices))
        prices[61:]+=100.;b=self.replay(prepared(prices=prices));cutoff=BASE+pd.Timedelta(seconds=30)
        for key,field in [('decisions','ts_event'),('fills','ts_event')]:
            self.assertEqual([r for r in a[key] if pd.Timestamp(r[field])<=cutoff],
                             [r for r in b[key] if pd.Timestamp(r[field])<=cutoff])
        for key,field in [('period_feedback','observed_at_ns'),('period_selections','selected_at_ns')]:
            self.assertEqual([r for r in a[key] if r[field]<=cutoff.value],
                             [r for r in b[key] if r[field]<=cutoff.value])
        frame=prepared()
        for c in frame:
            if c.startswith('future_'):frame[c]=np.nan
            if c.startswith('label_valid_') or c=='learning_ready':frame[c]=False
        self.assertEqual(a,self.replay(frame))

    def test_version_switch_and_gap_keep_old_positions_and_invalid_rewards(self):
        """换版不清零原仓位，旧反馈只更新旧版本；缺格桶保留不完整状态。"""
        frame=prepared();versions=np.where(np.arange(len(frame))<35,'old','new')
        r=self.replay(frame,versions=versions,holding_review_ms=1000,mode='round_robin')
        self.assertEqual(r['trades'][0]['model_version_id'],'old')
        first_new=next(p for p in r['period_selections'] if p['model_version_id']=='new')
        self.assertEqual(first_new['feedback_counts_before'],[0,0])
        self.assertTrue(any(p['model_version_id']=='old' and p['observed_at_ns']>=frame.ts_event.iloc[35].value for p in r['period_feedback']))
        stamps=list(pd.date_range(BASE,periods=100,freq='500ms'));del stamps[32]
        gap=self.replay(prepared(stamps),holding_review_ms=1000)
        self.assertGreater(gap['period_status_counts'].get('incomplete_orders',0),0)

    def test_zero_predictions_and_missing_tail_remain_explicit(self):
        """无成交仍探索且不学习；缺尾不能把最后未成熟期间当作零或成功反馈。"""
        frame=prepared();r=self.replay(frame,preds=np.zeros((len(frame),2)),force_replay_end=False)
        self.assertEqual(r['total_fills'],0);self.assertEqual(r['observed_rewards'],0)
        self.assertGreater(len(r['period_selections']),2)
        late=np.zeros((len(frame),2));late[-5:]=.01
        tail=self.replay(frame,preds=late,force_replay_end=False)
        self.assertGreater(tail['period_status_counts'].get('incomplete_period',0),0)
        self.assertGreater(tail['unmatured_fill_rewards_at_end'],0)


class PeriodExperimentTests(unittest.TestCase):
    """复用七 session 合成建库，验证配置冻结和 CLI 的多周完整对照。"""
    def setUp(self):
        self.fixture=library_fixture.LibraryOEIntegrationTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.f=self.fixture
        cfg=dict(schema_version=1,purpose='development_period_OE_UCB',oe_protocol_file='oe.json',
            selection_period_ms=300000,exploration_c_price=1.,reward_source='equal_weight_control',stages=['validation','test'])
        self.cfg=self.f.root/'period.json';self.cfg.write_text(json.dumps(cfg))

    def freeze(self):
        lp=self.f.root/'library-plan.json';lp.write_text(json.dumps(freeze_library(self.f.library_path,[self.f.dataset])))
        lib=build_library(lp);self.libpath=self.f.root/'library.json';self.libpath.write_text(json.dumps(lib))
        plan=main(['freeze','--config',str(self.cfg),'--library',str(self.libpath),'--output-dir',str(self.f.root/'period-frozen')])
        return lib,plan,self.f.root/'period-frozen/plan.json'

    def test_cli_keeps_all_controls_and_actual_weekly_versions(self):
        """16 个策略、未来周版本和独立日统计均保留，JSON/中文报告可重复。"""
        lib,plan,path=self.freeze()
        result=main(['run','--plan',str(path),'--detail','--output-dir',str(self.f.root/'period-result')])
        self.assertEqual(len(plan['strategies']),16)
        self.assertEqual(result,json.loads((self.f.root/'period-result/result.json').read_text()))
        self.assertEqual(result,run_period_experiment(path,detail=True))
        case=result['age_cases'][0]
        for phase,data in case['phases'].items():
            for day in data['daily']:
                index=0 if day['session_id']<'2025-10-06' else 1
                self.assertEqual(day['library_version_ids'],[lib['age_cases'][0]['versions'][index]['version_sha256']])
                for r in day['results'][:2]:
                    self.assertEqual(sum(p['orders'] for p in r['period_feedback']),r['total_fills'])
                    self.assertEqual(sum(sum(s['visits']) for s in r['selector_version_statistics'].values()),len(r['period_selections']))
        with self.assertRaises(FileExistsError):main(['run','--plan',str(path),'--output-dir',str(self.f.root/'period-result')])

    def test_frozen_source_library_and_data_guards(self):
        """源码、模型和质量数据改变不能静默沿用原实验身份；错误奖励源拒绝冻结。"""
        lib,plan,path=self.freeze()
        with patch('src.period_experiment.code_hashes',return_value={}):
            with self.assertRaises(ValueError):run_period_experiment(path)
        self.libpath.write_text(json.dumps(lib)+' ')
        with self.assertRaisesRegex(ValueError,'artifact'):run_period_experiment(path)
        self.libpath.write_text(json.dumps(lib))
        quality=self.f.dataset/'quality.json';quality.write_text(quality.read_text()+' ')
        with self.assertRaises(ValueError):run_period_experiment(path)
        config=json.loads(self.cfg.read_text());config['reward_source']='learned_from_unrepresentable_expert';self.cfg.write_text(json.dumps(config))
        with self.assertRaises(ValueError):freeze_period_experiment(self.cfg,self.libpath)


if __name__=='__main__':unittest.main()
