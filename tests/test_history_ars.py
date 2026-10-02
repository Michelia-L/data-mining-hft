"""ARS 子回测与当前可见评分的研究性质测试；不以收益正负判断实现正确。"""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_history_ars import main
from src.ars_experiment import freeze_ars_experiment,run_ars_experiment
from src.history_ars import HistoryARSSelector,HistoryWindowBacktester
from src.irl_reward import IRLRewardLearner
from src.model_selector import SingleModelSelector
from src.period_experiment import freeze_period_experiment,run_period_experiment
from src.snapshot_dataset import SessionCalendar,build_session_dataset
from src.session_experiment import fingerprint
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library
import test_period_ucb as period_fixture
from test_time_execution import BASE,prepared


def versions_for_test(calendar,feature='rel_spread'):
    """两个可序列化固定线性模型，排除拟合随机性，单独测试执行和可见性。"""
    models=[dict(model_id=name,status='fitted',family='Ridge',features=[feature],
        scaler_mean=[0.],scaler_scale=[1.],coefficients=[0. if feature=='rel_spread' else sign*4.],
        intercept=sign*.01 if feature=='rel_spread' else 0.) for name,sign in [('long',1),('short',-1)]]
    version=dict(available_at_utc=calendar.sessions['2025-09-22']['open'].isoformat(),
        interval_ms=500,max_age_ms=500,models=models)
    version['version_sha256']=fingerprint(version)
    return [version]


class HistoryWindowTests(unittest.TestCase):
    """手算完整订单均值，并和原时间执行器的完整成交账本逐项比较。"""
    def setUp(self):
        self.calendar=SessionCalendar();self.versions=versions_for_test(self.calendar)
        self.reward=IRLRewardLearner(horizons=[1000,3000],definition='paper_price_difference')
        self.stamps=list(pd.date_range(BASE,periods=200,freq='500ms'))
        prices=np.full(200,100.);prices[[33,37,81,85]]=[102.,106.,104.,108.]
        self.frame=prepared(self.stamps,prices)

    def scorer(self,frame=None,versions=None,**settings):
        return HistoryWindowBacktester(self.frame if frame is None else frame,
            self.versions if versions is None else versions,self.calendar,self.reward,**settings)

    def evaluate(self,scorer=None,now=43000,window=40000,alignment='matured',owners=None):
        return (scorer or self.scorer()).evaluate(BASE.value+now*1000000,'2025-09-22',
            self.versions[0]['version_sha256'],window,alignment,[0,1] if owners is None else owners)

    def test_hand_computed_two_order_OE_and_costs(self):
        """开仓未来均值+4，退出方向均值-6，整窗两单均值-1；空头相反。"""
        results=self.evaluate()
        self.assertEqual(results[0]['total_fills'],2);self.assertEqual(results[0]['status'],'matured')
        self.assertAlmostEqual(results[0]['score'],-1.);self.assertAlmostEqual(results[1]['score'],1.)
        self.assertAlmostEqual(results[0]['friction_usd'],27.5)
        self.assertEqual(results[0]['gross_pnl_usd'],0.)
        self.assertEqual(results[0]['matured_order_count'],2)
        self.assertEqual(results[0]['reward_observation_latest_required_ns'],BASE.value+42500000000)

    def test_optimized_account_matches_existing_engine_with_latency_and_gaps(self):
        """跳过无效 tick 的优化必须与完整回放一致：变向、750ms 延迟、暂停风险退出。"""
        stamps=self.stamps.copy();del stamps[110:113]
        prices=np.array([100.+.25*((i//4)%5) for i in range(len(stamps))])
        frame=prepared(stamps,prices);versions=versions_for_test(self.calendar,'ret_lag_1')
        scorer=self.scorer(frame,versions,latency_ms=750,holding_review_ms=1500)
        now=BASE.value+93000000000;start=BASE.value+20000000000;end=BASE.value+90000000000
        result=scorer.evaluate(now,'2025-09-22',versions[0]['version_sha256'],70000,'matured',[0,1],detail=True)
        raw=frame[(frame.ts_event>=pd.Timestamp(start,unit='ns',tz='UTC'))&(frame.ts_event<pd.Timestamp(end,unit='ns',tz='UTC'))]
        raw=raw[SNAPSHOT_SCHEMA.names+['session_id','segment_id']]
        historical,_,_=build_session_dataset(raw,self.calendar,500,(1000,3000))
        matrix,_=predict_library(versions,historical,500)
        for owner in (0,1):
            quotes=historical.copy();quotes['feature_valid'] &= np.isfinite(matrix).all(axis=1)
            selector=SingleModelSelector('gold',[SimpleNamespace(name='long'),SimpleNamespace(name='short')],fixed_idx=owner)
            selector.reward_type='OE'
            gold=TimeExecutionEngine(self.reward,latency_ms=750,holding_review_ms=1500).run_backtest(selector,quotes,matrix,detail=True)
            for key in ('gross_pnl_usd','friction_usd','net_pnl_usd','total_fills','terminal_position'):
                self.assertEqual(result[owner][key],gold[key],(owner,key))
            self.assertEqual(sum(result[owner]['order_statuses'].values()),gold['total_fills'])
            fields=('timestamp_ns','ts_event','side','owner','opening','mid','price','cost_usd','cost_ratio')
            self.assertEqual([{k:f[k] for k in fields} for f in result[owner]['fills']],
                             [{k:f[k] for k in fields} for f in gold['fills']])
        self.assertEqual(result,scorer.evaluate(now,'2025-09-22',versions[0]['version_sha256'],70000,'matured',[0,1],detail=True))

    def test_recent_window_cannot_read_future_and_keeps_pending_denominator(self):
        """最长3秒大于2秒窗时即使源容器有未来报价，也只报告未成熟或无订单。"""
        # 长窗口有交易，最后强退单也必须等待其未来价格，不提前回传。
        recent=self.evaluate(now=40000,window=40000,alignment='recent')
        self.assertEqual(recent[0]['status'],'pending_orders');self.assertIsNone(recent[0]['score'])
        self.assertEqual(recent[0]['total_fills'],2);self.assertEqual(recent[0]['order_statuses']['pending_orders'],1)
        truncated=self.frame[self.frame.ts_event<=BASE+pd.Timedelta(seconds=40)].copy()
        self.assertEqual(recent,self.evaluate(self.scorer(truncated),now=40000,window=40000,alignment='recent'))
        short=self.evaluate(now=40000,window=2000,alignment='recent')
        self.assertTrue(all(r['score'] is None for r in short.values()))

    def test_outside_trade_and_reward_envelope_cannot_change_score(self):
        """窗口前与奖励扩展后行情变化不影响评分，窗口后合法目标价格则可以影响。"""
        a=self.evaluate(now=60000,window=30000)
        modified=self.frame.copy();outside=(modified.ts_event<BASE+pd.Timedelta(seconds=27))|(modified.ts_event>BASE+pd.Timedelta(seconds=60))
        for c in modified:
            if c.startswith(('bid_px_','ask_px_')):modified.loc[outside,c]+=100.
            if c.startswith('future_'):modified[c]=np.nan
            if c.startswith('label_valid_') or c=='learning_ready':modified[c]=False
        self.assertEqual(a,self.evaluate(self.scorer(modified),now=60000,window=30000))
        b=self.evaluate();changed=self.frame.copy();target=BASE+pd.Timedelta(seconds=40.5)
        for c in changed:
            if c.startswith(('bid_px_','ask_px_')):changed.loc[changed.ts_event==target,c]+=10.
        self.assertNotEqual(b[0]['score'],self.evaluate(self.scorer(changed))[0]['score'])

    def test_missing_target_or_window_edge_remains_undefined(self):
        """缺一格目标整窗不学习，缺末退出格不在更早报价有利强平。"""
        gap=self.frame[self.frame.ts_event!=BASE+pd.Timedelta(seconds=42.5)].copy()
        r=self.evaluate(self.scorer(gap))[0]
        self.assertEqual(r['status'],'incomplete_orders');self.assertIsNone(r['score'])
        tail=self.frame[self.frame.ts_event!=BASE+pd.Timedelta(seconds=39.5)].copy()
        r=self.evaluate(self.scorer(tail))[0]
        self.assertEqual(r['status'],'incomplete_window_boundary');self.assertEqual(r['terminal_position'],1)
        leading=self.frame[self.frame.ts_event!=BASE].copy()
        self.assertEqual(self.evaluate(self.scorer(leading))[0]['status'],'incomplete_window_boundary')

    def test_no_orders_and_failed_model_do_not_become_zero_reward(self):
        """完全零预测的窗口 OE 为 None，模型失败为显式状态，不添加虚构现金均值。"""
        versions=deepcopy(self.versions);versions[0]['models'][0]['intercept']=0.
        versions[0]['models'][1]['status']='insufficient_training_rows'
        v=versions[0];v['version_sha256']=fingerprint({k:z for k,z in v.items() if k!='version_sha256'})
        r=self.scorer(versions=versions).evaluate(BASE.value+43000000000,'2025-09-22',v['version_sha256'],40000,'matured',[0,1])
        self.assertEqual(r[0]['status'],'no_orders');self.assertIsNone(r[0]['score'])
        self.assertEqual(r[1]['status'],'model_not_fitted');self.assertIsNone(r[1]['score'])

    def test_future_and_new_versions_obey_current_visibility(self):
        """未来版本不得提前影响评分；新版本不能把旧版历史窗口当作当前参数表现。"""
        versions=deepcopy(self.versions);future=deepcopy(versions[0]);future['available_at_utc']=(BASE+pd.Timedelta(seconds=80)).isoformat()
        future['models'][0]['intercept']=100.
        future['version_sha256']=fingerprint({k:v for k,v in future.items() if k!='version_sha256'});versions.append(future)
        self.assertEqual(self.evaluate(),self.evaluate(self.scorer(versions=versions)))
        scorer=self.scorer(versions=versions)
        r=scorer.evaluate(BASE.value+90000000000,'2025-09-22',future['version_sha256'],40000,'matured',[0,1])
        self.assertTrue(all(z['status']=='window_before_current_version' for z in r.values()))
        with self.assertRaisesRegex(ValueError,'obsolete'):self.evaluate(scorer,now=90000)
        with self.assertRaises(ValueError):self.evaluate(window=1100)


class ARSSelectionTests(unittest.TestCase):
    """真实期间与其他臂重回测分开，不延用失效窗口，也不把初始化零当观察。"""
    def setUp(self):
        self.backtester=SimpleNamespace(interval_ms=500,evaluate=lambda now,session,version,window,alignment,owners:
            {i:dict(owner=i,status='matured',score=[2.,-1.,2.][i],source='independent_history_replay') for i in owners})
        self.selector=HistoryARSSelector('ars',[SimpleNamespace(name=str(i)) for i in range(3)],self.backtester,period_ms=5000,window_ms=10000)

    def test_live_selected_and_other_window_scores_use_distinct_sources(self):
        """已选模型只有成熟实盘期间，其余模型从独立历史窗取分，负值不与假零比较。"""
        self.selector.observe_period('v',0,-2.,context=dict(period_index=0,observed_at_ns=100,last_origin_ns=0,session_id='day'))
        self.assertEqual(self.selector.select_period('day','v',1,200),2)
        p=self.selector.period_selections[-1]
        self.assertEqual(p['scores_before'],[-2.,-1.,2.]);self.assertEqual(p['previous_owner'],0)
        self.assertEqual(p['evaluations'][0]['source'],'live_selected_model')
        self.assertEqual(self.selector.select_period('day','v',1,300),2)
        self.assertEqual(len(self.selector.period_selections),1)

    def test_stale_window_disappears_and_cold_start_is_explicit(self):
        """新的窗口不可评价就丢弃旧窗口分数，全不可观察按预声明顺序轮换。"""
        self.selector.select_period('day','v',0,100)
        self.backtester.evaluate=(lambda now,session,version,window,alignment,owners:
            {i:dict(owner=i,status='pending_orders',score=None,source='independent_history_replay') for i in owners})
        self.selector.select_period('day','v',1,200)
        p=self.selector.period_selections[-1];self.assertTrue(p['cold_start']);self.assertEqual(p['owner'],1)
        self.assertEqual(p['scores_before'],[None,None,None])

    def test_live_feedback_version_and_time_identity_are_not_rewritten(self):
        """旧版奖励不进入新版，较老期间不覆盖较新，未来反馈不能提前选用。"""
        s=self.selector
        s.observe_period('old',0,5.,context=dict(period_index=2,observed_at_ns=100))
        s.observe_period('old',0,99.,context=dict(period_index=1,observed_at_ns=200))
        self.assertEqual(s.latest_live[('old',0)]['reward'],5.)
        s.select_period('day','new',0,300);self.assertIsNone(s.period_selections[-1]['scores_before'][0])
        s.observe_period('future',0,5.,context=dict(period_index=0,observed_at_ns=1000))
        with self.assertRaisesRegex(ValueError,'visible'):s.select_period('day','future',0,500)

    def test_greedy_ties_preserve_all_candidates(self):
        """其他臂并列时保留并列列表，按声明顺序取第一；不随机挑有利策略。"""
        self.backtester.evaluate=(lambda now,session,version,window,alignment,owners:
            {i:dict(owner=i,status='matured',score=1.,source='independent_history_replay') for i in owners})
        self.assertEqual(self.selector.select_period('day','v',0,100),1)
        self.assertEqual(self.selector.period_selections[-1]['tied_best_owners'],[1,2])


class ARSExperimentTests(unittest.TestCase):
    """七 session 合成模型库和 CLI 连通，原16对照逐项不变、双窗口均保留。"""
    def setUp(self):
        self.fixture=period_fixture.PeriodExperimentTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.f=self.fixture
        config=dict(schema_version=1,purpose='development_history_OE_ARS',period_protocol_file='period.json',
                    history_window_ms=1800000,alignments=['recent','matured'],tie_tolerance_reward=1e-9)
        self.cfg=self.f.f.root/'ars.json';self.cfg.write_text(json.dumps(config))

    def test_cli_double_controls_and_original_results_are_identical(self):
        """18 策略全保留，合成真实跨周可见版本正确，原 UCB/静态对照不变。"""
        lib,base,path=self.f.freeze()
        plan=main(['freeze','--config',str(self.cfg),'--library',str(self.f.libpath),'--output-dir',str(self.f.f.root/'ars-frozen')])
        planpath=self.f.f.root/'ars-frozen/plan.json'
        result=main(['run','--plan',str(planpath),'--output-dir',str(self.f.f.root/'ars-result')])
        self.assertEqual(len(plan['strategies']),18)
        self.assertEqual(result,json.loads((self.f.f.root/'ars-result/result.json').read_text()))
        old=run_period_experiment(path)
        for case,previous in zip(result['age_cases'],old['age_cases']):
            for phase,data in case['phases'].items():
                for day,old_day in zip(data['daily'],previous['phases'][phase]['daily']):
                    self.assertEqual(day['results'][:16],old_day['results'])
                    for r in day['results'][16:]:
                        self.assertEqual(sum(p['orders'] for p in r['period_feedback']),r['total_fills'])
                        self.assertEqual(sum(p['updated_selector'] for p in r['period_feedback']),r['observed_rewards'])
                        self.assertNotIn('exploration_c_price',r)
                        for p in r['period_selections']:
                            for e in p['evaluations']:
                                if e['source']=='independent_history_replay':
                                    self.assertLessEqual(e['window_end_ns'],p['selected_at_ns'])
                                    self.assertEqual(e['known_through_ns'],p['selected_at_ns'])
        with self.assertRaises(FileExistsError):main(['run','--plan',str(planpath),'--output-dir',str(self.f.f.root/'ars-result')])

    def test_plan_source_and_library_changes_are_rejected(self):
        """窗口、源码和模型文件改变不能无记录地复用冻结结果身份。"""
        self.f.freeze()
        plan=freeze_ars_experiment(self.cfg,self.f.libpath);p=self.f.f.root/'ars-plan.json';p.write_text(json.dumps(plan))
        with patch('src.ars_experiment.code_hashes',return_value={}):
            with self.assertRaises(ValueError):run_ars_experiment(p)
        self.f.libpath.write_text(self.f.libpath.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'artifact'):run_ars_experiment(p)
        plan['config']['history_window_ms']+=1000;p.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):run_ars_experiment(p)


if __name__=='__main__':unittest.main()
