"""验证多模型校准、因果周更新及订单版本归属，不用盈利与否代替正确性。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from run_library_oe import main
from src.irl_reward import IRLRewardLearner
from src.library_oe import freeze_library_oe, replay_candidates, run_library_oe
from src.model_selector import SingleModelSelector
from src.session_experiment import fingerprint, freeze_protocol
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_irl import replay_stage, policy_specs
from src.time_execution import TimeExecutionEngine
from src.weekly_library import build_library, freeze_library
import test_session_experiment as session_fixture
from test_snapshot_irl import config_for_test as irl_config_for_test
from test_weekly_library import config_for_test as library_config_for_test
from test_time_execution import BASE, Model, prepared, quote


class VersionAttributionTests(unittest.TestCase):
    """把旧版本订单留到换版后评价，手工检查所有权与真实资金连续性。"""
    def replay(self, frame, preds, versions=None, **settings):
        selector = SingleModelSelector('fixed', [Model()]);selector.reward_type = 'OE'
        return TimeExecutionEngine(IRLRewardLearner(horizons=[1000, 3000], definition='paper_price_difference'),
            **settings).run_backtest(selector, frame, preds, detail=True, model_version_ids=versions)

    def test_old_position_and_reward_keep_owner_version_after_switch(self):
        """新信号可触发退出，但开仓/退出账本与到期 OE 都归属旧版本，仓位不清零。"""
        frame = prepared();preds = np.full((len(frame), 1), .01);preds[35:, 0] = -.01
        versions = np.where(np.arange(len(frame)) < 35, 'old', 'new')
        result = self.replay(frame, preds, versions)
        first = result['trades'][0]
        self.assertEqual(first['model_version_id'], 'old')
        self.assertEqual(first['exit_trigger_model_version_id'], 'new')
        self.assertEqual(result['fills'][0]['model_version_id'], 'old')
        self.assertEqual(result['fills'][1]['model_version_id'], 'old')
        later = [r for r in result['reward_observations'] if r['model_version_id']=='old'
                 and pd.Timestamp(r['observed_time']) >= frame.ts_event.iloc[35]]
        self.assertTrue(later)
        self.assertGreater(result['cancelled_intents']['model_version_change'], 0)
        self.assertAlmostEqual(sum(t['net_pnl_usd'] for t in result['trades']), result['net_pnl_usd'])
        self.assertEqual(sum(a.get('OE_matured', 0) for a in result['model_version_audit'].values()),
                         result['matured_order_count'])

    def test_switch_cancels_unexecuted_old_intents(self):
        """长延迟的旧意图不得在新版本生效后才开仓，新意图仍等待自己的到期。"""
        frame = prepared();preds = np.full((len(frame),1), .01);preds[35:,0] = -.01
        versions = np.where(np.arange(len(frame)) < 35, 'old', 'new')
        result = self.replay(frame, preds, versions, latency_ms=5000)
        self.assertTrue(all(f['model_version_id']=='new' for f in result['fills']))
        self.assertEqual(pd.Timestamp(result['fills'][0]['intent_time']), frame.ts_event.iloc[35])
        self.assertEqual(pd.Timestamp(result['fills'][0]['ts_event']), frame.ts_event.iloc[45])

    def test_constant_version_preserves_default_numbers_and_ledger(self):
        """附加身份审计不能改变无换版基线的信号、成交、成本和奖励数值。"""
        frame = prepared();preds = np.full((len(frame), 1), .01)
        original = self.replay(frame, preds)
        tagged = self.replay(frame, preds, ['same']*len(frame))
        for key in ('total_fills','total_trades','net_pnl_usd','gross_pnl_usd','friction_usd',
                    'matured_order_count','order_feature_expectation','reward_status','cancelled_intents'):
            self.assertEqual(original[key], tagged[key])
        for a,b in zip(original['fills'],tagged['fills']):
            self.assertEqual(a,{k:v for k,v in b.items() if k not in ('model_version_id','trigger_model_version_id')})

    def test_missing_tail_and_version_validation_preserve_risk(self):
        """无尾部退出报价的旧版本持仓仍可见，版本形状/可见性不能模糊匹配。"""
        frame = prepared();preds = np.zeros((len(frame),1));preds[-5:,0] = .01
        result = self.replay(frame, preds, ['old']*len(frame), force_replay_end=False)
        self.assertFalse(result['terminal_position_liquidated'])
        self.assertEqual(result['terminal_position_model_version_id'],'old')
        self.assertGreater(result['unmatured_fill_rewards_at_end'],0)
        self.assertGreater(result['unexecuted_intents_at_end'],0)
        for invalid in [['old'], [None]*len(frame), ['']*len(frame)]:
            with self.subTest(invalid=invalid[:2]), self.assertRaisesRegex(ValueError, 'Visible version'):
                self.replay(frame,preds,invalid)


class LibraryOEIntegrationTests(unittest.TestCase):
    """七个合成完整 session 跨周，校准先于测试，全部候选和状态都保留。"""
    prepare = session_fixture.SessionExperimentTests.prepare

    def setUp(self):
        session_fixture.SessionExperimentTests.setUp(self)
        self.days += ['2025-10-06','2025-10-07']
        rows = []
        for d,day in enumerate(self.days):
            stamps = [self.calendar.sessions[day]['open']+pd.Timedelta(minutes=1)]
            stamps += list(pd.date_range(day+'T14:00:00Z',day+'T21:00:00Z',freq='min'))
            rows += [quote(t,100.+d+.25*(i%23)) for i,t in enumerate(stamps)]
        self.raw_rows = pd.DataFrame(rows)
        self.dataset = self.prepare(self.raw_rows,'seven-session-data')
        self.protocol['sessions'] = dict(train=self.days[:2],calibration=self.days[2:3],
                                        validation=self.days[3:5],test=self.days[5:])
        self.config.write_text(json.dumps(self.protocol))
        self.library_config = library_config_for_test() | dict(through_session=self.days[-1],evaluation_sessions=self.days[2:])
        self.library_path = self.root/'library-config.json';self.library_path.write_text(json.dumps(self.library_config))
        self.irl = irl_config_for_test();(self.root/'irl.json').write_text(json.dumps(self.irl))
        self.oe_config = dict(schema_version=1,purpose='development_library_time_OE_IRL',
            session_protocol_file='protocol.json',irl_config_file='irl.json',
            model_update_policy='predeclared_weekly_visible',session_boundary_policy='independent_daily_accounts_no_feedback_carry')
        self.oe_path = self.root/'oe.json';self.oe_path.write_text(json.dumps(self.oe_config))

    def frozen(self, dataset=None, name='experiment'):
        directory = self.root/name;directory.mkdir()
        library_plan = freeze_library(self.library_path,[dataset or self.dataset])
        lp = directory/'library-plan.json';lp.write_text(json.dumps(library_plan))
        artifact = build_library(lp)
        ap = directory/'library.json';ap.write_text(json.dumps(artifact))
        plan = main(['freeze','--config',str(self.oe_path),'--library',str(ap),'--output-dir',str(directory/'frozen')])
        return artifact, plan, directory/'frozen/plan.json'

    def test_cli_all_candidates_cross_week_and_frozen_selection(self):
        """16 个候选和现金/等权对照全保留，新周只换参数不按测试结果改身份。"""
        artifact,plan,path = self.frozen()
        result = main(['run','--plan',str(path),'--output-dir',str(self.root/'result')])
        self.assertEqual(result,run_library_oe(path))
        self.assertEqual(result,json.loads((self.root/'result/result.json').read_text()))
        self.assertEqual(len(plan['policies']),16)
        case = result['age_cases'][0]
        for phase,data in case['phases'].items():
            self.assertEqual(len(data['statistics']),16)
            for day in data['daily']:
                expected = artifact['age_cases'][0]['versions'][0 if day['session_id']<'2025-10-06' else 1]['version_sha256']
                self.assertEqual(day['library_version_ids'],[expected])
                for row in day['results']:
                    boundary = row['session_boundary']
                    self.assertEqual(boundary['unmatured_OE_not_forwarded'],row['unmatured_fill_rewards_at_end'])
                    self.assertEqual(boundary['unexecuted_intents_not_forwarded'],row['unexecuted_intents_at_end'])
                    self.assertEqual(row['replay_rows'],next(a for a in data['horizon_audits'] if a['session_id']==day['session_id'])['observed_rows'])
            if phase!='calibration':
                for name,fit in case['reward_fits'].items():
                    self.assertEqual(data['frozen_evaluations'][name]['selected_policy_id'],
                        fit['selected_policy']['selected_policy_id'] if fit['selected_policy'] else None)
                cash = data['frozen_evaluations']['cash']['result']
                self.assertEqual(cash['net_pnl_usd'],0.)
                self.assertIsNone(cash['order_feature_expectation'])
        with self.assertRaises(FileExistsError):
            main(['run','--plan',str(path),'--output-dir',str(self.root/'result')])

    def test_future_prices_cannot_change_calibration_or_early_versions(self):
        """改验证/测试价格仅可影响后续可用模型和后续盈亏，不能回流校准选择。"""
        original_library,_,first = self.frozen()
        original = run_library_oe(first)['age_cases'][0]
        rows = self.raw_rows.copy();mask=rows.ts_event>=pd.Timestamp('2025-10-01T22:00:00Z')
        for name in rows:
            if name.startswith(('bid_px_','ask_px_')):
                rows.loc[mask,name]+=100.
        changed_data = self.prepare(rows,'changed-future')
        changed_library,_,second = self.frozen(changed_data,'changed-experiment')
        changed = run_library_oe(second)['age_cases'][0]
        self.assertEqual(original['reward_fits'],changed['reward_fits'])
        self.assertEqual(original['phases']['calibration'],changed['phases']['calibration'])
        self.assertEqual(original_library['age_cases'][0]['versions'][0],changed_library['age_cases'][0]['versions'][0])
        self.assertNotEqual(original_library['age_cases'][0]['versions'][1],changed_library['age_cases'][0]['versions'][1])

    def test_future_cache_never_gates_model_or_order(self):
        """离线有效标签抹去后，全部成交、资金、成熟 OE、专家和权重仍一致。"""
        _,_,first = self.frozen();original=run_library_oe(first)['age_cases'][0]
        other=self.prepare(self.raw_rows,'censored-cache');path=other/'snapshots.parquet'
        table=pq.read_table(path);frame=table.to_pandas()
        for name in frame:
            if name.startswith('label_valid_'):frame[name]=False
            elif name.startswith(('future_return_','future_price_diff_')):frame[name]=np.nan
        frame['learning_ready']=False
        pq.write_table(pa.Table.from_pandas(frame,preserve_index=False).replace_schema_metadata(table.schema.metadata),path)
        _,_,second=self.frozen(other,'censored-experiment')
        self.assertEqual(original,run_library_oe(second)['age_cases'][0])

    def test_fixed_reference_still_matches_original_time_baseline(self):
        """新增库不得改变原固定 Ridge 门槛 1/2/4 的基线成交、成本或 OE。"""
        artifact,plan,_=self.frozen()
        reader=PreparedDatasetReader(self.dataset,calendar=self.calendar)
        from src.session_experiment import fit_streaming_ridge
        scaler,model,_=fit_streaming_ridge(reader,self.days[:2],60000,3660000)
        specs=policy_specs(self.irl,self.protocol['threshold'])
        old,_=replay_stage(reader,self.days[2:3],scaler,model,specs,self.protocol)
        new,_=replay_candidates(reader,self.days[2:3],artifact['age_cases'][0],plan['policies'],self.protocol)
        for a,b in zip(old[0]['results'],new[0]['results']):
            for key in a:
                self.assertEqual(a[key],b[key],key)

    def test_fractional_threshold_candidate_does_not_replace_x1_baseline(self):
        """候选可有小于 1 的乘数，但固定 ×1 对照不能误选列表中的首个门槛。"""
        self.irl['threshold_multipliers']=[.5,1,2]
        (self.root/'irl.json').write_text(json.dumps(self.irl))
        _,_,path=self.frozen()
        case=run_library_oe(path)['age_cases'][0]
        for phase in ('validation','test'):
            self.assertEqual(case['phases'][phase]['frozen_evaluations']['fixed_reference_x1']['selected_policy_id'],
                             'Ridge-threshold-x1')

    def test_solver_failure_preserves_all_candidates(self):
        """求解失败仍保留全部订单和负收益，不发布可用冻结选择。"""
        _,plan,path=self.frozen()
        with patch('src.library_oe.learn_calibration_reward',side_effect=lambda stats,hs,cfg,name:
                   {'status':'solver_failed','weights':None,'expert':None,'selected_policy':None,'equal_weight_selected_policy':None}):
            result=run_library_oe(path)['age_cases'][0]
        self.assertTrue(all(f['status']=='solver_failed' for f in result['reward_fits'].values()))
        self.assertTrue(all(len(p['statistics'])==16 for p in result['phases'].values()))
        self.assertTrue(all(p['frozen_evaluations']['simplex']['result'] is None for name,p in result['phases'].items() if name!='calibration'))

    def test_zero_maturity_does_not_fit_cash_only_reward(self):
        """校准块全部短于最长前瞻时，现金零向量不能被当作已完成奖励学习。"""
        rows=self.raw_rows.copy()
        day=self.calendar.sessions[self.days[2]]
        drop=((rows.ts_event>=day['open']) & (rows.ts_event<=day['close'])
              & rows.ts_event.dt.minute.isin([0,40]))
        other=self.prepare(rows.loc[~drop].copy(),'short-calibration-blocks')
        _,_,path=self.frozen(other,'no-maturity-experiment')
        case=run_library_oe(path)['age_cases'][0]
        self.assertTrue(all(s['matured_order_count']==0 for s in case['phases']['calibration']['statistics']))
        self.assertTrue(all(f['status']=='blocked_no_matured_trading_policy' for f in case['reward_fits'].values()))
        self.assertTrue(all(f['weights'] is None for f in case['reward_fits'].values()))

    def test_protocol_artifact_source_and_data_guards(self):
        """协议漂移、源码更换、模型文件或数据改写须重新冻结，不能省略候选。"""
        artifact,plan,path=self.frozen()
        with patch('src.library_oe.experiment_code_hashes',return_value={}):
            with self.assertRaisesRegex(ValueError,'Source changed'):run_library_oe(path)
        changed=deepcopy(plan);changed['policies'].pop();path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'integrity'):run_library_oe(path)
        path.write_text(json.dumps(plan))
        ap=self.root/'experiment/library.json';ap.write_text(ap.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Library artifact changed'):run_library_oe(path)
        ap.write_text(json.dumps(artifact))
        self.protocol['purge_ms']=1;self.config.write_text(json.dumps(self.protocol))
        with self.assertRaises(ValueError):freeze_library_oe(self.oe_path,ap)
        quality=self.dataset/'quality.json';quality.write_text(quality.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Prepared dataset changed'):run_library_oe(path)


if __name__=='__main__':
    unittest.main()
