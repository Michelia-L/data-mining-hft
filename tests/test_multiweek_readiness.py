"""多周真实版本覆盖、校准成熟资格与不消费后段交易收益的性质测试。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from run_multiweek_readiness import main
from src.multiweek_readiness import (calibration_prefix_coverage, freeze_readiness,
    prediction_day_audit, run_readiness, training_visibility_audit)
from src.library_oe import load_library
from src.session_experiment import read_day
from src.snapshot_backtest import PreparedDatasetReader
import test_period_ucb as period_fixture


class CalibrationPrefixTests(unittest.TestCase):
    """每日均值不能替代订单数合并；增加天数不自动赋予所有模型观察资格。"""
    def test_missing_mature_orders_resolve_only_when_actual_observations_arrive(self):
        """首日只有a成熟，第二日b实际成熟后才补齐；未平仓仍不可作净利专家。"""
        specs=[dict(policy_id=n,cash=False,threshold=.01,source='weekly_library') for n in ('a','b')]
        daily=[]
        for day,counts in [('d1',[2,0]),('d2',[0,3])]:
            results=[dict(strategy=n,total_trades=1,total_fills=4,terminal_position_liquidated=True,
                gross_pnl_usd=5.,friction_usd=2.,net_pnl_usd=3.,matured_order_count=k,
                unmatured_fill_rewards_at_end=4-k,order_feature_expectation=[1.,2.]) for n,k in zip(('a','b'),counts)]
            daily.append(dict(session_id=day,results=results))
        rows=calibration_prefix_coverage(daily,specs,[1000,3000],1)
        self.assertEqual(rows[0]['missing_library_policy_ids'],['b'])
        self.assertEqual(rows[1]['missing_library_policy_ids'],[])
        self.assertEqual(rows[1]['matured_orders_by_policy'],{'a':2,'b':3})
        daily[-1]['results'][1]['terminal_position_liquidated']=False
        self.assertEqual(calibration_prefix_coverage(daily,specs,[1000,3000],1)[-1]['missing_library_policy_ids'],['b'])


class MultiweekReadinessTests(unittest.TestCase):
    """七个合成session的两周夹具，单独测试三周配置门槛和真实预测版本。"""
    def setUp(self):
        """复用现有真实prepared/建库夹具；不导入TestCase别名以免重复收集测试。"""
        self.fixture=period_fixture.PeriodExperimentTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups);self.f=self.fixture
        self.library,self.base,_=self.f.freeze();self.case=self.library['age_cases'][0]
        self.config=dict(schema_version=1,purpose='development_multiweek_OE_readiness',
            oe_protocol_file='oe.json',minimum_weekly_versions=2)
        self.path=self.f.f.root/'readiness.json';self.path.write_text(json.dumps(self.config))
        _,self.calendar=load_library(self.f.libpath)
        self.reader=PreparedDatasetReader(self.f.f.dataset,calendar=self.calendar)

    def test_cli_actual_versions_calibration_identity_and_no_post_calibration_profit(self):
        """两个真实预测版本均有共同可预测行；只有校准有静态账本，后段无策略收益。"""
        frozen=self.f.f.root/'ready-plan';output=self.f.f.root/'ready-result'
        plan=main(['freeze','--config',str(self.path),'--library',str(self.f.libpath),'--output-dir',str(frozen)])
        result=main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output)])
        self.assertEqual(result,json.loads((output/'result.json').read_text()))
        self.assertTrue((output/'report.md').exists())
        case=result['age_cases'][0]
        self.assertEqual(plan['expected_update_sessions'],['2025-10-01','2025-10-06'])
        self.assertTrue(case['readiness']['real_weekly_versions_verified'])
        self.assertFalse(case['readiness']['dynamic_backtest_run'])
        self.assertFalse(case['readiness']['formal_replication_ready'])
        self.assertIsNone(case['active_reward_weights'])
        self.assertEqual(len(case['calibration']['statistics']),16)
        for phase in ('validation','test'):
            data=case['phases'][phase]
            self.assertFalse(data['strategy_profit_evaluated'])
            self.assertEqual(data['calibration_sha256'],case['calibration_sha256'])
            self.assertNotIn('net_pnl_usd',json.dumps(data))
            self.assertNotIn('results',data)
            for row in data['prediction_audits']:
                self.assertEqual(row['future_version_rows'],0)
                expected='2025-10-01' if row['session_id']<'2025-10-06' else '2025-10-06'
                self.assertEqual([v['update_session'] for v in row['versions']],[expected])
        with self.assertRaises(FileExistsError):main(['run','--plan',str(frozen/'plan.json'),'--output-dir',str(output)])

    def test_three_week_requirement_rejects_only_two_weeks(self):
        """只构建两个真实周版不能满足本轮三周配置，不把候选模型数量当版本数。"""
        self.config['minimum_weekly_versions']=3;self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,'distinct weekly'):freeze_readiness(self.path,self.f.libpath)
        self.config['minimum_weekly_versions']=True;self.path.write_text(json.dumps(self.config))
        with self.assertRaises(ValueError):freeze_readiness(self.path,self.f.libpath)

    def test_prediction_ignores_future_labels_and_later_model_parameters(self):
        """未来标签列全失效、未来周参数改变，先前日期预测哈希与覆盖完全相同。"""
        day='2025-10-02';protocol=self.base['cases'][0]['session_binding']['protocol']
        original=prediction_day_audit(self.reader,day,self.case,protocol['reward_horizons_ms'])
        frame=read_day(self.reader,day)
        for column in frame:
            if column.startswith('future_'):frame[column]=np.nan
            if column.startswith('label_valid_'):frame[column]=False
        changed=deepcopy(self.case)
        for model in changed['versions'][1]['models']:
            if model['family']=='Ridge':model['intercept']+=100.
        from src.session_experiment import fingerprint
        changed['versions'][1]['version_sha256']=fingerprint({k:v for k,v in changed['versions'][1].items() if k!='version_sha256'})
        with patch('src.multiweek_readiness.read_day',return_value=frame):
            self.assertEqual(original,prediction_day_audit(self.reader,day,changed,protocol['reward_horizons_ms']))

    def test_wrong_prediction_version_detected_independently(self):
        """模拟预测接口返回旧版身份，真实新周日期的审计应拒绝而非只保存该身份。"""
        from src.weekly_library import predict_library
        day='2025-10-06';frame=read_day(self.reader,day)
        matrix,ids=predict_library(self.case['versions'],frame,self.reader.report['interval_ms'])
        ids[:]=self.case['versions'][0]['version_sha256']
        with patch('src.multiweek_readiness.predict_library',return_value=(matrix,ids)):
            with self.assertRaisesRegex(ValueError,'future or stale'):
                prediction_day_audit(self.reader,day,self.case,[60000])

    def test_training_origins_and_counts_recomputed_before_version_visibility(self):
        """训练计数不能仅信模型元数据；未来训练时刻与缺最长前瞻purge都拒绝。"""
        config=self.base['library_configuration']
        rows=training_visibility_audit(self.reader,self.case,config)
        for v in rows:
            for m in v['models']:
                if m['latest_prediction_label_maturity_ns'] is not None:
                    self.assertLess(m['latest_prediction_label_maturity_ns'],v['available_at_ns'])
        broken=deepcopy(self.case);broken['versions'][0]['models'][0]['training_rows']+=1
        with self.assertRaisesRegex(ValueError,'coverage disagrees'):
            training_visibility_audit(self.reader,broken,config)
        bad_time=self.calendar.sessions['2025-10-01']['close'].value
        with patch('src.multiweek_readiness.history_samples',return_value=(None,None,np.array([bad_time]))):
            with self.assertRaisesRegex(ValueError,'purge'):
                training_visibility_audit(self.reader,self.case,config)

    def test_frozen_source_nested_plan_and_data_identity_rejected(self):
        """冻结后改学习源码、父协议或输入质量都不能无记录复用身份。"""
        plan=freeze_readiness(self.path,self.f.libpath);path=self.f.f.root/'ready-plan.json'
        path.write_text(json.dumps(plan))
        with patch('src.multiweek_readiness.code_hashes',return_value={}):
            with self.assertRaises(ValueError):run_readiness(path)
        from src.session_experiment import fingerprint
        broken=deepcopy(plan);broken['base_oe_plan']['irl_config']['minimum_matured_orders']+=1
        broken['plan_sha256']=fingerprint(broken);path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):run_readiness(path)
        path.write_text(json.dumps(plan));quality=self.f.f.dataset/'quality.json';quality.write_text(quality.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'Prepared dataset changed'):run_readiness(path)


if __name__=='__main__':unittest.main()
