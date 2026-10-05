"""冻结后段的研究性质：权重不重学、共同成熟、失败保留、日恢复和因果周更新。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from run_frozen_ablation import main
from src.frozen_ablation import (audit_day, frozen_rewards, jobs_for_case, load_prior,
    replay_day, run_ablation, summarize, validate_config)
from src.session_experiment import fingerprint, read_day
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import SessionCalendar, prepare_snapshot_dataset
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import fit_history
from test_history_ars import versions_for_test
from test_time_execution import BASE, prepared, quote


def previous_case():
    """可执行库内组及不可表示全候选专家并存；只构造冻结输入，不模拟拟合。"""
    groups = {}
    for scope in ('all_candidates', 'online_library'):
        weights = None if scope == 'all_candidates' else [1.]+[0.]*6
        groups[scope] = dict(active_reward_weights=weights, reward_fits=dict(sum_only=dict(weights=weights)),
            execution_gates=dict(sum_only=dict(usable_for_execution=weights is not None,
                reasons=['expert_outside_online_library'] if weights is None else [],
                expert_policy_id='cash' if weights is None else 'Library-Ridge-price-h1')))
    return dict(me_calibration=dict(scopes=deepcopy(groups)), oe_calibration=deepcopy(groups))


class FrozenAblationTests(unittest.TestCase):
    """小行情手算检查，不以亏损或获利来决定正确性。"""
    def setUp(self):
        self.calendar = SessionCalendar()
        self.config = json.loads(Path('config/esz5_final_frozen_ablation.json').read_text())

    def test_protocol_rejects_removed_age_window_scope_and_days(self):
        """禁止删失败来源或双ARS，也禁止择日删去连续评价区间或扩至十日以上。"""
        validate_config(self.config, self.calendar)
        for delta in (dict(age_candidates_ms=[1000]), dict(selectors=['UCB']),
                dict(expert_scopes=['online_library']), dict(maturity_policy='shortest_only'),
                dict(evaluation_sessions=self.config['evaluation_sessions'][::2]),
                dict(evaluation_sessions=[d for d in self.calendar.sessions if '2025-10-22' <= d <= '2025-11-10'])):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                validate_config(self.config | delta, self.calendar)

    def test_inherited_weights_unchanged_and_blocked_is_not_cash(self):
        """旧权重逐值保留，两来源各有三策略；阻断None不变成现金美元0。"""
        prior = previous_case(); unchanged = deepcopy(prior)
        rewards = frozen_rewards(prior, [5000,15000,45000,135000,405000,1215000,3645000])
        self.assertEqual(prior, unchanged)
        jobs = jobs_for_case(rewards)
        self.assertEqual(len(jobs), 24); self.assertEqual(sum(j['weights'] is None for j in jobs), 6)
        self.assertEqual(rewards['ME']['single_shortest']['weights'], [1.,0.,0.,0.,0.,0.,0.])
        self.assertIsNone(rewards['OE']['all_candidates']['weights'])
        prior['oe_calibration']['online_library']['active_reward_weights'] = [1/7]*7
        with self.assertRaisesRegex(ValueError, 'weights'): frozen_rewards(prior, [1,2,3,4,5,6,7])

    def short_replay(self, frame=None):
        """二尺度合成实验缩小运行规模：零长权仍等待3秒，不把毫秒当行数。"""
        frame = prepared() if frame is None else frame
        groups = previous_case()
        for kind in (groups['me_calibration']['scopes'], groups['oe_calibration']):
            kind['online_library']['active_reward_weights'] = [1.,0.]
            kind['online_library']['reward_fits']['sum_only']['weights'] = [1.,0.]
        rewards = frozen_rewards(groups, [1000,3000])
        versions = versions_for_test(self.calendar)
        case = dict(versions=versions, rewards=rewards, protocol=dict(reward_horizons_ms=[1000,3000],
            latency_ms=500, holding_review_ms=1000, threshold=.000015))
        plan = dict(model_ids=['long','short'], selector_parameters=dict(period_ms=5000,
            exploration_c_price=1., history_window_ms=40000, tie_tolerance_reward=1e-9),
            strategies=[j['strategy'] for j in jobs_for_case(rewards)]+['cash','Fixed-Ridge-price-h1','Mean-Ensemble'])
        reader = SimpleNamespace(report=dict(interval_ms=500), calendar=self.calendar)
        with patch('src.frozen_ablation.read_day', return_value=frame):
            rows = replay_day(reader, '2025-09-22', case, plan)
        audits = audit_day(rows, case, plan)
        return rows, case, plan, audits

    def test_zero_long_weights_keep_maturity_and_same_cost_ledger(self):
        """单尺度权重仍全向量成熟，所有策略同成本守恒；现金确实执行且为0。"""
        rows, case, plan, audits = self.short_replay()
        self.assertEqual(len(rows),27);self.assertEqual(sum(r['run_status']=='blocked' for r in rows),6)
        for row in rows:
            if row['run_status']=='blocked': self.assertIsNone(row['net_pnl_usd']);continue
            self.assertAlmostEqual(row['gross_pnl_usd']-row['friction_usd'],row['net_pnl_usd'])
            for period in row.get('period_feedback',[]):
                if period['updated_selector']:
                    self.assertGreaterEqual(period['observed_at_ns'],period['last_origin_ns']+3_000_000_000)
        cash=next(r for r in rows if r['strategy']=='cash');self.assertEqual(cash['net_pnl_usd'],0.)
        single=next(r for r in rows if r['strategy']=='ME-single_shortest-UCB')
        self.assertTrue(any(p['updated_selector'] for p in single['period_feedback']))
        corrupted=deepcopy(rows);next(r for r in corrupted if r['strategy']==single['strategy'])['reward_weights']=[0.,1.]
        with self.assertRaises(ValueError):audit_day(corrupted,case,plan)
        days=[dict(results=rows,dynamic_audits=audits)]*2
        totals=summarize(days,plan['strategies'])
        self.assertFalse(single['terminal_position_liquidated'])
        self.assertIsNone(next(r for r in totals if r['strategy']==single['strategy'])['net_pnl_usd'])
        self.assertEqual(next(r for r in totals if r['strategy']=='cash')['net_pnl_usd'],0.)
        self.assertIsNone(next(r for r in totals if r['run_status']=='blocked')['net_pnl_usd'])

    def test_changed_future_cannot_change_early_choices(self):
        """后段行情与离线标签修改不能影响此前ME/OE期间选择，奖励继承不拟合。"""
        a,_,_,_=self.short_replay();changed=prepared();mask=changed.ts_event>BASE+pd.Timedelta(seconds=35)
        for column in ('mid_price','bid_px_00','ask_px_00'):changed.loc[mask,column]+=50.
        for column in changed:
            if column.startswith(('future_', 'label_')):changed[column]=False if changed[column].dtype==bool else np.nan
        b,_,_,_=self.short_replay(changed)
        cutoff=BASE.value+35_000_000_000
        for x,y in zip(a,b):
            self.assertEqual([p for p in x.get('period_selections',[]) if p['selected_at_ns']<=cutoff],
                [p for p in y.get('period_selections',[]) if p['selected_at_ns']<=cutoff])

    def test_prior_wrong_result_hash_is_rejected(self):
        """旧证据不能与另一个完整结果混用，也不能把代码变更静默套在旧校准上。"""
        with tempfile.TemporaryDirectory() as root:
            evidence=Path(root)/'evidence.json';result=Path(root)/'old.json';result.write_text('{}')
            evidence.write_text(Path('results/final_report/four_combinations_snapshot.json').read_text())
            with self.assertRaisesRegex(ValueError,'integrity'):load_prior(evidence,result)


class FrozenPipelineTests(unittest.TestCase):
    """外部旧校准用小固定证据替代；新三年龄数据、训练、回放、CLI和恢复是真流程。"""
    def test_cli_weekly_extension_restore_and_frozen_source_rejection(self):
        """先冻结再建后周；已评价23/24日仅在27日训练可用，恢复不重训/重交易。"""
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);calendar=SessionCalendar()
            config=json.loads(Path('config/esz5_final_frozen_ablation.json').read_text())
            days=['2025-10-16','2025-10-17']+config['evaluation_sessions']
            horizons=[5000,15000,45000,135000,405000,1215000,3645000]
            rows=[]
            for i,day in enumerate(days):
                opening=calendar.sessions[day]['open']+pd.Timedelta(milliseconds=500)
                stamps=[opening]+list(pd.date_range(day+'T14:00:00Z',periods=180,freq='500ms'))+[calendar.sessions[day]['close']]
                rows += [quote(t,100.+i+.25*(j%19)) for j,t in enumerate(stamps)]
            raw=pd.DataFrame(rows);datasets=[]
            for age in config['age_candidates_ms']:
                directory=root/f'age-{age}';directory.mkdir();inputs=[]
                for day,part in raw.groupby(raw.ts_event.dt.date.astype(str),sort=True):
                    metadata=dict(sampling_clock='ts_recv',interval_ms='500',max_age_ms=str(age),
                        source_file='synthetic.raw',source_sha256='a'*64,source_date_utc=day,
                        partial_prefix='false',source_condition='available')
                    table=pa.Table.from_pandas(part[SNAPSHOT_SCHEMA.names],schema=SNAPSHOT_SCHEMA,preserve_index=False)
                    table=table.replace_schema_metadata({k.encode():v.encode() for k,v in metadata.items()})
                    path=directory/(day+'.parquet');pq.write_table(table,path);inputs.append(path)
                prepared_dir=directory/'prepared';prepare_snapshot_dataset(inputs,prepared_dir,horizons_ms=horizons)
                datasets.append(prepared_dir)
            library_config=json.loads(Path('config/esz5_multiweek_weekly_library.json').read_text())
            library_cases=[]
            for age,directory in zip(config['age_candidates_ms'],datasets):
                reader=PreparedDatasetReader(directory,calendar=calendar);models=[]
                for count in [1,2]:
                    for m in fit_history(reader,days[:2][-count:],library_config,calendar.sessions['2025-10-20']['open']):
                        models.append(m|dict(model_id=f'{m["family"]}-{m["feature_group"]}-h{count}',history_session_count=count))
                initial=dict(update_session='2025-10-20',available_at_utc=calendar.sessions['2025-10-20']['open'].isoformat(),
                    update_kind='weekly_first_session',training_sessions={'1':days[1:2],'2':days[:2]},models=models,
                    interval_ms=500,max_age_ms=age,prediction_horizon_ms=5000,purge_ms=3645000)
                initial['version_sha256']=fingerprint(initial)
                library_cases.append(dict(max_age_ms=age,versions=[initial]))
            protocol=json.loads(Path('config/esz5_multiweek_session_development.json').read_text())
            base=dict(library_configuration=library_config,model_ids=[m['model_id'] for m in models],
                cases=[dict(session_binding=dict(protocol=protocol))],config=dict(selection_period_ms=300000,exploration_c_price=1.))
            prior=dict(plan_sha256='prior-fixed',plan=dict(base_oe_dynamic_plan=dict(base_ars_plan=dict(base_period_plan=base,
                config=dict(history_window_ms=1800000,tie_tolerance_reward=1e-9)))),age_cases=[previous_case()|dict(max_age_ms=a) for a in config['age_candidates_ms']],
                source_result_sha256='')
            from src.snapshot_dataset import sha256_file
            prior_path=root/'prior.json';prior_path.write_text(json.dumps(prior));prior['source_result_sha256']=sha256_file(prior_path)
            evidence=root/'evidence.json';evidence.write_text(json.dumps(prior))
            cfg=root/'config.json';cfg.write_text(json.dumps(config));library=dict(age_cases=library_cases)
            with patch('src.frozen_ablation.load_prior',return_value=(prior,library,calendar)):
                plan=main(['freeze','--config',str(cfg),'--prior-evidence',str(evidence),'--prior-result',str(prior_path),
                    '--dataset-dirs',*[str(p) for p in datasets],'--output-dir',str(root/'frozen')])
                result=main(['run','--plan',str(root/'frozen/plan.json'),'--checkpoint-dir',str(root/'checks'),
                    '--output-dir',str(root/'result')])
                self.assertEqual(result,json.loads((root/'result/result.json').read_text()))
                self.assertTrue((root/'result/daily_net_usd.png').is_file())
                self.assertEqual(sum(len(d['results']) for c in result['age_cases'] for d in c['daily']),567)
                for case in result['age_cases']:
                    self.assertEqual([v['update_session'] for v in case['versions']],['2025-10-20','2025-10-27'])
                    self.assertEqual(case['versions'][1]['training_sessions']['2'],['2025-10-23','2025-10-24'])
                    for day in case['daily']:
                        expected='2025-10-20' if day['session_id']<'2025-10-27' else '2025-10-27'
                        self.assertEqual(day['prediction_audit']['versions'][0]['update_session'],expected)
                with patch('src.frozen_ablation.replay_day',side_effect=AssertionError('unexpected replay')), \
                     patch('src.frozen_ablation.build_library',side_effect=AssertionError('unexpected retraining')):
                    self.assertEqual(result,run_ablation(root/'frozen/plan.json',checkpoint_dir=root/'checks'))
                altered=deepcopy(plan);altered['cases'][0]['rewards']['ME']['online_library']['weights']=[1/7]*7
                altered['plan_sha256']=fingerprint(altered);bad=root/'bad.json';bad.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError,'inherited'):run_ablation(bad,checkpoint_dir=root/'checks')
                prior_path.write_text(prior_path.read_text()+' ')
                with self.assertRaisesRegex(ValueError,'source changed'):run_ablation(root/'frozen/plan.json',checkpoint_dir=root/'checks')
            with self.assertRaises(FileExistsError):main(['run','--plan',str(root/'frozen/plan.json'),
                '--checkpoint-dir',str(root/'checks'),'--output-dir',str(root/'result')])


if __name__=='__main__':
    unittest.main()
