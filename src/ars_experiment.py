"""最近窗口与平移成熟窗口 ARS 双对照，保留相同 UCB/轮换和静态基线。

研究目标是验证历史子回测独立性与因果评分，等权奖励不冒充已学成的 IRL。
选择期间、回测窗口和观察前瞻分别冻结，全部评价日期都为开发用途。
"""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR,INSTRUMENT_CONFIG
from src.history_ars import HistoryARSSelector,HistoryWindowBacktester
from src.irl_reward import IRLRewardLearner
from src.library_oe import load_library
from src.period_experiment import code_hashes as period_code_hashes, freeze_period_experiment, replay_day
from src.session_experiment import fingerprint,read_day,summarize_days
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library


def code_hashes():
    """绑定旧期间协议及新历史回测、选择和入口，不将旧实验身份用于新代码。"""
    return period_code_hashes()|{p:sha256_file(BASE_DIR/p) for p in
        ('src/history_ars.py','src/ars_experiment.py','run_history_ars.py')}


def freeze_ars_experiment(config_path,library_path):
    """借用已验证协议绑定数据，不重新按后续收益选年龄、期间或窗口对齐。"""
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text())
    required={'schema_version','purpose','period_protocol_file','history_window_ms','alignments','tie_tolerance_reward'}
    if (set(config)-required-{'notes'} or required-set(config) or config['schema_version']!=1
            or config['purpose']!='development_history_OE_ARS'
            or config['alignments']!=['recent','matured']
            or type(config['history_window_ms']) is not int or config['history_window_ms']<=0
            or type(config['tie_tolerance_reward']) not in (float,int)
            or not np.isfinite(config['tie_tolerance_reward']) or config['tie_tolerance_reward']<0):
        raise ValueError('Predeclared recent/matured ARS development controls required')
    base=freeze_period_experiment(config_path.parent/config['period_protocol_file'],library_path)
    if any(config['history_window_ms']%c['session_binding']['interval_ms'] for c in base['cases']):
        raise ValueError('Historical window must align with snapshot grid')
    plan=dict(schema_version=1,plan_kind='frozen_history_OE_ARS',config=config,
        config_source=dict(path=str(config_path),sha256=sha256_file(config_path)),
        base_period_plan=base,strategies=base['strategies']+['History-OE-ARS-'+a for a in config['alignments']],
        code_sha256=code_hashes(),holdout_claim='none_development_dates',age_selection='none',
        window_account_policy='independent_flat_force_exact_known_window_end_quote_else_undefined',
        current_model_reward_policy='latest_matured_live_period_original_version',
        other_model_reward_policy='fresh_window_replay_all_orders_or_undefined')
    plan['plan_sha256']=fingerprint(plan)
    return plan


def replay_ars_day(reader,day,case,protocol,plan,*,detail=False):
    """所有策略使用相同费用和因果周版本；新增 ARS 不改变已有对照执行路径。"""
    base=plan['base_period_plan']
    daily=replay_day(reader,day,case,protocol,base,detail=detail)
    frame=read_day(reader,day);matrix,versions=predict_library(case['versions'],frame,reader.report['interval_ms'])
    quotes=frame[SNAPSHOT_SCHEMA.names+['session_id','segment_id','mid_price','feature_valid']].copy()
    valid=frame.feature_valid.to_numpy();quotes['feature_valid']=valid & np.isfinite(matrix).all(axis=1)
    models=[SimpleNamespace(name=m) for m in base['model_ids']]
    for alignment in plan['config']['alignments']:
        reward=IRLRewardLearner(horizons=protocol['reward_horizons_ms'],definition='paper_price_difference')
        backtester=HistoryWindowBacktester(frame,case['versions'],reader.calendar,reward,
            interval_ms=reader.report['interval_ms'],latency_ms=protocol['latency_ms'],
            holding_review_ms=protocol['holding_review_ms'],threshold=protocol['threshold'])
        selector=HistoryARSSelector('History-OE-ARS-'+alignment,models,backtester,
            period_ms=base['config']['selection_period_ms'],window_ms=plan['config']['history_window_ms'],
            alignment=alignment,tie_tolerance=plan['config']['tie_tolerance_reward'])
        engine=TimeExecutionEngine(reward,interval_ms=reader.report['interval_ms'],calendar=reader.calendar,
            latency_ms=protocol['latency_ms'],holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'],force_replay_end=False)
        result=engine.run_backtest(selector,quotes,matrix,detail=detail,model_version_ids=versions)
        histories=[e for p in result['period_selections'] for e in p['evaluations'] if e['source']=='independent_history_replay']
        result.pop('exploration_c_price')  # ARS 没有 UCB 探索项，不让继承的占位 C=0 混入解释。
        result.update(history_window_ms=plan['config']['history_window_ms'],history_alignment=alignment,
            current_prediction_unavailable_rows=int((valid&~quotes.feature_valid.to_numpy()).sum()),
            history_status_counts=dict(Counter(e['status'] for e in histories)),
            scored_history_windows=sum(e['score'] is not None for e in histories),
            cold_start_periods=sum(p['cold_start'] for p in result['period_selections']),
            period_assumptions='hold_period_choice_latest_live_selected_fresh_history_others_observable_scores_only',
            version_statistics_role='cumulative_live_period_feedback_audit_not_ARS_ranking_W')
        daily['results'].append(result)
    return daily


def run_ars_experiment(plan_path,*,detail=False):
    """冻结后所有年龄和策略完整运行，历史评分只读取当前可见前缀。"""
    plan=json.loads(Path(plan_path).read_text());base=plan.get('base_period_plan',{})
    if (plan.get('schema_version')!=1 or plan.get('plan_kind')!='frozen_history_OE_ARS'
            or plan.get('plan_sha256')!=fingerprint(plan) or plan['code_sha256']!=code_hashes()
            or base.get('plan_sha256')!=fingerprint(base) or base['code_sha256']!=period_code_hashes()):
        raise ValueError('Frozen ARS plan/source integrity check failed')
    source=base['library_source']
    if sha256_file(source['path'])!=source['sha256']:raise ValueError('Library artifact changed')
    library,calendar=load_library(source['path'])
    if library['plan']['config']!=base['library_configuration']:raise ValueError('Library configuration changed')
    cases=[]
    with threadpool_limits(limits=1):
        for frozen in base['cases']:
            binding=frozen['session_binding'];protocol=binding['protocol']
            if binding['plan_sha256']!=fingerprint(binding):raise ValueError('Session binding changed')
            reader=PreparedDatasetReader(binding['dataset']['path'],calendar=calendar,
                allow_partial=protocol['allow_partial'],include_degraded=protocol['include_degraded'])
            if reader.provenance!=binding['dataset']:raise ValueError('Prepared dataset changed')
            case=next(c for c in library['age_cases'] if c['max_age_ms']==frozen['max_age_ms'])
            phases={}
            for phase in base['config']['stages']:
                daily=[replay_ars_day(reader,day,case,protocol,plan,detail=detail) for day in protocol['sessions'][phase]]
                statistics=[]
                for strategy in plan['strategies']:
                    total=summarize_days(daily,strategy);histories=Counter();periods=Counter();cold=scored=0
                    for day in daily:
                        r=next(r for r in day['results'] if r['strategy']==strategy)
                        histories.update(r.get('history_status_counts',{}));periods.update(r.get('period_status_counts',{}))
                        cold+=r.get('cold_start_periods',0);scored+=r.get('scored_history_windows',0)
                    statistics.append(total|dict(history_status_counts=dict(histories),period_status_counts=dict(periods),
                        cold_start_periods=cold,scored_history_windows=scored))
                phases[phase]=dict(daily=daily,statistics=statistics)
                print(f"完成 {frozen['max_age_ms']}ms {phase} 的 18 策略与 ARS 双窗口对照",flush=True)
            cases.append(dict(max_age_ms=frozen['max_age_ms'],phases=phases))
    if plan['code_sha256']!=code_hashes() or sha256_file(source['path'])!=source['sha256']:
        raise ValueError('Source/library changed during experiment')
    return dict(schema_version=1,result_kind='history_OE_ARS_development',plan=plan,plan_sha256=plan['plan_sha256'],
        age_cases=cases,selected_age_ms=None,instrument=INSTRUMENT_CONFIG['CME_ES'],
        environment=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ('numpy','pandas','pyarrow','scikit-learn','scipy','threadpoolctl')}),
        limits=['Algorithm 3 的明确工程实现；严格最近与向前平移成熟窗口均保留，不替论文消除歧义。',
                '当前模型用最新完整成熟真实期间，其他模型独立重回放；两种反馈时段/账户不同。',
                '窗口从空仓/空意图开始并重建因果特征；窗口末确切可交易网格主动退出为子回测假设。',
                '任何订单失效/未成熟、无订单、缺窗口边界或版本历史不足都不可评分，不补零。',
                '只在可观察分数中选最大值；全不可观察按声明顺序轮换，不延用过期历史窗分数。',
                '仍是等权奖励控制，没有成功 IRL、时间 ME IRL、真实跨周或全量复现。'])


def render_ars_experiment(result):
    """全部策略和历史评分失败覆盖从同一 JSON 输出，保留负结果。"""
    lines=['# 历史窗口 OE-ARS 双对照','',f"计划 `{result['plan_sha256']}`；等权奖励开发控制。",'',
        '| 年龄 ms | 阶段 | 策略 | 成交 | 成熟订单 | 毛利 USD | 摩擦 USD | 净利 USD | 可评分历史窗 | 冷启动期间 | 历史状态 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for phase,data in case['phases'].items():
            for row in data['statistics']:
                lines.append(f"| {case['max_age_ms']} | {phase} | {row['strategy']} | {row['total_fills']} | "
                    f"{row['matured_order_count']} | {row['gross_pnl_usd']} | {row['friction_usd']} | {row['net_pnl_usd']} | "
                    f"{row['scored_history_windows']} | {row['cold_start_periods']} | {row['history_status_counts']} |")
    lines+=['','## 限制','']+['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'
