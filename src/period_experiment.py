"""固定期间 OE-UCB 开发实验：预声明等权奖励，全部年龄与静态对照保留。

这里只检验 Algorithm 2 的选择/反馈时序；不使用此前不可表示专家的 IRL 权重。
复用同批周库和训练/执行协议，评价日期已有开发用途，不宣称未触碰测试。
"""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.irl_reward import IRLRewardLearner
from src.library_oe import experiment_code_hashes, freeze_library_oe, load_library
from src.model_selector import FlatSelector, SingleModelSelector
from src.period_ucb import PeriodOESelector
from src.session_experiment import fingerprint, read_day, summarize_days
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library, predict_model


def code_hashes():
    """期间协议与入口也绑定到计划；执行器改动后须另存重建模型库。"""
    return experiment_code_hashes() | {p:sha256_file(BASE_DIR/p) for p in
        ('src/period_ucb.py','src/period_experiment.py','run_period_ucb.py')}


def freeze_period_experiment(config_path, library_path):
    """不算收益就绑定协议、全部候选、时间参数与等权奖励控制。

    复用旧冻结函数的训练、质量、年龄及日历一致性检查，取其数据绑定；
    不继承旧入口的校准阶段角色或声称本入口重新学习过奖励。
    """
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text())
    required={'schema_version','purpose','oe_protocol_file','selection_period_ms',
              'exploration_c_price','reward_source','stages'}
    if (set(config)-required-{'notes'} or required-set(config) or config['schema_version']!=1
            or config['purpose']!='development_period_OE_UCB' or config['reward_source']!='equal_weight_control'
            or config['stages']!=['validation','test'] or type(config['selection_period_ms']) is not int
            or config['selection_period_ms']<=0 or type(config['exploration_c_price']) not in (int,float)
            or not np.isfinite(config['exploration_c_price']) or config['exploration_c_price']<0):
        raise ValueError('Unsupported fixed period experiment configuration')
    binding=freeze_library_oe(config_path.parent/config['oe_protocol_file'],library_path)
    if any(config['selection_period_ms']%c['session_binding']['interval_ms'] for c in binding['cases']):
        raise ValueError('Selection period must align with snapshot grid')
    models=[p['model_id'] for p in binding['policies'] if p['source']=='weekly_library']
    strategies=['Period-OE-UCB','Period-OE-RoundRobin','cash','Ridge-threshold-x1']+['Fixed-'+m for m in models]
    plan=dict(schema_version=1,plan_kind='frozen_period_OE_UCB',config=config,
        config_source=dict(path=str(config_path),sha256=sha256_file(config_path)),
        **{k:binding[k] for k in ('cases','library_source','library_configuration')},
        model_ids=models,strategies=strategies,code_sha256=code_hashes(),
        reward_source='predeclared_equal_weight_control_not_successful_IRL',
        stage_roles=dict(validation='development_period_protocol_diagnostics',test='development_no_retuning'),
        holdout_claim='none',age_selection='none',
        availability_policy='all_library_columns_finite_at_current_tick_for_dynamic_selectors')
    plan['plan_sha256']=fingerprint(plan)
    return plan


def replay_day(reader, day, case, protocol, plan, *, detail=False):
    """一日生成一次 N×K 预测；账户、反馈、模型选择统计不跨日迁移。

    库模型若失败/预热不足，动态选择器要求当下全部列可用才决策，不能根据
    未来标签屏蔽。固定候选各用自身可用性；现金及原 Ridge 保留独立对照。
    """
    frame=read_day(reader,day);valid=frame.feature_valid.to_numpy()
    matrix,versions=predict_library(case['versions'],frame,reader.report['interval_ms'])
    reference=case['fixed_reference']|dict(features=FEATURE_COLUMNS,family='Ridge')
    original=np.full((len(frame),1),np.nan)
    if reference['status']=='fitted' and valid.any():
        original[valid,0]=predict_model(reference,frame.loc[valid,FEATURE_COLUMNS])
    names=plan['model_ids'];models=[SimpleNamespace(name=n) for n in names]
    results=[]
    for strategy in plan['strategies']:
        if strategy.startswith('Period-'):
            preds=matrix;ids=versions
            selector=PeriodOESelector(strategy,models,period_ms=plan['config']['selection_period_ms'],
                c=plan['config']['exploration_c_price'],mode='ucb' if strategy=='Period-OE-UCB' else 'round_robin')
        elif strategy.startswith('Fixed-'):
            index=names.index(strategy[len('Fixed-'):]);preds=matrix[:,index:index+1];ids=versions
            selector=SingleModelSelector(strategy,[models[index]])
        else:
            preds=np.zeros((len(frame),1)) if strategy=='cash' else original
            ids=np.full(len(frame),'cash' if strategy=='cash' else 'fixed-'+fingerprint(reference),dtype=object)
            selector=(FlatSelector if strategy=='cash' else SingleModelSelector)(strategy,[SimpleNamespace(name=reference['name'])])
        selector.reward_type='OE'
        quotes=frame[SNAPSHOT_SCHEMA.names+['session_id','segment_id','mid_price','feature_valid']].copy()
        quotes['feature_valid']=valid & np.isfinite(preds).all(axis=1)
        reward=IRLRewardLearner(horizons=protocol['reward_horizons_ms'],definition='paper_price_difference')
        engine=TimeExecutionEngine(reward,interval_ms=reader.report['interval_ms'],calendar=reader.calendar,
            latency_ms=protocol['latency_ms'],holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'],force_replay_end=False)
        result=engine.run_backtest(selector,quotes,preds,detail=detail,model_version_ids=ids)
        result['current_prediction_unavailable_rows']=int((valid&~quotes.feature_valid.to_numpy()).sum())
        results.append(result)
    return dict(session_id=day,input_feature_valid_rows=int(valid.sum()),
                library_version_ids=sorted(set(versions)-{None}),results=results)


def run_period_experiment(plan_path, *, detail=False):
    """消费冻结身份；不接收参数覆盖，不根据后续收益改变 C、期间或候选。"""
    plan=json.loads(Path(plan_path).read_text())
    if (plan.get('schema_version')!=1 or plan.get('plan_kind')!='frozen_period_OE_UCB'
            or plan.get('plan_sha256')!=fingerprint(plan) or plan['code_sha256']!=code_hashes()):
        raise ValueError('Frozen period plan/source integrity check failed')
    source=plan['library_source']
    if sha256_file(source['path'])!=source['sha256']:raise ValueError('Library artifact changed')
    library,calendar=load_library(source['path']);cases=[]
    if library['plan']['config']!=plan['library_configuration']:raise ValueError('Library configuration changed')
    with threadpool_limits(limits=1):
        for frozen in plan['cases']:
            binding=frozen['session_binding'];protocol=binding['protocol']
            if binding['plan_sha256']!=fingerprint(binding):raise ValueError('Session binding changed')
            reader=PreparedDatasetReader(binding['dataset']['path'],calendar=calendar,
                allow_partial=protocol['allow_partial'],include_degraded=protocol['include_degraded'])
            if reader.provenance!=binding['dataset']:raise ValueError('Prepared dataset changed')
            case=next(c for c in library['age_cases'] if c['max_age_ms']==frozen['max_age_ms'])
            phases={}
            for phase in plan['config']['stages']:
                daily=[replay_day(reader,day,case,protocol,plan,detail=detail) for day in protocol['sessions'][phase]]
                summaries=[]
                for strategy in plan['strategies']:
                    total=summarize_days(daily,strategy)
                    totals=Counter()
                    for day in daily:
                        r=next(r for r in day['results'] if r['strategy']==strategy)
                        totals.update(r.get('period_status_counts',{}))
                    summaries.append(dict(**total,period_status_counts=dict(totals)))
                phases[phase]=dict(daily=daily,statistics=summaries)
            cases.append(dict(max_age_ms=frozen['max_age_ms'],phases=phases))
            print(f"完成 {frozen['max_age_ms']}ms 固定期间 UCB/轮换及全部固定模型对照",flush=True)
    if plan['code_sha256']!=code_hashes() or sha256_file(source['path'])!=source['sha256']:
        raise ValueError('Source/library changed during experiment')
    import importlib.metadata
    import platform
    return dict(schema_version=1,result_kind='period_OE_UCB_development',plan=plan,plan_sha256=plan['plan_sha256'],
        age_cases=cases,selected_age_ms=None,
        environment=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in
            ('numpy','pandas','pyarrow','scikit-learn','scipy','threadpoolctl')}),
        limits=['等权奖励控制，不是成功 IRL 的 FMATO-OE-UCBS。',
                '5 分钟取论文例子；C=1 价格点为项目假设，未按测试调参。',
                '期间锁定模型、等权期间 W、n=选择期间数、延迟/无订单/归属均为明确工程约定。',
                '完整期间中任何订单标签失效则整桶不学习；无订单/末端未成熟不补零。',
                '日账户/反馈独立；换版分开 W/n，旧反馈只归旧版本，不实现 ARS 或时间 ME IRL。',
                '真实开发日期已有使用记录，全部年龄及负结果保留，尚无全量或真实跨周实验。'])


def render_period_experiment(result):
    """用同一 JSON 展示所有策略成本及期间失败覆盖，不只列 UCB 收益。"""
    lines=['# 固定期间 OE-UCB 开发验证','',f"计划 `{result['plan_sha256']}`；预声明等权奖励控制。",'',
           '| 年龄 ms | 阶段 | 策略 | 成交 | 成熟订单 | 毛利 USD | 摩擦 USD | 净利 USD | 期间状态 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for phase,data in case['phases'].items():
            for row in data['statistics']:
                lines.append(f"| {case['max_age_ms']} | {phase} | {row['strategy']} | {row['total_fills']} | "
                    f"{row['matured_order_count']} | {row['gross_pnl_usd']} | {row['friction_usd']} | "
                    f"{row['net_pnl_usd']} | {row['period_status_counts']} |")
    lines+=['','## 限制','']+['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'
