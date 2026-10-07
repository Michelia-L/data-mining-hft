"""增量奖励的性质检查；由 run_project.py check --core 调用，不读取实盘行情。

缩短七尺度只为核对因果时序。检查真实单边成本归属、美元/价格点单位、
原版回归、零成本等价、成本改变选择、坏标签不反馈与未来扰动不改过去。
"""
import hashlib
import json
import math
from types import SimpleNamespace


HORIZONS = [500, 1000, 1500, 2000, 2500, 3000, 3500]
WEIGHTS = [1.2, -.2, 0., 0., 0., 0., 0.]
# 在未修改的 origin/main（37c2c5d）及本次数值环境上生成，含22笔成交、19次期间反馈。
ORIGINAL_DIGEST = '02b013e4b60ff10a7ba9a58f3f50b7c284a019f09b1056c1494fbaf32db1b04e'


def synthetic_quotes(future_shift=False):
    """半秒网格的七尺度样本，点差随当时行情变化；修改只发生在30秒之后。"""
    import pandas as pd
    from src.snapshot_dataset import SessionCalendar, build_session_dataset
    calendar = SessionCalendar()
    start = pd.Timestamp('2025-09-22T00:00:00Z')
    rows = []
    for i in range(140):
        stamp = start + pd.Timedelta(milliseconds=500*i)
        received = stamp - pd.Timedelta(milliseconds=100)
        bid = 100 + .25*i + (100 if future_shift and i > 60 else 0)
        spread = .25 * (1 + (i//12) % 3)
        row = dict(ts_event=stamp, source_ts_recv=received, source_ts_event=received,
            source_flags=128, age_ms=100., sequence=i, instrument_id=294973, symbol='ESZ5')
        for level in range(5):
            row.update({f'bid_px_{level:02d}':bid-.25*level,
                f'ask_px_{level:02d}':bid+spread+.25*level,
                f'bid_sz_{level:02d}':10, f'ask_sz_{level:02d}':12})
        rows.append(row)
    assigned, _, _ = calendar.assign(pd.DataFrame(rows))
    return build_session_dataset(assigned, calendar, 500, HORIZONS)[0], calendar, start


def original_replay():
    """供原版源码生成回归指纹；不依赖新增奖励，覆盖期间选择和完整 JSON。"""
    import numpy as np
    from src.period_ucb import PeriodOESelector
    from src.sum_only_reward import FrozenSumOnlyReward
    from src.time_execution import TimeExecutionEngine
    quotes, calendar, _ = synthetic_quotes()
    models = [SimpleNamespace(name='long'), SimpleNamespace(name='short')]
    selector = PeriodOESelector('original-regression', models, period_ms=5000, c=1.)
    return TimeExecutionEngine(FrozenSumOnlyReward(HORIZONS, WEIGHTS), calendar=calendar,
        holding_review_ms=1000).run_backtest(selector, quotes,
        np.tile([.01, -.01], (len(quotes), 1)), detail=True)


def result_digest(result):
    """完整 JSON 指纹用于核对增量入口没有改变原版路径的任何数值或字段。"""
    return hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def check_friction_snapshot(root, require):
    """标准库核对增量证据与表格；不要求原始行情，不重算或替换历史收益。"""
    from pathlib import Path
    import gzip
    directory = Path(root)/'results/friction_reward'
    summary = json.loads((directory/'summary.json').read_text())
    archive = directory/summary['source']['archive_path']
    compressed = archive.read_bytes(); raw_bytes = gzip.decompress(compressed)
    require(hashlib.sha256(compressed).hexdigest()==summary['source']['archive_sha256']
            and hashlib.sha256(raw_bytes).hexdigest()==summary['source']['sha256'], '新增完整回放快照身份改变')
    replay = json.loads(raw_bytes)
    require(summary['new_replay_performed'] is True
            and summary['result_kind']=='course_friction_reward_increment', '增量摘要必须来自实际新回放')
    plan = summary['plan']; config = plan['config']; protocol = config['protocol']
    library_bytes = (directory/summary['source']['library_archive_path']).read_bytes()
    require(hashlib.sha256(library_bytes).hexdigest()==summary['source']['library_archive_sha256']
            and hashlib.sha256(gzip.decompress(library_bytes)).hexdigest()==plan['library_sha256'], '共用模型库身份改变')
    experiment = plan['incremental_experiment']
    methods = [experiment['control'],experiment['treatment']]
    require(methods==['OE-online_library-UCB','OE-friction-online_library-UCB']
            and experiment['cost_coefficient']==1.
            and experiment['weight_source']=='shared_original_raw_OE_calibration_no_refit'
            and summary['selected_age_ms'] is None, '不能增加处理差异、另学权重或按收益挑年龄')
    require([c['max_age_ms'] for c in summary['age_cases']]==[500,1000,2000], '增量实验缺报价年龄')
    payload = {k:v for k,v in plan.items() if k!='plan_sha256'}
    actual = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,
        allow_nan=False,separators=(',', ':')).encode()).hexdigest()
    require(actual==summary['plan_sha256']==plan['plan_sha256'], '增量冻结计划身份改变')
    for name,sha in summary['historical_evidence_sha256'].items():
        require(hashlib.sha256((Path(root)/'results/final_report'/name).read_bytes()).hexdigest()==sha,
                '增量实验改变历史证据：'+name)
    require(set(summary['artifact_sha256'])=={'net_comparison.png'}, '增量图表清单改变')
    for name,sha in summary['artifact_sha256'].items():
        require(hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha, '增量图表身份改变')
    report = (Path(root)/'friction_reward_report.md').read_text()
    financial = ('gross_pnl_usd','friction_usd','net_pnl_usd')
    for case in summary['age_cases']:
        weights = case['reward_fit']['weights']; gate = case['execution_gate']['usable_for_execution']
        for phase in ('validation','test'):
            data = case['phases'][phase]
            raw_case = next(c for c in replay['age_cases'] if c['max_age_ms']==case['max_age_ms'])
            original_data = raw_case['phases'][phase]
            require(data['statistics']==original_data['statistics']
                    and case['reward_fit']==raw_case['reward_fit']
                    and case['execution_gate']==raw_case['execution_gate'], '摘要不等于新增原始回放')
            require([d['session_id'] for d in data['daily']]==protocol['sessions'][phase], '增量日期被挑选或遗漏')
            totals = data['statistics']; delta = data['adjusted_minus_original']
            require([r['strategy'] for r in totals]==methods, '两方法汇总错位')
            for day,raw_day in zip(data['daily'],original_data['daily']):
                require(day['session_id']==raw_day['session_id'], '原始回放与摘要日期错位')
                require([r['strategy'] for r in day['results']]==methods, '两方法日结果错位')
                for i,row in enumerate(day['results']):
                    raw_row = raw_day['results'][i]
                    require(all(raw_row[k]==v for k,v in row.items()), '摘要日记录不是原始回放值')
                    require(row['run_status']==('executed' if gate else 'blocked'), '两方法没有共用门控')
                    if not gate:
                        require(all(row[k] is None for k in financial+('total_fills',)), '阻断被补成零收益')
                        continue
                    require(row['reward_weights']==weights
                            and row['reward_horizons_ms']==protocol['reward_horizons_ms']
                            and row['selection_period_ms']==config['selection_period_ms']
                            and row['exploration_c_price']==config['exploration_c_price']
                            and row['latency_ms']==protocol['latency_ms']
                            and row['holding_review_ms']==protocol['holding_review_ms'], '两方法的受控条件改变')
                    require(math.isclose(row['gross_pnl_usd']-row['friction_usd'],row['net_pnl_usd'],abs_tol=1e-7),
                            '增量日账本不守恒')
                    periods = raw_row['period_feedback']
                    require(sum(p['orders'] for p in periods)==row['total_fills']
                            and sum(p['updated_selector'] for p in periods)==row['observed_rewards'], '实际期间计数不守恒')
                    for p in periods:
                        require(p['orders']==p['pending']+sum(p['statuses'].values()), '实际期间订单被丢弃')
                        if p['updated_selector']:
                            require(p['orders']==p['statuses'].get('matured',0) and p['pending']==0
                                    and p['observed_at_ns']>=max(p['end_ns'],p['last_origin_ns']+
                                                              max(protocol['reward_horizons_ms'])*1_000_000),
                                    '真实回放期间在七尺度成熟前或按子集反馈')
                        else:
                            require(p['reward'] is None, '真实回放坏期间被补零')
                    if i==0:
                        require(row['reward_cost_mode']=='PaperOE-no-cost', '原对照奖励被扣费')
                    else:
                        require(row['reward_cost_mode']=='FillOE-actual-single-side-friction', '扩展奖励未扣实际单边费用')
                        audit = row['friction_reward_audit']; count = row['matured_order_count']
                        require(audit['matured_fills']==count and audit['coefficient']==1.
                                and audit['account_costs_deducted_again'] is False, '奖励重复扣账本或混入未成熟单')
                        require(math.isclose(audit['cost_sum_usd']/(summary['instrument']['multiplier']*row['quantity']),
                                             audit['cost_sum_price'],abs_tol=1e-7)
                                and math.isclose(audit['raw_reward_sum_price']-audit['cost_sum_price'],
                                                 audit['adjusted_reward_sum_price'],abs_tol=1e-7), '费用奖励单位或加总错误')
                        require(math.isclose(sum(p['reward_sum'] for p in periods),audit['adjusted_reward_sum_price'],abs_tol=1e-7),
                                '实际期间奖励与成熟成交扣费评分不一致')
                        require(0. <= audit['cost_sum_usd'] <= row['friction_usd']+1e-7, '成熟费用超过全部实际费用')
                        require(count>0 or all(audit[k] is None for k in
                            ('raw_reward_mean_price','cost_mean_price','adjusted_reward_mean_price')), '无成熟奖励均值应未定义')
            for i,total in enumerate(totals):
                rows = [d['results'][i] for d in data['daily']]
                if not gate:
                    require(all(total[k] is None for k in financial), '阻断总收益不应定义')
                    continue
                for key in ('total_fills','matured_order_count'):
                    require(sum(r[key] for r in rows)==total[key], '增量计数加总错位')
                for key in financial:
                    expected = sum(r[key] for r in rows) if key=='friction_usd' or total['pnl_aggregation_defined'] else None
                    require(total[key] is None if expected is None else math.isclose(total[key],expected,abs_tol=1e-7),
                            '增量金额加总错位')
            for key in financial+('total_fills',):
                expected = totals[1][key]-totals[0][key] if all(t[key] is not None for t in totals) else None
                require(delta[key]==expected, '增量差值错位')
            if all(delta[k] is not None for k in financial):
                require(math.isclose(delta['gross_pnl_usd']-delta['friction_usd'],delta['net_pnl_usd'],abs_tol=1e-7),
                        '净利差不是毛利差减摩擦差')
            values = [totals[0]['net_pnl_usd'],totals[1]['net_pnl_usd'],delta['net_pnl_usd'],
                      delta['gross_pnl_usd'],delta['friction_usd']]
            texts = ['阻断/未定义' if v is None else f'{v:.2f}' for v in values]
            fills = '未定义' if delta['total_fills'] is None else str(delta['total_fills'])
            line = f"| {case['max_age_ms']}ms | "+' | '.join(texts)+' | '+fills+' |'
            require(line in report, '增量报告对照表与新证据不符')
            if phase=='test':
                main_report = (Path(root)/'final_replication_report.md').read_text()
                money = ['阻断' if v is None else f'{v:.2f}' for v in values[:2]]
                change = '未定义' if delta['net_pnl_usd'] is None else f"{delta['net_pnl_usd']:.2f}"
                main_line = f"| {case['max_age_ms']}ms | "+' | '.join(money+[change])+' |'
                require(main_line in main_report, '课程报告增量表与新证据不符')
    # 后段的原OE对照需与历史快照逐日复核，不能用新方法替换老实验。
    historical = json.loads((Path(root)/'results/final_report/frozen_ablation_snapshot.json').read_text())
    for case,old in zip(summary['age_cases'],historical['age_cases']):
        for day,old_day in zip(case['phases']['test']['daily'],old['daily']):
            control = day['results'][0]
            previous = next(r for r in old_day['results'] if r['strategy']==methods[0])
            require(day['session_id']==old_day['session_id'] and all(control.get(k)==previous.get(k)
                    for k in financial+('total_fills','matured_order_count')), '原OE后段财务或成交计数偏离历史')
    print('通过：新增完整快照、两阶段/三年龄、共用权重与门控、实际七尺度成熟、费用/期间审计、日加总及原OE历史回归。')


def check_friction_reward(require):
    """独立检查经济单位、真实成交成本、完整成熟、选择效果和默认行为。"""
    import numpy as np
    import pandas as pd
    from src.friction_reward import FrozenFrictionOEReward
    from src.model_selector import SingleModelSelector
    from src.period_ucb import PeriodOESelector
    from src.sum_only_reward import FrozenSumOnlyReward
    from src.time_execution import TimeExecutionEngine

    require(result_digest(original_replay()) == ORIGINAL_DIGEST, '新增扩展改变了原版完整回放结果')
    raw = FrozenSumOnlyReward(HORIZONS, WEIGHTS)
    adjusted = FrozenFrictionOEReward(HORIZONS, WEIGHTS)
    values = np.arange(1., 8.)
    require(adjusted.cost_in_price_points(35., 50., 2.) == .35, '美元成本未正确除以点值和张数')
    require(math.isclose(adjusted.score(values, cost_price=.35), raw.score(values)-.35),
            '负权时费用应在最终奖励恰好扣一次')
    require(adjusted.score(values, cost_price=0.) == raw.score(values), '零成本不等价于原奖励')
    for bad in (-1., float('nan'), float('inf')):
        try:
            adjusted.score(values, cost_price=bad)
        except ValueError:
            pass
        else:
            raise ValueError('无效成本被接受')

    quotes, calendar, start = synthetic_quotes()
    models = [SimpleNamespace(name='long'), SimpleNamespace(name='short')]
    def replay(frame, reward, periodic=False, no_cost=False):
        selector = (PeriodOESelector('synthetic', models, period_ms=5000, c=1.) if periodic
                    else SingleModelSelector('synthetic', models, fixed_idx=0))
        selector.reward_type = 'OE'
        engine = TimeExecutionEngine(reward, calendar=calendar, holding_review_ms=1000, quantity=2.)
        if no_cost:
            engine.config = engine.config | dict(slippage_ticks=0., commission_per_order=0.)
            frame = frame.copy()
            frame['bid_px_00'] = frame['ask_px_00'] = frame['mid_price']
        return engine.run_backtest(selector, frame, np.tile([.01, -.01], (len(frame), 1)), detail=True)

    base = replay(quotes, raw)
    treatment = replay(quotes, adjusted)
    for key in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd', 'total_fills', 'trades', 'decisions'):
        require(base[key] == treatment[key], '固定决策时奖励扣费不应再次改变账本：'+key)
    require(treatment['total_fills'] > 0 and treatment['matured_order_count'] > 0,
            '费用检查必须包含实际成交和完整成熟反馈')
    fills = {f['ts_event']:f for f in treatment['fills']}
    require(math.isclose(sum(f['cost_usd'] for f in fills.values()), treatment['friction_usd']),
            '实际费用没有按全部开平仓成交各记一次')
    for observation in treatment['reward_observations']:
        fill = fills[observation['origin_time']]
        require(observation['cost_usd'] == fill['cost_usd']
                and observation['cost_price'] == fill['cost_usd']/100.
                and math.isclose(observation['reward'], observation['raw_reward']-observation['cost_price']),
                '奖励使用的不是本次实际单边摩擦')
        require(pd.Timestamp(observation['observed_time']) >= pd.Timestamp(observation['due_time']),
                '已知成本不允许绕过七尺度完整成熟')
    audit = treatment['friction_reward_audit']
    require(math.isclose(audit['raw_reward_sum_price']-audit['cost_sum_price'],
                        audit['adjusted_reward_sum_price'], abs_tol=1e-10), '奖励成本审计不守恒')
    require(audit['matured_fills'] == treatment['matured_order_count'], '奖励审计混入未成熟成交')

    # 无摩擦市场下，原版和扩展应有完全一致的期间选择、成交和反馈。
    zero_raw = replay(quotes, raw, periodic=True, no_cost=True)
    zero_adjusted = replay(quotes, adjusted, periodic=True, no_cost=True)
    for key in ('decisions', 'period_selections', 'period_feedback', 'net_pnl_usd'):
        require(zero_raw[key] == zero_adjusted[key], '零成本回放改变了原版协议：'+key)

    first = replay(quotes, adjusted, periodic=True)
    shifted = replay(synthetic_quotes(True)[0], adjusted, periodic=True)
    cutoff = start+pd.Timedelta(seconds=30)
    for key, clock in (('decisions','ts_event'), ('fills','ts_event'),
                       ('reward_observations','observed_time')):
        require([r for r in first[key] if pd.Timestamp(r[clock]) <= cutoff] ==
                [r for r in shifted[key] if pd.Timestamp(r[clock]) <= cutoff], '未来价格改变过去：'+key)
    require(first['observed_rewards'] > 0, '扩展必须真正更新期间选择器')
    for period in first['period_feedback']:
        if period['updated_selector']:
            require(period['orders'] == period['statuses'].get('matured',0)
                    and period['observed_at_ns'] >= max(period['end_ns'],period['last_origin_ns']+3_500_000_000),
                    '费用奖励按子集或提前更新期间')
        else:
            require(period['reward'] is None, '无订单或未成熟期间不能补成现金零奖励')
    # 删去一格未来目标，确实产生坏标签；该期间不得由其余订单替代。
    gap = replay(quotes.drop(index=80).reset_index(drop=True), adjusted, periodic=True)
    incomplete = [p for p in gap['period_feedback'] if p['status']=='incomplete_orders']
    require(incomplete and all(not p['updated_selector'] for p in incomplete), '扩展没有阻断坏标签期间')

    # 同一原始评价偏好，在成本不同时应能改变 UCB 的经济排序，探索项固定为0。
    choices = []
    for costs in ((0.,0.), (.6,.1)):
        selector = PeriodOESelector('cost-choice', models, period_ms=5000, c=0.)
        selector.select_period('s','v',0,0)
        selector.select_period('s','v',1,1)
        for owner, (gross, cost) in enumerate(zip((1.,.8), costs)):
            selector.observe_period('v',owner,adjusted.score(np.full(7,gross),cost_price=cost))
        choices.append(selector.select_period('s','v',2,2))
    require(choices == [0,1], '费用未实际进入模型选择')
    print('通过：费用单位/单次扣除、账本隔离、零费用等价、七尺度成熟/缺格、未来扰动与费用改变选择。')
