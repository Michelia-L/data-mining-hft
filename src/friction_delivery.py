"""只为单项费用奖励实验生成独立摘要、对照表与图，不读旧快照作新收益。

摘要保留两阶段所有日和三个报价年龄，绑定真实回放身份；阻断保持None。
此导出仅由显式 run-friction 调用，课程默认五策略摘要与两图保持原流程。
"""
import json
from pathlib import Path

from src.config import BASE_DIR
from src.friction_reward import FRICTION_EXPERIMENT
from src.snapshot_dataset import sha256_file


DAY_FIELDS = ('strategy', 'run_status', 'blocking_reasons', 'total_trades', 'total_fills',
    'gross_pnl_usd', 'friction_usd', 'net_pnl_usd', 'terminal_position_liquidated',
    'matured_order_count', 'observed_rewards', 'period_status_counts', 'reward_definition',
    'reward_weights', 'reward_cost_mode', 'reward_horizons_ms', 'reward_source',
    'selection_period_ms', 'exploration_c_price', 'latency_ms', 'holding_review_ms',
    'quantity', 'max_drawdown_usd', 'model_versions_seen', 'friction_reward_audit')
FINANCIAL = ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd')


def paired_difference(rows):
    """逐项保持未定义；净利差=毛利差−摩擦差，正数表示扩展的净利较高。"""
    original, adjusted = rows
    return {k: adjusted[k]-original[k] if adjusted.get(k) is not None and original.get(k) is not None
            else None for k in FINANCIAL+('total_fills',)}


def build_friction_summary(result, source):
    """从本次真实回放提取摘要，保留原权重与门控，不按结果筛选年龄或日期。"""
    if result['result_kind'] != 'course_friction_reward_increment':
        raise ValueError('Need an actual incremental friction replay')
    summary = {k:result[k] for k in ('schema_version','result_kind','plan','plan_sha256',
                                   'selected_age_ms','instrument','environment')}
    summary.update(new_replay_performed=True,
        source=dict(path=str(Path(source).resolve()),sha256=sha256_file(source)), age_cases=[])
    for case in result['age_cases']:
        phases = {}
        for phase, data in case['phases'].items():
            daily = [dict(session_id=day['session_id'],results=[
                {k:r[k] for k in DAY_FIELDS if k in r} for r in day['results']]) for day in data['daily']]
            phases[phase] = dict(daily=daily,statistics=data['statistics'],
                                adjusted_minus_original=paired_difference(data['statistics']))
        summary['age_cases'].append(dict(max_age_ms=case['max_age_ms'], reward_fit=case['reward_fit'],
            execution_gate=case['execution_gate'], calibration_statistics=case['calibration']['statistics'], phases=phases))
    old = BASE_DIR/'results/final_report'
    files = ('development_snapshot.json','four_combinations_snapshot.json','frozen_ablation_snapshot.json',
             'figures/daily_net_usd.png','figures/cumulative_daily_net_usd.png')
    summary['historical_evidence_sha256'] = {name:sha256_file(old/name) for name in files}
    return summary


def format_amount(value):
    """None表示未执行/未定义，不能格式化为金额0。"""
    return '阻断/未定义' if value is None else f'{value:.2f}'


def comparison_table(summary, phase):
    """三个报价年龄与毛利/摩擦/净利差同表展示，避免仅展示有利口径。"""
    lines = ['| 报价年龄 | 原OE净利 | 费用内化OE净利 | 净利差 | 毛利差 | 摩擦差 | 成交数差 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for case in summary['age_cases']:
        data = case['phases'][phase]
        a,b = data['statistics']; d = data['adjusted_minus_original']
        values = [a['net_pnl_usd'], b['net_pnl_usd'], d['net_pnl_usd'],
                  d['gross_pnl_usd'], d['friction_usd']]
        fill = '未定义' if d['total_fills'] is None else str(d['total_fills'])
        lines.append(f"| {case['max_age_ms']}ms | "+' | '.join(map(format_amount,values))+' | '+fill+' |')
    return '\n'.join(lines)


def plot_comparison(summary, path):
    """仅画新实验；未执行单元格灰色并标阻断，避免把null画成现金零收益。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei','Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    all_values = [r['net_pnl_usd'] for c in summary['age_cases'] for p in c['phases'].values()
                  for r in p['statistics'] if r['net_pnl_usd'] is not None]
    limit = max(map(abs,all_values),default=1.) or 1.
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    cmap = matplotlib.colormaps['RdYlGn'].copy(); cmap.set_bad('#e5e7eb')
    norm = colors.TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit)
    for ax,phase,title in zip(axes,('validation','test'),('前段七日开发评价','后段七日开发复核')):
        values = np.array([[c['phases'][phase]['statistics'][i]['net_pnl_usd']
                           for c in summary['age_cases']] for i in range(2)],float)
        ax.imshow(np.ma.masked_invalid(values),cmap=cmap,norm=norm,aspect='auto')
        ax.set_xticks(range(3),[f"{c['max_age_ms']}ms" for c in summary['age_cases']])
        ax.set_yticks(range(2),['原OE-UCB','费用内化OE-UCB'])
        ax.set_title(title); ax.set_xlabel('允许报价年龄；500ms快照网格')
        for i in range(2):
            for j in range(3):
                value = values[i,j]
                row = summary['age_cases'][j]['phases'][phase]['statistics'][i]
                missing = '阻断\n未执行' if row['run_status']=='blocked' else '未定义\n未平仓'
                ax.text(j,i,missing if np.isnan(value) else f'{value:,.2f}',
                        ha='center',va='center',fontsize=11,color='#111827',
                        bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=3))
    fig.suptitle('单项增量实验：原OE奖励 vs 扣本次实际成交摩擦的OE奖励',fontsize=14)
    fig.supxlabel('单位USD；独立日净利加总；全部日期用于开发；灰格为阻断或未定义，不能读作现金0',fontsize=10)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def export_friction_result(result, directory):
    """在本次新输出中发布摘要、中文对照表和图；拒绝替换已经存在的产物。"""
    directory = Path(directory).resolve()
    names = ('summary.json','net_comparison.png','incremental_report.md')
    if any((directory/name).exists() for name in names):
        raise FileExistsError('Refusing to overwrite incremental experiment artifacts')
    summary = build_friction_summary(result,directory/'result.json')
    plot_comparison(summary,directory/'net_comparison.png')
    summary['artifact_sha256'] = {'net_comparison.png':sha256_file(directory/'net_comparison.png')}
    (directory/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report = ('# 费用内化奖励增量实验\n\n'
              '两方法共用原校准权重、模型、UCB与执行条件，唯一处理是奖励减本次实际单边摩擦。'
              '全部日期已用于开发；原历史证据另行保留。金额单位USD，差值为扩展减原OE。\n\n')
    for phase,title in (('validation','前段七日开发评价'),('test','后段七日开发复核')):
        report += f'## {title}\n\n'+comparison_table(summary,phase)+'\n\n'
    report += '![两阶段三个年龄的净利对照](net_comparison.png)\n\n净利差=毛利差−摩擦差；阻断/未定义不是现金零收益。\n'
    (directory/'incremental_report.md').write_text(report,encoding='utf-8')
    print(report)
    return summary
