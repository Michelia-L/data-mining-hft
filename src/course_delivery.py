"""从固定历史证据生成课程摘要与两张结果图，不重新计算行情收益。

只读 frozen_ablation_snapshot 的预声明五项与三个报价年龄；不作选优。
金额直接从快照读取；组员可以据此制作各自的展示材料。
"""
import hashlib
import json
from pathlib import Path

SOURCE = 'results/final_report/frozen_ablation_snapshot.json'
STRATEGIES = ('cash', 'Fixed-Ridge-price-h1', 'Mean-Ensemble', 'OE-equal-UCB', 'OE-online_library-UCB')
LABELS = ('现金', '固定库候选', '均值集成', '等权OE-UCB', '学习权重OE-UCB')
FIELDS = ('run_status', 'gross_pnl_usd', 'friction_usd', 'net_pnl_usd', 'total_fills', 'matured_order_count')


def build_summary(root):
    """提取七日独立账户的5×3金额/状态和逐日明细，阻断保留null。"""
    root = Path(root)
    raw = (root / SOURCE).read_bytes()
    source = json.loads(raw)
    cases = []
    for case in source['age_cases']:
        lookup = {r['strategy']: r for r in case['statistics']}
        rows = [dict(strategy=name, label=label, **{k: lookup[name].get(k) for k in FIELDS})
                for name, label in zip(STRATEGIES, LABELS)]
        differences = {k: None if rows[-1][k] is None else rows[-1][k]-rows[-2][k]
                       for k in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd')}
        cases.append(dict(max_age_ms=case['max_age_ms'], statistics=rows,
            learned_minus_equal=differences, daily=[dict(session_id=d['session_id'], results=[
                {k: r.get(k) for k in ('strategy',)+FIELDS} for name in STRATEGIES
                for r in d['results'] if r['strategy']==name]) for d in case['daily']]))
    return dict(schema_version=1, result_kind='course_view_of_frozen_evidence',
        source=dict(path=SOURCE, sha256=hashlib.sha256(raw).hexdigest(),
                    source_commit='c406441c504cf8612822ed2b0bbb0242b63f2b46', plan_sha256=source['plan_sha256']),
        evaluation_sessions=[d['session_id'] for d in source['age_cases'][0]['daily']],
        interval_ms=500, currency='USD', account_policy='independent_daily_accounts_no_compounding',
        holdout_claim='none_development_dates', new_replay_performed=False, age_cases=cases)


def show_summary(summary):
    """终端金额直接来自快照；明示这是结果查看而非重训。"""
    print('FMATO课程部分复现｜既有后段七日净利（USD）')
    print('策略\t500ms\t1000ms\t2000ms')
    for i, label in enumerate(LABELS):
        values = [c['statistics'][i]['net_pnl_usd'] for c in summary['age_cases']]
        print(label+'\t'+'\t'.join('阻断' if v is None else f'{v:.2f}' for v in values))
    print('所有执行的非现金策略累计亏损；三个口径均保留。默认未读取行情或重新训练。')


def draw_figures(summary, directory):
    """标准静态图：明确展示阻断与零，以及净利差的毛利/摩擦拆解。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('WenQuanYi Micro Hei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei') if f in fonts), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font, 'axes.unicode_minus':False, 'font.size':13,
                         'savefig.facecolor':'white'})
    values = np.array([[c['statistics'][i]['net_pnl_usd'] if c['statistics'][i]['net_pnl_usd'] is not None else np.nan
                        for c in summary['age_cases']] for i in range(5)])
    fig, ax = plt.subplots(figsize=(11.5,4.2))
    ax.set_axis_off()
    cells = [[('阻断（未执行）' if np.isnan(v) else f'{v:,.2f}') for v in row] for row in values]
    table = ax.table(cellText=cells, rowLabels=LABELS, colLabels=['500ms报价年龄','1000ms报价年龄','2000ms报价年龄'],
                     cellLoc='center', rowLoc='center', bbox=[.23,.15,.76,.80])
    table.auto_set_font_size(False); table.set_fontsize(16)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('white'); cell.set_linewidth(3)
        cell.set_facecolor('#e9f1f3' if r%2 else '#f5f8f9')
        if r == 0:
            cell.set_facecolor('#17324d'); cell.set_text_props(color='white', weight='bold')
        elif c == -1:
            cell.set_text_props(color='#17324d', weight='bold')
        elif r==5 and c==0:
            cell.set_facecolor('#dfe2e5'); cell.set_text_props(color='#525960')
        else:
            cell.set_text_props(color='#a43c32' if values[r-1,c]<0 else '#146d73')
    ax.text(.02,.015,'2025-10-22至10-30 · 7个交易session · 独立日净利加总（USD）', transform=ax.transAxes, color='#566372', fontsize=12)
    fig.savefig(directory / 'net_comparison.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4.5))
    x = np.arange(2); width=.24
    colors = ('#7d91a5','#d98a48','#176e73')
    fields = ('gross_pnl_usd','friction_usd','net_pnl_usd')
    labels = ('毛利差','摩擦差','净利差')
    for i,(field,label,color) in enumerate(zip(fields,labels,colors)):
        vals=[c['learned_minus_equal'][field] for c in summary['age_cases'][1:]]
        bars=ax.bar(x+(i-1)*width, vals, width, label=label, color=color)
        for bar,v in zip(bars,vals):
            ax.annotate(f'{v:,.0f}' if float(v).is_integer() else f'{v:,.2f}',
                (bar.get_x()+bar.get_width()/2,v), xytext=(0,5 if v>=0 else -6),
                textcoords='offset points', ha='center', va='bottom' if v>=0 else 'top', fontsize=12)
    ax.axhline(0,color='#687789',linewidth=1)
    ax.set_xticks(x,['1000ms报价年龄','2000ms报价年龄'])
    ax.set_ylabel('学习组 − 等权组（USD）')
    ax.set_ylim(-11900,7600); ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper right',frameon=False,ncol=3)
    ax.set_title('净利差＝毛利差－摩擦差',loc='left',pad=15,color='#17324d')
    fig.tight_layout(); fig.savefig(directory / 'cost_difference.png',dpi=180); plt.close(fig)


def export_course(root, output_directory):
    """只向新目录导出摘要与两图；原有证据与本地参考材料保持原字节。"""
    root=Path(root); output=Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite export directory: {output}')
    summary=build_summary(root)
    output.mkdir(parents=True)
    draw_figures(summary,output)
    summary['artifact_sha256'] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ('net_comparison.png','cost_difference.png')}
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
