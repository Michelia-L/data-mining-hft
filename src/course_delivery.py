"""从固定历史证据生成课程摘要、两张图与可编辑PPT，不重新计算行情收益。

只读 frozen_ablation_snapshot 的预声明五项与三个报价年龄；不作选优。
幻灯片文字和备注从 presentation.md 读取，金额从快照读取，减少手工抄错。
"""
import hashlib
import json
from pathlib import Path
import re

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


def read_slides(path):
    """解析课程稿中11个编号区块，保留每页三条要点、备注和讲述秒数。"""
    text = Path(path).read_text(encoding='utf-8')
    blocks = re.findall(r'<!-- slide:(\d+) seconds:(\d+) -->\s*## ([^\n]+)\n(.*?)(?=<!-- slide:|\n## 教师问答)', text, re.S)
    slides = []
    for number, seconds, title, body in blocks:
        points, notes = body.split('讲稿：', 1)
        slides.append(dict(number=int(number), seconds=int(seconds), title=title,
                           bullets=re.findall(r'^- (.+)$', points, re.M), notes=notes.strip()))
    if ([s['number'] for s in slides] != list(range(1,12))
            or sum(s['seconds'] for s in slides) != 560 or any(len(s['bullets']) != 3 for s in slides)):
        raise ValueError('Need 8 main + 3 backup slides, three points each and 560 seconds')
    return slides


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


def make_deck(slides, directory, target):
    """16:9课程稿；文本/流程图可编辑，图表用标准绘图，逐页讲稿写入备注。"""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    navy='17324D'; teal='176E73'; grey='647586'; pale='EDF3F5'; orange='D98A48'
    def box(slide,text,x,y,w,h,size=23,color=navy,bold=False,fill=None):
        shape=(slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
               if fill else slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)))
        if fill:
            shape.fill.solid(); shape.fill.fore_color.rgb=RGBColor.from_string(fill); shape.line.fill.background()
        tf=shape.text_frame; tf.word_wrap=True
        tf.margin_left=tf.margin_right=Inches(0 if w<=.5 else .12)
        tf.margin_top=Inches(.1)
        for i,line in enumerate(text.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line
            p.font.name='Microsoft YaHei'; p.font.size=Pt(size); p.font.bold=bold
            p.font.color.rgb=RGBColor.from_string(color); p.space_after=Pt(4)
        return shape
    elapsed=0
    for spec in slides:
        n=spec['number']; slide=prs.slides.add_slide(prs.slide_layouts[6])
        box(slide,'FMATO  /  课程部分复现',.65,.20,10,.35,size=12,color=teal,bold=True)
        box(slide,spec['title'],.65,.65,12,.8,size=32,bold=True)
        end=elapsed+spec['seconds']; clock=f'{elapsed//60}:{elapsed%60:02d}—{end//60}:{end%60:02d}' if n<=8 else '提问时使用'
        box(slide,f'{n:02d}  /  {clock}',.65,7.02,12,.3,size=11,color=grey)
        slide.notes_slide.notes_text_frame.text=spec['notes']
        if n<=8: elapsed=end
        if n==6:
            box(slide,spec['bullets'][0],.7,1.5,12,.55,size=22,color=teal,bold=True)
            slide.shapes.add_picture(str(directory / 'net_comparison.png'), Inches(1.42), Inches(1.95), width=Inches(10.5))
            box(slide,'\n'.join(spec['bullets'][1:]),.8,6.04,12,.9,size=18)
        elif n==7:
            box(slide,spec['bullets'][0],.7,1.5,12,.55,size=24,color=teal,bold=True)
            slide.shapes.add_picture(str(directory / 'cost_difference.png'), Inches(.7), Inches(2.35), width=Inches(8.0))
            box(slide,'1000ms口径\n毛利更差4125\n摩擦减少9565\n→ 少亏5440美元',9,2.45,3.5,2.3,size=22,fill=pale)
            box(slide,'反馈稀疏、短历史\n替代市场与执行假设\n少亏不等于预测更准确',9,5.0,3.5,1.4,size=18,color=grey)
        else:
            for i,point in enumerate(spec['bullets']):
                box(slide,point,.8,1.7+i*.78,11.8,.7,size=23 if n<9 else 20)
            if n==2:
                for i,label in enumerate(('线性 / 浅树','三类特征','1 / 2日历史')):
                    box(slide,label,1+i*4,4.55,3.65,.8,size=24,fill=pale,bold=True)
                box(slide,'2 × 3 × 2 ＝ 12个候选；按周更新',1,5.7,11.2,.7,size=27,color=teal,bold=True)
            elif n==3:
                for i,label in enumerate(('真实时间盘口','轻模型库','UCB选模型','延迟成交','美元账本')):
                    box(slide,label,.8+i*2.5,4.45,2.2,.85,size=19,fill=pale,bold=True)
                    if i<4: box(slide,'→',3.01+i*2.5,4.60,.3,.5,size=20,color=teal)
                box(slide,'历史校准 → 冻结权重；成交 → 完整成熟评分 → 更新UCB',.9,5.9,11.7,.9,size=22,color=teal)
            elif n==4:
                box(slide,'成交 t → 5秒 → 15秒 → … → 3645秒',1,4.6,11.2,.8,size=30,color=teal,bold=True,fill=pale)
                box(slide,'完整奖励成熟  ＋  期间结束  ＋  全部订单可评价',1,5.75,11.2,.7,size=24,bold=True)
            elif n==5:
                box(slide,'固定网格500ms\n报价年龄500 / 1000 / 2000ms',.9,4.6,5.6,1.3,size=22,fill=pale)
                box(slide,'1张ES；点值50美元\n点差 + 滑点 + 单边手续费',6.8,4.6,5.6,1.3,size=22,fill=pale)
            elif n==8:
                box(slide,'核心机制可以解释，收益优势尚未得到支持。',.9,5.05,11.5,1.0,size=28,color=teal,bold=True,fill=pale)
            elif n==1:
                box(slide,'一个研究问题，一条方法链路，一份如实的结果。',.9,5.2,11.4,.8,size=27,color=teal,bold=True,fill=pale)
    prs.save(target)


def export_course(root, output_directory):
    """只向新目录导出；源码证据与已提交PPT不被原地覆盖。"""
    root=Path(root); output=Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite export directory: {output}')
    summary=build_summary(root); slides=read_slides(root / 'presentation.md')
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    draw_figures(summary,output)
    make_deck(slides,output,output / 'presentation.pptx')
    summary['artifact_sha256'] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ('net_comparison.png','cost_difference.png','presentation.pptx')}
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
