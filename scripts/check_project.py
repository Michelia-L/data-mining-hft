"""课程交付小核对：标准库查结果；--core另查合成时序/账本和奖励拟合。

检查不证明盈利、原行情真实性、全部逐笔因果性或论文等价复现。
历史源码身份保留在快照；当前源码不必等于旧源码。合成样本缩短时间参数，
只检查性质，不把它叫作论文七尺度实验或替换正式数据结果。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
PINS = {
    'evidence_index.json':'35c4656c2fa2066782dec4752eb6cc901d29096171e7c967a45ffde34f025555',
    'development_snapshot.json':'8dd3caffea6da0b400dd2a42ee89151d9649af8345bf3ef8ec9b8ec2a7ea190f',
    'four_combinations_snapshot.json':'0bc4a073b191573ee55b9f359ed8480429ca38c0ca54cd8a742beb0b1e5d869a',
    'frozen_ablation_snapshot.json':'3765d39e51748a72147e86653d0ca0edfcaf50d3e452a262962bc56fdc1e85b6',
    'figures/daily_net_usd.png':'176f9d925c9e9603f43698a370de7cfbd832b3291b42f52c8f88da995083407e',
    'figures/cumulative_daily_net_usd.png':'a6eb4778466fdfec88e15f66bbe432de5a712d3fa4763557b1afde0c9278f2d7',
}
STRATEGIES = ('cash','Fixed-Ridge-price-h1','Mean-Ensemble','OE-equal-UCB','OE-online_library-UCB')
LABELS = ('现金','固定库候选','均值集成','等权OE-UCB','学习权重OE-UCB')
FINANCIAL = ('gross_pnl_usd','friction_usd','net_pnl_usd')


def require(condition, message):
    """即使Python以优化模式运行，也不能关闭交付约束。"""
    if not condition:
        raise ValueError(message)


def digest(path):
    """这里只哈希小型交付文件，不在课堂读取原行情。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_delivery():
    """独立核对15格金额、全部27项日加总、当前阅读链接与11页PPT备注。"""
    evidence = ROOT / 'results/final_report'
    for path, sha in PINS.items():
        require(digest(evidence/path)==sha, f'历史证据身份改变：{path}')
    source=json.loads((evidence/'frozen_ablation_snapshot.json').read_text())
    summary=json.loads((ROOT/'results/presentation/summary.json').read_text())
    require(summary['source']['sha256']==PINS['frozen_ablation_snapshot.json']
            and summary['new_replay_performed'] is False, '不能把课程摘要写成新回放')
    require([c['max_age_ms'] for c in summary['age_cases']]==[500,1000,2000], '三个报价年龄必须保留')
    for original, view in zip(source['age_cases'],summary['age_cases']):
        lookup={r['strategy']:r for r in original['statistics']}
        require([r['strategy'] for r in view['statistics']]==list(STRATEGIES), '五项主比较必须按声明顺序保留')
        require([r['label'] for r in view['statistics']]==list(LABELS), '中文标签与策略错位')
        require(len(original['daily'])==7 and [d['session_id'] for d in original['daily']]==summary['evaluation_sessions'], '七日范围改变')
        for total in original['statistics']:
            rows=[next(r for r in d['results'] if r['strategy']==total['strategy']) for d in original['daily']]
            if total['run_status']=='blocked':
                require(all(r['run_status']=='blocked' and all(r[k] is None for k in FINANCIAL) for r in rows), '阻断被补成收益')
                require(all(total[k] is None for k in FINANCIAL), '阻断汇总金额应未定义')
            else:
                for r in rows+[total]:
                    require(math.isclose(r['gross_pnl_usd']-r['friction_usd'],r['net_pnl_usd'],abs_tol=1e-7), '毛利−成本不等于净利')
                for k in FINANCIAL+('total_fills',):
                    require(math.isclose(sum(r[k] for r in rows),total[k],abs_tol=1e-7), '独立日加总错位')
        for row in view['statistics']:
            for key in ('run_status',)+FINANCIAL+('total_fills','matured_order_count'):
                require(row[key]==lookup[row['strategy']].get(key), f'摘要数据改变：{row["strategy"]}/{key}')
        for old_day,view_day in zip(original['daily'],view['daily']):
            require(old_day['session_id']==view_day['session_id'], '摘要日次错位')
            for row in view_day['results']:
                old=next(r for r in old_day['results'] if r['strategy']==row['strategy'])
                require(all(value==old.get(key) for key,value in row.items()), '逐日摘要错位')
        equal=lookup['OE-equal-UCB']; learned=lookup['OE-online_library-UCB']
        for k in FINANCIAL:
            expected=None if learned[k] is None else learned[k]-equal[k]
            require(view['learned_minus_equal'][k]==expected,'成本差异拆解错位')
    require(summary['age_cases'][0]['statistics'][0]['net_pnl_usd']==0
            and summary['age_cases'][0]['statistics'][-1]['net_pnl_usd'] is None, '现金与阻断不可混淆')
    report=(ROOT/'final_replication_report.md').read_text()
    for i,label in enumerate(LABELS):
        values=[c['statistics'][i]['net_pnl_usd'] for c in summary['age_cases']]
        expected='| '+label+' | '+' | '.join('阻断' if v is None else f'{v:.2f}' for v in values)+' |'
        require(expected in report,'课程报告主表与证据不符：'+label)
    for doc in ('README.md','presentation.md','final_replication_report.md','results/final_report/README.md'):
        text=(ROOT/doc).read_text()
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            if not link.startswith(('https:','http:','#')):
                require((ROOT/doc).parent.joinpath(link.split('#')[0]).exists(),f'失效链接：{doc}/{link}')
    # 幻灯片从同一稿生成，核对备注、主/备页结构和嵌入的图像身份。
    from src.course_delivery import read_slides
    slides=read_slides(ROOT/'presentation.md')
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with ZipFile(ROOT/'presentation.pptx') as deck:
        require(len([n for n in deck.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])==11,'PPT页数应为11')
        for spec in slides:
            notes=ET.fromstring(deck.read(f'ppt/notesSlides/notesSlide{spec["number"]}.xml'))
            texts=''.join(n.text or '' for n in notes.findall('.//a:t',ns))
            require(spec['notes'] in texts,'PPT备注与讲稿不一致')
        media={hashlib.sha256(deck.read(n)).hexdigest() for n in deck.namelist() if n.startswith('ppt/media/')}
        for name in ('net_comparison.png','cost_difference.png'):
            require(digest(ROOT/'results/presentation'/name) in media,'PPT图像与课程图不一致')
    for name,sha in summary['artifact_sha256'].items():
        path=ROOT/name if name=='presentation.pptx' else ROOT/'results/presentation'/name
        require(digest(path)==sha,'课程导出产物身份改变：'+name)
    print('通过：三份原快照/两图、15格主表、全部27项日加总、文档链接、11页PPT及备注。')


def check_core():
    """少量性质检查：未来扰动、期间完整成熟、金额守恒和历史拟合一致。

    使用1/3秒合成前瞻和5秒期间，避免为课程检查读取大型行情。数值环境
    在此函数中导入，因此默认交付核对仍能用 python -S 运行。
    """
    from types import SimpleNamespace
    import numpy as np
    import pandas as pd
    from src.calibration import execution_gate, qualify_calibration
    from src.course_experiment import load_config
    from src.period_ucb import PeriodOESelector, PeriodOrderBook
    from src.snapshot_dataset import SessionCalendar, build_session_dataset
    from src.sum_only_reward import FrozenSumOnlyReward, learn_sum_only
    from src.time_execution import TimeExecutionEngine
    config,calendar=load_config(ROOT/'config/course_experiment.json')
    start=pd.Timestamp('2025-09-22T00:00:00Z')
    models=[SimpleNamespace(name='long'),SimpleNamespace(name='short')]
    def frame(future_shift=False):
        rows=[]
        for i in range(120):
            stamp=start+pd.Timedelta(milliseconds=500*i); received=stamp-pd.Timedelta(milliseconds=100)
            price=100+.25*i+(100 if future_shift and i>60 else 0)
            row=dict(ts_event=stamp,source_ts_recv=received,source_ts_event=received,
                source_flags=128,age_ms=100.,sequence=i,instrument_id=294973,symbol='ESZ5')
            for level in range(5):
                row.update({f'bid_px_{level:02d}':price-.25*level,f'ask_px_{level:02d}':price+.25*(level+1),
                            f'bid_sz_{level:02d}':10,f'ask_sz_{level:02d}':12})
            rows.append(row)
        assigned,_,_=calendar.assign(pd.DataFrame(rows))
        return build_session_dataset(assigned,calendar,500,(1000,3000))[0]
    def replay(quotes):
        selector=PeriodOESelector('synthetic',models,period_ms=5000,c=1.)
        return TimeExecutionEngine(FrozenSumOnlyReward([1000,3000],[.5,.5]),calendar=calendar,
            holding_review_ms=1000).run_backtest(selector,quotes,np.tile([.01,-.01],(len(quotes),1)),detail=True)
    first=replay(frame()); second=replay(frame(True)); cutoff=start+pd.Timedelta(seconds=30)
    for key,clock in (('decisions','ts_event'),('fills','ts_event'),('reward_observations','observed_time')):
        require([r for r in first[key] if pd.Timestamp(r[clock])<=cutoff]==
                [r for r in second[key] if pd.Timestamp(r[clock])<=cutoff], '未来价格改变过去'+key)
    require(first['total_fills']>0 and first['observed_rewards']>0,'合成样本应真正成交并更新期间反馈')
    periods=first['period_feedback']
    require(sum(p['orders'] for p in periods)==first['total_fills'],'期间订单不守恒')
    for p in periods:
        require(p['orders']==p['pending']+sum(p['statuses'].values()),'未成熟订单被丢弃')
        if p['updated_selector']:
            require(p['statuses'].get('matured',0)==p['orders'] and p['observed_at_ns']>=
                    max(p['end_ns'],p['last_origin_ns']+3_000_000_000),'期间提前或按子集反馈')
    require(math.isclose(first['gross_pnl_usd']-first['friction_usd'],first['net_pnl_usd'],abs_tol=1e-8)
            and math.isclose(sum(t['net_pnl_usd'] for t in first['trades']),first['net_pnl_usd'],abs_tol=1e-8), '交易账本不守恒')
    selector=PeriodOESelector('gate',models,period_ms=5000,c=1.)
    book=PeriodOrderBook(selector,calendar,3_000_000_000); t=start.value
    book.choose('2025-09-22','v',t)
    a=book.add_order('2025-09-22','v',0,t+1_000_000_000)
    b=book.add_order('2025-09-22','v',0,t+4_000_000_000)
    book.resolve(a,'matured',2.); book.advance(t+5_000_000_000)
    require(not selector.reward_history,'最后订单未成熟时不得反馈')
    book.resolve(b,'missing_target'); book.advance(t+7_000_000_000)
    require(not selector.reward_history and book.completed[0]['reward'] is None,'坏订单不能补零或取成熟子集')
    # 同一历史校准矩阵复算，验证去掉盒约束诊断没有改变专家/资格/学习结果。
    archive=json.loads((ROOT/'results/final_report/development_snapshot.json').read_text())
    for case in archive['age_cases']:
        stats=case['calibration']['statistics']; group=case['calibration']['scopes']['online_library']
        old=group['reward_fits']['sum_only']; experts=group['expert_candidate_policy_ids']
        q=qualify_calibration(stats,config['protocol']['reward_horizons_ms'],config['calibration'],expert_policy_ids=experts)
        fit=learn_sum_only(stats,config['protocol']['reward_horizons_ms'],config['calibration'],q)
        require(fit['status']==old['status'] and fit['expert']==old['expert']
                and fit['eligible_policy_ids']==old['eligible_policy_ids'],'历史校准资格发生变化')
        if old['weights'] is not None:
            require(np.allclose(fit['weights'],old['weights'],rtol=1e-8,atol=1e-8),'历史奖励权重发生变化')
        policies=[dict(policy_id=p,source='weekly_library') for p in experts]
        gate=execution_gate(fit,policies)
        require(gate==group['execution_gates']['sum_only'],'库内执行门控改变')
    print('通过：未来扰动、完整期间成熟/缺单门控、美元账本、三个年龄历史校准拟合与执行门控。')


def main():
    """一次执行需要的检查；失败显示明确性质，避免长回测掩盖错误。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(ROOT))
    check_delivery()
    if args.core:
        check_core()


if __name__=='__main__':
    main()
