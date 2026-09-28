"""Generate a self-contained offline dashboard from versioned experiment results."""
import argparse
import json
from pathlib import Path
from src.config import BASE_DIR, SUMMARY_JSON_PATH

TEMPLATE = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FMATO 因果回测实验</title>
<style>
:root{color-scheme:dark;font:15px system-ui;color:#dbe5ee;background:#0b1220}body{max-width:1440px;margin:auto;padding:28px}
h1{font-size:26px}h2{font-size:18px}p{color:#a5b4c5;line-height:1.7}select{padding:9px;background:#182337;color:inherit;border:1px solid #334155;border-radius:6px}
.card{background:#111c2e;border:1px solid #26344b;border-radius:10px;padding:18px;margin:18px 0;overflow:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .card{margin:0}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}
th,td{text-align:right;border-bottom:1px solid #26344b;padding:9px;font-size:13px}th:first-child,td:first-child{text-align:left}
svg{width:100%;height:260px}button{background:#26344b;color:inherit;border:0;padding:8px;cursor:pointer}.muted{color:#94a3b8}
@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:14px}}a{color:#7dd3fc}
</style>
<h1>FMATO 因果回测实验</h1>
<p>有限策略奖励学习 · 延迟反馈 · 下一事件成交 · 美元记账与逐事件盯市。当前数据仅支持同一文件内多个时间段的比较。</p>
<label>品种 <select id="dataset"></select></label> <label>时间窗口 <select id="window"></select></label>
<div class="card" id="scope"></div>
<div class="grid"><div class="card"><h2>账户收益曲线（初始资金 $100,000）</h2><label>策略 <select id="strategy"></select></label> <label>曲线 <select id="metric"><option value="net">净收益</option><option value="gross">毛收益</option></select></label><div id="curve"></div></div>
<div class="card"><h2>多尺度奖励权重</h2><div id="weights"></div><p>有限可执行策略库的校准结果；权重不唯一，不代表尺度预测能力的独立证明。</p></div></div>
<div class="grid"><div class="card"><h2>测试起点前五档盘口</h2><p id="quote-time"></p><div id="book"></div></div>
<div class="card"><h2>模型选择频率</h2><div id="actions"></div></div></div>
<div class="card"><h2>测试策略明细</h2><div id="performance"></div></div>
<div class="card"><h2>预测与推理诊断</h2><p id="baseline"></p><div id="models"></div></div>
<div class="card"><h2>跨窗口汇总（美元）</h2><p>各窗重置资金；合计不是连续账户收益，窗口也不是独立交易日。</p><div id="aggregate"></div></div>
<div class="card"><h2>评估约定与限制</h2><p>数据按训练/奖励校准/验证/测试四段切分，并清除跨边界标签。仅在最长奖励周期到期后更新选择器。ME 按预测信号评价，OE 按真实或独立影子账户成交评价。成本包含点差、配置滑点与手续费，期末强制平仓。</p>
<p>最大回撤用完整盯市曲线计算，展示曲线经过抽样。年化夏普不适用于当前短样本，记为 N/A。延迟为单条预测 P50/P95，不包括特征提取、选择器及网络链路。撮合没有模拟排队、部分成交或市场冲击；不能直接外推实盘收益。</p></div>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (x, digits=2) => x == null ? 'N/A' : Number(x).toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits});
const pct = x => x == null ? 'N/A' : fmt(x*100)+'%';
function table(id, headers, rows){
 const t=document.createElement('table'), head=document.createElement('tr');
 headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.appendChild(th)});t.appendChild(head);
 rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});t.appendChild(tr)});
 $(id).replaceChildren(t);
}
function options(id, entries){$(id).replaceChildren(...entries.map(([value,text])=>new Option(text,value)))}
function current(){return DATA.experiments[$('dataset').value].windows[Number($('window').value)]}
function plot(){
 const w=current(), s=w.strategies[Number($('strategy').value)], ys=s[$('metric').value==='gross'?'gross_curve':'net_curve'].map(v=>v*100);
 const low=Math.min(0,...ys), high=Math.max(0,...ys), span=Math.max(high-low,.001), n=ys.length;
 const points=ys.map((y,i)=>`${50+800*(s.curve_steps[i]/Math.max(1,s.curve_steps[n-1]))},${220-180*(y-low)/span}`).join(' ');
 $('curve').innerHTML=`<svg viewBox="0 0 900 260" role="img" aria-label="策略净收益曲线"><line x1="50" x2="850" y1="220" y2="220" stroke="#475569"/><polyline points="${points}" stroke="#38bdf8" stroke-width="2" fill="none"/><text x="0" y="35" fill="#94a3b8">${fmt(high)}%</text><text x="0" y="222" fill="#94a3b8">${fmt(low)}%</text><text x="50" y="250" fill="#94a3b8">0</text><text x="700" y="250" fill="#94a3b8">${s.curve_steps[n-1]} events</text></svg>`;
}
function render(){
 const e=DATA.experiments[$('dataset').value], w=current(), t=w.partitions.test;
 $('scope').textContent=`${w.symbol} · 测试 ${t.start} → ${t.end} · ${fmt(t.duration_seconds)} 秒 · ${t.rows.toLocaleString()} 事件 · 前三段各清除至少 ${w.purge_events} 事件 · 门槛 ${w.tuning.threshold} · 验证最佳模型 ${w.tuning.validation_best_model}`;
 options('strategy',w.strategies.map((s,i)=>[i,s.strategy]));plot();
 const q=w.l2_snapshot; $('quote-time').textContent=q.timestamp;
 table('book',['档位','买量','买价','卖价','卖量'],q.bids.map((b,i)=>[i+1,fmt(b.size,0),fmt(b.price,3),fmt(q.asks[i].price,3),fmt(q.asks[i].size,0)]));
 table('actions',['策略',...w.model_eval.map(m=>m.name)],w.strategies.filter(s=>s.strategy.startsWith('FMATO-')).map(s=>[s.strategy,...w.model_eval.map(m=>pct(s.action_distribution[m.name]||0))]));
 table('weights',['奖励','10 events','30 events','90 events','最终间隔','专家可表示'],Object.entries(w.rewards).map(([k,r])=>[k,...r.weights.map(x=>fmt(x,4)),fmt(r.diagnostics.final_margin,6),String(r.diagnostics.expert_representable)]));
 table('performance',['策略','净盈亏 $','毛盈亏 $','成本 $','平仓笔数','胜率','每笔净盈亏 $','最大回撤','夏普'],w.strategies.map(s=>[s.strategy,fmt(s.net_pnl_usd),fmt(s.gross_pnl_usd),fmt(s.friction_usd),s.total_trades,pct(s.win_rate),fmt(s.avg_trade_net_usd),pct(s.max_drawdown),fmt(s.sharpe_ratio)]));
 const b=w.prediction_baselines;
 $('baseline').textContent=`零收益预测准确率 ${fmt(b.zero_direction_accuracy)}%，训练多数类基线 ${fmt(b.train_majority_accuracy)}%；零收益 MSE ${b.zero_prediction_mse.toExponential(3)}。有标签 ${b.evaluated_rows} 行；尾部 ${b.unlabeled_tail} 行仅参与执行。`;
 table('models',['模型','准确率 %','平衡准确率 %','非零方向 %','MSE','单条 P50 μs','单条 P95 μs','批量 μs/行'],w.model_eval.map(m=>[m.name,fmt(m.direction_accuracy),fmt(m.balanced_accuracy),fmt(m.nonzero_direction_accuracy),m.mse.toExponential(3),fmt(m.latency.single_p50_us),fmt(m.latency.single_p95_us),fmt(m.latency.batch_us_per_row,3)]));
 table('aggregate',['策略','合计净盈亏 $','平均 $','最差 $','最好 $','总平仓笔数'],e.aggregate.map(a=>[a.strategy,fmt(a.net_pnl_usd_sum),fmt(a.net_pnl_usd_mean),fmt(a.net_pnl_usd_min),fmt(a.net_pnl_usd_max),a.total_trades]));
}
function datasetChanged(){options('window',DATA.experiments[$('dataset').value].windows.map((w,i)=>[i,`${i+1} · offset ${w.source_offset}`]));render()}
options('dataset',Object.keys(DATA.experiments).map(k=>[k,k]));
$('dataset').onchange=datasetChanged;$('window').onchange=render;$('strategy').onchange=plot;$('metric').onchange=plot;datasetChanged();
</script></html>'''


def render_dashboard(data):
    if data.get('schema_version') != 2:
        raise ValueError('Regenerate experiments with schema version 2')
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return TEMPLATE.replace('__DATA__', payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=SUMMARY_JSON_PATH)
    parser.add_argument('--output', default=str(BASE_DIR / 'visualization_dashboard.html'))
    args = parser.parse_args()
    Path(args.output).write_text(render_dashboard(json.loads(Path(args.input).read_text())))


if __name__ == '__main__':
    main()
