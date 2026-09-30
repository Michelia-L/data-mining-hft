"""生成无需联网的单文件交互看板：Python 注入 JSON，浏览器负责切换与绘图。

TEMPLATE 内包含 HTML 布局、CSS 样式和原生 JavaScript；使用 SVG 绘制曲线，
因此不依赖 CDN 图表库。数据只来自实验结果，不在浏览器里重新计算交易策略。
阅读时可先看 render_dashboard 如何嵌入数据，再看模板内 render/plot 如何展示。"""
import argparse
import json
from pathlib import Path
from src.config import BASE_DIR, SUMMARY_JSON_PATH

# 保持模板与数据分离：占位符只在 render_dashboard 中一次性替换。
TEMPLATE = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FMATO 思路的课程工程实验（非论文数值复现）</title>
<style>
/* 深色卡片和两列网格只负责布局；窄屏时改为单列，宽表允许横向滚动。 */
:root{color-scheme:dark;font:15px system-ui;color:#dbe5ee;background:#0b1220}body{max-width:1440px;margin:auto;padding:28px}
h1{font-size:26px}h2{font-size:18px}p{color:#a5b4c5;line-height:1.7}select{padding:9px;background:#182337;color:inherit;border:1px solid #334155;border-radius:6px}
.card{background:#111c2e;border:1px solid #26344b;border-radius:10px;padding:18px;margin:18px 0;overflow:auto}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .card{margin:0}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}
th,td{text-align:right;border-bottom:1px solid #26344b;padding:9px;font-size:13px}th:first-child,td:first-child{text-align:left}
svg{width:100%;height:260px}button{background:#26344b;color:inherit;border:0;padding:8px;cursor:pointer}.muted{color:#94a3b8}
@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:14px}}a{color:#7dd3fc}
</style>
<h1>FMATO 思路的课程工程实验（非论文数值复现）</h1>
<p>有限策略奖励学习 · 延迟反馈 · 下一事件成交 · 美元记账与逐事件盯市。当前数据仅支持同一文件内多个时间段的比较。CME/ICE 数据不复现论文中国期货实验数值。</p>
<label>品种 <select id="dataset"></select></label> <label>时间窗口 <select id="window"></select></label>
<div class="card" id="scope"></div><div class="card"><h2>验证选择与并列项</h2><div id="choices"></div></div>
<div class="grid"><div class="card"><h2>账户收益曲线（初始资金 $100,000）</h2><label>策略 <select id="strategy"></select></label> <label>曲线 <select id="metric"><option value="net">净收益</option><option value="gross">毛收益</option></select></label><div id="curve"></div></div>
<div class="card"><h2>多尺度奖励权重</h2><div id="weights"></div><p>有限可执行策略库的校准结果；权重不唯一，不代表尺度预测能力的独立证明。</p></div></div>
<div class="grid"><div class="card"><h2>测试起点前五档盘口</h2><p id="quote-time"></p><div id="book"></div></div>
<div class="card"><h2>模型选择频率</h2><div id="actions"></div></div></div>
<div class="card"><h2>测试策略明细</h2><div id="performance"></div></div>
<div class="card"><h2>预测与推理诊断</h2><p id="baseline"></p><div id="models"></div></div>
<div class="card"><h2>文件内多窗口描述统计（非独立重复实验）</h2><p>各窗重置资金；合计不是连续账户收益，窗口也不是独立交易日。</p><div id="aggregate"></div></div>
<div class="card"><h2>评估约定与限制</h2><p>数据按训练/奖励校准/验证/测试四段切分，并清除跨边界标签。仅在最长奖励周期到期后更新选择器。ME 使用与执行一致的门槛，OE 校准按成熟订单平均，默认不扣成本；NetOE 另列敏感性。成本包含点差、配置滑点与手续费，期末强制平仓。</p>
<p>默认价格差奖励不缩放、不裁剪；归一化收益率是另一种工程模式。非负权重是附加假设，SignedBox 检查负权敏感性。CausalShadowARS 是连续影子账户的 300 事件滑窗，未实现 Algorithm 3 的过去 30 分钟重回测；CausalEventUCB 每事件选择、无成交反馈零，未实现固定期间更新。模型库使用同一特征与训练期；短验证段重复选参有过拟合风险。详见仓库 docs/paper_alignment.md。</p><p>最大回撤用完整盯市曲线计算，展示曲线经过抽样。当前样本不足以形成独立日收益，因此不报告 Sharpe。延迟为单条预测 P50/P95，不包括特征提取、选择器及网络链路。撮合没有模拟排队、部分成交或市场冲击；不能直接外推实盘收益。</p></div>
<script>
// Python 在保存页面前替换占位符；打开 HTML 后直接读取内嵌数据，不请求远端服务。
const DATA = __DATA__;
const $ = id => document.getElementById(id);
// null 表示该指标当前无法定义（例如没有交易时的胜率），显示 N/A，而不是误导性的 0。
const fmt = (x, digits=2) => x == null ? 'N/A' : Number(x).toLocaleString('en-US',{minimumFractionDigits:digits,maximumFractionDigits:digits});
const pct = x => x == null ? 'N/A' : fmt(x*100)+'%';
// 通用表格渲染器：headers 为列名，rows 为二维单元格数组。
// 文本写入 textContent，避免把策略名称等数据字段当作 HTML 解释。
function table(id, headers, rows){
 const t=document.createElement('table'), head=document.createElement('tr');
 headers.forEach(h=>{const th=document.createElement('th');th.textContent=h;head.appendChild(th)});t.appendChild(head);
 rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td)});t.appendChild(tr)});
 $(id).replaceChildren(t);
}
// entries 的每项为 [内部值, 显示文本]；重建选项时默认回到第一项。
function options(id, entries){$(id).replaceChildren(...entries.map(([value,text])=>new Option(text,value)))}
// 浏览器下拉框的 value 是字符串；窗口索引需转成数字再取对应实验。
function current(){return DATA.experiments[$('dataset').value].windows[Number($('window').value)]}
// 只绘制当前策略和所选毛/净收益曲线；收益率乘 100 后显示为百分数。
function plot(){
 const w=current(), s=w.strategies[Number($('strategy').value)], ys=s[$('metric').value==='gross'?'gross_curve':'net_curve'].map(v=>v*100);
 // 纵轴包含零基准；全零曲线也给跨度设置下限，避免映射坐标时除零。
 const low=Math.min(0,...ys), high=Math.max(0,...ys), span=Math.max(high-low,.001), n=ys.length;
 // 横坐标使用真实事件序号而非点的数组下标，因为展示抽样的末端间距可能不同。
 // SVG 纵坐标向下增加，所以用 220 减去收益映射后的高度；这里只拼接数值坐标。
 const points=ys.map((y,i)=>`${50+800*(s.curve_steps[i]/Math.max(1,s.curve_steps[n-1]))},${220-180*(y-low)/span}`).join(' ');
 $('curve').innerHTML=`<svg viewBox="0 0 900 260" role="img" aria-label="策略净收益曲线"><line x1="50" x2="850" y1="220" y2="220" stroke="#475569"/><polyline points="${points}" stroke="#38bdf8" stroke-width="2" fill="none"/><text x="0" y="35" fill="#94a3b8">${fmt(high)}%</text><text x="0" y="222" fill="#94a3b8">${fmt(low)}%</text><text x="50" y="250" fill="#94a3b8">0</text><text x="700" y="250" fill="#94a3b8">${s.curve_steps[n-1]} events</text></svg>`;
}
// 切换品种/窗口后刷新全部表格和曲线；交易绩效直接取后端保存值，不重新年化。
function render(){
 const e=DATA.experiments[$('dataset').value], w=current(), t=w.partitions.test;
 $('scope').textContent=`${w.symbol} · 测试 ${t.start} → ${t.end} · ${fmt(t.duration_seconds)} 秒 · ${t.rows.toLocaleString()} 事件 · 前三段各清除至少 ${w.purge_events} 事件 · 门槛 ${w.tuning.threshold} · 奖励模式 ${w.reward_definition}`;
 // 索引日期是 UTC 文件分区；保留质量状态，避免把降级输入误认成正常完整交易日。
 const daily=DATA.metadata.daily_source;
 if(daily) $('scope').textContent+=` · UTC 文件日期 ${daily.file_date_utc} · 数据质量 ${daily.condition} · 单文件窗口，尚未跨日训练`;
 // 并列规则与候选完全来自后端记录，不根据测试表现再排序。
 const choices=[['门槛',w.tuning.threshold_choice],['固定模型',w.tuning.fixed_choice],['校准专家',w.calibration_expert_choice],...Object.entries(w.tuning.c_choices).map(([k,c])=>['UCB-'+k,c])];
 table('choices',['项目','选中','最佳/并列最佳候选','规则'],choices.map(([label,c])=>[label,c.selected,(c.tied_best_candidates.length>1?'并列最佳：':'最佳：')+c.tied_best_candidates.join(', '),c.tie_break_rule]));
 options('strategy',w.strategies.map((s,i)=>[i,s.strategy]));plot();
 const q=w.l2_snapshot; $('quote-time').textContent=q.timestamp;
 table('book',['档位','买量','买价','卖价','卖量'],q.bids.map((b,i)=>[i+1,fmt(b.size,0),fmt(b.price,3),fmt(q.asks[i].price,3),fmt(q.asks[i].size,0)]));
 table('actions',['策略',...w.model_eval.map(m=>m.name)],w.strategies.filter(s=>s.strategy.startsWith('Variant-')).map(s=>[s.strategy,...w.model_eval.map(m=>pct(s.action_distribution[m.name]||0))]));
 table('weights',['奖励','10 events','30 events','90 events','最终间隔','专家可表示'],[...Object.entries(w.rewards),...Object.entries(w.signed_weight_sensitivity).map(([k,r])=>[k+' SignedBox',r])].map(([k,r])=>[k,...r.weights.map(x=>fmt(x,4)),fmt(r.diagnostics.final_margin,6),String(r.diagnostics.expert_representable)]));
 table('performance',['策略','净盈亏 $','毛盈亏 $','成本 $','平仓笔数','胜率','每笔净盈亏 $','最大回撤'],w.strategies.map(s=>[s.strategy,fmt(s.net_pnl_usd),fmt(s.gross_pnl_usd),fmt(s.friction_usd),s.total_trades,pct(s.win_rate),fmt(s.avg_trade_net_usd),pct(s.max_drawdown)]));
 const b=w.prediction_baselines;
 $('baseline').textContent=`零收益预测准确率 ${fmt(b.zero_direction_accuracy)}%，训练多数类基线 ${fmt(b.train_majority_accuracy)}%；零收益 MSE ${b.zero_prediction_mse.toExponential(3)}。有标签 ${b.evaluated_rows} 行；尾部 ${b.unlabeled_tail} 行仅参与执行。`;
 table('models',['模型','原始符号 %','门槛三类 %','有效信号 %','覆盖率','分类 argmax %','MSE','单条 P50 μs','单条 P95 μs','批量 μs/行'],w.model_eval.map(m=>[m.name,fmt(m.raw_sign_accuracy),fmt(m.thresholded_signal_accuracy),fmt(m.active_signal_accuracy),pct(m.signal_coverage),fmt(m.classifier_argmax_accuracy),m.mse.toExponential(3),fmt(m.latency.single_p50_us),fmt(m.latency.single_p95_us),fmt(m.latency.batch_us_per_row,3)]));
 table('aggregate',['策略','合计净盈亏 $','平均 $','最差 $','最好 $','总平仓笔数'],e.aggregate.map(a=>[a.strategy,fmt(a.net_pnl_usd_sum),fmt(a.net_pnl_usd_mean),fmt(a.net_pnl_usd_min),fmt(a.net_pnl_usd_max),a.total_trades]));
}
// 品种变化时先重建其窗口列表，再刷新内容，避免沿用另一个品种的窗口索引。
function datasetChanged(){options('window',DATA.experiments[$('dataset').value].windows.map((w,i)=>[i,`${i+1} · offset ${w.source_offset}`]));render()}
// 初始化选项并绑定事件：策略/毛净切换只重画曲线，品种/窗口切换刷新整个看板。
options('dataset',Object.keys(DATA.experiments).map(k=>[k,k]));
$('dataset').onchange=datasetChanged;$('window').onchange=render;$('strategy').onchange=plot;$('metric').onchange=plot;datasetChanged();
</script></html>'''


def render_dashboard(data):
    """校验结果版本，将整份结果嵌入 HTML 模板并返回字符串。

    JSON 中的中文原样保留；NaN/Infinity 被禁止。把小于号替换为 JSON 的
    Unicode 转义，防止数据中的结束脚本标签提前结束 HTML 脚本节点。
    这项转义不会改变浏览器解码后的字段值，且只用于展示数据的安全嵌入。"""
    if data.get('schema_version') != 3:
        raise ValueError('Regenerate experiments with schema version 3')
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    return TEMPLATE.replace('__DATA__', payload)


def main():
    """读取结果文件并输出完整离线 HTML；可用命令行指定临时输入和输出做快速检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=SUMMARY_JSON_PATH)
    parser.add_argument('--output', default=str(BASE_DIR / 'visualization_dashboard.html'))
    args = parser.parse_args()
    Path(args.output).write_text(render_dashboard(json.loads(Path(args.input).read_text())))


if __name__ == '__main__':
    main()
