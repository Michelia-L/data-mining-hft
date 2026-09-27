"""
可视化展示网页生成脚本 (generate_dashboard.py)
读取 results/experiment_summary.json，生成独立单页面 HTML 看板 (visualization_dashboard.html)。
风格设计：简约低调、低饱和度现代暗调学术风，内容饱满丰富，涵盖 L2 深度、模型库基准、IRL 权重、选择分布与净值曲线。
"""

import os
import json
from src.config import SUMMARY_JSON_PATH, BASE_DIR

OUTPUT_HTML_PATH = os.path.join(BASE_DIR, "visualization_dashboard.html")

def build_dashboard():
    if not os.path.exists(SUMMARY_JSON_PATH):
        raise FileNotFoundError(f"未找到实验结果文件: {SUMMARY_JSON_PATH}，请先运行 run_experiments.py")

    with open(SUMMARY_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取默认数据集 (ICE_BRENT 或 CME_ES) 的代表性数据
    exp_brent = data["experiments"].get("ICE_BRENT", {})
    exp_es = data["experiments"].get("CME_ES", {})
    
    # 序列化为安全内嵌 JSON 字符串
    json_str = json.dumps(data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FMATO 核心方法复现看板 - 数据挖掘量化实验</title>
    <!-- Chart.js 现代数据可视化库 -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-primary: #0f141c;
            --bg-secondary: #161e2b;
            --bg-card: #1c2638;
            --border-color: #2b3952;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-amber: #fbbf24;
            --accent-rose: #fb7185;
            --accent-purple: #c084fc;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1360px;
            margin: 0 auto;
        }}

        /* 页头设计 */
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}

        .header-title h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }}

        .header-title p {{
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .dataset-selector {{
            display: flex;
            gap: 8px;
            background-color: var(--bg-secondary);
            padding: 4px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .btn-tab {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.8125rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-tab.active {{
            background-color: var(--bg-card);
            color: #fff;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}

        /* KPI 卡片栏 */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px 20px;
        }}

        .kpi-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f1f5f9;
        }}

        .kpi-subtext {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        /* 栅格内容区 */
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        @media (max-width: 992px) {{
            .grid-2 {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: #f1f5f9;
        }}

        .card-desc {{
            font-size: 0.8125rem;
            color: var(--text-muted);
        }}

        /* 表格样式 */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8125rem;
            text-align: left;
        }}

        .data-table th {{
            color: var(--text-muted);
            font-weight: 500;
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
        }}

        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid rgba(43, 57, 82, 0.4);
            color: var(--text-secondary);
        }}

        .data-table tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
            color: #fff;
        }}

        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        .badge-positive {{
            background-color: rgba(52, 211, 153, 0.15);
            color: var(--accent-green);
        }}

        .badge-negative {{
            background-color: rgba(251, 113, 133, 0.15);
            color: var(--accent-rose);
        }}

        .badge-neutral {{
            background-color: rgba(148, 163, 184, 0.15);
            color: var(--text-secondary);
        }}

        /* 盘口深度 Ladder */
        .ladder-container {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 10px;
        }}

        .ladder-row {{
            display: flex;
            align-items: center;
            font-size: 0.8125rem;
            height: 24px;
        }}

        .ladder-label {{
            width: 50px;
            color: var(--text-muted);
            font-size: 0.75rem;
        }}

        .ladder-price {{
            width: 80px;
            font-family: monospace;
            font-weight: 600;
        }}

        .ladder-bar-wrap {{
            flex: 1;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }}

        .ladder-bar {{
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }}

        .ladder-size {{
            position: absolute;
            right: 8px;
            top: 2px;
            font-size: 0.75rem;
            color: #fff;
            font-family: monospace;
        }}

        .ask-side .ladder-price {{ color: var(--accent-rose); }}
        .ask-side .ladder-bar {{ background-color: rgba(251, 113, 133, 0.35); }}
        .bid-side .ladder-price {{ color: var(--accent-green); }}
        .bid-side .ladder-bar {{ background-color: rgba(52, 211, 153, 0.35); }}

        /* 缺陷与差距警告框 */
        .discrepancy-box {{
            background-color: rgba(251, 191, 36, 0.05);
            border: 1px solid rgba(251, 191, 36, 0.2);
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
        }}

        .discrepancy-title {{
            color: var(--accent-amber);
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .discrepancy-list {{
            list-style: none;
            font-size: 0.8125rem;
            color: #cbd5e1;
            line-height: 1.6;
        }}

        .discrepancy-list li {{
            margin-bottom: 6px;
            padding-left: 16px;
            position: relative;
        }}

        .discrepancy-list li::before {{
            content: "•";
            position: absolute;
            left: 4px;
            color: var(--accent-amber);
        }}

        footer {{
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>

<div class="container">
    <!-- 顶部状态栏 -->
    <header>
        <div class="header-title">
            <h1>FMATO 核心方法最小复现看板</h1>
            <p>基于强化学习的高频价格预测模型在线调优 · 数据挖掘课程实验 (Pattern Recognition 2022)</p>
        </div>
        <div class="dataset-selector">
            <button class="btn-tab active" onclick="switchDataset('ICE_BRENT')">ICE Brent 原油期货</button>
            <button class="btn-tab" onclick="switchDataset('CME_ES')">CME E-mini 标普期货</button>
        </div>
    </header>

    <!-- KPI 核心指标栏 -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">高频事件样本数</div>
            <div class="kpi-value" id="kpi-sample">59,904</div>
            <div class="kpi-subtext" id="kpi-sample-sub">训练 41,932 / 测试 17,972</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">多尺度 IRL 权重 (10t / 30t / 90t)</div>
            <div class="kpi-value" style="font-size: 1.15rem; font-family: monospace;" id="kpi-irl">0.16 / 0.27 / 0.57</div>
            <div class="kpi-subtext">长周期趋势获得更高权重分配</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">最高策略毛收益 (Gross Alpha)</div>
            <div class="kpi-value" style="color: var(--accent-green);" id="kpi-gross">+2.20%</div>
            <div class="kpi-subtext" id="kpi-gross-sub">FMATO-OE-UCBS (未扣费)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">交易摩擦侵蚀 (Friction Drag)</div>
            <div class="kpi-value" style="color: var(--accent-rose);" id="kpi-friction">-7.94%</div>
            <div class="kpi-subtext">点差与手续费完全吞噬微观毛利</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">轻量模型推理延迟</div>
            <div class="kpi-value" id="kpi-latency">0.28 <span style="font-size: 0.875rem; font-weight: normal; color: var(--text-muted);">us/tick</span></div>
            <div class="kpi-subtext">满足高频微秒级实时响应</div>
        </div>
    </div>

    <!-- 图表栏 1: 策略净值走势对比与模型选择分布 -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">策略累计毛收益曲线 (Gross PnL - 纯信号能力)</div>
                    <div class="card-desc">对比 4 种 FMATO 变体与 3 个基准策略在测试集上的预测毛收益</div>
                </div>
            </div>
            <div style="height: 320px;">
                <canvas id="chartGrossPnL"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">真实累计净收益曲线 (Net PnL - 扣除点差与手续费)</div>
                    <div class="card-desc">如实反映高频策略在面临真实流动性成本摩擦时的真实净值</div>
                </div>
            </div>
            <div style="height: 320px;">
                <canvas id="chartNetPnL"></canvas>
            </div>
        </div>
    </div>

    <!-- 图表栏 2: 模型库性能基准与 IRL 权重分配 -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">轻量模型库单体性能基准 (Model Library Benchmark)</div>
                    <div class="card-desc">5 个异构动作模型在测试集上的方向准确率与推理耗时</div>
                </div>
            </div>
            <div style="height: 280px;">
                <canvas id="chartModelBenchmark"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">逆强化学习 (IRL) 多尺度时域期望权重</div>
                    <div class="card-desc">通过专家轨迹特征期望匹配学得的最优权重向量 w*</div>
                </div>
            </div>
            <div style="height: 280px;">
                <canvas id="chartIRLWeights"></canvas>
            </div>
        </div>
    </div>

    <!-- 盘口切片与回测绩效详细指标表 -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">L2 订单簿微观结构深度切片 (MBP-10)</div>
                    <div class="card-desc">当前测试样本盘口前5档挂单深度与买卖不平衡度 (OBI)</div>
                </div>
            </div>
            <div class="ladder-container" id="l2-ladder">
                <!-- 动态填充盘口挂单 -->
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div>
                    <div class="card-title">策略在线选择频率分布 (Action Selection Rate)</div>
                    <div class="card-desc">展示 UCB 与 ARS 算法在时序动态演变中挑选不同候选模型的占比</div>
                </div>
            </div>
            <div style="height: 240px;">
                <canvas id="chartActionDist"></canvas>
            </div>
        </div>
    </div>

    <!-- 全策略综合回测结果表格 -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
            <div>
                <div class="card-title">全策略实证回测指标明细表 (Performance Summary)</div>
                <div class="card-desc">真实测试集执行统计，严格区分毛收益、扣费摩擦与真实净收益</div>
            </div>
        </div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>策略名称</th>
                    <th>成交笔数</th>
                    <th>胜率 (Win Rate)</th>
                    <th>盈亏比 (P/L)</th>
                    <th>累计毛收益 (Gross)</th>
                    <th>交易摩擦扣减 (Friction)</th>
                    <th>真实净收益 (Net)</th>
                    <th>夏普比率 (Sharpe)</th>
                    <th>最大回撤 (MDD)</th>
                </tr>
            </thead>
            <tbody id="table-strategy-body">
                <!-- 动态填充 -->
            </tbody>
        </table>

        <!-- 差异与缺陷说明框 (严格符合用户要求) -->
        <div class="discrepancy-box">
            <div class="discrepancy-title">
                <span>⚠</span> 复现差距、缺陷与不足深度说明 (Discrepancy & Limitations)
            </div>
            <ul class="discrepancy-list">
                <li><strong>数据尺度差距</strong>：原论文使用中国商品期货交易所数月甚至全年全量 L2 逐笔快照，具备跨周期宏观趋势；本复现受限于本地数据，仅能使用单日高频事件切片，无法评估跨日与长期周期迁移。</li>
                <li><strong>撮合排队队列简化</strong>：原论文具有实盘高频交易接入系统与精细回测子系统，支持排队位置仿真；本复现基于对价吃单与半点差滑点简化假设，未考虑挂单排队被动成交以及订单取消撤单延迟。</li>
                <li><strong>高频交易摩擦吞噬</strong>：实验客观证明，FMATO 的信号预测能力能够产生正向毛收益（Gross Return 为 +1% ~ +2%），但由于高频频繁调仓（数百笔），交易所手续费与半点差将毛利完全吞噬，导致净收益为负。这正是学术界高频策略复现中最常见的真实挑战。</li>
                <li><strong>探索与过度交易的权衡</strong>：UCB 算法因不确定性激励，探索频次较高，导致交易过于活跃；而 ARS 算法交易更为谨慎保守，摩擦损失显著降低，提示高频场景下应严格增加“开仓惩罚项”。</li>
            </ul>
        </div>
    </div>

    <footer>
        <p>数据挖掘课程作业 · FMATO 算法最小可行复现项目 · 独立单页可视化展示系统</p>
    </footer>
</div>

<script>
    // 注入后端完整实验数据
    const EXPERIMENT_DATA = {json_str};

    let currentDatasetKey = "ICE_BRENT";
    let chartGross = null;
    let chartNet = null;
    let chartBenchmark = null;
    let chartIRL = null;
    let chartAction = null;

    // 颜色配置
    const colors = [
        '#38bdf8', '#34d399', '#fbbf24', '#fb7185',
        '#c084fc', '#94a3b8', '#f43f5e'
    ];

    function renderDashboard(datasetKey) {{
        const exp = EXPERIMENT_DATA.experiments[datasetKey];
        if (!exp) return;

        // 1. 更新 KPI 指标
        document.getElementById('kpi-sample').innerText = exp.sample_size.toLocaleString();
        document.getElementById('kpi-sample-sub').innerText = `训练 ${{exp.train_size.toLocaleString()}} / 测试 ${{exp.test_size.toLocaleString()}}`;

        const w = exp.irl_weights;
        document.getElementById('kpi-irl').innerText = `${{w["10_ticks"]}} / ${{w["30_ticks"]}} / ${{w["90_ticks"]}}`;

        // 寻找最高毛收益策略
        let maxGross = -999;
        let maxGrossStrat = "";
        let maxFriction = 0;
        exp.strategies.forEach(s => {{
            if (s.gross_return > maxGross) {{
                maxGross = s.gross_return;
                maxGrossStrat = s.strategy;
            }}
            if (s.total_friction > maxFriction) {{
                maxFriction = s.total_friction;
            }}
        }});
        document.getElementById('kpi-gross').innerText = (maxGross * 100).toFixed(2) + "%";
        document.getElementById('kpi-gross-sub').innerText = maxGrossStrat + " (未扣费)";
        document.getElementById('kpi-friction').innerText = "-" + (maxFriction * 100).toFixed(2) + "%";

        // 平均延迟
        const avgLat = exp.model_eval.reduce((a, b) => a + b.latency_us, 0) / exp.model_eval.length;
        document.getElementById('kpi-latency').innerHTML = `${{avgLat.toFixed(2)}} <span style="font-size: 0.875rem; font-weight: normal; color: var(--text-muted);">us/tick</span>`;

        // 2. 渲染策略表格
        const tbody = document.getElementById('table-strategy-body');
        tbody.innerHTML = '';
        exp.strategies.forEach(s => {{
            const tr = document.createElement('tr');
            const grossBadge = s.gross_return >= 0 ? 'badge-positive' : 'badge-negative';
            const netBadge = s.net_return >= 0 ? 'badge-positive' : 'badge-negative';

            tr.innerHTML = `
                <td style="font-weight: 500; color: #f8fafc;">${{s.strategy}}</td>
                <td>${{s.total_trades}}</td>
                <td>${{(s.win_rate * 100).toFixed(1)}}%</td>
                <td>${{s.pl_ratio.toFixed(2)}}</td>
                <td><span class="badge ${{grossBadge}}">${{(s.gross_return * 100).toFixed(2)}}%</span></td>
                <td style="color: var(--accent-rose);">${{(s.total_friction * 100).toFixed(2)}}%</td>
                <td><span class="badge ${{netBadge}}">${{(s.net_return * 100).toFixed(2)}}%</span></td>
                <td>${{s.sharpe_ratio.toFixed(2)}}</td>
                <td>${{(s.max_drawdown * 100).toFixed(2)}}%</td>
            `;
            tbody.appendChild(tr);
        }});

        // 3. 渲染盘口切片 Ladder
        const ladderEl = document.getElementById('l2-ladder');
        ladderEl.innerHTML = '';
        if (exp.l2_snapshot && exp.l2_snapshot.length > 0) {{
            const snap = exp.l2_snapshot[0];
            const maxVol = Math.max(
                ...snap.asks.map(a => a.size),
                ...snap.bids.map(b => b.size)
            );

            // 倒序展示 Asks (从 Ask 4 到 Ask 0)
            for (let i = 4; i >= 0; i--) {{
                const item = snap.asks[i];
                const pct = (item.size / maxVol) * 100;
                ladderEl.innerHTML += `
                    <div class="ladder-row ask-side">
                        <div class="ladder-label">卖 ${{i+1}}</div>
                        <div class="ladder-price">${{item.price.toFixed(2)}}</div>
                        <div class="ladder-bar-wrap">
                            <div class="ladder-bar" style="width: ${{pct}}%;"></div>
                            <span class="ladder-size">${{item.size}}</span>
                        </div>
                    </div>
                `;
            }}

            ladderEl.innerHTML += `
                <div style="text-align: center; font-size: 0.75rem; color: var(--text-muted); padding: 4px 0; border-top: 1px dashed var(--border-color); border-bottom: 1px dashed var(--border-color); margin: 4px 0;">
                    Spread: ${{snap.spread.toFixed(2)}} | OBI: ${{snap.obi_l1.toFixed(3)}}
                </div>
            `;

            // 正序展示 Bids (从 Bid 0 到 Bid 4)
            for (let i = 0; i < 5; i++) {{
                const item = snap.bids[i];
                const pct = (item.size / maxVol) * 100;
                ladderEl.innerHTML += `
                    <div class="ladder-row bid-side">
                        <div class="ladder-label">买 ${{i+1}}</div>
                        <div class="ladder-price">${{item.price.toFixed(2)}}</div>
                        <div class="ladder-bar-wrap">
                            <div class="ladder-bar" style="width: ${{pct}}%;"></div>
                            <span class="ladder-size">${{item.size}}</span>
                        </div>
                    </div>
                `;
            }}
        }}

        // 4. 渲染图表
        renderCharts(exp);
    }}

    function renderCharts(exp) {{
        const labels = exp.strategies[0].timestamps.map((t, idx) => `T${{idx*50}}`);

        // A. 毛收益曲线
        if (chartGross) chartGross.destroy();
        chartGross = new Chart(document.getElementById('chartGrossPnL'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: exp.strategies.map((s, i) => ({{
                    label: s.strategy,
                    data: s.gross_curve.map(v => (v * 100).toFixed(2)),
                    borderColor: colors[i % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    tension: 0.1
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#94a3b8', callback: v => v + '%' }}
                    }}
                }},
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1', boxWidth: 12, font: {{ size: 10 }} }} }}
                }}
            }}
        }});

        // B. 真实净收益曲线
        if (chartNet) chartNet.destroy();
        chartNet = new Chart(document.getElementById('chartNetPnL'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: exp.strategies.map((s, i) => ({{
                    label: s.strategy,
                    data: s.net_curve.map(v => (v * 100).toFixed(2)),
                    borderColor: colors[i % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    tension: 0.1
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ display: false }},
                    y: {{
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        ticks: {{ color: '#94a3b8', callback: v => v + '%' }}
                    }}
                }},
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ color: '#cbd5e1', boxWidth: 12, font: {{ size: 10 }} }} }}
                }}
            }}
        }});

        // C. 模型库性能
        if (chartBenchmark) chartBenchmark.destroy();
        chartBenchmark = new Chart(document.getElementById('chartModelBenchmark'), {{
            type: 'bar',
            data: {{
                labels: exp.model_eval.map(m => m.name),
                datasets: [
                    {{
                        label: '方向预测准确率 (%)',
                        data: exp.model_eval.map(m => m.direction_accuracy),
                        backgroundColor: 'rgba(56, 189, 248, 0.4)',
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                        yAxisID: 'y'
                    }},
                    {{
                        label: '单次推理延迟 (us)',
                        data: exp.model_eval.map(m => m.latency_us),
                        backgroundColor: 'rgba(251, 191, 36, 0.4)',
                        borderColor: '#fbbf24',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1', font: {{ size: 10 }} }} }},
                    y: {{ position: 'left', grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#38bdf8' }} }},
                    y1: {{ position: 'right', grid: {{ display: false }}, ticks: {{ color: '#fbbf24' }} }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 11 }} }} }}
                }}
            }}
        }});

        // D. IRL 权重
        if (chartIRL) chartIRL.destroy();
        const wData = exp.irl_weights;
        chartIRL = new Chart(document.getElementById('chartIRLWeights'), {{
            type: 'doughnut',
            data: {{
                labels: ['10 Ticks (短周期)', '30 Ticks (中周期)', '90 Ticks (长周期)'],
                datasets: [{{
                    data: [wData['10_ticks'], wData['30_ticks'], wData['90_ticks']],
                    backgroundColor: ['#38bdf8', '#34d399', '#c084fc'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'right', labels: {{ color: '#cbd5e1', font: {{ size: 11 }} }} }}
                }}
            }}
        }});

        // E. 动作选择分布 (对比 FMATO-ME-UCBS 与 FMATO-ME-ARS)
        if (chartAction) chartAction.destroy();
        const ucbs = exp.strategies.find(s => s.strategy === 'FMATO-ME-UCBS');
        const ars = exp.strategies.find(s => s.strategy === 'FMATO-ME-ARS');
        const modelNames = exp.model_eval.map(m => m.name);

        chartAction = new Chart(document.getElementById('chartActionDist'), {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [
                    {{
                        label: 'FMATO-ME-UCBS 选择占比',
                        data: modelNames.map(m => ((ucbs.action_distribution[m] || 0) * 100).toFixed(1)),
                        backgroundColor: 'rgba(56, 189, 248, 0.5)',
                        borderColor: '#38bdf8',
                        borderWidth: 1
                    }},
                    {{
                        label: 'FMATO-ME-ARS 选择占比',
                        data: modelNames.map(m => ((ars.action_distribution[m] || 0) * 100).toFixed(1)),
                        backgroundColor: 'rgba(52, 211, 153, 0.5)',
                        borderColor: '#34d399',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ ticks: {{ color: '#cbd5e1', font: {{ size: 10 }} }} }},
                    y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8', callback: v => v + '%' }} }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#cbd5e1', font: {{ size: 10 }} }} }}
                }}
            }}
        }});
    }}

    function switchDataset(key) {{
        currentDatasetKey = key;
        document.querySelectorAll('.btn-tab').forEach(b => b.classList.remove('active'));
        if (event && event.target) {{
            event.target.classList.add('active');
        }}
        renderDashboard(key);
    }}

    // 页面初始化
    window.onload = () => {{
        renderDashboard('ICE_BRENT');
    }};
</script>
</body>
</html>
"""

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✔ 可视化展示网页已成功生成: {OUTPUT_HTML_PATH}")


if __name__ == "__main__":
    build_dashboard()
