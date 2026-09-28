# data-mining-hft

高频量化交易模型在线自适应调优 (FMATO) 核心方法最小复现项目。

## 参考论文
- **论文**：《Auto-tuning of price prediction models for high-frequency trading via reinforcement learning》
- **文件**：[1679894.pdf](1679894.pdf)

## 数据文件
仓库中已包含两个高频 MBP-10（买卖十档深度）订单簿全量数据集：
- `databento_glbx.mdp3_mbp_10.parquet`：CME 标普500 E-mini 期货
- `databento_ifeu.impact_mbp_10.parquet`：ICE 布伦特原油期货

## 运行步骤

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 执行回测实验
自动完成高频数据特征提取、轻量模型库训练、IRL 多尺度奖励权重学习与在线动态选择回测：
```bash
python run_experiments.py
```
实验指标与结果汇总将输出至 `results/experiment_summary.json`。

### 3. 生成可视化看板
```bash
python generate_dashboard.py
```
运行后将生成单文件交互式看板 `visualization_dashboard.html`，可在任意浏览器中直接打开查看。

## 交付产物
- **复现实验报告**：[replication_report.md](replication_report.md)（实事求是记录算法原理、回测结果及与原论文的差距缺陷）
- **可视化看板**：[visualization_dashboard.html](visualization_dashboard.html)（网页交互式图表）

## BTCUSDT 1 秒数据能力边界复现

新增一条独立的 BTCUSDT 1 秒 K 线实验路径，不修改原有 ES/Brent MBP-10 实验。由于 BTC 数据不含 L2 盘口，实验不伪造 OBI、microprice 或真实 bid/ask spread，而是使用 OHLCV、成交笔数、taker-buy ratio 与成交方向失衡等可观测特征来复现 FMATO 的核心算法框架。

数据：`data/BTCUSDT-1s-2024-03-01.part*.csv`（原始 CSV 无损拆成 9 个顺序分片；合计 86,400 行，覆盖 2024-03-01 UTC 全天连续 1 秒样本）。

运行：

```bash
python run_btc_experiment.py
```

输出：`results/btc_experiment_summary.json`。

完整方法、运行结果、成本敏感性、因果延迟奖励设计及局限说明见 [btc_replication_report.md](btc_replication_report.md)。
