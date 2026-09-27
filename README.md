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
