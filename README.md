# data-mining-hft

个人研究记录项目：使用高频行情数据尝试复现论文核心方法。

## 参考论文
- **论文**：《Auto-tuning of price prediction models for high-frequency trading via reinforcement learning》
- **文件**：[1679894.pdf](1679894.pdf)

## 数据说明
本项目包含以下两个高频 MBP-10（买卖十档深度）订单簿全量数据文件：
- `databento_glbx.mdp3_mbp_10.parquet`（CME 标普500 E-mini 期货，约 55MB）
- `databento_ifeu.impact_mbp_10.parquet`（ICE 布伦特原油期货，约 44MB）

### 为什么采用 Parquet 格式？
原始 CSV 单文件均超过 1GB（合计超 2.7GB），无法直接推送至 GitHub。本项目采用现代量化与数据科学通用的 **Parquet 列式无损存储（Zstandard 压缩）**：
1. **100% 全量无损**：全部 510 万行逐笔高频事件、纳秒时间戳、十档量价毫厘不失；
2. **高效轻便**：体积从 2.7GB 优化至约 99MB，支持直接随仓库完整开源，克隆即可用；
3. **极速读取**：相比传统 CSV，读取性能提升 20 倍以上。

### 复现需要转回 CSV 吗？
**完全不需要！** 项目代码已原生适配 Parquet 格式，克隆仓库后直接运行实验流水线即可：
```bash
python run_experiments.py
python generate_dashboard.py
```

> **注（可选转换）**：若需在外部特定软件中查看原始 CSV 格式，可在 Python 中执行单行代码一键还原：
> ```python
> import pandas as pd
> pd.read_parquet("databento_glbx.mdp3_mbp_10.parquet").to_csv("databento_glbx.mdp3_mbp_10.csv", index=False)
> ```

## 交付产物
- **复现实验报告**：[replication_report.md](replication_report.md)（实事求是剖析算法原理、真实回测及与原论文的差距缺陷）
- **可视化看板**：[visualization_dashboard.html](visualization_dashboard.html)（简约低调单页交互式图表系统）
