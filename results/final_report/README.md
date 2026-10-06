# 原始实验摘要证据

组员默认阅读 [课程报告](../../final_replication_report.md)与 [展示讲稿](../../presentation.md)。本目录用于核查和教师追问，三份快照、索引及两张原图按清理前字节保留。

| 文件 | 含义 | 对应原源码提交 |
| --- | --- | --- |
| [development_snapshot.json](development_snapshot.json) | OE开发对照；24项状态，评价10-13至10-21 | 115bc8d |
| [four_combinations_snapshot.json](four_combinations_snapshot.json) | ME/OE与UCB/ARS扩展；34项状态，旧OE继承而非重跑 | acc7e85 |
| [frozen_ablation_snapshot.json](frozen_ablation_snapshot.json) | 后段七日冻结复核；27项状态，课程主表来源 | c406441 |

这些是摘要，未包含完整逐笔账本和原始行情。市场真实性、全部成交因果性与统计显著性不能靠摘要哈希证明。

[evidence_index.json](evidence_index.json)保留原交付索引及SHA256。其`delivery_documents`列的是原交付时文档，部分已在课程瘦身中删除；它是历史清单，不作为当前导航。当前核对由`scripts/check_project.py`执行，固定快照和图身份，检查课程摘要及展示引用，不要求当前源码等于历史源码，也不会更新原哈希。

两张原图为 [完整逐日净利](figures/daily_net_usd.png)和 [独立日净利累计](figures/cumulative_daily_net_usd.png)。后者不是复利账户。课堂使用的两张简图另存`results/presentation/`，不会覆盖原图。
