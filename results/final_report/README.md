# 最终报告证据与复核索引

主文档为[中文FMATO部分复现报告](../../final_replication_report.md)。本目录保存
三批已完成实验的紧凑证据与两张同源日图；定稿保留其原字节，不重新拟合、
选择日期或生成新收益。完整身份由[evidence_index.json](evidence_index.json)
固定；它是文件完整性索引，不是原数据真实性或统计有效性证明。

## 三批证据各回答什么

| 文件 | 报告位置 | 评价日期（2025年） | 内容和边界 |
| --- | --- | --- | --- |
| [development_snapshot.json](development_snapshot.json) | §5.1–§5.3、§7.1–§7.3 | 10-13/14/15/16/17/20/21 | 原OE开发24策略、504策略日；保留校准、失败门控、分阶段与逐日摘要 |
| [four_combinations_snapshot.json](four_combinations_snapshot.json) | §5.4、§7.4、开发附表 | 同七日 | 共34策略、714策略日；原24项继承，新增ME/集成，不把继承结果重复当作新回放 |
| [frozen_ablation_snapshot.json](frozen_ablation_snapshot.json) | §5.5、§7.5 | 10-22/23/24/27/28/29/30 | 冻结旧权重/门控，27策略、567策略日；所有控制/来源/双ARS完整保留，无未触碰测试声明 |

三年龄均为500/1000/2000ms，不选最赚钱年龄。阻断为null；现金0是实际
无交易；正式累计净利只用于已平仓日。以上策略日是状态条目，不是独立样本。
原504条已包含于四组合714条，不能相加声称增加独立验证。

## 同源图表

- [逐日热图](figures/daily_net_usd.png)：三年龄、七日、全部27项；灰格是阻断/未定义。
- [独立日净利累计图](figures/cumulative_daily_net_usd.png)：预声明UCB赋权与必要参照；双ARS完整结果在正文和热图，不因表现删项。

图源为冻结复核同一JSON，金额为USD。账户每日重启，累计只是日净利相加，
不是复利账户或实盘净值。归档采用`legend.loc='center left'`避免图例遮挡。
原CLI默认图仍在原本地产物路径；不同绘图环境不保证PNG字节相同。

## 无需行情的最短核对

从仓库根目录使用Python 3.12或兼容标准库环境运行：

```bash
python scripts/verify_final_delivery.py
# 额外核对历史Git对象；浅克隆缺对象时会明确失败。
python scripts/verify_final_delivery.py --source-commits
```

第一条先固定三批实验ID/路径、两图路径及七份必要文档，拒绝删掉检查项后
少做检查；文档可新增但不能重复。随后检查三份快照/计划身份、逐日金额与汇总、
旧24项逐日共有结果/动态审计及汇总继承、旧学习权重、
当前36项冻结数值文件、两图哈希、34×3和27×3主策略收益表及冻结差额/反馈
表、本地文档链接；不下载数据、不运行模型、不写文件。第二条逐文件读取
三批实际数值提交的Git对象，缺对象时须先获取索引中的`source_commit`。
哈希匹配说明与已审阅归档一致，不说明论文实验等价或逐笔执行已经重审。
CI使用`python -S scripts/verify_final_delivery.py`禁用site-packages；
新增标准库测试在临时副本中验证删除清单、重复条目和保持汇总不变的日结果
改写均被拒绝，不修改正式快照。

## 数值源码、环境与外部产物

| 批次 | 实际数值提交 | 完整结果/账本的原本地入口 |
| --- | --- | --- |
| 原OE | `115bc8d6b3e15899dbb8a75e6c99783a2baed08e` | `/tmp/data-mining-multiweek-dynamic-oe-resumable/evaluation/result.json`及对应检查点 |
| 四组合 | `acc7e856f763cf0fa9cf887ab58a52976a2875d9` | `/tmp/fmato-final-four/evaluation/result.json`及对应检查点 |
| 冻结复核 | `c406441c504cf8612822ed2b0bbb0242b63f2b46` | `/tmp/fmato-final-ablation/evaluation/result.json`、`checkpoints` |

三份JSON内保留原结果SHA、计划、实际依赖、数据/模型/日历绑定与审计。
源码提交是产生收益的版本，合并/报告提交不同不意味着数值代码被改写。
[环境快照](../../environment-snapshot.txt)记录实际版本；研究种子42，计算
线程1。归档读取只需标准库，完整重跑需要数值依赖。

上述`/tmp`是当次运行者的原路径，不是克隆仓库即可获得的文件，也不承诺
永久保存。原行情在忽略的`data/`与项目数据集中；prepared数据、周模型库、
完整日/期间/历史窗口账本没有打包到这些快照。正式产物未被本次覆盖，
需要完整审计或恢复时应向持有者取得与记录SHA匹配的原文件，保留独立副本。
只有紧凑快照时可核对本报告，无法重新证明每一笔成交因果性。

## 有数据时怎样复核

1. 按[仓库README](../../README.md)安装研究环境，检查原数据、周库和旧结果身份。
2. 原OE步骤见[多周动态验证](../../docs/multiweek_dynamic_oe_validation_2026-10-03.md)；ME增量见[四组合验证](../../docs/four_combinations_validation_2026-10-04.md)。
3. 新后段按[冻结复核准备记录](../../docs/frozen_ablation_validation_2026-10-05.md)显式准备七尺度，再执行README的freeze/run；不要用默认三尺度目录。
4. 路径或环境不同必须保存新身份；另选不存在的输出目录。恢复已有检查点也保留原产物，不冒称新回放或相同PNG字节。

完整回放可能耗时很长；无数据的交付核对、合成/小窗口验证与实际历史
重放是不同验证层次。最终交付只证明所列实现与证据一致，部分复现边界见
[报告§2.4/§6/§8](../../final_replication_report.md)和[方法对照](../../docs/paper_alignment.md)。
