# 收尾4/4：最终报告与交付核对

## 范围与原文依据

2026-10-05，用户授权合并PR #22后，从最新main `1786d33`创建独立定稿
分支。重新通读`1679894.pdf`全部12页，重点核对物理第3–5页§3.1–§3.4、
Eq.(1)–(6)、Algorithms 1–3，第5–8页§4.1–§4.5、Tables 3–11，以及
第9–10页§4.6–§5。没有新增正式研究训练、奖励拟合、交易回放、日期扩展
或参数选择；既有入口小窗口只作工程联动，不产生报告新收益。

原文把模型选择作为动作，按特征/历史时期构建周模型；Eq.(2)为原始中间
价差，Eq.(3)权重和为1，Eq.(5)为期间订单均值。项目已有工程实现与负
结果，此轮缩小的是报告可解释性和交付可核对性差异，未修改方法定义。
原三个月/末月测试、原市场实盘、完整生产策略与ME极端筛选仍未复现。

## 定稿改动

- 主报告正式标注部分复现定稿；统一摘要、两批日期/训练可见性、结果解释与结论，保持所有既有数值表。
- 明确原24项已包含于四组合34项，不重复计数；冻结27项单独列出。三年龄、两专家来源、双ARS、失败和负结果保留。
- 补齐原奖励基线、最佳单模型/准确率赋权集成、预测期望/准确率和行情接收至发单耗时的未覆盖边界，不拿不同口径替代原文指标。
- 新[交付索引](../results/final_report/README.md)、[机器身份索引](../results/final_report/evidence_index.json)与标准库只读核对入口，说明无行情可核对什么、完整重跑缺哪些外部文件。
- GitHub CI在数值依赖安装前运行同一只读交付核对；浅克隆不要求历史Git对象。
- 三份归档快照、两张图、36项冻结数值文件不改字节；旧时期验证文档按其日期解读，最终报告为当前结论入口。

## 验证命令与结果

从仓库根目录，在项目虚拟环境运行：

```bash
# -S禁用site-packages，验证无数值依赖也能核对归档。
python -S scripts/verify_final_delivery.py
python scripts/verify_final_delivery.py --source-commits
python run_experiments.py --rows 3000 --windows 1 --output /tmp/fmato-final-delivery-smoke.json
python generate_report.py --input /tmp/fmato-final-delivery-smoke.json --output /tmp/fmato-final-delivery-smoke-report.md
python generate_dashboard.py --input /tmp/fmato-final-delivery-smoke.json --output /tmp/fmato-final-delivery-smoke-dashboard.html
git diff --check
```

| 检查 | 实际结果 |
| --- | --- |
| 标准库只读核对 | 通过；三批24/34/27策略，504/714/567状态，全部三年龄、日金额/反馈与汇总一致 |
| 历史源码对象 | 三批原数值提交逐文件通过；当前36项冻结数值文件哈希一致 |
| 归档与原main比较 | 三份快照及两张PNG逐字节不变；完整结果身份在机器索引保留 |
| 报告表格 | 原§5全部结果表、开发附表及单位/成本表与合并基线逐行不变；只读入口另核对34×3、27×3及冻结差额/反馈 |
| 临时副本拒绝验证 | 七项均退出1：金额改写、删除策略、插入假策略、快照字节损坏、图像改变、数值源码变化、缺失引用；恢复副本后通过 |
| 历史对象缺失 | 临时无Git副本的可选核对明确失败；默认无行情核对通过，没有误报源码已核验 |
| 小窗口联动 | 3000行/单窗口实验、Markdown报告、HTML看板通过；全部另存上述/tmp路径 |
| 文档/命令/差异 | 本地链接、代码片段语法、diff空白检查通过；没有跟踪data/或新行情产物 |

日志保留于`/tmp/fmato-final-delivery-verify.log`、`-stdlib.log`、`-negative.log`、
`-smoke.log`。拒绝验证只修改临时交付副本，未损坏仓库正式归档。
本轮不在本地重复全量模型训练/历史回放或234项研究单测：36项数值文件
未改，PR #22最终[CI全部通过](https://github.com/Michelia-L/data-mining-hft/actions/runs/37259092096)，
本轮按交付改动执行上述检查。现有GitHub工作流仍会在最终PR运行全量单测
与小窗口联动，并新增标准库交付核对；该PR的实际CI状态记在PR正文。

## 交付限制

只读核对验证文件身份、日金额/汇总、旧权重继承、表格和链接，不替代
原消息、完整日账本或逐笔因果审计。外部`/tmp`记录是当次运行路径而非
可永久下载的数据包；原产物保持原路径，需持有者保存独立副本后方能恢复。
不同依赖/字体环境可有不同PNG字节；索引只固定已归档图，不重新交易。

本轮结束四个PR收尾范围。没有未触碰样本外或稳定盈利结论；不把程序
检查通过当作完整论文复现。此最终PR仍待组员审阅合并。

## PR #23审阅修复（2026-10-05）

继续更新未合并的`docs/final-replication-delivery`分支，修复三条交付校验
评论。在本会话已通读全文的基础上，重新核对PDF物理第3–5页§3、Eq.(1)–(6)、
Algorithms 1–3及第7–8页§4.2–§4.5、Tables 4–11。论文的模型动作、
价格差奖励、订单平均评价和成本/收益指标不变；本修复只约束证据检查范围
与旧结果继承，未重新拟合奖励、交易或改变任何正式研究数值。

- 独立固定三批experiment IDs及对应快照路径、两个figure路径、七份必要交付文档。清单缺项、重复或实验/图像路径替换均失败；文档可追加但不能删除原必要项。
- 旧24项除阶段`statistics`外，按三年龄、阶段、session和策略顺序逐日核对全部共有结果字段，覆盖金额、成交、奖励及完整动态审计中的选择/反馈计数与版本。日级预测审计和OE校准SHA也须一致，不能用跨日抵消保持总额来伪装继承。
- 已核实投影差异显式排除：旧OE独有`terminal_position`、`holding_ms_mean`、`pnl_aggregation_defined`；四组合独有`blocking_reasons`、`scored_history_windows`。不动态取字段交集，共有字段被删除仍会失败。
- CI交付命令改为`python -S scripts/verify_final_delivery.py`，防止预装第三方包掩盖外部依赖；新增[标准库回归测试](../tests/test_final_delivery.py)，自动纳入原全量unittest发现流程。

```bash
python -S -m unittest discover -s tests -p test_final_delivery.py -v
python -S scripts/verify_final_delivery.py --source-commits
git diff --check
```

| 检查 | 实际结果 |
| --- | --- |
| 修复前回归复现 | 原归档正例通过；新测试出现28个失败断言，暴露清单完整性和逐日继承覆盖缺口 |
| 修复后标准库回归 | 7项测试通过；包含28个负例与一个完整归档只读正例 |
| 清单负例 | 逐项删除三批实验/两图/七文档、清空集合、重复条目、替换实验ID/路径均被拒绝 |
| 逐日继承负例 | 两日盈亏或成交数互相加减、选择次数迁移、版本改写、共有字段/审计删除均被拒绝；负例主动刷新临时快照SHA，排除旧哈希先行失败的假验证 |
| 正式交付与源码 | `-S`核对通过；三份快照、两图、当前36项数值文件及三批历史Git对象身份一致 |
| 改动范围 | 归档、图、正式报告及数值文件未改；方法说明/交付索引说明同步，diff空白检查通过 |

回归日志为`/tmp/fmato-pr23-before-fix.log`及`/tmp/fmato-pr23-review-tests.log`。
所有改写只在测试临时目录中进行，不修改索引或正式证据。修复阶段本地
仅运行相关7项标准库测试与交付核对；数值实现未改，未重复234项研究
单测、小窗口训练或正式多周回放。更新PR后由现有CI运行全部241项测试
及实验/报告/看板联动，其实际状态另记PR正文。
