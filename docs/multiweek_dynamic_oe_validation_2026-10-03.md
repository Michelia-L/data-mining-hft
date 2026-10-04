# 真实多周 OE 动态对照验证（2026-10-03）

基线为 PR #18 合并后的 main `696a56f7b29cb879c6cb68992df9316f77427b37`。
本记录于2026-10-03开始，真实回放和完整结果核对于2026-10-04完成。
本轮研究问题是：库内工程专家已通过校准门控后，冻结奖励能否在真实后续
周版本中用于UCB/双ARS，以及实际收到多少可用反馈。盈利不是通过条件。

## 论文依据与实现对应

本会话已通读12页原文，本轮编码前重读物理第3–5页 §3.1–§3.4、
Eq.(1)–(6)、Algorithms 1–3，以及第7页 §4.2–§4.5。

- Eq.(1)动作是模型；Algorithm 1没有公开专家候选范围与生产参数优化器。
  全候选/库内净利专家都作为工程假设并列保留，最佳专家无成熟OE时不换人。
- Eq.(2)–(4)使用七尺度原始中间价差及和为1的权重。完整有限策略最大间隔
  替代原文策略搜索，最优面最小L1仅为工程消歧，不声称权重唯一或有效。
- Eq.(5)在线反馈是完整期间全部订单均值；校准仍使用成熟订单子集，两者
  统计总体不同。在线期间任何订单失效/未成熟时不以子集均值训练选择器。
- Algorithm 2采用原5分钟期间与C=1价格点；Algorithm 3保留严格最近与
  平移最长前瞻的两种30分钟窗口。论文没有明确一小时以上奖励的在线成熟
  细节，两个口径及冷启动轮换仍是工程假设，不宣称等价复刻原系统。

新增入口只编排既有求解和执行模块，没有改动原执行器、选择器、专家规则
默认值或模型库绑定。旧入口及旧产物完整保留。

## 数据、冻结和策略范围

复用PR #17准备的18个UTC源分区、157,057,286条原始消息和14个完整声明
session；prepared缓存和真实三版周库均通过数据/源码/参数身份检查。没有
重新下载或改变原始行情，没有加入前缀或degraded输入。

| 阶段 | 2025年trade date | 本轮用途 |
| --- | --- | --- |
| train | 10-02、10-03 | 启动周库；重算训练样本合法性 |
| calibration | 10-06至10-10 | 重新回放16个静态参照，冻结两种来源的奖励 |
| validation | 10-13至10-17 | 24项动态/控制策略状态与因果审计 |
| test | 10-20、10-21 | 24项冻结策略状态；仍为开发测试 |

上述日期均已有开发用途，不是新未触碰测试集。旧session配置中的后段
“预测检查”说明对应就绪审计；本轮新计划的 `stage_roles` 明确扩为动态
收益评价，未改写旧配置或旧结果。模型10-06启动，10-13和10-20更新；
后两周才计作动态评价覆盖，不能用校准周或离线未来参数凑数。

网格500ms，最大报价年龄500/1000/2000ms全部保留。预测前瞻5000ms，
延迟500ms，持仓复核15000ms，门槛0.000015。七奖励尺度为
`[5000,15000,45000,135000,405000,1215000,3645000]` ms。
一手ES，乘数50、最小价格跳动0.25、单订单费用1.25美元、不利滑点0.5跳
及实测点差，全部沿用既有主动成交假设；奖励不扣成本，美元账本扣成本。

两种来源使用同一完整成熟候选矩阵。三种权约束均保留诊断，只有预声明
`sum_only`进入原动作门控；不根据后段收益改选权约束、来源、年龄或权重。
每个年龄只在全部五日校准结束后学习一次，随后奖励冻结不变。
部分权重为0也不缩短成熟要求，仍检查完整七尺度向量。这一保守工程约定
与此前校准/在线门控一致，反馈覆盖因此不能只按非零权重的最长尺度解释。

| 策略组 | 数量 | 作用 |
| --- | --- | --- |
| 原等权/静态控制 | 18 | UCB、同期间轮换、现金、原固定Ridge、12个固定库策略、双ARS |
| `Learned-OE-*` | 3 | 原全候选专家的UCB和双ARS，门控失败保留阻断 |
| `LibraryExpert-OE-*` | 3 | 库内工程专家的UCB和双ARS，通过原门控才执行 |

“固定库策略”固定的是候选ID，其数值参数仍按同一周计划更新；原固定Ridge
参照保持启动训练参数。动态策略要求当下全部库模型有预测，静态候选按自身
可用性交易，延续旧对照口径，不能把交易次数差异全部归因于选择器。

账户、选择器与奖励队列每天重启，数值模型按周更新。每日已结算净利只作
加总，不是跨日连续资金或复利；无尾部平仓行情时保留未平仓与未定义汇总。

## 验证与复现

```bash
.venv/bin/python -m unittest discover -s tests -p test_multiweek_dynamic_oe.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_multiweek_dynamic_oe.py freeze \
  --config config/esz5_multiweek_dynamic_oe.json \
  --library /tmp/data-mining-multiweek-readiness/library/library.json \
  --output-dir /tmp/data-mining-multiweek-dynamic-oe-resumable/frozen
.venv/bin/python run_multiweek_dynamic_oe.py run \
  --plan /tmp/data-mining-multiweek-dynamic-oe-resumable/frozen/plan.json \
  --checkpoint-dir /tmp/data-mining-multiweek-dynamic-oe-resumable/checkpoints \
  --output-dir /tmp/data-mining-multiweek-dynamic-oe-resumable/evaluation
.venv/bin/python run_experiments.py --rows 3000 --windows 1 --output /tmp/multiweek-dynamic-final-smoke.json
.venv/bin/python generate_report.py --input /tmp/multiweek-dynamic-final-smoke.json --output /tmp/multiweek-dynamic-final-smoke-report.md
.venv/bin/python generate_dashboard.py --input /tmp/multiweek-dynamic-final-smoke.json --output /tmp/multiweek-dynamic-final-smoke-dashboard.html
```

新增10项性质测试通过（45.395秒），全量219项通过（128.521秒）。覆盖两组
门控/负专家/无成熟最佳专家、未来扰动、全期间订单守恒与提前反馈拒绝、
ARS冻结权重和可见边界、阻断None及不允许后段改门控、两评价周CLI、
原18控制逐字段一致、校准只拟合一次、配置/源码/父计划/模型身份与禁止覆盖，
以及日结果恢复逐字段一致、错误计划/明细/环境与损坏/未完整发布检查点拒绝。
旧3000行单窗口实验、报告和看板联动均通过；只说明旧入口兼容，旧生成器
不能直接读取新增schema。新入口自带同源JSON/中文报告生成。

运行时还独立重算训练样本时刻与数量，逐行预测/实际选择检查最新可见周版，
核对成交与期间桶/成熟订单/选择器访问反馈计数，检查完整期间均值及最早
成熟时刻、ARS可见观察上限与完整订单均值的权重内积。成熟期间桶数量与
选择次数不是同一总体，不将二者的比值解释为反馈概率。

真实产物进一步核对通过：三个年龄的校准逐日账本与PR #17就绪归档逐字段
一致，两组专家/三约束拟合/门控与PR #18归档逐字段一致；后段逐日预测审计
与原就绪检查一致。全24项策略顺序和冻结权重齐全，阻断为null，执行结果
均满足毛利减摩擦等于净利。禁止调用交易回放模块后恢复全部36个检查点，
最终结果与首次发布JSON逐字段相同，证明真实恢复跳过重交易但保留审计。

## 长运行中断与恢复

首次无检查点回放的工具会话结束，最终结果未发布，旧日志只记录到1000ms
的第二个评价日；不能将日志当作完整实验结果。旧冻结计划保留在
`/tmp/data-mining-multiweek-dynamic-oe/frozen`，日志为
`/tmp/multiweek-dynamic-real.log`。本轮没有从这些不完整日志恢复收益数值。

增加完整日检查点后另冻新计划，使用相同日期、模型、参数与成本重新回放。
检查点保存完整数据而非摘要，每份绑定计划（含源码/数据/模型）、明细开关、
实际依赖版本及内容哈希，原子发布且不覆盖旧单位。恢复先重新验证输入身份，
校准统计和权重由完整校准账本重算，评价日重新核对预测及动态审计；不会因
缓存命中跳过原计划的年龄、策略或日期。最终结果与有无缓存路径无关。

本次用户会话中断时，后台进程仍在运行；先确认进程状态而未重复启动。
最终36个检查点齐全，包含三个年龄各五个校准日和七个评价日。此前无
检查点的计划/日志保留，新结果没有覆盖PR #17/#18归档或原始行情。

## 真实运行结果

以下为开发数据结果。利润按七个独立日账户加总，非连续资金复利；完整逐日
账本与失败状态保存在JSON中，报告由同一JSON汇总全部策略。

| 年龄 ms | 专家来源 | 专家 | 校准净利 USD | 专家成熟订单 | 仅等式拟合 | 门控 |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | all_candidates | Ridge-threshold-x4 | 22.50 | 0 | blocked_expert_reward_unobserved | False |
| 500 | online_library | Library-Ridge-price-h1 | -8842.50 | 0 | blocked_expert_reward_unobserved | False |
| 1000 | all_candidates | cash | 0.00 | 0 | fitted | False |
| 1000 | online_library | Library-Ridge-price-h1 | -10487.50 | 4 | fitted | True |
| 2000 | all_candidates | cash | 0.00 | 0 | fitted | False |
| 2000 | online_library | Library-Ridge-price_volume-h2 | -2212.50 | 22 | fitted | True |

### 全部24项策略的七日净利（USD）

| 策略 | 500ms | 1000ms | 2000ms |
| --- | --- | --- | --- |
| Period-OE-UCB | -110215.00 | -105730.00 | -139520.00 |
| Period-OE-RoundRobin | -108087.50 | -101490.00 | -145780.00 |
| cash | 0.00 | 0.00 | 0.00 |
| Ridge-threshold-x1 | -222817.50 | -320502.50 | -311957.50 |
| Fixed-Ridge-price-h1 | -66142.50 | -74235.00 | -79505.00 |
| Fixed-DecisionTree-price-h1 | -11980.00 | -7602.50 | -10542.50 |
| Fixed-Ridge-displayed_volume-h1 | -158225.00 | -172897.50 | -132992.50 |
| Fixed-DecisionTree-displayed_volume-h1 | -83560.00 | -89600.00 | -91420.00 |
| Fixed-Ridge-price_volume-h1 | -117677.50 | -71610.00 | -27310.00 |
| Fixed-DecisionTree-price_volume-h1 | -141230.00 | -69210.00 | -293232.50 |
| Fixed-Ridge-price-h2 | -67617.50 | -69005.00 | -72100.00 |
| Fixed-DecisionTree-price-h2 | -39537.50 | -14607.50 | -103665.00 |
| Fixed-Ridge-displayed_volume-h2 | -240675.00 | -266662.50 | -145417.50 |
| Fixed-DecisionTree-displayed_volume-h2 | -45075.00 | -108405.00 | -257762.50 |
| Fixed-Ridge-price_volume-h2 | -146247.50 | -129450.00 | -25385.00 |
| Fixed-DecisionTree-price_volume-h2 | -161935.00 | -99220.00 | -464147.50 |
| History-OE-ARS-recent | -101447.50 | -93852.50 | -159700.00 |
| History-OE-ARS-matured | -97172.50 | -101737.50 | -153025.00 |
| Learned-OE-UCB | 未执行/未定义 | 未执行/未定义 | 未执行/未定义 |
| Learned-OE-ARS-recent | 未执行/未定义 | 未执行/未定义 | 未执行/未定义 |
| Learned-OE-ARS-matured | 未执行/未定义 | 未执行/未定义 | 未执行/未定义 |
| LibraryExpert-OE-UCB | 未执行/未定义 | -99247.50 | -142930.00 |
| LibraryExpert-OE-ARS-recent | 未执行/未定义 | -93852.50 | -159700.00 |
| LibraryExpert-OE-ARS-matured | 未执行/未定义 | -105845.00 | -142912.50 |

### 库内学习组的分阶段账本与反馈

| 年龄 ms | 阶段 | 策略 | 毛利 USD | 摩擦 USD | 净利 USD | 成交 | 成熟期间桶 | 选择次数 | 可评分/全部历史窗 | 冷启动期间 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | validation | LibraryExpert-OE-UCB | -4337.50 | 72812.50 | -77150.00 | 4970 | 43 | 1329 | 0/0 | 0 |
| 1000 | validation | LibraryExpert-OE-ARS-recent | -6037.50 | 74520.00 | -80557.50 | 5086 | 53 | 1329 | 0/14619 | 1077 |
| 1000 | validation | LibraryExpert-OE-ARS-matured | -5556.25 | 75228.75 | -80785.00 | 5138 | 48 | 1329 | 394/14619 | 1086 |
| 1000 | test | LibraryExpert-OE-UCB | -268.75 | 21828.75 | -22097.50 | 1568 | 15 | 506 | 0/0 | 0 |
| 1000 | test | LibraryExpert-OE-ARS-recent | 781.25 | 14076.25 | -13295.00 | 1006 | 14 | 506 | 0/5566 | 399 |
| 1000 | test | LibraryExpert-OE-ARS-matured | 743.75 | 25803.75 | -25060.00 | 1858 | 11 | 506 | 59/5566 | 406 |
| 2000 | validation | LibraryExpert-OE-UCB | 1775.00 | 131015.00 | -129240.00 | 8952 | 58 | 1365 | 0/0 | 0 |
| 2000 | validation | LibraryExpert-OE-ARS-recent | -1800.00 | 147820.00 | -149620.00 | 10186 | 72 | 1365 | 0/15015 | 968 |
| 2000 | validation | LibraryExpert-OE-ARS-matured | -4393.75 | 128043.75 | -132437.50 | 8750 | 59 | 1365 | 443/15015 | 995 |
| 2000 | test | LibraryExpert-OE-UCB | -225.00 | 13465.00 | -13690.00 | 962 | 21 | 545 | 0/0 | 0 |
| 2000 | test | LibraryExpert-OE-ARS-recent | 350.00 | 10430.00 | -10080.00 | 744 | 24 | 545 | 0/5995 | 447 |
| 2000 | test | LibraryExpert-OE-ARS-matured | 312.50 | 10787.50 | -10475.00 | 770 | 23 | 545 | 135/5995 | 451 |

### 实际跨周与独立审计

全部评价包括3个年龄×7个session×24项策略，即504项日状态：420项执行、
84项阻断。1000/2000ms的库内UCB与双ARS均实际选择了10-13与10-20的
模型版本；全候选各年龄及500ms库内组均阻断，没有动态版本覆盖。

| 年龄 ms | 全动态对照选择次数 | 完整成熟期间桶 | 全部历史窗 | 可评分历史窗 |
| --- | --- | --- | --- | --- |
| 500 | 5264 | 205 | 28952 | 224 |
| 1000 | 12845 | 457 | 80740 | 907 |
| 2000 | 13370 | 600 | 84040 | 1161 |

以上合计包含等权UCB、轮换、双ARS及通过门控的学习组，不包含静态参照；
不同策略历史回放独立，窗口数不能当作独立市场样本数。库内学习组单独
可评分的平移历史窗为1000ms的453/20185、2000ms的578/21010。
严格最近窗均为0；其逐日实际选择、成交数与净利均与等权最近ARS相同，
不是删去失败策略后的结果。

### 冻结权重与负结果解释

500ms两个来源的最佳专家都没有成熟OE，且完整库缺一个候选的成熟评价；
1000/2000ms全候选专家为现金，现金仍在动作空间外，不放宽门控。只有
库内组的两个年龄通过，专家均为负净利；1000ms仅4/840个订单成熟，
2000ms为22/120，门控通过不能解释为专家质量或统计可靠性已获验证。

按七尺度顺序，1000ms权重为`[0,0,0,0,0,1,0]`；2000ms为
`[0.4594281745572404,0.279546243778215,0,-0.09248755643014231,0.3535131380946869,0,0]`。
JSON保留求解器的有符号零；上述写法仅简化零的展示。含负权的仅等式约束
与Eq.(3)权重和为1一致，但数值由工程专家与有限候选优化决定，不是论文参数。

所有执行的非现金策略在各年龄的七日累计净利均为负。库内UCB相对等权
UCB在1000ms少亏6482.50美元、在2000ms多亏3410.00美元；平移ARS的
差异方向也不一致，严格最近ARS没有区别。不能据此支持学习奖励稳定改善
收益、优于固定模型或复现论文收益的结论。

库内学习组各阶段摩擦成本均大于毛利的绝对值，主动成交成本是本协议净
亏损的重要组成；部分阶段毛利为正仍不抵成本。这是沿用原成本条件后的
观察，不以测试结果重选年龄、门槛或成本。报价缺失、短训练历史、预测信号
和执行假设的各自因果贡献仍需预声明对照才能拆分。

在七尺度全部成熟约定下，30分钟严格最近窗不能提供满足最长3645秒前瞻的
完整订单反馈；平移窗有少量可评分历史，但多数选择仍处于冷启动。此结果
揭示了长期奖励与短历史窗衔接的困难，不能把工程平移窗称为论文已公开方案。

## 产物身份与实际环境

新产物根目录为`/tmp/data-mining-multiweek-dynamic-oe-resumable`。完整JSON
包含逐日账本、拟合诊断、门控、训练/预测/选择审计及实际周版本哈希；报告
来自同一JSON。362,051,618字节的开发JSON和完整检查点留在本地，不提交
大文件。仓库本记录保留全部24项七日净利和库内组分阶段反馈证据。

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 冻结计划 | `frozen/plan.json` | `ac246f46b0ceef67f2eda90698cb20143a5303fee7e89ff4eadc89c21d528823` |
| 完整结果 | `evaluation/result.json` | `d9aa5e4b416d4d9d28cc05ec22bdeed6a608c80e50f6a6119931c82c57b7e5ae` |
| 同源报告 | `evaluation/report.md` | `0d3e4c758819cfc2231953c5c13bac53bf63b22316ac6c43b231640ceb26e4f8` |

计划内容指纹为`bb2145612fece2e08249399207862f727d12fd81cda202589567bd07274b0639`，
与文件字节SHA不同；源码、配置、数据和库身份均由计划及父计划绑定。
实际环境为Python 3.12.14、NumPy 2.5.3、pandas 3.0.6、PyArrow 25.0.1、
scikit-learn 1.9.1、SciPy 1.18.1、threadpoolctl 3.7.0。

复用库文件SHA为`d18d6d01dbb51eb1dd8676651cd31d34b900aff08d82195e896857a0a954e1ab`；
核对的旧就绪结果SHA为`1c6e94e8c6539d3b8fa9716e933734fbecde663e01ed7414929125da333b1e4c`，
旧专家来源结果SHA为`de6fd00769d57d0730f77fb0f528da134a143af14f4a6cdb69d669b3ff859b39`。
完整测试、真实运行、旧归档核对及恢复核对日志分别保留为
`/tmp/multiweek-dynamic-final-tests.log`、`/tmp/multiweek-dynamic-resumable-real.log`、
`/tmp/multiweek-dynamic-review.log`与`/tmp/multiweek-dynamic-restore-check.log`。

## 剩余边界

本轮只回答OE学习奖励的真实多周执行和反馈覆盖问题。时间ME学习、
完整四变体/集成/奖励消融及正式长期冻结测试仍未完成。CME与中国商品
期货的差异、工程特征和1/2session训练历史、有限候选优化、稀疏成熟反馈
和主动成交成本假设仍限制结论；不能从测试通过或单次盈利推断原论文复现成功。
