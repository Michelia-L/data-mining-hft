# 项目术语词典

给准备10分钟展示、初次阅读项目的组员使用。已检查 README、课程报告、证据目录说明，以及21个 Python 文件的注释和文档字符串，整理为 **90个词条、7组**；同类叫法和代码字段合并为一个词条。

先读第一组的8个核心词，其他内容遇到时用 **Ctrl+F 搜索缩写、英文或中文**。出处中的页码均为 [论文](1679894.pdf)的 **PDF物理页码**；标注的是该文使用或解释术语的位置，不表示这些通用术语由论文作者发明。“项目”表示我们的具体实现或设定。

## 1. 先懂这8个核心词

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **FMATO** | 论文给“自动调整价格预测模型”这套方法起的名字。正文未明确展开字母全称，沿用原名即可。 | [第4页 Fig.2；第5页 §4](1679894.pdf#page=4)。本项目是核心机制的部分复现。 |
| **HFT** — High-Frequency Trading，高频交易 | 在很短的时间尺度上作交易决策。本项目用半秒网格回放行情。 | [第1页标题；第3页 §3.1](1679894.pdf#page=3)。半秒回放不等于已经建成实盘高频系统。 |
| **RL** — Reinforcement Learning，强化学习 | 试过一些选择，拿到反馈，再调整之后选什么。在这里，“选择”主要是选预测模型。 | [第2页 Introduction；第3页 §3、Eq.(1)](1679894.pdf#page=3)。 |
| **IRL** — Inverse Reinforcement Learning，逆强化学习 | 先有历史参照策略，再倒推“怎样给表现打分”。项目用历史校准近似学习各时间尺度的评分权重，之后冻结。 | [第2页 §2.3；第4页 §3.3、Algorithm 1](1679894.pdf#page=4)。有限候选求解是项目近似。 |
| **UCB / UCBS** — Upper Confidence Bound，上置信界；UCB selection | 过去得分高的模型多用，了解不够的模型也试一试；打分包含“历史平均分＋探索加分”。论文方案名中的 UCBS 对应 UCB 选模型。 | [第2页 §2.2；第5页 §3.4.1、Eq.(6)、Algorithm 2](1679894.pdf#page=5)。加分不保证交易盈利。 |
| **ME** — 论文称 Model Prediction Expectation，模型预测期望 | 从模型发出预测的时刻往后看，行情后来怎么走，用它评价预测。评价起点是预测时刻。 | [第5页 §3.3.1](1679894.pdf#page=5)。历史扩展结果中有 ME；当前课程主线使用 OE。 |
| **OE** — Order Traded Expectation，订单成交期望 | 从实际成交的时刻往后看，中间价后来变了多少，用它评价成交后的走势。它是价格差评分。 | [第4页 Eq.(2)；第5页 §3.3.2](1679894.pdf#page=5)。项目对卖单反转符号，成本另记。 |
| **ARS** — Average Reward Selection，平均奖励选择 | 用历史回测给候选模型打分，再挑平均分最高者；论文还回测没有被选中的模型。 | [第5页 §3.4.2、Algorithm 3](1679894.pdf#page=5)。历史工程变体的结果保留，当前课程回放主线是 UCB。 |

例如，一笔买单成交时的中间价是100，5秒后是101，则这一尺度的 OE 是 **＋1价格点**。它还没有说明后来何时平仓、交了多少成本或净赚多少美元。名字 `FMATO-OE-UCBS` 可以拆成“论文方法＋成交后评价＋UCB选模型”。

## 2. 模型与评分：谁预测，谁选择，怎样打分

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **轻模型 / 模型库** — light model / model library | 把多个计算较简单的预测模型放在一个候选名单里，供系统选择。 | [第3–4页 §3.2、Table 2](1679894.pdf#page=4)。项目是两类模型×三类特征×两种历史长度，共12个候选。 |
| **动作 / 臂 / MAB** — action / arm / Multi-Armed Bandit，多臂老虎机 | 想成有多个选项，每次选一个，之后观察它的表现；“臂”就是一个选项。本项目的动作/臂是一个预测模型。 | [第2页 §2.2；第3页 Eq.(1)](1679894.pdf#page=3)；“未访臂”见 [选择器注释](src/period_ucb.py#L37)。 |
| **智能体 / 策略** — agent / policy | 智能体是作决策的系统；策略是“遇到什么情况怎样做”的规则。模型给出预测，交易策略据此决定交易意图。 | [第3页 §3、Fig.1](1679894.pdf#page=3)。 |
| **MDP** — Markov Decision Process，马尔可夫决策过程 | 用“当前情况、可选动作、下一步情况和评分”描述连续决策问题的一种框架。 | [第2页 §2.3](1679894.pdf#page=2)中的 IRL 背景用语；与数据集代码 `MDP3` 分别解释。 |
| **特征 / 标签** — feature / label | 特征是模型作答时的输入，例如当时的价格变化；标签是训练时的参考答案，例如随后5秒的相对收益。历史标签可用于训练，当时决策只能看已知输入。 | 特征分类见 [第4页 Table 2](1679894.pdf#page=4)；项目见 [特征与标签注释](src/snapshot_dataset.py#L207)。 |
| **Ridge** — 岭回归 | 在线性回归上增加约束系数过大的惩罚，让模型尽量少受不稳定系数影响。项目用它预测未来相对收益。 | [Ridge 官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)；[项目适配器](src/model_library.py)。原文写线性回归，Ridge是项目具体选择。 |
| **浅树** — shallow decision tree | 用少量“如果……那么……”判断分支作预测；“浅”表示分支层数较少。 | [第4页 §3.2](1679894.pdf#page=4)使用决策树；具体树深度是 [项目参数](config/course_experiment.json)。 |
| **Ridge-price-h1 / 固定库候选** | 名字拆开看：Ridge模型、价格特征、最近1个完整交易日的训练历史；`h1`表示一个历史session。固定候选身份，参数仍可按周更新。 | [项目模型库](src/weekly_library.py)。`Fixed-Ridge-price-h1`不是事后挑出的收益冠军，也不表示参数永久不变。 |
| **集成 / Mean-Ensemble** — ensemble | 多个模型各自作预测，再把这些预测合起来；本项目取12个预测的算术平均。 | [第7页 §4.5](1679894.pdf#page=7)；[项目均值集成](src/model_selector.py#L57)。 |
| **门槛 / threshold** | 预测变化足够大才考虑交易；很小的预测变化可能不足以覆盖交易摩擦。 | [项目协议](config/course_experiment.json)与[执行器](src/time_execution.py)；具体门槛是项目设定。 |
| **期望 / 奖励** — expectation / reward；`E(T)`、`R` | 论文中的 `E(T)`是向后观察T时间的中间价差；多个尺度按权重合成奖励评分。“期望”在这里不能读成保证发生的利润。 | [第4页 §3.3、Eq.(2)–(4)](1679894.pdf#page=4)；单位为价格点，净利润另算。 |
| **多尺度 / 权重 / 等权** — multi-scale / weights / equal weights | 同时观察短、中、长时间后的价格变化；权重决定各项怎样合成评分。等权是每项同样重要，本项目七项各为1/7。 | [第4页 Eq.(2)–(4)](1679894.pdf#page=4)；`OE-equal-UCB`用人工等权，`OE-online_library-UCB`用历史校准的学习权重。 |
| **专家 / expert** | 用来学习评分规则的历史参照。项目按校准净利在已平仓库内候选中选最高者，即使这个最高者仍亏损，也叫“专家”。 | [第4页 Algorithm 1](1679894.pdf#page=4)；候选范围和净利筛选是 [项目设定](src/calibration.py#L53)。 |
| **期间平均奖励** — period mean reward | 把同一模型在固定期间内的订单评分相加，再除以订单数。项目等期间结束且全部订单完整可评价后反馈。 | [第5页 Eq.(5)、Algorithm 2](1679894.pdf#page=5)；[项目期间协议](src/period_ucb.py#L94)。历史校准的成熟订单子集与在线完整期间分开处理。 |

## 3. 行情数据：文件和字段说的是什么

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **CME / NYSE** | CME是芝加哥商品交易所，NYSE是纽约证券交易所。不同市场有不同交易时间，代码里的提醒是“ES用自己的期货日历”。 | [项目日历注释](src/snapshot_dataset.py#L51)。 |
| **ES / ESZ5** | ES是E-mini标普500股指期货；Z表示12月，本数据年份中的5表示2025年。ESZ5指该年12月的这一合约。 | [CME合约代码说明](https://www.cmegroup.com/education/courses/introduction-to-futures/understanding-contract-trading-codes)；项目使用单一合约。 |
| **Databento / Hugging Face** | Databento提供原始行情；Hugging Face在本项目用于共享数据下载文件。看到下载平台名时仍要区分行情来源。 | [README的数据集介绍](README.md#数据集)。 |
| **GLBX.MDP3** | Databento给CME Globex的MDP 3.0行情数据集使用的代码，表示我们使用哪路行情。 | [Databento数据集说明](https://databento.com/docs/venues-and-datasets/glbx-mdp3)。 |
| **MBP-10 / L2行情** — Market By Price | 按价格汇总买卖各十档的报价、挂单数量等。“L2行情”是市场深度数据的层级称呼。 | [Databento MBP-10定义](https://databento.com/docs/schemas-and-data-formats/mbp-10)。项目取前五档特征；此处的L2与回归的L2惩罚含义不同。 |
| **盘口 / 深度** — order book / depth | 盘口是市场上等待成交的买卖报价；深度是在这些价格上有多少挂单。挂单量描述待成交数量，与实际已经成交的量有区别。 | [第6页 Figs.3–4](1679894.pdf#page=6)；[项目特征注释](src/weekly_library.py#L25)。 |
| **买一 / 卖一** — best bid / best ask | 买一是最高的买方报价，卖一是最低的卖方报价。主动买入通常要付卖一，主动卖出通常拿买一。 | [第4页 Table 1](1679894.pdf#page=4)；[项目主动成交](src/account.py#L26)。 |
| **中间价 / midprice** | 买一和卖一的平均值，例如100和101的中间价是100.5。它用来评价走势，并不是保证能成交的价格。 | [第4页 Table 1、Eq.(2)](1679894.pdf#page=4)。 |
| **消息 / 完整事件** — message / complete event | 一个交易所事件可能分多条消息才传完；收到事件的一部分时，盘口还在更新。消息条数也包含挂单更新等内容。 | [项目采样注释](src/timed_snapshots.py#L55)；因此原始消息条数不能直接当成交笔数。 |
| **快照 / snapshot** | 在一个时刻给盘口“拍一张照片”。本项目每500ms用当时已经收到的完整行情生成快照。 | [第3页 §3.1](1679894.pdf#page=3)；[项目快照工具](src/timed_snapshots.py)。 |
| **degraded** — 数据质量降级 | 数据提供方标记这天的数据质量有问题；它是一种来源质量状态。当前协议默认排除这些日期。 | [README](README.md#数据集)与[数据目录检查](src/data_catalog.py#L16)。 |
| **DBN / Parquet** | DBN是Databento保存行情的二进制格式，英文为Databento Binary Encoding；Parquet是保存大表的文件格式。项目保留源DBN，并读取转成表格的Parquet。 | [DBN官方定义](https://databento.com/docs/standards-and-conventions/databento-binary-encoding)；[项目输入](src/data_catalog.py)。 |

## 4. 时间：几个容易混淆的“多久”

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **UTC / IANA时区 / ISO日期** | UTC是统一世界时；IANA时区用`America/Chicago`等地区名称表达规则，会处理夏令时；本项目ISO日期写作`YYYY-MM-DD`，即年-月-日。文件日期按UTC，交易日按交易所日历。 | [数据日期检查](src/data_catalog.py#L27)、[交易日历](src/snapshot_dataset.py#L51)。 |
| **session / trade date / segment** | session是一个完整交易日，可从前一晚开始；trade date是它归属的交易日日期；segment用于区分其中的休市暂停。 | [项目日历注释](src/snapshot_dataset.py#L51)。完整session可能跨相邻UTC文件。 |
| **ts_recv / source_ts_recv** | 接收端收到原始行情的时刻；项目用它判断决策时是否已经知道这条消息。快照保留来源的接收时刻为`source_ts_recv`。 | [Databento时间字段](https://databento.com/docs/schemas-and-data-formats/mbp-10)；[项目采样](src/timed_snapshots.py#L55)严格要求接收时刻早于网格。 |
| **ts_event / source_ts_event** | 原始行情中的`ts_event`指撮合引擎收到事件的时刻。项目快照里的同名字段是决策网格时刻，原始事件时刻保存在`source_ts_event`。 | [原始字段定义](https://databento.com/docs/schemas-and-data-formats/mbp-10)与[项目字段说明](src/timed_snapshots.py#L1)。先确认正在看哪种文件。 |
| **ms / us / ns** — 毫秒 / 微秒 / 纳秒 | 1秒＝1000ms；1ms＝1000us；1us＝1000ns。配置多用毫秒，内部时钟多用纳秒，两者须换算。 | [项目时钟注释](src/time_execution.py#L24)。 |
| **tick** | 论文§3.1主要用它指一次定时行情快照；交易中“价格tick”又可指最小报价跳动。代码里的`slippage_ticks`使用后者。 | [第3页 §3.1；第5页 §4](1679894.pdf#page=3)分别有0.5秒和每秒四次快照的描述；项目明确采用500ms网格。 |
| **网格间隔 / interval_ms** | 计划多久检查一次行情；当前每500ms一个网格。两个有输出的相邻行之间若缺了格，实际可能相隔更久。 | [项目采样](src/timed_snapshots.py#L55)。 |
| **报价年龄 / max_age_ms** — quote age | 决策时刻减去来源报价的接收时刻，即“这条报价已经有多旧”。500/1000/2000ms是允许的最大年龄。 | [项目质量条件](src/timed_snapshots.py#L55)；它与生成快照的500ms间隔分别配置。 |
| **前瞻期 / horizon** | 从一个起点往后观察多久。预测前瞻是模型要预测的5秒；奖励前瞻是用来评价的七个尺度，最长3645秒＝60.75分钟。 | [第4页 Eq.(2)–(4)](1679894.pdf#page=4)；[项目毫秒参数](config/course_experiment.json)。 |
| **延迟 / latency_ms** | 作出交易意图后，到允许执行之间要等多久。项目设500ms，并使用到期后实际可用的行情成交。 | [项目执行假设](src/time_execution.py#L53)。 |
| **选择期间 / selection_period_ms** | 多久作一次模型选择并归集订单评价。项目按交易日开盘锚定5分钟期间，期间内锁定模型；反馈还须等待成熟。 | [第5页 Eq.(5)、Algorithm 2](1679894.pdf#page=5)；锚定与锁定是 [项目约定](src/period_ucb.py)。 |
| **持仓复核 / holding_review_ms** | 隔多久重新考虑已有持仓是否继续。项目正常复核须距上次至少15秒，同向信号可以续持；15秒不是最大持有时长。 | [项目账户规则](src/account.py#L63)；15秒是项目设定。 |

## 5. 交易与结果：评分怎样对应到账本

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **PnL / PNL / USD** — Profit and Loss / 美元 | PnL就是盈亏，正数赚、负数亏；USD是金额单位。看到PnL还要看它是毛盈亏还是净盈亏。 | [项目账本](src/account.py)；论文的原实验金额用人民币，见 [第7页 §4.2](1679894.pdf#page=7)。 |
| **毛利 / 摩擦 / 净利** — gross / friction / net | 本项目毛利按中间价变动记；摩擦包括点差、滑点、手续费；净利＝毛利−摩擦。摩擦减少也可能导致少亏。 | [项目账本](src/account.py#L45)。`gross_pnl_usd`、`friction_usd`、`net_pnl_usd`分别对应三项。 |
| **价格点 / 点值 / multiplier** | 价格点是报价变化的单位；点值把它换成钱。项目每张ES每变动1点对应50美元，最小跳动0.25点对应12.50美元。 | [项目合约单位](src/config.py#L7)。因此1价格点奖励不能直接读成1美元净利。 |
| **点差 / 滑点 / 手续费** — spread / slippage / commission | 点差是买一与卖一之间的差；滑点是成交比基准报价更不利的部分；手续费是每次买卖另付的费用。 | [项目成本](src/config.py#L8)：额外0.5价格跳动滑点、单边1.25美元手续费，均为执行假设。 |
| **多头 / 空头 / 空仓；position / quantity** | 多头受益于上涨，空头受益于下跌，空仓没有持仓。`position`的＋1/−1/0记录方向，`quantity`记录合约张数。 | [项目账户注释](src/account.py#L8)；方向与张数分别记录。 |
| **fills / trades** | fill是一次买入或卖出的成交；trade是从开仓到平仓的一笔完整交易。本项目一开一平通常有两次fill。 | [项目账户注释](src/account.py#L8)。成交笔数与完整交易数分别统计。 |
| **turnover** — 成交名义金额 | 把各次成交按“成交价×点值×张数”累加，表示交易金额规模。来回交易可使它很大，利润仍可能为负。 | [项目账户注释](src/account.py#L14)。 |
| **盯市 / 回撤 / 夏普** — mark-to-market / drawdown / Sharpe | 盯市按当前价格给未平仓资产估值；回撤看资金从之前峰值回落多少；夏普衡量收益相对波动的表现。 | 盯市见 [账户注释](src/account.py#L93)，论文提夏普见 [第2页 Introduction](1679894.pdf#page=2)；项目不据七日结果作年化推断。 |
| **独立日账户 / 复利** | 项目每天重置同样的初始资金，再把各日利润相加；复利则会让前一天的资金变化影响下一天。图中的累计金额采用前一种口径。 | [项目汇总说明](src/session_experiment.py#L216)。 |

## 6. 实验与质量：什么时候能相信、什么时候能打分

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **训练 / 校准 / 开发评价 / 冻结复核** | 训练拟合预测模型；校准学习奖励权重；开发评价检查方案；冻结复核固定方案后看后段表现。当前全部实验日期已参与开发。 | [项目四段协议](config/course_experiment.json)；原文训练/测试设置见 [第7页 §4.2](1679894.pdf#page=7)。四段划分是项目设定。 |
| **在线 / 离线** — online / offline | 在线表示随行情与成熟反馈到来调整模型选择；离线表示利用已有历史数据作训练、校准或统计。项目在线选择模型，奖励权重在评价阶段冻结。 | [项目执行器](src/time_execution.py)与[课程报告](final_replication_report.md)。 |
| **样本外 / holdout / 未触碰测试集** | 样本外是用训练以外的样本评价；未触碰还要求没有看其表现来改方案。当前配置虽有`test`字段，这些日期已用于开发，不能再称未触碰。 | [项目协议注释](src/session_experiment.py#L34)与[课程报告](final_replication_report.md)。 |
| **因果性 / 前视 / 信息泄漏** — causal / look-ahead / leakage | 当时的决定只能用当时已知信息。若让未来价格、未来标签或后来学出的参数影响过去的选择，就是提前“偷看答案”。 | [项目执行时序](src/time_execution.py#L87)、[特征说明](src/weekly_library.py#L37)。 |
| **purge** — 切分边界隔离 | 给训练结束边界留出标签观察的缓冲；删掉那些参考答案会用到后段行情的训练样本。 | [项目训练边界](src/snapshot_backtest.py#L157)；隔离长度按最长前瞻设定。 |
| **成熟 / matured；pending；联合完整** | pending表示还在等待；完整成熟表示所需的七个未来价格都已到达且有效。每个尺度各有样本，也可能找不到一笔七项齐全的订单。 | [第4页 Eq.(3)–(4)](1679894.pdf#page=4)；[项目到期评价](src/time_execution.py#L164)。`missing_target`表示到期目标行情缺失。 |
| **缺格 / 预热 / 前填** — gap / warm-up / forward-fill | 缺格是预期时点没有可用快照；预热是等够历史才能算滚动特征；前填是用旧值填后面的空位。项目不跨缺口拼历史，也不随意前填成连续行情。 | [项目连续区间规则](src/snapshot_dataset.py#L207)。 |
| **门控 / blocked / 现金** — gate / cash | 门控检查数据与拟合等条件是否允许启用策略；blocked表示条件未满足而未执行，收益为空。现金策略实际选择不交易，所以收益为0。 | [项目门控](src/calibration.py#L102)与[阻断结果](src/calibration.py#L115)。空收益与现金0分别解释。 |
| **基线 / 消融** — baseline / ablation | 基线是拿来比较的简单做法，例如固定模型、集成或现金；消融是改动一部分，检查它的作用，例如等权与学习权重对照。 | [第7页 §4.4–§4.5](1679894.pdf#page=7)；项目主比较见 [课程报告](final_replication_report.md)。 |
| **统计充分 / 统计显著性** | 能算一个均值，只说明数据够计算；要证明效果稳定可靠，还需要足够且合适的证据，处理波动与样本相关性。 | [项目资格注释](src/calibration.py#L53)；七日及少量成熟订单没有建立长期收益或显著性结论。 |
| **合成数据 / 合成检查** — synthetic data | 用程序构造的简短行情，用来检查代码是否遵守时序、金额守恒等规则。通过这种检查不等于在真实市场有收益优势。 | [课程性质检查](scripts/check_project.py#L98)。 |

## 7. 查代码注释时再看：优化与文件用语

| 词条 | 最直白的意思 | 出处与本项目用法 |
| --- | --- | --- |
| **sum_only / signed weights** | 只要求权重加起来等于1，允许有限的正权和负权；这些权重用来评分，不能当成概率。 | 权重和见 [第4页 Eq.(3)](1679894.pdf#page=4)；具体求解见 [项目实现](src/sum_only_reward.py)。 |
| **L1 / L2** | L1把系数的绝对值加起来；L2把系数平方相加再开根号。项目用最小L1在最优解间消歧，Ridge用L2的平方惩罚过大系数。 | [奖励求解注释](src/sum_only_reward.py#L15)、[Ridge定义](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。与“L2行情”分别解释。 |
| **最大间隔 / max margin；oracle** | 最大间隔想让专家高于最难胜过的候选；这里oracle就是在完整候选中找最高分的步骤。专家自身也在比较中，所以本项目最优间隔至多为0。 | [项目优化器说明](src/sum_only_reward.py#L1)。这是替代论文未公开生产优化器的工程近似。 |
| **LP / 求解器 / 收敛** — Linear Programming / solver | LP是在线性条件下找最优值；求解器是计算工具；收敛/成功表示求解器认为找到了目标解。项目还会另查数值条件、专家可表示与执行资格。 | [项目求解过程](src/sum_only_reward.py#L15)；算出权重不等于能盈利。 |
| **专家可表示 / 唯一识别** | 可表示是存在合规的评分权重，让专家至少并列最佳；唯一识别是只有一组权重能符合条件。能做到前者，不一定能做到后者。 | [项目资格说明](src/calibration.py#L53)、[求解诊断](src/sum_only_reward.py)。 |
| **盒约束 / box constraint** | 给每个权重另设上下界，例如强制它在0和1之间。它比“权重和为1”多加了限制，当前sum_only不采用它。 | [项目求解约定](src/sum_only_reward.py#L15)；论文Eq.(3)未明确要求非负。 |
| **对偶乘子 / 驻点 / 残差** | 对偶乘子是配给约束的辅助系数；驻点检查目标与约束的作用是否抵消；残差表示相应检查还差多少。它们帮助核对数值解。 | [项目数值复核注释](src/sum_only_reward.py#L35)，用于求解检查。 |
| **消歧 / 并列 / tie-breaking** | 多个候选或权重同样好时，用预先固定的规则选一个。这样避免每次临时挑喜欢的结果。 | [候选冻结顺序](src/calibration.py#L45)、[最小L1约定](src/sum_only_reward.py#L1)。 |
| **标准化 / StandardScaler** | 特征减去训练均值，再除以训练标准差，让不同数量级更容易比较。预测时沿用历史训练参数。 | [官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)与[项目训练](src/session_experiment.py#L166)。 |
| **归一化 / 裁剪** — normalization / clipping | 归一化把量按某个尺度重新表达；裁剪把越界值压到边界。项目Eq.(2)的价格差奖励保留原始单位，不作这些处理。 | [项目奖励定义](src/irl_reward.py#L22)。特征标准化与奖励处理分别看待。 |
| **flags / F_LAST / F_SNAPSHOT / F_BAD_TS_RECV** | flags是一条消息可同时带的标记：F_LAST表示事件的最后一条；F_SNAPSHOT表示回放/源快照消息；F_BAD_TS_RECV表示接收时间不可靠。 | [Databento标记定义](https://databento.com/docs/standards-and-conventions)与[项目筛选](src/timed_snapshots.py#L19)。源快照标记与项目定时快照分别解释。 |
| **向量 / 矩阵 / 零向量；N×K / N×F / P×H** | 向量是一组数，零向量每项都是0；矩阵是一张数表。这里N是样本行数，K/F/P/H分别是模型/特征/策略/尺度数；N×K就是每个时刻、每个模型各有一个预测的表。 | [预测矩阵](src/weekly_library.py#L272)、[校准矩阵](src/calibration.py#L53)。 |
| **UCB变量：W / n / N / C；feedback_counts** | 项目中W是已成熟期间的平均分，n/N是当前模型/全部模型被选的期间数，C控制探索加分；feedback_counts才是收到的完整期间反馈次数。被选过不代表已经收到评价。 | [第5页 Eq.(6)](1679894.pdf#page=5)、[项目计数约定](src/period_ucb.py#L16)。这里N表示访问总数，矩阵形状中的N表示样本行数。 |
| **mask / feature_valid / label_valid / learning_ready** | mask是一张保留/排除的真假清单；feature_valid检查当下特征能否计算，后两者涉及未来标签的有效性等离线信息。不能因后来查不到标签，就提前删掉当时的行情或决策。 | [特征预热](src/snapshot_dataset.py#L207)、[行情读取](src/snapshot_backtest.py#L65)。 |
| **NaN / None / null** | NaN常表示数值缺失或无效；None是Python空值，写成JSON时对应null。结合状态解释它们，不能直接补成现金的零收益。 | [缺失特征](src/snapshot_dataset.py#L207)、[序列化](src/artifacts.py#L8)、[阻断结果](src/calibration.py#L115)。 |
| **JSON / schema / 序列化** | JSON是用字段名和数值等保存信息的文本格式；schema约定有哪些字段及其类型；序列化就是把程序里的对象写成文件。 | [项目文件输出](src/artifacts.py)与[数据格式核验](src/snapshot_dataset.py#L136)。 |
| **元数据 / metadata；provenance / 来源记录** | 元数据是数据的说明，例如来源、日期、行数和质量；来源记录说明文件从哪里来、怎样得到。它们帮助解释和追溯结果。 | [项目来源记录](src/timed_snapshots.py#L200)、[冻结输入](src/session_experiment.py#L78)。 |
| **哈希 / SHA256 / fingerprint** | 给文件内容算一个“指纹”，方便检查是否换过内容。相同指纹用于核对同一份文件，不能单凭它证明数据真实或方法正确。 | [项目文件身份](src/snapshot_dataset.py#L29)、[结果核对](scripts/check_project.py#L35)。SHA256是这里使用的哈希算法。 |
| **流式 / 分批 / partial_fit / 充分统计量** | 流式、分批是一次处理部分数据；partial_fit逐批累计或更新。这里充分统计量是保留求模型所需的累计量，避免拼出整个季度的大表。 | [项目两遍训练](src/session_experiment.py#L166)、[周模型训练](src/weekly_library.py#L186)。 |
| **行组 / footer** — row group / 文件尾部元数据 | Parquet把行分成批次存放，这种批次叫行组；footer记有行数等说明。代码可以先看这些说明来少读数据。 | [只读行数](src/data_catalog.py#L50)、[按行组读取](src/snapshot_backtest.py#L84)。 |
| **CLI / I/O / ID** | CLI是命令行接口，例如输入`python run_project.py`；I/O是数据读写；ID是区分对象的编号或名称。模型ID标识候选规则，版本ID标识某一周的参数。 | [运行入口](run_project.py)、[模型版本注释](src/weekly_library.py#L272)。 |
| **随机种子 / seed** | 随机抽样使用的固定起始编号，方便在同一设置下重复抽样过程。它不是按收益挑选“好样本”的参数。 | [项目抽样规则](src/weekly_library.py#L186)。 |
| **float32** | 用32个二进制位表示一个浮点数，会有舍入误差。树预测保持同样精度，以免同一个值落到不同判断分支。 | [项目预测精度说明](src/weekly_library.py#L250)。 |
| **影子评估 / CausalEventUCB** | 影子评估是在模拟中给未选模型打分，不直接形成账户成交；CausalEventUCB是旧的逐事件选择变体。当前主线按完整期间反馈，不能把旧变体名称当作论文原算法。 | [接口开关](src/model_selector.py#L4)、[拒绝逐订单更新](src/period_ucb.py#L70)、[真实期间归属](src/period_ucb.py#L94)。 |
