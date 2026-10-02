# 固定期间 OE-UCB 验证（2026-10-02）

## 问题与论文依据

PR #12 已接通轻模型、真实时间执行和 OE 校准，但只用静态候选回放，不能检验
Algorithm 2 的在线选择。开发前重新阅读论文物理第 3–7 页，重点核对第 5 页
§3.3.2–§3.4、Eq.(5)–(6)、Algorithm 2，以及第 3–4 页 §3.1/§3.2、
Table 2、Eq.(2)–(4) 与第 7 页实验表格。论文用固定期间交易，按订单平均
评价动作，再以平均奖励和探索次数构成 UCB 分数。

本次增加独立期间 OE-UCB 协议，保留事件变体的原命名。没有恢复论文全部
生产参数：原文同时写“每 tick 选择”与“固定期间交易”，未公开期间内换模、
长前瞻反馈、零订单、W 运行平均和版本更新细节。下述约定均是项目解释。

## 明确的协议

- 每个 session 开盘锚定墙钟 `[start,end)`，5 分钟为论文 Algorithm 2 的例子。
  当前期间首次有效决策分配一次模型，此后逐 tick 锁定到期间结束。休市、
  跳格不压缩时间，没有有效决策的期间不生成假访问。
- Eq.(6) 为 `Q=W+C*sqrt(2*ln(N)/n)`，`n` 数实际模型选择期间，`N=sum(n)`。
  在选择时记访问，反馈到达时记反馈数；未访臂视为正无穷，按候选声明顺序
  消歧。无反馈的 W 暂为初始化 0，实测均值另列为 None。C=1 个价格点是
  开发假设，不是论文公开值，未根据后两段收益调参。
- 全部真实成交按成交时刻、模型、数值版本和期间分桶。期间结束且所有订单
  的七尺度 OE 标签均解决后才评价；全部成熟才用 `sum(score)/orders`，
  完整期间中任何订单缺格/跨 session 均令整桶不可用。无订单和末端未成熟
  不补零；W 是已观察完整期间均值的等权平均，不按期间订单数加权。
- 共享账户换模型不强制退出；原仓位和退出归属原开仓模型/版本，触发信号
  另记。因此同期间可能有多个模型桶，退出桶可能没有当期新访问。该归属
  是现有项目账本约定，不是论文公开的唯一规则。
- 数值版本各有独立 W/n，旧奖励只改旧统计；换版撤销旧待成交意图。正常
  模型切换保留仍合法的到期意图。每日新账户/选择器/队列，不跨日迁移。
  缺尾未平仓保留并使多日净利汇总未定义，不任意回溯强平。
- 只在当前特征与全部库模型预测均有限时进行动态决策，静态候选用自身
  可用性。没有未来标签掩码；本次真实动态不可用预测行数均为 0。

本次**预声明等权奖励控制**，不使用 PR #12 不可表示专家的诊断权重，不
声称成功 IRL 的 FMATO-OE-UCBS。候选为全部 12 个库模型，不把现金当作
预测臂；对照保留同期间轮换、现金、原固定 Ridge 与 12 个固定库模型，共
16 个策略。轮换不消费反馈来选模型，仍记录相同期间评价以核对覆盖。

## 数据、单位和成本

复用 PR #9 同批完整 UTC 源文件及三个 prepared 年龄对照，所有历史结果
与原始数据保留。训练 2025-09-29/30；原校准日 10-01 不在本入口重新拟合
奖励；评价为验证 10-02、测试 10-03。均是已经使用的开发日期，没有未触碰
测试集。训练/版本可见性沿用周库，启动 10-01，后续按预声明周计划更新。

500ms 网格与执行延迟、5000ms 预测前瞻、15000ms 持仓复核、300000ms
模型选择期间分别独立。OE 为 5000/15000/45000/135000/405000/1215000/
3645000ms 七尺度中间价差，卖单方向符号为项目扩展；不扣费、不缩放、不裁剪。
每订单最长超过一小时，不能因选择期间只有五分钟就提前读取其未来价格。
报价年龄 500ms 严格基线与 1000/2000ms 敏感性全部保留，不按收益选年龄。
论文半秒与每秒四快照的频率差异仍未消除。

ES 一手，合约乘数 50、最小跳动 0.25、单订单手续费 1.25 美元、不利
滑点 0.5 跳及实测点差，沿用所有策略相同的主动成交假设。盘口量不是论文
成交量，特征/短历史/树抽样仍为周库工程近似，CME 不能替代中国商品期货
原始数值。毛利、摩擦和净利分列。

## 真实负结果与反馈覆盖

UCB 在两个评价日及全部年龄均亏损，没有超越现金。严格 500ms 时，相比
同期间轮换分别少亏 10/345 美元；1000/2000ms 均更差。不能据此声称稳定
收益改进，也不选择最有利年龄发布。与原 Ridge 的亏损差异同时包含模型
身份、交易次数和成本变化，不能单独归因于奖励学习。

即使单笔订单已有成熟 OE，也经常不能形成完整可评价期间；期间内一笔坏
标签就会导致整桶不更新。严格基线两个评价日 UCB 仅收到 1/13 次反馈。
12 份动态日回放（UCB+轮换、三年龄、两天）共 171 次完整期间反馈。
此覆盖和大量无订单/不完整状态属于结果，不能把全部访问当成已学样本。

| 年龄 ms | 阶段 | 策略 | 选择期间 | 成交 | 成熟订单 | 完整期间反馈 | 毛利 USD | 摩擦 USD | 净利 USD | 期间状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | validation | Period-OE-UCB | 132 | 708 | 28 | 1 | -187.5 | 9772.5 | -9960.0 | {'no_orders': 66, 'incomplete_orders': 83, 'matured': 1, 'pending_orders': 3} |
| 500 | validation | Period-OE-RoundRobin | 132 | 736 | 28 | 1 | 187.5 | 10157.5 | -9970.0 | {'no_orders': 64, 'incomplete_orders': 82, 'matured': 1, 'pending_orders': 5} |
| 500 | test | Period-OE-UCB | 122 | 694 | 136 | 13 | 231.25 | 9586.25 | -9355.0 | {'no_orders': 54, 'incomplete_orders': 68, 'matured': 13, 'pending_orders': 6} |
| 500 | test | Period-OE-RoundRobin | 122 | 690 | 136 | 13 | -162.5 | 9537.5 | -9700.0 | {'no_orders': 58, 'incomplete_orders': 68, 'matured': 13, 'pending_orders': 5} |
| 1000 | validation | Period-OE-UCB | 235 | 970 | 237 | 24 | 468.75 | 13468.75 | -13000.0 | {'no_orders': 110, 'incomplete_orders': 114, 'matured': 24, 'pending_orders': 5} |
| 1000 | validation | Period-OE-RoundRobin | 235 | 882 | 145 | 17 | 800.0 | 12252.5 | -11452.5 | {'no_orders': 125, 'incomplete_orders': 115, 'matured': 17, 'pending_orders': 5} |
| 1000 | test | Period-OE-UCB | 238 | 1112 | 178 | 14 | 1287.5 | 15390.0 | -14102.5 | {'no_orders': 109, 'incomplete_orders': 118, 'matured': 14, 'pending_orders': 7} |
| 1000 | test | Period-OE-RoundRobin | 238 | 788 | 146 | 17 | 106.25 | 10928.75 | -10822.5 | {'no_orders': 123, 'incomplete_orders': 109, 'matured': 17, 'pending_orders': 7} |
| 2000 | validation | Period-OE-UCB | 273 | 1358 | 185 | 22 | 1425.0 | 18822.5 | -17397.5 | {'no_orders': 149, 'incomplete_orders': 123, 'matured': 22, 'pending_orders': 7} |
| 2000 | validation | Period-OE-RoundRobin | 273 | 1136 | 192 | 20 | 1087.5 | 15770.0 | -14682.5 | {'no_orders': 162, 'incomplete_orders': 119, 'matured': 20, 'pending_orders': 7} |
| 2000 | test | Period-OE-UCB | 272 | 1434 | 129 | 16 | -1662.5 | 19892.5 | -21555.0 | {'no_orders': 157, 'incomplete_orders': 130, 'matured': 16, 'pending_orders': 7} |
| 2000 | test | Period-OE-RoundRobin | 272 | 1266 | 87 | 13 | -493.75 | 17588.75 | -18082.5 | {'no_orders': 167, 'incomplete_orders': 129, 'matured': 13, 'pending_orders': 5} |

完整期间桶数不一定等于选择次数：原仓位在其他期间退出会产生未当期访问
模型的真实成交桶。JSON 保留 `selected`、原 owner/version、所有订单状态、
开始/结束/观察时刻和奖励分母，不能把这些桶误当成追加的模型访问。

### 全部策略对照

以下按声明顺序保留全部净利，完整费用/订单/期间统计见生成的 `report.md`
和 `result.json`。单个固定模型的局部盈利保留，不能据后续结果反选它。

| 年龄 ms | 策略 | 验证净利 USD | 测试净利 USD |
| --- | --- | --- | --- |
| 500 | Period-OE-UCB | -9960.0 | -9355.0 |
| 500 | Period-OE-RoundRobin | -9970.0 | -9700.0 |
| 500 | cash | 0.0 | 0.0 |
| 500 | Ridge-threshold-x1 | -24215.0 | -29135.0 |
| 500 | Fixed-Ridge-price-h1 | -47.5 | -897.5 |
| 500 | Fixed-DecisionTree-price-h1 | -852.5 | -955.0 |
| 500 | Fixed-Ridge-displayed_volume-h1 | -12020.0 | -11652.5 |
| 500 | Fixed-DecisionTree-displayed_volume-h1 | -2527.5 | -2295.0 |
| 500 | Fixed-Ridge-price_volume-h1 | -9640.0 | -10227.5 |
| 500 | Fixed-DecisionTree-price_volume-h1 | -23530.0 | -24422.5 |
| 500 | Fixed-Ridge-price-h2 | -12.5 | -367.5 |
| 500 | Fixed-DecisionTree-price-h2 | -1302.5 | -1715.0 |
| 500 | Fixed-Ridge-displayed_volume-h2 | -14947.5 | -14292.5 |
| 500 | Fixed-DecisionTree-displayed_volume-h2 | -11310.0 | -8387.5 |
| 500 | Fixed-Ridge-price_volume-h2 | -12990.0 | -12827.5 |
| 500 | Fixed-DecisionTree-price_volume-h2 | -26802.5 | -26327.5 |
| 1000 | Period-OE-UCB | -13000.0 | -14102.5 |
| 1000 | Period-OE-RoundRobin | -11452.5 | -10822.5 |
| 1000 | cash | 0.0 | 0.0 |
| 1000 | Ridge-threshold-x1 | -27217.5 | -31977.5 |
| 1000 | Fixed-Ridge-price-h1 | -82.5 | -1000.0 |
| 1000 | Fixed-DecisionTree-price-h1 | -900.0 | -1235.0 |
| 1000 | Fixed-Ridge-displayed_volume-h1 | -15490.0 | -13765.0 |
| 1000 | Fixed-DecisionTree-displayed_volume-h1 | -11045.0 | -6695.0 |
| 1000 | Fixed-Ridge-price_volume-h1 | -12787.5 | -10937.5 |
| 1000 | Fixed-DecisionTree-price_volume-h1 | -32542.5 | -29072.5 |
| 1000 | Fixed-Ridge-price-h2 | -90.0 | -387.5 |
| 1000 | Fixed-DecisionTree-price-h2 | -1797.5 | -2942.5 |
| 1000 | Fixed-Ridge-displayed_volume-h2 | -17135.0 | -14755.0 |
| 1000 | Fixed-DecisionTree-displayed_volume-h2 | -20720.0 | -18592.5 |
| 1000 | Fixed-Ridge-price_volume-h2 | -15570.0 | -14330.0 |
| 1000 | Fixed-DecisionTree-price_volume-h2 | -17017.5 | -14077.5 |
| 2000 | Period-OE-UCB | -17397.5 | -21555.0 |
| 2000 | Period-OE-RoundRobin | -14682.5 | -18082.5 |
| 2000 | cash | 0.0 | 0.0 |
| 2000 | Ridge-threshold-x1 | -30862.5 | -33827.5 |
| 2000 | Fixed-Ridge-price-h1 | 50.0 | -705.0 |
| 2000 | Fixed-DecisionTree-price-h1 | -872.5 | -1342.5 |
| 2000 | Fixed-Ridge-displayed_volume-h1 | -9862.5 | -10707.5 |
| 2000 | Fixed-DecisionTree-displayed_volume-h1 | -16457.5 | -13992.5 |
| 2000 | Fixed-Ridge-price_volume-h1 | -10042.5 | -7612.5 |
| 2000 | Fixed-DecisionTree-price_volume-h1 | -61145.0 | -60195.0 |
| 2000 | Fixed-Ridge-price-h2 | -227.5 | -862.5 |
| 2000 | Fixed-DecisionTree-price-h2 | -2825.0 | -5085.0 |
| 2000 | Fixed-Ridge-displayed_volume-h2 | -5732.5 | -6127.5 |
| 2000 | Fixed-DecisionTree-displayed_volume-h2 | -15622.5 | -14292.5 |
| 2000 | Fixed-Ridge-price_volume-h2 | -4862.5 | -3117.5 |
| 2000 | Fixed-DecisionTree-price_volume-h2 | -70220.0 | -63715.0 |

## 验证与复现身份

- `.venv/bin/python -m unittest discover -s tests -v`：156 项通过（新增 14 项）。
  日志 `/tmp/period-ucb-full-tests.log`。
- 手算三单 2/4/6 均值 4；期间未结束或最后一单未成熟均不反馈；一坏单令
  整桶不可用，无订单/未成熟不补零；W 等权期间平均与精确 ln(N) 分数验证。
- 改变未来价格，此前 tick/期间选择、成交和期间反馈不变；抹去离线未来
  标签仍完全相同。访问数与反馈数独立，缺格不压缩，换版不污染新统计。
- 七个合成 session 端到端覆盖真实日历的下一周版本，另外单次回放换版
  验证旧仓位/反馈归属。真实输入仍只一个交易周，10-06 更新版仅建库，
  没有当日真实成交评估，不混淆两种证据。
- 三年龄所有库参数/版本与 PR #12 产物完全一致。84 份静态对照的原有
  回放字段逐项一致（只排除策略命名及旧入口独有日边界外层注记）。
- 12 份真实动态回放核对全部成交、成熟/待成熟订单、访问及完整反馈计数
  守恒；反馈严格晚于期间结束与最后订单最长前瞻，精确 UCB 分数逐条核对。
  当前源码哈希与冻结结果一致。
- 源码、模型文件、质量数据改变拒绝沿用冻结身份；错误奖励源与覆盖输出
  拒绝运行。语法、文档链接和 diff 检查通过。
- 老入口小窗口 `run_experiments.py --rows 3000 --windows 1`、报告及看板
  联动成功，另存 `/tmp/period-ucb-legacy-smoke.json`、
  `/tmp/period-ucb-legacy-report.md`、`/tmp/period-ucb-legacy-dashboard.html`。

实际命令（新路径不覆盖旧产物）：

```bash
.venv/bin/python run_weekly_library.py freeze \
  --config config/esz5_weekly_library_development.json \
  --dataset-dirs /tmp/data-mining-snapshot-age-audit/age-500/prepared \
                 /tmp/data-mining-snapshot-age-audit/age-1000/prepared \
                 /tmp/data-mining-snapshot-age-audit/age-2000/prepared \
  --output-dir /tmp/data-mining-period-ucb/library-plan
.venv/bin/python run_weekly_library.py build \
  --plan /tmp/data-mining-period-ucb/library-plan/plan.json \
  --output-dir /tmp/data-mining-period-ucb/library
.venv/bin/python run_period_ucb.py freeze \
  --config config/esz5_period_ucb_development.json \
  --library /tmp/data-mining-period-ucb/library/library.json \
  --output-dir /tmp/data-mining-period-ucb/frozen
.venv/bin/python run_period_ucb.py run \
  --plan /tmp/data-mining-period-ucb/frozen/plan.json \
  --output-dir /tmp/data-mining-period-ucb/evaluation
```

产物留在本地 `/tmp`，GitHub 不提交原始行情或大结果。冻结计划记录完整
配置、源码/数据/日历和模型文件哈希；模型文件保存实际训练参数，运行结果
记录依赖。代码改变须另存重建，旧身份不可套到新代码。

| 产物 | 文件 SHA-256 | 计划身份 |
| --- | --- | --- |
| `library-plan/plan.json` | `cf84c126370999fed8380158c21a6218c2ae8220f8a2a953822a20b7d12337da` | `68da6dc6cf70b0010c9f94d2fe8281c159cd005539f02001be005f40e1220871` |
| `library/library.json` | `4a848272c2207cc3b3cd45ee29e9327fb328a156af9a44368a66890724bee2f2` | `68da6dc6cf70b0010c9f94d2fe8281c159cd005539f02001be005f40e1220871` |
| `frozen/plan.json` | `6dc49f040665b22335d06d0edc7b7f14ad93f748bae1ec748da88cbd13d1e6b2` | `f30ae70ce1239013ac88bd0f30be538b4dff1ecf6c0285d9f4c725fd9d6a3ec9` |
| `evaluation/result.json` | `196916146b55e1a6dee87e5e0a96b428fc2ec07de22675165111a4ffb78f4dd6` | `f30ae70ce1239013ac88bd0f30be538b4dff1ecf6c0285d9f4c725fd9d6a3ec9` |

依赖：python 3.12.14；numpy 2.5.3；pandas 3.0.6；pyarrow 25.0.1；scikit-learn 1.9.1；scipy 1.18.1；threadpoolctl 3.7.0。

## 剩余差异

本次缩小“选择器仍按事件或逐订单更新”的差异，新增可审计固定期间与延迟
反馈机制。等权控制不是学成的奖励，期间锁定/运行平均/完整订单/零订单/
版本归属仍是项目约定，不能声称 Algorithm 2 完全等价实现或盈利复现。

还缺可用 IRL 奖励联动、时间 ME IRL、Algorithm 3 固定历史窗口回测、真实
跨周覆盖与三个月全量运行。长奖励前瞻与短历史回测窗口的可见性须先明确，
不能用当前时点之后的价格来计算所谓历史窗口奖励。扩展数据量不能自动
解决专家不可表示或点差/费用主导的亏损。
