# FMATO 思路的课程工程实验（非论文数值复现）

> 旧版含前视偏差的收益结论已撤回。本报告由 JSON 自动生成。

## 论文定义与工程边界

详见 [逐项论文对照](docs/paper_alignment.md) 与 [原论文](1679894.pdf)。

- 默认 `paper_price_difference` 按 Eq.(2) 计算绝对价格差，不归一化、不裁剪；卖单取方向符号为项目扩展。`normalized_return` 另以训练 P99 缩放相对收益并裁剪，不混用两种单位。
- Eq.(3) 明确要求权重和为 1；非负约束是项目假设，另报告允许负权的 SignedBox 敏感性。有限策略优化不等于论文未公开的生产策略优化器。
- Eq.(5) 校准 OE 除以成熟订单数。无订单均值未定义，仅在有限策略优化中约定现金零向量。默认 OE 不扣成本；NetOE 在固定权重下单列敏感性。资金账本始终扣实际费用。
- CausalShadowARS 是连续影子账户、300 个奖励到达事件滑窗，尚未实现 Algorithm 3 的过去固定时间窗口重新回测。CausalEventUCB 每事件选择、每成熟决策反馈一次，无成交反馈零，不是 Algorithm 2 的固定期间订单平均。
- 五种算法使用同一特征及训练期；尚未实现论文不同特征集、历史时期与市场分布构成的候选库或每周更新。
- ME 与执行共用门槛，只平均过门槛的校准信号；EventUCB 对未交易决策反馈零。未实现论文强调的极端 top-1/1000 信号评价协议。
- 10/30/90 为不等间隔订单簿事件数，不是秒或定时快照；只有三个短尺度，未覆盖论文一小时以上时域。
- 数据按 50%/15%/15%/20% 切分，前三段清除跨段标签。先选共同验证门槛，再重放校准拟合权重，随后选 c/固定模型。测试隔离，但同一短验证段反复选参存在过拟合风险，尚未做嵌套验证。
- CME/ICE 的合约、市场制度、费用、更新机制不同于论文中国期货数据，未复现原论文实验数值。文件内窗口不是跨日独立重复。
- 下一事件主动成交；逐事件盯市；期末平仓；每窗资金重置为 $100,000。未模拟排队、部分成交、市场冲击、保证金与端到端延迟；短样本不报告年化夏普。

## 配置与复核

种子 42；每品种 3 个窗口；每窗 60,000 条原始事件；Python 3.12.14；计算线程 1。

JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。

## CME_ES：文件内多窗口描述统计（非独立重复实验）

数据 `databento_glbx.mdp3_mbp_10.parquet`，共 2,807,571 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 |
|---|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1470.00 | -490.00 | -1100.00 | -152.50 | 53 |
| Variant-ME-CausalShadowARS | -1202.50 | -400.83 | -897.50 | -127.50 | 46 |
| Variant-OE-EventUCB | -330.00 | -110.00 | -205.00 | -27.50 | 12 |
| Variant-OE-CausalShadowARS | -1225.00 | -408.33 | -892.50 | -155.00 | 45 |
| Baseline-Single-Ridge_Linear | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Single-Logistic_Direction | -1692.50 | -564.17 | -1345.00 | 0.00 | 62 |
| Baseline-Single-Decision_Tree | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Single-Hist_GBDT | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Single-Rule_Momentum | -245.00 | -81.67 | -245.00 | 0.00 | 8 |
| Baseline-Static-Ensemble | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Random | -787.50 | -262.50 | -510.00 | -82.50 | 30 |
| Baseline-RoundRobin | -1002.50 | -334.17 | -685.00 | -112.50 | 36 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Random-seed-43 | -825.00 | -275.00 | -577.50 | -122.50 | 30 |
| Baseline-Random-seed-44 | -775.00 | -258.33 | -552.50 | -110.00 | 30 |
| Ablation-ME-CausalShadowARS-Equal | -1325.00 | -441.67 | -952.50 | -177.50 | 50 |
| Ablation-ME-CausalShadowARS-Single-10 | -1392.50 | -464.17 | -1020.00 | -177.50 | 52 |
| Ablation-ME-CausalShadowARS-Single-30 | -1255.00 | -418.33 | -897.50 | -177.50 | 47 |
| Ablation-ME-CausalShadowARS-Single-90 | -1050.00 | -350.00 | -745.00 | -127.50 | 40 |
| Ablation-OE-CausalShadowARS-Equal | -1157.50 | -385.83 | -852.50 | -127.50 | 43 |
| Ablation-OE-CausalShadowARS-Single-10 | -922.50 | -307.50 | -617.50 | -127.50 | 34 |
| Ablation-OE-CausalShadowARS-Single-30 | -1032.50 | -344.17 | -647.50 | -177.50 | 38 |
| Ablation-OE-CausalShadowARS-Single-90 | -1225.00 | -408.33 | -892.50 | -155.00 | 45 |
| Sensitivity-ME-SignedBox | -1187.50 | -395.83 | -897.50 | -112.50 | 45 |
| Sensitivity-OE-SignedBox | -1170.00 | -390.00 | -852.50 | -140.00 | 43 |
| Sensitivity-NetOE-FixedWeights | -510.00 | -170.00 | -220.00 | -112.50 | 19 |

### 窗口 1

测试 2025-09-22 00:56:01.100857925+00:00 → 2025-09-22 01:05:42.326551493+00:00，581.23 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Decision_Tree', 'Hist_GBDT']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Logistic_Direction', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`；UCB-OE：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 1.0000 / 0.0000 / 0.0000 | True | True | -0.000000 |
| OE / simplex | 1.0000 / 0.0000 / 0.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | True | -0.000000 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | True | -0.000000 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 34.35% | 53.46% | N/A | 0.00% | N/A | 43.29 / 114.62 |
| Logistic_Direction | 34.26% | 53.46% | N/A | 0.00% | 56.35% | 64.03 / 253.02 |
| Decision_Tree | 33.99% | 53.46% | N/A | 0.00% | N/A | 47.39 / 93.07 |
| Hist_GBDT | 34.69% | 53.46% | N/A | 0.00% | N/A | 162.05 / 300.93 |
| Rule_Momentum | 31.85% | 53.38% | 21.88% | 0.27% | N/A | 3.35 / 3.92 |

零信号准确率 53.46%；训练多数类准确率 53.46%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 1.2549504662, '0.5': 5.124046678999999, '0.9': 11.5675809418}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-09-22 13:59:49.662763235+00:00 → 2025-09-22 14:00:19.921952383+00:00，30.26 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.7720 / 0.2280 | True | False | -0.053495 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.005208 |
| ME / signed_box | -0.1828 / 1.0000 / 0.1828 | True | False | -0.047845 |
| OE / signed_box | -0.7917 / 1.0000 / 0.7917 | True | True | -0.000000 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.16% | 75.45% | N/A | 0.00% | N/A | 42.59 / 73.25 |
| Logistic_Direction | 22.73% | 75.90% | 54.19% | 1.50% | 75.75% | 60.23 / 96.38 |
| Decision_Tree | 21.53% | 75.45% | N/A | 0.00% | N/A | 49.08 / 79.22 |
| Hist_GBDT | 22.74% | 75.45% | N/A | 0.00% | N/A | 170.31 / 445.12 |
| Rule_Momentum | 21.47% | 75.45% | N/A | 0.00% | N/A | 3.27 / 3.56 |

零信号准确率 75.45%；训练多数类准确率 75.45%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.04134640420000002, '0.5': 0.33158887800000003, '0.9': 0.7446652415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：72/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：2/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-09-22 15:58:06.928056965+00:00 → 2025-09-22 15:59:59.951415627+00:00，113.02 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.080357 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.015625 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.066964 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.015625 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 16.96% | 82.34% | N/A | 0.00% | N/A | 39.86 / 58.14 |
| Logistic_Direction | 17.23% | 82.52% | 83.33% | 0.25% | 84.82% | 60.49 / 120.72 |
| Decision_Tree | 16.77% | 82.34% | N/A | 0.00% | N/A | 48.06 / 67.73 |
| Hist_GBDT | 17.18% | 82.34% | N/A | 0.00% | N/A | 159.24 / 350.38 |
| Rule_Momentum | 16.24% | 82.34% | N/A | 0.00% | N/A | 3.07 / 3.36 |

零信号准确率 82.34%；训练多数类准确率 82.34%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0501033618, '0.5': 0.676730374, '0.9': 1.5287187415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：16/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。
## ICE_BRENT：文件内多窗口描述统计（非独立重复实验）

数据 `databento_ifeu.impact_mbp_10.parquet`，共 2,292,832 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 |
|---|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -2647.00 | -882.33 | -2387.00 | -23.00 | 109 |
| Variant-ME-CausalShadowARS | -4544.00 | -1514.67 | -4271.00 | -23.00 | 178 |
| Variant-OE-EventUCB | -7485.00 | -2495.00 | -7245.00 | -23.00 | 305 |
| Variant-OE-CausalShadowARS | -3823.00 | -1274.33 | -3573.00 | -23.00 | 151 |
| Baseline-Single-Ridge_Linear | -128.00 | -42.67 | -79.00 | 0.00 | 6 |
| Baseline-Single-Logistic_Direction | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-Single-Decision_Tree | -7914.00 | -2638.00 | -7914.00 | 0.00 | 318 |
| Baseline-Single-Hist_GBDT | -7434.00 | -2478.00 | -7197.00 | -46.00 | 288 |
| Baseline-Single-Rule_Momentum | -372.00 | -124.00 | -372.00 | 0.00 | 14 |
| Baseline-Static-Ensemble | -7084.00 | -2361.33 | -7084.00 | 0.00 | 278 |
| Baseline-Random | -7494.00 | -2498.00 | -7349.00 | -23.00 | 298 |
| Baseline-RoundRobin | -7718.00 | -2572.67 | -7458.00 | -23.00 | 316 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| Baseline-ValidationBest | -49.00 | -16.33 | -49.00 | 0.00 | 3 |
| Baseline-Random-seed-43 | -7318.00 | -2439.33 | -7147.00 | 0.00 | 296 |
| Baseline-Random-seed-44 | -7449.00 | -2483.00 | -7219.00 | -46.00 | 303 |
| Ablation-ME-CausalShadowARS-Equal | -4414.00 | -1471.33 | -4141.00 | -23.00 | 178 |
| Ablation-ME-CausalShadowARS-Single-10 | -3594.00 | -1198.00 | -3321.00 | -23.00 | 148 |
| Ablation-ME-CausalShadowARS-Single-30 | -4220.00 | -1406.67 | -3970.00 | -23.00 | 170 |
| Ablation-ME-CausalShadowARS-Single-90 | -4544.00 | -1514.67 | -4271.00 | -23.00 | 178 |
| Ablation-OE-CausalShadowARS-Equal | -4753.00 | -1584.33 | -4503.00 | -23.00 | 191 |
| Ablation-OE-CausalShadowARS-Single-10 | -3369.00 | -1123.00 | -3119.00 | -23.00 | 133 |
| Ablation-OE-CausalShadowARS-Single-30 | -3823.00 | -1274.33 | -3573.00 | -23.00 | 151 |
| Ablation-OE-CausalShadowARS-Single-90 | -4290.00 | -1430.00 | -3994.00 | -23.00 | 170 |
| Sensitivity-ME-SignedBox | -4424.00 | -1474.67 | -4151.00 | -23.00 | 178 |
| Sensitivity-OE-SignedBox | -5119.00 | -1706.33 | -4823.00 | -23.00 | 203 |
| Sensitivity-NetOE-FixedWeights | -345.00 | -115.00 | -227.00 | -23.00 | 15 |

### 窗口 1

测试 2025-06-09 01:22:55.205014002+00:00 → 2025-06-09 01:38:08.188182002+00:00，912.98 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00024`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Decision_Tree', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Logistic_Direction', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：并列最佳 `[0.1, 0.8]`，按预先声明顺序选 `0.1`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.011111 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.000556 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.011111 |
| OE / signed_box | 1.0000 / -1.0000 / 1.0000 | True | True | -0.000000 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 33.79% | 52.78% | N/A | 0.00% | N/A | 40.04 / 66.05 |
| Logistic_Direction | 33.79% | 52.78% | N/A | 0.00% | 51.81% | 61.07 / 174.26 |
| Decision_Tree | 35.29% | 52.78% | N/A | 0.00% | N/A | 48.65 / 148.01 |
| Hist_GBDT | 32.42% | 52.83% | 100.00% | 0.05% | N/A | 169.48 / 369.73 |
| Rule_Momentum | 34.12% | 52.78% | N/A | 0.00% | N/A | 3.16 / 3.56 |

零信号准确率 52.78%；训练多数类准确率 52.78%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.4946915994000001, '0.5': 5.168116001, '0.9': 22.5388601014}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：18/True；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-06-09 13:41:48.627059002+00:00 → 2025-06-09 13:44:10.342476002+00:00，141.72 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[0.00012, 0.00024]`，按预先声明顺序选 `0.00012`。固定模型：并列最佳 `['Logistic_Direction', 'Decision_Tree']`，按预先声明顺序选 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.8`；UCB-OE：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.022500 |
| OE / simplex | 1.0000 / 0.0000 / 0.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.015000 |
| OE / signed_box | 0.5000 / -0.5000 / 1.0000 | True | True | -0.000000 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.14% | 70.70% | 100.00% | 0.03% | N/A | 39.74 / 52.36 |
| Logistic_Direction | 23.04% | 70.67% | N/A | 0.00% | 71.03% | 60.80 / 131.04 |
| Decision_Tree | 22.25% | 70.67% | N/A | 0.00% | N/A | 47.86 / 127.73 |
| Hist_GBDT | 23.24% | 70.66% | 47.62% | 0.18% | N/A | 165.65 / 367.09 |
| Rule_Momentum | 22.23% | 70.63% | 15.00% | 0.17% | N/A | 3.42 / 4.20 |

零信号准确率 70.67%；训练多数类准确率 70.67%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0211009952, '0.5': 0.6737915005, '0.9': 2.4399680977}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：10/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：2/True；Baseline-Single-Hist_GBDT：6/True；Baseline-Single-Rule_Momentum：12/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-06-09 20:31:36.437188002+00:00 → 2025-06-09 22:00:00.245000+00:00，5303.81 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00024`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Single-Rule_Momentum', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.032500 |
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | False | -0.000085 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.032500 |
| OE / signed_box | -0.6894 / 1.0000 / 0.6894 | True | False | -0.000085 |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 28.76% | 60.30% | 100.00% | 0.05% | N/A | 41.84 / 121.95 |
| Logistic_Direction | 29.09% | 60.25% | N/A | 0.00% | 60.75% | 63.56 / 109.75 |
| Decision_Tree | 22.05% | 26.07% | 13.97% | 60.95% | N/A | 48.58 / 105.99 |
| Hist_GBDT | 22.50% | 29.29% | 10.32% | 50.94% | N/A | 172.81 / 347.96 |
| Rule_Momentum | 27.49% | 60.25% | N/A | 0.00% | N/A | 3.02 / 3.24 |

零信号准确率 60.25%；训练多数类准确率 60.25%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.035222500000000004, '0.5': 5.053657501, '0.9': 23.3169947982}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：4/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：235/True；Baseline-Single-Hist_GBDT：204/True；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：204/True；Baseline-Cash：0/False。

## 解释限制与后续工作

求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。

当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。
