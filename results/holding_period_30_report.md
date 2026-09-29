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
- 下一事件主动成交；逐事件盯市；期末平仓；每窗资金重置为 $100,000。未模拟排队、部分成交、市场冲击、保证金与端到端延迟；当前样本不足以形成独立日收益，因此不报告 Sharpe。

## 配置与复核

种子 42；每品种 3 个窗口；每窗 60,000 条原始事件；holding period 30 events；Python 3.12.14；计算线程 1。

JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。

## CME_ES：文件内多窗口描述统计（非独立重复实验）

数据 `databento_glbx.mdp3_mbp_10.parquet`，共 2,807,571 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1442.50 | -480.83 | -1072.50 | -152.50 | 52 | 36.54% | 0.00% | 1.92 | 29.66 | 30.58 |
| Variant-ME-CausalShadowARS | -1175.00 | -391.67 | -870.00 | -127.50 | 45 | 42.22% | 0.00% | 3.19 | 29.31 | 30.67 |
| Variant-OE-EventUCB | -330.00 | -110.00 | -205.00 | -27.50 | 12 | 16.67% | 0.00% | 0.52 | 28.02 | 30.00 |
| Variant-OE-CausalShadowARS | -1197.50 | -399.17 | -865.00 | -155.00 | 44 | 34.09% | 0.00% | 2.13 | 29.35 | 30.68 |
| Baseline-Single-Ridge_Linear | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Logistic_Direction | -1637.50 | -545.83 | -1290.00 | 0.00 | 60 | 41.67% | 0.00% | 2.60 | 29.90 | 31.00 |
| Baseline-Single-Decision_Tree | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Hist_GBDT | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Rule_Momentum | -205.00 | -68.33 | -205.00 | 0.00 | 7 | 14.29% | 0.00% | -0.89 | 28.39 | 30.00 |
| Baseline-Static-Ensemble | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random | -787.50 | -262.50 | -510.00 | -82.50 | 30 | 33.33% | 0.00% | 2.71 | 28.96 | 30.00 |
| Baseline-RoundRobin | -975.00 | -325.00 | -657.50 | -112.50 | 35 | 31.43% | 0.00% | 1.25 | 29.11 | 30.86 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -825.00 | -275.00 | -577.50 | -122.50 | 30 | 40.00% | 0.00% | 2.29 | 29.79 | 30.00 |
| Baseline-Random-seed-44 | -775.00 | -258.33 | -552.50 | -110.00 | 30 | 40.00% | 0.00% | 3.33 | 29.17 | 30.00 |
| Ablation-ME-CausalShadowARS-Equal | -1297.50 | -432.50 | -925.00 | -177.50 | 49 | 38.78% | 0.00% | 2.81 | 29.29 | 30.61 |
| Ablation-ME-CausalShadowARS-Single-10 | -1365.00 | -455.00 | -992.50 | -177.50 | 51 | 37.25% | 0.00% | 2.57 | 29.34 | 30.59 |
| Ablation-ME-CausalShadowARS-Single-30 | -1227.50 | -409.17 | -870.00 | -177.50 | 46 | 39.13% | 0.00% | 2.72 | 29.40 | 30.65 |
| Ablation-ME-CausalShadowARS-Single-90 | -1050.00 | -350.00 | -745.00 | -127.50 | 40 | 42.50% | 0.00% | 3.12 | 29.38 | 30.00 |
| Ablation-OE-CausalShadowARS-Equal | -1130.00 | -376.67 | -825.00 | -127.50 | 42 | 38.10% | 0.00% | 2.38 | 29.29 | 30.71 |
| Ablation-OE-CausalShadowARS-Single-10 | -922.50 | -307.50 | -617.50 | -127.50 | 34 | 38.24% | 0.00% | 2.21 | 29.34 | 30.00 |
| Ablation-OE-CausalShadowARS-Single-30 | -1032.50 | -344.17 | -647.50 | -177.50 | 38 | 36.84% | 0.00% | 2.30 | 29.47 | 30.00 |
| Ablation-OE-CausalShadowARS-Single-90 | -1197.50 | -399.17 | -865.00 | -155.00 | 44 | 34.09% | 0.00% | 2.13 | 29.35 | 30.68 |
| Sensitivity-ME-SignedBox | -1160.00 | -386.67 | -870.00 | -112.50 | 44 | 40.91% | 0.00% | 2.98 | 29.35 | 30.68 |
| Sensitivity-OE-SignedBox | -1142.50 | -380.83 | -825.00 | -140.00 | 42 | 30.95% | 0.00% | 1.93 | 29.14 | 30.71 |
| Sensitivity-NetOE-FixedWeights | -510.00 | -170.00 | -220.00 | -112.50 | 19 | 26.32% | 0.00% | 1.64 | 28.49 | 30.00 |

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

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 7 | 30.00 | 30.00 | 30.00 | -2.68 | 28.39 | 14.29% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Variant-OE-EventUCB | 7 | 30.00 | 30.00 | 30.00 | -0.89 | 28.39 | 14.29% | 0.00% |
| Variant-OE-CausalShadowARS | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 7 | 30.00 | 30.00 | 30.00 | -0.89 | 28.39 | 14.29% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 3 | 30.00 | 30.00 | 30.00 | 2.08 | 29.58 | 33.33% | 0.00% |
| Baseline-RoundRobin | 7 | 30.00 | 30.00 | 30.00 | -0.89 | 28.39 | 14.29% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 30.00 | 30.00 | 30.00 | -3.12 | 27.50 | 0.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 30.00 | 30.00 | 30.00 | 1.56 | 29.06 | 25.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Sensitivity-OE-SignedBox | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 6 | 30.00 | 30.00 | 30.00 | -1.04 | 28.54 | 16.67% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 34.35% | 53.46% | N/A | 0.00% | N/A | 99.17 / 127.25 |
| Logistic_Direction | 34.26% | 53.46% | N/A | 0.00% | 56.35% | 143.26 / 173.42 |
| Decision_Tree | 33.99% | 53.46% | N/A | 0.00% | N/A | 112.56 / 148.19 |
| Hist_GBDT | 34.69% | 53.46% | N/A | 0.00% | N/A | 369.00 / 421.71 |
| Rule_Momentum | 31.85% | 53.38% | 21.88% | 0.27% | N/A | 6.94 / 7.33 |

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

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 39 | 30.77 | 30.00 | 30.00 | 2.56 | 30.06 | 41.03% | 0.00% |
| Variant-ME-CausalShadowARS | 33 | 30.91 | 30.00 | 30.00 | 3.41 | 29.77 | 45.45% | 0.00% |
| Variant-OE-EventUCB | 1 | 30.00 | 30.00 | 30.00 | 0.00 | 27.50 | 0.00% | 0.00% |
| Variant-OE-CausalShadowARS | 31 | 30.97 | 30.00 | 30.00 | 2.02 | 29.92 | 35.48% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 46 | 31.30 | 30.00 | 30.00 | 2.04 | 30.08 | 39.13% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 19 | 30.00 | 30.00 | 30.00 | 2.63 | 29.47 | 36.84% | 0.00% |
| Baseline-RoundRobin | 23 | 31.30 | 30.00 | 30.00 | 1.09 | 29.67 | 34.78% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 21 | 30.00 | 30.00 | 30.00 | 2.98 | 30.48 | 47.62% | 0.00% |
| Baseline-Random-seed-44 | 21 | 30.00 | 30.00 | 30.00 | 3.27 | 29.58 | 42.86% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 35 | 30.86 | 30.00 | 30.00 | 3.21 | 29.64 | 42.86% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 37 | 30.81 | 30.00 | 30.00 | 2.87 | 29.70 | 40.54% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 33 | 30.91 | 30.00 | 30.00 | 3.41 | 29.77 | 45.45% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 28 | 30.00 | 30.00 | 30.00 | 3.35 | 29.96 | 46.43% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 30 | 31.00 | 30.00 | 30.00 | 2.29 | 29.79 | 40.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 22 | 30.00 | 30.00 | 30.00 | 1.99 | 30.06 | 40.91% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 24 | 30.00 | 30.00 | 30.00 | 3.12 | 30.10 | 45.83% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 31 | 30.97 | 30.00 | 30.00 | 2.02 | 29.92 | 35.48% | 0.00% |
| Sensitivity-ME-SignedBox | 33 | 30.91 | 30.00 | 30.00 | 3.41 | 29.77 | 45.45% | 0.00% |
| Sensitivity-OE-SignedBox | 30 | 31.00 | 30.00 | 30.00 | 2.08 | 29.58 | 33.33% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 8 | 30.00 | 30.00 | 30.00 | 1.56 | 29.06 | 25.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.16% | 75.45% | N/A | 0.00% | N/A | 98.65 / 125.65 |
| Logistic_Direction | 22.73% | 75.90% | 54.19% | 1.50% | 75.75% | 142.55 / 171.69 |
| Decision_Tree | 21.53% | 75.45% | N/A | 0.00% | N/A | 113.17 / 139.87 |
| Hist_GBDT | 22.74% | 75.45% | N/A | 0.00% | N/A | 369.76 / 392.40 |
| Rule_Momentum | 21.47% | 75.45% | N/A | 0.00% | N/A | 7.00 / 7.16 |

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

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 30.00 | 30.00 | 30.00 | 3.12 | 28.54 | 33.33% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 30.00 | 30.00 | 30.00 | 6.25 | 27.50 | 50.00% | 0.00% |
| Variant-OE-EventUCB | 4 | 30.00 | 30.00 | 30.00 | 3.12 | 27.50 | 25.00% | 0.00% |
| Variant-OE-CausalShadowARS | 7 | 30.00 | 30.00 | 30.00 | 5.36 | 27.50 | 42.86% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 14 | 30.00 | 30.00 | 30.00 | 4.46 | 29.29 | 50.00% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 8 | 30.00 | 30.00 | 30.00 | 3.12 | 27.50 | 25.00% | 0.00% |
| Baseline-RoundRobin | 5 | 30.00 | 30.00 | 30.00 | 5.00 | 27.50 | 40.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 5 | 30.00 | 30.00 | 30.00 | 3.75 | 28.75 | 40.00% | 0.00% |
| Baseline-Random-seed-44 | 5 | 30.00 | 30.00 | 30.00 | 5.00 | 27.50 | 40.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 8 | 30.00 | 30.00 | 30.00 | 3.91 | 28.28 | 37.50% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 8 | 30.00 | 30.00 | 30.00 | 3.91 | 28.28 | 37.50% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 7 | 30.00 | 30.00 | 30.00 | 2.68 | 28.39 | 28.57% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 30.00 | 30.00 | 30.00 | 6.25 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 6 | 30.00 | 30.00 | 30.00 | 6.25 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 6 | 30.00 | 30.00 | 30.00 | 6.25 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 8 | 30.00 | 30.00 | 30.00 | 2.34 | 28.28 | 25.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 7 | 30.00 | 30.00 | 30.00 | 5.36 | 27.50 | 42.86% | 0.00% |
| Sensitivity-ME-SignedBox | 5 | 30.00 | 30.00 | 30.00 | 5.00 | 27.50 | 40.00% | 0.00% |
| Sensitivity-OE-SignedBox | 6 | 30.00 | 30.00 | 30.00 | 4.17 | 27.50 | 33.33% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 5 | 30.00 | 30.00 | 30.00 | 5.00 | 27.50 | 40.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 16.96% | 82.34% | N/A | 0.00% | N/A | 99.80 / 128.40 |
| Logistic_Direction | 17.23% | 82.52% | 83.33% | 0.25% | 84.82% | 143.72 / 171.45 |
| Decision_Tree | 16.77% | 82.34% | N/A | 0.00% | N/A | 114.36 / 142.45 |
| Hist_GBDT | 17.18% | 82.34% | N/A | 0.00% | N/A | 365.28 / 387.60 |
| Rule_Momentum | 16.24% | 82.34% | N/A | 0.00% | N/A | 7.03 / 7.31 |

零信号准确率 82.34%；训练多数类准确率 82.34%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0501033618, '0.5': 0.676730374, '0.9': 1.5287187415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：16/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。
## ICE_BRENT：文件内多窗口描述统计（非独立重复实验）

数据 `databento_ifeu.impact_mbp_10.parquet`，共 2,292,832 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -4175.00 | -1391.67 | -4004.00 | -23.00 | 155 | 30.32% | 0.00% | -0.39 | 26.55 | 61.35 |
| Variant-ME-CausalShadowARS | -1907.00 | -635.67 | -1703.00 | -23.00 | 69 | 28.99% | 0.00% | -0.51 | 27.13 | 73.48 |
| Variant-OE-EventUCB | -3118.00 | -1039.33 | -2990.00 | -23.00 | 116 | 42.24% | 0.00% | 0.65 | 27.53 | 81.98 |
| Variant-OE-CausalShadowARS | -2169.00 | -723.00 | -1965.00 | -23.00 | 73 | 35.62% | 0.00% | -0.62 | 29.10 | 63.85 |
| Baseline-Single-Ridge_Linear | -572.00 | -190.67 | -516.00 | 0.00 | 14 | 35.71% | 0.00% | 0.71 | 41.57 | 28.07 |
| Baseline-Single-Logistic_Direction | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Decision_Tree | -3096.00 | -1032.00 | -3096.00 | 0.00 | 102 | 42.16% | 0.00% | -0.83 | 29.52 | 93.64 |
| Baseline-Single-Hist_GBDT | -3321.00 | -1107.00 | -3130.00 | -46.00 | 117 | 30.77% | 0.00% | -0.98 | 27.40 | 77.44 |
| Baseline-Single-Rule_Momentum | -593.00 | -197.67 | -396.00 | 0.00 | 21 | 42.86% | 0.00% | 0.24 | 28.48 | 31.43 |
| Baseline-Static-Ensemble | -2863.00 | -954.33 | -2863.00 | 0.00 | 101 | 42.57% | 0.00% | -0.25 | 28.10 | 94.46 |
| Baseline-Random | -5630.00 | -1876.67 | -5508.00 | -23.00 | 220 | 19.09% | 0.00% | -0.55 | 25.05 | 41.05 |
| Baseline-RoundRobin | -2756.00 | -918.67 | -2585.00 | -23.00 | 102 | 44.12% | 0.98% | 0.69 | 27.71 | 92.94 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -5659.00 | -1886.33 | -5557.00 | 0.00 | 223 | 21.97% | 0.00% | -0.04 | 25.33 | 39.82 |
| Baseline-Random-seed-44 | -5954.00 | -1984.67 | -5724.00 | -46.00 | 228 | 21.93% | 0.00% | -0.33 | 25.79 | 40.18 |
| Ablation-ME-CausalShadowARS-Equal | -1887.00 | -629.00 | -1683.00 | -23.00 | 69 | 31.88% | 0.00% | -0.29 | 27.06 | 76.09 |
| Ablation-ME-CausalShadowARS-Single-10 | -2021.00 | -673.67 | -1817.00 | -23.00 | 67 | 37.31% | 0.00% | -0.45 | 29.72 | 67.66 |
| Ablation-ME-CausalShadowARS-Single-30 | -2206.00 | -735.33 | -2025.00 | -23.00 | 72 | 26.39% | 0.00% | -1.60 | 29.04 | 71.40 |
| Ablation-ME-CausalShadowARS-Single-90 | -1907.00 | -635.67 | -1703.00 | -23.00 | 69 | 28.99% | 0.00% | -0.51 | 27.13 | 73.48 |
| Ablation-OE-CausalShadowARS-Equal | -2225.00 | -741.67 | -2021.00 | -23.00 | 85 | 36.47% | 0.00% | 0.59 | 26.76 | 63.18 |
| Ablation-OE-CausalShadowARS-Single-10 | -1965.00 | -655.00 | -1761.00 | -23.00 | 65 | 36.92% | 0.00% | -0.62 | 29.62 | 58.32 |
| Ablation-OE-CausalShadowARS-Single-30 | -2169.00 | -723.00 | -1965.00 | -23.00 | 73 | 35.62% | 0.00% | -0.62 | 29.10 | 63.85 |
| Ablation-OE-CausalShadowARS-Single-90 | -2719.00 | -906.33 | -2515.00 | -23.00 | 93 | 32.26% | 0.00% | -0.91 | 28.32 | 58.51 |
| Sensitivity-ME-SignedBox | -1943.00 | -647.67 | -1739.00 | -23.00 | 71 | 30.99% | 0.00% | -0.28 | 27.08 | 72.25 |
| Sensitivity-OE-SignedBox | -2266.00 | -755.33 | -2062.00 | -23.00 | 82 | 31.71% | 0.00% | -0.49 | 27.15 | 68.78 |
| Sensitivity-NetOE-FixedWeights | -753.00 | -251.00 | -572.00 | -23.00 | 21 | 38.10% | 0.00% | -0.71 | 35.14 | 34.43 |

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

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Variant-ME-CausalShadowARS | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Variant-OE-EventUCB | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Variant-OE-CausalShadowARS | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 2 | 30.00 | 30.00 | 30.00 | 2.50 | 25.50 | 50.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Baseline-RoundRobin | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-44 | 2 | 30.00 | 30.00 | 30.00 | 2.50 | 25.50 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Sensitivity-ME-SignedBox | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Sensitivity-OE-SignedBox | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 1 | 30.00 | 30.00 | 30.00 | 5.00 | 28.00 | 100.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 33.79% | 52.78% | N/A | 0.00% | N/A | 98.96 / 125.35 |
| Logistic_Direction | 33.79% | 52.78% | N/A | 0.00% | 51.81% | 142.60 / 171.31 |
| Decision_Tree | 35.29% | 52.78% | N/A | 0.00% | N/A | 113.95 / 140.16 |
| Hist_GBDT | 32.42% | 52.83% | 100.00% | 0.05% | N/A | 366.25 / 387.94 |
| Rule_Momentum | 34.12% | 52.78% | N/A | 0.00% | N/A | 7.05 / 7.48 |

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
| OE / signed_box | 1.0000 / -0.0000 / -0.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 30.00 | 30.00 | 30.00 | 2.50 | 27.17 | 66.67% | 0.00% |
| Variant-ME-CausalShadowARS | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Variant-OE-EventUCB | 5 | 36.00 | 30.00 | 48.00 | 5.00 | 26.00 | 60.00% | 0.00% |
| Variant-OE-CausalShadowARS | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Baseline-Single-Ridge_Linear | 2 | 30.00 | 30.00 | 30.00 | -2.50 | 25.50 | 0.00% | 0.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 5 | 36.00 | 30.00 | 48.00 | -4.00 | 25.00 | 20.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 9 | 33.33 | 30.00 | 36.00 | 3.33 | 25.22 | 44.44% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 3 | 30.00 | 30.00 | 30.00 | -10.00 | 23.00 | 0.00% | 0.00% |
| Baseline-RoundRobin | 6 | 30.00 | 30.00 | 30.00 | 1.67 | 26.33 | 50.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 30.00 | 30.00 | 30.00 | 1.25 | 26.75 | 50.00% | 0.00% |
| Baseline-Random-seed-44 | 8 | 30.00 | 30.00 | 30.00 | 1.88 | 24.87 | 37.50% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 30.00 | 30.00 | 30.00 | 0.00 | 26.33 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Sensitivity-ME-SignedBox | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Sensitivity-OE-SignedBox | 7 | 30.00 | 30.00 | 30.00 | 0.00 | 25.86 | 28.57% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 6 | 30.00 | 30.00 | 30.00 | 0.00 | 26.33 | 33.33% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.14% | 70.70% | 100.00% | 0.03% | N/A | 98.52 / 124.33 |
| Logistic_Direction | 23.04% | 70.67% | N/A | 0.00% | 71.03% | 142.11 / 170.44 |
| Decision_Tree | 22.25% | 70.67% | N/A | 0.00% | N/A | 114.53 / 140.73 |
| Hist_GBDT | 23.24% | 70.66% | 47.62% | 0.18% | N/A | 373.10 / 398.88 |
| Rule_Momentum | 22.23% | 70.63% | 15.00% | 0.17% | N/A | 7.09 / 7.27 |

零信号准确率 70.67%；训练多数类准确率 70.67%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0211009952, '0.5': 0.6737915005, '0.9': 2.4399680977}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：2/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：2/True；Baseline-Single-Hist_GBDT：2/True；Baseline-Single-Rule_Momentum：8/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-06-09 20:31:36.437188002+00:00 → 2025-06-09 22:00:00.245000+00:00，5303.81 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00012`。固定模型：选中 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.8`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.023333 |
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | False | -0.000274 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.023333 |
| OE / signed_box | 0.4572 / 1.0000 / -0.4572 | True | False | -0.000274 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 148 | 62.84 | 30.00 | 150.00 | -0.54 | 26.51 | 28.38% | 0.00% |
| Variant-ME-CausalShadowARS | 61 | 79.18 | 60.00 | 150.00 | -0.66 | 27.26 | 27.87% | 0.00% |
| Variant-OE-EventUCB | 110 | 84.55 | 60.00 | 153.00 | 0.41 | 27.59 | 40.91% | 0.00% |
| Variant-OE-CausalShadowARS | 65 | 68.02 | 30.00 | 150.00 | -0.77 | 29.46 | 35.38% | 0.00% |
| Baseline-Single-Ridge_Linear | 12 | 27.75 | 30.00 | 30.00 | 1.25 | 44.25 | 41.67% | 0.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 102 | 93.64 | 60.00 | 180.00 | -0.83 | 29.52 | 42.16% | 0.00% |
| Baseline-Single-Hist_GBDT | 110 | 80.18 | 60.00 | 180.00 | -0.91 | 27.55 | 30.91% | 0.00% |
| Baseline-Single-Rule_Momentum | 12 | 30.00 | 30.00 | 30.00 | -2.08 | 30.92 | 41.67% | 0.00% |
| Baseline-Static-Ensemble | 101 | 94.46 | 60.00 | 180.00 | -0.25 | 28.10 | 42.57% | 0.00% |
| Baseline-Random | 216 | 41.25 | 30.00 | 60.00 | -0.44 | 25.06 | 18.98% | 0.00% |
| Baseline-RoundRobin | 95 | 97.58 | 60.00 | 198.00 | 0.58 | 27.79 | 43.16% | 1.05% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 219 | 40.00 | 30.00 | 60.00 | -0.07 | 25.31 | 21.46% | 0.00% |
| Baseline-Random-seed-44 | 218 | 40.64 | 30.00 | 60.00 | -0.44 | 25.82 | 21.10% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 61 | 82.13 | 60.00 | 180.00 | -0.41 | 27.18 | 31.15% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 59 | 72.76 | 30.00 | 156.00 | -0.59 | 30.20 | 37.29% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 65 | 75.86 | 60.00 | 180.00 | -1.85 | 29.31 | 24.62% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 61 | 79.18 | 60.00 | 150.00 | -0.66 | 27.26 | 27.87% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 77 | 66.62 | 30.00 | 132.00 | 0.58 | 26.83 | 36.36% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 57 | 62.30 | 30.00 | 120.00 | -0.79 | 30.11 | 36.84% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 65 | 68.02 | 30.00 | 150.00 | -0.77 | 29.46 | 35.38% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 85 | 61.19 | 30.00 | 108.00 | -1.06 | 28.53 | 31.76% | 0.00% |
| Sensitivity-ME-SignedBox | 63 | 77.62 | 60.00 | 150.00 | -0.40 | 27.21 | 30.16% | 0.00% |
| Sensitivity-OE-SignedBox | 74 | 72.97 | 30.00 | 171.00 | -0.61 | 27.26 | 31.08% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 14 | 36.64 | 30.00 | 51.00 | -1.43 | 39.43 | 35.71% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 28.76% | 60.44% | 80.56% | 0.30% | N/A | 99.03 / 123.85 |
| Logistic_Direction | 29.09% | 60.25% | N/A | 0.00% | 60.75% | 143.55 / 172.72 |
| Decision_Tree | 22.05% | 26.17% | 14.18% | 61.16% | N/A | 114.05 / 142.20 |
| Hist_GBDT | 22.50% | 28.22% | 12.33% | 55.18% | N/A | 368.46 / 407.52 |
| Rule_Momentum | 27.49% | 60.36% | 30.51% | 0.49% | N/A | 7.15 / 7.38 |

零信号准确率 60.25%；训练多数类准确率 60.25%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.035222500000000004, '0.5': 5.053657501, '0.9': 23.3169947982}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：4/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：73/True；Baseline-Single-Hist_GBDT：68/True；Baseline-Single-Rule_Momentum：12/True；Baseline-Static-Ensemble：73/True；Baseline-Cash：0/False。

## 解释限制与后续工作

求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。

当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。
