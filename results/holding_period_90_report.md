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

种子 42；每品种 3 个窗口；每窗 60,000 条原始事件；holding period 90 events；Python 3.12.14；计算线程 1。

JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。

## CME_ES：文件内多窗口描述统计（非独立重复实验）

数据 `databento_glbx.mdp3_mbp_10.parquet`，共 2,807,571 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1362.50 | -454.17 | -955.00 | -177.50 | 45 | 40.00% | 0.00% | -0.56 | 29.72 | 90.00 |
| Variant-ME-CausalShadowARS | -1165.00 | -388.33 | -835.00 | -127.50 | 41 | 43.90% | 0.00% | 0.76 | 29.18 | 90.00 |
| Variant-OE-EventUCB | -670.00 | -223.33 | -410.00 | -82.50 | 23 | 30.43% | 0.00% | -0.54 | 28.59 | 90.00 |
| Variant-OE-CausalShadowARS | -945.00 | -315.00 | -682.50 | -112.50 | 33 | 42.42% | 0.00% | 0.57 | 29.20 | 90.00 |
| Baseline-Single-Ridge_Linear | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Logistic_Direction | -1462.50 | -487.50 | -1117.50 | 0.00 | 50 | 44.00% | 0.00% | 0.88 | 30.12 | 90.00 |
| Baseline-Single-Decision_Tree | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Hist_GBDT | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Rule_Momentum | -230.00 | -76.67 | -230.00 | 0.00 | 7 | 42.86% | 0.00% | -4.46 | 28.39 | 90.00 |
| Baseline-Static-Ensemble | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random | -715.00 | -238.33 | -452.50 | -70.00 | 26 | 38.46% | 0.00% | 1.20 | 28.70 | 90.00 |
| Baseline-RoundRobin | -930.00 | -310.00 | -587.50 | -112.50 | 32 | 37.50% | 0.00% | -0.39 | 28.67 | 90.00 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -782.50 | -260.83 | -560.00 | -110.00 | 28 | 50.00% | 0.00% | 1.79 | 29.73 | 90.00 |
| Baseline-Random-seed-44 | -795.00 | -265.00 | -572.50 | -97.50 | 28 | 35.71% | 0.00% | 0.89 | 29.29 | 90.00 |
| Ablation-ME-CausalShadowARS-Equal | -1275.00 | -425.00 | -877.50 | -195.00 | 45 | 42.22% | 0.00% | 0.83 | 29.17 | 90.00 |
| Ablation-ME-CausalShadowARS-Single-10 | -1327.50 | -442.50 | -930.00 | -195.00 | 46 | 39.13% | 0.00% | 0.41 | 29.27 | 90.00 |
| Ablation-ME-CausalShadowARS-Single-30 | -1217.50 | -405.83 | -835.00 | -180.00 | 42 | 40.48% | 0.00% | 0.30 | 29.29 | 90.00 |
| Ablation-ME-CausalShadowARS-Single-90 | -1067.50 | -355.83 | -737.50 | -127.50 | 37 | 45.95% | 0.00% | 0.51 | 29.36 | 90.00 |
| Ablation-OE-CausalShadowARS-Equal | -917.50 | -305.83 | -655.00 | -112.50 | 32 | 43.75% | 0.00% | 0.39 | 29.06 | 90.00 |
| Ablation-OE-CausalShadowARS-Single-10 | -820.00 | -273.33 | -557.50 | -112.50 | 28 | 35.71% | 0.00% | -0.22 | 29.06 | 90.00 |
| Ablation-OE-CausalShadowARS-Single-30 | -902.50 | -300.83 | -655.00 | -97.50 | 31 | 41.94% | 0.00% | 0.20 | 29.31 | 90.00 |
| Ablation-OE-CausalShadowARS-Single-90 | -835.00 | -278.33 | -572.50 | -112.50 | 29 | 41.38% | 0.00% | 0.43 | 29.22 | 90.00 |
| Sensitivity-ME-SignedBox | -1150.00 | -383.33 | -835.00 | -112.50 | 40 | 42.50% | 0.00% | 0.47 | 29.22 | 90.00 |
| Sensitivity-OE-SignedBox | -957.50 | -319.17 | -710.00 | -97.50 | 33 | 39.39% | 0.00% | 0.19 | 29.20 | 90.00 |
| Sensitivity-NetOE-FixedWeights | -452.50 | -150.83 | -205.00 | -97.50 | 16 | 37.50% | 0.00% | 0.00 | 28.28 | 90.00 |

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
| Variant-ME-EventUCB | 7 | 90.00 | 90.00 | 90.00 | -4.46 | 28.39 | 42.86% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Variant-OE-EventUCB | 6 | 90.00 | 90.00 | 90.00 | -1.04 | 28.54 | 50.00% | 0.00% |
| Variant-OE-CausalShadowARS | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 7 | 90.00 | 90.00 | 90.00 | -4.46 | 28.39 | 42.86% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 3 | 90.00 | 90.00 | 90.00 | 4.17 | 27.50 | 66.67% | 0.00% |
| Baseline-RoundRobin | 7 | 90.00 | 90.00 | 90.00 | -4.46 | 28.39 | 42.86% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 90.00 | 90.00 | 90.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 90.00 | 90.00 | 90.00 | 4.69 | 29.06 | 75.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 90.00 | 90.00 | 90.00 | -6.25 | 27.50 | 33.33% | 0.00% |
| Sensitivity-OE-SignedBox | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 5 | 90.00 | 90.00 | 90.00 | -2.50 | 27.50 | 40.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 34.35% | 53.46% | N/A | 0.00% | N/A | 100.76 / 129.81 |
| Logistic_Direction | 34.26% | 53.46% | N/A | 0.00% | 56.35% | 145.48 / 172.97 |
| Decision_Tree | 33.99% | 53.46% | N/A | 0.00% | N/A | 116.89 / 143.66 |
| Hist_GBDT | 34.69% | 53.46% | N/A | 0.00% | N/A | 375.87 / 399.75 |
| Rule_Momentum | 31.85% | 53.38% | 21.88% | 0.27% | N/A | 7.23 / 7.84 |

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
| OE / simplex | 0.0000 / 0.9697 / 0.0303 | True | False | -0.005682 |
| ME / signed_box | -0.1828 / 1.0000 / 0.1828 | True | False | -0.047845 |
| OE / signed_box | -0.0275 / 1.0000 / 0.0275 | True | False | -0.005161 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 32 | 90.00 | 90.00 | 90.00 | 0.39 | 30.23 | 40.62% | 0.00% |
| Variant-ME-CausalShadowARS | 29 | 90.00 | 90.00 | 90.00 | 1.08 | 29.87 | 41.38% | 0.00% |
| Variant-OE-EventUCB | 14 | 90.00 | 90.00 | 90.00 | -0.45 | 28.84 | 21.43% | 0.00% |
| Variant-OE-CausalShadowARS | 23 | 90.00 | 90.00 | 90.00 | 0.27 | 29.95 | 39.13% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 37 | 90.00 | 90.00 | 90.00 | 0.17 | 30.37 | 40.54% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 16 | 90.00 | 90.00 | 90.00 | 1.17 | 29.45 | 37.50% | 0.00% |
| Baseline-RoundRobin | 20 | 90.00 | 90.00 | 90.00 | -0.31 | 29.06 | 30.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 19 | 90.00 | 90.00 | 90.00 | 0.99 | 30.46 | 47.37% | 0.00% |
| Baseline-Random-seed-44 | 19 | 90.00 | 90.00 | 90.00 | -0.33 | 29.80 | 26.32% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 31 | 90.00 | 90.00 | 90.00 | 1.41 | 29.72 | 41.94% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 32 | 90.00 | 90.00 | 90.00 | 0.78 | 29.84 | 37.50% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 29 | 90.00 | 90.00 | 90.00 | 1.08 | 29.87 | 41.38% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 25 | 90.00 | 90.00 | 90.00 | 0.75 | 30.25 | 44.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 22 | 90.00 | 90.00 | 90.00 | 0.00 | 29.77 | 40.91% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 18 | 90.00 | 90.00 | 90.00 | -1.04 | 29.93 | 27.78% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 22 | 90.00 | 90.00 | 90.00 | 0.28 | 30.06 | 40.91% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 19 | 90.00 | 90.00 | 90.00 | 0.00 | 30.13 | 36.84% | 0.00% |
| Sensitivity-ME-SignedBox | 29 | 90.00 | 90.00 | 90.00 | 1.08 | 29.87 | 41.38% | 0.00% |
| Sensitivity-OE-SignedBox | 24 | 90.00 | 90.00 | 90.00 | 0.26 | 29.84 | 37.50% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 7 | 90.00 | 90.00 | 90.00 | 0.00 | 29.29 | 28.57% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.16% | 75.45% | N/A | 0.00% | N/A | 99.86 / 124.91 |
| Logistic_Direction | 22.73% | 75.90% | 54.19% | 1.50% | 75.75% | 144.01 / 173.50 |
| Decision_Tree | 21.53% | 75.45% | N/A | 0.00% | N/A | 115.15 / 141.09 |
| Hist_GBDT | 22.74% | 75.45% | N/A | 0.00% | N/A | 373.32 / 396.07 |
| Rule_Momentum | 21.47% | 75.45% | N/A | 0.00% | N/A | 7.14 / 7.32 |

零信号准确率 75.45%；训练多数类准确率 75.45%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.04134640420000002, '0.5': 0.33158887800000003, '0.9': 0.7446652415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：54/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：2/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-09-22 15:58:06.928056965+00:00 → 2025-09-22 15:59:59.951415627+00:00，113.02 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.080357 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.026786 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.066964 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.017857 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 90.00 | 90.00 | 90.00 | -1.04 | 28.54 | 33.33% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 90.00 | 90.00 | 90.00 | 6.25 | 27.50 | 66.67% | 0.00% |
| Variant-OE-EventUCB | 3 | 90.00 | 90.00 | 90.00 | 0.00 | 27.50 | 33.33% | 0.00% |
| Variant-OE-CausalShadowARS | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 13 | 90.00 | 90.00 | 90.00 | 2.88 | 29.42 | 53.85% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 7 | 90.00 | 90.00 | 90.00 | 0.00 | 27.50 | 28.57% | 0.00% |
| Baseline-RoundRobin | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 5 | 90.00 | 90.00 | 90.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Baseline-Random-seed-44 | 5 | 90.00 | 90.00 | 90.00 | 2.50 | 27.50 | 40.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 8 | 90.00 | 90.00 | 90.00 | 3.91 | 28.28 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 8 | 90.00 | 90.00 | 90.00 | 3.91 | 28.28 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 7 | 90.00 | 90.00 | 90.00 | 2.68 | 28.39 | 42.86% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 90.00 | 90.00 | 90.00 | 6.25 | 27.50 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 4 | 90.00 | 90.00 | 90.00 | 3.12 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Sensitivity-ME-SignedBox | 5 | 90.00 | 90.00 | 90.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Sensitivity-OE-SignedBox | 4 | 90.00 | 90.00 | 90.00 | 3.12 | 27.50 | 50.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 4 | 90.00 | 90.00 | 90.00 | 3.12 | 27.50 | 50.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 16.96% | 82.34% | N/A | 0.00% | N/A | 101.15 / 126.79 |
| Logistic_Direction | 17.23% | 82.52% | 83.33% | 0.25% | 84.82% | 146.89 / 257.94 |
| Decision_Tree | 16.77% | 82.34% | N/A | 0.00% | N/A | 115.83 / 146.25 |
| Hist_GBDT | 17.18% | 82.34% | N/A | 0.00% | N/A | 372.14 / 403.33 |
| Rule_Momentum | 16.24% | 82.34% | N/A | 0.00% | N/A | 7.20 / 7.63 |

零信号准确率 82.34%；训练多数类准确率 82.34%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0501033618, '0.5': 0.676730374, '0.9': 1.5287187415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：14/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。
## ICE_BRENT：文件内多窗口描述统计（非独立重复实验）

数据 `databento_ifeu.impact_mbp_10.parquet`，共 2,292,832 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -850.00 | -283.33 | -719.00 | -13.00 | 30 | 43.33% | 0.00% | 1.50 | 29.83 | 131.13 |
| Variant-ME-CausalShadowARS | -1020.00 | -340.00 | -856.00 | -13.00 | 40 | 30.00% | 2.50% | 0.50 | 26.00 | 166.50 |
| Variant-OE-EventUCB | -1571.00 | -523.67 | -1453.00 | -13.00 | 57 | 43.86% | 0.00% | -0.61 | 26.95 | 194.21 |
| Variant-OE-CausalShadowARS | -1442.00 | -480.67 | -1278.00 | -13.00 | 44 | 36.36% | 0.00% | -3.86 | 28.91 | 140.48 |
| Baseline-Single-Ridge_Linear | -486.00 | -162.00 | -440.00 | 0.00 | 12 | 41.67% | 8.33% | -2.92 | 37.58 | 87.58 |
| Baseline-Single-Logistic_Direction | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Decision_Tree | -1447.00 | -482.33 | -1447.00 | 0.00 | 49 | 46.94% | 0.00% | -1.12 | 28.41 | 216.73 |
| Baseline-Single-Hist_GBDT | -1486.00 | -495.33 | -1325.00 | -36.00 | 52 | 36.54% | 0.00% | -1.83 | 26.75 | 214.62 |
| Baseline-Single-Rule_Momentum | -547.00 | -182.33 | -340.00 | 0.00 | 19 | 47.37% | 5.26% | 5.26 | 34.05 | 88.63 |
| Baseline-Static-Ensemble | -1447.00 | -482.33 | -1447.00 | 0.00 | 49 | 46.94% | 0.00% | -1.12 | 28.41 | 216.73 |
| Baseline-Random | -2423.00 | -807.67 | -2334.00 | -23.00 | 91 | 34.07% | 0.00% | -0.93 | 25.69 | 117.69 |
| Baseline-RoundRobin | -1476.00 | -492.00 | -1335.00 | -13.00 | 52 | 51.92% | 0.00% | 1.15 | 29.54 | 214.04 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -2301.00 | -767.00 | -2199.00 | 0.00 | 87 | 36.78% | 0.00% | -0.63 | 25.82 | 122.07 |
| Baseline-Random-seed-44 | -2732.00 | -910.67 | -2545.00 | -36.00 | 94 | 32.98% | 0.00% | -1.54 | 27.52 | 120.67 |
| Ablation-ME-CausalShadowARS-Equal | -954.00 | -318.00 | -790.00 | -13.00 | 38 | 36.84% | 2.63% | 1.32 | 26.42 | 180.00 |
| Ablation-ME-CausalShadowARS-Single-10 | -1372.00 | -457.33 | -1208.00 | -13.00 | 44 | 38.64% | 0.00% | -0.68 | 30.50 | 141.20 |
| Ablation-ME-CausalShadowARS-Single-30 | -1067.00 | -355.67 | -926.00 | -13.00 | 39 | 33.33% | 0.00% | -1.03 | 26.33 | 170.77 |
| Ablation-ME-CausalShadowARS-Single-90 | -1020.00 | -340.00 | -856.00 | -13.00 | 40 | 30.00% | 2.50% | 0.50 | 26.00 | 166.50 |
| Ablation-OE-CausalShadowARS-Equal | -1364.00 | -454.67 | -1200.00 | -13.00 | 48 | 45.83% | 2.08% | 0.21 | 28.62 | 149.40 |
| Ablation-OE-CausalShadowARS-Single-10 | -1227.00 | -409.00 | -1063.00 | -13.00 | 39 | 38.46% | 0.00% | -2.31 | 29.15 | 126.18 |
| Ablation-OE-CausalShadowARS-Single-30 | -1442.00 | -480.67 | -1278.00 | -13.00 | 44 | 36.36% | 0.00% | -3.86 | 28.91 | 140.48 |
| Ablation-OE-CausalShadowARS-Single-90 | -1513.00 | -504.33 | -1349.00 | -13.00 | 51 | 39.22% | 1.96% | -0.59 | 29.08 | 140.61 |
| Sensitivity-ME-SignedBox | -987.00 | -329.00 | -823.00 | -13.00 | 39 | 30.77% | 2.56% | 0.77 | 26.08 | 170.77 |
| Sensitivity-OE-SignedBox | -1504.00 | -501.33 | -1340.00 | -13.00 | 48 | 39.58% | 0.00% | -1.77 | 29.56 | 123.81 |
| Sensitivity-NetOE-FixedWeights | -686.00 | -228.67 | -522.00 | -13.00 | 22 | 36.36% | 4.55% | -0.68 | 30.50 | 109.14 |

### 窗口 1

测试 2025-06-09 01:22:55.205014002+00:00 → 2025-06-09 01:38:08.188182002+00:00，912.98 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00024`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Decision_Tree', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Logistic_Direction', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.011111 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.011111 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Variant-ME-CausalShadowARS | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Variant-OE-EventUCB | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Variant-OE-CausalShadowARS | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 2 | 90.00 | 90.00 | 90.00 | 5.00 | 23.00 | 50.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 1 | 90.00 | 90.00 | 90.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Baseline-RoundRobin | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-44 | 2 | 90.00 | 90.00 | 90.00 | 5.00 | 23.00 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Sensitivity-ME-SignedBox | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Sensitivity-OE-SignedBox | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 1 | 90.00 | 90.00 | 90.00 | 10.00 | 23.00 | 100.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 33.79% | 52.78% | N/A | 0.00% | N/A | 99.09 / 124.26 |
| Logistic_Direction | 33.79% | 52.78% | N/A | 0.00% | 51.81% | 145.29 / 177.19 |
| Decision_Tree | 35.29% | 52.78% | N/A | 0.00% | N/A | 115.14 / 141.48 |
| Hist_GBDT | 32.42% | 52.83% | 100.00% | 0.05% | N/A | 368.48 / 390.64 |
| Rule_Momentum | 34.12% | 52.78% | N/A | 0.00% | N/A | 7.12 / 7.36 |

零信号准确率 52.78%；训练多数类准确率 52.78%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.4946915994000001, '0.5': 5.168116001, '0.9': 22.5388601014}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：16/True；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-06-09 13:41:48.627059002+00:00 → 2025-06-09 13:44:10.342476002+00:00，141.72 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[0.00012, 0.00024]`，按预先声明顺序选 `0.00012`。固定模型：并列最佳 `['Logistic_Direction', 'Decision_Tree']`，按预先声明顺序选 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.8`；UCB-OE：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.022500 |
| OE / simplex | 0.2000 / 0.0000 / 0.8000 | True | False | -0.000500 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.015000 |
| OE / signed_box | 0.2500 / -0.0833 / 0.8333 | True | False | -0.000417 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 90.00 | 90.00 | 90.00 | 6.67 | 26.33 | 66.67% | 0.00% |
| Variant-ME-CausalShadowARS | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Variant-OE-EventUCB | 5 | 90.00 | 90.00 | 90.00 | 3.00 | 24.00 | 40.00% | 0.00% |
| Variant-OE-CausalShadowARS | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Baseline-Single-Ridge_Linear | 2 | 90.00 | 90.00 | 90.00 | 2.50 | 25.50 | 50.00% | 0.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 5 | 90.00 | 90.00 | 90.00 | -1.00 | 24.00 | 40.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 9 | 90.00 | 90.00 | 90.00 | 2.22 | 25.22 | 33.33% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 2 | 90.00 | 90.00 | 90.00 | -10.00 | 23.00 | 0.00% | 0.00% |
| Baseline-RoundRobin | 6 | 90.00 | 90.00 | 90.00 | 4.17 | 25.50 | 50.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 90.00 | 90.00 | 90.00 | -0.00 | 25.50 | 25.00% | 0.00% |
| Baseline-Random-seed-44 | 7 | 90.00 | 90.00 | 90.00 | 2.86 | 24.43 | 28.57% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 90.00 | 90.00 | 90.00 | 4.17 | 25.50 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Sensitivity-ME-SignedBox | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Sensitivity-OE-SignedBox | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 7 | 90.00 | 90.00 | 90.00 | 3.57 | 25.14 | 42.86% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.14% | 70.70% | 100.00% | 0.03% | N/A | 100.53 / 125.49 |
| Logistic_Direction | 23.04% | 70.67% | N/A | 0.00% | 71.03% | 145.33 / 172.67 |
| Decision_Tree | 22.25% | 70.67% | N/A | 0.00% | N/A | 115.89 / 141.96 |
| Hist_GBDT | 23.24% | 70.66% | 47.62% | 0.18% | N/A | 368.13 / 390.29 |
| Rule_Momentum | 22.23% | 70.63% | 15.00% | 0.17% | N/A | 7.20 / 7.41 |

零信号准确率 70.67%；训练多数类准确率 70.67%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0211009952, '0.5': 0.6737915005, '0.9': 2.4399680977}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：2/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：2/True；Baseline-Single-Hist_GBDT：2/True；Baseline-Single-Rule_Momentum：8/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-06-09 20:31:36.437188002+00:00 → 2025-06-09 22:00:00.245000+00:00，5303.81 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00012`。固定模型：选中 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.023333 |
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.023333 |
| OE / signed_box | 1.0000 / 1.0000 / -1.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 23 | 143.65 | 90.00 | 360.00 | -0.22 | 31.04 | 34.78% | 0.00% |
| Variant-ME-CausalShadowARS | 32 | 185.62 | 90.00 | 360.00 | -0.47 | 26.28 | 25.00% | 3.12% |
| Variant-OE-EventUCB | 51 | 206.47 | 180.00 | 360.00 | -1.18 | 27.31 | 43.14% | 0.00% |
| Variant-OE-CausalShadowARS | 36 | 151.69 | 90.00 | 315.00 | -5.69 | 29.81 | 33.33% | 0.00% |
| Baseline-Single-Ridge_Linear | 10 | 87.10 | 90.00 | 90.00 | -4.00 | 40.00 | 40.00% | 10.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 49 | 216.73 | 180.00 | 468.00 | -1.12 | 28.41 | 46.94% | 0.00% |
| Baseline-Single-Hist_GBDT | 45 | 234.00 | 180.00 | 450.00 | -2.22 | 27.22 | 35.56% | 0.00% |
| Baseline-Single-Rule_Momentum | 10 | 87.40 | 90.00 | 90.00 | 8.00 | 42.00 | 60.00% | 10.00% |
| Baseline-Static-Ensemble | 49 | 216.73 | 180.00 | 468.00 | -1.12 | 28.41 | 46.94% | 0.00% |
| Baseline-Random | 88 | 118.64 | 90.00 | 180.00 | -0.74 | 25.78 | 35.23% | 0.00% |
| Baseline-RoundRobin | 45 | 233.33 | 180.00 | 540.00 | 0.56 | 30.22 | 51.11% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 83 | 123.61 | 90.00 | 180.00 | -0.66 | 25.83 | 37.35% | 0.00% |
| Baseline-Random-seed-44 | 85 | 123.92 | 90.00 | 180.00 | -2.06 | 27.88 | 32.94% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 30 | 204.00 | 180.00 | 378.00 | 0.50 | 26.83 | 33.33% | 3.33% |
| Ablation-ME-CausalShadowARS-Single-10 | 36 | 152.58 | 90.00 | 360.00 | -1.81 | 31.75 | 36.11% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 32 | 188.44 | 90.00 | 360.00 | -2.34 | 26.59 | 28.12% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 32 | 185.62 | 90.00 | 360.00 | -0.47 | 26.28 | 25.00% | 3.12% |
| Ablation-OE-CausalShadowARS-Equal | 40 | 161.28 | 90.00 | 360.00 | -0.62 | 29.37 | 45.00% | 2.50% |
| Ablation-OE-CausalShadowARS-Single-10 | 31 | 135.52 | 90.00 | 270.00 | -4.03 | 30.26 | 35.48% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 36 | 151.69 | 90.00 | 315.00 | -5.69 | 29.81 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 43 | 150.02 | 90.00 | 342.00 | -1.51 | 29.86 | 37.21% | 2.33% |
| Sensitivity-ME-SignedBox | 31 | 191.61 | 90.00 | 360.00 | -0.16 | 26.39 | 25.81% | 3.23% |
| Sensitivity-OE-SignedBox | 40 | 130.57 | 90.00 | 189.00 | -3.00 | 30.50 | 37.50% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 14 | 120.07 | 90.00 | 180.00 | -3.57 | 33.71 | 28.57% | 7.14% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 28.76% | 60.44% | 80.56% | 0.30% | N/A | 99.25 / 126.34 |
| Logistic_Direction | 29.09% | 60.25% | N/A | 0.00% | 60.75% | 143.87 / 171.49 |
| Decision_Tree | 22.05% | 26.17% | 14.18% | 61.16% | N/A | 114.06 / 140.64 |
| Hist_GBDT | 22.50% | 28.22% | 12.33% | 55.18% | N/A | 372.56 / 391.71 |
| Rule_Momentum | 27.49% | 60.36% | 30.51% | 0.49% | N/A | 7.22 / 7.41 |

零信号准确率 60.25%；训练多数类准确率 60.25%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.035222500000000004, '0.5': 5.053657501, '0.9': 23.3169947982}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：4/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：41/True；Baseline-Single-Hist_GBDT：37/True；Baseline-Single-Rule_Momentum：12/True；Baseline-Static-Ensemble：41/True；Baseline-Cash：0/False。

## 解释限制与后续工作

求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。

当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。
