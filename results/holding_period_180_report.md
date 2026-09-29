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

种子 42；每品种 3 个窗口；每窗 60,000 条原始事件；holding period 180 events；Python 3.12.14；计算线程 1。

JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。

## CME_ES：文件内多窗口描述统计（非独立重复实验）

数据 `databento_glbx.mdp3_mbp_10.parquet`，共 2,807,571 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1217.50 | -405.83 | -837.50 | -165.00 | 37 | 27.03% | 0.00% | -3.04 | 29.86 | 184.84 |
| Variant-ME-CausalShadowARS | -995.00 | -331.67 | -680.00 | -100.00 | 33 | 33.33% | 0.00% | -0.57 | 29.58 | 185.45 |
| Variant-OE-EventUCB | -712.50 | -237.50 | -530.00 | -57.50 | 25 | 20.00% | 0.00% | 0.25 | 28.75 | 187.20 |
| Variant-OE-CausalShadowARS | -617.50 | -205.83 | -355.00 | -112.50 | 22 | 31.82% | 0.00% | 1.14 | 29.20 | 180.00 |
| Baseline-Single-Ridge_Linear | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Logistic_Direction | -1210.00 | -403.33 | -972.50 | 0.00 | 39 | 38.46% | 0.00% | -0.48 | 30.54 | 184.62 |
| Baseline-Single-Decision_Tree | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Hist_GBDT | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Rule_Momentum | -215.00 | -71.67 | -215.00 | 0.00 | 6 | 16.67% | 0.00% | -8.33 | 27.50 | 180.00 |
| Baseline-Static-Ensemble | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random | -642.50 | -214.17 | -432.50 | -57.50 | 22 | 27.27% | 0.00% | -0.28 | 28.92 | 180.00 |
| Baseline-RoundRobin | -777.50 | -259.17 | -462.50 | -100.00 | 26 | 30.77% | 0.00% | -1.44 | 28.46 | 186.92 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -762.50 | -254.17 | -527.50 | -112.50 | 25 | 32.00% | 0.00% | -0.75 | 29.75 | 180.00 |
| Baseline-Random-seed-44 | -635.00 | -211.67 | -490.00 | -72.50 | 24 | 37.50% | 0.00% | 2.60 | 29.06 | 180.00 |
| Ablation-ME-CausalShadowARS-Equal | -1077.50 | -359.17 | -735.00 | -127.50 | 36 | 36.11% | 0.00% | -0.35 | 29.58 | 185.00 |
| Ablation-ME-CausalShadowARS-Single-10 | -1050.00 | -350.00 | -707.50 | -127.50 | 35 | 34.29% | 0.00% | -0.54 | 29.46 | 190.29 |
| Ablation-ME-CausalShadowARS-Single-30 | -1022.50 | -340.83 | -680.00 | -127.50 | 34 | 35.29% | 0.00% | -0.37 | 29.71 | 185.29 |
| Ablation-ME-CausalShadowARS-Single-90 | -915.00 | -305.00 | -600.00 | -100.00 | 31 | 35.48% | 0.00% | 0.00 | 29.52 | 180.00 |
| Ablation-OE-CausalShadowARS-Equal | -862.50 | -287.50 | -600.00 | -112.50 | 30 | 36.67% | 0.00% | 0.83 | 29.58 | 180.00 |
| Ablation-OE-CausalShadowARS-Single-10 | -547.50 | -182.50 | -312.50 | -85.00 | 19 | 31.58% | 0.00% | 0.33 | 29.14 | 180.00 |
| Ablation-OE-CausalShadowARS-Single-30 | -617.50 | -205.83 | -355.00 | -112.50 | 22 | 36.36% | 0.00% | 1.42 | 29.49 | 180.00 |
| Ablation-OE-CausalShadowARS-Single-90 | -725.00 | -241.67 | -462.50 | -112.50 | 25 | 28.00% | 0.00% | 0.25 | 29.25 | 180.00 |
| Sensitivity-ME-SignedBox | -995.00 | -331.67 | -680.00 | -100.00 | 33 | 33.33% | 0.00% | -0.57 | 29.58 | 185.45 |
| Sensitivity-OE-SignedBox | -697.50 | -232.50 | -435.00 | -112.50 | 24 | 33.33% | 0.00% | 0.52 | 29.58 | 180.00 |
| Sensitivity-NetOE-FixedWeights | -345.00 | -115.00 | -150.00 | -85.00 | 13 | 30.77% | 0.00% | 1.44 | 27.98 | 180.00 |

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
| Variant-ME-EventUCB | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Variant-OE-EventUCB | 3 | 180.00 | 180.00 | 180.00 | 8.33 | 27.50 | 33.33% | 0.00% |
| Variant-OE-CausalShadowARS | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 3 | 180.00 | 180.00 | 180.00 | 8.33 | 27.50 | 33.33% | 0.00% |
| Baseline-RoundRobin | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 180.00 | 180.00 | 180.00 | -3.12 | 27.50 | 25.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 180.00 | 180.00 | 180.00 | 9.38 | 27.50 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 180.00 | 180.00 | 180.00 | -8.33 | 27.50 | 16.67% | 0.00% |
| Sensitivity-OE-SignedBox | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 5 | 180.00 | 180.00 | 180.00 | -2.50 | 27.50 | 20.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 34.35% | 53.46% | N/A | 0.00% | N/A | 99.64 / 128.14 |
| Logistic_Direction | 34.26% | 53.46% | N/A | 0.00% | 56.35% | 144.84 / 172.95 |
| Decision_Tree | 33.99% | 53.46% | N/A | 0.00% | N/A | 114.08 / 140.96 |
| Hist_GBDT | 34.69% | 53.46% | N/A | 0.00% | N/A | 362.03 / 387.48 |
| Rule_Momentum | 31.85% | 53.38% | 21.88% | 0.27% | N/A | 7.06 / 7.82 |

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
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | True | -0.000000 |
| ME / signed_box | -0.1828 / 1.0000 / 0.1828 | True | False | -0.047845 |
| OE / signed_box | 0.1875 / 1.0000 / -0.1875 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 25 | 187.20 | 180.00 | 180.00 | -2.75 | 30.75 | 28.00% | 0.00% |
| Variant-ME-CausalShadowARS | 22 | 188.18 | 180.00 | 180.00 | -0.28 | 30.62 | 31.82% | 0.00% |
| Variant-OE-EventUCB | 17 | 190.59 | 180.00 | 180.00 | -1.84 | 29.34 | 17.65% | 0.00% |
| Variant-OE-CausalShadowARS | 12 | 180.00 | 180.00 | 180.00 | 1.04 | 30.62 | 33.33% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 29 | 186.21 | 180.00 | 180.00 | -2.80 | 30.73 | 27.59% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 13 | 180.00 | 180.00 | 180.00 | -3.37 | 29.90 | 23.08% | 0.00% |
| Baseline-RoundRobin | 15 | 192.00 | 180.00 | 180.00 | -1.67 | 29.17 | 26.67% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 16 | 180.00 | 180.00 | 180.00 | -2.34 | 30.62 | 25.00% | 0.00% |
| Baseline-Random-seed-44 | 16 | 180.00 | 180.00 | 180.00 | -0.78 | 29.84 | 25.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 24 | 187.50 | 180.00 | 180.00 | -0.26 | 30.36 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 23 | 195.65 | 180.00 | 180.00 | -0.54 | 30.22 | 30.43% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 22 | 188.18 | 180.00 | 180.00 | -0.28 | 30.62 | 31.82% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 20 | 180.00 | 180.00 | 180.00 | 0.62 | 30.62 | 35.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 20 | 180.00 | 180.00 | 180.00 | 0.31 | 30.31 | 35.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 10 | 180.00 | 180.00 | 180.00 | -0.62 | 30.62 | 30.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 12 | 180.00 | 180.00 | 180.00 | 1.04 | 30.62 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 15 | 180.00 | 180.00 | 180.00 | -0.42 | 30.42 | 26.67% | 0.00% |
| Sensitivity-ME-SignedBox | 22 | 188.18 | 180.00 | 180.00 | -0.28 | 30.62 | 31.82% | 0.00% |
| Sensitivity-OE-SignedBox | 14 | 180.00 | 180.00 | 180.00 | -0.45 | 30.62 | 28.57% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 4 | 180.00 | 180.00 | 180.00 | 1.56 | 29.06 | 25.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.16% | 75.45% | N/A | 0.00% | N/A | 100.08 / 126.87 |
| Logistic_Direction | 22.73% | 75.90% | 54.19% | 1.50% | 75.75% | 147.25 / 179.74 |
| Decision_Tree | 21.53% | 75.45% | N/A | 0.00% | N/A | 116.47 / 144.84 |
| Hist_GBDT | 22.74% | 75.45% | N/A | 0.00% | N/A | 360.80 / 385.23 |
| Rule_Momentum | 21.47% | 75.45% | N/A | 0.00% | N/A | 7.05 / 7.26 |

零信号准确率 75.45%；训练多数类准确率 75.45%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.04134640420000002, '0.5': 0.33158887800000003, '0.9': 0.7446652415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：42/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：2/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-09-22 15:58:06.928056965+00:00 → 2025-09-22 15:59:59.951415627+00:00，113.02 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：并列最佳 `[0.1, 0.8]`，按预先声明顺序选 `0.1`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.080357 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.041667 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.066964 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.041667 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 179.83 | 180.00 | 180.00 | 1.04 | 28.54 | 33.33% | 0.00% |
| Variant-ME-CausalShadowARS | 5 | 180.00 | 180.00 | 180.00 | 7.50 | 27.50 | 60.00% | 0.00% |
| Variant-OE-EventUCB | 5 | 180.00 | 180.00 | 180.00 | 2.50 | 27.50 | 20.00% | 0.00% |
| Variant-OE-CausalShadowARS | 5 | 180.00 | 180.00 | 180.00 | 5.00 | 27.50 | 40.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 10 | 180.00 | 180.00 | 180.00 | 6.25 | 30.00 | 70.00% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 6 | 180.00 | 180.00 | 180.00 | 2.08 | 27.50 | 33.33% | 0.00% |
| Baseline-RoundRobin | 5 | 180.00 | 180.00 | 180.00 | 7.50 | 27.50 | 60.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 5 | 180.00 | 180.00 | 180.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 180.00 | 180.00 | 180.00 | 9.38 | 27.50 | 75.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 180.00 | 180.00 | 180.00 | 7.29 | 28.54 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 180.00 | 180.00 | 180.00 | 7.29 | 28.54 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 180.00 | 180.00 | 180.00 | 7.29 | 28.54 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 5 | 180.00 | 180.00 | 180.00 | 7.50 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 180.00 | 180.00 | 180.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 4 | 180.00 | 180.00 | 180.00 | 6.25 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 5 | 180.00 | 180.00 | 180.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 5 | 180.00 | 180.00 | 180.00 | 5.00 | 27.50 | 40.00% | 0.00% |
| Sensitivity-ME-SignedBox | 5 | 180.00 | 180.00 | 180.00 | 7.50 | 27.50 | 60.00% | 0.00% |
| Sensitivity-OE-SignedBox | 5 | 180.00 | 180.00 | 180.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 4 | 180.00 | 180.00 | 180.00 | 6.25 | 27.50 | 50.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 16.96% | 82.34% | N/A | 0.00% | N/A | 99.33 / 126.13 |
| Logistic_Direction | 17.23% | 82.52% | 83.33% | 0.25% | 84.82% | 142.20 / 171.22 |
| Decision_Tree | 16.77% | 82.34% | N/A | 0.00% | N/A | 113.88 / 141.70 |
| Hist_GBDT | 17.18% | 82.34% | N/A | 0.00% | N/A | 362.95 / 388.50 |
| Rule_Momentum | 16.24% | 82.34% | N/A | 0.00% | N/A | 7.04 / 7.28 |

零信号准确率 82.34%；训练多数类准确率 82.34%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0501033618, '0.5': 0.676730374, '0.9': 1.5287187415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：12/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。
## ICE_BRENT：文件内多窗口描述统计（非独立重复实验）

数据 `databento_ifeu.impact_mbp_10.parquet`，共 2,292,832 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1304.00 | -434.67 | -1193.00 | -13.00 | 48 | 56.25% | 0.00% | 3.13 | 30.29 | 264.75 |
| Variant-ME-CausalShadowARS | -1273.00 | -424.33 | -1152.00 | -13.00 | 51 | 50.98% | 0.00% | 4.12 | 29.08 | 185.69 |
| Variant-OE-EventUCB | -1113.00 | -371.00 | -969.00 | -13.00 | 41 | 58.54% | 0.00% | 4.27 | 31.41 | 314.34 |
| Variant-OE-CausalShadowARS | -1364.00 | -454.67 | -1256.00 | 0.00 | 48 | 45.83% | 2.08% | 1.77 | 30.19 | 203.83 |
| Baseline-Single-Ridge_Linear | -1255.00 | -418.33 | -1209.00 | 0.00 | 45 | 48.89% | 2.22% | 1.67 | 29.56 | 185.42 |
| Baseline-Single-Logistic_Direction | -580.00 | -193.33 | -580.00 | 0.00 | 20 | 55.00% | 5.00% | 8.25 | 37.25 | 176.30 |
| Baseline-Single-Decision_Tree | -910.00 | -303.33 | -910.00 | 0.00 | 30 | 56.67% | 0.00% | 2.67 | 33.00 | 380.13 |
| Baseline-Single-Hist_GBDT | -1058.00 | -352.67 | -910.00 | -56.00 | 36 | 55.56% | 0.00% | 2.22 | 31.61 | 346.78 |
| Baseline-Single-Rule_Momentum | -1407.00 | -469.00 | -1223.00 | 0.00 | 59 | 47.46% | 1.69% | 4.49 | 28.34 | 187.85 |
| Baseline-Static-Ensemble | -864.00 | -288.00 | -864.00 | 0.00 | 28 | 57.14% | 0.00% | 1.96 | 32.82 | 408.57 |
| Baseline-Random | -1427.00 | -475.67 | -1358.00 | -13.00 | 59 | 52.54% | 1.69% | 4.15 | 28.34 | 203.19 |
| Baseline-RoundRobin | -1025.00 | -341.67 | -927.00 | -13.00 | 35 | 60.00% | 0.00% | 3.57 | 32.86 | 357.94 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | -580.00 | -193.33 | -580.00 | 0.00 | 20 | 55.00% | 5.00% | 8.25 | 37.25 | 176.30 |
| Baseline-Random-seed-43 | -1301.00 | -433.67 | -1209.00 | 0.00 | 47 | 46.81% | 0.00% | 1.60 | 29.28 | 258.89 |
| Baseline-Random-seed-44 | -1165.00 | -388.33 | -971.00 | -56.00 | 45 | 44.44% | 2.22% | 3.22 | 29.11 | 286.42 |
| Ablation-ME-CausalShadowARS-Equal | -1366.00 | -455.33 | -1245.00 | -13.00 | 52 | 48.08% | 0.00% | 2.79 | 29.06 | 185.58 |
| Ablation-ME-CausalShadowARS-Single-10 | -1273.00 | -424.33 | -1152.00 | -13.00 | 51 | 50.98% | 0.00% | 4.12 | 29.08 | 185.69 |
| Ablation-ME-CausalShadowARS-Single-30 | -1366.00 | -455.33 | -1245.00 | -13.00 | 52 | 53.85% | 0.00% | 3.27 | 29.54 | 189.04 |
| Ablation-ME-CausalShadowARS-Single-90 | -1350.00 | -450.00 | -1229.00 | -13.00 | 50 | 48.00% | 0.00% | 2.70 | 29.70 | 203.80 |
| Ablation-OE-CausalShadowARS-Equal | -1257.00 | -419.00 | -1149.00 | 0.00 | 49 | 53.06% | 2.04% | 3.98 | 29.63 | 203.35 |
| Ablation-OE-CausalShadowARS-Single-10 | -1280.00 | -426.67 | -1172.00 | 0.00 | 50 | 46.00% | 4.00% | 4.40 | 30.00 | 203.54 |
| Ablation-OE-CausalShadowARS-Single-30 | -1360.00 | -453.33 | -1252.00 | 0.00 | 50 | 52.00% | 2.00% | 2.40 | 29.60 | 199.28 |
| Ablation-OE-CausalShadowARS-Single-90 | -1207.00 | -402.33 | -1099.00 | 0.00 | 49 | 51.02% | 2.04% | 4.80 | 29.43 | 211.20 |
| Sensitivity-ME-SignedBox | -1290.00 | -430.00 | -1169.00 | -13.00 | 50 | 48.00% | 0.00% | 3.70 | 29.50 | 185.80 |
| Sensitivity-OE-SignedBox | -1436.00 | -478.67 | -1328.00 | 0.00 | 52 | 46.15% | 1.92% | 2.31 | 29.92 | 191.62 |
| Sensitivity-NetOE-FixedWeights | -1023.00 | -341.00 | -915.00 | 0.00 | 41 | 60.98% | 4.88% | 5.49 | 30.44 | 190.34 |

### 窗口 1

测试 2025-06-09 01:22:55.205014002+00:00 → 2025-06-09 01:38:08.188182002+00:00，912.98 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00024`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Decision_Tree', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Logistic_Direction', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.011111 |
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | False | -0.000937 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.011111 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.000938 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Variant-ME-CausalShadowARS | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Variant-OE-EventUCB | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Variant-OE-CausalShadowARS | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 2 | 180.00 | 180.00 | 180.00 | 0.00 | 28.00 | 50.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 1 | 180.00 | 180.00 | 180.00 | 10.00 | 23.00 | 100.00% | 0.00% |
| Baseline-RoundRobin | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-44 | 2 | 180.00 | 180.00 | 180.00 | 0.00 | 28.00 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-10 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-30 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-90 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Sensitivity-ME-SignedBox | 1 | 180.00 | 180.00 | 180.00 | 15.00 | 28.00 | 100.00% | 0.00% |
| Sensitivity-OE-SignedBox | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Sensitivity-NetOE-FixedWeights | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 33.79% | 52.78% | N/A | 0.00% | N/A | 100.59 / 127.96 |
| Logistic_Direction | 33.79% | 52.78% | N/A | 0.00% | 51.81% | 144.97 / 172.88 |
| Decision_Tree | 35.29% | 52.78% | N/A | 0.00% | N/A | 115.03 / 141.27 |
| Hist_GBDT | 32.42% | 52.83% | 100.00% | 0.05% | N/A | 364.10 / 387.02 |
| Rule_Momentum | 34.12% | 52.78% | N/A | 0.00% | N/A | 7.01 / 7.23 |

零信号准确率 52.78%；训练多数类准确率 52.78%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.4946915994000001, '0.5': 5.168116001, '0.9': 22.5388601014}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：16/True；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-06-09 13:41:48.627059002+00:00 → 2025-06-09 13:44:10.342476002+00:00，141.72 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[0.00012, 0.00024]`，按预先声明顺序选 `0.00012`。固定模型：并列最佳 `['Logistic_Direction', 'Decision_Tree']`，按预先声明顺序选 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.8`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.022500 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.015000 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 180.00 | 180.00 | 180.00 | 10.00 | 26.33 | 83.33% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Variant-OE-EventUCB | 7 | 180.00 | 180.00 | 180.00 | 7.14 | 25.86 | 71.43% | 0.00% |
| Variant-OE-CausalShadowARS | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Baseline-Single-Ridge_Linear | 2 | 180.00 | 180.00 | 180.00 | 2.50 | 25.50 | 50.00% | 0.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 4 | 180.00 | 180.00 | 180.00 | 1.25 | 24.25 | 50.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 8 | 180.00 | 180.00 | 180.00 | 2.50 | 25.50 | 37.50% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 2 | 180.00 | 180.00 | 180.00 | -5.00 | 23.00 | 0.00% | 0.00% |
| Baseline-RoundRobin | 5 | 180.00 | 180.00 | 180.00 | 9.00 | 26.00 | 80.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 180.00 | 180.00 | 180.00 | 2.50 | 25.50 | 50.00% | 0.00% |
| Baseline-Random-seed-44 | 6 | 180.00 | 180.00 | 180.00 | 1.67 | 24.67 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Sensitivity-OE-SignedBox | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 6 | 180.00 | 180.00 | 180.00 | 7.50 | 25.50 | 66.67% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.14% | 70.70% | 100.00% | 0.03% | N/A | 101.56 / 129.39 |
| Logistic_Direction | 23.04% | 70.67% | N/A | 0.00% | 71.03% | 144.60 / 173.63 |
| Decision_Tree | 22.25% | 70.67% | N/A | 0.00% | N/A | 116.16 / 145.07 |
| Hist_GBDT | 23.24% | 70.66% | 47.62% | 0.18% | N/A | 366.93 / 393.08 |
| Rule_Momentum | 22.23% | 70.63% | 15.00% | 0.17% | N/A | 7.07 / 7.31 |

零信号准确率 70.67%；训练多数类准确率 70.67%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0211009952, '0.5': 0.6737915005, '0.9': 2.4399680977}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：2/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：2/True；Baseline-Single-Hist_GBDT：2/True；Baseline-Single-Rule_Momentum：8/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-06-09 20:31:36.437188002+00:00 → 2025-06-09 22:00:00.245000+00:00，5303.81 秒；奖励模式 `paper_price_difference`。

门槛：选中 `6e-05`。固定模型：选中 `Logistic_Direction`。

校准专家：选中 `Baseline-Cash`。

UCB-ME：选中 `0.8`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 1.0000 / 0.0000 / 0.0000 | True | False | -0.002373 |
| OE / simplex | 0.5634 / 0.4366 / 0.0000 | True | False | -0.000704 |
| ME / signed_box | 1.0000 / 0.0773 / -0.0773 | True | False | -0.002354 |
| OE / signed_box | 0.3488 / 1.0000 / -0.3488 | True | False | -0.000581 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 41 | 279.22 | 180.00 | 540.00 | 1.83 | 30.93 | 51.22% | 0.00% |
| Variant-ME-CausalShadowARS | 44 | 186.59 | 180.00 | 180.00 | 3.41 | 29.59 | 47.73% | 0.00% |
| Variant-OE-EventUCB | 33 | 346.91 | 180.00 | 684.00 | 3.33 | 32.70 | 54.55% | 0.00% |
| Variant-OE-CausalShadowARS | 42 | 207.24 | 180.00 | 180.00 | 0.95 | 30.86 | 42.86% | 2.38% |
| Baseline-Single-Ridge_Linear | 43 | 185.67 | 180.00 | 180.00 | 1.63 | 29.74 | 48.84% | 2.33% |
| Baseline-Single-Logistic_Direction | 20 | 176.30 | 180.00 | 180.00 | 8.25 | 37.25 | 55.00% | 5.00% |
| Baseline-Single-Decision_Tree | 30 | 380.13 | 180.00 | 756.00 | 2.67 | 33.00 | 56.67% | 0.00% |
| Baseline-Single-Hist_GBDT | 30 | 380.13 | 180.00 | 756.00 | 2.50 | 32.83 | 56.67% | 0.00% |
| Baseline-Single-Rule_Momentum | 51 | 189.08 | 180.00 | 180.00 | 4.80 | 28.78 | 49.02% | 1.96% |
| Baseline-Static-Ensemble | 28 | 408.57 | 360.00 | 828.00 | 1.96 | 32.82 | 57.14% | 0.00% |
| Baseline-Random | 56 | 204.43 | 180.00 | 360.00 | 4.38 | 28.62 | 53.57% | 1.79% |
| Baseline-RoundRobin | 29 | 394.76 | 180.00 | 792.00 | 2.24 | 34.21 | 55.17% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 20 | 176.30 | 180.00 | 180.00 | 8.25 | 37.25 | 55.00% | 5.00% |
| Baseline-Random-seed-43 | 43 | 266.23 | 180.00 | 360.00 | 1.51 | 29.63 | 46.51% | 0.00% |
| Baseline-Random-seed-44 | 37 | 309.43 | 180.00 | 540.00 | 3.65 | 29.89 | 45.95% | 2.70% |
| Ablation-ME-CausalShadowARS-Equal | 45 | 186.44 | 180.00 | 180.00 | 1.89 | 29.56 | 44.44% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 44 | 186.59 | 180.00 | 180.00 | 3.41 | 29.59 | 47.73% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 45 | 190.44 | 180.00 | 180.00 | 2.44 | 30.11 | 51.11% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 43 | 207.67 | 180.00 | 324.00 | 1.74 | 30.33 | 44.19% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 43 | 206.60 | 180.00 | 180.00 | 3.49 | 30.21 | 51.16% | 2.33% |
| Ablation-OE-CausalShadowARS-Single-10 | 44 | 206.75 | 180.00 | 360.00 | 3.98 | 30.61 | 43.18% | 4.55% |
| Ablation-OE-CausalShadowARS-Single-30 | 44 | 201.91 | 180.00 | 180.00 | 1.70 | 30.16 | 50.00% | 2.27% |
| Ablation-OE-CausalShadowARS-Single-90 | 43 | 215.56 | 180.00 | 324.00 | 4.42 | 29.98 | 48.84% | 2.33% |
| Sensitivity-ME-SignedBox | 43 | 186.74 | 180.00 | 180.00 | 2.91 | 30.09 | 44.19% | 0.00% |
| Sensitivity-OE-SignedBox | 46 | 193.13 | 180.00 | 180.00 | 1.63 | 30.50 | 43.48% | 2.17% |
| Sensitivity-NetOE-FixedWeights | 35 | 192.11 | 180.00 | 180.00 | 5.14 | 31.29 | 60.00% | 5.71% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 28.76% | 60.89% | 55.00% | 3.51% | N/A | 99.84 / 138.30 |
| Logistic_Direction | 29.09% | 61.62% | 57.35% | 3.98% | 60.75% | 144.85 / 174.58 |
| Decision_Tree | 22.05% | 25.00% | 16.14% | 66.53% | N/A | 113.66 / 141.61 |
| Hist_GBDT | 22.50% | 25.13% | 16.12% | 66.30% | N/A | 361.31 / 390.15 |
| Rule_Momentum | 27.49% | 56.89% | 36.28% | 16.45% | N/A | 6.99 / 7.23 |

零信号准确率 60.25%；训练多数类准确率 60.25%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.035222500000000004, '0.5': 5.053657501, '0.9': 23.3169947982}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：61/True；Baseline-Single-Logistic_Direction：12/True；Baseline-Single-Decision_Tree：17/True；Baseline-Single-Hist_GBDT：17/True；Baseline-Single-Rule_Momentum：71/True；Baseline-Static-Ensemble：19/True；Baseline-Cash：0/False。

## 解释限制与后续工作

求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。

当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。
