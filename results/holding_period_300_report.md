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

种子 42；每品种 3 个窗口；每窗 60,000 条原始事件；holding period 300 events；Python 3.12.14；计算线程 1。

JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。

## CME_ES：文件内多窗口描述统计（非独立重复实验）

数据 `databento_glbx.mdp3_mbp_10.parquet`，共 2,807,571 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1105.00 | -368.33 | -662.50 | -177.50 | 32 | 31.25% | 0.00% | -4.49 | 30.04 | 296.22 |
| Variant-ME-CausalShadowARS | -997.50 | -332.50 | -620.00 | -112.50 | 29 | 31.03% | 0.00% | -4.96 | 29.44 | 299.03 |
| Variant-OE-EventUCB | -777.50 | -259.17 | -430.00 | -82.50 | 21 | 19.05% | 0.00% | -8.04 | 28.99 | 298.71 |
| Variant-OE-CausalShadowARS | -587.50 | -195.83 | -380.00 | -97.50 | 20 | 40.00% | 0.00% | 0.00 | 29.38 | 298.60 |
| Baseline-Single-Ridge_Linear | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Logistic_Direction | -940.00 | -313.33 | -690.00 | 0.00 | 31 | 38.71% | 0.00% | 0.00 | 30.32 | 308.81 |
| Baseline-Single-Decision_Tree | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Hist_GBDT | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Single-Rule_Momentum | -265.00 | -88.33 | -265.00 | 0.00 | 6 | 33.33% | 0.00% | -16.67 | 27.50 | 300.00 |
| Baseline-Static-Ensemble | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random | -612.50 | -204.17 | -352.50 | -95.00 | 20 | 30.00% | 0.00% | -1.56 | 29.06 | 300.00 |
| Baseline-RoundRobin | -785.00 | -261.67 | -407.50 | -112.50 | 24 | 33.33% | 0.00% | -4.17 | 28.54 | 311.33 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-Random-seed-43 | -747.50 | -249.17 | -487.50 | -112.50 | 24 | 37.50% | 0.00% | -1.82 | 29.32 | 298.88 |
| Baseline-Random-seed-44 | -657.50 | -219.17 | -475.00 | -72.50 | 23 | 52.17% | 0.00% | 0.54 | 29.13 | 300.00 |
| Ablation-ME-CausalShadowARS-Equal | -1052.50 | -350.83 | -675.00 | -112.50 | 31 | 29.03% | 0.00% | -4.64 | 29.31 | 299.10 |
| Ablation-ME-CausalShadowARS-Single-10 | -1052.50 | -350.83 | -675.00 | -112.50 | 31 | 29.03% | 0.00% | -4.64 | 29.31 | 299.10 |
| Ablation-ME-CausalShadowARS-Single-30 | -997.50 | -332.50 | -620.00 | -112.50 | 29 | 31.03% | 0.00% | -4.96 | 29.44 | 299.03 |
| Ablation-ME-CausalShadowARS-Single-90 | -957.50 | -319.17 | -580.00 | -112.50 | 28 | 32.14% | 0.00% | -4.69 | 29.51 | 299.00 |
| Ablation-OE-CausalShadowARS-Equal | -697.50 | -232.50 | -475.00 | -110.00 | 24 | 41.67% | 0.00% | 0.52 | 29.58 | 311.33 |
| Ablation-OE-CausalShadowARS-Single-10 | -587.50 | -195.83 | -365.00 | -110.00 | 20 | 40.00% | 0.00% | -0.31 | 29.06 | 298.60 |
| Ablation-OE-CausalShadowARS-Single-30 | -602.50 | -200.83 | -380.00 | -110.00 | 21 | 42.86% | 0.00% | 0.60 | 29.29 | 298.67 |
| Ablation-OE-CausalShadowARS-Single-90 | -707.50 | -235.83 | -500.00 | -97.50 | 23 | 34.78% | 0.00% | -1.09 | 29.67 | 311.83 |
| Sensitivity-ME-SignedBox | -997.50 | -332.50 | -620.00 | -112.50 | 29 | 31.03% | 0.00% | -4.96 | 29.44 | 299.03 |
| Sensitivity-OE-SignedBox | -692.50 | -230.83 | -485.00 | -97.50 | 22 | 31.82% | 0.00% | -1.99 | 29.49 | 312.36 |
| Sensitivity-NetOE-FixedWeights | -395.00 | -131.67 | -187.50 | -97.50 | 13 | 30.77% | 0.00% | -1.92 | 28.46 | 297.85 |

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
| Variant-ME-EventUCB | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Variant-ME-CausalShadowARS | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Variant-OE-EventUCB | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Variant-OE-CausalShadowARS | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 3 | 300.00 | 300.00 | 300.00 | -2.08 | 29.58 | 33.33% | 0.00% |
| Baseline-RoundRobin | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 300.00 | 300.00 | 300.00 | -9.38 | 27.50 | 50.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 300.00 | 300.00 | 300.00 | -16.67 | 27.50 | 33.33% | 0.00% |
| Sensitivity-OE-SignedBox | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 4 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 50.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 34.35% | 53.46% | N/A | 0.00% | N/A | 102.27 / 129.84 |
| Logistic_Direction | 34.26% | 53.46% | N/A | 0.00% | 56.35% | 145.62 / 185.09 |
| Decision_Tree | 33.99% | 53.46% | N/A | 0.00% | N/A | 115.97 / 145.13 |
| Hist_GBDT | 34.69% | 53.46% | N/A | 0.00% | N/A | 367.99 / 397.93 |
| Rule_Momentum | 31.85% | 53.38% | 21.88% | 0.27% | N/A | 7.06 / 7.44 |

零信号准确率 53.46%；训练多数类准确率 53.46%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 1.2549504662, '0.5': 5.124046678999999, '0.9': 11.5675809418}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-09-22 13:59:49.662763235+00:00 → 2025-09-22 14:00:19.921952383+00:00，30.26 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.7720 / 0.2280 | True | False | -0.053495 |
| OE / simplex | 0.0000 / 1.0000 / 0.0000 | True | True | -0.000000 |
| ME / signed_box | -0.1828 / 1.0000 / 0.1828 | True | False | -0.047845 |
| OE / signed_box | 1.0000 / -1.0000 / 1.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 20 | 300.00 | 300.00 | 300.00 | -1.88 | 31.25 | 30.00% | 0.00% |
| Variant-ME-CausalShadowARS | 18 | 300.00 | 300.00 | 300.00 | -3.82 | 30.62 | 22.22% | 0.00% |
| Variant-OE-EventUCB | 12 | 300.00 | 300.00 | 300.00 | -5.73 | 30.10 | 8.33% | 0.00% |
| Variant-OE-CausalShadowARS | 12 | 300.00 | 300.00 | 300.00 | -1.04 | 30.62 | 33.33% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 21 | 314.29 | 300.00 | 300.00 | -2.08 | 30.77 | 28.57% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 11 | 300.00 | 300.00 | 300.00 | -2.27 | 29.77 | 27.27% | 0.00% |
| Baseline-RoundRobin | 13 | 323.08 | 300.00 | 300.00 | -1.92 | 29.42 | 23.08% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 15 | 300.00 | 300.00 | 300.00 | -2.50 | 30.00 | 26.67% | 0.00% |
| Baseline-Random-seed-44 | 15 | 300.00 | 300.00 | 300.00 | -1.67 | 30.00 | 46.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 20 | 300.00 | 300.00 | 300.00 | -3.44 | 30.31 | 20.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 20 | 300.00 | 300.00 | 300.00 | -3.44 | 30.31 | 20.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 18 | 300.00 | 300.00 | 300.00 | -3.82 | 30.62 | 22.22% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 17 | 300.00 | 300.00 | 300.00 | -3.31 | 30.81 | 23.53% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 15 | 320.00 | 300.00 | 300.00 | -0.83 | 30.83 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 11 | 300.00 | 300.00 | 300.00 | -2.84 | 30.34 | 27.27% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 12 | 300.00 | 300.00 | 300.00 | -1.04 | 30.62 | 33.33% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 15 | 320.00 | 300.00 | 300.00 | -2.50 | 30.83 | 26.67% | 0.00% |
| Sensitivity-ME-SignedBox | 18 | 300.00 | 300.00 | 300.00 | -3.82 | 30.62 | 22.22% | 0.00% |
| Sensitivity-OE-SignedBox | 14 | 321.43 | 300.00 | 300.00 | -4.02 | 30.62 | 21.43% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 5 | 300.00 | 300.00 | 300.00 | -7.50 | 30.00 | 0.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.16% | 75.45% | N/A | 0.00% | N/A | 100.99 / 126.61 |
| Logistic_Direction | 22.73% | 75.90% | 54.19% | 1.50% | 75.75% | 146.15 / 174.12 |
| Decision_Tree | 21.53% | 75.45% | N/A | 0.00% | N/A | 115.39 / 141.46 |
| Hist_GBDT | 22.74% | 75.45% | N/A | 0.00% | N/A | 370.62 / 396.26 |
| Rule_Momentum | 21.47% | 75.45% | N/A | 0.00% | N/A | 7.10 / 7.50 |

零信号准确率 75.45%；训练多数类准确率 75.45%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.04134640420000002, '0.5': 0.33158887800000003, '0.9': 0.7446652415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：34/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：2/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-09-22 15:58:06.928056965+00:00 → 2025-09-22 15:59:59.951415627+00:00，113.02 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[3e-05, 6e-05]`，按预先声明顺序选 `3e-05`。固定模型：并列最佳 `['Ridge_Linear', 'Decision_Tree', 'Hist_GBDT', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Hist_GBDT', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：选中 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.080357 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.068182 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.066964 |
| OE / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.068182 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 279.83 | 300.00 | 300.00 | -1.04 | 28.54 | 33.33% | 0.00% |
| Variant-ME-CausalShadowARS | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Variant-OE-EventUCB | 3 | 291.00 | 300.00 | 300.00 | 0.00 | 27.50 | 33.33% | 0.00% |
| Variant-OE-CausalShadowARS | 4 | 293.00 | 300.00 | 300.00 | 3.12 | 27.50 | 50.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 10 | 297.30 | 300.00 | 300.00 | 4.38 | 29.38 | 60.00% | 0.00% |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 6 | 300.00 | 300.00 | 300.00 | 0.00 | 27.50 | 33.33% | 0.00% |
| Baseline-RoundRobin | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 5 | 294.60 | 300.00 | 300.00 | 6.25 | 28.75 | 60.00% | 0.00% |
| Baseline-Random-seed-44 | 4 | 300.00 | 300.00 | 300.00 | 9.38 | 27.50 | 75.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 4 | 293.00 | 300.00 | 300.00 | 3.12 | 27.50 | 50.00% | 0.00% |
| Sensitivity-ME-SignedBox | 5 | 294.40 | 300.00 | 300.00 | 5.00 | 27.50 | 60.00% | 0.00% |
| Sensitivity-OE-SignedBox | 4 | 293.00 | 300.00 | 300.00 | 3.12 | 27.50 | 50.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 4 | 293.00 | 300.00 | 300.00 | 3.12 | 27.50 | 50.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 16.96% | 82.34% | N/A | 0.00% | N/A | 101.79 / 130.29 |
| Logistic_Direction | 17.23% | 82.52% | 83.33% | 0.25% | 84.82% | 144.64 / 172.76 |
| Decision_Tree | 16.77% | 82.34% | N/A | 0.00% | N/A | 115.19 / 186.06 |
| Hist_GBDT | 17.18% | 82.34% | N/A | 0.00% | N/A | 368.58 / 389.96 |
| Rule_Momentum | 16.24% | 82.34% | N/A | 0.00% | N/A | 7.11 / 7.84 |

零信号准确率 82.34%；训练多数类准确率 82.34%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0501033618, '0.5': 0.676730374, '0.9': 1.5287187415999999}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：11/True；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：0/False；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。
## ICE_BRENT：文件内多窗口描述统计（非独立重复实验）

数据 `databento_ifeu.impact_mbp_10.parquet`，共 2,292,832 行。每窗账户重置，合计不是连续账户收益。

| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | -1028.00 | -342.67 | -927.00 | -23.00 | 36 | 44.44% | 2.78% | 1.53 | 30.08 | 382.44 |
| Variant-ME-CausalShadowARS | -1114.00 | -371.33 | -973.00 | -23.00 | 38 | 44.74% | 0.00% | 0.26 | 29.58 | 307.68 |
| Variant-OE-EventUCB | -912.00 | -304.00 | -764.00 | -23.00 | 24 | 37.50% | 0.00% | -5.21 | 32.79 | 562.17 |
| Variant-OE-CausalShadowARS | -902.00 | -300.67 | -787.00 | 0.00 | 34 | 58.82% | 0.00% | 3.97 | 30.50 | 303.24 |
| Baseline-Single-Ridge_Linear | -996.00 | -332.00 | -940.00 | 0.00 | 32 | 46.88% | 0.00% | -0.94 | 30.19 | 291.84 |
| Baseline-Single-Logistic_Direction | -678.00 | -226.00 | -678.00 | 0.00 | 16 | 50.00% | 0.00% | -5.31 | 37.06 | 299.19 |
| Baseline-Single-Decision_Tree | -622.00 | -207.33 | -622.00 | 0.00 | 14 | 42.86% | 0.00% | -5.71 | 38.71 | 838.50 |
| Baseline-Single-Hist_GBDT | -637.00 | -212.33 | -479.00 | -66.00 | 19 | 42.11% | 0.00% | 0.79 | 34.32 | 710.11 |
| Baseline-Single-Rule_Momentum | -996.00 | -332.00 | -845.00 | 0.00 | 42 | 52.38% | 2.38% | 5.48 | 29.19 | 299.74 |
| Baseline-Static-Ensemble | -469.00 | -156.33 | -469.00 | 0.00 | 13 | 46.15% | 0.00% | 1.15 | 37.23 | 899.38 |
| Baseline-Random | -1098.00 | -366.00 | -1009.00 | -23.00 | 36 | 36.11% | 2.78% | -0.97 | 29.53 | 348.97 |
| Baseline-RoundRobin | -790.00 | -263.33 | -682.00 | -23.00 | 20 | 45.00% | 0.00% | -3.50 | 36.00 | 675.70 |
| Baseline-Cash | 0.00 | 0.00 | 0.00 | 0.00 | 0 | N/A | N/A | 0.00 | 0.00 | N/A |
| Baseline-ValidationBest | -678.00 | -226.00 | -678.00 | 0.00 | 16 | 50.00% | 0.00% | -5.31 | 37.06 | 299.19 |
| Baseline-Random-seed-43 | -849.00 | -283.00 | -757.00 | 0.00 | 33 | 48.48% | 3.03% | 4.70 | 30.42 | 389.97 |
| Baseline-Random-seed-44 | -1102.00 | -367.33 | -878.00 | -66.00 | 34 | 38.24% | 2.94% | -1.91 | 30.50 | 413.88 |
| Ablation-ME-CausalShadowARS-Equal | -1147.00 | -382.33 | -1006.00 | -23.00 | 39 | 51.28% | 0.00% | 0.00 | 29.41 | 301.00 |
| Ablation-ME-CausalShadowARS-Single-10 | -1114.00 | -371.33 | -973.00 | -23.00 | 38 | 44.74% | 0.00% | 0.26 | 29.58 | 307.68 |
| Ablation-ME-CausalShadowARS-Single-30 | -1094.00 | -364.67 | -953.00 | -23.00 | 38 | 44.74% | 0.00% | 1.32 | 30.11 | 293.13 |
| Ablation-ME-CausalShadowARS-Single-90 | -1067.00 | -355.67 | -926.00 | -23.00 | 39 | 46.15% | 0.00% | 2.44 | 29.79 | 310.51 |
| Ablation-OE-CausalShadowARS-Equal | -1012.00 | -337.33 | -897.00 | 0.00 | 34 | 44.12% | 0.00% | 0.88 | 30.65 | 320.88 |
| Ablation-OE-CausalShadowARS-Single-10 | -872.00 | -290.67 | -757.00 | 0.00 | 34 | 61.76% | 0.00% | 5.15 | 30.79 | 294.41 |
| Ablation-OE-CausalShadowARS-Single-30 | -1018.00 | -339.33 | -903.00 | 0.00 | 36 | 52.78% | 0.00% | 1.94 | 30.22 | 294.72 |
| Ablation-OE-CausalShadowARS-Single-90 | -996.00 | -332.00 | -881.00 | 0.00 | 32 | 46.88% | 0.00% | 0.00 | 31.12 | 331.75 |
| Sensitivity-ME-SignedBox | -1068.00 | -356.00 | -927.00 | -23.00 | 36 | 47.22% | 0.00% | 0.14 | 29.81 | 316.44 |
| Sensitivity-OE-SignedBox | -898.00 | -299.33 | -783.00 | 0.00 | 36 | 52.78% | 2.78% | 5.00 | 29.94 | 293.39 |
| Sensitivity-NetOE-FixedWeights | -728.00 | -242.67 | -613.00 | 0.00 | 26 | 57.69% | 3.85% | 4.42 | 32.42 | 380.35 |

### 窗口 1

测试 2025-06-09 01:22:55.205014002+00:00 → 2025-06-09 01:38:08.188182002+00:00，912.98 秒；奖励模式 `paper_price_difference`。

门槛：选中 `0.00024`。固定模型：并列最佳 `['Ridge_Linear', 'Logistic_Direction', 'Decision_Tree', 'Rule_Momentum']`，按预先声明顺序选 `Ridge_Linear`。

校准专家：并列最佳 `['Baseline-Single-Ridge_Linear', 'Baseline-Single-Logistic_Direction', 'Baseline-Single-Decision_Tree', 'Baseline-Single-Rule_Momentum', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Ridge_Linear`。

UCB-ME：选中 `0.01`；UCB-OE：并列最佳 `[0.1, 0.8]`，按预先声明顺序选 `0.1`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.011111 |
| OE / simplex | 0.0000 / 0.0000 / 1.0000 | True | True | -0.000000 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.011111 |
| OE / signed_box | 1.0000 / -0.7500 / 0.7500 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Variant-ME-CausalShadowARS | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Variant-OE-EventUCB | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Variant-OE-CausalShadowARS | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Ridge_Linear | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 2 | 300.00 | 300.00 | 300.00 | -10.00 | 23.00 | 0.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Baseline-RoundRobin | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-44 | 2 | 300.00 | 300.00 | 300.00 | -10.00 | 23.00 | 0.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-10 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-30 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Ablation-OE-CausalShadowARS-Single-90 | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Sensitivity-ME-SignedBox | 1 | 300.00 | 300.00 | 300.00 | 0.00 | 23.00 | 0.00% | 0.00% |
| Sensitivity-OE-SignedBox | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Sensitivity-NetOE-FixedWeights | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 33.79% | 52.78% | N/A | 0.00% | N/A | 101.09 / 125.92 |
| Logistic_Direction | 33.79% | 52.78% | N/A | 0.00% | 51.81% | 144.19 / 177.61 |
| Decision_Tree | 35.29% | 52.78% | N/A | 0.00% | N/A | 114.64 / 141.13 |
| Hist_GBDT | 32.42% | 52.83% | 100.00% | 0.05% | N/A | 365.49 / 392.74 |
| Rule_Momentum | 34.12% | 52.78% | N/A | 0.00% | N/A | 6.98 / 7.46 |

零信号准确率 52.78%；训练多数类准确率 52.78%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.4946915994000001, '0.5': 5.168116001, '0.9': 22.5388601014}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：0/False；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：0/False；Baseline-Single-Hist_GBDT：14/True；Baseline-Single-Rule_Momentum：0/False；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 2

测试 2025-06-09 13:41:48.627059002+00:00 → 2025-06-09 13:44:10.342476002+00:00，141.72 秒；奖励模式 `paper_price_difference`。

门槛：并列最佳 `[0.00012, 0.00024]`，按预先声明顺序选 `0.00012`。固定模型：并列最佳 `['Logistic_Direction', 'Decision_Tree']`，按预先声明顺序选 `Logistic_Direction`。

校准专家：并列最佳 `['Baseline-Single-Logistic_Direction', 'Baseline-Static-Ensemble', 'Baseline-Cash']`，按预先声明顺序选 `Baseline-Single-Logistic_Direction`。

UCB-ME：选中 `0.8`；UCB-OE：并列最佳 `[0.01, 0.1, 0.8]`，按预先声明顺序选 `0.01`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 0.0000 / 0.0000 / 1.0000 | True | False | -0.022500 |
| OE / simplex | 0.4286 / 0.0000 / 0.5714 | True | False | -0.001071 |
| ME / signed_box | -1.0000 / 1.0000 / 1.0000 | True | False | -0.015000 |
| OE / signed_box | 1.0000 / -1.0000 / 1.0000 | True | True | -0.000000 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 6 | 300.00 | 300.00 | 300.00 | 15.83 | 28.83 | 83.33% | 16.67% |
| Variant-ME-CausalShadowARS | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Variant-OE-EventUCB | 5 | 300.00 | 300.00 | 300.00 | 3.00 | 28.00 | 60.00% | 0.00% |
| Variant-OE-CausalShadowARS | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Baseline-Single-Ridge_Linear | 2 | 300.00 | 300.00 | 300.00 | 2.50 | 30.50 | 50.00% | 0.00% |
| Baseline-Single-Logistic_Direction | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Decision_Tree | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Single-Hist_GBDT | 4 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 50.00% | 0.00% |
| Baseline-Single-Rule_Momentum | 7 | 300.00 | 300.00 | 300.00 | 6.43 | 28.00 | 57.14% | 0.00% |
| Baseline-Static-Ensemble | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random | 2 | 300.00 | 300.00 | 300.00 | -7.50 | 25.50 | 0.00% | 0.00% |
| Baseline-RoundRobin | 5 | 300.00 | 300.00 | 300.00 | 12.00 | 29.00 | 80.00% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-Random-seed-43 | 4 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 75.00% | 0.00% |
| Baseline-Random-seed-44 | 6 | 300.00 | 300.00 | 300.00 | 1.67 | 28.00 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Equal | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Sensitivity-ME-SignedBox | 6 | 300.00 | 300.00 | 300.00 | 9.17 | 28.83 | 66.67% | 0.00% |
| Sensitivity-OE-SignedBox | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |
| Sensitivity-NetOE-FixedWeights | 5 | 300.00 | 300.00 | 300.00 | 5.00 | 28.00 | 60.00% | 0.00% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 23.14% | 70.70% | 100.00% | 0.03% | N/A | 101.49 / 128.27 |
| Logistic_Direction | 23.04% | 70.67% | N/A | 0.00% | 71.03% | 145.70 / 174.44 |
| Decision_Tree | 22.25% | 70.67% | N/A | 0.00% | N/A | 115.56 / 142.28 |
| Hist_GBDT | 23.24% | 70.66% | 47.62% | 0.18% | N/A | 363.01 / 392.95 |
| Rule_Momentum | 22.23% | 70.63% | 15.00% | 0.17% | N/A | 7.01 / 7.47 |

零信号准确率 70.67%；训练多数类准确率 70.67%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.0211009952, '0.5': 0.6737915005, '0.9': 2.4399680977}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：2/True；Baseline-Single-Logistic_Direction：0/False；Baseline-Single-Decision_Tree：2/True；Baseline-Single-Hist_GBDT：2/True；Baseline-Single-Rule_Momentum：8/True；Baseline-Static-Ensemble：0/False；Baseline-Cash：0/False。

### 窗口 3

测试 2025-06-09 20:31:36.437188002+00:00 → 2025-06-09 22:00:00.245000+00:00，5303.81 秒；奖励模式 `paper_price_difference`。

门槛：选中 `6e-05`。固定模型：选中 `Logistic_Direction`。

校准专家：选中 `Baseline-Cash`。

UCB-ME：选中 `0.1`；UCB-OE：选中 `0.8`。

| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |
|---|---|---|---|---:|
| ME / simplex | 1.0000 / 0.0000 / 0.0000 | True | False | -0.002373 |
| OE / simplex | 0.7458 / 0.2542 / 0.0000 | True | False | -0.000932 |
| ME / signed_box | 1.0000 / 0.0773 / -0.0773 | True | False | -0.002354 |
| OE / signed_box | 0.3929 / 1.0000 / -0.3929 | True | False | -0.000655 |

| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Variant-ME-EventUCB | 29 | 402.34 | 300.00 | 600.00 | -1.38 | 30.59 | 37.93% | 0.00% |
| Variant-ME-CausalShadowARS | 31 | 309.42 | 300.00 | 300.00 | -1.45 | 29.94 | 41.94% | 0.00% |
| Variant-OE-EventUCB | 18 | 649.56 | 300.00 | 1590.00 | -7.78 | 34.67 | 33.33% | 0.00% |
| Variant-OE-CausalShadowARS | 29 | 303.79 | 300.00 | 300.00 | 3.79 | 30.93 | 58.62% | 0.00% |
| Baseline-Single-Ridge_Linear | 30 | 291.30 | 300.00 | 300.00 | -1.17 | 30.17 | 46.67% | 0.00% |
| Baseline-Single-Logistic_Direction | 16 | 299.19 | 300.00 | 300.00 | -5.31 | 37.06 | 50.00% | 0.00% |
| Baseline-Single-Decision_Tree | 14 | 838.50 | 450.00 | 1920.00 | -5.71 | 38.71 | 42.86% | 0.00% |
| Baseline-Single-Hist_GBDT | 13 | 899.38 | 600.00 | 1980.00 | 1.15 | 38.00 | 46.15% | 0.00% |
| Baseline-Single-Rule_Momentum | 35 | 299.69 | 300.00 | 300.00 | 5.29 | 29.43 | 51.43% | 2.86% |
| Baseline-Static-Ensemble | 13 | 899.38 | 592.00 | 1800.00 | 1.15 | 37.23 | 46.15% | 0.00% |
| Baseline-Random | 33 | 353.42 | 300.00 | 300.00 | -0.61 | 29.97 | 39.39% | 3.03% |
| Baseline-RoundRobin | 14 | 836.71 | 450.00 | 1830.00 | -9.29 | 39.43 | 35.71% | 0.00% |
| Baseline-Cash | 0 | N/A | N/A | N/A | 0.00 | 0.00 | N/A | N/A |
| Baseline-ValidationBest | 16 | 299.19 | 300.00 | 300.00 | -5.31 | 37.06 | 50.00% | 0.00% |
| Baseline-Random-seed-43 | 29 | 402.38 | 300.00 | 600.00 | 4.66 | 30.76 | 44.83% | 3.45% |
| Baseline-Random-seed-44 | 26 | 448.92 | 300.00 | 900.00 | -2.12 | 31.65 | 38.46% | 3.85% |
| Ablation-ME-CausalShadowARS-Equal | 32 | 301.22 | 300.00 | 300.00 | -1.72 | 29.72 | 50.00% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-10 | 31 | 309.42 | 300.00 | 300.00 | -1.45 | 29.94 | 41.94% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-30 | 31 | 291.58 | 300.00 | 300.00 | -0.16 | 30.58 | 41.94% | 0.00% |
| Ablation-ME-CausalShadowARS-Single-90 | 32 | 312.81 | 300.00 | 300.00 | 1.25 | 30.19 | 43.75% | 0.00% |
| Ablation-OE-CausalShadowARS-Equal | 29 | 324.48 | 300.00 | 360.00 | 0.17 | 31.10 | 41.38% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-10 | 29 | 293.45 | 300.00 | 300.00 | 5.17 | 31.28 | 62.07% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-30 | 31 | 293.87 | 300.00 | 300.00 | 1.45 | 30.58 | 51.61% | 0.00% |
| Ablation-OE-CausalShadowARS-Single-90 | 27 | 337.63 | 300.00 | 300.00 | -0.93 | 31.70 | 44.44% | 0.00% |
| Sensitivity-ME-SignedBox | 29 | 320.41 | 300.00 | 300.00 | -1.72 | 30.24 | 44.83% | 0.00% |
| Sensitivity-OE-SignedBox | 31 | 292.32 | 300.00 | 300.00 | 5.00 | 30.26 | 51.61% | 3.23% |
| Sensitivity-NetOE-FixedWeights | 21 | 399.48 | 300.00 | 600.00 | 4.29 | 33.48 | 57.14% | 4.76% |

| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |
|---|---:|---:|---:|---:|---:|---|
| Ridge_Linear | 28.76% | 60.89% | 55.00% | 3.51% | N/A | 102.76 / 134.44 |
| Logistic_Direction | 29.09% | 61.62% | 57.35% | 3.98% | 60.75% | 144.87 / 172.44 |
| Decision_Tree | 22.05% | 25.00% | 16.14% | 66.53% | N/A | 116.28 / 146.31 |
| Hist_GBDT | 22.50% | 25.13% | 16.12% | 66.30% | N/A | 367.66 / 387.19 |
| Rule_Momentum | 27.49% | 56.89% | 36.28% | 16.45% | N/A | 7.17 / 7.34 |

零信号准确率 60.25%；训练多数类准确率 60.25%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 30 行仍参与执行。

90 事件对应秒数分位数：`{'0.1': 0.035222500000000004, '0.5': 5.053657501, '0.9': 23.3169947982}`。

校准订单数/均值是否定义：Baseline-Single-Ridge_Linear：47/True；Baseline-Single-Logistic_Direction：12/True；Baseline-Single-Decision_Tree：21/True；Baseline-Single-Hist_GBDT：23/True；Baseline-Single-Rule_Momentum：51/True；Baseline-Static-Ensemble：23/True；Baseline-Cash：0/False。

## 解释限制与后续工作

求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。

当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。
