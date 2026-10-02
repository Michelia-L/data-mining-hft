# OE IRL 校准可表示性审计（2026-10-02）

## 问题与论文阅读

PR #14 完成独立历史窗口双对照，但动态入口仍用等权奖励；PR #12 的
多模型 OE IRL 在三个报价年龄下没有可用冻结选择。此前的失败是在项目
非负单纯形或 `[-1,1]` 盒约束下观察到，不能概括为原文 Eq.(3) 都不可能。

本次开发前通读 `1679894.pdf` 全文，重点复核物理第 4 页 §3.3、
Eq.(2)–(4)/Algorithm 1，第 5 页 §3.3.2/§3.4、Eq.(5)–(6) 及
Algorithm 2–3，第 7–8 页时间切分和奖励对照。原文规定原始中间价差的
线性多尺度组合、权重和为1；没有明确非负或逐项盒边界，也没有公开完整
参数优化器。Algorithm 1 的专家与策略最优响应学习不能用一次几何检查替代。

本项聚焦失败归因，为之后的奖励联动提供依据；不更改原专家、旧拟合、
执行器、模型参数或费用，不把诊断见证当学成的奖励。

## 三个集合与证据定义

输入 `mu` 为 P×H 校准候选特征矩阵，按完整成熟订单数量加权合并，仍是
可观测订单子集，不是固定期间全部订单的完整 Eq.(5)。现金无实测 OE，
仅在优化中为零向量。专家按所有已结算候选的校准净盈亏选择，缺观察不换
次优交易策略；资格和最少一单条件沿用原协议，不声称统计充分。

令 `D_j=mu_expert-mu_j`，可表示要求 `D_j·w >= -tau` 和 `sum(w)=1`。
这里 tau 为原校准 `1e-8` **价格点**容差，与美元净盈亏和奖励并列容差不同。

| 集合 | 权重约束 | 依据 |
| --- | --- | --- |
| simplex | `0<=w_i<=1; sum(w)=1` | 项目非负附加假设 |
| signed_box | `-1<=w_i<=1; sum(w)=1` | 项目有界负权敏感性 |
| sum_only | `sum(w)=1`，各项无界 | Eq.(3) 明确的等式条件 |

可行时解 `min sum(|w_i|)`，保存权重、候选分数、最大权幅、L1、等式残差
及每个不等式的最大违例。最小 L1 是诊断见证选择，不是原文最大间隔目标。
不直接放开现有最大间隔求解器的边界：仅等式集合的某些间隔问题可能无界，
需先定义学习/正则化与执行协议。

不可行时将候选分离和盒边界都写成 `A w<=b`，再求数值证书：

```text
y >= 0
A.T @ y + z * ones(H) = 0
b @ y + z = -1
```

假设可行，则 `y·A w+z·sum(w)=0`，却必须小于等于 `y·b+z=-1`，矛盾。
输出每一行的候选/边界身份及全部乘子，让组员能从同一矩阵复算。

HiGHS 原始/对偶可行精度为 `1e-9`，证书独立复核精度为 `1e-9`。求解失败、
非有限见证或复核不通过都返回未定义，不把单个 `infeasible` 状态当证明。
下述结论为浮点输入和预声明精度下的数值结论，不宣称精确有理数证明。
不可表示仅针对当前专家和成熟子集矩阵，不能推广到所有市场或其他策略空间。

## 历史数据与信息边界

只读 PR #12 归档 `/tmp/data-mining-library-oe/evaluation/result.json` 中
2025-10-01 校准段。输入来自原 500/1000/2000ms 报价年龄、500ms 网格、
5000ms 预测、500ms 延迟、15000ms 复核及七尺度最长 3645000ms 前瞻。
原训练为09-29/30，后段10-02/03已经用于开发，没有未触碰测试声明。

归档仍含其他日期，但冻结载荷只提取校准日的统计；逐日校准摘要按原账本
重新合并，必须与归档统计完全一致。输入整个文件 SHA-256 用于追溯，不
代表验证/测试数值用于求解。测试证明任意改变后两段不会改变校准诊断。

归档当时源码/模型/数据身份与本次诊断源码分别保存。允许历史源码不同于
当前源码，明确为旧结果诊断；没有用当前执行器重新回放原行情、训练模型
或新增正式损益。旧模型、行情、PR #12/13/14 结果都保留。

## 真实审计结果

| 年龄 ms | 原专家 | 约束 | 可表示性 | 原拟合状态 | 解释 |
| --- | --- | --- | --- | --- | --- |
| 500 | Library-Ridge-price-h1 | 全部 | 未进行几何判断 | blocked_no_matured_trading_policy | 缺少成熟交易观察 |
| 1000 | cash | simplex | not_representable | fitted_expert_not_representable | not_representable_even_with_sum_only |
| 1000 | cash | signed_box | not_representable | fitted_expert_not_representable | not_representable_even_with_sum_only |
| 1000 | cash | sum_only | not_representable | 无旧拟合，仅诊断 | not_representable_even_with_sum_only |
| 2000 | cash | simplex | not_representable | fitted_expert_not_representable | signed_box_bound_excludes_witness |
| 2000 | cash | signed_box | not_representable | fitted_expert_not_representable | signed_box_bound_excludes_witness |
| 2000 | cash | sum_only | representable | 无旧拟合，仅诊断 | signed_box_bound_excludes_witness |

500ms 的原专家为 `Library-Ridge-price-h1`，校准净利 +20 USD，但全部
交易候选没有成熟七尺度 OE，不能由现金零向量或次优专家补出几何结论。

1000ms 的现金专家在全部三个集合都不可表示。仅等式证书还可转换为
其他候选的非负凸组合，其七个尺度均值都约为 **0.0254345429416 价格点**。
在 `sum(w)=1` 时，这一组合的奖励仍为该正数，而现金优化得分为0；
因而至少有一个候选得分高于现金，远超过1e-8容差。下表列出全部非专家
可观察候选，零系数也保留；完整精度和矩阵在 JSON 中。

| 候选 | 凸组合系数，约值 |
| --- | --- |
| Ridge-threshold-x1 | 0 |
| Library-Ridge-displayed_volume-h1 | 0.107938978792483 |
| Library-DecisionTree-displayed_volume-h1 | 0.122647924863146 |
| Library-Ridge-price_volume-h1 | 0 |
| Library-DecisionTree-price_volume-h1 | 0.237152888964199 |
| Library-DecisionTree-price-h2 | 0.0332518020660165 |
| Library-Ridge-displayed_volume-h2 | 0.228808420861683 |
| Library-DecisionTree-displayed_volume-h2 | 0 |
| Library-Ridge-price_volume-h2 | 0.164650444001246 |
| Library-DecisionTree-price_volume-h2 | 0.105549540451228 |

该凸组合只是线性分离的数学反例，不是一项新交易策略或真实持仓混合。

2000ms 的现金专家在 simplex / signed_box 下均不可表示，sum_only 则有
容差内可行见证，说明原盒约束确实排除了这一解。对应七尺度的权重为：

| 前瞻 ms | 权重，约值 |
| --- | --- |
| 5000 | 5.08284595988394 |
| 15000 | -6.94000183773459 |
| 45000 | -2.60009202373591 |
| 135000 | 6.42752474253381 |
| 405000 | -6.28057181904419 |
| 1215000 | 4.07793500915101 |
| 3645000 | 1.23235996894593 |

权重和为 `0.9999999999999998`，L1为 `32.641331361029394`，最大绝对权重为 `6.940001837734593`。
最大不等式违例 `5.718648644797578e-17`，等式残差 `2.220446049250313e-16`。
专家减最优候选得分 `-1.0000000057186487e-08`，是 **1e-8容差内匹配**，
不等于严格领先，也不能直接套用现有1e-9奖励并列规则或发布为学习成功。

两个年龄的专家都是现金，身份不在当前纯模型动态候选中。即使几何可行，
也尚未验证该奖励/专家在实际在线策略空间可执行。原 simplex/signed_box
拟合都仍为不可表示，没有被见证覆盖成成功状态。

独立复算通过：六份原拟合完整字典与归档逐字段一致；5份不可行数值证书
和1份可行见证均满足冻结精度，证书最大驻点残差为 `8.881784197001252e-16`。
校准摘要与每日订单统计吻合；源输入字节哈希未变，没有权重激活或新增回测收益。

## 验证与产物

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_experiments.py --rows 3000 --windows 1 --output /tmp/reward-audit-legacy-smoke.json
.venv/bin/python generate_report.py --input /tmp/reward-audit-legacy-smoke.json --output /tmp/reward-audit-legacy-report.md
.venv/bin/python generate_dashboard.py --input /tmp/reward-audit-legacy-smoke.json --output /tmp/reward-audit-legacy-dashboard.html
.venv/bin/python run_reward_audit.py freeze \
  --config config/esz5_reward_audit_development.json \
  --input /tmp/data-mining-library-oe/evaluation/result.json \
  --output-dir /tmp/data-mining-reward-audit-final/frozen
.venv/bin/python run_reward_audit.py run \
  --plan /tmp/data-mining-reward-audit-final/frozen/plan.json \
  --output-dir /tmp/data-mining-reward-audit-final/evaluation
```

180 项全套测试通过（50.787 秒），新增11项覆盖手算边界分离、全常数
候选矛盾、负权敏感性、凸包并列、无信息/未观测专家阻断、数值失败与假
证书拒绝、后段扰动不变、摘要/归档身份/冻结源码完整性和中文CLI联动。
最后将“权重唯一”元字段改为“未评估唯一性”后，11项审计测试再次通过；
未把未评估错误写成数学上证明不唯一。原入口3000行/1窗口及报告、看板
联动通过。新诊断只读取旧校准账本，没有无必要地重跑模型或大行情。

| 文件 | SHA-256 |
| --- | --- |
| 历史输入 | `1334c1345871e8536750c1fa447b43d9ff1cf0c98164961c8c120df799d1ccbe` |
| 新 frozen/plan.json | `f99b90043aa38d447353e81fdb8c774aadff1aec07d5ffa11be8fd52d28a2b51` |
| 新 evaluation/result.json | `a4d14f231ef24be7c6a39e2d09dea0b44f1022c9ef2ae38888138385c88d4fab` |

新计划身份 `4dd810d3e43c0093a6e10b59f5a00d3788b3db8b0035c3e7df100d8e0d097ec8`；历史来源计划 `d73f7b343ecb8c265c718430b34cd03f708ae257552d07d1e83cfed4e284f77e`。
本次实际依赖：python 3.12.14；numpy 2.5.3；pandas 3.0.6；scipy 1.18.1；threadpoolctl 3.7.0。

正式旧结果与最初诊断产物均保留，最终产物另存新目录；仓库不提交原行情
或大 JSON。命令输出目录必须不存在，冻结后更改源码、输入或载荷会拒绝。

## 下一阶段与剩余差异

本次缩小的是“无法判断 IRL 失败是否由项目额外约束造成”的差异。旧拟合
依旧不可用，数值可表示见证只用于诊断，没有把较差或可观测模型换成专家。

2000ms 说明应独立定义仅等式或明确正则化的学习目标并验证策略最优响应，
不能把大幅正负见证直接塞进现有 bounded learner。当前现金专家也不是纯
模型候选入口的一个模型身份；须明确学习策略空间与执行策略空间的关系。
1000ms 表明仅移除盒边界不能解决当前矩阵的目标差异，仍需保留原始价差
奖励与净费用收益口径的区别、成熟覆盖和负结果。

还缺可用 OE IRL 在线联动、时间 ME IRL、真实跨周和全量长期运行。不能
根据这些已用于开发的日期反选参数或声称原文收益复现成功。
