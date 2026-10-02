# 多模型时间 OE 校准与冻结选择验证（2026-10-02）

## 目标与论文依据

在 PR #11 的按周轻模型库基础上，接通预测、真实时间成交、成熟订单 OE、校准
与冻结选择后续评估。开发前重新阅读 `1679894.pdf` 物理第 3–5 页 §3.1–§3.4、
Table 2、Eq.(2)–(6)、Algorithm 1–3，并核对第 7 页实验口径。§3.2 要求不同
特征和历史期的轻模型每周更新；Eq.(2) 是中间价差，Eq.(3) 权重和为 1，
Eq.(5) 是期间订单平均评价。这里只实现有限候选的 Algorithm 1 工程近似，
不把每日回放误称为 Algorithm 2/3 的固定期间在线选择。

候选为现金、原固定 Ridge 门槛 ×1/×2/×4，以及全部 12 个轻模型的原门槛
策略，共 16 个。轻模型来自三组特征 × Ridge/浅树 × 1/2 历史 session。
盘口挂单量替代成交量、特征公式、短历史、树抽样、门槛和时间尺度仍为工程
假设，继承 [模型库验证记录](weekly_library_validation_2026-10-02.md)，未调整
日期或参数来改善本次结果。

## 冻结和因果边界

训练为 2025-09-29/30，校准为 10-01，验证为 10-02，测试为 10-03；全部
已经用于开发，**没有未触碰测试集**。先校准净利专家、OE 权重和候选身份，再
运行后两段。模型 ID 固定算法、特征和历史规则；数值参数允许按预声明周计划
使用此前结束且标签成熟的历史更新。因此这是因果 walk-forward，不能说成
数值模型全程冻结。冻结产物保存实际阶段角色，旧单模型协议中的历史备注不
代表本入口仍未使用校准段。

三份 prepared 数据复用 PR #9 的同批完整 UTC 原始文件，没有前缀或降级输入。
网格 500ms，预测前瞻 5000ms，执行延迟 500ms，持仓复核 15000ms；奖励
前瞻为 5000/15000/45000/135000/405000/1215000/3645000ms。报价年龄
500ms 为严格基线，1000/2000ms 为预声明对照，完整保留，不按收益选年龄。
论文 0.5 秒与每秒四次快照的差异、项目长期尺度扩展仍需披露。

每个候选每天独立账户、选择器和队列，不跨日搬运待成交意图或未成熟反馈。
缺尾未平仓保留，令跨日净利汇总未定义；不以任意尾部价格虚构退出成交。
这个入口没有跨日账户状态或跨日奖励反馈机制。执行器本身可在单次回放换版：
撤销旧待成交意图，保留原仓位和奖励身份，退出订单/OE 归原开仓版本，触发
退出的新版本另记。这是项目账本约定，论文未公开唯一版本归属规则。

OE 用全部尺度完整成熟订单的原始价格差均值，只使用成熟子集，不用未来
标签掩码删除当下决策。零订单的 OE 未定义，现金零向量只是优化约定。专家
按全部可结算净利选择；奖励拟合仅使用成熟订单足够的策略，专家没有成熟 OE
时不能凭盈利强行拟合。最低一单只检查可计算性，不代表统计充分。

成本沿用相同 ES 配置：一手、合约乘数 50、最小价格跳动 0.25、单订单
手续费 1.25 美元、滑点 0.5 个价格跳动，加实测买卖价差。OE 不扣费，
毛利、摩擦成本和净利另列。订单包括开仓和退出成交，不等于交易笔数。

## 真实数据结果

500ms 校准所有交易候选成熟 OE 均为 0，两个约束都返回
`blocked_no_matured_trading_policy`。净利专家为 `Library-Ridge-price-h1`，
54 次成交净利 +20 美元，但后两段分别 -47.5/-897.5 美元；这不能支持论文
盈利或奖励拟合成功的结论。

1000/2000ms 校准专家均为现金，并列原 Ridge ×2/×4 的零交易策略；交易
候选净利均为负。simplex 和 signed_box 均收敛，但最终间隔为负，返回
`fitted_expert_not_representable`。扩充策略集后，当前原始价差特征与约束不能
表示现金净利专家，这与 PR #10 单模型候选结果不同。代码保留权重和全部
诊断、可以展示冻结权重评分，但学习后的可执行选择为 `None`，不拿候选
响应或评价期赢家冒充成功选择。simplex 非负和 signed_box 边界也是额外约束，
不是论文只要求权重和为 1 的完整定义。

| 报价年龄 ms | 权约束 | 求解收敛 | 专家可表示 | 最终间隔 | 学习后的选择 |
| --- | --- | --- | --- | --- | --- |
| 500 | simplex | None | None | None | None |
| 500 | signed_box | None | None | None | None |
| 1000 | simplex | True | False | -0.125 | None |
| 1000 | signed_box | True | False | -0.028854131743459806 | None |
| 2000 | simplex | True | False | -0.05357142857142857 | None |
| 2000 | signed_box | True | False | -0.013655500137878505 | None |

### 冻结对照的实际净利

下表中的专家与等权选择只来自校准。等权是按等权 OE 选择一个候选，不是
混合下单的集成策略；现金零净利仍无实测 OE。两个学习约束在全部年龄都没有
可用选择，其后续策略收益应记为未定义，不能记作零。

| 年龄 ms | 校准选择规则 | 固定候选 ID | 验证净利 USD | 测试净利 USD |
| --- | --- | --- | --- | --- |
| 500 | cash | cash | 0.0 | 0.0 |
| 500 | fixed_reference_x1 | Ridge-threshold-x1 | -24215.0 | -29135.0 |
| 500 | calibration_net_expert | Library-Ridge-price-h1 | -47.5 | -897.5 |
| 500 | equal_weight | None | None | None |
| 500 | simplex | None | None | None |
| 500 | signed_box | None | None | None |
| 1000 | cash | cash | 0.0 | 0.0 |
| 1000 | fixed_reference_x1 | Ridge-threshold-x1 | -27217.5 | -31977.5 |
| 1000 | calibration_net_expert | cash | 0.0 | 0.0 |
| 1000 | equal_weight | Library-DecisionTree-price-h2 | -1797.5 | -2942.5 |
| 1000 | simplex | None | None | None |
| 1000 | signed_box | None | None | None |
| 2000 | cash | cash | 0.0 | 0.0 |
| 2000 | fixed_reference_x1 | Ridge-threshold-x1 | -30862.5 | -33827.5 |
| 2000 | calibration_net_expert | cash | 0.0 | 0.0 |
| 2000 | equal_weight | Library-Ridge-displayed_volume-h2 | -5732.5 | -6127.5 |
| 2000 | simplex | None | None | None |
| 2000 | signed_box | None | None | None |

### 全部候选校准覆盖与后续净利

按配置顺序保留全部候选，不筛选赚钱的模型。校准毛利减摩擦等于净利；
成熟订单数不是全部成交数。后两段完整订单覆盖、OE、费用与各权重评分见
产物 `result.json`；`report.md` 另列覆盖和成本，此处保留所有候选的后续净利。

| 年龄 ms | 候选 | 校准成交 | 校准成熟 OE | 校准毛利 USD | 校准摩擦 USD | 校准净利 USD | 验证净利 USD | 测试净利 USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | cash | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 500 | Ridge-threshold-x1 | 1748 | 0 | 731.25 | 24341.25 | -23610.0 | -24215.0 | -29135.0 |
| 500 | Ridge-threshold-x2 | 2 | 0 | -18.75 | 58.75 | -77.5 | 0.0 | 0.0 |
| 500 | Ridge-threshold-x4 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 500 | Library-Ridge-price-h1 | 54 | 0 | 806.25 | 786.25 | 20.0 | -47.5 | -897.5 |
| 500 | Library-DecisionTree-price-h1 | 114 | 0 | 362.5 | 1617.5 | -1255.0 | -852.5 | -955.0 |
| 500 | Library-Ridge-displayed_volume-h1 | 1220 | 0 | 2725.0 | 16962.5 | -14237.5 | -12020.0 | -11652.5 |
| 500 | Library-DecisionTree-displayed_volume-h1 | 644 | 0 | 368.75 | 9123.75 | -8755.0 | -2527.5 | -2295.0 |
| 500 | Library-Ridge-price_volume-h1 | 942 | 0 | 825.0 | 13202.5 | -12377.5 | -9640.0 | -10227.5 |
| 500 | Library-DecisionTree-price_volume-h1 | 1498 | 0 | 712.5 | 20885.0 | -20172.5 | -23530.0 | -24422.5 |
| 500 | Library-Ridge-price-h2 | 54 | 0 | -75.0 | 792.5 | -867.5 | -12.5 | -367.5 |
| 500 | Library-DecisionTree-price-h2 | 184 | 0 | 212.5 | 2630.0 | -2417.5 | -1302.5 | -1715.0 |
| 500 | Library-Ridge-displayed_volume-h2 | 1262 | 0 | 2175.0 | 17627.5 | -15452.5 | -14947.5 | -14292.5 |
| 500 | Library-DecisionTree-displayed_volume-h2 | 1092 | 0 | 418.75 | 15208.75 | -14790.0 | -11310.0 | -8387.5 |
| 500 | Library-Ridge-price_volume-h2 | 1228 | 0 | 1543.75 | 17191.25 | -15647.5 | -12990.0 | -12827.5 |
| 500 | Library-DecisionTree-price_volume-h2 | 1666 | 0 | -162.5 | 23032.5 | -23195.0 | -26802.5 | -26327.5 |
| 1000 | cash | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1000 | Ridge-threshold-x1 | 2036 | 117 | 2131.25 | 28776.25 | -26645.0 | -27217.5 | -31977.5 |
| 1000 | Ridge-threshold-x2 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1000 | Ridge-threshold-x4 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1000 | Library-Ridge-price-h1 | 70 | 0 | 362.5 | 1025.0 | -662.5 | -82.5 | -1000.0 |
| 1000 | Library-DecisionTree-price-h1 | 126 | 0 | 343.75 | 1801.25 | -1457.5 | -900.0 | -1235.0 |
| 1000 | Library-Ridge-displayed_volume-h1 | 1540 | 88 | 1962.5 | 21725.0 | -19762.5 | -15490.0 | -13765.0 |
| 1000 | Library-DecisionTree-displayed_volume-h1 | 842 | 58 | 18.75 | 11658.75 | -11640.0 | -11045.0 | -6695.0 |
| 1000 | Library-Ridge-price_volume-h1 | 1248 | 48 | 368.75 | 17778.75 | -17410.0 | -12787.5 | -10937.5 |
| 1000 | Library-DecisionTree-price_volume-h1 | 1626 | 149 | -81.25 | 22676.25 | -22757.5 | -32542.5 | -29072.5 |
| 1000 | Library-Ridge-price-h2 | 70 | 0 | -150.0 | 1025.0 | -1175.0 | -90.0 | -387.5 |
| 1000 | Library-DecisionTree-price-h2 | 356 | 2 | 331.25 | 5088.75 | -4757.5 | -1797.5 | -2942.5 |
| 1000 | Library-Ridge-displayed_volume-h2 | 1736 | 66 | 1818.75 | 24626.25 | -22807.5 | -17135.0 | -14755.0 |
| 1000 | Library-DecisionTree-displayed_volume-h2 | 1848 | 83 | 1437.5 | 26235.0 | -24797.5 | -20720.0 | -18592.5 |
| 1000 | Library-Ridge-price_volume-h2 | 1680 | 61 | 456.25 | 23843.75 | -23387.5 | -15570.0 | -14330.0 |
| 1000 | Library-DecisionTree-price_volume-h2 | 1878 | 67 | 556.25 | 26566.25 | -26010.0 | -17017.5 | -14077.5 |
| 2000 | cash | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2000 | Ridge-threshold-x1 | 1470 | 112 | 1593.75 | 20743.75 | -19150.0 | -30862.5 | -33827.5 |
| 2000 | Ridge-threshold-x2 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2000 | Ridge-threshold-x4 | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2000 | Library-Ridge-price-h1 | 88 | 0 | 212.5 | 1285.0 | -1072.5 | 50.0 | -705.0 |
| 2000 | Library-DecisionTree-price-h1 | 122 | 0 | 193.75 | 1746.25 | -1552.5 | -872.5 | -1342.5 |
| 2000 | Library-Ridge-displayed_volume-h1 | 1092 | 78 | 1762.5 | 15390.0 | -13627.5 | -9862.5 | -10707.5 |
| 2000 | Library-DecisionTree-displayed_volume-h1 | 1276 | 409 | 1481.25 | 17576.25 | -16095.0 | -16457.5 | -13992.5 |
| 2000 | Library-Ridge-price_volume-h1 | 938 | 22 | 1062.5 | 13372.5 | -12310.0 | -10042.5 | -7612.5 |
| 2000 | Library-DecisionTree-price_volume-h1 | 2554 | 518 | 712.5 | 35717.5 | -35005.0 | -61145.0 | -60195.0 |
| 2000 | Library-Ridge-price-h2 | 114 | 0 | 6.25 | 1636.25 | -1630.0 | -227.5 | -862.5 |
| 2000 | Library-DecisionTree-price-h2 | 394 | 2 | 1356.25 | 5586.25 | -4230.0 | -2825.0 | -5085.0 |
| 2000 | Library-Ridge-displayed_volume-h2 | 640 | 14 | 775.0 | 9075.0 | -8300.0 | -5732.5 | -6127.5 |
| 2000 | Library-DecisionTree-displayed_volume-h2 | 1794 | 614 | 1243.75 | 24823.75 | -23580.0 | -15622.5 | -14292.5 |
| 2000 | Library-Ridge-price_volume-h2 | 406 | 6 | 118.75 | 5776.25 | -5657.5 | -4862.5 | -3117.5 |
| 2000 | Library-DecisionTree-price_volume-h2 | 3342 | 633 | 850.0 | 46790.0 | -45940.0 | -70220.0 | -63715.0 |

## 可重复性与验证

真实评估仅覆盖一个交易周。三年龄均建成 10-01 启动版和 10-06 周更新版，
本次真实回放只使用启动版；没有 10-06 行情，不能声称验证了真实跨周成交。
七个合成 session 测试覆盖下一周可见更新；另有单次回放换版测试覆盖待成交
撤销、原仓位资金连续性、旧反馈成熟和版本归属。两种证据的范围分开解释。

- `.venv/bin/python -m unittest discover -s tests -v`：142 项通过。
- 小窗口联动：`run_experiments.py --rows 3000 --windows 1`，以及
  `generate_report.py`、`generate_dashboard.py` 都成功；输出
  `/tmp/library-oe-legacy-smoke.json`、`/tmp/library-oe-legacy-report.md`、
  `/tmp/library-oe-legacy-dashboard.html`。日志见 `/tmp/library-oe-full-tests.log`。
- 合成测试核对未来验证/测试价格不能改变此前参数与校准结果；下一周按规则
  使用此前已结束历史则允许变动。不根据未来标签有效性删掉当下决策。
- 实际比较三年龄所有模型参数/版本与 PR #11 旧产物完全相同；固定模型仅新增
  显式 `fitted` 状态，相同训练参数保持不变。
- 三年龄 × 三评价段 × 四旧候选，共 36 份每日结果的所有原有字段，与 PR #10
  原产物逐项相同，包括资金曲线、费用、OE、动作分布。
- 全部 144 份候选/日回放的版本成交和成熟 OE 计数守恒；真实日期仅出现启动版。
  校验当前源码哈希与冻结结果一致，旧结果和原始数据均未覆盖。
- 注入求解失败、校准无成熟订单、尾部未平仓、源码/模型/数据质量产物篡改，
  均验证显式状态或拒绝运行；新输出目录不得覆盖已有产物。

本次实际运行的命令（输出均另存）：

```bash
.venv/bin/python run_weekly_library.py freeze \
  --config config/esz5_weekly_library_development.json \
  --dataset-dirs /tmp/data-mining-snapshot-age-audit/age-500/prepared \
                 /tmp/data-mining-snapshot-age-audit/age-1000/prepared \
                 /tmp/data-mining-snapshot-age-audit/age-2000/prepared \
  --output-dir /tmp/data-mining-library-oe/library-plan
.venv/bin/python run_weekly_library.py build \
  --plan /tmp/data-mining-library-oe/library-plan/plan.json \
  --output-dir /tmp/data-mining-library-oe/library
.venv/bin/python run_library_oe.py freeze \
  --config config/esz5_library_oe_development.json \
  --library /tmp/data-mining-library-oe/library/library.json \
  --output-dir /tmp/data-mining-library-oe/frozen
.venv/bin/python run_library_oe.py run \
  --plan /tmp/data-mining-library-oe/frozen/plan.json \
  --output-dir /tmp/data-mining-library-oe/evaluation
```

JSON 保存源文件、prepared/质量报告/日历、源码哈希、完整配置和模型参数。
`/tmp` 产物为本地验证材料，未提交行情或大体积结果。源码改变后须重新冻结并
另存建库，旧验证记录不能套用到新身份。

| 本次产物 | 身份/文件 SHA-256 |
| --- | --- |
| `library-plan/plan.json` | 文件 `92c9403af3de07ea2cc9bdcf739e7f6b7db0fceb435fed3d7ba149b683893ba2`；计划 `f2cf2a1211d0c2ed285a849ed25e1a60969ea060c6a07ae23e222bc7f14659ee` |
| `library/library.json` | 文件 `ec5f3af1f0a76ca9853394c1570aac103c76e19ec4baebe2aa675030a3bc4009`；计划 `f2cf2a1211d0c2ed285a849ed25e1a60969ea060c6a07ae23e222bc7f14659ee` |
| `frozen/plan.json` | 文件 `4c7f4e949e3a4c5405a4fec244eed0fb2da73bc6013705d3d92f59bb124c14cf`；计划 `d73f7b343ecb8c265c718430b34cd03f708ae257552d07d1e83cfed4e284f77e` |
| `evaluation/result.json` | 文件 `1334c1345871e8536750c1fa447b43d9ff1cf0c98164961c8c120df799d1ccbe`；计划 `d73f7b343ecb8c265c718430b34cd03f708ae257552d07d1e83cfed4e284f77e` |

依赖版本：python 3.12.14；numpy 2.5.3；pandas 3.0.6；pyarrow 25.0.1；scikit-learn 1.9.1；scipy 1.18.1；threadpoolctl 3.7.0。

## 剩余差异与下一阶段

本次缩小的是“多模型库尚未接入真实订单 OE 与冻结选择评估”的差异，仍没有
时间 ME IRL、Algorithm 2/3 固定期间 UCB/ARS、跨日反馈迁移、季度规模
训练/交易或中国商品期货原始实验。当前无法拟合可执行专家的结果属于研究
证据，不能通过挑窗口、改变测试参数或把费用塞入原始 Eq.(2) 来掩盖。

下一阶段优先明确并实现固定期间的选择与成熟订单反馈协议，先用手工可核对
合成数据验证 Eq.(5) 与 Algorithm 2 的时序，再决定真实多周窗口；对未成熟
OE、无订单和不可表示专家保持显式状态。扩到全量数据前仍需真实跨周覆盖、
可比基线和耗时/内存记录。
