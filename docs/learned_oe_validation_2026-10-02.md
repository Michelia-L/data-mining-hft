# OE 学习与在线选择联动验证（2026-10-02）

基线为 PR #15 合并后的 main `31fddf32040ef8247c5b7046cbe686768468ff45`。
本次建立校准学习到 UCB/双 ARS 的因果接口，并区分数值拟合与在线可执行性。
真实开发日期仍未得到可执行学习奖励，合成数据成功不等于真实论文复现成功。

## 论文依据与目标

在本会话全文阅读基础上，重新核对 `1679894.pdf` 物理第3–5页问题定义、
模型库、Eq.(2)–(6)、Algorithm 1–3，及第7–8页奖励与变体对照。
原文 Eq.(3) 仅明确权重和为1，没有公开完整专家生成细节与生产参数优化器。
因此保持原校准净利专家和资格口径，显式新增有限策略最大间隔近似。

令 `D_j=mu_expert-mu_j`，先求 `max margin`，约束 `D_j·w>=margin`、
`sum(w)=1`。完整 oracle 包含专家自身，零差行将 margin 封顶0；不再
在无限正负权的未完整约束子集上求解。再固定同一最优 margin，最小化
`sum(abs(w))` 消歧。此两阶段学习不等于上次仅可行性诊断，也不等于
论文未公开的连续策略优化。两阶段复核原始/对偶与 oracle 误差，保留负间隔。

原 simplex/signed_box 同矩阵拟合仍单独记录；不把权重边界下的失败视为
仅等式下失败，不用失败权重交易。专家可表示不代表奖励唯一或交易盈利。

## 动作集与时序

在线保留原12个轻模型。现金与原 Ridge 门槛对照用于校准/评估，不作为
额外在线动作。门控要求拟合成功、专家属于原库、库中全部候选满足成熟
校准 OE 资格；这是保守项目假设，不能写作论文规定。未通过时记录原因，
三项学习策略的成交、收益皆为 null，与真实运行的现金零收益区分。

校准 μ 沿用成熟订单子集均值；在线期间与历史子回测使用各自全部订单，
任一订单无效/未成熟均不能评分。每个年龄先完成校准学习并绑定身份，
再进入验证/测试。冻结权重同时传入 UCB 期间反馈与双 ARS 历史重回放，
不能复用等权历史评分。当前模型参数可按预声明周规则变化，权重不再重学。
每日账户/反馈独立；最近与平移成熟双窗口、C=1价格点、5分钟选择期、
30分钟历史窗等沿用已披露控制。反馈不扣费，美元账本扣费，大正负权
保留原值，探索相对尺度变化属于限制而非按测试调参。

## 验证命令

```bash
.venv/bin/python -m unittest discover -s tests -p test_learned_oe.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_experiments.py --rows 3000 --windows 1 --output /tmp/data-mining-learned-oe-smoke.json
.venv/bin/python generate_report.py --input /tmp/data-mining-learned-oe-smoke.json --output /tmp/data-mining-learned-oe-smoke-report.md
.venv/bin/python generate_dashboard.py --input /tmp/data-mining-learned-oe-smoke.json --output /tmp/data-mining-learned-oe-smoke-dashboard.html
.venv/bin/python run_learned_oe.py freeze --config config/esz5_learned_oe_development.json \
  --library /tmp/data-mining-period-ucb/library/library.json --output-dir /tmp/data-mining-learned-oe/frozen
.venv/bin/python run_learned_oe.py run --plan /tmp/data-mining-learned-oe/frozen/plan.json \
  --output-dir /tmp/data-mining-learned-oe/evaluation
```

新增11项测试覆盖仅等式大权重手算、不可表示专家的负最优间隔、完整 oracle
并列、缺观察阻断、冻结权重校验、成熟反馈、两类历史评分同权重、未来扰动、
CLI 与嵌套协议源码身份。手工校准/序列化模型测试用于隔离数学与执行性质。
另有七个合成 session 的实际建库→实际校准→学习→跨周执行的 CLI 测试，
通过动作门控且三个学习策略实际运行；一个旧控制日期逐项完全一致。

新增11项及全套191项均通过（全套76.623秒）；3000行旧入口烟雾实验、
中文报告与看板生成通过。新增CLI直接生成同源JSON与中文报告，旧报告/
看板的检查仅说明旧入口兼容，不代表它们能直接读取新增schema。

## 真实开发结果

沿用09-29/30训练、10-01校准、10-02验证、10-03开发测试，500ms网格，
奖励前瞻 `[5000,15000,45000,135000,405000,1215000,3645000]`ms，
全部500/1000/2000ms报价年龄均运行，不选有利年龄。使用已有当前源码兼容
库 `/tmp/data-mining-period-ucb/library/library.json`，旧产物保留。

| 年龄 ms | 仅等式校准状态 | 结果解释 |
| --- | --- | --- |
| 500 | blocked_no_matured_trading_policy | 无足够成熟交易观察，不导出学习权重 |
| 1000 | fitted_expert_not_representable | 仅等式完整最大间隔仍为负，不能解释原净利专家 |
| 2000 | fitted | 可解释现金专家，但专家不在动作集，且库校准OE不完整，阻断执行 |

三种年龄的学习 UCB/双 ARS 都未执行。原18个控制照常运行并保留全部
收益和订单失败覆盖；没有用等权结果、现金收益或诊断见证填补学习策略。

1000ms 最优间隔为 `-0.02543454294159724` 价格点，与此前仅等式
不可表示证书一致。2000ms 最优间隔为0，专家与 oracle 数值差
`-2.0816681711721685e-17`，权重为：

```text
[5.082849818803779, -6.940007459873251, -2.6000937754591944,
 6.42752903608299, -6.280576665633624, 4.077938024017188,
 1.2323610220621115]
```

2000ms 最大绝对权重约6.940007，L1约32.641356，仍无在线激活权重。
两个可求解年龄的全部原始/对偶复核最大误差 `1.1368683772161603e-13`，
低于冻结容差 `1e-8`。2000ms 的3个库候选缺校准OE（两个price-h1、
Ridge-price-h2），专家为现金；仅等式可表示无法消除执行空间与覆盖阻碍。

与上次真实 ARS 结果逐个对象比较，`3年龄×2日×18策略=108` 个策略日
结果完全一致。新增 `3年龄×2日×3策略=18` 个策略日如实阻断。
新计划身份 `9de80b90bbf3e0215cd984dfc55c79a1b5917edc7ff0c14a7cf50ded387cc4fb`；
结果字节 SHA256 `fd143c21ed9cc9c21d3409465ac5ff49156517c1ffcfe30605a4c8fb35948ddc`。
旧对照保留在 `/tmp/data-mining-history-ars/evaluation/result.json`，新输出为
`/tmp/data-mining-learned-oe/evaluation/{result.json,report.md}`。

## 距离正式实验

学习与执行接口已具备，当前研究阻碍是长期成熟标签覆盖与校准专家/在线
策略空间一致性。下一项应在新的真实多周开发区间扩大成熟覆盖，按冻结
规则检验专家及模型库；若仍失败应保留结论，不能只扩大数据后声称成功。
之后仍需时间 ME 的学习/选择对照、四变体范围审计、真实跨周、正式实验
协议与独立长期评价。根目录旧报告与看板仍为旧事件实验，非正式全量报告。
