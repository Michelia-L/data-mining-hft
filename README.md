# data-mining-hft

FMATO 思路的课程工程实验，参考 [原论文](1679894.pdf)。**尚未复现原论文完整算法或实验数值。** 默认入口实现事件级 UCB、连续影子账户 ARS、有限策略奖励学习及美元记账；另有定时快照、时间 OE 校准和按周轻模型库的独立入口。方法对照见 [论文定义与 20 项审阅处理](docs/paper_alignment.md)。

**旧版回测包含前视偏差，其收益与延迟结论已撤回。** 当前版本清除跨切分标签，奖励成熟后才更新选择器，按下一事件行情成交，逐事件盯市并在期末平仓。论文未公开的策略优化器使用明确标注的有限策略近似；不宣称等价复刻原生产系统。

## 运行

推荐 Python 3.12，先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment-snapshot.txt
python -m unittest discover -s tests -v
python run_experiments.py
python generate_report.py
python generate_dashboard.py
```

`environment-snapshot.txt` 只是生成随仓库结果时使用的环境版本快照，不是包含全部传递依赖和平台条件的完整锁文件。其他兼容环境可以安装 `requirements.txt`；数值、模型行为与延迟可能随版本和硬件变化，实验会记录实际版本。

默认对每个数据文件选取首段、中段、末段三个不重叠窗口，每窗 60,000 条原始事件。每窗按 50%/15%/15%/20% 分为训练、奖励校准、参数验证、测试；前三段尾部清除至少 90 个事件。窗口内重新拟合，测试不参与调参。每次运行固定随机种子、使用单线程，源代码和数据 SHA-256 随结果保存。

快速验证可另存结果，不覆盖正式产物：

```bash
python run_experiments.py --rows 3000 --windows 1 --output /tmp/smoke.json
python generate_report.py --input /tmp/smoke.json --output /tmp/report.md
python generate_dashboard.py --input /tmp/smoke.json --output /tmp/dashboard.html
```

使用 `--rows`、`--windows`、`--seed` 调整实验；窗口必须不重叠。输入数据优先读取 Parquet，按批次加载请求窗口，兼容同列 CSV。只允许单一合约，按事件时间和交易所序号稳定排序。

### 从新增 ESZ5 数据中选择日期

新增 `data/ESZ5/index.json` 索引入口，可以指定一个 UTC 文件日期做实验：

```bash
python run_experiments.py --data-index data/ESZ5/index.json --data-date 2025-09-22 \
  --rows 3000 --windows 1 --output /tmp/esz5-20250922-smoke.json
python generate_report.py --input /tmp/esz5-20250922-smoke.json --output /tmp/esz5-report.md
python generate_dashboard.py --input /tmp/esz5-20250922-smoke.json --output /tmp/esz5-dashboard.html
```

这是向论文多日实验迁移的第一步：本次只运行所选文件的 ESZ5，不会同时运行旧 Brent。
`--rows` 和 `--windows` 仍表示该文件内部的抽样窗口，**该回测入口尚未接入定时快照、跨日训练或长期奖励**。
UTC 文件日期不是交易所 session；分区内可能包含事件时间稍早的初始记录。
不传索引参数时，仍运行原来的根目录数据。新入口要求显式指定 `--output`。

索引顶层包含 `dataset=GLBX.MDP3`、`schema=mbp-10`、`symbol=ESZ5`、
`stype_in=raw_symbol` 和 `files` 列表；每条记录包含 `date`、`parquet`、`rows`、
`condition`，可另有 `parquet_bytes`。相对数据路径以项目根目录为基准，不是索引所在目录。
默认拒绝 `degraded` 日期；仅在明确做质量敏感性实验时加 `--include-degraded`。
结果保存索引哈希、所选 Parquet 哈希、文件日期与质量状态；报告和看板也显示质量状态。
大体积原始行情由运行者在本地准备，测试使用临时合成 Parquet，不需要下载这三个月数据。

### 制作固定时间盘口快照

论文的 tick 是定时间隔快照；Databento 的 MBP-10 文件是一条条不等间隔消息。
下面可从选定的 ESZ5 日期生成 0.5 秒网格快照。先用 `--max-records` 处理前缀检查；
去掉该参数会处理所选文件的全部原始消息。输出路径必须是新文件，不会覆盖旧结果。

```bash
python build_timed_snapshots.py --data-index data/ESZ5/index.json --data-date 2025-09-22 \
  --max-records 300000 --output /tmp/esz5-20250922-500ms-preview.parquet
```

边界时刻只使用此前已收到、带 `F_LAST` 的完整盘口；依据 `ts_recv` 确认行情到达，
并保留交易所原始 `source_ts_event`、原始 `source_ts_recv` 和消息标志。
两个时钟可能未同步；因果采样只依据可信的接收时间，带 `F_BAD_TS_RECV` 的消息不推进网格。
默认只接受年龄不超过 0.5 秒的报价，遇到未完成事件、损坏盘口或长空档则跳过该网格。
输出 `ts_event` 是网格时间，Parquet 元数据记录采样设置、源哈希和是否只读取前缀。
**跳过的网格不能在后续模型中当成连续 tick**；可使用下述数据准备入口划分 session 并隔离时间标签。
这个工具目前只生成快照文件，尚未改变 `run_experiments.py` 的事件级默认实验。

### 整理 session、缺口与时间标签

`prepare_snapshot_dataset.py` 读取上一步生成的快照，按交易 session 缓冲，
输出带特征、标签和有效性标志的 Parquet，以及同一口径的 JSON/中文质量报告：

```bash
# 沿用前缀快照做开发验证，必须显式允许预览；输出目录必须尚不存在。
python prepare_snapshot_dataset.py \
  --snapshots /tmp/esz5-20250922-500ms-preview.parquet \
  --allow-partial --output-dir /tmp/esz5-20250922-dataset-preview
```

正式整理时去掉 `--allow-partial`，提供按时间排列的多个完整快照文件：

```bash
python prepare_snapshot_dataset.py \
  --snapshots data/ESZ5/snapshots/2025-09-21.parquet \
              data/ESZ5/snapshots/2025-09-22.parquet \
              data/ESZ5/snapshots/2025-09-23.parquet \
  --output-dir data/ESZ5/prepared/2025-09-22-23
```

这些正式文件需先用 `build_timed_snapshots.py` 生成；不能以原始 MBP 文件代替。
UTC 午夜不切断同一 session。完整交易 session 通常需要相邻 UTC 分区；
质量报告会列出输入时间范围内未提供的 session，以及每个 session 的前尾和内部缺格。
`coverage_complete` 仅说明该 session 的计划交易网格全部存在，不代表原始消息没有丢失。

默认前瞻期为 `--horizons-ms 5000 15000 45000`，在半秒网格上对应 10/30/90 ticks。
按精确时间生成 `future_price_diff_5000ms` 等价格差标签，另存 `future_return_5000ms` 等相对收益。
`label_end_5000ms` 是计划到期时刻，**不能将离线未来标签作为当下可见信息**。
目标格缺失、区间内任何缺格、交易暂停或 session 结束都会使相应标签失效；
缺格后的动量/波动率重新预热。`feature_valid` 和各尺度 `label_valid_*` 分别表示有效性，
`learning_ready` 要求特征及全部配置尺度的标签都有效。NaN 不补成零。

如已确定训练/校准/验证/测试边界，按升序重复指定
`--split-boundary 2025-09-23T22:00:00Z` 等带时区的时刻；达到或跨过边界的标签被清除。
该工具不会自行选取四段日期，也不会拟合标准化参数。前瞻期必须是网格间隔的整数倍。
默认拒绝降级数据；研究质量敏感性时显式使用 `--include-degraded`，质量状态保留在报告中。

交易日历使用 [版本化 ESZ5 配置](config/cme_es_sessions_2025.json)，覆盖 trade date
2025-09-15 至 2025-12-15，按 `America/Chicago` 处理夏令时和每日暂停。
根据 [CME 感恩节原表（AMP 转载）](https://www.ampfutures.com/hubfs/CME%20Group%20Globex%20-%20Thanksgiving%20Holiday%20Schedule%20-%20November%2026-28%2C%202025.png)，
11/27 的早盘、当晚重开和 11/28 的早收盘归入 11/28 trade date，暂停两侧仍为不同连续区间。
常规时段来源见配置及 [CME 交易时间说明](https://www.cmegroup.com/trading/equity-index/fairvaluefaq.html)。
该配置不是通用交易所日历，越界输入会报错；其他日期需先核对规则并通过 `--calendar` 显式提供。
快照允许 `(open, close]` 边界上的历史盘口，来源报价必须在同一开放时段内收到；
这不表示 close 时刻仍能成交。临时停机通过降级标志与缺口暴露，不推断缺失价格。

本步骤只准备数据。下面的新入口可用这份数据训练固定模型并按时间回放；
`run_experiments.py` 的默认事件实验仍按事件数推进，两种结果须按各自单位解读。
本阶段真实数据检查与样本损失统计见 [session 时间标签验证记录](docs/snapshot_dataset_validation_2026-10-01.md)。

### 用快照训练固定模型并按真实时间回放

`run_snapshot_backtest.py` 是独立的开发验证入口：训练一个冻结的 Ridge，
比较固定模型与现金基线；默认使用等权价格差奖励，尚未接入 IRL 或论文选择器。
必须给出带时区的训练/回放日期，所有区间为 `[start,end)`；下面只验证小窗口链路：

```bash
python run_snapshot_backtest.py \
  --dataset-dir data/ESZ5/prepared/2025-09-22-23 \
  --train-start 2025-09-22T13:00:00Z --train-end 2025-09-22T14:00:00Z \
  --test-start 2025-09-22T14:00:00Z --test-end 2025-09-22T14:15:00Z \
  --prediction-horizon-ms 5000 --latency-ms 500 --holding-review-ms 15000 \
  --detail --output-dir /tmp/esz5-time-backtest
```

输出新目录中的 `result.json` 与中文 `report.md`，拒绝覆盖。`--detail` 增加成交、
完整交易、决策与奖励观察时点，方便组员审计。新结果有自己的 `result_kind`/schema，
不交给旧 `generate_report.py` 或 `generate_dashboard.py`。JSON 保留数据与源码哈希、
来源质量、日历、实际依赖、时间切分、标准化参数与冻结模型系数。

标准化与 Ridge 仅拟合训练样本；训练末端独立隔离最长奖励前瞻期，
即使数据准备时未传 `--split-boundary` 也不允许训练标签跨越训练结束时刻。
训练样本与收益根据当前窗口、精确目标行情和连续区间重新计算，不沿用 prepared
数据生成时旧切分的 `label_valid_*` 筛选；旧有效标签只做一致性审计，旧边界保留在来源记录。
默认预测 5 秒相对收益，奖励用数据集配置的 5/15/45 秒中间价差。
没有有效训练或回放特征时直接报错；`feature_valid=False` 的行情仍用于盯市与风险处理。
前缀、降级数据须显式使用 `--allow-partial`/`--include-degraded`，并保留标记。

执行延迟与持仓复核按毫秒计算。意图只能在到期后首条连续可用行情执行，
不会按下一行提前成交；奖励必须等待最长尺度到期，精确目标不存在或中间有缺口则
不反馈、不伪造零奖励。**回放不以 `label_valid_*` 或 `learning_ready` 筛选决策**，
执行器的输入已移除离线未来标签。

意外缺口到达后撤销旧意图，在首条可交易行情风险平仓；承担缺口期间的价格损益。
日历已知的暂停/收盘采用最后一个严格早于关闭边界的网格退出；关闭边界快照仅供
盯市及奖励观察。若退出格缺失，不能回溯平仓，持仓延续到下一条可交易行情。
期末仍无可交易退出报价时如实保留仓位与未实现盈亏；未成熟奖励单列，绝不提前更新。
主动成交和风险/期末退出均是报价模拟，未模拟队列、冲击、保证金及实盘交易资格。

这一步缩小时间单位和快照接入差异；该短窗口入口没有跨日协议、长期奖励校准、按周模型库
或 Algorithm 2/3。运行窗口记录为开发验证用途，多次观察过的窗口不能宣称是未触碰的
正式测试集。验证中的负收益与限制见 [时间回放验证记录](docs/time_backtest_validation_2026-10-01.md)。

### 冻结完整 session 的跨日开发协议

`run_session_experiment.py` 按完整交易日安排训练、校准、验证和测试。先检查相邻
UTC 原始分区齐备，保存配置、数据、日历和运行源码哈希，再消费冻结文件运行；
运行时不能覆盖日期、门槛或奖励参数。**校准/验证目前仅诊断，没有接入 IRL 或调参。**

```bash
# prepared 需由完整 UTC 2025-09-28 至 10-03 六个快照文件整理，默认拒绝前缀/降级。
python run_session_experiment.py freeze \
  --protocol config/esz5_session_development.json \
  --dataset-dir data/ESZ5/prepared/2025-09-29-10-03 \
  --output-dir /tmp/esz5-session-plan
python run_session_experiment.py run \
  --plan /tmp/esz5-session-plan/plan.json --output-dir /tmp/esz5-session-results
```

示例协议使用 trade date 09-29/30 训练、10-01 校准、10-02 验证、10-03 测试。
日期必须按时间连续、互不重叠；不会因没有特征或亏损删掉某一天。标准化与
Ridge 使用两遍逐日充分统计量训练，每日仅物化当前 session；模型固定后回放三个
诊断阶段。所有日期都记录为开发验证，不声称未触碰的正式测试。

以半秒网格将三倍奖励尺度扩展至 3,645 秒（7,290 网格），重算精确目标、
各尺度/联合有效量及缺口原因；等权价格差仍是工程基线。训练每天末端隔离最长
尺度，预测目标为 5 秒相对收益。原始 UTC 分区齐备不等于网格齐备，缺格不前填。

日账户独立使用固定资金。完整 session 模式不会因文件最后一行而提前强平：
按已知日历退出，关闭边界只盯市；缺少退出报价则保留仓位，汇总净盈亏标为未定义。
短窗口入口仍保留原末行平仓策略。新入口生成独立 JSON/中文报告，结果不可覆盖，
不使用旧报告与看板。数据可用率、负收益和长期校准阻碍见
[跨日协议验证记录](docs/session_protocol_validation_2026-10-01.md)。

### 诊断采样缺格与长期标签可用性

`run_snapshot_audit.py` 在相同原始消息、交易日历和标签规则上，比较预先声明的
500/1000/2000ms 完成盘口年龄上限。默认采样仍为 500ms，诊断不计算策略收益，
不自动选择默认值。工程规则不能当作论文要求；更旧的报价可能不代表可成交盘口。

```bash
python run_snapshot_audit.py freeze \
  --config config/esz5_snapshot_age_audit.json --data-index data/ESZ5/index.json \
  --output-dir /tmp/esz5-age-plan
python run_snapshot_audit.py run --plan /tmp/esz5-age-plan/plan.json \
  --max-age-ms 500 --output-dir /tmp/esz5-age-500
python run_snapshot_audit.py run --plan /tmp/esz5-age-plan/plan.json \
  --max-age-ms 1000 --output-dir /tmp/esz5-age-1000
python run_snapshot_audit.py run --plan /tmp/esz5-age-plan/plan.json \
  --max-age-ms 2000 --output-dir /tmp/esz5-age-2000
python run_snapshot_audit.py compare --plan /tmp/esz5-age-plan/plan.json \
  --results /tmp/esz5-age-500/result.json /tmp/esz5-age-1000/result.json \
            /tmp/esz5-age-2000/result.json --output-dir /tmp/esz5-age-comparison
```

冻结配置、源数据/索引、日历和代码哈希后才运行；仅接受计划里的候选，汇总要求
全部候选齐备。每个候选保存独立快照、prepared 数据、JSON 和中文报告，拒绝
覆盖。原始文件按完整日期流式处理，前缀验证不作为该对照的正式输入。

诊断把已结算网格的状态分为完成盘口超龄、事件未完成、坏盘口标记、非法盘口、
不可信接收状态和缺少完成盘口，并统计可信消息接收静默与产出报价年龄。只累计
交易日历内的网格，休市单独排除；文件首尾未结算的网格记录为未知，不能说成
供应商丢包。扩大年龄仍拒绝未完成/异常盘口、休市前报价和跨缺格标签；不做长
空档前填。每个 session 的原因计数与计划量守恒，逐尺度与联合有效量另列。

真实数据结果及各原因解释见
[快照年龄与长期标签对照](docs/snapshot_age_validation_2026-10-01.md)。

### 校准时间尺度 OE 奖励并冻结后续评估

`run_snapshot_irl.py` 将真实成交后成熟的七尺度 OE 向量接入有限策略 IRL。
训练段拟合冻结 Ridge，校准段按净盈亏选专家、学习权重和选择策略；验证及测试
只消费冻结选择。该入口补充上面的诊断基线，使用同一日期和执行协议；
`run_session_experiment.py` 本身仍只运行等权诊断。

```bash
# 沿用缺格审计生成的三个 prepared 目录，全部候选必须齐备且来自同一批原始消息。
python run_snapshot_irl.py freeze --config config/esz5_time_irl_development.json \
  --dataset-dirs /tmp/esz5-age-500/prepared /tmp/esz5-age-1000/prepared \
                 /tmp/esz5-age-2000/prepared --output-dir /tmp/esz5-time-irl-plan
python run_snapshot_irl.py run --plan /tmp/esz5-time-irl-plan/plan.json \
  --output-dir /tmp/esz5-time-irl-results
```

配置预先声明现金、Ridge 门槛乘数 1/2/4 和 simplex/signed_box 两种权约束。
最大报价年龄 500ms 基线及 1000/2000ms 对照全部保留，不从收益选择年龄。
运行不能覆盖参数，输出新目录的 JSON 和中文报告；源码或数据变化须重新冻结。
这是 Algorithm 1 的有限策略近似，没有实现时间 ME IRL、按周模型库或 Algorithm 2/3。
OE 均值仅覆盖全部尺度完整成熟的订单，零订单为未定义；现金零向量仅用于优化。
静态成交不随权重变化，收益差异只能来自校准时冻结的策略选择。

真实五 session 中，500ms 校准段仍无成熟 OE，明确阻止拟合；1000/2000ms
对照能够求解，但专家和学习后的选择均为现金，没有正利润交易专家。详细结果、
假设与验证见 [时间 OE IRL 验证记录](docs/time_irl_validation_2026-10-02.md)。

### 按周构建多特征与历史期轻模型库

`run_weekly_library.py` 对应论文 §3.2/Table 2：价格、盘口深度、价量交叉
三组特征 × Ridge/浅树 × 两种完整历史 session 窗口，每版 12 个候选。
量特征使用前五档双边挂单量，**不是论文的成交量数据**；完整公式、1/2 session
窗口、树抽样与启动日都是明确工程设定。

```bash
# 复用之前的全部三个 prepared 对照，不重新选日期或更改严格采样。
python run_weekly_library.py freeze --config config/esz5_weekly_library_development.json \
  --dataset-dirs /tmp/esz5-age-500/prepared /tmp/esz5-age-1000/prepared \
                 /tmp/esz5-age-2000/prepared --output-dir /tmp/esz5-library-plan
python run_weekly_library.py build --plan /tmp/esz5-library-plan/plan.json \
  --output-dir /tmp/esz5-library-result
```

输出 `library.json` 保存可序列化 Ridge/树参数、训练行/日期、版本可用时刻和
源码/数据哈希，`report.md` 展示全部候选的预测诊断。路径必须尚不存在；运行不能
覆盖参数。首次启动允许显式周中日期，此后每周第一个实际交易 session 开盘
换版，只使用此前已结束且标签成熟的历史；UTC 文件日期与 trade date 分开。

`src.weekly_library.predict_library(versions, frame, interval_ms)` 接收
`PreparedDatasetReader` 的完整行情，输出 `N×K` 预测矩阵和逐行版本 ID。
按当前可见时刻取版本，启动前或预热不足保持 NaN；JSON 树不使用 pickle。
Ridge 使用全部合法样本，树用固定种子的最多 50,000 行均匀抽样，避免拼接全量
季度训练矩阵。固定原 11 特征 Ridge 对照在启动后保持不更新。

默认开发配置生成 10-01 首版和 10-06 周更新版；输入仅覆盖 09-29 至 10-03，
因此 **10-06 只建库，没有该日预测评估**。原诊断日结束后可成为下一周训练历史，
不再把它们称作未触碰测试。本入口尚未接入时间 IRL、执行器周界状态或论文
选择器，预测误差不等于交易收益。结果与限制见
[按周轻模型库验证记录](docs/weekly_library_validation_2026-10-02.md)。

### 多模型时间 OE 校准与冻结选择交易评估

`run_library_oe.py` 将上述模型库接入相同真实时间执行器：现金、原固定 Ridge
门槛 ×1/×2/×4，加全部 12 个轻模型的原门槛候选，共 16 个。校准段选择专家、
学习 OE 权重和冻结候选 ID，验证/测试不重新选优；模型参数按冻结周规则更新，
因此冻结的是身份和规则，不是全程相同的数值系数。

```bash
# 执行器新增身份审计后源码哈希改变，旧模型产物须另存新计划重建。
python run_weekly_library.py freeze --config config/esz5_weekly_library_development.json \
  --dataset-dirs /tmp/esz5-age-500/prepared /tmp/esz5-age-1000/prepared \
                 /tmp/esz5-age-2000/prepared --output-dir /tmp/esz5-library-oe-build-plan
python run_weekly_library.py build --plan /tmp/esz5-library-oe-build-plan/plan.json \
  --output-dir /tmp/esz5-library-oe-models
python run_library_oe.py freeze --config config/esz5_library_oe_development.json \
  --library /tmp/esz5-library-oe-models/library.json --output-dir /tmp/esz5-library-oe-plan
python run_library_oe.py run --plan /tmp/esz5-library-oe-plan/plan.json \
  --output-dir /tmp/esz5-library-oe-result
```

输出新目录中的 `result.json` 与中文 `report.md`，保留全候选成熟订单/费用/净利、
现金/等权/校准净利专家/学习后的冻结选择和求解失败。`--detail` 增加订单、
交易、决策和到期 OE 的版本 ID，资金退出和奖励沿用原开仓归属，触发退出的
版本另列；换版撤销旧待成交意图，不清零旧仓位或改写未成熟反馈。

沿用每日独立账户，日末未执行意图和未成熟反馈显式记录、不传给下一日模型；
缺尾未平仓使汇总净利未定义。这是工程边界协议，不是论文未公开细节的还原。
静态候选不按奖励切换，线性权重重评不改变同一候选的成交/账本。
完整样本、失败与限制见 [多模型时间 OE 验证](docs/library_oe_validation_2026-10-02.md)。
当前真实窗口未跨周，跨周更新/归属已用合成 session 验证；该入口本身没有
在线 UCB/ARS。下面的独立入口新增期间 UCB 控制，仍不能称为完整 FMATO。

### 固定期间 OE-UCB 与同期间轮换对照

`run_period_ucb.py` 对应论文 Eq.(5)–(6)/Algorithm 2：以 session 开盘锚定
5 分钟墙钟期间，首次有效决策选择一个模型，本期间逐 tick 使用该模型。
`n(a)` 数实际选择期间，`N=sum(n)`；完整期间奖励用全部成交订单均值，
`W(a)` 用已观察期间均值的运行平均，探索项严格使用 `ln(N)`。

```bash
# 先用当前源码另存重建上面的周库，再冻结期间实验；旧产物和结果保留。
python run_period_ucb.py freeze --config config/esz5_period_ucb_development.json \
  --library /tmp/esz5-current-library/library.json --output-dir /tmp/esz5-period-plan
python run_period_ucb.py run --plan /tmp/esz5-period-plan/plan.json \
  --output-dir /tmp/esz5-period-result
```

期间必须结束且其中全部订单的完整多尺度标签都成熟才反馈；缺格整桶不学习，
无订单和末端未成熟均为未定义，不补零。访问数与反馈数分开；无反馈的初始
`W=0` 是算法初始化，不能说成观测到零奖励。期间、持仓复核、预测前瞻与
奖励前瞻分别配置。休市/缺格不压缩墙钟，不为跳过的期间伪造访问。

本入口使用**预声明等权奖励控制**，因为当前多模型 IRL 尚无可用学习选择；
不消费不可表示专家的诊断权重。5 分钟取论文例子，`C=1` 个价格点是项目
设定；模型在期间内锁定、期间均值如何合并、无订单处理和延迟均为原文未
公开的工程约定。没有宣称恢复完整生产算法或论文收益。

保存所有年龄及 UCB/轮换/现金/原 Ridge/12 个固定模型的成本、成熟覆盖和
负结果，共 16 个策略。动态决策仅在当下全部库预测可用时进行，不按未来
标签删样本；静态候选按自身可用性回放。每天独立账户和统计，数值换版
分开 `W/n`，旧奖励仍只更新旧版本；保留原持仓和成交归属，不因换模型强平。
`--detail` 加订单/单笔到期审计，期间选择/反馈及版本统计默认保存。

手算、未来扰动、合成跨周和真实结果见
[固定期间 OE-UCB 验证](docs/period_ucb_validation_2026-10-02.md)。尚缺成功
IRL 奖励联动、时间 ME IRL、真实跨周与全量长期评估；下一入口增加历史窗口 ARS。

### 历史窗口 OE-ARS：严格最近与平移成熟双对照

`run_history_ars.py` 对应论文第 5 页 Algorithm 3：当前模型使用真实账户
最新完整成熟期间 OE，其他模型每次独立回测 30 分钟，按可观察分数最大值
选择下一 5 分钟模型。历史账户每次空仓重启，特征在窗口内重新预热；
不同于旧 `CausalShadowARS` 从回放起点持续运行的影子账户。

最长奖励前瞻为 3645000ms，超过 30 分钟，原文没有说明最近窗口内订单
如何完整成熟。因此同时冻结两种工程对照，不按结果选择其中一种：

- `recent`：严格 `[t-W,t)`。订单未成熟就保留 None，不使用 t 之后价格。
- `matured`：向前平移最长前瞻，`[t-H_max-W,t-H_max)`；评价最多观察到 t。
  这不是论文唯一规定，缺格、跨 session 或版本历史不足仍不能评分。

```bash
# 当前源码身份有效的周库可复用；库依赖源码改变时需另存计划重建。
python run_history_ars.py freeze --config config/esz5_history_ars_development.json \
  --library /tmp/esz5-current-library/library.json --output-dir /tmp/esz5-history-ars-plan
python run_history_ars.py run --plan /tmp/esz5-history-ars-plan/plan.json \
  --output-dir /tmp/esz5-history-ars-result
```

输出 `result.json` 和中文 `report.md`；保留原 16 个策略，加两种 ARS 共 18 个。
历史窗精确末格可交易时主动退出，缺边界报价则完整评分未定义；窗口中任意
订单失效/未成熟、无订单均不补零。旧历史窗口分数不延用；全不可观察时
按声明顺序轮换，负奖励不与虚构初始化零比较。上述边界、冷启动、期间锁定、
当前模型使用最新成熟期间及稳定并列消歧均是工程约定。实时共享账户换模型
不强平，每日状态独立，旧版本反馈不更新新版。

仍使用预声明等权奖励，没有成功 IRL 或完整 FMATO 收益复现。全过程只读
当时可见行情，三种报价年龄及负结果均保留，不增加未触碰测试集的声明。
方法与真实覆盖见 [历史窗口 OE-ARS 验证](docs/history_ars_validation_2026-10-02.md)。

### OE IRL 失败归因：校准可表示性审计

`run_reward_audit.py` 分别检查非负单纯形、`[-1,1]` 有符号盒和仅
`sum(w)=1` 三个集合。最后一个对应 Eq.(3) 的明确条件；前两种为项目
附加约束。可行时保存最小 L1 诊断见证，不可行时保存可重算的数值证书，
区分“附加边界排除了专家”和“当前特征/专家在仅等式条件下仍不可表示”。

```bash
# 输入是已有多模型 OE 结果，允许保留其旧源码身份；只审计其中校准段。
python run_reward_audit.py freeze --config config/esz5_reward_audit_development.json \
  --input /tmp/esz5-library-oe-result/result.json --output-dir /tmp/esz5-reward-audit-plan
python run_reward_audit.py run --plan /tmp/esz5-reward-audit-plan/plan.json \
  --output-dir /tmp/esz5-reward-audit-result
```

输出新建目录的 `result.json` 和中文 `report.md`。重算校准摘要以核对
每日账本，保留所有年龄、旧专家/拟合、资格排除原因、矩阵和证书；验证/
测试数值不参与诊断。历史输入文件、历史源码身份和本次诊断源码分别绑定。

最小 L1 是诊断消歧，不是 Algorithm 1 的学习目标；见证不用于交易。
真实归档中 500ms 缺成熟观察、1000ms 仅等式仍不可行、2000ms 仅等式
可行但盒约束不可行。2000ms 见证最大绝对权重约 6.94，现金专家身份
也不在当前模型候选中，不能说已获得可用的动态交易奖励。
证据与限制见 [OE IRL 可表示性审计](docs/reward_audit_validation_2026-10-02.md)。

## 数据与产物

- `data/` 目录中的新增 ESZ5 数据可从 [Hugging Face 数据集 badraldine/datamining_hft_SUFE](https://huggingface.co/datasets/badraldine/datamining_hft_SUFE) 获取；本地按 `data/ESZ5/` 结构存放，供上述按日期实验入口使用。
- `databento_glbx.mdp3_mbp_10.parquet`：CME ES 合约订单簿事件。
- `databento_ifeu.impact_mbp_10.parquet`：ICE Brent 合约订单簿事件。
- [实验结果](results/experiment_summary.json)：切分时间、调参轨迹、模型诊断、策略指标、奖励消融及资金曲线。
- [复现报告](replication_report.md)：从同一结果文件自动生成，注明实现与论文差异。
- [离线看板](visualization_dashboard.html)：单文件，无 CDN 依赖，可切换品种和窗口。
- [复现瓶颈评估](docs/bottleneck_assessment_2026-09-30.md)：论文、旧结果和新增数据的对照，注明评估基线与后续进度。

## 分支与开发约定

`main` 保存已合并版本；每项新工作从最新 `main` 创建功能分支，小步提交并发布到对应远程分支，再通过 PR 合并。已合并的 PR 分支不继续追加新功能。功能分支已发布不代表 GitHub 默认展示的 `main` 已更新。

本地数据、备份分支及本轮整理记录见 [仓库状态与开发流程](docs/repository_workflow.md)。

## 评估口径与论文差异

- 默认 `paper_price_difference` 按 Eq.(2) 计算绝对价格差，不缩放、不裁剪。`--reward-definition normalized_return` 单独运行训练 P99 缩放、裁剪的收益率变体，建议用 `--output /tmp/normalized.json` 保存。两种奖励的数值单位与 UCB 系数含义不同。
- Eq.(3) 明确要求权重和为 1，但没有明确要求非负。默认非负单纯形是项目假设，另报告允许负权的 SignedBox 敏感性。
- Eq.(5) 的 OE 校准均值除以成熟订单数；零订单均值标记为未定义，优化中约定零向量。默认不扣奖励成本；NetOE 固定权重敏感性单列。账本始终扣实际成本。
- `CausalEventUCB` 每事件选择、每成熟决策一次反馈，无成交为零。`CausalShadowARS` 使用连续影子账户与 300 个奖励到达事件窗口。两者**没有实现**论文 Algorithm 2/3 的固定期间更新与历史时间窗口重新回测，策略名已撤回 FMATO-UCBS/ARS 的等价表述。
- 五种算法使用相同训练段与特征集，未实现论文不同历史时期/特征子集的模型库及每周更新。ME 使用共同交易门槛，不等于论文强调的 top-1/1000 极端信号协议。
- 10/30/90 步是订单簿**事件数**，不是秒或定时快照。每窗初始资金 $100,000、固定一张合约；主动成交、期末平仓、逐事件盯市，不含排队、部分成交、冲击或保证金约束。
- 门槛先在验证段选定，再用于校准与测试。专家、门槛、c、固定模型都保存并列候选与预先声明的消歧规则。同一短验证段反复选参存在选择过拟合，尚未实现嵌套时间验证。
- 预测指标区分原始收益符号、门槛后的三类信号、有效信号准确率与覆盖率；逻辑回归分类 argmax 独立报告。无有效信号时准确率记为 null，而非零或满分。
- 对比所有固定模型、验证选定模型、集成、现金、轮换、三个随机种子、八个权重消融、两个负权敏感性与一个 NetOE 敏感性。报告和看板从 schema 3 JSON 自动生成。
- 单条预测 P50/P95 与批量耗时分开保存，不等于完整交易链路延迟。短样本不年化夏普。

CME/ICE 数据不同于论文中国期货市场、品种与制度，因此当前结果不复现论文数值。文件内多个窗口可能相关，不能视为跨日独立重复，不能验证跨周泛化或证明实盘盈利。下一阶段需要多日数据、论文模型库设计、固定时间选择协议以及更真实的执行模拟。
