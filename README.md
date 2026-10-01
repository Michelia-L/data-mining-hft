# data-mining-hft

FMATO 思路的课程工程实验，参考 [原论文](1679894.pdf)。**尚未复现原论文完整算法或实验数值。** 当前实现的是事件级 UCB、连续影子账户 ARS、有限策略奖励学习及美元记账，方法对照见 [论文定义与 20 项审阅处理](docs/paper_alignment.md)。

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

这一步缩小时间单位和快照接入差异，尚未实现跨日训练协议、长期奖励校准、按周模型库
或 Algorithm 2/3。运行窗口记录为开发验证用途，多次观察过的窗口不能宣称是未触碰的
正式测试集。验证中的负收益与限制见 [时间回放验证记录](docs/time_backtest_validation_2026-10-01.md)。

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
