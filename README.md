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
`--rows` 和 `--windows` 仍表示该文件内部的抽样窗口，**尚未实现跨日训练、定时快照或长期奖励**。
UTC 文件日期不是交易所 session；分区内可能包含事件时间稍早的初始记录。
不传索引参数时，仍运行原来的根目录数据。新入口要求显式指定 `--output`。

索引顶层包含 `dataset=GLBX.MDP3`、`schema=mbp-10`、`symbol=ESZ5`、
`stype_in=raw_symbol` 和 `files` 列表；每条记录包含 `date`、`parquet`、`rows`、
`condition`，可另有 `parquet_bytes`。相对数据路径以项目根目录为基准，不是索引所在目录。
默认拒绝 `degraded` 日期；仅在明确做质量敏感性实验时加 `--include-degraded`。
结果保存索引哈希、所选 Parquet 哈希、文件日期与质量状态；报告和看板也显示质量状态。
大体积原始行情由运行者在本地准备，测试使用临时合成 Parquet，不需要下载这三个月数据。

## 数据与产物

- `data/` 目录中的新增 ESZ5 数据可从 [Hugging Face 数据集 badraldine/datamining_hft_SUFE](https://huggingface.co/datasets/badraldine/datamining_hft_SUFE) 获取；本地按 `data/ESZ5/` 结构存放，供上述按日期实验入口使用。
- `databento_glbx.mdp3_mbp_10.parquet`：CME ES 合约订单簿事件。
- `databento_ifeu.impact_mbp_10.parquet`：ICE Brent 合约订单簿事件。
- [实验结果](results/experiment_summary.json)：切分时间、调参轨迹、模型诊断、策略指标、奖励消融及资金曲线。
- [复现报告](replication_report.md)：从同一结果文件自动生成，注明实现与论文差异。
- [离线看板](visualization_dashboard.html)：单文件，无 CDN 依赖，可切换品种和窗口。

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
