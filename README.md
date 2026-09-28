# data-mining-hft

FMATO 思路的因果回测与有限策略复现实验，参考 [原论文](1679894.pdf)。支持轻量模型库、延迟 ME/OE 奖励、UCB/ARS 选择、独立影子账户及美元记账。

**旧版回测包含前视偏差，其收益与延迟结论已撤回。** 当前版本清除跨切分标签，奖励成熟后才更新选择器，按下一事件行情成交，逐事件盯市并在期末平仓。论文未公开的策略优化器使用明确标注的有限策略近似；不宣称等价复刻原生产系统。

## 运行

推荐 Python 3.12，先创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python run_experiments.py
python generate_report.py
python generate_dashboard.py
```

`requirements-lock.txt` 记录生成随仓库结果时使用的具体版本。其他兼容环境可以安装 `requirements.txt`；数值、模型行为与延迟可能随版本和硬件变化，实验会记录实际版本。

默认对每个数据文件选取首段、中段、末段三个不重叠窗口，每窗 60,000 条原始事件。每窗按 50%/15%/15%/20% 分为训练、奖励校准、参数验证、测试；前三段尾部清除至少 90 个事件。窗口内重新拟合，测试不参与调参。每次运行固定随机种子、使用单线程，源代码和数据 SHA-256 随结果保存。

快速验证可另存结果，不覆盖正式产物：

```bash
python run_experiments.py --rows 3000 --windows 1 --output /tmp/smoke.json
python generate_report.py --input /tmp/smoke.json --output /tmp/report.md
python generate_dashboard.py --input /tmp/smoke.json --output /tmp/dashboard.html
```

使用 `--rows`、`--windows`、`--seed` 调整实验；窗口必须不重叠。输入数据优先读取 Parquet，按批次加载请求窗口，兼容同列 CSV。只允许单一合约，按事件时间和交易所序号稳定排序。

## 数据与产物

- `databento_glbx.mdp3_mbp_10.parquet`：CME ES 合约订单簿事件。
- `databento_ifeu.impact_mbp_10.parquet`：ICE Brent 合约订单簿事件。
- [实验结果](results/experiment_summary.json)：切分时间、调参轨迹、模型诊断、策略指标、奖励消融及资金曲线。
- [复现报告](replication_report.md)：从同一结果文件自动生成，注明实现与论文差异。
- [离线看板](visualization_dashboard.html)：单文件，无 CDN 依赖，可切换品种和窗口。

## 评估口径

- 10/30/90 步是订单簿**事件数**，不是秒；实际时间跨度记录在结果中。
- 每窗初始资金 $100,000，固定一张合约。主动成交跨越点差，另计配置滑点和逐笔手续费；不含排队、部分成交、冲击或保证金约束。
- ME 从预测时刻计时；OE 从实际买卖成交计时并扣单边成本，退出归属原开仓模型。ARS 的 OE 为每个候选模型维护独立影子账户。
- 多尺度特征按训练集尺度归一化。专家从校准段可执行静态策略中选取，线性规划与有限策略最优响应交替求权重；权重可能不唯一。
- 对比所有固定模型、验证最佳模型、集成、现金、轮换、三个随机种子，并提供 ME/OE 的等权与三个单尺度 ARS 消融。
- 方向指标附带类别分布、混淆矩阵、平衡准确率、非零方向准确率、零收益和训练多数类基线。
- 单条预测 P50/P95 与批量均摊耗时分开报告。短样本不报告年化夏普；收益率以固定初始资金为分母，最大回撤以完整盯市资金峰值为分母。

现有窗口不能替代多个独立交易日，不足以验证跨周更新或证明实盘盈利。详见报告中的限制与后续工作。
