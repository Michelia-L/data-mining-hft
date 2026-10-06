# 从固定模型到动态选模型：FMATO 的课程部分复现

本项目用于 **10分钟小组展示**。我们研究：在相同交易与成本条件下，用多尺度成交后评价动态选择预测模型，是否比固定模型和简单集成更有效？原方法见 [论文](1679894.pdf)。

目前实现了轻模型库、历史奖励校准、期间 UCB 选择和因果交易回放。现有七日复核中，所有已执行的非现金策略累计亏损，学习权重的改善方向也不稳定。因此结论是完成核心机制的部分复现，当前结果没有支持稳定收益优势。

组员按下面的顺序准备即可，无需阅读源码、配置和原始 JSON：

1. 打开 [可编辑 PPT](presentation.pptx)：8页正文约9分20秒，预留40秒机动；另外3页备用。讲稿也在 PPT 备注中。
2. 阅读 [逐页讲稿与教师问答](presentation.md)，按页分工、计时排练。
3. 用 [简短课程报告](final_replication_report.md)核实方法、实验条件和结论边界。

![五策略七日净利比较](results/presentation/net_comparison.png)

上图是2025-10-22至10-30的七个交易 session，单位为美元。500ms是快照网格；横轴500/1000/2000ms是允许的报价年龄。灰格“阻断”没有收益观察，不能读作现金的零收益。各日账户独立重启，图中金额是日利润加总。

数据路线只保留本地 `data/ESZ5/` 的 Databento CME 行情。当前索引有78个 UTC 日期，其中3日 degraded；实际项目证据涉及21个交易 session，主展示使用最后七日，**没有完成全季度回测**。两份早期根目录 Parquet、旧报告、旧看板及其批次产物已退出当前版本。原始新数据继续忽略，不提交 Git。

查看和核对结果只需要 Python，无需下载行情：

```bash
python run_project.py
python -S scripts/check_project.py
```

维护者推荐使用 Python 3.12，安装 `requirements.txt`。若已有虚拟环境，使用 `.venv/bin/python`：

```bash
# 加做合成数据的因果性、成熟时序和账本检查。
python run_project.py check --core
# 重新导出展示包，输出必须是新目录。
python run_project.py export --output-dir /tmp/course-export
# 完整重跑较耗时；只处理唯一协议所需日期，另存新数据和结果。
python run_project.py prepare --output-dir /tmp/course-inputs
python run_project.py run --prepared-dir /tmp/course-inputs --output-dir /tmp/course-replay
```

默认查看和导出均使用既有证据，不能称为新代码重跑收益。唯一协议是 [course_experiment.json](config/course_experiment.json)；源码位于 `src/`。历史三份快照与两张图在 [证据目录](results/final_report/README.md)按原字节保留，其中也包含 ME、ARS、不利结果与失败状态，不属于组员的默认阅读任务。删减前版本可通过提交 [0338afc](https://github.com/Michelia-L/data-mining-hft/tree/0338afc2e8a67ee22285d05d41e4d78373329859)追溯。
