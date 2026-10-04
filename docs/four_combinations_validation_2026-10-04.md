# 收尾2/4：真实时间ME与最小四组合验证

## 问题与论文依据

PR #19实现了真实多周OE，PR #20交付了报告初稿；本轮补最小真实时间ME，
让ME/OE×UCB/ARS均有执行或阻断条目，并补均值集成参照。用户明确授权
合并#19/#20，本地与远程main已同步到`3d22c09c32ea9dd913d0f4da219f38a3656bf39f`。
本轮从该基线创建独立功能分支，不在已合并分支追加工作。

编码前通读12页原文，并核对物理第3–5页§3.1–§3.4、Eq.(1)–(6)、
Algorithms 1–3，以及第6–8页§4.2、§4.4–§4.5/Tables 4–11。ME从预测
时刻取未来中间价差，OE从成交时刻取差；论文未公开ME期间聚合、极端
千分位筛选和长奖励成熟的完整细节，不能把下列假设写成论文规定。

## 实现及固定范围

- 信号为5秒相对收益预测中`abs(prediction)>1.5e-5`的方向预测；符号乘原始价格点差，不扣费、
  不归一化、不裁剪。卖方向处理和既有门槛筛选是工程假设，不实现千分位。
- ME校准按成熟信号数合并候选均值；在线期间/历史窗均用全部信号分母，
  全部七尺度成熟才更新选择器。零信号、缺目标、跨缺格、未成熟或期间
  未结束都不反馈。ME均值不是Eq.(5)的成交订单均值。
- 校准专家仍来自同16候选的完整美元净利，保留全候选/库内两来源及三
  权约束诊断，只激活sum_only且通过原完整库门控的权重。不换缺成熟
  最佳专家，不把无收益观察补成现金零。
- ME预测反馈独立于成交，因果生成逐时刻选择后由原`TimeExecutionEngine`
  执行；代理消费的动作逐条核对。原执行器、账本、期间UCB和ARS选择
  源码未改动，费用/延迟/持仓/版本归属条件可比，不增加执行器家族。
- 14个既有开发session不变：10-02/03训练、10-06至10-10校准、10-13至
  10-17开发验证、10-20/21开发测试。500/1000/2000ms报价年龄全保留。
  七尺度为5/15/45/135/405/1215/3645秒；500ms网格/延迟、15秒持仓复核、
  5分钟选择、30分钟双ARS、C=1价格点、1/2session历史、种子42均不变。
- 原24项OE/参照继承自PR #19完整结果，有独立源码/文件/计划身份；
  现行旧依赖和环境必须逐字段一致，重新核对全部动态审计与逐日汇总。
  新算均值集成、3个ME等权控制、6个ME学习状态，共34项状态。
  固定模型参照预声明为`Ridge-price-h1`，全部12个固定候选仍保留。

完整结果/快照中的原计划含当次本地绝对路径；迁移位置或环境时保存新
身份。仓库快照是摘要证据，不能代替完整输入明细启动新回放。

历史ME的`latest_signal_required_ns`来自实际最后一个过阈值信号；窗口尾
推导的`reward_observation_latest_required_ns`表示潜在末格前瞻上界。评分
成熟性按实际信号逐项核对，不能把没有产生的末格信号加入分母。七尺度
全部成熟的约定仍不因某个权重为零而缩短；无信号窗口继续不可评分。

## 验证命令

```bash
.venv/bin/python -m unittest discover -s tests -p test_time_me.py -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_four_combinations.py freeze \
  --config config/esz5_four_combinations_development.json \
  --oe-result /tmp/data-mining-multiweek-dynamic-oe-resumable/evaluation/result.json \
  --output-dir /tmp/fmato-final-four/frozen
.venv/bin/python run_four_combinations.py run \
  --plan /tmp/fmato-final-four/frozen/plan.json \
  --checkpoint-dir /tmp/fmato-final-four/checkpoints \
  --output-dir /tmp/fmato-final-four/evaluation
```

新增9项测试通过（73.263秒），覆盖手算预测起点/卖方向、全尺度成熟、零
信号不学习、缺格不缩分母、未来价格/标签隔离、原执行条件等价、双历史
窗可见性、错版/提前反馈/丢分母拒绝，以及七session实际建库的CLI、OE
继承逐字段一致、冻结门控和禁止再次回放的完整日恢复。完整研究检查和
真实结果在下文记录，测试通过不等于统计充分、盈利或论文复现成功。

全量228项测试通过（198.868秒）。3000行单窗口旧入口、报告和看板联动
通过，输出为`/tmp/fmato-four-smoke.json`、`/tmp/fmato-four-smoke-report.md`及
`/tmp/fmato-four-smoke-dashboard.html`；未覆盖或改写仓库正式历史结果。
本轮检查日志为`/tmp/fmato-four-me-tests.log`、`/tmp/fmato-four-full-tests.log`、
`/tmp/fmato-four-smoke.log`和`/tmp/fmato-four-real.log`。

## 剩余限制

ME/OE的工程聚合、阈值与方向符号、有限策略优化、短历史、CME替代市场、
原论文时间频率歧义、严格最近/平移成熟窗口及每日重启仍须披露。四组合
条目完整不等于四种原生产策略均已等价恢复。所有日期已有开发用途；
一次固定范围冻结后段复核和最小奖励消融留给收尾3/4，最终定稿为4/4。
