# 收尾3/4：冻结后段复核与最小奖励消融

## 收益运行前声明

2026-10-05，从合并PR #21后的`main`（`1d36874`）开始，重新通读论文
12页并核对PDF物理第3–5页§3.1–§3.4、Eq.(1)–(6)、Algorithms 1–3及
第6–8页§4.2、§4.4–§4.5、Tables 4–11。论文Eq.(3)要求权重和为1，
§4.4对比短尺度、人工线性组合与学习奖励；本轮只补最小赋权消融，
不声称完整重现其生产策略和三个基线。

在该段策略收益运行前保存[配置](../config/esz5_final_frozen_ablation.json)：
旧末日2025-10-21后首七个交易session，即10-22/23/24/27/28/29/30。
七日与开发评价长度相同，覆盖10-20旧版与10-27新版；不按收益选日期。
新增范围小于十session上限，不扩至全季度。原始季度文件已经在本地，
仅处理构成这七日与必要训练日的分区。历史用途无法完全证明未触碰，
统一称冻结后复核，不称未触碰测试或论文三个月长期评估。

旧版训练16/17日，10-27版仅用已经结束的23/24日；最近1/2session、
种子42、浅树抽样、500ms网格/延迟、5秒预测、15秒持仓复核、5分钟
选择、30分钟双ARS、三种报价年龄、成本等全部沿用。
ME/OE奖励及门控从PR #21五日校准证据冻结继承，不使用后段重新拟合。

冻结计划沿用旧协议容器：`sessions.train`列16/17/23/24的在线训练历史
并集，`sessions.test`列七日评价；它不是把未来23/24提前交给10-20版的
普通全局切分。实际逐版训练日期与成熟边界由周计划、版本和训练审计确定。

27项状态包括：两奖励×等权/最短尺度×UCB/双ARS共12控制，两奖励×
两专家来源×UCB/双ARS共12学习状态，现金、首候选`Ridge-price-h1`、
均值集成三参照。无成熟信号、专家不在动作库或不可表示等失败保持阻断。
固定候选是固定模型ID，数值模型按声明周计划更新。

最短尺度组在原七维向量上赋`[1,0,0,0,0,0,0]`，仍等全部七尺度成熟；
这隔离赋权贡献，避免同时改变反馈时延。它不是论文只观察10tick的等价
实验。严格最近与平移成熟窗都保留，不能因反馈稀疏改窗或缩短成熟条件。
源质量非available、前缀/降级/缺必要分区或无行情时拒绝该年龄日，保留缺测；
普通内部缺格沿用原规则使相关标签/订单/信号失效，不任意前填。

## 验证及结果

三年龄×七日×27项共567状态：399执行、168阻断。全部汇总与逐日状态见
[报告§5.5](../final_replication_report.md)、[归档快照](../results/final_report/frozen_ablation_snapshot.json)。
所有执行日均已平仓；三个年龄下所有非现金策略的七日净利均为负。
库内学习相对等权有少亏也有多亏，不支持稳定改善。1000ms OE-UCB
少亏5440美元来自摩擦减少9565美元，毛利反而减少4125美元；不能解释为
预测质量提升。完整毛利/成本差、UCB成熟期间与双ARS反馈覆盖保留在报告。

## 所需分区及数据准备命令

仅使用UTC分区10-15/16/17/21/22/23/24/26/27/28/29/30。
前三日覆盖旧10-20版的16/17训练；21～30覆盖七日评价与23/24后周训练。
沿用旧产物的15/16/17/21快照，其余八个分区新采样；不读取其它季度分区。
输入源均必须是完整available分区，内部缺格不被改成连续tick。
以下命令复用原准备函数，不增加采样/标签算法；输出路径必须另选新目录。

```bash
# 项目虚拟环境中，从仓库根目录执行；需要PR #21绑定的旧prepared/快照。
python - <<'PY'
from pathlib import Path
import json
from src.data_catalog import select_daily_source
from src.snapshot_dataset import prepare_snapshot_dataset
from src.timed_snapshots import write_timed_snapshots

root = Path('/tmp/fmato-ablation-data')
root.mkdir(exist_ok=False)
config = json.loads(Path('config/esz5_final_frozen_ablation.json').read_text())
(root/'pre_returns_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
prior = json.loads(Path('results/final_report/four_combinations_snapshot.json').read_text())
base = prior['plan']['base_oe_dynamic_plan']['base_ars_plan']['base_period_plan']
dates = ['2025-10-15','2025-10-16','2025-10-17','2025-10-21','2025-10-22',
         '2025-10-23','2025-10-24','2025-10-26','2025-10-27','2025-10-28',
         '2025-10-29','2025-10-30']
for age in config['age_candidates_ms']:
    old = next(c for c in base['cases'] if c['max_age_ms'] == age)
    quality = json.loads((Path(old['session_binding']['dataset']['path'])/'quality.json').read_text())
    available = {i['metadata']['source_date_utc']: Path(i['path']) for i in quality['inputs']}
    output = root/f'age-{age}'/'snapshots'
    output.mkdir(parents=True)
    paths = []
    for day in dates:
        if day in available:
            paths.append(available[day])
            continue
        source = select_daily_source('data/ESZ5/index.json', day)
        target = output/f'{day}.parquet'
        write_timed_snapshots(source['source_file'], target, interval_ms=500,
            max_age_ms=age, source_date=day, condition=source['condition'])
        paths.append(target)
    prepare_snapshot_dataset(paths, root/f'age-{age}'/'prepared-seven',
        horizons_ms=[5000,15000,45000,135000,405000,1215000,3645000])
    print('完成数据准备', age, flush=True)
PY
```

按README的freeze/run命令绑定上述新prepared目录即可。数据来源、每个输入
快照和原始文件SHA在quality.json中保留；冻结计划再绑定prepared Parquet与
质量JSON的SHA。正式复核本地路径是`/tmp/fmato-final-ablation`，上述示例
使用新路径以保护已完成产物。旧全量结果SHA及快照身份必须一致，否则拒绝。

## 实际验证与身份

- 全量234项测试通过，196.955秒；最后首候选身份/未平仓显示细化后，相关6项通过，14.532秒。
- 3000行旧实验、报告与看板联动通过：`/tmp/fmato-final-ablation-smoke*`。
- 先保留配置声明，再准备数据；冻结检查拒绝默认三尺度，显式七尺度另存`prepared-seven`后才运行收益。原三尺度/拒绝日志保留，范围参数未变。
- 新三年龄仅处理八个新UTC分区，另引用四个既有快照分区；所有源available、完整非前缀，无降级放行。
- 三年龄/两个真实版本/27状态/567日条目全部齐全，21个日回放与1个周扩展检查点完成。
- 禁止周扩展重训与日交易后恢复22个完整单位，完整JSON的SHA256与首次发布逐字节相同。
- 独立核对声明、源码Git对象、原结果SHA、旧权重/初版、训练日期、全部金额/汇总、成熟审计和文档图表，一致；`git diff --check`通过。
- 计划：`5b3c9b8eaae508462edc7e620572e797895e4cf0a1298722f36bccc475919f7f`；结果：`c2afc09be43d0282a27fc53b79e18cd680f78ecd0e63cb230e2ad07779a0b524`；快照：`3765d39e51748a72147e86653d0ca0edfcaf50d3e452a262962bc56fdc1e85b6`。
- 数值提交：`c406441c504cf8612822ed2b0bbb0242b63f2b46`；本轮后续仅文档/证据/图片，不重复训练或交易。
- 正式本地产物：`/tmp/fmato-final-ablation/frozen`、`evaluation`、`checkpoints`；源码/数据/日历/环境及完整日payload身份保存在计划与结果中。
- 数值提交的[GitHub CI](https://github.com/Michelia-L/data-mining-hft/actions/runs/37255426312)通过。

实测过程中对若干完整日并行预计算检查点，所有任务直接调用同一冻结
`replay_day`、预测/日审计与原检查点入口，各任务计算线程为1。随后原
`run_ablation`恢复并重新审计全部21日；只改变离线计算顺序，没有改变
虚拟行情时钟、信息可见性、模型更新或策略状态，也没有筛选有利日期。

图源只有同一结果JSON。CLI首次导出图保留在`evaluation`；审阅发现图例
遮挡阻断标注后，仅在同一`plot_ablation`函数外设置图例位置，另存
`figures-reviewed`再归档，不重训或重交易。该样式导出的复核方法如下，
输出必须另选新目录；图像字节不冒称与CLI默认样式相同。

归档逐日图SHA256：`176f9d925c9e9603f43698a370de7cfbd832b3291b42f52c8f88da995083407e`；
累计图SHA256：`a6eb4778466fdfec88e15f66bbe432de5a712d3fa4763557b1afde0c9278f2d7`。

```bash
MPLCONFIGDIR=/tmp/fmato-ablation-mpl-cache python - <<'PY'
import json
from pathlib import Path
import matplotlib
from src.frozen_ablation import plot_ablation
result = json.loads(Path('/tmp/fmato-final-ablation/evaluation/result.json').read_text())
output = Path('/tmp/fmato-ablation-new-figures')
output.mkdir(exist_ok=False)
with matplotlib.rc_context({'legend.loc': 'center left'}):
    plot_ablation(result, output)
PY
```

外部旧完整结果约625MiB，完整期间明细由日检查点承载；仓库只归档同源摘要、审计、模型与身份，不冒称包含全部原始消息。后周子建库入口的首版记录为bootstrap，是扩展子任务启动标签；全局日历核验强制其为周首session，不在看到收益后择日。
