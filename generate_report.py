"""从实验 JSON 自动生成中文复现报告，保持文字表格与看板的数据来源一致。

本模块不重新训练、不重新调参，只把已经保存的指标和诊断转换成 Markdown。
方法边界是固定说明；窗口时间、收益、权重和专家匹配情况都从结果中提取，
避免手工复制数字后忘记同步，或根据某次漂亮结果预先写死结论。"""
import argparse
import json
from pathlib import Path
from src.config import BASE_DIR, SUMMARY_JSON_PATH


def render_report(data):
    """把 schema_version=2 的实验结果组织成报告字符串，不写文件。

    先解释方法口径，再逐品种展示窗口、策略汇总、模型预测与奖励诊断，
    最后从全部窗口统计现金专家和无法表示专家的数量。
    金额保留两位小数、权重/间隔保留更多位只是展示格式，原始精度仍在 JSON 中。
    lines 中的空字符串用于生成 Markdown 空行，保证标题和表格正确渲染。"""
    lines = [
        '# FMATO 因果回测与有限策略复现实验', '',
        '> 本报告由 `generate_report.py` 从实验 JSON 自动生成。旧版含前视偏差的收益结论已撤回。', '',
        '## 方法与复现边界', '',
        '参考本仓库 `1679894.pdf`：Auto-tuning of price prediction models for high-frequency trading via reinforcement learning。', '',
        '- 使用异构轻量模型库与 ME/OE、UCB/ARS 框架；不是原论文生产系统的逐项等价复刻。',
        '- 原论文 §3.1 使用定时行情快照；本项目的 10/30/90 步均指清洗并排序后的订单簿事件，不是秒，也不是最小报价跳动。仅三个尺度，不涵盖论文的一小时以上尺度。',
        '- 每个窗口按 50%/15%/15%/20% 分为训练、奖励校准、参数验证、测试。前三段尾部至少清除 90 个事件，保证所有标签终点严格早于下一段；标准化和奖励尺度仅由训练段估计。',
        '- 训练后冻结模型；校准段比较五个固定模型、等权集成与现金策略的实际净盈亏，选出可执行专家。ME/OE 分别学习权重。测试区间不参与专家选择或调参。',
        '- 奖励学习实现有限策略 apprenticeship 近似：交替求解最大间隔线性规划、在候选策略库中重新求最优策略、更新特征期望约束，记录完整迭代与终止误差。它不是原论文未公开的连续参数优化器；权重可能不唯一或位于单一尺度，不能解释成已发现稳定市场规律。',
        '- 每个尺度先除以训练段绝对收益的 99% 分位数，再加权并裁剪到 [-1,1]，使 UCB 的奖励与探索项具有一致尺度。统一交易门槛由验证段的静态集成表现选取，UCB 系数分别在验证段调优。',
        '- ME 从信号时刻计时；OE 从每笔实际成交（开仓和退出）计时并扣该笔成交成本。退出订单归属原开仓模型。只有最长尺度到期后才反馈，不读取回测标签。OE 衡量成交后盯市期望，不等于一笔完整交易的净盈亏。',
        '- ARS 的 OE 为每个候选模型维护独立影子账户，按已成熟成交更新事件时间滑窗。冷启动轮换，未观察奖励记为零；UCB 将选择次数与已收到反馈次数分开。',
        '- 交易信号默认下一事件成交。买入按卖一加滑点，卖出按买一减滑点，按一张合约及合约乘数记美元账；手续费逐成交扣除。持仓超时或反向信号平仓，期末强制平仓。',
        '- 初始资金 $100,000，每事件按中间价盯市，最大回撤由完整资金曲线计算。曲线仅在展示时抽样，保留终点。当前样本不足以估计可靠年化夏普，输出 null。',
        '- 单事件推理延迟为预热后的 200 次 P50/P95；批量均摊耗时另列。未测订单链路端到端延迟，不能由此宣称达到实盘性能。', '',
        '## 实验配置与可复核性', '',
        f"- 根随机种子：{data['metadata']['seed']}；每品种窗口数：{data['metadata']['windows']}；每窗原始事件：{data['metadata']['rows']:,}。",
        f"- Python：{data['metadata']['python']}；计算线程：{data['metadata']['threads']}。",
        '- 数据 SHA-256、源代码 SHA-256、实际依赖版本、切分时间、归一化参数、调参候选、奖励迭代均记录在 `results/experiment_summary.json`。',
        '- 随机基线使用三个独立固定种子；窗口内分别重置模型与选择器。不同时间窗口可能相关，不能视为独立交易日；下表不提供虚假的置信区间或显著性结论。',
        '- 消融固定 ARS、成本、门槛和尺度归一化，只替换权重：校准权重、等权、单独 10/30/90 事件。单模型基线全部保留，并提供仅由验证段选出的最佳模型。', '',
    ]
    # experiments 按品种组织，windows 保存各时间段，aggregate 是同名策略的跨窗汇总。
    for key, experiment in data['experiments'].items():
        lines += [f'## {key}', '', f"数据：`{experiment['source_file']}`，总行数 {experiment['source_rows']:,}。", '',
                  '| 窗口 | 原始偏移 | 测试起点 UTC | 测试终点 UTC | 测试秒数 | 不变类别占比 |',
                  '|---|---:|---|---|---:|---:|']
        for i, w in enumerate(experiment['windows'], 1):
            t = w['partitions']['test']
            flat = w['prediction_baselines']['class_proportions']['0']
            lines.append(f"| {i} | {w['source_offset']:,} | {t['start']} | {t['end']} | {t['duration_seconds']:.2f} | {flat:.2%} |")
        lines += ['', '### 各窗口净盈亏汇总（美元）', '',
                  '每窗资金重置；合计是各次独立回放盈亏之和，不是连续账户年化收益。', '',
                  '| 策略 | 总净盈亏 | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in experiment['aggregate']:
            lines.append(f"| {r['strategy']} | {r['net_pnl_usd_sum']:.2f} | {r['net_pnl_usd_mean']:.2f} | {r['net_pnl_usd_min']:.2f} | {r['net_pnl_usd_max']:.2f} | {r['total_trades']} |")
        for i, w in enumerate(experiment['windows'], 1):
            lines += ['', f'### 窗口 {i}：模型诊断与校准', '',
                      f"验证段选定门槛 `{w['tuning']['threshold']}`；最佳固定模型 `{w['tuning']['validation_best_model']}`；UCB 系数 `{w['tuning']['c_by_reward']}`。", '',
                      '| 奖励 | 10 / 30 / 90 权重 | 迭代数 | 间隔求解收敛 | 最终间隔 | 专家可表示 | 校准专家 |', '|---|---|---:|---|---:|---|---|']
            # 同时展示求解收敛、最终间隔和专家可表示性，防止把数值收敛等同于成功模仿。
            for reward, item in w['rewards'].items():
                d = item['diagnostics']
                weights = ' / '.join(f'{v:.4f}' for v in item['weights'])
                lines.append(f"| {reward} | {weights} | {len(d['history'])} | {d['converged']} | {d['final_margin']:.6f} | {d['expert_representable']} | {d['policy_names'][d['expert_index']]} |")
            lines += ['', '| 模型 | 方向准确率 | 平衡准确率 | 非零涨跌准确率 | MSE | 单条 P50 μs | 单条 P95 μs | 批量 μs/行 |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            for m in w['model_eval']:
                latency = m['latency']
                nz = 'N/A' if m['nonzero_direction_accuracy'] is None else f"{m['nonzero_direction_accuracy']:.2f}%"
                lines.append(f"| {m['name']} | {m['direction_accuracy']:.2f}% | {m['balanced_accuracy']:.2f}% | {nz} | {m['mse']:.3e} | {latency['single_p50_us']:.2f} | {latency['single_p95_us']:.2f} | {latency['batch_us_per_row']:.3f} |")
            b = w['prediction_baselines']
            lines += ['', f"零收益预测：MSE `{b['zero_prediction_mse']:.3e}`，方向准确率 **{b['zero_direction_accuracy']:.2f}%**；训练多数类基线准确率 **{b['train_majority_accuracy']:.2f}%**。方向指标仅评估已有真实标签的 {b['evaluated_rows']} 行，尾部 {b['unlabeled_tail']} 行仍参与交易回放与平仓。", '',
                      f"90 个事件对应秒数的 10%/50%/90% 分位数：`{w['horizon_90_seconds_quantiles']}`。"]
    # 总结也由实际诊断计算；若以后增加数据或改变模型，这段结论会随结果自动更新。
    windows = [w for e in data['experiments'].values() for w in e['windows']]
    cash_experts = sum(w['rewards']['ME']['diagnostics']['policy_names'][
        w['rewards']['ME']['diagnostics']['expert_index']] == 'Baseline-Cash' for w in windows)
    unrepresentable = sum(not w['rewards']['ME']['diagnostics']['expert_representable'] for w in windows)
    lines += ['', '## 本次运行观察', '',
              f'{len(windows)} 个窗口中，{cash_experts} 个窗口的校准净盈亏最优策略是现金（不交易）；{unrepresentable} 个窗口的 ME 专家无法由当前非负线性奖励表示。求解器收敛只表示有限策略最大间隔问题已求解，不表示专家匹配成功或获得有效交易优势。', '',
              '这些诊断限制了本次奖励学习结果的解释范围。单尺度顶点权重可能来自线性规划退化或缺乏有效专家，不能解释为发现了最优预测周期。', '']
    lines += ['', '## 结论边界与剩余工作', '',
              '本实验用于验证因果实现及比较指定数据窗口中的方法表现。负净收益本身不否定复现，正毛收益也不证明存在可交易优势；判断必须同时参考固定模型、现金、轮换、多随机种子及奖励消融。', '',
              '现有数据无法验证跨日、跨周泛化或论文每周模型更新。尚未实现被动挂单排队、部分成交、网络延迟、市场冲击及保证金管理。主动成交是假设，不代表任何规模订单都保证成交。一步事件延迟不是固定墙钟延迟。', '',
              '下一阶段应获取多交易日数据，按真实日期滚动训练与评估，补充手续费/滑点敏感性与更真实撮合；只有足够多独立交易日后再报告收益置信区间和年化风险指标。', '']
    return '\n'.join(lines)


def main():
    """读取命令行指定的 JSON，校验版本后保存报告。

    --input/--output 允许给快速验证另设文件，防止覆盖正式实验报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=SUMMARY_JSON_PATH)
    parser.add_argument('--output', default=str(BASE_DIR / 'replication_report.md'))
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    if data.get('schema_version') != 2:
        raise ValueError('Regenerate experiments with schema version 2')
    Path(args.output).write_text(render_report(data))


if __name__ == '__main__':
    main()
