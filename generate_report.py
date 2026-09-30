"""由 schema 3 结果生成课程报告，明确论文公式、工程变体及证据边界。

数值只从 JSON 提取。方法说明引用固定的论文对照文档，不根据收益好坏改写方法。
"""
import argparse
import json
from pathlib import Path
from src.config import BASE_DIR, SUMMARY_JSON_PATH


def choice_text(choice):
    """同时显示选中项与全部并列项，不把固定顺序消歧包装成唯一最优。"""
    ties = choice['tied_best_candidates']
    return (f"并列最佳 `{ties}`，按预先声明顺序选 `{choice['selected']}`" if len(ties) > 1
            else f"选中 `{choice['selected']}`")


def metric(value, suffix='', digits=2):
    """把可空交易统计格式化为报告文本；无交易时保持 N/A，而不是伪造 0。"""
    return 'N/A' if value is None else f"{value:.{digits}f}{suffix}"


def render_report(data):
    """从同一份结果生成方法边界、文件内描述统计、逐窗诊断与选择记录。"""
    lines = ['# FMATO 思路的课程工程实验（非论文数值复现）', '',
        '> 旧版含前视偏差的收益结论已撤回。本报告由 JSON 自动生成。', '',
        '## 论文定义与工程边界', '',
        '详见 [逐项论文对照](docs/paper_alignment.md) 与 [原论文](1679894.pdf)。', '',
        '- 默认 `paper_price_difference` 按 Eq.(2) 计算绝对价格差，不归一化、不裁剪；卖单取方向符号为项目扩展。`normalized_return` 另以训练 P99 缩放相对收益并裁剪，不混用两种单位。',
        '- Eq.(3) 明确要求权重和为 1；非负约束是项目假设，另报告允许负权的 SignedBox 敏感性。有限策略优化不等于论文未公开的生产策略优化器。',
        '- Eq.(5) 校准 OE 除以成熟订单数。无订单均值未定义，仅在有限策略优化中约定现金零向量。默认 OE 不扣成本；NetOE 在固定权重下单列敏感性。资金账本始终扣实际费用。',
        '- CausalShadowARS 是连续影子账户、300 个奖励到达事件滑窗，尚未实现 Algorithm 3 的过去固定时间窗口重新回测。CausalEventUCB 每事件选择、每成熟决策反馈一次，无成交反馈零，不是 Algorithm 2 的固定期间订单平均。',
        '- 五种算法使用同一特征及训练期；尚未实现论文不同特征集、历史时期与市场分布构成的候选库或每周更新。',
        '- ME 与执行共用门槛，只平均过门槛的校准信号；EventUCB 对未交易决策反馈零。未实现论文强调的极端 top-1/1000 信号评价协议。',
        '- 10/30/90 为不等间隔订单簿事件数，不是秒或定时快照；只有三个短尺度，未覆盖论文一小时以上时域。',
        '- 数据按 50%/15%/15%/20% 切分，前三段清除跨段标签。先选共同验证门槛，再重放校准拟合权重，随后选 c/固定模型。测试隔离，但同一短验证段反复选参存在过拟合风险，尚未做嵌套验证。',
        '- CME/ICE 的合约、市场制度、费用、更新机制不同于论文中国期货数据，未复现原论文实验数值。文件内窗口不是跨日独立重复。',
        '- 下一事件主动成交；逐事件盯市；期末平仓；每窗资金重置为 $100,000。未模拟排队、部分成交、市场冲击、保证金与端到端延迟；当前样本不足以形成独立日收益，因此不报告 Sharpe。', '',
        '## 配置与复核', '',
        f"种子 {data['metadata']['seed']}；每品种 {data['metadata']['windows']} 个窗口；每窗 {data['metadata']['rows']:,} 条原始事件；holding period {data['metadata'].get('execution', {}).get('holding_period', 'N/A')} events；Python {data['metadata']['python']}；计算线程 {data['metadata']['threads']}。", '',
        'JSON 保存数据与源码哈希、依赖实际版本、时间范围、候选分数、并列选择、奖励迭代。`environment-snapshot.txt` 是环境版本快照，不是完整依赖锁。', '']
    # 日期来自下载分区，展示时避免将它误称为完整交易日；旧 JSON 没有此字段仍可读取。
    if daily := data['metadata'].get('daily_source'):
        lines += [f"索引输入：ESZ5；UTC 文件日期 `{daily['file_date_utc']}`；"
                  f"数据质量 `{daily['condition']}`。本次仍为单文件窗口实验，尚未跨日训练。", '']
    for key, experiment in data['experiments'].items():
        lines += [f'## {key}：文件内多窗口描述统计（非独立重复实验）', '',
            f"数据 `{experiment['source_file']}`，共 {experiment['source_rows']:,} 行。每窗账户重置，合计不是连续账户收益。", '',
            '| 策略 | 总净盈亏 USD | 平均 | 最差窗口 | 最好窗口 | 平仓笔数 | 毛胜率 | 净胜率 | 平均毛盈亏/笔 | 平均摩擦/笔 | 平均持有 events |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in experiment['aggregate']:
            lines.append(
                f"| {r['strategy']} | {r['net_pnl_usd_sum']:.2f} | {r['net_pnl_usd_mean']:.2f} | "
                f"{r['net_pnl_usd_min']:.2f} | {r['net_pnl_usd_max']:.2f} | {r['total_trades']} | "
                f"{metric(r.get('gross_win_rate'), '%', 2) if r.get('gross_win_rate') is None else metric(r['gross_win_rate'] * 100, '%', 2)} | "
                f"{metric(r.get('net_win_rate'), '%', 2) if r.get('net_win_rate') is None else metric(r['net_win_rate'] * 100, '%', 2)} | "
                f"{r.get('avg_trade_gross_usd', 0.):.2f} | {r.get('avg_trade_friction_usd', 0.):.2f} | "
                f"{metric(r.get('holding_events_mean'))} |")
        for i, w in enumerate(experiment['windows'], 1):
            t, tuning = w['partitions']['test'], w['tuning']
            lines += ['', f'### 窗口 {i}', '',
                f"测试 {t['start']} → {t['end']}，{t['duration_seconds']:.2f} 秒；奖励模式 `{w['reward_definition']}`。", '',
                f"门槛：{choice_text(tuning['threshold_choice'])}。固定模型：{choice_text(tuning['fixed_choice'])}。", '',
                f"校准专家：{choice_text(w['calibration_expert_choice'])}。", '',
                f"UCB-ME：{choice_text(tuning['c_choices']['ME'])}；UCB-OE：{choice_text(tuning['c_choices']['OE'])}。", '',
                '| 奖励 / 权重约束 | 10 / 30 / 90 事件权重 | 收敛 | 专家可表示 | 最终间隔 |',
                '|---|---|---|---|---:|']
            for group, variants in [('simplex', w['rewards']), ('signed_box', w['signed_weight_sensitivity'])]:
                for reward, item in variants.items():
                    d = item['diagnostics']
                    weights = ' / '.join(f'{v:.4f}' for v in item['weights'])
                    lines.append(f"| {reward} / {group} | {weights} | {d['converged']} | {d['expert_representable']} | {d['final_margin']:.6f} |")
            lines += ['', '| 交易策略 | 平仓笔数 | mean holding | median holding | P90 holding | 毛盈亏/笔 USD | 摩擦/笔 USD | 毛胜率 | 净胜率 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for s in w['strategies']:
                gross_wr = None if s.get('gross_win_rate') is None else s['gross_win_rate'] * 100
                net_wr = None if s.get('net_win_rate') is None else s['net_win_rate'] * 100
                lines.append(
                    f"| {s['strategy']} | {s['total_trades']} | {metric(s.get('holding_events_mean'))} | "
                    f"{metric(s.get('holding_events_median'))} | {metric(s.get('holding_events_p90'))} | "
                    f"{s.get('avg_trade_gross_usd', 0.):.2f} | {s.get('avg_trade_friction_usd', 0.):.2f} | "
                    f"{metric(gross_wr, '%')} | {metric(net_wr, '%')} |")
            lines += ['', '| 模型 | 原始符号准确率 | 门槛三类准确率 | 有效信号准确率 | 信号覆盖率 | 分类 argmax 准确率 | 单条 P50 / P95 μs |',
                '|---|---:|---:|---:|---:|---:|---|']
            for m in w['model_eval']:
                active = 'N/A' if m['active_signal_accuracy'] is None else f"{m['active_signal_accuracy']:.2f}%"
                argmax = 'N/A' if 'classifier_argmax_accuracy' not in m else f"{m['classifier_argmax_accuracy']:.2f}%"
                lines.append(f"| {m['name']} | {m['raw_sign_accuracy']:.2f}% | {m['thresholded_signal_accuracy']:.2f}% | {active} | {m['signal_coverage']:.2%} | {argmax} | {m['latency']['single_p50_us']:.2f} / {m['latency']['single_p95_us']:.2f} |")
            b = w['prediction_baselines']
            lines += ['', f"零信号准确率 {b['zero_direction_accuracy']:.2f}%；训练多数类准确率 {b['train_majority_accuracy']:.2f}%。原始符号、阈值后信号、分类 argmax 是不同指标，不可混读；未标注尾部 {b['unlabeled_tail']} 行仍参与执行。", '',
                f"90 事件对应秒数分位数：`{w['horizon_90_seconds_quantiles']}`。", '',
                '校准订单数/均值是否定义：' + '；'.join(f"{r['strategy']}：{r['matured_order_count']}/{r['order_mean_defined']}" for r in w['calibration_policies']) + '。']
    lines += ['', '## 解释限制与后续工作', '',
        '求解收敛不意味着专家可表示；若多个不交易策略并列零收益，专家选择不能证明交易优势。权重位于单一尺度也不能解释为发现了稳定最优周期。', '',
        '当前只验证所述工程变体及文件内表现。要继续做论文忠实复现，需要多日数据、不同历史期/特征子集的模型库、固定时间期间的 UCB、过去固定时间窗口重新回测的 ARS、极端信号协议，以及嵌套时间验证和更真实的撮合。', '']
    return '\n'.join(lines)


def main():
    """读取 schema 3 文件；旧版结果不再解释为新公式结果，必须重新运行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=SUMMARY_JSON_PATH)
    parser.add_argument('--output', default=str(BASE_DIR / 'replication_report.md'))
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    if data.get('schema_version') != 3:
        raise ValueError('Regenerate experiments with schema version 3')
    Path(args.output).write_text(render_report(data))


if __name__ == '__main__':
    main()
