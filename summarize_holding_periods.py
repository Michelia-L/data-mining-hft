"""汇总 holding period 单因素敏感性实验。

输入必须来自同一源码、数据、种子、窗口和奖励定义；脚本只比较 holding_period，
不会按测试结果反向选择参数。正式的逐窗完整结果仍保存在各自 JSON 中。
"""
import argparse
import json
from pathlib import Path

from src.config import BASE_DIR


DEFAULT_PERIODS = (30, 90, 180, 300)
DEFAULT_INPUTS = [BASE_DIR / 'results' / f'holding_period_{p}.json' for p in DEFAULT_PERIODS]


def fmt_pct(value):
    return 'N/A' if value is None else f'{value * 100:.2f}%'


def fmt_num(value, digits=2):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def load_runs(paths):
    runs = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        period = int(data['metadata']['execution']['holding_period'])
        runs.append((period, data))
    runs.sort(key=lambda item: item[0])
    return runs


def check_comparable(runs):
    """确认除 holding period/运行时间外的关键实验输入保持一致。"""
    first_period, first = runs[0]
    keys = ('reward_definition', 'seed', 'rows', 'windows', 'split_ratios',
            'instruments', 'ucb_candidates', 'quantity', 'initial_capital')
    for period, data in runs[1:]:
        for key in keys:
            if data['metadata'].get(key) != first['metadata'].get(key):
                raise ValueError(f'holding {period}: metadata {key} differs from holding {first_period}')
        if data.get('horizon_events') != first.get('horizon_events'):
            raise ValueError(f'holding {period}: horizon_events differs')
        if data['metadata'].get('source_sha256') != first['metadata'].get('source_sha256'):
            raise ValueError(f'holding {period}: source code hashes differ')
        for instrument in first['experiments']:
            if data['experiments'][instrument]['source_sha256'] != first['experiments'][instrument]['source_sha256']:
                raise ValueError(f'holding {period}: dataset hash differs for {instrument}')


def render(runs):
    check_comparable(runs)
    periods = [p for p, _ in runs]
    lines = [
        '# Holding period 单因素敏感性',
        '',
        f"比较 holding period：{', '.join(map(str, periods))} events。除 holding period 外，源码、数据哈希、种子、窗口、奖励定义、阈值候选、UCB 候选和交易成本配置保持一致。",
        '',
        '这里只做描述性比较，不根据测试结果把某个 holding period 重新选作参数。',
        ''
    ]
    instruments = list(runs[0][1]['experiments'])
    for instrument in instruments:
        lines += [f'## {instrument}', '',
                  '### Variant 汇总', '',
                  '| holding | 策略 | 净盈亏 USD | 毛盈亏 USD | 摩擦 USD | 交易数 | 毛胜率 | 净胜率 | 毛盈亏/笔 | 摩擦/笔 | mean holding |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for period, data in runs:
            aggregate = data['experiments'][instrument]['aggregate']
            for row in aggregate:
                if not row['strategy'].startswith('Variant-'):
                    continue
                lines.append(
                    f"| {period} | {row['strategy']} | {row['net_pnl_usd_sum']:.2f} | "
                    f"{row['gross_pnl_usd_sum']:.2f} | {row['friction_usd_sum']:.2f} | "
                    f"{row['total_trades']} | {fmt_pct(row['gross_win_rate'])} | "
                    f"{fmt_pct(row['net_win_rate'])} | {row['avg_trade_gross_usd']:.2f} | "
                    f"{row['avg_trade_friction_usd']:.2f} | {fmt_num(row['holding_events_mean'])} |")
        lines += ['', '### Variant 逐窗持仓分布', '',
                  '| holding | 窗口 | 策略 | trades | mean | median | P90 | 毛胜率 | 净胜率 |',
                  '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
        for period, data in runs:
            for window_index, window in enumerate(data['experiments'][instrument]['windows'], 1):
                for row in window['strategies']:
                    if not row['strategy'].startswith('Variant-'):
                        continue
                    lines.append(
                        f"| {period} | {window_index} | {row['strategy']} | {row['total_trades']} | "
                        f"{fmt_num(row['holding_events_mean'])} | {fmt_num(row['holding_events_median'])} | "
                        f"{fmt_num(row['holding_events_p90'])} | {fmt_pct(row['gross_win_rate'])} | "
                        f"{fmt_pct(row['net_win_rate'])} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', default=[str(p) for p in DEFAULT_INPUTS])
    parser.add_argument('--output', default=str(BASE_DIR / 'holding_period_sensitivity.md'))
    args = parser.parse_args()
    runs = load_runs(args.inputs)
    Path(args.output).write_text(render(runs))


if __name__ == '__main__':
    main()
