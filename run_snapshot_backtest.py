"""快照固定模型时间回放入口；日期、时间单位和输出目录必须明确。

示例见 README；默认只做开发验证，不替代 run_experiments 的旧事件实验。
"""
import argparse
from pathlib import Path

from src.snapshot_backtest import run_snapshot_experiment, write_results
from src.snapshot_dataset import DEFAULT_CALENDAR


def main(argv=None):
    """以显式时间切分冻结模型；命令行参数不会根据测试净收益自动优化。"""
    parser = argparse.ArgumentParser(description='ESZ5 快照的真实时间固定模型回放（开发验证）')
    parser.add_argument('--dataset-dir', required=True, help='含 snapshots.parquet 和 quality.json 的目录')
    for name in ('train-start', 'train-end', 'test-start', 'test-end'):
        parser.add_argument('--' + name, required=True, help='带时区的时刻；所有区间为 [start,end)')
    parser.add_argument('--output-dir', required=True, help='尚不存在的新结果目录')
    parser.add_argument('--calendar', default=str(DEFAULT_CALENDAR), help='与数据准备一致的日历 JSON')
    parser.add_argument('--prediction-horizon-ms', type=int, default=5000, help='Ridge 的预测前瞻期，毫秒')
    parser.add_argument('--latency-ms', type=int, default=500, help='决策到成交的最短延迟，毫秒')
    parser.add_argument('--holding-review-ms', type=int, default=15000, help='持仓复核最短间隔，毫秒')
    parser.add_argument('--threshold', type=float, help='预先固定的相对收益门槛；不在测试段调参')
    parser.add_argument('--allow-partial', action='store_true', help='明确允许前缀数据并保留标记')
    parser.add_argument('--include-degraded', action='store_true', help='明确允许降级数据并保留标记')
    parser.add_argument('--detail', action='store_true', help='额外保存逐笔交易、决策与奖励观察审计')
    args = parser.parse_args(argv)
    if Path(args.output_dir).exists():
        raise FileExistsError('Output directory must not exist')
    result = run_snapshot_experiment(args.dataset_dir,
        **{key: getattr(args, key) for key in ('train_start', 'train_end', 'test_start', 'test_end',
            'prediction_horizon_ms', 'latency_ms', 'holding_review_ms', 'threshold',
            'allow_partial', 'include_degraded', 'detail')}, calendar_path=args.calendar)
    write_results(result, args.output_dir)
    print(f"已保存时间回放结果：{args.output_dir}；训练 {result['training_rows']} 行，回放 {result['test_rows']} 行。")
    return result


if __name__ == '__main__':
    main()
