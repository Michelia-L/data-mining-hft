"""将 ESZ5 定时快照整理为按交易 session 隔离的时间标签数据集。"""
import argparse
import json

from src.snapshot_dataset import DEFAULT_CALENDAR, DEFAULT_HORIZONS_MS, prepare_snapshot_dataset


def main(argv=None):
    """创建新目录，显式允许预览/降级数据；不覆盖快照或正式实验结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', nargs='+', required=True, help='Snapshot Parquet files in time order')
    parser.add_argument('--output-dir', required=True, help='New directory for dataset and quality reports')
    parser.add_argument('--calendar', default=str(DEFAULT_CALENDAR), help='Versioned ESZ5 session calendar JSON')
    parser.add_argument('--horizons-ms', nargs='+', type=int, default=DEFAULT_HORIZONS_MS,
                        help='Increasing positive forecast milliseconds, each a grid multiple')
    parser.add_argument('--split-boundary', action='append', default=[],
                        help='Timezone-aware partition boundary; labels reaching it become invalid')
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly allow prefix snapshot previews')
    parser.add_argument('--include-degraded', action='store_true', help='Explicitly allow degraded source data')
    args = parser.parse_args(argv)
    try:
        result = prepare_snapshot_dataset(args.snapshots, args.output_dir, calendar_path=args.calendar,
            horizons_ms=args.horizons_ms, boundaries=args.split_boundary,
            allow_partial=args.allow_partial, include_degraded=args.include_degraded)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output_dir=args.output_dir, rows=result['output_rows'],
                          sessions=len(result['sessions']), partial_input=result['partial_input'],
                          degraded_input=result['degraded_input']), ensure_ascii=False))


if __name__ == '__main__':
    main()
