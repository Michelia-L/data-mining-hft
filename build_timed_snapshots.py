"""按指定 UTC 文件日期从 Databento MBP-10 制作因果定时盘口快照。"""
import argparse
import json

from src.data_catalog import select_daily_source
from src.timed_snapshots import write_timed_snapshots


def main(argv=None):
    """只处理一个索引日期；输出路径必须由调用者指定，避免覆盖正式结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-index', required=True, help='Local Databento index JSON')
    parser.add_argument('--data-date', required=True, help='UTC file date YYYY-MM-DD')
    parser.add_argument('--output', required=True, help='New Parquet path for timed snapshots')
    parser.add_argument('--interval-ms', type=int, default=500, help='UTC snapshot grid in milliseconds')
    parser.add_argument('--max-age-ms', type=int,
                        help='Oldest eligible completed book; default is one interval')
    parser.add_argument('--max-records', type=int,
                        help='Read only a raw-message prefix for validation; marks output partial')
    parser.add_argument('--include-degraded', action='store_true',
                        help='Explicitly allow a date marked degraded by Databento')
    args = parser.parse_args(argv)
    try:
        source = select_daily_source(args.data_index, args.data_date, args.include_degraded)
        stats = write_timed_snapshots(source['source_file'], args.output,
                                      interval_ms=args.interval_ms, max_age_ms=args.max_age_ms,
                                      max_records=args.max_records, source_date=source['file_date_utc'],
                                      condition=source['condition'])
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(source_date_utc=source['file_date_utc'],
                          condition=source['condition'], output=args.output,
                          interval_ms=args.interval_ms,
                          max_age_ms=args.max_age_ms or args.interval_ms,
                          partial_prefix=args.max_records is not None, **stats), ensure_ascii=False))


if __name__ == '__main__':
    main()
