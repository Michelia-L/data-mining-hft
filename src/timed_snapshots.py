"""把单日 ESZ5 MBP-10 消息流转换为按接收时间对齐的定时盘口快照。

论文以定时间隔的盘口快照定义 tick；Databento MBP-10 则逐消息更新，且同一
交易所事件可能有多条记录。快照是课程模型与回放的输入，
输出中的 ts_event 是可决策的网格时刻，
source_ts_event/source_ts_recv 则保留所选原始消息的真实时间供审计。
"""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Databento 位标志必须按位检查；例如 160 同时含 F_LAST 和 F_SNAPSHOT。
F_LAST = 1 << 7
F_BAD_TS_RECV = 1 << 3
F_MAYBE_BAD_BOOK = 1 << 2
PRICE_FIELDS = [f'{side}_px_{level:02d}' for level in range(5) for side in ('bid', 'ask')]
SIZE_FIELDS = [f'{side}_sz_{level:02d}' for level in range(5) for side in ('bid', 'ask')]
INPUT_FIELDS = ['ts_recv', 'ts_event', 'flags', 'sequence', 'instrument_id', 'symbol',
                *PRICE_FIELDS, *SIZE_FIELDS]
SNAPSHOT_SCHEMA = pa.schema([
    pa.field('ts_event', pa.timestamp('ns', tz='UTC')),
    pa.field('source_ts_event', pa.timestamp('ns', tz='UTC')),
    pa.field('source_ts_recv', pa.timestamp('ns', tz='UTC')),
    pa.field('sequence', pa.uint32()),
    pa.field('instrument_id', pa.uint32()),
    pa.field('symbol', pa.string()),
    pa.field('source_flags', pa.uint8()),
    pa.field('age_ms', pa.float64()),
    *[pa.field(field, pa.float64()) for field in PRICE_FIELDS],
    *[pa.field(field, pa.uint32()) for field in SIZE_FIELDS],
])


def _utc_timestamp(value_ns):
    """从纳秒整数构造 UTC 时间，避免 Python datetime 丢失纳秒尾数。"""
    return pd.Timestamp(value_ns, unit='ns', tz='UTC')


def _valid_book(values, row):
    """只有完整事件末尾的前五档报价满足基本双边盘口约束时才可作为快照。"""
    prices = [float(values[name][row]) for name in PRICE_FIELDS]
    sizes = [float(values[name][row]) for name in SIZE_FIELDS]
    return (np.isfinite(prices).all() and np.isfinite(sizes).all()
            and prices[0] > 0 and prices[1] >= prices[0]
            and sizes[0] > 0 and sizes[1] > 0 and all(size >= 0 for size in sizes))


def iter_timed_snapshots(source_path, interval_ms=500, max_age_ms=None,
                         max_records=None, batch_size=65536, statistics=None,
                         grid_observer=None):
    """逐批读取 MBP-10，仅产生在网格时刻已经完整且足够新的盘口。

    以 UTC 整数纳秒网格为准，在边界 B 生成快照时只用 ts_recv < B 的记录。
    恰好在 B 才收到的记录归入下个网格，避免边界上的前视。非 F_LAST 消息表示
    当前事件尚未结束；若它跨过边界，该边界不能沿用上一笔完整盘口。
    默认最大报价年龄为一个网格周期。行情空档不生成虚构快照；后续训练必须
    按网格缺口隔离标签，不能把相邻输出行当作必然相隔一个 tick。

    statistics 是可选的调用方字典，用于边流式处理边收集行数与跳过原因。
    max_records 只做开发期前缀验证，不能代表完整交易日。
    grid_observer 是只读诊断回调，接收 (网格起点ns,排他终点ns,状态,
    完整盘口接收时间ns或None,最近可信接收时间ns或None)。范围按 interval_ms
    递增；长空档一次报告，不逐格展开。回调不能改变采样或补齐缺失盘口。
    """
    if type(interval_ms) is not int or interval_ms <= 0:
        raise ValueError('interval_ms must be a positive integer')
    if max_age_ms is None:
        max_age_ms = interval_ms
    if type(max_age_ms) is not int or max_age_ms <= 0:
        raise ValueError('max_age_ms must be a positive integer')
    if max_records is not None and (type(max_records) is not int or max_records <= 0):
        raise ValueError('max_records must be a positive integer')
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    interval_ns, max_age_ns = interval_ms * 1_000_000, max_age_ms * 1_000_000
    stats = statistics if statistics is not None else {}
    stats.update(source_records=0, snapshots=0, skipped_intervals=0,
                 invalid_completed_books=0, unreliable_receive_records=0,
                 skipped_by_reason={})
    parquet = pq.ParquetFile(source_path)
    missing = set(INPUT_FIELDS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f'MBP-10 snapshot input is missing columns: {sorted(missing)}')
    next_boundary = last_recv = last_complete = None
    event_open = False
    # 原有采样状态保持不变；原因只解释为什么当前可见状态不能产出网格。
    blocked_reason = 'no_completed_book'
    for batch in parquet.iter_batches(batch_size=batch_size, columns=INPUT_FIELDS):
        # ts_recv 在 Databento 的 pandas 元数据里也可能是索引；按 Arrow 列读取
        # 可避免 pandas 隐式重建索引，并完整保留纳秒精度。
        values = {}
        for i, field in enumerate(INPUT_FIELDS):
            column = batch.column(i)
            if column.null_count:
                raise ValueError(f'Input contains null {field}; cannot determine a causal book')
            values[field] = (column.to_pylist() if field == 'symbol'
                             # 历史文件为 ns，测试或其他导出器可能写成 us；先
                             # 显式转 ns 再取整数，避免把 500 ms 错当成 500 秒。
                             else column.cast(pa.timestamp('ns', tz='UTC')).cast(pa.int64()).to_numpy(
                                 zero_copy_only=False)
                             if field in ('ts_recv', 'ts_event')
                             else column.to_numpy(zero_copy_only=False))
        for row in range(batch.num_rows):
            if max_records is not None and stats['source_records'] >= max_records:
                return
            stats['source_records'] += 1
            if values['symbol'][row] != 'ESZ5':
                raise ValueError('Timed snapshot input must contain only ESZ5')
            flags = int(values['flags'][row])
            if flags & F_BAD_TS_RECV:
                # 此行的接收时间不可信，连“网格边界已经过去”都不能由它证明。
                # 因而在比较时间、推进边界或更新 last_recv 之前隔离该行；
                # 已知盘口随即失效，下一个可信的完整事件到来后才重新启用。
                last_complete = None
                event_open = True
                blocked_reason = 'unreliable_receive_state'
                stats['unreliable_receive_records'] += 1
                if flags & F_LAST:
                    stats['invalid_completed_books'] += 1
                continue
            recv_ns = int(values['ts_recv'][row])
            if last_recv is not None and recv_ns < last_recv:
                raise ValueError('ts_recv moved backwards; snapshots require receive-ordered input')
            if next_boundary is None:
                next_boundary = (recv_ns // interval_ns + 1) * interval_ns
            # 当前消息尚未到达时，先结算所有到期网格。无新报价的长空档用
            # 整数跳转，不为休市逐个构造百万个虚假 tick。
            while next_boundary <= recv_ns:
                fresh = (last_complete is not None and not event_open
                         and next_boundary - last_complete['recv_ns'] <= max_age_ns)
                if fresh:
                    chosen = last_complete
                    if grid_observer is not None:
                        grid_observer(next_boundary, next_boundary + interval_ns, 'emitted',
                                      chosen['recv_ns'], last_recv)
                    snapshot = dict(ts_event=_utc_timestamp(next_boundary),
                                    source_ts_event=_utc_timestamp(chosen['event_ns']),
                                    source_ts_recv=_utc_timestamp(chosen['recv_ns']),
                                    sequence=chosen['sequence'], instrument_id=chosen['instrument_id'],
                                    symbol=chosen['symbol'], source_flags=chosen['flags'],
                                    age_ms=(next_boundary - chosen['recv_ns']) / 1_000_000,
                                    **chosen['book'])
                    stats['snapshots'] += 1
                    yield snapshot
                    next_boundary += interval_ns
                else:
                    remaining = (recv_ns - next_boundary) // interval_ns + 1
                    reason = (blocked_reason if last_complete is None or event_open
                              else 'stale_completed_book')
                    counts = stats['skipped_by_reason']
                    counts[reason] = counts.get(reason, 0) + int(remaining)
                    if grid_observer is not None:
                        grid_observer(next_boundary, next_boundary + remaining * interval_ns, reason,
                                      None if last_complete is None else last_complete['recv_ns'], last_recv)
                    stats['skipped_intervals'] += remaining
                    next_boundary += remaining * interval_ns
            last_recv = recv_ns
            if not flags & F_LAST:
                event_open = True
                # 尚未恢复可信完整盘口时保留先前异常原因；正常盘口被新事件
                # 打断则标记 incomplete_event，不把未完成事件当成超龄。
                if last_complete is not None or blocked_reason == 'no_completed_book':
                    blocked_reason = 'incomplete_event'
                continue
            event_open = False
            event_ns = int(values['ts_event'][row])
            if flags & F_MAYBE_BAD_BOOK or not _valid_book(values, row):
                # 盘口可能损坏时不沿用旧报价；交易所与接收端时钟未必同步，
                # 因此不比较 event_ns 和 recv_ns 的大小来筛选合法消息。
                # 事件时间仅作为来源信息保存，快照因果性始终由接收时间决定。
                last_complete = None
                blocked_reason = ('bad_book_flag' if flags & F_MAYBE_BAD_BOOK else 'invalid_completed_book')
                stats['invalid_completed_books'] += 1
                continue
            blocked_reason = None
            last_complete = dict(recv_ns=recv_ns, event_ns=event_ns,
                                 sequence=int(values['sequence'][row]),
                                 instrument_id=int(values['instrument_id'][row]),
                                 symbol=values['symbol'][row], flags=flags,
                                 book={field: float(values[field][row]) for field in PRICE_FIELDS}
                                      | {field: int(values[field][row]) for field in SIZE_FIELDS})


def _sha256(path):
    """逐块计算输入文件哈希，让快照产物可追溯到具体原始文件。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_timed_snapshots(source_path, output_path, *, interval_ms=500,
                          max_age_ms=None, max_records=None, source_date=None,
                          condition=None, batch_size=65536, grid_observer=None):
    """流式写出 Parquet；成功后原子替换临时文件，错误时不留下半成品。

    产物元数据保留采样时钟、源哈希、日期、质量和是否只读了前缀。传入的
    source_date 是下载分区日期，不能代替交易所 session 的定义。
    """
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite existing snapshots: {output_path}')
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {output_path.parent}')
    if max_age_ms is None:
        max_age_ms = interval_ms
    metadata = {
        b'sampling_clock': b'ts_recv', b'interval_ms': str(interval_ms).encode(),
        b'max_age_ms': str(max_age_ms).encode(), b'source_file': source_path.name.encode(),
        b'source_sha256': _sha256(source_path).encode(),
        b'partial_prefix': str(max_records is not None).lower().encode(),
        b'source_date_utc': str(source_date or '').encode(),
        b'source_condition': str(condition or '').encode(),
    }
    schema = SNAPSHOT_SCHEMA.with_metadata(metadata)
    stats = {}
    # 同目录临时文件保证最终 rename 原子性；大日数据也只缓冲少量快照。
    with tempfile.TemporaryDirectory(prefix='.snapshots-', dir=output_path.parent) as directory:
        temporary = Path(directory) / 'part.parquet'
        with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
            rows = []
            for snapshot in iter_timed_snapshots(source_path, interval_ms, max_age_ms,
                                                 max_records, batch_size, stats, grid_observer):
                rows.append(snapshot)
                if len(rows) == 2048:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        if stats['snapshots'] == 0:
            raise ValueError('No valid timed snapshots in the selected input')
        os.replace(temporary, output_path)
    return stats
