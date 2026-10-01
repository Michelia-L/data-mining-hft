"""将定时盘口快照整理为有 session、缺口和时间标签的数据集。

论文第 3 页 §3.1 的 tick 是定时快照，第 4 页 Eq.(2) 是未来中间价差。
这里按明确毫秒前瞻期生成离线标签，绝不以“后面第 h 行”代替目标时刻。
缺一格就断开连续区间，是本项目的保守质量规则；标签和滚动特征均不跨缺口。
本模块只准备数据，不训练模型，也不修改事件级回测引擎。
"""
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path
import platform
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import BASE_DIR
from src.timed_snapshots import PRICE_FIELDS, SIZE_FIELDS, SNAPSHOT_SCHEMA


DEFAULT_CALENDAR = BASE_DIR / 'config' / 'cme_es_sessions_2025.json'
DEFAULT_HORIZONS_MS = (5000, 15000, 45000)


def sha256_file(path):
    """分块记录实际输入内容的哈希，而非依靠可变的文件名判断数据版本。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc_ns(values):
    """显式转纳秒后取整数；兼容 pandas 不同版本默认的 us/ns 时间精度。"""
    return pd.DatetimeIndex(values).as_unit('ns').asi8


def parse_boundary(value):
    """切分时刻必须自带时区，不能把运行者本地时间偷偷解释成 UTC。"""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Split boundaries must be explicit timezone-aware timestamps')
    return stamp.tz_convert('UTC')


class SessionCalendar:
    """读取有限日期范围的交易日规则，并展开为可匹配的 UTC 连续交易时段。

    session_id 是配置中的 trade date，可包含多个交易时段；segment_id 区分
    同一 trade date 内的暂停。America/Chicago 使用 IANA 时区处理夏令时，
    不固定减六小时。假期由版本化配置覆盖，不能把 NYSE 日历套到 ES 上。
    """

    def __init__(self, path=DEFAULT_CALENDAR):
        self.path = Path(path).resolve()
        self.spec = json.loads(self.path.read_text(encoding='utf-8'))
        self.sha256 = sha256_file(self.path)
        spec = self.spec
        if spec['symbol'] != 'ESZ5' or spec['timezone'] != 'America/Chicago':
            raise ValueError('Calendar must describe ESZ5 in America/Chicago')
        self.first, self.last = date.fromisoformat(spec['valid_from']), date.fromisoformat(spec['valid_through'])
        if self.first > self.last or not spec['calendar_id'] or not spec['sources']:
            raise ValueError('Calendar needs valid coverage, identity and sources')
        excluded = set(spec['closed_session_dates'])
        overrides = spec['overrides']
        for key in excluded | set(overrides):
            day = date.fromisoformat(key)
            if not self.first <= day <= self.last or day.weekday() >= 5:
                raise ValueError('Calendar exceptions must be covered weekday session dates')
        if excluded & set(overrides):
            raise ValueError('A session cannot be both closed and overridden')
        self.sessions, segments = {}, []
        zone = ZoneInfo(spec['timezone'])

        def local(day, offset, clock):
            """日期偏移基于本地日历日；负偏移明确标识夜盘属于哪个 trade date。"""
            if type(offset) is not int or not -7 <= offset <= 0:
                raise ValueError('Calendar day offsets must be integers from -7 to 0')
            parsed = time.fromisoformat(clock)
            if parsed.tzinfo is not None:
                raise ValueError('Calendar clocks must be local wall times')
            return pd.Timestamp(datetime.combine(day + timedelta(days=offset), parsed, zone)).tz_convert('UTC')

        for day in (self.first + timedelta(days=i) for i in range((self.last - self.first).days + 1)):
            key = day.isoformat()
            if day.weekday() >= 5 or key in excluded:
                continue
            rule = overrides.get(key, spec['regular'])
            opening, closing = local(day, rule['open_day_offset'], rule['open']), local(day, 0, rule['close'])
            breaks = [(local(day, b['day_offset'], b['start']), local(day, b['day_offset'], b['end']))
                      for b in rule['breaks']]
            cursor = opening
            intervals = []
            for start, end in breaks:
                if not cursor < start < end < closing:
                    raise ValueError('Calendar breaks must be ordered within the session')
                intervals.append((cursor, start))
                cursor = end
            if cursor >= closing:
                raise ValueError('Calendar opening must precede closing')
            intervals.append((cursor, closing))
            self.sessions[key] = dict(open=opening, close=closing, intervals=intervals)
            for number, (start, end) in enumerate(intervals):
                if segments and start < segments[-1]['end']:
                    raise ValueError('Calendar trading intervals overlap')
                segments.append(dict(session_id=key, segment_id=f'{key}:{number}', start=start, end=end))
        if not segments:
            raise ValueError('Calendar contains no trading sessions')
        self.segments = pd.DataFrame(segments)
        self.starts, self.ends = utc_ns(self.segments.start), utc_ns(self.segments.end)

    def assign(self, frame):
        """按快照决策时刻匹配 (open, close]，并要求来源报价来自同一开放时段。

        收盘边界只包含 close 之前收到的信息；开盘边界不能沿用休市前报价。
        未匹配行返回空 session，供报告统计后排除。覆盖范围之外则直接报错。
        """
        times, received = utc_ns(frame.ts_event), utc_ns(frame.source_ts_recv)
        if (times < self.starts[0]).any() or (times > self.ends[-1]).any():
            raise ValueError('Snapshot timestamps fall outside calendar coverage')
        indices = np.searchsorted(self.starts, times, side='left') - 1
        safe = np.maximum(indices, 0)
        opened = (indices >= 0) & (times <= self.ends[safe])
        available = opened & (received >= self.starts[safe])
        result = frame.copy()
        result['session_id'] = np.where(available, self.segments.session_id.to_numpy()[safe], '')
        result['segment_id'] = np.where(available, self.segments.segment_id.to_numpy()[safe], '')
        return result, int((~opened).sum()), int((opened & ~available).sum())


def inspect_snapshot(path, allow_partial=False, include_degraded=False):
    """验证输入的来源元数据，拒绝原始 MBP、未知质量或未经明确允许的预览文件。"""
    path = Path(path).resolve(strict=True)
    parquet = pq.ParquetFile(path)
    metadata = {key.decode(): value.decode() for key, value in (parquet.schema_arrow.metadata or {}).items()}
    required = {'sampling_clock', 'interval_ms', 'max_age_ms', 'source_file', 'source_sha256',
                'source_date_utc', 'source_condition', 'partial_prefix'}
    if not required <= metadata.keys() or metadata['sampling_clock'] != 'ts_recv':
        raise ValueError('Input must carry timed snapshot provenance metadata')
    if metadata.get('dataset_kind'):
        raise ValueError('Prepared datasets cannot be reused as raw snapshots')
    if metadata['partial_prefix'] not in ('true', 'false'):
        raise ValueError('Invalid partial_prefix metadata')
    if metadata['partial_prefix'] == 'true' and not allow_partial:
        raise ValueError('Partial snapshot input requires --allow-partial')
    condition = metadata['source_condition']
    if condition not in ('available', 'degraded'):
        raise ValueError('Unknown snapshot source condition')
    if condition == 'degraded' and not include_degraded:
        raise ValueError('Degraded input requires --include-degraded')
    interval, age = int(metadata['interval_ms']), int(metadata['max_age_ms'])
    if interval <= 0 or age <= 0:
        raise ValueError('Snapshot interval and maximum age must be positive')
    source_date = metadata['source_date_utc']
    if date.fromisoformat(source_date).isoformat() != source_date:
        raise ValueError('Invalid UTC source file date')
    if (len(metadata['source_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in metadata['source_sha256'])
            or not metadata['source_file']):
        raise ValueError('Invalid snapshot source identity')
    if not set(SNAPSHOT_SCHEMA.names) <= set(parquet.schema_arrow.names):
        raise ValueError('Snapshot input is missing required columns')
    for name in ('ts_event', 'source_ts_event', 'source_ts_recv'):
        dtype = parquet.schema_arrow.field(name).type
        if not pa.types.is_timestamp(dtype) or dtype.tz != 'UTC':
            raise ValueError('Snapshot clocks must be UTC Arrow timestamps')
    # 允许时间精度为 us/ns，但其他字段沿用快照工具的 schema；否则浮点 flags
    # 转整数会悄悄改变可靠性位，字符串价量也可能掩盖错误的导出格式。
    for field in SNAPSHOT_SCHEMA:
        if not pa.types.is_timestamp(field.type) and parquet.schema_arrow.field(field.name).type != field.type:
            raise ValueError(f'Snapshot column type disagrees with schema: {field.name}')
    return dict(path=str(path), sha256=sha256_file(path), rows=parquet.metadata.num_rows,
                metadata=metadata, interval_ms=interval, max_age_ms=age)


def validate_batch(frame, interval_ms, max_age_ms):
    """复核因果采样约束；错误输入必须报错，不能清洗后把错误隐去。"""
    if frame.isna().any().any():
        raise ValueError('Snapshot input contains null fields')
    for name in ('ts_event', 'source_ts_event', 'source_ts_recv'):
        frame[name] = frame[name].dt.as_unit('ns')
    times, received = utc_ns(frame.ts_event), utc_ns(frame.source_ts_recv)
    if (times % (interval_ms * 1_000_000)).any():
        raise ValueError('Snapshot timestamps are off the declared grid')
    ages = (times - received) / 1_000_000
    if ((ages <= 0).any() or (ages > max_age_ms).any()
            or not np.allclose(ages, frame.age_ms, rtol=0, atol=1e-6)):
        raise ValueError('Snapshot age or receive-time causality is invalid')
    flags = frame.source_flags.to_numpy(dtype=np.int64)
    if ((flags & 128) == 0).any() or (flags & (8 | 4)).any():
        raise ValueError('Snapshot source must be a reliable completed book')
    numbers = frame[PRICE_FIELDS + SIZE_FIELDS].to_numpy(dtype=float)
    if (not np.isfinite(numbers).all() or (frame.bid_px_00 <= 0).any()
            or (frame.ask_px_00 < frame.bid_px_00).any()
            or (frame[SIZE_FIELDS] < 0).any().any()
            or (frame.bid_sz_00 <= 0).any() or (frame.ask_sz_00 <= 0).any()):
        raise ValueError('Invalid snapshot book')
    if not frame.symbol.eq('ESZ5').all():
        raise ValueError('Snapshot dataset currently requires ESZ5')


def build_session_dataset(frame, calendar, interval_ms, horizons_ms, boundaries=()):
    """对一个 session 构建连续区间、因果特征与精确目标时刻的离线标签。

    特征共 11 项，与现有微观结构特征对应；动量与波动率按连续区间重启，
    预热期保留 NaN 并设置 feature_valid=False，不拿零值伪装已观测的历史。
    标签只在目标时刻存在且整个前瞻区间连续、没有跨切分时有效。
    """
    frame = frame.reset_index(drop=True).copy()
    times = utc_ns(frame.ts_event)
    step_ns = interval_ms * 1_000_000
    new_block = np.r_[True, (np.diff(times) != step_ns)
                      | (frame.segment_id.to_numpy()[1:] != frame.segment_id.to_numpy()[:-1])]
    frame['continuous_block'] = np.cumsum(new_block)
    frame['mid_price'] = (frame.ask_px_00 + frame.bid_px_00) / 2
    frame['spread'] = frame.ask_px_00 - frame.bid_px_00
    frame['rel_spread'] = frame.spread / frame.mid_price
    size_sum = frame.bid_sz_00.astype(float) + frame.ask_sz_00.astype(float)
    frame['obi_l1'] = (frame.bid_sz_00.astype(float) - frame.ask_sz_00) / size_sum
    weights = (1., .5, .33, .25, .2)
    bid = sum(frame[f'bid_sz_{i:02d}'].astype(float) * w for i, w in enumerate(weights))
    ask = sum(frame[f'ask_sz_{i:02d}'].astype(float) * w for i, w in enumerate(weights))
    frame['obi_multi'] = (bid - ask) / (bid + ask)
    micro = (frame.ask_px_00 * frame.bid_sz_00 + frame.bid_px_00 * frame.ask_sz_00) / size_sum
    frame['micro_dev'] = (micro - frame.mid_price) / (frame.spread + 1e-8)
    features = ['spread', 'rel_spread', 'obi_l1', 'obi_multi', 'micro_dev']
    grouped = frame.groupby('continuous_block', sort=False).mid_price
    for lag in (1, 5, 10, 30):
        name = f'ret_lag_{lag}'
        frame[name] = frame.mid_price / grouped.shift(lag) - 1
        features.append(name)
    returns = frame.mid_price / grouped.shift(1) - 1
    for window in (10, 30):
        name = f'vol_{window}'
        frame[name] = returns.groupby(frame.continuous_block, sort=False).transform(
            lambda part: part.rolling(window, min_periods=window).std())
        features.append(name)
    frame['feature_valid'] = np.isfinite(frame[features].to_numpy()).all(axis=1)
    session = calendar.sessions[frame.session_id.iloc[0]]
    closing_ns = session['close'].value
    end_by_segment = dict(zip(calendar.segments.segment_id, calendar.ends))
    segment_ends = frame.segment_id.map(end_by_segment).to_numpy(dtype=np.int64)
    block = frame.continuous_block.to_numpy()
    split_ns = np.array([parse_boundary(b).value for b in boundaries], dtype=np.int64)
    if len(split_ns) and (np.diff(split_ns) <= 0).any():
        raise ValueError('Split boundaries must be strictly increasing')
    mid = frame.mid_price.to_numpy()
    ready = frame.feature_valid.to_numpy().copy()
    quality = {}
    for horizon in horizons_ms:
        target = times + horizon * 1_000_000
        positions = np.searchsorted(times, target)
        safe = np.minimum(positions, len(times) - 1)
        exact = (positions < len(times)) & (times[safe] == target)
        # 按明确优先级记录一个互斥原因，让报告中失效数量可与总行数核对。
        reason = np.full(len(frame), 'valid', dtype=object)
        reason[~exact] = 'missing_target'
        reason[exact & (block[safe] != block)] = 'gap'
        crosses_split = (np.searchsorted(split_ns, times, side='right')
                         != np.searchsorted(split_ns, target, side='right'))
        reason[crosses_split] = 'split_boundary'
        reason[target > segment_ends] = 'scheduled_break'
        reason[target > closing_ns] = 'session_boundary'
        valid = reason == 'valid'
        suffix = f'{horizon}ms'
        frame[f'label_end_{suffix}'] = pd.to_datetime(target, utc=True)
        frame[f'label_valid_{suffix}'] = valid
        frame[f'label_reason_{suffix}'] = reason
        # Eq.(2) 的中间价差为主标签；相对收益另存，不能静默替换奖励单位。
        frame[f'future_price_diff_{suffix}'] = np.where(valid, mid[safe] - mid, np.nan)
        frame[f'future_return_{suffix}'] = np.where(valid, (mid[safe] - mid) / mid, np.nan)
        ready &= valid
        counts = pd.Series(reason).value_counts().to_dict()
        quality[str(horizon)] = dict(valid=int(valid.sum()),
                                    feature_and_label_valid=int((valid & frame.feature_valid).sum()),
                                    reasons={k: int(v) for k, v in counts.items()})
    frame['learning_ready'] = ready
    expected = sum(end.value // step_ns - start.value // step_ns for start, end in session['intervals'])
    gaps = np.diff(times)
    same_segment = frame.segment_id.to_numpy()[1:] == frame.segment_id.to_numpy()[:-1]
    internal_missing = int(((gaps[same_segment] // step_ns) - 1).sum())
    report = dict(session_id=frame.session_id.iloc[0], open_utc=str(session['open']),
                  close_utc=str(session['close']), observed_snapshots=len(frame),
                  expected_grid_points=int(expected), missing_grid_points=int(expected - len(frame)),
                  coverage_complete=len(frame) == expected, internal_missing_grid_points=internal_missing,
                  continuous_blocks=int(new_block.sum()), first_snapshot=str(frame.ts_event.iloc[0]),
                  last_snapshot=str(frame.ts_event.iloc[-1]), feature_valid=int(frame.feature_valid.sum()),
                  learning_ready=int(ready.sum()), labels=quality)
    return frame, features, report


def render_quality_report(report):
    """从同一份 JSON 生成中文概览，缺口不等于整个数据源都缺失行情。"""
    lines = ['# 定时快照数据质量报告', '',
             f"日历：`{report['calendar']['id']}`；间隔：{report['interval_ms']} 毫秒。", '',
             '缺口按当前提供的文件统计；缺少相邻 UTC 分区、前缀预览及采样跳过均可能造成覆盖不足。',
             '数据准备检查不代表论文复现完成，也不是可用于旧事件级引擎的新回测结果。', '',
             '| session | 有效快照 | 计划网格 | 缺失网格 | 特征可用 | 全尺度可学习 | 覆盖完整 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in report['sessions']:
        lines.append(f"| {row['session_id']} | {row['observed_snapshots']} | {row['expected_grid_points']} | "
                     f"{row['missing_grid_points']} | {row['feature_valid']} | {row['learning_ready']} | "
                     f"{'是' if row['coverage_complete'] else '否'} |")
    lines += ['', '## 标签失效原因', '',
              '`missing_target`：目标时刻未提供；`gap`：端点存在但中间缺格；',
              '`split_boundary`：达到或跨过切分；`scheduled_break`：跨交易暂停；',
              '`session_boundary`：超过 session 收盘。原因按明确优先级互斥计数。', '',
              '| session | 前瞻毫秒 | 标签可用 | 特征及标签可用 | 原因计数 |',
              '| --- | ---: | ---: | ---: | --- |']
    for row in report['sessions']:
        for horizon, values in row['labels'].items():
            counts = ', '.join(f'{key}={count}' for key, count in values['reasons'].items())
            lines.append(f"| {row['session_id']} | {horizon} | {values['valid']} | "
                         f"{values['feature_and_label_valid']} | {counts} |")
    lines += ['', f"排除非交易时段快照：{report['excluded_outside_trading']}；"
              f"排除来源报价不属于当前交易时段：{report['excluded_preopen_source']}。", '',
              '输入哈希、质量、前缀状态、切分与源码版本见同目录 `quality.json`。']
    return '\n'.join(lines) + '\n'


def prepare_snapshot_dataset(paths, output_dir, *, calendar_path=DEFAULT_CALENDAR,
                             horizons_ms=DEFAULT_HORIZONS_MS, boundaries=(),
                             allow_partial=False, include_degraded=False, batch_size=65536):
    """跨 UTC 文件流式读取、按一个 session 缓冲并输出，失败不发布半份产物。

    输入必须按时间排列，重复或倒序直接报错；不为去重而事后重排整个数据集。
    新目录包含 snapshots.parquet、quality.json 和 quality.md；正式结果不被覆盖。
    """
    if not paths or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Need input snapshots and positive batch size')
    inputs = [inspect_snapshot(p, allow_partial, include_degraded) for p in paths]
    if len({i['path'] for i in inputs}) != len(inputs):
        raise ValueError('Duplicate snapshot input path')
    interval = inputs[0]['interval_ms']
    if any((i['interval_ms'], i['max_age_ms']) != (interval, inputs[0]['max_age_ms']) for i in inputs):
        raise ValueError('All snapshot inputs must share sampling settings')
    horizons_ms = tuple(horizons_ms)
    if (not horizons_ms or any(type(h) is not int or h <= 0 or h > 7 * 86400000
                              or h % interval for h in horizons_ms)
            or tuple(sorted(set(horizons_ms))) != horizons_ms):
        raise ValueError('Horizons must be increasing unique positive grid multiples within seven days')
    boundaries = tuple(parse_boundary(b) for b in boundaries)
    if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError('Split boundaries must be strictly increasing')
    calendar = SessionCalendar(calendar_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset directory: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, dataset_kind='session_time_labeled_snapshots', symbol='ESZ5',
                  interval_ms=interval, horizons_ms=list(horizons_ms),
                  label_policy='exact_target_same_continuous_block',
                  snapshot_boundary='(open, close]', split_boundaries=[str(b) for b in boundaries],
                  feature_lag_ms=[interval * lag for lag in (1, 5, 10, 30)],
                  feature_volatility_window_ms=[interval * window for window in (10, 30)],
                  calendar=dict(id=calendar.spec['calendar_id'], sha256=calendar.sha256,
                                spec=calendar.spec), inputs=inputs,
                  partial_input=any(i['metadata']['partial_prefix'] == 'true' for i in inputs),
                  degraded_input=any(i['metadata']['source_condition'] == 'degraded' for i in inputs),
                  environment=dict(python=platform.python_version(), pandas=pd.__version__,
                                   numpy=np.__version__, pyarrow=pa.__version__),
                  code_sha256={name: sha256_file(BASE_DIR / name) for name in
                               ('src/snapshot_dataset.py', 'src/timed_snapshots.py', 'prepare_snapshot_dataset.py')},
                  excluded_outside_trading=0, excluded_preopen_source=0, sessions=[])
    previous = first_seen = instrument = None
    current, chunks, writer = None, [], None
    with tempfile.TemporaryDirectory(prefix='.snapshot-dataset-', dir=output.parent) as directory:
        staging = Path(directory) / 'dataset'
        staging.mkdir()

        def flush():
            """只在 session 结束时计算，跨文件/批次的同一 session 不重复预热。"""
            nonlocal writer, chunks
            if not chunks:
                return
            frame, features, quality = build_session_dataset(pd.concat(chunks, ignore_index=True),
                                                            calendar, interval, horizons_ms, boundaries)
            report['feature_columns'] = features
            table = pa.Table.from_pandas(frame, preserve_index=False)
            # 移除 pandas 索引元数据；算法配置与来源在质量 JSON 中有完整可追溯记录。
            metadata = {b'dataset_kind': b'session_time_labeled_snapshots',
                        b'calendar_sha256': calendar.sha256.encode(), b'interval_ms': str(interval).encode(),
                        b'horizons_ms': json.dumps(horizons_ms).encode(),
                        b'partial_input': str(report['partial_input']).lower().encode()}
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(staging / 'snapshots.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            report['sessions'].append(quality)
            chunks = []

        try:
            for input_id, item in enumerate(inputs):
                for batch in pq.ParquetFile(item['path']).iter_batches(batch_size=batch_size,
                                                                      columns=SNAPSHOT_SCHEMA.names):
                    frame = batch.to_pandas()
                    if frame.empty:
                        continue
                    validate_batch(frame, interval, item['max_age_ms'])
                    times = utc_ns(frame.ts_event)
                    if (np.diff(times) <= 0).any() or (previous is not None and times[0] <= previous):
                        raise ValueError('Duplicate or non-increasing snapshot timestamps across inputs')
                    first_seen = times[0] if first_seen is None else first_seen
                    previous = times[-1]
                    ids = frame.instrument_id.unique()
                    if len(ids) != 1 or (instrument is not None and ids[0] != instrument):
                        raise ValueError('Snapshot inputs must share a single instrument_id')
                    instrument = ids[0]
                    frame, outside, preopen = calendar.assign(frame)
                    report['excluded_outside_trading'] += outside
                    report['excluded_preopen_source'] += preopen
                    frame['snapshot_input_id'] = input_id
                    frame = frame.loc[frame.session_id.ne('')]
                    for session_id, part in frame.groupby('session_id', sort=False):
                        if current != session_id:
                            flush()
                            current = session_id
                        chunks.append(part)
            flush()
        finally:
            if writer is not None:
                writer.close()
        if not report['sessions']:
            raise ValueError('No eligible trading-session snapshots')
        # 输入时间范围内整段未提供的 session 也列出来，避免质量报告只展示有数据的天。
        observed = {row['session_id'] for row in report['sessions']}
        step = interval * 1_000_000
        for key, session in calendar.sessions.items():
            if key not in observed and session['open'].value < previous and session['close'].value >= first_seen:
                expected = sum(end.value // step - start.value // step for start, end in session['intervals'])
                report['sessions'].append(dict(session_id=key, open_utc=str(session['open']),
                    close_utc=str(session['close']), observed_snapshots=0, expected_grid_points=int(expected),
                    missing_grid_points=int(expected), coverage_complete=False, internal_missing_grid_points=0,
                    continuous_blocks=0, feature_valid=0, learning_ready=0, first_snapshot=None,
                    last_snapshot=None, labels={str(h): dict(valid=0, feature_and_label_valid=0, reasons={})
                                               for h in horizons_ms}))
        report['sessions'].sort(key=lambda row: row['session_id'])
        report['output_rows'] = sum(row['observed_snapshots'] for row in report['sessions'])
        (staging / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False,
                                                       indent=2) + '\n', encoding='utf-8')
        (staging / 'quality.md').write_text(render_quality_report(report), encoding='utf-8')
        staging.rename(output)
    return report
