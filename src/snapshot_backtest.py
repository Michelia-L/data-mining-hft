"""只读按 session 准备的数据，构造不越过训练边界的标签。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.snapshot_dataset import SessionCalendar, parse_boundary, sha256_file, utc_ns, validate_batch
from src.timed_snapshots import SNAPSHOT_SCHEMA

# 特征均来自截至当前格的盘口；未来标签不得成为预测的可用性条件。
FEATURE_COLUMNS = ['spread', 'rel_spread', 'obi_l1', 'obi_multi', 'micro_dev',
                   'ret_lag_1', 'ret_lag_5', 'ret_lag_10', 'ret_lag_30', 'vol_10', 'vol_30']

def inspect_prepared_dataset(directory, calendar, allow_partial=False, include_degraded=False):
    """一次校验 prepared 元数据与文件哈希，供多 session 顺序读取复用。

    文件级哈希绑定实际内容；原始消息的来源记录继承数据准备报告，不能假称已
    重新验证原始消息。后续窗口读取仍逐条检查报价、因果时钟和 session。
    """
    directory = Path(directory).resolve()
    quality_path, parquet_path = directory / 'quality.json', directory / 'snapshots.parquet'
    report = json.loads(quality_path.read_text(encoding='utf-8'))
    if (report.get('schema_version') != 1 or report.get('dataset_kind') != 'session_time_labeled_snapshots'
            or report.get('symbol') != 'ESZ5' or report.get('feature_columns') != FEATURE_COLUMNS
            or report.get('label_policy') != 'exact_target_same_continuous_block'
            or report['calendar']['sha256'] != calendar.sha256 or report['calendar']['spec'] != calendar.spec):
        raise ValueError('Unsupported prepared dataset or calendar provenance')
    for name in ('partial_input', 'degraded_input'):
        if type(report.get(name)) is not bool:
            raise ValueError('Quality status must be explicit boolean values')
    if report['partial_input'] and not allow_partial:
        raise ValueError('Partial prepared input requires --allow-partial')
    if report['degraded_input'] and not include_degraded:
        raise ValueError('Degraded prepared input requires --include-degraded')
    interval, horizons = report['interval_ms'], report['horizons_ms']
    if (type(interval) is not int or interval <= 0 or not horizons
            or any(type(h) is not int or h <= 0 or h % interval or h > 7 * 86400000 for h in horizons)
            or sorted(set(horizons)) != horizons):
        raise ValueError('Invalid prepared sampling or horizon metadata')
    inputs = report['inputs']
    if (not inputs or len({(item['interval_ms'], item['max_age_ms']) for item in inputs}) != 1
            or inputs[0]['interval_ms'] != interval or type(inputs[0]['max_age_ms']) is not int
            or inputs[0]['max_age_ms'] <= 0):
        raise ValueError('Inconsistent prepared source settings')
    for item in inputs:
        meta = item['metadata']
        if (meta.get('partial_prefix') not in ('true', 'false')
                or meta.get('source_condition') not in ('available', 'degraded')
                or meta.get('sampling_clock') != 'ts_recv'):
            raise ValueError('Invalid prepared raw-source provenance')
    if (report['partial_input'] != any(i['metadata']['partial_prefix'] == 'true' for i in inputs)
            or report['degraded_input'] != any(i['metadata']['source_condition'] == 'degraded' for i in inputs)):
        raise ValueError('Quality status disagrees with recorded raw sources')
    file = pq.ParquetFile(parquet_path)
    metadata = file.schema_arrow.metadata or {}
    expected = {b'dataset_kind': b'session_time_labeled_snapshots',
                b'calendar_sha256': calendar.sha256.encode(), b'interval_ms': str(interval).encode(),
                b'horizons_ms': json.dumps(horizons).encode(),
                b'partial_input': str(report['partial_input']).lower().encode()}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Parquet metadata disagrees with quality report')
    if file.metadata.num_rows != report['output_rows']:
        raise ValueError('Prepared row count disagrees with quality report')
    # 明确选择所需列，不把 learning_ready 等未来有效性条件用作行情加载过滤器。
    columns = SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'continuous_block',
        'mid_price', 'feature_valid'] + FEATURE_COLUMNS
    for horizon in horizons:
        columns += [f'label_valid_{horizon}ms', f'label_end_{horizon}ms', f'future_return_{horizon}ms']
    if not set(columns).issubset(file.schema_arrow.names):
        raise ValueError('Prepared data is missing required columns')
    for field in SNAPSHOT_SCHEMA:
        actual = file.schema_arrow.field(field.name).type
        valid_type = (pa.types.is_timestamp(actual) and actual.tz == 'UTC'
                      if pa.types.is_timestamp(field.type) else actual == field.type)
        # pandas 新版可能将 Arrow string 写成 large_string；两者都是原样文本。
        if field.name == 'symbol':
            valid_type = pa.types.is_string(actual) or pa.types.is_large_string(actual)
        if not valid_type:
            raise ValueError(f'Prepared source schema mismatch: {field.name}')
    return file, report, dict(parquet_sha256=sha256_file(parquet_path),
                              quality_sha256=sha256_file(quality_path), path=str(directory))

def read_prepared_quotes(file, report, calendar, start, end, batch_size=65536):
    """根据 Parquet 行组时间范围跳过无关 session，只物化请求的 [start,end)。

    行组统计只用于减少 IO；缺少统计时保守读取，最终仍按真实时间筛选并校验。
    不根据 learning_ready 或未来标签剔除行情，也不事后排序掩盖输入逆序。
    """
    start, end = parse_boundary(start), parse_boundary(end)
    if start >= end or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Need an increasing explicit time window and positive batch size')
    interval, inputs = report['interval_ms'], report['inputs']
    columns = SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'continuous_block',
        'mid_price', 'feature_valid'] + FEATURE_COLUMNS
    for horizon in report['horizons_ms']:
        columns += [f'label_valid_{horizon}ms', f'label_end_{horizon}ms', f'future_return_{horizon}ms']
    clock_column = file.schema_arrow.names.index('ts_event')
    groups = []
    for i in range(file.metadata.num_row_groups):
        stats = file.metadata.row_group(i).column(clock_column).statistics
        if (stats is None or not stats.has_min_max
                or (pd.Timestamp(stats.max) >= start and pd.Timestamp(stats.min) < end)):
            groups.append(i)
    chunks = []
    for batch in file.iter_batches(batch_size=batch_size, columns=columns, row_groups=groups):
        frame = batch.to_pandas()
        selected = frame.loc[(frame.ts_event >= start) & (frame.ts_event < end)].copy()
        if not selected.empty:
            # 原始盘口字段不得缺失；滚动特征预热 NaN 与离线失效标签属于合法状态。
            validate_batch(selected[SNAPSHOT_SCHEMA.names].copy(), interval, inputs[0]['max_age_ms'])
            chunks.append(selected)
    if not chunks:
        raise ValueError('Requested window has no prepared snapshots')
    frame = pd.concat(chunks, ignore_index=True)
    times = utc_ns(frame.ts_event)
    if (np.diff(times) <= 0).any() or frame.instrument_id.nunique() != 1:
        raise ValueError('Prepared window must contain increasing single-contract snapshots')
    assigned, outside, preopen = calendar.assign(frame)
    if (outside or preopen or not assigned.session_id.equals(frame.session_id)
            or not assigned.segment_id.equals(frame.segment_id)):
        raise ValueError('Prepared session metadata disagrees with calendar')
    if (not np.isfinite(frame.mid_price).all()
            or not np.allclose(frame.mid_price, (frame.bid_px_00 + frame.ask_px_00) / 2, rtol=0, atol=1e-9)):
        raise ValueError('Prepared midprice disagrees with quotes')
    finite = np.isfinite(frame[FEATURE_COLUMNS].to_numpy(float)).all(axis=1)
    if frame.feature_valid.dtype != bool or not np.array_equal(finite, frame.feature_valid):
        raise ValueError('Prepared feature availability disagrees with feature values')
    actual_change = ((np.diff(times) != interval * 1_000_000)
                     | (frame.session_id.to_numpy()[1:] != frame.session_id.to_numpy()[:-1])
                     | (frame.segment_id.to_numpy()[1:] != frame.segment_id.to_numpy()[:-1]))
    declared_change = ((frame.continuous_block.to_numpy()[1:] != frame.continuous_block.to_numpy()[:-1])
                       | (frame.session_id.to_numpy()[1:] != frame.session_id.to_numpy()[:-1]))
    if not np.array_equal(actual_change, declared_change):
        raise ValueError('Prepared continuous blocks disagree with observed gaps')
    return frame

class PreparedDatasetReader:
    """复用一次元数据/哈希检查的读取器，内存仅包含当前 session 的行情。

    大训练集不必拼成完整季度 DataFrame；调用者可重复迭代训练 session 并累计
    统计量。文件在迭代期间改变时明确拒绝，避免一个实验混用两个数据版本。
    """
    def __init__(self, directory, *, calendar=None, allow_partial=False, include_degraded=False):
        self.calendar = calendar or SessionCalendar()
        self.file, self.report, self.provenance = inspect_prepared_dataset(
            directory, self.calendar, allow_partial, include_degraded)
        self.paths = [Path(directory) / name for name in ('snapshots.parquet', 'quality.json')]
        self.identities = [(p.stat().st_size, p.stat().st_mtime_ns) for p in self.paths]

    def read_window(self, start, end, batch_size=65536):
        """保留所有行情；特征与标签的训练筛选由上层独立处理。"""
        if [(p.stat().st_size, p.stat().st_mtime_ns) for p in self.paths] != self.identities:
            raise ValueError('Prepared files changed during experiment')
        return read_prepared_quotes(self.file, self.report, self.calendar, start, end, batch_size)

def training_samples(frame, start, end, prediction_horizon_ms, purge_ms, *, minimum_samples=2):
    """只筛训练样本，并验证未来目标确实在训练段内；不改变回放序列。

    本次窗口是唯一训练切分依据；旧 prepared split-boundary 只影响标签缓存，
    不能静默删掉当前窗口的合法样本。目标必须精确存在且在同一 session/连续
    区间，并满足最长奖励尺度的训练末端隔离。收益用匹配行情重新计算。
    旧 label_valid/return/end 仅在声称有效时做一致性审计，不参与样本筛选或 y。
    """
    start, end = parse_boundary(start), parse_boundary(end)
    if (start >= end or type(prediction_horizon_ms) is not int or prediction_horizon_ms <= 0
            or type(purge_ms) is not int or purge_ms < prediction_horizon_ms
            or type(minimum_samples) is not int or minimum_samples < 0):
        raise ValueError('Need an increasing training window and purge covering prediction horizon')
    times = utc_ns(frame.ts_event)
    horizon = prediction_horizon_ms * 1_000_000
    indices = np.searchsorted(times, times + horizon)
    safe = np.minimum(indices, len(frame) - 1)
    same = ((frame.session_id.to_numpy()[safe] == frame.session_id.to_numpy())
            & (frame.continuous_block.to_numpy()[safe] == frame.continuous_block.to_numpy()))
    suffix = f'{prediction_horizon_ms}ms'
    eligible = ((times >= start.value) & (times + purge_ms * 1_000_000 < end.value)
                & frame.feature_valid.to_numpy())
    declared = frame[f'label_valid_{suffix}'].to_numpy()
    if declared.dtype != bool:
        raise ValueError('Training label availability must be boolean')
    exact = (indices < len(times)) & (times[safe] == times + horizon) & same
    if (eligible & declared & ~exact).any():
        raise ValueError('Declared valid training label crosses a gap or has no exact target')
    keep = eligible & exact
    # 多 session 累积时允许本日零/一个样本，最后对全训练段检查数量；旧入口仍须至少两个。
    if keep.sum() < minimum_samples:
        raise ValueError('Need at least two finite, purged training samples')
    prices = frame.mid_price.to_numpy(float)
    # 相对收益是本基线的预测目标，Eq.(2) 的价格差奖励仍由执行器独立计算。
    y = (prices[safe[keep]] - prices[keep]) / prices[keep]
    audit = keep & declared
    cached = frame.loc[audit, f'future_return_{suffix}'].to_numpy(float)
    expected = (prices[safe[audit]] - prices[audit]) / prices[audit]
    ends = utc_ns(frame.loc[audit, f'label_end_{suffix}'])
    if (not np.isfinite(y).all() or not np.isfinite(cached).all()
            or not np.allclose(cached, expected, rtol=1e-10, atol=1e-12)
            or not np.array_equal(ends, times[audit] + horizon)):
        raise ValueError('Training labels disagree with observed target prices or times')
    return frame.loc[keep, FEATURE_COLUMNS].to_numpy(float), y, keep
