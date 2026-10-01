"""快照数据集到固定模型时间回放的独立小窗口实验。

论文第 3–4 页 §3.1/§3.2 要求历史训练模型；第 7 页使用按时间分离的测试。
这里固定 Ridge、训练段标准化、等权价格差奖励及现金基线，不进行测试选参，
也不声称已实现按周模型库、IRL 校准或论文选择器。只物化显式请求的时间窗口。
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.artifacts import pretty_json
from src.config import BASE_DIR, INSTRUMENT_CONFIG, SEED
from src.irl_reward import IRLRewardLearner
from src.model_library import RidgeModel
from src.model_selector import FlatSelector, SingleModelSelector
from src.snapshot_dataset import (DEFAULT_CALENDAR, SessionCalendar, parse_boundary,
                                  sha256_file, utc_ns, validate_batch)
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.time_execution import TimeExecutionEngine


# 与数据准备模块的 11 个因果特征保持一致；不能允许质量 JSON 把未来列列为特征。
FEATURE_COLUMNS = ['spread', 'rel_spread', 'obi_l1', 'obi_multi', 'micro_dev',
                   'ret_lag_1', 'ret_lag_5', 'ret_lag_10', 'ret_lag_30', 'vol_10', 'vol_30']


def read_prepared_window(directory, start, end, *, calendar=None, allow_partial=False,
                         include_degraded=False, batch_size=65536):
    """按批次读显式的 [start,end) 窗口，校验来源、时钟、盘口和 session。

    返回完整行情序列及质量记录；不能按 learning_ready/标签有效性过滤行情。
    训练标签在下一层单独使用，执行器只接收行情与因果特征。文件级哈希记录实际
    内容；原始数据的记录来自数据准备 provenance，不假称重新验证全部原始消息。
    """
    start, end = parse_boundary(start), parse_boundary(end)
    if start >= end or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Need an increasing explicit time window and positive batch size')
    directory = Path(directory).resolve()
    quality_path, parquet_path = directory / 'quality.json', directory / 'snapshots.parquet'
    report = json.loads(quality_path.read_text(encoding='utf-8'))
    calendar = calendar or SessionCalendar()
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
    chunks = []
    for batch in file.iter_batches(batch_size=batch_size, columns=columns):
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
    return frame, report, dict(parquet_sha256=sha256_file(parquet_path),
                              quality_sha256=sha256_file(quality_path), path=str(directory))


def training_samples(frame, start, end, prediction_horizon_ms, purge_ms):
    """只筛训练样本，并验证未来目标确实在训练段内；不改变回放序列。

    即使文件生成时没有传 split-boundary，也独立清除最长奖励尺度跨越训练末端
    的样本。只训练一个预测尺度，因此不要求其他离线标签均有效。目标必须精确
    存在且在同一 session/连续区间，防止被伪造的 label_valid 掩盖缺口或泄漏。
    """
    start, end = parse_boundary(start), parse_boundary(end)
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
    keep = eligible & declared & exact
    if keep.sum() < 2:
        raise ValueError('Need at least two finite, purged training samples')
    y = frame.loc[keep, f'future_return_{suffix}'].to_numpy(float)
    expected = frame.mid_price.to_numpy()[safe[keep]] / frame.mid_price.to_numpy()[keep] - 1
    ends = utc_ns(frame.loc[keep, f'label_end_{suffix}'])
    if (not np.isfinite(y).all() or not np.allclose(y, expected, rtol=1e-10, atol=1e-12)
            or not np.array_equal(ends, times[keep] + horizon)):
        raise ValueError('Training labels disagree with observed target prices or times')
    return frame.loc[keep, FEATURE_COLUMNS].to_numpy(float), y, keep


def run_snapshot_experiment(directory, *, train_start, train_end, test_start, test_end,
                            calendar_path=DEFAULT_CALENDAR, prediction_horizon_ms=5000,
                            latency_ms=500, holding_review_ms=15000, threshold=None,
                            allow_partial=False, include_degraded=False, detail=False):
    """冻结一个 Ridge 和现金基线，在用户明确的开发窗口上验证时间执行链路。

    训练与测试按时间分离，不比较参数候选，不校准 IRL。该入口的测试窗口默认
    记为开发验证用途，不能将多次观察过的窗口声称为未触碰的正式样本外测试。
    """
    bounds = [parse_boundary(v) for v in (train_start, train_end, test_start, test_end)]
    a, b, c, d = bounds
    if not a < b <= c < d:
        raise ValueError('Require train_start < train_end <= test_start < test_end')
    calendar = SessionCalendar(calendar_path)
    frame, quality, provenance = read_prepared_window(directory, a, d, calendar=calendar,
        allow_partial=allow_partial, include_degraded=include_degraded)
    horizons = quality['horizons_ms']
    if type(prediction_horizon_ms) is not int or prediction_horizon_ms not in horizons:
        raise ValueError('Prediction horizon must be present in prepared dataset')
    reward = IRLRewardLearner(horizons=horizons, definition='paper_price_difference')
    engine = TimeExecutionEngine(reward, interval_ms=quality['interval_ms'], latency_ms=latency_ms,
                                holding_review_ms=holding_review_ms, calendar=calendar, threshold=threshold)
    X, y, training_mask = training_samples(frame, a, b, prediction_horizon_ms, max(horizons))
    scaler, model = StandardScaler(), RidgeModel(alpha=1.)
    test = frame.loc[(frame.ts_event >= c) & (frame.ts_event < d)].reset_index(drop=True)
    if len(test) < 2 or not test.feature_valid.any():
        raise ValueError('Test window needs quotes and causal feature-valid rows')
    if test.instrument_id.iloc[0] != frame.loc[training_mask, 'instrument_id'].iloc[0]:
        raise ValueError('Training and test contracts must match')
    predictions = np.full((len(test), 1), np.nan)
    valid = test.feature_valid.to_numpy()
    with threadpool_limits(limits=1):
        model.fit(scaler.fit_transform(X), y)
        predictions[valid, 0] = model.predict(scaler.transform(test.loc[valid, FEATURE_COLUMNS].to_numpy(float)))
    # 从回放输入中移除全部未来标签，强化代码层面的信息边界。
    replay_columns = SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'mid_price', 'feature_valid']
    replay = test[replay_columns]
    fixed = SingleModelSelector('Time-Baseline-Fixed-Ridge', [model])
    fixed.reward_type = 'OE'
    cash = FlatSelector('Time-Baseline-Cash', [model])
    results = [engine.run_backtest(selector, replay, predictions, detail=detail) for selector in (fixed, cash)]
    return dict(schema_version=1, result_kind='snapshot_time_baseline', seed=SEED,
        experiment_purpose='development_validation_not_untouched_holdout',
        paper_basis='PDF pp.3–5 §3.1–3.3 Eq.(2)–(4); p.7 time-separated training/test',
        split={name: stamp.isoformat() for name, stamp in zip(
            ('train_start', 'train_end', 'test_start', 'test_end'), bounds)},
        training_rows=int(training_mask.sum()), training_purge_ms=max(horizons),
        training_first_time=str(frame.loc[training_mask, 'ts_event'].iloc[0]),
        training_last_time=str(frame.loc[training_mask, 'ts_event'].iloc[-1]),
        test_rows=len(test), test_feature_valid_rows=int(valid.sum()),
        prediction_horizon_ms=prediction_horizon_ms, feature_columns=FEATURE_COLUMNS,
        model=dict(name=model.name, alpha=1., coefficients=model.model.coef_.tolist(),
                   intercept=float(model.model.intercept_), scaler_mean=scaler.mean_.tolist(),
                   scaler_scale=scaler.scale_.tolist()),
        execution=dict(instrument=INSTRUMENT_CONFIG['CME_ES'], threshold=engine.threshold,
                       latency_ms=latency_ms, holding_review_ms=holding_review_ms,
                       interval_ms=quality['interval_ms'], reward_weights=reward.weights.tolist(),
                       reward_horizons_ms=horizons, reward_weights_source='fixed_equal_not_IRL'),
        dataset=dict(**provenance, calendar_sha256=calendar.sha256,
            partial_input=quality['partial_input'], degraded_input=quality['degraded_input'],
            source_inputs=quality['inputs'], quality_sessions=quality['sessions']),
        code_sha256={name: sha256_file(BASE_DIR / name) for name in
            ('run_snapshot_backtest.py', 'src/snapshot_backtest.py', 'src/time_execution.py',
             'src/execution_engine.py', 'src/irl_reward.py', 'src/model_library.py', 'src/model_selector.py')},
        environment=dict(python=platform.python_version(), **{name: importlib.metadata.version(name)
            for name in ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'threadpoolctl')}),
        limits=['固定模型与等权奖励；尚未接入 IRL 或论文 Algorithm 2/3。',
                'CME ESZ5 与论文中国商品期货不同；本次收益不代表原论文数值复现。',
                '只物化明确时间窗口；尚未支持全季度训练的外存算法。',
                '风险退出与期末平仓采用报价模拟，无排队、冲击、保证金或实盘成交保证。'],
        results=results)


def render_report(result):
    """输出本入口自己的中文报告，避免把新时间结构冒充旧事件实验 schema。"""
    lines = ['# 快照时间回放开发验证', '',
        '用途：开发链路验证；不是未触碰的正式测试，也不是论文完整 FMATO 实验。', '',
        f"训练样本：{result['training_rows']}；训练末端隔离：{result['training_purge_ms']} ms。",
        f"测试行情：{result['test_rows']}；特征有效：{result['test_feature_valid_rows']}。", '',
        '| 基线 | 完整交易 | 成交 | 毛盈亏 USD | 成本 USD | 净盈亏 USD | 成熟 OE | 期末未成熟 OE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['results']:
        lines.append(f"| {row['strategy']} | {row['total_trades']} | {row['total_fills']} | "
            f"{row['gross_pnl_usd']:.4f} | {row['friction_usd']:.4f} | {row['net_pnl_usd']:.4f} | "
            f"{row['matured_order_count']} | {row['unmatured_fill_rewards_at_end']} |")
        if not row['terminal_position_liquidated']:
            lines += ['', f"{row['strategy']} 期末未能平仓，仓位={row['terminal_position']}；净盈亏含未实现盯市。"]
    lines += ['', '## 切分与单位', '', pretty_json(result['split']), '',
              pretty_json(result['execution']), '', '## 边界与反馈审计', '']
    for row in result['results']:
        lines += [row['strategy'], '', pretty_json({key: row[key] for key in
            ('reward_status', 'risk_exits', 'cancelled_intents', 'unexecuted_intents_at_end')}), '']
    lines += ['## 复现限制', ''] + ['- ' + note for note in result['limits']]
    return '\n'.join(lines) + '\n'


def write_results(result, directory):
    """在新目录发布 JSON 和中文报告，不覆盖已有正式结果或失败后留下半份报告。"""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite results: {directory}')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.snapshot-backtest-', dir=directory.parent) as temporary:
        staging = Path(temporary) / 'result'
        staging.mkdir()
        (staging / 'result.json').write_text(pretty_json(result) + '\n', encoding='utf-8')
        (staging / 'report.md').write_text(render_report(result), encoding='utf-8')
        staging.rename(directory)
