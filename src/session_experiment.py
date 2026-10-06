"""四段时间协议、逐日读取、冻结身份与独立日美元汇总。

物理第3页 §3.1、第4页 Eq.(2)–(4)及第7页时间隔离。所有日期仍属开发，
不声称未触碰测试；源码变动后必须另存新计划，历史身份不能迁就当前目录。
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.artifacts import pretty_json
from src.config import BASE_DIR
from src.model_library import RidgeModel
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader, training_samples
from src.snapshot_dataset import DEFAULT_CALENDAR, SessionCalendar, sha256_file, utc_ns

PHASES = ('train', 'calibration', 'validation', 'test')
CODE_FILES = ('run_project.py',) + tuple(sorted(str(p.relative_to(BASE_DIR))
              for p in (BASE_DIR / 'src').glob('*.py')))

def code_hashes():
    """冻结代码与数据一起绑定；修复代码后须另存新计划，不冒充原冻结实验。"""
    return {name: sha256_file(BASE_DIR / name) for name in CODE_FILES}

def fingerprint(document):
    """规范化 JSON 内容哈希，检测冻结计划被无意编辑；这不是密码学签名。"""
    payload = {k: v for k, v in document.items() if k != 'plan_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()

def validate_protocol(protocol, calendar, interval_ms):
    """四段必须包含连续、不重复、有序的完整 session，不能按收益跳过日期。

    日期是 trade date，不是 UTC 文件日期；星期休市自然不在交易日历中。
    先固定诊断窗口，所有配置由冻结文件提供，运行阶段没有覆盖参数的入口。
    """
    required = {'schema_version', 'purpose', 'sessions', 'prediction_horizon_ms',
                'reward_horizons_ms', 'latency_ms', 'holding_review_ms', 'threshold',
                'allow_partial', 'include_degraded'}
    if set(protocol) - (required | {'notes'}) or required - set(protocol):
        raise ValueError('Protocol keys must match the documented schema')
    if protocol['schema_version'] != 1 or protocol['purpose'] != 'development_validation':
        raise ValueError('This entry supports explicit development-validation protocols only')
    if set(protocol['sessions']) != set(PHASES):
        raise ValueError('Need train/calibration/validation/test session lists')
    flattened = []
    for phase in PHASES:
        days = protocol['sessions'][phase]
        if (not isinstance(days, list) or not days or any(not isinstance(d, str) for d in days)
                or any(d not in calendar.sessions for d in days) or days != sorted(set(days))):
            raise ValueError(f'Invalid ordered calendar session list: {phase}')
        flattened.extend(days)
    if flattened != sorted(set(flattened)):
        raise ValueError('Experiment phases must be strictly chronological and disjoint')
    expected = [d for d in calendar.sessions if flattened[0] <= d <= flattened[-1]]
    if flattened != expected:
        raise ValueError('Protocol cannot silently omit an intervening trading session')
    horizons = protocol['reward_horizons_ms']
    if (not isinstance(horizons, list) or not horizons
            or any(type(h) is not int or h <= 0 or h % interval_ms or h > 7 * 86400000 for h in horizons)
            or horizons != sorted(set(horizons))):
        raise ValueError('Reward horizons must be increasing positive grid multiples')
    if (type(protocol['prediction_horizon_ms']) is not int
            or protocol['prediction_horizon_ms'] not in horizons):
        raise ValueError('Prediction horizon must be one of the configured reward horizons')
    if any(type(protocol[k]) is not int or protocol[k] <= 0 for k in ('latency_ms', 'holding_review_ms')):
        raise ValueError('Execution intervals must be positive integer milliseconds')
    if (type(protocol['threshold']) not in (float, int) or not np.isfinite(protocol['threshold'])
            or protocol['threshold'] < 0):
        raise ValueError('Threshold must be a finite nonnegative relative return')
    if any(type(protocol[k]) is not bool for k in ('allow_partial', 'include_degraded')):
        raise ValueError('Source-quality permissions must be explicit booleans')
    return flattened

def freeze_protocol(protocol_path, directory, *, calendar_path=DEFAULT_CALENDAR):
    """绑定协议、源码、日历与 prepared 文件哈希，不计算测试收益或选择参数。

    每个完整 session 所需的相邻 UTC 源分区都必须存在；网格完整率另记，不能
    因原始分区存在就声称全网格完整。前缀/降级默认拒绝，显式允许也保留标记。
    """
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))['protocol']
    calendar = SessionCalendar(calendar_path)
    # 先验证控制项再读取来源，避免非布尔值被当成隐式授权。
    if any(type(protocol.get(k)) is not bool for k in ('allow_partial', 'include_degraded')):
        raise ValueError('Protocol must declare boolean source-quality permissions')
    reader = PreparedDatasetReader(directory, calendar=calendar,
        allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
    days = validate_protocol(protocol, calendar, reader.report['interval_ms'])
    if protocol['prediction_horizon_ms'] not in reader.report['horizons_ms']:
        raise ValueError('Prepared input must carry the prediction horizon for cache auditing')
    dates = {item['metadata']['source_date_utc'] for item in reader.report['inputs']}
    quality = {row['session_id']: row for row in reader.report['sessions']}
    schedules, coverage = {}, []
    for day in days:
        session = calendar.sessions[day]
        needed = {t.date().isoformat() for t in pd.date_range(
            session['open'].normalize(), session['close'].normalize(), freq='D')}
        if not needed <= dates:
            raise ValueError(f'Session {day} needs missing UTC source partitions: {sorted(needed - dates)}')
        if day not in quality or quality[day]['observed_snapshots'] < 2:
            raise ValueError(f'Session {day} has no usable quote coverage')
        schedules[day] = dict(open_utc=session['open'].isoformat(), close_utc=session['close'].isoformat())
        coverage.append(dict(session_id=day, required_utc_partitions=sorted(needed),
            source_partitions_present=True, observed_snapshots=quality[day]['observed_snapshots'],
            expected_grid_points=quality[day]['expected_grid_points'],
            grid_coverage_complete=quality[day]['coverage_complete']))
    plan = dict(schema_version=1, plan_kind='frozen_session_protocol', protocol=protocol,
        protocol_source=dict(path=str(protocol_path), sha256=sha256_file(protocol_path)),
        dataset=reader.provenance, calendar=dict(path=str(Path(calendar_path).resolve()), sha256=calendar.sha256),
        interval_ms=reader.report['interval_ms'], schedules=schedules, source_coverage=coverage,
        prepared_split_boundaries=reader.report['split_boundaries'],
        partial_input=reader.report['partial_input'], degraded_input=reader.report['degraded_input'],
        code_sha256=code_hashes(),
        stage_roles=dict(train='bootstrap_history', calibration='expert_and_frozen_reward',
                         validation='development_no_retuning', test='frozen_development_evaluation'),
        holdout_claim='not_claimed_all_windows_are_development_validation')
    plan['plan_sha256'] = fingerprint(plan)
    return plan

def read_day(reader, day):
    """按完整 (open,close] 调度，保留关闭边界快照作盯市/反馈，不在边界成交。"""
    session = reader.calendar.sessions[day]
    frame = reader.read_window(session['open'], session['close'] + pd.Timedelta(nanoseconds=1))
    if len(frame) < 2 or not frame.session_id.eq(day).all():
        raise ValueError(f'Bad single-session quote coverage: {day}')
    return frame

def audit_horizons(frame, calendar, horizons_ms):
    """按当前行情重算各尺度的有效量和缺口原因，不使用 prepared 旧切分标签。

    只输出数据可用性统计，不把这些未来条件交给执行器筛选决策。最长尺度的
    联合有效量是后续 IRL 的数据前提，单个尺度有样本不代表完整奖励有样本。
    """
    times = utc_ns(frame.ts_event)
    sessions = frame.session_id.to_numpy()
    blocks = frame.continuous_block.to_numpy()
    segments = frame.segment_id.to_numpy()
    end_by_segment = dict(zip(calendar.segments.segment_id, calendar.ends))
    segment_ends = np.array([end_by_segment[s] for s in segments], dtype=np.int64)
    closing = calendar.sessions[sessions[0]]['close'].value
    available = frame.feature_valid.to_numpy()
    joint = available.copy()
    labels = {}
    for horizon in horizons_ms:
        target = times + horizon * 1_000_000
        index = np.searchsorted(times, target)
        safe = np.minimum(index, len(times) - 1)
        exact = (index < len(times)) & (times[safe] == target)
        reason = np.full(len(frame), 'valid', dtype=object)
        reason[~exact] = 'missing_target'
        reason[exact & ((blocks[safe] != blocks) | (sessions[safe] != sessions))] = 'gap'
        reason[target > segment_ends] = 'scheduled_break'
        reason[target > closing] = 'session_boundary'
        valid = reason == 'valid'
        joint &= valid
        labels[str(horizon)] = dict(valid_rows=int(valid.sum()),
            feature_and_label_valid_rows=int((available & valid).sum()),
            invalid_reasons={k: int(v) for k, v in Counter(reason[~valid]).items()})
    return dict(session_id=sessions[0], observed_rows=len(frame), feature_valid_rows=int(available.sum()),
                all_horizons_feature_valid_rows=int(joint.sum()), labels=labels)

def fit_streaming_ridge(reader, days, prediction_horizon_ms, purge_ms):
    """两遍逐 session 训练，内存上界为一天，不拼接整个季度的训练矩阵。

    第一遍 StandardScaler.partial_fit 仅累计训练均值/方差；第二遍在标准化
    特征上累计 X'X、X'y、列和与标签和。带截距的 Ridge 解为
    beta=(X'X-sx*sx'/N+alpha*I)^-1 (X'y-sx*sy/N)，alpha 固定为 1。
    这是 sklearn Ridge 同一目标的充分统计量计算，不是新论文模型或参数搜索。
    """
    scaler, rows = StandardScaler(), []
    total = 0

    def samples(day):
        frame = read_day(reader, day)
        session = reader.calendar.sessions[day]
        return frame, training_samples(frame, session['open'],
            session['close'] + pd.Timedelta(nanoseconds=1), prediction_horizon_ms, purge_ms, minimum_samples=0)

    for day in days:
        frame, (X, y, keep) = samples(day)
        if len(y):
            scaler.partial_fit(X)
        total += len(y)
        rows.append(dict(session_id=day, training_rows=len(y),
            first_training_time=str(frame.loc[keep, 'ts_event'].iloc[0]) if len(y) else None,
            last_training_time=str(frame.loc[keep, 'ts_event'].iloc[-1]) if len(y) else None))
    if total < 2:
        raise ValueError('Not enough purged samples across training sessions')
    width = len(FEATURE_COLUMNS)
    xx, xy, sx = np.zeros((width, width)), np.zeros(width), np.zeros(width)
    sy = 0.
    for day in days:
        _, (X, y, _) = samples(day)
        if not len(y):
            continue
        z = scaler.transform(X)
        xx += z.T @ z
        xy += z.T @ y
        sx += z.sum(axis=0)
        sy += y.sum()
    coefficients = np.linalg.solve(xx - np.outer(sx, sx) / total + np.eye(width), xy - sx * sy / total)
    intercept = float(sy / total - (sx / total) @ coefficients)
    model = RidgeModel(alpha=1.)
    # 赋值的是上述精确 Ridge 解；n_features_in_ 让 sklearn 的冻结预测接口正常校验。
    model.model.coef_, model.model.intercept_, model.model.n_features_in_ = coefficients, intercept, width
    model.is_trained = True
    metadata = dict(name=model.name, alpha=1., fit_method='two_pass_session_sufficient_statistics',
        training_rows=total, training_sessions=rows, coefficients=coefficients.tolist(), intercept=intercept,
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
    return scaler, model, metadata

def summarize_days(days, strategy):
    """日账户独立固定资金；只有所有日都平仓才定义总净盈亏，避免抹掉缺尾风险。

    保留无交易与亏损日。日资金重置不是单一复利账户；不足以年化夏普。
    尚未平仓日包含未实现盯市，不能混成全部已实现的累计收益。
    """
    rows = [next(r for r in day['results'] if r['strategy'] == strategy) for day in days]
    settled = all(r['terminal_position_liquidated'] for r in rows)
    return dict(strategy=strategy, sessions=len(rows), total_trades=sum(r['total_trades'] for r in rows),
        total_fills=sum(r['total_fills'] for r in rows), pnl_aggregation_defined=settled,
        gross_pnl_usd=float(sum(r['gross_pnl_usd'] for r in rows)) if settled else None,
        friction_usd=float(sum(r['friction_usd'] for r in rows)),
        net_pnl_usd=float(sum(r['net_pnl_usd'] for r in rows)) if settled else None,
        unsettled_sessions=[day['session_id'] for day, row in zip(days, rows) if not row['terminal_position_liquidated']],
        matured_order_count=sum(r['matured_order_count'] for r in rows),
        unmatured_fill_rewards_at_end=sum(r['unmatured_fill_rewards_at_end'] for r in rows),
        account_policy='independent_daily_accounts_no_compounding', sharpe_ratio=None)

def publish_bundle(document, report, directory, json_name):
    """输出必须是新目录；原子发布计划或结果，不覆盖历史实验。"""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite experiment directory: {directory}')
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.session-protocol-', dir=directory.parent) as temporary:
        staging = Path(temporary) / 'output'
        staging.mkdir()
        (staging / json_name).write_text(pretty_json(document) + '\n', encoding='utf-8')
        (staging / 'report.md').write_text(report, encoding='utf-8')
        staging.rename(directory)
