"""§3.2 / Table 2 的轻模型库：两类算法×三类特征×两个历史期，按周更新。

具体特征、短历史和12个候选均为工程设定；盘口挂单深度不等于成交量。
各版本只能使用当时已结束历史，未来已离线生成的版本不能提前预测。
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits
from src.session_experiment import code_hashes, fingerprint, fit_streaming_ridge, read_day
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader, training_samples
from src.snapshot_dataset import DEFAULT_CALENDAR, SessionCalendar, sha256_file, utc_ns
from src.timed_snapshots import SIZE_FIELDS

def library_code_hashes():
    """新模型绑定当前完整主线；不会更新旧快照的历史源码身份。"""
    return code_hashes()

# Table 2 的三类是原文依据；下列列名/公式/10 格窗口是项目明示的具体化。
# 价格特征为相对收益/相对均价/相对方差；量的单位是前五档双边挂单合约数。
FEATURE_GROUPS = {
    'price': ['ret_lag_1', 'ret_lag_5', 'ret_lag_10', 'ret_lag_30', 'vol_10', 'vol_30',
              'mid_ma_rel_10', 'mid_var_rel_10'],
    'displayed_volume': ['displayed_depth', 'depth_diff_1', 'depth_ma_10', 'depth_var_10',
                         'depth_speed_5', 'obi_l1', 'obi_multi'],
    'price_volume': ['price_diff_1_x_depth', 'price_diff_1_div_depth', 'micro_dev', 'rel_spread'],
}
ALL_LIBRARY_FEATURES = list(dict.fromkeys(c for cols in FEATURE_GROUPS.values() for c in cols))


def library_features(frame, interval_ms):
    """输入 N 条完整 session 行情，输出同索引 N×20 因果特征（三组使用其中 19 列）。

    使用截至当前格的右对齐 10 格均值/总体方差和 1/5 格差分。每个实际连续块
    独立预热，缺格/暂停后的历史不能借用；每格是 interval_ms 毫秒，五格速度
    的单位为合约/秒。价量乘/除分别为价格×合约、价格/合约。双边一档正深度
    保证总深度非零，不用未来标签或后续价格确定特征。原有 11 特征保持原值。
    """
    output = frame[FEATURE_COLUMNS].copy()
    blocks = pd.MultiIndex.from_frame(frame[['session_id', 'continuous_block']])
    depth = frame[SIZE_FIELDS].sum(axis=1).astype(float)
    mid = frame.mid_price.astype(float)

    def rolling(series, operation):
        groups = series.groupby(blocks, sort=False).rolling(10, min_periods=10)
        values = groups.mean() if operation == 'mean' else groups.var(ddof=0)
        return values.reset_index(level=0, drop=True).reindex(frame.index)

    depth_group, mid_group = depth.groupby(blocks, sort=False), mid.groupby(blocks, sort=False)
    difference = mid - mid_group.shift(1)
    output['mid_ma_rel_10'] = rolling(mid, 'mean') / mid - 1
    output['mid_var_rel_10'] = rolling(mid, 'var') / mid.pow(2)
    output['displayed_depth'] = depth
    output['depth_diff_1'] = depth - depth_group.shift(1)
    output['depth_ma_10'] = rolling(depth, 'mean')
    output['depth_var_10'] = rolling(depth, 'var')
    output['depth_speed_5'] = (depth - depth_group.shift(5)) / (5 * interval_ms / 1000)
    output['price_diff_1_x_depth'] = difference * depth
    output['price_diff_1_div_depth'] = difference / depth
    return output

def validate_library_config(config, calendar):
    """开发计划预声明窗口、启动例外、模型超参数和日期，不允许运行覆盖。"""
    required = {'schema_version', 'purpose', 'history_start_session', 'bootstrap_session', 'through_session',
        'evaluation_sessions', 'history_session_counts', 'age_candidates_ms', 'prediction_horizon_ms',
        'purge_ms', 'ridge_alpha', 'tree_max_depth', 'tree_min_samples_leaf', 'tree_sample_limit',
        'seed', 'allow_partial', 'include_degraded'}
    if (set(config) - (required | {'notes'}) or required - set(config)
            or config['schema_version'] != 1 or config['purpose'] != 'development_weekly_light_library'):
        raise ValueError('Unsupported weekly library configuration')
    a, b, c = (config[k] for k in ('history_start_session', 'bootstrap_session', 'through_session'))
    if any(d not in calendar.sessions for d in (a, b, c)) or not a < b <= c:
        raise ValueError('Need chronological calendar history/bootstrap/through sessions')
    days = config['evaluation_sessions']
    if (not isinstance(days, list) or not days or days != sorted(set(days))
            or any(d not in calendar.sessions or not b <= d <= c for d in days)):
        raise ValueError('Evaluation must be declared after bootstrap and within schedule')
    for key in ('history_session_counts', 'age_candidates_ms'):
        values = config[key]
        if (not isinstance(values, list) or not values or values != sorted(set(values))
                or any(type(v) is not int or v <= 0 for v in values)):
            raise ValueError('Need ordered positive history/age candidates')
    if len(config['history_session_counts']) < 2:
        raise ValueError('Need at least two different historical periods')
    for key in ('prediction_horizon_ms', 'purge_ms', 'tree_max_depth', 'tree_min_samples_leaf', 'tree_sample_limit'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError('Need positive explicit model/time parameters')
    if config['purge_ms'] < config['prediction_horizon_ms']:
        raise ValueError('Purge must cover prediction maturity')
    if (type(config['ridge_alpha']) not in (int, float) or not np.isfinite(config['ridge_alpha'])
            or config['ridge_alpha'] <= 0 or type(config['seed']) is not int or not 0 <= config['seed'] < 2**32):
        raise ValueError('Need finite positive regularization and explicit random seed')
    if config['tree_sample_limit'] < 2 or any(type(config[k]) is not bool for k in ('allow_partial', 'include_degraded')):
        raise ValueError('Need usable sample bound and explicit data permissions')

def weekly_schedule(config, calendar):
    """首版允许显式日中周启动，后续只在每个 ISO 周首个交易 session 开盘换版。

    使用 trade date 的周，开盘的 UTC 文件日期可能是前一日。节假日从版本化
    日历确定该周首个交易日，不用固定 UTC 星期一或看到收益后择日。历史窗口
    必须是启动日前的最近 k 个完整 session，缺数据报错而不回退到更老日期。
    """
    bootstrap = config['bootstrap_session']
    first_by_week = {}
    for day in calendar.sessions:
        week = pd.Timestamp(day).isocalendar()[:2]
        first_by_week.setdefault(tuple(week), day)
    updates = [bootstrap] + [d for d in first_by_week.values() if bootstrap < d <= config['through_session']]
    schedule = []
    for day in updates:
        history = [d for d in calendar.sessions if config['history_start_session'] <= d < day]
        if len(history) < max(config['history_session_counts']):
            raise ValueError('Not enough complete historical sessions before update')
        schedule.append(dict(update_session=day, available_at_utc=calendar.sessions[day]['open'].isoformat(),
            update_kind='bootstrap' if day == bootstrap else 'weekly_first_session',
            training_sessions={str(k): history[-k:] for k in config['history_session_counts']}))
    return schedule

def freeze_library(config_path, directories, calendar_path=DEFAULT_CALENDAR):
    """绑定全部年龄与来源；在交易评价之前冻结模型计划。"""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))['library']
    calendar = SessionCalendar(calendar_path)
    validate_library_config(config, calendar)
    schedule = weekly_schedule(config, calendar)
    required_days = sorted(set(config['evaluation_sessions']) | {
        d for s in schedule for days in s['training_sessions'].values() for d in days})
    cases, signatures = [], []
    if len(directories) != len(config['age_candidates_ms']):
        raise ValueError('Need every declared age dataset exactly once')
    for directory in directories:
        reader = PreparedDatasetReader(directory, calendar=calendar,
            allow_partial=config['allow_partial'], include_degraded=config['include_degraded'])
        interval = reader.report['interval_ms']
        if (config['prediction_horizon_ms'] not in reader.report['horizons_ms']
                or config['prediction_horizon_ms'] % interval or config['purge_ms'] % interval):
            raise ValueError('Prediction/purge must be declared exact grid multiples')
        dates = {r['metadata']['source_date_utc'] for r in reader.report['inputs']}
        quality = {r['session_id']: r for r in reader.report['sessions']}
        for day in required_days:
            session = calendar.sessions[day]
            needed = {t.date().isoformat() for t in pd.date_range(session['open'].normalize(),
                                                                session['close'].normalize(), freq='D')}
            if (not needed <= dates or day not in quality or quality[day]['observed_snapshots'] < 2):
                raise ValueError(f'Missing complete source coverage for {day}')
        signatures.append([(r['metadata']['source_date_utc'], r['metadata']['source_sha256'],
                            r['metadata']['source_condition'], r['metadata']['partial_prefix'])
                           for r in reader.report['inputs']])
        cases.append(dict(max_age_ms=reader.report['inputs'][0]['max_age_ms'], dataset=reader.provenance,
                          interval_ms=interval, source_coverage_sessions=required_days))
    cases.sort(key=lambda r: r['max_age_ms'])
    if [r['max_age_ms'] for r in cases] != config['age_candidates_ms']:
        raise ValueError('Age dataset does not match declared candidates')
    if any(s != signatures[0] for s in signatures) or len({r['interval_ms'] for r in cases}) != 1:
        raise ValueError('Age candidates must share raw sources and time grid')
    if cases[0]['interval_ms'] not in config['age_candidates_ms']:
        raise ValueError('Strict one-grid-age baseline required')
    plan = dict(schema_version=1, plan_kind='frozen_weekly_light_library', config=config,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)),
        calendar=dict(path=str(Path(calendar_path).resolve()), sha256=calendar.sha256), cases=cases,
        schedule=schedule, feature_groups=FEATURE_GROUPS, code_sha256=library_code_hashes(),
        holdout_claim='none_all_dates_used_for_development', age_selection='none')
    plan['plan_sha256'] = fingerprint(plan)
    return plan

def history_samples(reader, day, config):
    """精确重算 5 秒等预测标签，返回 N×19 特征、N 标签和 N 个样本时刻。

    训练只用合法连续块，每日末端 purge 覆盖原长期奖励尺度。附加特征共同
    有限掩码使所有候选采用同一可用性标准，不用未来标签有效标记筛选预测。
    """
    frame = read_day(reader, day)
    session = reader.calendar.sessions[day]
    _, y, keep = training_samples(frame, session['open'], session['close'] + pd.Timedelta(nanoseconds=1),
        config['prediction_horizon_ms'], config['purge_ms'], minimum_samples=0)
    features = library_features(frame, reader.report['interval_ms']).loc[keep, ALL_LIBRARY_FEATURES]
    finite = np.isfinite(features.to_numpy(float)).all(axis=1)
    return features.loc[finite].to_numpy(float), y[finite], utc_ns(frame.loc[keep].loc[finite, 'ts_event'])

def fit_history(reader, days, config, available_at):
    """两遍训练三组 Ridge，并对有界均匀样本拟合三棵浅树。

    Ridge 用全部合法样本的标准化充分统计量，不拼接季度矩阵。树不支持
    partial_fit，因此用每行独立 U(0,1) 优先级取最小 K 项，等价于均匀无放回
    抽样；固定种子且与标签无关，不取有利价格或只取文件前缀。内存上界为
    当前 session 加 K 行。抽样/树参数均是项目假设，样本时刻哈希可追溯。
    """
    scalers = {name: StandardScaler() for name in FEATURE_GROUPS}
    indices = {name: [ALL_LIBRARY_FEATURES.index(c) for c in cols] for name, cols in FEATURE_GROUPS.items()}
    rng = np.random.default_rng(config['seed'])
    sample_x, sample_y, sample_t, priorities = np.empty((0, len(ALL_LIBRARY_FEATURES))), np.empty(0), np.empty(0, dtype='int64'), np.empty(0)
    counts, total = [], 0
    for day in days:
        if reader.calendar.sessions[day]['close'] >= pd.Timestamp(available_at):
            raise ValueError('Training session not completed before version availability')
        x, y, times = history_samples(reader, day, config)
        counts.append(dict(session_id=day, samples=len(y)))
        total += len(y)
        if not len(y):
            continue
        for name, cols in indices.items():
            scalers[name].partial_fit(x[:, cols])
        sample_x = np.concatenate([sample_x, x]); sample_y = np.concatenate([sample_y, y])
        sample_t = np.concatenate([sample_t, times]); priorities = np.concatenate([priorities, rng.random(len(y))])
        count = min(config['tree_sample_limit'], len(priorities))
        selected = np.argpartition(priorities, count - 1)[:count]
        # 排序使同一优先级样本的树训练输入确定；不能按收益或标签排样本。
        selected = selected[np.argsort(priorities[selected])]
        sample_x, sample_y, sample_t, priorities = (v[selected] for v in (sample_x, sample_y, sample_t, priorities))
    if total < 2:
        return [dict(family=family, feature_group=name, status='insufficient_samples',
                     training_rows=total, training_sessions=counts)
                for name in FEATURE_GROUPS for family in ('Ridge', 'DecisionTree')]
    sums = {name: [np.zeros((len(cols), len(cols))), np.zeros(len(cols)), np.zeros(len(cols)), 0.]
            for name, cols in indices.items()}
    for day in days:
        x, y, _ = history_samples(reader, day, config)
        if not len(y):
            continue
        for name, cols in indices.items():
            z = scalers[name].transform(x[:, cols])
            stats = sums[name]
            stats[0] += z.T @ z; stats[1] += z.T @ y; stats[2] += z.sum(axis=0); stats[3] += y.sum()
    models = []
    for name, cols in indices.items():
        scaler = scalers[name]
        xx, xy, sx, sy = sums[name]
        coefficients = np.linalg.solve(xx - np.outer(sx, sx) / total + config['ridge_alpha'] * np.eye(len(cols)),
                                       xy - sx * sy / total)
        common = dict(feature_group=name, features=FEATURE_GROUPS[name], status='fitted', training_rows=total,
            training_sessions=counts, scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
        models.append(common | dict(family='Ridge', coefficients=coefficients.tolist(),
            intercept=float(sy / total - (sx / total) @ coefficients), alpha=config['ridge_alpha']))
        tree = DecisionTreeRegressor(max_depth=config['tree_max_depth'], min_samples_leaf=config['tree_min_samples_leaf'],
                                     random_state=config['seed']).fit(scaler.transform(sample_x[:, cols]), sample_y)
        state = tree.tree_
        models.append(common | dict(family='DecisionTree', tree_sample_rows=len(sample_y),
            tree_sample_times_sha256=hashlib.sha256(sample_t.astype('<i8').tobytes()).hexdigest(),
            tree=dict(children_left=state.children_left.tolist(), children_right=state.children_right.tolist(),
                feature=state.feature.tolist(), threshold=state.threshold.tolist(),
                value=state.value.reshape(-1).tolist(), n_node_samples=state.n_node_samples.tolist())))
    return models

def predict_model(model, features):
    """消费 JSON 模型参数，输出 N 个相对收益预测；树参数不需要 pickle。

    sklearn 的树预测将输入转 float32，本函数也如此，避免阈值附近因精度
    不同而走另一分支。标准化只消费该版本训练期均值/尺度，不能在预测段拟合。
    """
    x = features[model['features']].to_numpy(float)
    z = (x - np.asarray(model['scaler_mean'])) / np.asarray(model['scaler_scale'])
    if model['family'] == 'Ridge':
        return z @ np.asarray(model['coefficients']) + model['intercept']
    tree, z = model['tree'], z.astype(np.float32)
    nodes = np.zeros(len(z), dtype=int)
    left, right = np.asarray(tree['children_left']), np.asarray(tree['children_right'])
    split, thresholds = np.asarray(tree['feature']), np.asarray(tree['threshold'])
    active = left[nodes] != -1
    while active.any():
        rows = np.flatnonzero(active); current = nodes[rows]
        goes_left = z[rows, split[current]] <= thresholds[current]
        nodes[rows] = np.where(goes_left, left[current], right[current])
        active = left[nodes] != -1
    return np.asarray(tree['value'])[nodes]

def predict_library(versions, frame, interval_ms):
    """按每行可见时间选择版本，输出 N×K 预测和 N 个版本 ID。

    未来版本虽已在离线文件中建好，也不能提前应用。启动前保留 NaN 和 None，
    预热/不足样本仍保留 NaN。候选 ID 在版本间稳定，版本 ID 则绑定实际模型。
    真实时间引擎保留原版本持仓/反馈归属，换版时取消未执行的旧意图。
    """
    if not versions:
        raise ValueError('Need scheduled model versions')
    available = pd.DatetimeIndex([v['available_at_utc'] for v in versions])
    if not available.is_monotonic_increasing or available.has_duplicates:
        raise ValueError('Version availability must increase strictly')
    for version in versions:
        if version['version_sha256'] != fingerprint({k: v for k, v in version.items() if k != 'version_sha256'}):
            raise ValueError('Model version integrity check failed')
        if version['interval_ms'] != interval_ms:
            raise ValueError('Prediction grid disagrees with model version')
    if (np.diff(utc_ns(frame.ts_event)) <= 0).any():
        raise ValueError('Prediction quotes must increase strictly')
    ids = [m['model_id'] for m in versions[0]['models']]
    if any([m['model_id'] for m in v['models']] != ids for v in versions):
        raise ValueError('Candidate model identities must stay stable')
    # pandas 的字符串 DatetimeIndex 可能为微秒精度；不能与行情纳秒整数直接比较。
    # 两侧显式转纳秒，防止未来版本因单位差异看起来已经可用。
    positions = np.searchsorted(utc_ns(pd.Series(available)), utc_ns(frame.ts_event), side='right') - 1
    features = library_features(frame, interval_ms)
    valid = frame.feature_valid.to_numpy() & np.isfinite(features[ALL_LIBRARY_FEATURES].to_numpy(float)).all(axis=1)
    predictions, version_ids = np.full((len(frame), len(ids)), np.nan), np.full(len(frame), None, dtype=object)
    for i, version in enumerate(versions):
        selected = positions == i
        if (frame.loc[selected, 'age_ms'] > version['max_age_ms']).any():
            raise ValueError('Prediction quote age exceeds frozen version bound')
        version_ids[selected] = version['version_sha256']
        rows = np.flatnonzero(selected & valid)
        for j, model in enumerate(version['models']):
            if len(rows) and model['status'] == 'fitted':
                predictions[rows, j] = predict_model(model, features.iloc[rows])
    return predictions, version_ids

def build_library(plan_path):
    """生成可序列化周版本和校准参照；逐年龄保留，不读取后段收益。"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if (plan.get('plan_kind') != 'frozen_weekly_light_library' or plan.get('schema_version') != 1
            or plan.get('plan_sha256') != fingerprint(plan)):
        raise ValueError('Frozen library plan integrity check failed')
    if library_code_hashes() != plan['code_sha256']:
        raise ValueError('Source changed after freezing')
    calendar = SessionCalendar(plan['calendar']['path'])
    if calendar.sha256 != plan['calendar']['sha256']:
        raise ValueError('Calendar changed after freezing')
    config = plan['config']
    validate_library_config(config, calendar)
    if plan['schedule'] != weekly_schedule(config, calendar) or plan['feature_groups'] != FEATURE_GROUPS:
        raise ValueError('Schedule/features disagree with frozen definition')
    cases = []
    with threadpool_limits(limits=1):
        for case in plan['cases']:
            reader = PreparedDatasetReader(case['dataset']['path'], calendar=calendar,
                allow_partial=config['allow_partial'], include_degraded=config['include_degraded'])
            if reader.provenance != case['dataset']:
                raise ValueError('Prepared dataset changed after freezing')
            versions = []
            for schedule in plan['schedule']:
                models = []
                for count in config['history_session_counts']:
                    for model in fit_history(reader, schedule['training_sessions'][str(count)], config, schedule['available_at_utc']):
                        models.append(model | dict(model_id=f'{model["family"]}-{model["feature_group"]}-h{count}',
                                                   history_session_count=count))
                version = schedule | dict(models=models, feature_groups=FEATURE_GROUPS,
                    prediction_horizon_ms=config['prediction_horizon_ms'], purge_ms=config['purge_ms'],
                    interval_ms=reader.report['interval_ms'], max_age_ms=case['max_age_ms'])
                version['version_sha256'] = fingerprint(version)
                versions.append(version)
            days = plan['schedule'][0]['training_sessions'][str(max(config['history_session_counts']))]
            try:
                _, _, reference = fit_streaming_ridge(reader, days, config['prediction_horizon_ms'], config['purge_ms'])
                reference['status'] = 'fitted'
            except ValueError as error:
                # 只保留已知的零/不足样本状态，数据或公式错误不能被吞成缺样本。
                if str(error) != 'Not enough purged samples across training sessions':
                    raise
                reference = dict(name='Ridge_Linear', status='insufficient_samples', error=str(error))
            cases.append(dict(max_age_ms=case['max_age_ms'], versions=versions,
                              fixed_reference=reference))
            print(f'完成 {case["max_age_ms"]}ms：{len(versions)} 个版本，每版 {len(models)} 个候选', flush=True)
    if library_code_hashes() != plan['code_sha256']:
        raise ValueError('Source changed during build')
    result = dict(schema_version=1, result_kind='weekly_light_library_development', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, selected_age_ms=None,
        environment=dict(python=platform.python_version(), **{n: importlib.metadata.version(n) for n in
                         ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'threadpoolctl')}),
        limits=['Table 2 特征和历史窗口是工程具体化；盘口深度不是成交量。',
                '启动日可为周中，之后按 trade date 每周首个 session 开盘更新；不模拟训练耗时。',
                '树使用固定种子的有界均匀抽样，Ridge 使用全量合法历史样本。',
                '全部日期为开发；未来版本不得提前使用，已结束的旧诊断日可成为后续历史。',
                '建库输出本身不含交易收益；需另行运行课程 OE-UCB 链路。'])
    return result
