"""唯一课程实验链路：新数据准备→周模型库→历史 OE 校准→冻结奖励→UCB。

保留论文 §3.2、Eq.(2)–(6) 的接口，复用原时间引擎和完整期间反馈。
有限候选专家/最大间隔、短历史、周统计重启和执行成本均是项目假设。
此入口生成新身份与新输出；随仓库报告与图表的数值仍来自原冻结证据。
"""
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.calibration import (blocked_result, execution_gate, policy_specs,
    pooled_policy_statistics, qualify_calibration, replay_candidates)
from src.config import BASE_DIR, INSTRUMENT_CONFIG
from src.data_catalog import select_daily_source
from src.friction_reward import FRICTION_EXPERIMENT, FrozenFrictionOEReward
from src.model_selector import EnsembleSelector, FlatSelector, SingleModelSelector
from src.period_ucb import PeriodOESelector
from src.session_experiment import (code_hashes, fingerprint, freeze_protocol,
    publish_bundle, read_day, summarize_days, validate_protocol)
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import SessionCalendar, prepare_snapshot_dataset, sha256_file
from src.sum_only_reward import FrozenSumOnlyReward, learn_sum_only
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA, write_timed_snapshots
from src.weekly_library import build_library, freeze_library, predict_library, validate_library_config

STRATEGIES = ('cash', 'Fixed-Ridge-price-h1', 'Mean-Ensemble', 'OE-equal-UCB',
              'OE-online_library-UCB')


def load_config(path):
    """运行前检查唯一协议；所有年龄保留，禁止通过 CLI 按收益换参数。"""
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    calendar = SessionCalendar(BASE_DIR / config['calendar'])
    if (config['schema_version'] != 1 or config['purpose'] != 'course_core_OE_UCB_development'
            or config['expert_scope'] != 'online_library' or config['weight_constraint'] != 'sum_only'
            or config['fixed_model_id'] != 'Ridge-price-h1'
            or config['library']['age_candidates_ms'] != [500, 1000, 2000]
            or config['interval_ms'] != 500):
        raise ValueError('Unsupported course protocol or missing declared comparison')
    days = validate_protocol(config['protocol'], calendar, config['interval_ms'])
    library = config['library']
    validate_library_config(library, calendar)
    protocol = config['protocol']
    if (library['bootstrap_session'] != protocol['sessions']['calibration'][0]
            or library['evaluation_sessions'] != days[len(protocol['sessions']['train']):]
            or library['prediction_horizon_ms'] != protocol['prediction_horizon_ms']
            or library['purge_ms'] != max(protocol['reward_horizons_ms'])
            or any(library[k] != protocol[k] for k in ('allow_partial', 'include_degraded'))):
        raise ValueError('Library and execution time/quality protocols differ')
    if (type(config['selection_period_ms']) is not int or config['selection_period_ms'] <= 0
            or config['selection_period_ms'] % config['interval_ms']
            or not np.isfinite(config['exploration_c_price']) or config['exploration_c_price'] < 0):
        raise ValueError('Invalid fixed-period UCB parameters')
    calibration = config['calibration']
    if (calibration['threshold_multipliers'] != [1, 2, 4]
            or type(calibration['minimum_matured_orders']) is not int
            or calibration['minimum_matured_orders'] <= 0
            or any(not np.isfinite(calibration[k]) or calibration[k] <= 0
                   for k in ('solver_tolerance', 'tie_tolerance_usd', 'tie_tolerance_reward'))):
        raise ValueError('Invalid calibration computability/tie rules')
    return config, calendar


def prepare_course(config_path, directory):
    """逐 UTC 日期、逐年龄分批准备；只读取协议所需日期，不把季度称作回测。

    每个 session 需要相邻 UTC 分区，先核对全部来源和 degraded 状态再写。
    输出必须为新目录，原始 data/ 与已有证据均保留；缺格不前填为连续 tick。
    """
    config, calendar = load_config(config_path)
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite prepared directory: {directory}')
    days = [d for phase in ('train', 'calibration', 'validation', 'test')
            for d in config['protocol']['sessions'][phase]]
    dates = sorted({t.date().isoformat() for day in days for t in pd.date_range(
        calendar.sessions[day]['open'].normalize(), calendar.sessions[day]['close'].normalize(), freq='D')})
    index_sha = sha256_file(BASE_DIR / config['data_index'])
    sources = [select_daily_source(BASE_DIR / config['data_index'], day,
               config['protocol']['include_degraded']) for day in dates]
    config_sha = sha256_file(config_path)
    directory.mkdir(parents=True)
    datasets = []
    for age in config['library']['age_candidates_ms']:
        paths = []
        for source in sources:
            path = directory / 'snapshots' / f'age-{age}' / (source['file_date_utc'] + '.parquet')
            path.parent.mkdir(parents=True, exist_ok=True)
            write_timed_snapshots(source['source_file'], path, interval_ms=config['interval_ms'],
                max_age_ms=age, source_date=source['file_date_utc'], condition=source['condition'])
            paths.append(path)
            print(f"已准备 {source['file_date_utc']} / {age}ms", flush=True)
        dataset = directory / f'age-{age}'
        prepare_snapshot_dataset(paths, dataset, calendar_path=calendar.path,
            horizons_ms=config['protocol']['reward_horizons_ms'],
            allow_partial=config['protocol']['allow_partial'],
            include_degraded=config['protocol']['include_degraded'])
        datasets.append(str(dataset))
    if sha256_file(config_path) != config_sha or sha256_file(BASE_DIR / config['data_index']) != index_sha:
        raise ValueError('Configuration or data index changed during preparation')
    binding = dict(config_sha256=config_sha, datasets=datasets, source_dates_utc=dates,
                   data_index_sha256=index_sha)
    (directory / 'inputs.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2)+'\n')
    return binding


def replay_course_day(reader, day, library_case, config, fit, gate, *, detail=False,
                      friction_experiment=False):
    """原五项或显式增量对照共享 N×12 当时预测、执行条件与完整库可用性。

    固定候选身份提前声明，参数仍按周更新。学习奖励只取历史校准；阻断
    保留 None。每笔成交等全部尺度成熟，整期间所有订单完整才更新 UCB。
    friction_experiment=True 只对照原学习OE与扣实际单边摩擦的学习OE；
    两者使用同一份原始OE校准权重和资格门控，不再校准或增加其他方法。
    """
    frame = read_day(reader, day)
    matrix, versions = predict_library(library_case['versions'], frame, reader.report['interval_ms'])
    models = [SimpleNamespace(name=m['model_id']) for m in library_case['versions'][0]['models']]
    ids = [m.name for m in models]
    quotes = frame[SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
    quotes['feature_valid'] &= np.isfinite(matrix).all(axis=1)
    horizons = config['protocol']['reward_horizons_ms']
    results = []
    strategies = (FRICTION_EXPERIMENT['control'], FRICTION_EXPERIMENT['treatment']) if friction_experiment else STRATEGIES
    for name in strategies:
        calibrated = name in (STRATEGIES[-1], FRICTION_EXPERIMENT['treatment'])
        if calibrated and not gate['usable_for_execution']:
            results.append(blocked_result(name, gate))
            continue
        weights = fit['weights'] if calibrated else [1/len(horizons)] * len(horizons)
        reward_class = FrozenFrictionOEReward if name == FRICTION_EXPERIMENT['treatment'] else FrozenSumOnlyReward
        reward = reward_class(horizons, weights)
        if name == 'cash':
            selector = FlatSelector(name, models)
        elif name == STRATEGIES[1]:
            selector = SingleModelSelector(name, models, fixed_idx=ids.index(config['fixed_model_id']))
        elif name == 'Mean-Ensemble':
            selector = EnsembleSelector(name, models)
        else:
            selector = PeriodOESelector(name, models, period_ms=config['selection_period_ms'],
                                        c=config['exploration_c_price'])
        selector.reward_type = 'OE'
        protocol = config['protocol']
        engine = TimeExecutionEngine(reward, interval_ms=reader.report['interval_ms'], calendar=reader.calendar,
            latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'], force_replay_end=False)
        row = engine.run_backtest(selector, quotes, matrix, detail=detail, model_version_ids=versions)
        row.update(run_status='executed', reward_source='frozen_calibration_sum_only'
                   if calibrated else 'declared_equal_weights')
        results.append(row)
    return dict(session_id=day, results=results)


def run_course(config_path, prepared_directory, output_directory, *, friction_experiment=False):
    """冻结当前源码/输入后运行校准和评价；逐日保存，完成后再发布总结果。

    所有日期已用于开发，权重冻结不等于未触碰测试。新课程代码没有重跑
    ME/ARS、专家来源或约束搜索；历史27项完整结果留在原快照。
    显式增量模式在单独输出目录生成新回放，两臂只在奖励扣费项上有差异。
    """
    config, calendar = load_config(config_path)
    if type(friction_experiment) is not bool:
        raise ValueError('Incremental experiment selection must be boolean')
    strategies = (FRICTION_EXPERIMENT['control'], FRICTION_EXPERIMENT['treatment']) if friction_experiment else STRATEGIES
    inputs = json.loads((Path(prepared_directory) / 'inputs.json').read_text())
    if inputs['config_sha256'] != sha256_file(config_path):
        raise ValueError('Prepared data and course configuration differ')
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite result directory: {output}')
    plan = freeze_library(config_path, inputs['datasets'], calendar_path=calendar.path)
    publish_bundle(plan, '# 新课程模型库冻结计划\n', output, 'library_plan.json')
    library = build_library(output / 'library_plan.json')
    (output / 'library.json').write_text(json.dumps(library, ensure_ascii=False, allow_nan=False))
    # 建库已按实际质量元数据排序三个年龄，执行绑定沿用同一顺序，避免输入列表错位。
    bindings = [freeze_protocol(config_path, row['dataset']['path'], calendar_path=calendar.path)
                for row in plan['cases']]
    execution_plan = dict(config=config, config_sha256=sha256_file(config_path),
        cases=bindings, data_index_sha256=inputs['data_index_sha256'],
        library_sha256=sha256_file(output / 'library.json'), code_sha256=code_hashes(),
        account_policy='independent_daily_accounts_no_compounding', holdout_claim='none_development_dates')
    if friction_experiment:
        # 在任何评价收益产生前冻结单项扩展；仍只在历史原始 OE 上校准一次。
        execution_plan['incremental_experiment'] = FRICTION_EXPERIMENT.copy()
        execution_plan['prepared_input_manifest'] = dict(
            sha256=sha256_file(Path(prepared_directory)/'inputs.json'),
            preparation_cache=inputs.get('preparation_cache'))
    execution_plan['plan_sha256'] = fingerprint(execution_plan)
    (output / 'execution_plan.json').write_text(json.dumps(execution_plan, ensure_ascii=False, indent=2))
    cases = []
    with threadpool_limits(limits=1):
        for library_case, binding in zip(library['age_cases'], bindings):
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=config['protocol']['allow_partial'], include_degraded=config['protocol']['include_degraded'])
            if reader.provenance != binding['dataset']:
                raise ValueError('Prepared data changed after freezing')
            age = library_case['max_age_ms']
            policies = [p | dict(source='fixed_reference', model_id=None) for p in
                        policy_specs(config['calibration'], config['protocol']['threshold'])]
            policies += [dict(policy_id='Library-'+m['model_id'], source='weekly_library',
                model_id=m['model_id'], cash=False, threshold=config['protocol']['threshold'])
                for m in library_case['versions'][0]['models']]
            daily, audits = replay_candidates(reader, config['protocol']['sessions']['calibration'],
                                               library_case, policies, config['protocol'])
            stats = pooled_policy_statistics(daily, policies, config['protocol']['reward_horizons_ms'])
            qualification = qualify_calibration(stats, config['protocol']['reward_horizons_ms'],
                config['calibration'], expert_policy_ids=[p['policy_id'] for p in policies if p['source']=='weekly_library'])
            fit = learn_sum_only(stats, config['protocol']['reward_horizons_ms'], config['calibration'], qualification)
            gate = execution_gate(fit, policies)
            phases = {}
            for phase in ('validation', 'test'):
                rows = []
                for day in config['protocol']['sessions'][phase]:
                    row = replay_course_day(reader, day, library_case, config, fit, gate,
                                            friction_experiment=friction_experiment)
                    publish_bundle(row, '# 新课程独立日结果\n', output / f'age-{age}' / day, 'day.json')
                    rows.append(row)
                    print(f'已保存 {age}ms / {day} {len(strategies)}项状态', flush=True)
                totals = [blocked_result(name, gate) if name in (STRATEGIES[-1], FRICTION_EXPERIMENT['treatment'])
                          and not gate['usable_for_execution'] else summarize_days(rows, name) | dict(run_status='executed')
                          for name in strategies]
                phases[phase] = dict(daily=rows, statistics=totals)
            cases.append(dict(max_age_ms=age, calibration=dict(daily=daily, audits=audits, statistics=stats),
                              reward_fit=fit, execution_gate=gate, phases=phases))
    if (code_hashes() != execution_plan['code_sha256']
            or sha256_file(config_path) != execution_plan['config_sha256']
            or sha256_file(output / 'library.json') != execution_plan['library_sha256']):
        raise ValueError('Source, configuration or model library changed during replay')
    packages = ('numpy','pandas','pyarrow','scikit-learn','scipy','threadpoolctl')
    if friction_experiment:
        packages += ('matplotlib',)  # 新增实验同时导出独立图，记录实际绘图库版本。
    return dict(schema_version=1, result_kind='course_friction_reward_increment' if friction_experiment
                else 'course_core_OE_UCB_development', plan=execution_plan,
        plan_sha256=execution_plan['plan_sha256'], age_cases=cases, selected_age_ms=None,
        instrument=INSTRUMENT_CONFIG['CME_ES'], environment=dict(python=platform.python_version(),
        **{n: importlib.metadata.version(n) for n in packages}))
