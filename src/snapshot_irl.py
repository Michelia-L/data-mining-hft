"""按真实时间成熟的 OE 特征接入有限策略 IRL，并冻结后续评估。

论文第 4 页 Eq.(2)–(4)/Algorithm 1：先取得专家策略，再迭代求线性奖励与
最优策略响应。此处复用有限策略最大间隔近似；原文没有公开生产参数优化器。
策略是冻结 Ridge 加预声明门槛或现金，不是按周模型库或 Algorithm 2/3。
"""
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from src.artifacts import pretty_json
from src.config import BASE_DIR, INSTRUMENT_CONFIG, SEED
from src.irl_reward import IRLRewardLearner
from src.model_selector import FlatSelector, SingleModelSelector
from src.session_experiment import (CODE_FILES, audit_horizons, fingerprint,
    fit_streaming_ridge, freeze_protocol, read_day, summarize_days, validate_protocol)
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader
from src.snapshot_dataset import DEFAULT_CALENDAR, SessionCalendar, sha256_file
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA


def irl_code_hashes():
    """额外绑定新入口与研究逻辑，保留既有基线源码不改写。"""
    return {name: sha256_file(BASE_DIR / name) for name in
            (*CODE_FILES, 'src/snapshot_irl.py', 'run_snapshot_irl.py')}


def validate_irl_config(config):
    """候选、约束与消歧规则都在收益评估前声明；运行阶段不能覆盖。"""
    required = {'schema_version', 'purpose', 'session_protocol_file', 'age_candidates_ms',
        'threshold_multipliers', 'weight_constraints', 'minimum_matured_orders',
        'solver_tolerance', 'solver_max_iter', 'tie_tolerance_usd', 'tie_tolerance_reward'}
    if (set(config) - (required | {'notes'}) or required - set(config)
            or config['schema_version'] != 1 or config['purpose'] != 'development_time_OE_IRL'):
        raise ValueError('Unsupported time IRL configuration')
    ages = config['age_candidates_ms']
    if (not isinstance(ages, list) or not ages or ages != sorted(set(ages))
            or any(type(a) is not int or a <= 0 for a in ages)):
        raise ValueError('Need ordered positive millisecond age candidates')
    multipliers = config['threshold_multipliers']
    if (not isinstance(multipliers, list) or not multipliers or multipliers != sorted(set(multipliers))
            or any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in multipliers)
            or 1 not in multipliers):
        raise ValueError('Threshold multipliers must retain the fixed baseline')
    constraints = config['weight_constraints']
    if (not isinstance(constraints, list) or not constraints or len(constraints) != len(set(constraints))
            or any(v not in ('simplex', 'signed_box') for v in constraints) or 'simplex' not in constraints):
        raise ValueError('Need simplex baseline and optional declared signed-box sensitivity')
    if any(type(config[k]) is not int or config[k] <= 0
           for k in ('minimum_matured_orders', 'solver_max_iter')):
        raise ValueError('Need positive minimum order count and iteration limit')
    for name in ('solver_tolerance', 'tie_tolerance_usd', 'tie_tolerance_reward'):
        if type(config[name]) not in (int, float) or not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError('Need positive finite numerical tolerances')


def policy_specs(config, threshold):
    """现金优先的预声明顺序用于消歧，门槛乘数不从测试段选优。"""
    return [dict(policy_id='cash', cash=True, threshold=threshold)] + [
        dict(policy_id=f'Ridge-threshold-x{m:g}', cash=False, threshold=threshold * m)
        for m in config['threshold_multipliers']]


def freeze_time_irl(config_path, directories, calendar_path=DEFAULT_CALENDAR):
    """绑定同一协议下全部年龄数据；相邻源分区和质量规则沿用基线冻结检查。"""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_irl_config(config)
    protocol_path = (config_path.parent / config['session_protocol_file']).resolve()
    bindings = [freeze_protocol(protocol_path, directory, calendar_path=calendar_path) for directory in directories]
    if len(bindings) != len(config['age_candidates_ms']):
        raise ValueError('Need every declared age dataset exactly once')
    cases, signatures = [], []
    for binding in bindings:
        quality = json.loads((Path(binding['dataset']['path']) / 'quality.json').read_text(encoding='utf-8'))
        age = quality['inputs'][0]['max_age_ms']
        cases.append(dict(max_age_ms=age, session_binding=binding))
        signatures.append([(row['metadata']['source_date_utc'], row['metadata']['source_sha256'],
                            row['metadata']['source_condition'], row['metadata']['partial_prefix'])
                           for row in quality['inputs']])
    cases.sort(key=lambda c: c['max_age_ms'])
    if [c['max_age_ms'] for c in cases] != config['age_candidates_ms']:
        raise ValueError('Prepared age values do not match declared candidates')
    if (any(s != signatures[0] for s in signatures) or
            any(b['protocol'] != bindings[0]['protocol'] or b['interval_ms'] != bindings[0]['interval_ms']
                for b in bindings)):
        raise ValueError('Age datasets must share raw-source provenance, sessions and grid')
    interval = bindings[0]['interval_ms']
    if interval not in config['age_candidates_ms']:
        raise ValueError('Need one-grid-age strict baseline')
    protocol = bindings[0]['protocol']
    if protocol['threshold'] <= 0:
        raise ValueError('Positive baseline threshold required for distinct multiplier policies')
    plan = dict(schema_version=1, plan_kind='frozen_snapshot_time_IRL', config=config,
        config_source=dict(path=str(config_path), sha256=sha256_file(config_path)), cases=cases,
        policies=policy_specs(config, protocol['threshold']), code_sha256=irl_code_hashes(),
        stage_roles=dict(train='fit_frozen_Ridge_only', calibration='expert_and_OE_reward_fit',
                         validation='frozen_diagnostics_no_selection', test='frozen_development_evaluation'),
        expert_rule='maximum_settled_calibration_net_USD',
        tie_rule='first_declared_policy_cash_first_with_recorded_ties',
        age_selection='none_all_declared_candidates_retained')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def pooled_policy_statistics(daily, specs, horizons):
    """按成熟订单数量合并日均值，不能把每天均值等权平均或除以行情行数。

    Eq.(5) 的订单分母保留，但这里只评价校准段已成熟的可观测子集，不是所有
    发出订单，也不是论文固定选择期间。零成熟订单的 OE 为 None；现金仅在
    有限策略优化中约定零向量，不伪装成观测到零 OE。
    """
    output = []
    for spec in specs:
        summary = summarize_days(daily, spec['policy_id'])
        rows = [next(r for r in day['results'] if r['strategy'] == spec['policy_id']) for day in daily]
        count = summary['matured_order_count']
        sums = sum((np.asarray(r['order_feature_expectation']) * r['matured_order_count'] for r in rows),
                   np.zeros(len(horizons)))
        output.append(dict(policy_id=spec['policy_id'], cash=spec['cash'], threshold=spec['threshold'],
            **{k: v for k, v in summary.items() if k != 'strategy'},
            order_feature_expectation=(sums / count).tolist() if count else None,
            order_feature_expectation_defined=bool(count),
            reward_feature_source='matured_filled_orders_only',
            matured_fraction_of_fills=float(count / summary['total_fills']) if summary['total_fills'] else None))
    return output


def best_with_ties(rows, values, tolerance):
    """最大值附近全部并列都记录，按冻结列表顺序取首项，数值单位由调用方声明。"""
    if not rows:
        return None
    best = max(values)
    ties = [row['policy_id'] for row, value in zip(rows, values) if abs(value - best) <= tolerance]
    return dict(selected_policy_id=ties[0], tied_best_policy_ids=ties, best_score=float(best))


def learn_calibration_reward(statistics, horizons, config, constraint, *, expert_policy_ids=None):
    """只消费校准期 P×H 成熟订单特征，保留无数据、无信息和求解失败结果。

    专家先在所有已平仓候选中按净盈亏选择；若其奖励不可观测，不偷换成次优
    专家。现金是零向量约定；至少需要一个可观测非现金策略才允许拟合。
    求解收敛、专家可表示、正利润专家和权重唯一识别是不同概念。
    expert_policy_ids 仅限制专家来源，不删减用于求解的候选矩阵。默认 None
    保持旧的全候选规则；显式子集属于工程对照，并列仍按 statistics 的冻结
    顺序消歧。先选净利专家，再检查成熟资格，不能从可观测子集倒选专家。
    """
    if expert_policy_ids is not None:
        names = [r['policy_id'] for r in statistics]
        if (not expert_policy_ids or len(set(expert_policy_ids)) != len(expert_policy_ids)
                or not set(expert_policy_ids).issubset(names)):
            raise ValueError('Expert scope must be a nonempty unique subset of declared policies')
    settled = [r for r in statistics if r['pnl_aggregation_defined']]
    expert_rows = [r for r in settled if expert_policy_ids is None or r['policy_id'] in expert_policy_ids]
    expert = best_with_ties(expert_rows, [r['net_pnl_usd'] for r in expert_rows], config['tie_tolerance_usd'])
    eligible, excluded = [], []
    for row in statistics:
        reason = ('unsettled_account' if not row['pnl_aggregation_defined'] else
                  'insufficient_matured_orders' if not row['cash'] and
                  row['matured_order_count'] < config['minimum_matured_orders'] else None)
        if reason:
            excluded.append(dict(policy_id=row['policy_id'], reason=reason,
                                 matured_order_count=row['matured_order_count']))
        else:
            eligible.append(row)
    base = dict(weight_constraint=constraint, expert=expert, excluded_policies=excluded,
        eligible_policy_ids=[r['policy_id'] for r in eligible], weights=None, diagnostics=None,
        selected_policy=None, equal_weight_selected_policy=None,
        cash_zero_vector_is_optimization_convention=True,
        minimum_order_count_is_computability_not_statistical_sufficiency=True)
    if expert is None:
        return base | dict(status='blocked_no_settled_expert')
    winner = next(r for r in settled if r['policy_id'] == expert['selected_policy_id'])
    base['expert_is_cash'] = winner['cash']
    base['profitable_trading_expert'] = not winner['cash'] and winner['net_pnl_usd'] > config['tie_tolerance_usd']
    if not any(not r['cash'] for r in eligible):
        return base | dict(status='blocked_no_matured_trading_policy')
    if winner['policy_id'] not in base['eligible_policy_ids']:
        return base | dict(status='blocked_expert_reward_unobserved')
    mu = np.array([np.zeros(len(horizons)) if r['cash'] else r['order_feature_expectation'] for r in eligible])
    equal_scores = mu.mean(axis=1)
    base['equal_weight_selected_policy'] = best_with_ties(eligible, equal_scores, config['tie_tolerance_reward'])
    if np.max(np.abs(mu - mu[0])) <= config['solver_tolerance']:
        return base | dict(status='blocked_uninformative_expectations')
    reward = IRLRewardLearner(horizons=horizons, definition='paper_price_difference', weight_constraint=constraint)
    try:
        reward.fit_reward_weights(mu, base['eligible_policy_ids'].index(winner['policy_id']),
            policy_names=base['eligible_policy_ids'], tolerance=config['solver_tolerance'],
            max_iter=config['solver_max_iter'])
    except RuntimeError as error:
        return base | dict(status='solver_failed', error=str(error))
    base['weights'], base['diagnostics'] = reward.weights.tolist(), reward.diagnostics
    scores = mu @ reward.weights
    base['candidate_response'] = best_with_ties(eligible, scores, config['tie_tolerance_reward'])
    base['calibration_policy_scores'] = dict(zip(base['eligible_policy_ids'], map(float, scores)))
    status = ('solver_not_converged' if not reward.diagnostics['converged'] else
              'fitted_expert_not_representable' if not reward.diagnostics['expert_representable'] else 'fitted')
    # 不可表示或未收敛时保留权重和反例供诊断，不把它发布成可用冻结策略。
    base['selected_policy'] = base['candidate_response'] if status == 'fitted' else None
    return base | dict(status=status, reward_uniquely_identified=False)


def replay_stage(reader, days, scaler, model, specs, protocol, *, detail=False):
    """候选策略冻结，行情预热/标签失效行仍回放，OE 等待最长尺度后才累计。

    静态策略不依赖反馈权重，因此用等权回放一次即可保留完全相同的成交；后续
    学习权重对成熟向量线性评分。这个等价性不能推广到动态 UCB/ARS。
    """
    horizons = protocol['reward_horizons_ms']
    reward = IRLRewardLearner(horizons=horizons, definition='paper_price_difference')
    daily, audits = [], []
    for day in days:
        frame = read_day(reader, day)
        audits.append(audit_horizons(frame, reader.calendar, horizons))
        valid = frame.feature_valid.to_numpy()
        predictions = np.full((len(frame), 1), np.nan)
        if valid.any():
            predictions[valid, 0] = model.predict(scaler.transform(frame.loc[valid, FEATURE_COLUMNS].to_numpy(float)))
        quotes = frame[SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'mid_price', 'feature_valid']]
        results = []
        for spec in specs:
            selector = (FlatSelector if spec['cash'] else SingleModelSelector)(spec['policy_id'], [model])
            selector.reward_type = 'OE'
            engine = TimeExecutionEngine(reward, interval_ms=reader.report['interval_ms'], calendar=reader.calendar,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
                threshold=spec['threshold'], force_replay_end=False)
            results.append(engine.run_backtest(selector, quotes, predictions, detail=detail))
        daily.append(dict(session_id=day, results=results))
    return daily, audits


def evaluate_frozen_weights(statistics, fits):
    """先固定校准权重，再对验证/测试期可观测成熟订单均值评分，不重选策略。

    线性 Eq.(3) 下 mean(w·u)=w·mean(u)。零成熟订单的实测评分为 None，
    包括现金；校准优化的现金零向量约定不会被写成验证期的实测 OE。
    """
    output = []
    for row in statistics:
        vector = row['order_feature_expectation']
        scores = {name: float(np.dot(fit['weights'], vector))
                  if fit['weights'] is not None and vector is not None else None for name, fit in fits.items()}
        output.append(row | dict(equal_weight_order_reward=float(np.mean(vector)) if vector is not None else None,
                                frozen_weight_order_rewards=scores))
    return output


def run_time_irl(plan_path, *, detail=False):
    """所有年龄都保留；训练→校准求解→冻结选择→验证/测试，不使用未来区间选优。"""
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if (plan.get('plan_kind') != 'frozen_snapshot_time_IRL' or plan.get('schema_version') != 1
            or plan.get('plan_sha256') != fingerprint(plan)):
        raise ValueError('Frozen IRL plan integrity check failed')
    if irl_code_hashes() != plan['code_sha256']:
        raise ValueError('Source changed after freezing')
    config = plan['config']
    validate_irl_config(config)
    cases = []
    with threadpool_limits(limits=1):
        for case in plan['cases']:
            binding = case['session_binding']
            if binding['plan_sha256'] != fingerprint(binding):
                raise ValueError('Session binding changed')
            calendar = SessionCalendar(binding['calendar']['path'])
            if calendar.sha256 != binding['calendar']['sha256']:
                raise ValueError('Calendar changed after freezing')
            protocol = binding['protocol']
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            validate_protocol(protocol, calendar, reader.report['interval_ms'])
            if reader.provenance != binding['dataset']:
                raise ValueError('Prepared dataset changed after freezing')
            scaler, model, fitted = fit_streaming_ridge(reader, protocol['sessions']['train'],
                protocol['prediction_horizon_ms'], max(protocol['reward_horizons_ms']))
            daily, audits = replay_stage(reader, protocol['sessions']['calibration'], scaler, model,
                                         plan['policies'], protocol, detail=detail)
            calibration = pooled_policy_statistics(daily, plan['policies'], protocol['reward_horizons_ms'])
            fits = {constraint: learn_calibration_reward(calibration, protocol['reward_horizons_ms'], config, constraint)
                    for constraint in config['weight_constraints']}
            phases = dict(calibration=dict(daily=daily, statistics=calibration, horizon_audits=audits))
            for phase in ('validation', 'test'):
                daily, audits = replay_stage(reader, protocol['sessions'][phase], scaler, model,
                                             plan['policies'], protocol, detail=detail)
                statistics = pooled_policy_statistics(daily, plan['policies'], protocol['reward_horizons_ms'])
                # 别名只引用校准期冻结的策略，不根据本阶段的净利或奖励重新选择。
                selections = {name: dict(selected_policy_id=fit['selected_policy']['selected_policy_id']
                    if fit['selected_policy'] is not None else None,
                    usable_fit=fit['status'] == 'fitted', fit_status=fit['status']) for name, fit in fits.items()}
                phases[phase] = dict(daily=daily, statistics=evaluate_frozen_weights(statistics, fits),
                    horizon_audits=audits, calibration_frozen_selections=selections)
            cases.append(dict(max_age_ms=case['max_age_ms'], model=fitted, reward_fits=fits, phases=phases))
            print(f"完成 {case['max_age_ms']}ms：" + ', '.join(f'{name}={fit["status"]}' for name, fit in fits.items()), flush=True)
    if irl_code_hashes() != plan['code_sha256']:
        raise ValueError('Source changed during experiment')
    return dict(schema_version=1, result_kind='snapshot_time_OE_IRL', plan=plan, plan_sha256=plan['plan_sha256'],
        seed=SEED, instrument=INSTRUMENT_CONFIG['CME_ES'], age_cases=cases, selected_age_ms=None,
        environment=dict(python=platform.python_version(), **{name: importlib.metadata.version(name) for name in
            ('numpy', 'pandas', 'pyarrow', 'scikit-learn', 'scipy', 'threadpoolctl')}),
        limits=['Algorithm 1 使用有限门槛策略最大间隔近似，原文生产优化器未公开。',
                'OE 仅覆盖成熟订单子集，零订单均值未定义；现金零向量仅是优化约定。',
                '静态成交与权重无关；学习权重线性重评，收益变化只可能来自冻结策略选择。',
                '求解收敛不等于专家可表示、存在正利润专家或奖励唯一识别。',
                '尚无 ME IRL、按周模型库或论文固定期间 UCB/ARS；全部日期用于开发。'])


def render_time_irl(result):
    """从同一份 JSON 展示所有年龄、求解失败与负收益，不把现金标成零订单实测奖励。"""
    lines = ['# 时间尺度 OE IRL 开发验证', '', f"计划：`{result['plan_sha256']}`；全部年龄保留。", '',
        '| 最大年龄 ms | 权约束 | 状态 | 专家 | 正利润交易专家 | 冻结策略 | 权重 |',
        '| ---: | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for name, fit in case['reward_fits'].items():
            lines.append(f"| {case['max_age_ms']} | {name} | {fit['status']} | "
                f"{fit['expert']['selected_policy_id'] if fit['expert'] else '未定义'} | "
                f"{fit.get('profitable_trading_expert', False)} | "
                f"{fit['selected_policy']['selected_policy_id'] if fit['selected_policy'] else '未定义'} | {fit['weights']} |")
    lines += ['', '## 全候选订单覆盖与成本', '',
        '| 年龄 ms | 阶段 | 策略 | 成交 | 成熟 OE | 毛利 USD | 成本 USD | 净利 USD |',
        '| ---: | --- | --- | ---: | ---: | --- | ---: | --- |']
    for case in result['age_cases']:
        for phase, data in case['phases'].items():
            for row in data['statistics']:
                lines.append(f"| {case['max_age_ms']} | {phase} | {row['policy_id']} | {row['total_fills']} | "
                    f"{row['matured_order_count']} | {row['gross_pnl_usd']} | {row['friction_usd']:.4f} | {row['net_pnl_usd']} |")
    lines += ['', '## 冻结权重的实测 OE 评分', '',
        '分母为完整成熟订单数；无成熟订单（包括现金）显示未定义。不可表示或未收敛的权重仅供诊断。', '',
        '| 年龄 ms | 阶段 | 策略 | 等权 OE | simplex OE | signed_box OE |',
        '| ---: | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for phase in ('validation', 'test'):
            for row in case['phases'][phase]['statistics']:
                values = [row['equal_weight_order_reward']] + [
                    row['frozen_weight_order_rewards'].get(name) for name in ('simplex', 'signed_box')]
                scores = ['未定义' if value is None else f'{value:.6g}' for value in values]
                lines.append(f"| {case['max_age_ms']} | {phase} | {row['policy_id']} | " + ' | '.join(scores) + ' |')
    lines += ['', '## 权重与可表示性诊断', '']
    for case in result['age_cases']:
        lines += [f"### {case['max_age_ms']}ms", '', '```json', pretty_json(case['reward_fits']), '```', '']
    lines += ['## 复现限制', ''] + ['- ' + note for note in result['limits']]
    return '\n'.join(lines) + '\n'
