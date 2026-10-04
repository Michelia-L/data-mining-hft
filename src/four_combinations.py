"""收尾2/4的最小真实时间四组合证据，不扩大数据或重选开发参数。

OE 的24项已完成结果作为有独立身份的历史证据继承；新算同日期、同模型、
同费用的ME等权/学习UCB及双ARS，并补均值集成。原始OE文件/旧证据保留。
所有年龄和专家来源都输出；不可表示、无信号或专家不在库时保持阻断。
"""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR
from src.expert_scope import SCOPES
from src.history_ars import HistoryARSSelector
from src.irl_reward import IRLRewardLearner
from src.learned_oe_experiment import blocked_result
from src.library_oe import load_library
from src.model_selector import EnsembleSelector
from src.multiweek_dynamic_oe import (ReplayCheckpoints, audit_dynamic_result, code_hashes as oe_hashes,
    environment_versions, fit_groups, summarize_phase)
from src.multiweek_readiness import prediction_day_audit, training_visibility_audit
from src.period_ucb import PeriodOESelector
from src.session_experiment import fingerprint, read_day
from src.snapshot_backtest import FEATURE_COLUMNS, PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.sum_only_reward import FrozenSumOnlyReward
from src.time_execution import TimeExecutionEngine
from src.time_me import PredictionHistoryBacktester, calibration_me, replay_me
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library, predict_model


ME_CONTROLS = ['Period-ME-UCB', 'History-ME-ARS-recent', 'History-ME-ARS-matured']
ENSEMBLE = 'Mean-Ensemble'


def me_names(scope):
    """学习组按来源显式命名，ARS的两个窗口属于同一组合的工程对照。"""
    if scope not in SCOPES:
        raise ValueError('Unknown expert scope')
    prefix = 'Learned-ME-' if scope == 'all_candidates' else 'LibraryExpert-ME-'
    return [prefix+'UCB', prefix+'ARS-recent', prefix+'ARS-matured']


def code_hashes():
    """旧OE依赖逐字节绑定，新代码另外绑定，旧库不因新增模块被伪装成重训。"""
    return oe_hashes() | {p: sha256_file(BASE_DIR/p) for p in
        ('src/time_me.py', 'src/four_combinations.py', 'run_four_combinations.py')}


def validate_config(config):
    """限定收尾范围：三年龄由原协议决定，来源/窗口/学习口径不允许选择成功项。"""
    expected = dict(schema_version=1, purpose='development_four_combinations',
        expert_scopes=list(SCOPES), active_constraint='sum_only',
        me_sample_policy='existing_threshold_nonzero_signed_prediction',
        me_online_aggregation='all_signals_in_complete_period_or_window',
        fixed_model_rule='first_declared_model_without_return_selection')
    if {k: v for k, v in config.items() if k != 'notes'} != expected:
        raise ValueError('Predeclared minimal four-combination protocol required')


def load_oe(path):
    """校验完整历史OE文件的计划、现行未改动源码、策略及独立动态审计。

    只有相同执行/训练代码才能沿用这些美元账本作为可比参照；改变旧算法必须
    另算旧对照，不能仅重写旧哈希。新ME函数不输入后段收益来拟合权重。
    """
    result = json.loads(Path(path).read_text()); plan = result.get('plan', {})
    if (result.get('result_kind') != 'multiweek_dynamic_OE_development'
            or result.get('plan_sha256') != plan.get('plan_sha256')
            or plan.get('plan_sha256') != fingerprint(plan)
            or plan.get('code_sha256') != oe_hashes()
            or result.get('environment') != environment_versions()):
        raise ValueError('Archived OE plan/source/environment integrity differs')
    base = plan['base_ars_plan']['base_period_plan']
    oe = plan['base_oe_plan']
    for value in (plan['base_ars_plan'], base, oe):
        if value['plan_sha256'] != fingerprint(value):
            raise ValueError('Archived OE nested plan differs')
    library, calendar = load_library(base['library_source']['path'])
    if (sha256_file(base['library_source']['path']) != base['library_source']['sha256']
            or [c['max_age_ms'] for c in result['age_cases']] != [c['max_age_ms'] for c in base['cases']]):
        raise ValueError('Archived OE library or age binding differs')
    for case, frozen in zip(result['age_cases'], base['cases']):
        protocol = frozen['session_binding']['protocol']
        libcase = next(c for c in library['age_cases'] if c['max_age_ms'] == case['max_age_ms'])
        for phase in ('validation', 'test'):
            daily = case['phases'][phase]['daily']
            if [d['session_id'] for d in daily] != protocol['sessions'][phase]:
                raise ValueError('Archived OE dates differ')
            for day in daily:
                if [r['strategy'] for r in day['results']] != plan['strategies']:
                    raise ValueError('Archived OE lost declared strategies')
                audits = {r['strategy']: audit_dynamic_result(r, libcase['versions'],
                    max(protocol['reward_horizons_ms']),
                    case['calibration']['scopes'][r['expert_scope']]['active_reward_weights']
                    if 'expert_scope' in r else None) for r in day['results']}
                if audits != day['dynamic_audits']:
                    raise ValueError('Archived OE dynamic audit differs')
            if summarize_phase(daily, plan['strategies']) != case['phases'][phase]['statistics']:
                raise ValueError('Archived OE statistics differ from daily ledger')
    return result, library, calendar


def freeze_four(config_path, oe_path):
    """先保存ME假设、所有日期/模型和历史OE身份，再执行新ME校准或收益。"""
    path = Path(config_path).resolve(); source = Path(oe_path).resolve()
    config = json.loads(path.read_text()); validate_config(config)
    digest = sha256_file(source); old, _, _ = load_oe(source)
    if digest != sha256_file(source):
        raise ValueError('Archived OE changed during freeze')
    base = old['plan']['base_ars_plan']['base_period_plan']
    strategies = old['plan']['strategies'] + [ENSEMBLE] + ME_CONTROLS + [n for s in SCOPES for n in me_names(s)]
    plan = dict(schema_version=1, plan_kind='frozen_four_combinations', config=config,
        config_source=dict(path=str(path), sha256=sha256_file(path)),
        archived_oe_source=dict(path=str(source), sha256=digest, plan_sha256=old['plan_sha256']),
        base_oe_dynamic_plan=old['plan'], strategies=strategies, code_sha256=code_hashes(),
        fixed_reference_model_id=base['model_ids'][0],
        stage_roles=dict(calibration='new_ME_only_inherited_static_USD_then_freeze',
                         validation='development_four_combinations', test='development_no_retuning'),
        holdout_claim='none_all_dates_already_development', selected_age_ms=None, selected_expert_scope=None,
        inherited_OE_role='historical_result_same_unmodified_dependencies_not_new_replay',
        account_policy='independent_daily_accounts_and_feedback')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def me_fit_groups(statistics, horizons, config, policies):
    """将ME均值接入既有有限求解器；兼容字段只在局部使用，不冒称订单。

    原求解器要求 order_feature_expectation/count 字段；这里映射同形的P×H
    信号均值和成熟信号数。收益专家仍来自校准完整美元账本，保持相同专家
    来源规则。最低信号数沿用旧计算资格阈值，只代表可计算，不代表统计充分。
    """
    adapted = [row | dict(matured_order_count=row['matured_signal_count'],
        order_feature_expectation=row['prediction_feature_expectation'],
        order_feature_expectation_defined=row['prediction_feature_expectation'] is not None) for row in statistics]
    groups = fit_groups(adapted, horizons, config, policies)
    def clarify(value):
        """删除输出中OE专属措辞，保留求解数值和专家身份，不改变门控决定。"""
        if isinstance(value, dict):
            return {k.replace('minimum_order_count', 'minimum_signal_count')
                     .replace('matured_order_count', 'matured_signal_count'): clarify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clarify(v) for v in value]
        if isinstance(value, str):
            return (value.replace('calibration_OE', 'calibration_ME')
                    .replace('matured_trading_policy', 'matured_signal_policy')
                    .replace('insufficient_matured_orders', 'insufficient_matured_signals'))
        return value
    return clarify(groups)


def calibration_day(reader, day, libcase, protocol, policies, inherited):
    """一日只生成一次预测；完整静态美元账本引用旧OE，ME另外计算预测均值。"""
    frame = read_day(reader, day); interval = reader.report['interval_ms']
    matrix, _ = predict_library(libcase['versions'], frame, interval)
    valid = frame.feature_valid.to_numpy()
    reference = libcase['fixed_reference'] | dict(features=FEATURE_COLUMNS, family='Ridge')
    original = np.full(len(frame), np.nan)
    if reference['status'] == 'fitted' and valid.any():
        original[valid] = predict_model(reference, frame.loc[valid, FEATURE_COLUMNS])
    ids = [m['model_id'] for m in libcase['versions'][0]['models']]
    reward = IRLRewardLearner(horizons=protocol['reward_horizons_ms'], definition='paper_price_difference')
    rows = []
    if [r['strategy'] for r in inherited['results']] != [p['policy_id'] for p in policies]:
        raise ValueError('Inherited calibration candidate order differs')
    for spec, old in zip(policies, inherited['results']):
        values = (np.zeros(len(frame)) if spec['cash'] else original if spec['source'] == 'fixed_reference'
                  else matrix[:, ids.index(spec['model_id'])])
        me = calibration_me(frame, values, reader.calendar, reward, interval, spec['threshold'])
        rows.append(dict(policy_id=spec['policy_id'], cash=spec['cash'], threshold=spec['threshold'],
            pnl_aggregation_defined=old['terminal_position_liquidated'],
            net_pnl_usd=old['net_pnl_usd'], **me))
    return dict(session_id=day, statistics=rows)


def pool_me(daily, policies, horizons):
    """按成熟信号数合并校准，不等权合并每天均值；未平仓时总净利未定义。"""
    output = []
    for spec in policies:
        rows = [next(r for r in d['statistics'] if r['policy_id'] == spec['policy_id']) for d in daily]
        count = sum(r['matured_signal_count'] for r in rows)
        feature_sum = sum((np.asarray(r['prediction_feature_expectation'])*r['matured_signal_count']
            for r in rows if r['matured_signal_count']), np.zeros(len(horizons)))
        settled = all(r['pnl_aggregation_defined'] for r in rows)
        output.append(dict(policy_id=spec['policy_id'], cash=spec['cash'], threshold=spec['threshold'],
            pnl_aggregation_defined=settled, net_pnl_usd=sum(r['net_pnl_usd'] for r in rows) if settled else None,
            matured_signal_count=count, prediction_signal_count=sum(r['prediction_signal_count'] for r in rows),
            prediction_feature_expectation=(feature_sum/count).tolist() if count else None,
            reward_feature_source='matured_thresholded_predictions_calibration_subset'))
    return output


def replay_additions(reader, day, libcase, protocol, plan, groups, *, detail=False):
    """均值集成、ME等权控制和两来源学习组共10项；失败返回null而非现金零。

    同一N×K预测矩阵服务全部动态策略及集成；每策略账户/ME队列完全独立。
    """
    oldplan = plan['base_oe_dynamic_plan']; ars = oldplan['base_ars_plan']; base = ars['base_period_plan']
    frame = read_day(reader, day); interval = reader.report['interval_ms']
    matrix, versions = predict_library(libcase['versions'], frame, interval)
    models = [SimpleNamespace(name=m) for m in base['model_ids']]
    equal = np.full(len(protocol['reward_horizons_ms']), 1./len(protocol['reward_horizons_ms'])).tolist()
    jobs = [(n, equal, None) for n in ME_CONTROLS]
    for scope in SCOPES:
        jobs += [(n, groups[scope]['active_reward_weights'], scope) for n in me_names(scope)]
    reward = FrozenSumOnlyReward(protocol['reward_horizons_ms'], equal)
    quotes = frame[SNAPSHOT_SCHEMA.names+['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
    valid = frame.feature_valid.to_numpy(); quotes['feature_valid'] = valid & np.isfinite(matrix).all(axis=1)
    ensemble = EnsembleSelector(ENSEMBLE, models); ensemble.reward_type = 'OE'
    engine = TimeExecutionEngine(reward, interval_ms=interval, calendar=reader.calendar,
        latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
        threshold=protocol['threshold'], force_replay_end=False)
    rows = [engine.run_backtest(ensemble, quotes, matrix, detail=detail, model_version_ids=versions)]
    rows[0].update(run_status='executed', reference_policy='equal_mean_predictions_not_calibration_expert')
    for name, weights, scope in jobs:
        if weights is None:
            rows.append(blocked_result(name, groups[scope]['execution_gates']['sum_only']) |
                        dict(expert_scope=scope, reward_type='ME')); continue
        reward = FrozenSumOnlyReward(protocol['reward_horizons_ms'], weights)
        if name.endswith('UCB'):
            selector = PeriodOESelector(name, models, period_ms=base['config']['selection_period_ms'],
                c=base['config']['exploration_c_price'])
        else:
            alignment = name.rsplit('-', 1)[1]
            history = PredictionHistoryBacktester(frame, libcase['versions'], reader.calendar, reward,
                interval_ms=interval, latency_ms=protocol['latency_ms'],
                holding_review_ms=protocol['holding_review_ms'], threshold=protocol['threshold'])
            selector = HistoryARSSelector(name, models, history, period_ms=base['config']['selection_period_ms'],
                window_ms=ars['config']['history_window_ms'], alignment=alignment,
                tie_tolerance=ars['config']['tie_tolerance_reward'])
        row = replay_me(selector, frame, matrix, versions, reward, reader.calendar, interval_ms=interval,
            latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'], detail=detail)
        row.update(run_status='executed', reward_source='frozen_calibration_sum_only' if scope else 'equal_weight_control',
            current_prediction_unavailable_rows=int((valid & ~quotes.feature_valid.to_numpy()).sum()))
        if scope:
            row['expert_scope'] = scope
        rows.append(row)
    return rows


def audit_me(result, versions, longest_ms, weights=None):
    """独立核对信号分母、价格点内积、成熟/版本及ARS观察边界，费用另核对。"""
    if result.get('run_status') == 'blocked':
        if result['net_pnl_usd'] is not None or result['total_fills'] is not None:
            raise ValueError('Blocked ME must not have observed profit')
        return dict(run_status='blocked', selections=0, mature_periods=0, history_windows=0, scored_windows=0,
                    selected_version_ids=[])
    if 'period_feedback' not in result:
        return dict(run_status='static_control')
    if weights is not None and result['reward_weights'] != list(weights):
        raise ValueError('ME changed frozen weights')
    if not np.isclose(result['gross_pnl_usd']-result['friction_usd'], result['net_pnl_usd']):
        raise ValueError('ME dollar ledger does not conserve costs')
    feedback = result['period_feedback']; longest = longest_ms*1_000_000
    if (sum(p['signals'] for p in feedback) != result['prediction_signal_count']
            or sum(p['statuses'].get('matured', 0) for p in feedback) != result['matured_signal_count']):
        raise ValueError('ME ledger does not conserve prediction signals')
    matured = histories = scored = 0
    for p in feedback:
        if p['signals'] != p['pending']+sum(p['statuses'].values()):
            raise ValueError('ME signal denominator lost unresolved samples')
        if p['updated_selector']:
            if (p['status'] != 'matured' or not p['signals'] or p['pending']
                    or p['statuses'].get('matured', 0) != p['signals']
                    or p['observed_at_ns'] < max(p['end_ns'], p['last_origin_ns']+longest)
                    or not np.isclose(p['reward'], np.dot(result['reward_weights'],
                        p['signal_feature_sum'])/p['signals'], rtol=1e-10, atol=1e-10)):
                raise ValueError('ME complete-period reward or maturity differs')
            matured += 1
    selections = result['period_selections']
    by_id = {v['version_sha256']: v for v in versions}
    for p in selections:
        visible = [v for v in versions if pd.Timestamp(v['available_at_utc']).value <= p['selected_at_ns']]
        if not visible or visible[-1]['version_sha256'] != p['model_version_id']:
            raise ValueError('ME used future or stale version')
        for e in p.get('evaluations', []):
            if e['model_version_id'] != p['model_version_id']:
                raise ValueError('ME ARS uses another model version')
            if e['source'] == 'live_selected_model':
                if e['live_period'] and e['live_period']['observed_at_ns'] > p['selected_at_ns']:
                    raise ValueError('ME live score comes from future')
                continue
            histories += 1
            if e['known_through_ns'] != p['selected_at_ns'] or e['window_end_ns'] > e['known_through_ns']:
                raise ValueError('ME history exceeds current time')
            if e['score'] is not None:
                if (e['status'] != 'matured' or not e['total_signals']
                        or e['matured_signal_count'] != e['total_signals']
                        or e['latest_signal_required_ns'] > e['known_through_ns']
                        or e['window_start_ns'] < pd.Timestamp(by_id[e['model_version_id']]['available_at_utc']).value
                        or not np.isclose(e['score'], np.dot(result['reward_weights'],
                            e['prediction_feature_expectation']), rtol=1e-10, atol=1e-10)):
                    raise ValueError('ME history signal mean is not fully mature')
                scored += 1
    states = result['selector_version_statistics']
    if (sum(sum(s['visits']) for s in states.values()) != len(selections)
            or sum(sum(s['feedback_counts']) for s in states.values()) != matured):
        raise ValueError('ME selector counts differ from period ledger')
    return dict(run_status='executed', selections=len(selections), mature_periods=matured,
        history_windows=histories, scored_windows=scored,
        selected_version_ids=sorted({p['model_version_id'] for p in selections}))


def run_four(plan_path, *, detail=False, checkpoint_dir=None):
    """按日流式新增证据，ME校准冻结后才进入评价；旧OE数值从原文件继承。"""
    plan = json.loads(Path(plan_path).read_text()); validate_config(plan['config'])
    source = plan['archived_oe_source']
    if (plan.get('plan_kind') != 'frozen_four_combinations' or plan['plan_sha256'] != fingerprint(plan)
            or plan['code_sha256'] != code_hashes() or sha256_file(source['path']) != source['sha256']):
        raise ValueError('Frozen four-combination source integrity differs')
    old, library, calendar = load_oe(source['path'])
    if old['plan'] != plan['base_oe_dynamic_plan']:
        raise ValueError('Inherited OE plan differs')
    base = old['plan']['base_ars_plan']['base_period_plan']; oe = old['plan']['base_oe_plan']
    expected = old['plan']['strategies'] + [ENSEMBLE] + ME_CONTROLS + [n for s in SCOPES for n in me_names(s)]
    if plan['strategies'] != expected or plan['fixed_reference_model_id'] != base['model_ids'][0]:
        raise ValueError('Declared strategy or fixed reference differs')
    checkpoints = ReplayCheckpoints(checkpoint_dir, plan['plan_sha256'], detail); cases = []
    with threadpool_limits(limits=1):
        for oldcase, frozen in zip(old['age_cases'], base['cases']):
            binding = frozen['session_binding']; protocol = binding['protocol']; age = frozen['max_age_ms']
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            if binding['plan_sha256'] != fingerprint(binding) or reader.provenance != binding['dataset']:
                raise ValueError('Prepared dataset binding differs')
            libcase = next(c for c in library['age_cases'] if c['max_age_ms'] == age)
            training = training_visibility_audit(reader, libcase, base['library_configuration'])
            if training != oldcase['training_visibility_audits']:
                raise ValueError('Training visibility differs from inherited experiment')
            daily = []
            for day, inherited in zip(protocol['sessions']['calibration'], oldcase['calibration']['daily']):
                if inherited['session_id'] != day:
                    raise ValueError('Calibration date differs')
                key = f'age-{age}-ME-calibration-{day}'
                row = checkpoints.load(key)
                if row is None:
                    row = calibration_day(reader, day, libcase, protocol, oe['policies'], inherited)
                    checkpoints.save(key, row)
                if row['session_id'] != day or [r['policy_id'] for r in row['statistics']] != [p['policy_id'] for p in oe['policies']]:
                    raise ValueError('ME calibration checkpoint differs')
                daily.append(row); print(f'完成/恢复 {age}ms ME校准 {day}', flush=True)
            statistics = pool_me(daily, oe['policies'], protocol['reward_horizons_ms'])
            groups = me_fit_groups(statistics, protocol['reward_horizons_ms'], oe['irl_config'], oe['policies'])
            identity = fingerprint(dict(statistics=statistics, groups=groups)); phases = {}
            print(f'{age}ms ME冻结门控：'+str({s: groups[s]['active_reward_weights'] is not None for s in SCOPES}), flush=True)
            for phase in ('validation', 'test'):
                days = []
                for day, inherited in zip(protocol['sessions'][phase], oldcase['phases'][phase]['daily']):
                    prediction = prediction_day_audit(reader, day, libcase, protocol['reward_horizons_ms'])
                    if prediction != inherited['prediction_audit']:
                        raise ValueError('ME/OE evaluation predictions differ')
                    key = f'age-{age}-ME-{phase}-{day}'
                    added = checkpoints.load(key); restored = added is not None
                    if added is None:
                        added = dict(session_id=day, calibration_sha256=identity, prediction_audit=prediction,
                            results=replay_additions(reader, day, libcase, protocol, plan, groups, detail=detail))
                    if (added['session_id'] != day or added['calibration_sha256'] != identity
                            or added['prediction_audit'] != prediction):
                        raise ValueError('ME evaluation checkpoint differs')
                    audits = {r['strategy']: audit_me(r, libcase['versions'], max(protocol['reward_horizons_ms']),
                        groups[r['expert_scope']]['active_reward_weights'] if 'expert_scope' in r else None)
                        for r in added['results']}
                    if restored and audits != added['dynamic_audits']:
                        raise ValueError('Restored ME audit differs')
                    added['dynamic_audits'] = audits
                    if not restored:
                        checkpoints.save(key, added)
                    rows = inherited['results']+added['results']
                    if [r['strategy'] for r in rows] != expected:
                        raise ValueError('Four-combination strategy order differs')
                    days.append(dict(session_id=day, results=rows,
                        dynamic_audits=inherited['dynamic_audits'] | audits, prediction_audit=prediction,
                        me_calibration_sha256=identity, oe_calibration_sha256=oldcase['calibration_sha256']))
                    print(f'完成/恢复 {age}ms {phase} {day}：{len(expected)}项状态', flush=True)
                summaries = summarize_phase(days, expected)
                for summary in summaries:
                    name = summary['strategy']; rows = [next(r for r in d['results'] if r['strategy'] == name) for d in days]
                    if any('matured_signal_count' in r for r in rows):
                        summary.update(matured_signal_count=sum(r['matured_signal_count'] for r in rows),
                            prediction_signal_count=sum(r['prediction_signal_count'] for r in rows))
                phases[phase] = dict(daily=days, statistics=summaries)
            cases.append(dict(max_age_ms=age, me_calibration=dict(daily=daily, statistics=statistics, scopes=groups),
                me_calibration_sha256=identity, oe_calibration=oldcase['calibration']['scopes'],
                training_visibility_audits=training, phases=phases))
    if code_hashes() != plan['code_sha256'] or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source changed during four-combination run')
    return dict(schema_version=1, result_kind='four_combinations_development', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, environment=environment_versions(),
        instrument=old['instrument'], formal_replication_ready=False, selected_age_ms=None, selected_expert_scope=None,
        limits=['ME阈值筛选、方向符号及信号平均都是项目假设；未恢复论文极端千分位筛选。',
            'ME校准成熟子集与在线完整信号期间不同；ME均值不是Eq.(5)订单均值。',
            'OE24项是独立历史结果继承；新增ME/集成按相同数据模型和未改动执行器重放。',
            '仅等式有限最大间隔及最优面最小L1为Algorithm 1工程近似。',
            '严格最近与平移成熟ARS均保留；全尺度成熟、缺格、每日重启限制反馈。',
            '全部14session已有开发用途；尚缺下一收尾的固定范围冻结复核与奖励消融。'])


def compact_evidence(result, result_sha256):
    """仓库归档保留全部汇总、逐日收益/审计、校准和身份，不复制巨量期间明细。"""
    fields = ('strategy', 'run_status', 'blocking_reasons', 'total_trades', 'total_fills', 'gross_pnl_usd',
        'friction_usd', 'net_pnl_usd', 'terminal_position_liquidated', 'reward_type', 'reward_weights',
        'matured_order_count', 'unmatured_fill_rewards_at_end', 'prediction_signal_count', 'matured_signal_count',
        'prediction_feature_expectation', 'period_status_counts', 'history_status_counts', 'cold_start_periods',
        'scored_history_windows', 'expert_scope')
    cases = []
    for case in result['age_cases']:
        phases = {phase: data | dict(daily=[{k: v for k, v in day.items() if k != 'results'} |
            dict(results=[{k: row[k] for k in fields if k in row} for row in day['results']])
            for day in data['daily']]) for phase, data in case['phases'].items()}
        cases.append(case | dict(phases=phases))
    return {k: v for k, v in result.items() if k != 'age_cases'} | dict(age_cases=cases,
        evidence_kind='fmato_four_combinations_snapshot', source_result_sha256=result_sha256)


def render_four(result):
    """四组合与全部参照从同一JSON生成；阻断null、并列及亏损原样保留。"""
    lines = ['# 收尾2/4：真实时间ME/OE最小四组合', '',
        '全部日期为开发用途。ARS保留两个窗口，学习组保留两种专家来源；门控失败显示阻断。', '',
        f"计划：`{result['plan_sha256']}`。固定模型候选预声明为`{result['plan']['fixed_reference_model_id']}`。", '',
        '| 年龄ms | 策略 | 状态 | 七日净利USD | 成熟ME信号 | 完整成熟期间 | 可评分历史窗 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for case in result['age_cases']:
        by_phase = [case['phases'][p]['statistics'] for p in ('validation', 'test')]
        for name in result['plan']['strategies']:
            rows = [next(r for r in stats if r['strategy'] == name) for stats in by_phase]
            blocked = all(r['run_status'] == 'blocked' for r in rows)
            net = None if blocked or any(r['net_pnl_usd'] is None for r in rows) else sum(r['net_pnl_usd'] for r in rows)
            signals = sum(r.get('matured_signal_count', 0) for r in rows) if not blocked else None
            lines.append(f"| {case['max_age_ms']} | {name} | {'阻断' if blocked else '执行'} | "
                f"{net} | {signals} | {sum(r.get('mature_periods',0) for r in rows)} | {sum(r.get('scored_windows',0) for r in rows)} |")
    lines += ['', '## 限制', ''] + ['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'
