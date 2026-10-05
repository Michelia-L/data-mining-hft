"""收尾3/4：冻结后段复核及赋权消融，不重学奖励或选择有利日期。

论文物理第4页Eq.(2)–(4)/Algorithm 1及第7–8页§4.4/Tables 8–9提供
短尺度/人工组合/学习组合的依据。这里七维单尺度赋权仍沿用共同成熟条件，
只隔离赋权贡献；不是论文E(10tick)仅等短尺度的等价基线。ME聚合、有限
优化、双ARS、短历史等原工程假设不变。冻结文件同时绑定旧奖励和新数据。
"""
from collections import Counter
from copy import deepcopy
import json
import importlib.metadata
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import BASE_DIR, INSTRUMENT_CONFIG
from src.four_combinations import audit_me, code_hashes as four_hashes
from src.history_ars import HistoryARSSelector, HistoryWindowBacktester
from src.library_oe import load_library
from src.model_selector import EnsembleSelector, FlatSelector, SingleModelSelector
from src.multiweek_dynamic_oe import ReplayCheckpoints, audit_dynamic_result, environment_versions
from src.multiweek_readiness import prediction_day_audit, training_visibility_audit
from src.period_ucb import PeriodOESelector
from src.session_experiment import fingerprint, read_day
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.sum_only_reward import FrozenSumOnlyReward
from src.time_execution import TimeExecutionEngine
from src.time_me import PredictionHistoryBacktester, replay_me
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import build_library, freeze_library, predict_library, weekly_schedule


SELECTORS = ['UCB', 'ARS-recent', 'ARS-matured']
SCOPES = ['all_candidates', 'online_library']
ROW_FIELDS = ('strategy', 'run_status', 'blocking_reasons', 'reward_type', 'reward_group',
    'expert_scope', 'reward_weights', 'total_trades', 'total_fills', 'gross_pnl_usd',
    'friction_usd', 'net_pnl_usd', 'terminal_position_liquidated', 'matured_order_count',
    'prediction_signal_count', 'matured_signal_count', 'period_status_counts',
    'history_status_counts', 'cold_start_periods', 'scored_history_windows')


def code_hashes():
    """既有方法逐字节保持，入口/绘图另绑定；结果身份包括实际依赖环境。"""
    return four_hashes() | {p: sha256_file(BASE_DIR/p) for p in
        ('src/frozen_ablation.py', 'run_frozen_ablation.py', 'requirements.txt')}


def validate_config(config, calendar):
    """最多十个连续后段session，不允许略去年龄、专家、窗口或失败条目。"""
    expected = dict(schema_version=1, purpose='frozen_followup_reward_ablation',
        age_candidates_ms=[500, 1000, 2000], reward_types=['ME', 'OE'],
        reward_controls=['equal', 'single_shortest'], expert_scopes=SCOPES, selectors=SELECTORS,
        maturity_policy='all_seven_scales_even_when_weight_is_zero',
        reward_update_policy='inherit_previous_calibration_never_refit',
        holdout_claim='frozen_recheck_no_untouched_claim',
        quality_policy='reject_partial_degraded_missing_source_or_no_quotes_keep_internal_gaps')
    if any(config.get(k) != v for k, v in expected.items()) or set(config) != set(expected) | {
            'evaluation_sessions', 'initial_version_session', 'weekly_updates', 'training_sessions', 'notes'}:
        raise ValueError('Predeclared ages/scopes/windows/maturity required')
    days = config['evaluation_sessions']; initial = config['initial_version_session']
    if (not days or len(days) > 10 or days != sorted(set(days)) or initial not in calendar.sessions
            or any(d not in calendar.sessions or d <= initial for d in days)
            or days != [d for d in calendar.sessions if days[0] <= d <= days[-1]]):
        raise ValueError('Need at most ten consecutive post-baseline sessions')


def load_prior(evidence_path, result_path):
    """旧已归档快照绑定完整结果SHA；读取已冻结校准，不使用后段收益拟合。

    快照是PR #21完整结果的紧凑同源证据。完整结果保留原处，必须与已归档
    source_result_sha256相同；本轮不更写旧文件或旧源码哈希来绕过身份检查。
    """
    prior = json.loads(Path(evidence_path).read_text()); plan = prior.get('plan', {})
    if (prior.get('evidence_kind') != 'fmato_four_combinations_snapshot'
            or prior.get('result_kind') != 'four_combinations_development'
            or prior.get('plan_sha256') != plan.get('plan_sha256')
            or plan.get('plan_sha256') != fingerprint(plan)
            or plan.get('code_sha256') != four_hashes()
            or prior.get('environment') != environment_versions()
            or sha256_file(result_path) != prior['source_result_sha256']):
        raise ValueError('Prior evidence/result/source/environment integrity differs')
    old = plan['base_oe_dynamic_plan']; base = old['base_ars_plan']['base_period_plan']
    for nested in (old, old['base_ars_plan'], base, old['base_oe_plan']):
        if nested['plan_sha256'] != fingerprint(nested):
            raise ValueError('Prior nested plan differs')
    source = base['library_source']
    if sha256_file(source['path']) != source['sha256']:
        raise ValueError('Prior model library changed')
    library, calendar = load_library(source['path'])
    if [c['max_age_ms'] for c in prior['age_cases']] != [500, 1000, 2000]:
        raise ValueError('Prior evidence lost age cases')
    return prior, library, calendar


def jobs_for_case(rewards):
    """生成24个动态状态：两奖励×两控制/两来源×三个选择口径，失败不删项。"""
    jobs = []
    for kind in ('ME', 'OE'):
        for group in ('equal', 'single_shortest', *SCOPES):
            for selector in SELECTORS:
                jobs.append(dict(strategy=f'{kind}-{group}-{selector}', reward_type=kind,
                    reward_group=group, selector=selector, **rewards[kind][group]))
    return jobs


def frozen_rewards(prior_case, horizons):
    """原校准权重/门控逐值继承；控制与学习采用相同H维价格点定义。

    H维单尺度控制并非删掉其它标签，零权也保留共同成熟条件，避免混入更早
    反馈的收益效果。阻断组weights=None，不改成现金或挑另一位专家。
    """
    output = {}
    for kind in ('ME', 'OE'):
        groups = prior_case['me_calibration']['scopes'] if kind == 'ME' else prior_case['oe_calibration']
        output[kind] = dict(equal=dict(weights=[1/len(horizons)]*len(horizons), reasons=[]),
            single_shortest=dict(weights=[1.]+[0.]*(len(horizons)-1), reasons=[]))
        for scope in SCOPES:
            group = groups[scope]; gate = group['execution_gates']['sum_only']
            weights = group['active_reward_weights']
            if (gate['usable_for_execution'] != (weights is not None)
                    or weights is not None and weights != group['reward_fits']['sum_only']['weights']):
                raise ValueError('Inherited reward gate/weights differ')
            if weights is not None:
                FrozenSumOnlyReward(horizons, weights)  # Eq.(3)和有限性独立校验，不归一化。
            output[kind][scope] = dict(weights=deepcopy(weights), reasons=list(gate['reasons']),
                expert_policy_id=gate['expert_policy_id'], calibration_scope_sha256=fingerprint(dict(group=group)))
    return output


def freeze_ablation(config_path, evidence_path, result_path, directories):
    """所有收益回放前绑定七日范围、旧奖励/初版、未来周训练及新行情身份。

    未来数值模型此时未拟合；只声明因果历史。初版直接继承旧库10-20版，
    后续只在日历周首日训练。新一周使用此前评估日训练是预声明在线适应，
    不将这些日的收益、评价标签或最佳模型回传到奖励校准。
    """
    path = Path(config_path).resolve(); config = json.loads(path.read_text())
    prior, library, calendar = load_prior(evidence_path, result_path); validate_config(config, calendar)
    old = prior['plan']['base_oe_dynamic_plan']; ars = old['base_ars_plan']; base = ars['base_period_plan']
    if base['model_ids'][0] != 'Ridge-price-h1':
        raise ValueError('Predeclared fixed reference model identity changed')
    last = base['cases'][0]['session_binding']['protocol']['sessions']['test'][-1]
    days = config['evaluation_sessions']
    if days[0] != next(d for d in calendar.sessions if d > last):
        raise ValueError('Follow-up must start with the next session after prior evaluation')
    first_week = tuple(pd.Timestamp(config['initial_version_session']).isocalendar()[:2])
    first_by_week = {}
    for day in calendar.sessions:
        first_by_week.setdefault(tuple(pd.Timestamp(day).isocalendar()[:2]), day)
    updates = [d for w, d in first_by_week.items() if w != first_week and days[0] <= d <= days[-1]]
    if not updates or updates != config['weekly_updates']:
        raise ValueError('Need predeclared calendar weekly updates and multiple versions')
    extension_config = base['library_configuration'] | dict(bootstrap_session=updates[0],
        through_session=days[-1], evaluation_sessions=[d for d in days if d >= updates[0]],
        notes='冻结后段的周扩展：只用当时已结束且标签成熟的最近1/2session；奖励不重学。')
    # 复用原训练入口的严格冻结规则，临时配置仅适配路径接口，最终计划包含全文。
    with tempfile.TemporaryDirectory(prefix='fmato-weekly-freeze-') as root:
        config_file = Path(root)/'weekly.json'; config_file.write_text(json.dumps(extension_config))
        extension = freeze_library(config_file, directories, calendar.path)
    extension.pop('config_source'); extension['plan_sha256'] = fingerprint(extension)
    schedule = weekly_schedule(extension_config, calendar)
    if [s['update_session'] for s in schedule] != updates:
        raise ValueError('Extension schedule disagrees with frozen follow-up')
    cases = []
    for frozen, prior_case in zip(extension['cases'], prior['age_cases']):
        age = frozen['max_age_ms']; libcase = next(c for c in library['age_cases'] if c['max_age_ms'] == age)
        initial = next(v for v in libcase['versions'] if v['update_session'] == config['initial_version_session'])
        expected_initial = max(v['update_session'] for v in libcase['versions']
            if pd.Timestamp(v['available_at_utc']) <= calendar.sessions[days[0]]['open'])
        if expected_initial != initial['update_session']:
            raise ValueError('Follow-up initial version is not the latest visible prior version')
        training = {initial['update_session']: initial['training_sessions']['2']} | {
            s['update_session']: s['training_sessions']['2'] for s in schedule}
        if training != config['training_sessions']:
            raise ValueError('Predeclared training dates differ')
        reader = PreparedDatasetReader(frozen['dataset']['path'], calendar=calendar)
        require_days = sorted(set(days) | {d for ds in training.values() for d in ds})
        source_dates = {i['metadata']['source_date_utc'] for i in reader.report['inputs']}
        quality = {r['session_id']: r for r in reader.report['sessions']}
        for day in require_days:
            session = calendar.sessions[day]
            needed = {t.date().isoformat() for t in pd.date_range(session['open'].normalize(),
                session['close'].normalize(), freq='D')}
            if not needed <= source_dates or day not in quality or quality[day]['observed_snapshots'] < 2:
                raise ValueError(f'Follow-up source/session unavailable: {age}ms {day}')
        protocol = deepcopy(base['cases'][0]['session_binding']['protocol'])
        if reader.report['horizons_ms'] != protocol['reward_horizons_ms'] or reader.report['interval_ms'] != 500:
            raise ValueError('Follow-up horizon/grid differs')
        protocol['sessions'] = dict(train=sorted({d for ds in training.values() for d in ds}),
            calibration=[], validation=[], test=days)
        cases.append(dict(max_age_ms=age, dataset=frozen['dataset'], initial_version=initial,
            protocol=protocol, rewards=frozen_rewards(prior_case, protocol['reward_horizons_ms']),
            quality=[quality[d] for d in require_days]))
    strategies = [j['strategy'] for j in jobs_for_case(cases[0]['rewards'])] + ['cash', 'Fixed-Ridge-price-h1', 'Mean-Ensemble']
    plan = dict(schema_version=1, plan_kind='frozen_followup_ablation', config=config,
        config_source=dict(path=str(path), sha256=sha256_file(path)),
        prior_evidence=dict(path=str(Path(evidence_path).resolve()), sha256=sha256_file(evidence_path)),
        prior_result=dict(path=str(Path(result_path).resolve()), sha256=prior['source_result_sha256']),
        prior_plan_sha256=prior['plan_sha256'], extension_library_plan=extension, cases=cases,
        selector_parameters=dict(period_ms=base['config']['selection_period_ms'],
            exploration_c_price=base['config']['exploration_c_price'], history_window_ms=ars['config']['history_window_ms'],
            tie_tolerance_reward=ars['config']['tie_tolerance_reward']),
        model_ids=base['model_ids'], strategies=strategies, code_sha256=code_hashes(),
        calendar=extension['calendar'], environment=environment_versions(),
        learned_weights_update_policy='never', selected_age_ms=None, selected_expert_scope=None)
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def blocked_row(job):
    """阻断既无交易也无利润观察；None与实际执行现金0必须区分。"""
    return dict(strategy=job['strategy'], reward_type=job['reward_type'], reward_group=job['reward_group'],
        expert_scope=job['reward_group'], run_status='blocked', blocking_reasons=job['reasons'],
        total_fills=None, net_pnl_usd=None, gross_pnl_usd=None, friction_usd=None)


def replay_day(reader, day, case, plan, *, detail=False):
    """每日仅生成一个N×K预测矩阵，27项用相同报价、成本和可见周版本。

    七维零权不改变有效性/成熟要求。每个策略有独立账户、反馈和历史回放；
    只复用当前可见预测输入，不复用其它组反馈或用未来标签筛选交易时刻。
    """
    protocol = case['protocol']; parameters = plan['selector_parameters']
    frame = read_day(reader, day); interval = reader.report['interval_ms']
    matrix, versions = predict_library(case['versions'], frame, interval)
    quotes = frame[SNAPSHOT_SCHEMA.names+['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
    valid = frame.feature_valid.to_numpy(); quotes['feature_valid'] = valid & np.isfinite(matrix).all(axis=1)
    models = [SimpleNamespace(name=m) for m in plan['model_ids']]; rows = []
    for job in jobs_for_case(case['rewards']):
        if job['weights'] is None:
            rows.append(blocked_row(job)); continue
        reward = FrozenSumOnlyReward(protocol['reward_horizons_ms'], job['weights'])
        if job['selector'] == 'UCB':
            selector = PeriodOESelector(job['strategy'], models, period_ms=parameters['period_ms'],
                c=parameters['exploration_c_price'])
        else:
            alignment = job['selector'].removeprefix('ARS-')
            history_class = PredictionHistoryBacktester if job['reward_type'] == 'ME' else HistoryWindowBacktester
            history = history_class(frame, case['versions'], reader.calendar, reward, interval_ms=interval,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'], threshold=protocol['threshold'])
            selector = HistoryARSSelector(job['strategy'], models, history, period_ms=parameters['period_ms'],
                window_ms=parameters['history_window_ms'], alignment=alignment,
                tie_tolerance=parameters['tie_tolerance_reward'])
        if job['reward_type'] == 'ME':
            row = replay_me(selector, frame, matrix, versions, reward, reader.calendar, interval_ms=interval,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
                threshold=protocol['threshold'], detail=detail)
        else:
            engine = TimeExecutionEngine(reward, calendar=reader.calendar, interval_ms=interval,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
                threshold=protocol['threshold'], force_replay_end=False)
            row = engine.run_backtest(selector, quotes, matrix, detail=detail, model_version_ids=versions)
            if job['selector'] != 'UCB':
                row.pop('exploration_c_price')
                histories = [e for p in row['period_selections'] for e in p['evaluations']
                    if e['source'] == 'independent_history_replay']
                row.update(history_alignment=alignment, history_window_ms=parameters['history_window_ms'],
                    history_status_counts=dict(Counter(e['status'] for e in histories)),
                    scored_history_windows=sum(e['score'] is not None for e in histories),
                    cold_start_periods=sum(p['cold_start'] for p in row['period_selections']))
        row.update(run_status='executed', reward_type=job['reward_type'], reward_group=job['reward_group'],
            current_prediction_unavailable_rows=int((valid & ~quotes.feature_valid.to_numpy()).sum()))
        if job['reward_group'] in SCOPES:
            row['expert_scope'] = job['reward_group']
        rows.append(row)
    equal = FrozenSumOnlyReward(protocol['reward_horizons_ms'], [1/len(protocol['reward_horizons_ms'])]*len(protocol['reward_horizons_ms']))
    refs = [FlatSelector('cash', models), SingleModelSelector('Fixed-Ridge-price-h1', models, 0),
        EnsembleSelector('Mean-Ensemble', models)]
    for selector in refs:
        selector.reward_type = 'OE'
        engine = TimeExecutionEngine(equal, calendar=reader.calendar, interval_ms=interval,
            latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'], force_replay_end=False)
        row = engine.run_backtest(selector, quotes, matrix, detail=detail, model_version_ids=versions)
        row.update(run_status='executed', reward_group='reference'); rows.append(row)
    return rows


def audit_day(rows, case, plan):
    """重新审计每次运行或恢复：顺序/权重、金额、共同成熟条件及未来版本隔离。"""
    if [r['strategy'] for r in rows] != plan['strategies']:
        raise ValueError('Follow-up lost declared strategy states')
    jobs = {j['strategy']: j for j in jobs_for_case(case['rewards'])}; output = {}
    for row in rows:
        job = jobs.get(row['strategy'])
        if job and (row['run_status'] == 'blocked') != (job['weights'] is None):
            raise ValueError('Inherited calibration gate changed')
        if row['run_status'] == 'executed' and not np.isclose(row['gross_pnl_usd']-row['friction_usd'], row['net_pnl_usd']):
            raise ValueError('Follow-up dollar ledger does not conserve costs')
        audit = audit_me if job and job['reward_type'] == 'ME' else audit_dynamic_result
        output[row['strategy']] = audit(row, case['versions'], max(case['protocol']['reward_horizons_ms']),
            job['weights'] if job else None)
    return output


def summarize(daily, strategies):
    """所有预声明策略完整加总；若未平仓净利总体未定义，不择日汇总收益。"""
    output = []
    for name in strategies:
        rows = [next(r for r in d['results'] if r['strategy'] == name) for d in daily]
        if any(r['run_status'] == 'blocked' for r in rows):
            if not all(r['run_status'] == 'blocked' for r in rows):
                raise ValueError('Frozen reward gate changed across sessions')
            output.append(rows[0] | dict(scheduled_days=len(daily), executed_days=0, blocked_days=len(daily))); continue
        settled = all(r['terminal_position_liquidated'] for r in rows)
        total = dict(strategy=name, run_status='executed', scheduled_days=len(daily), executed_days=len(daily),
            blocked_days=0, pnl_aggregation_defined=settled)
        for field in ('gross_pnl_usd', 'friction_usd', 'net_pnl_usd'):
            total[field] = sum(r[field] for r in rows) if settled else None
        for field in ('total_fills', 'matured_order_count', 'prediction_signal_count', 'matured_signal_count'):
            total[field] = sum(r.get(field, 0) for r in rows)
        for field in ('selections', 'mature_periods', 'history_windows', 'scored_windows'):
            total[field] = sum(d['dynamic_audits'][name].get(field, 0) for d in daily)
        output.append(total)
    return output


def run_ablation(plan_path, *, checkpoint_dir, detail=False):
    """只消费冻结计划，按完整日发布并可恢复；完整期间账本留在日检查点。

    最终结果保留27项日摘要、独立审计、数据质量、模型参数及校准身份，不把
    大量重复期间明细放入仓库。日账本SHA另绑定；恢复重做审计，不重跑交易。
    每日账户/队列重启，加总不是连续跨日资金曲线。
    """
    plan = json.loads(Path(plan_path).read_text())
    if (plan.get('plan_kind') != 'frozen_followup_ablation' or plan.get('plan_sha256') != fingerprint(plan)
            or plan.get('code_sha256') != code_hashes() or plan.get('environment') != environment_versions()):
        raise ValueError('Frozen follow-up plan/source/environment integrity differs')
    for source in (plan['prior_evidence'], plan['prior_result'], plan['config_source']):
        if sha256_file(source['path']) != source['sha256']:
            raise ValueError('Frozen source changed')
    prior, library, calendar = load_prior(plan['prior_evidence']['path'], plan['prior_result']['path'])
    validate_config(plan['config'], calendar)
    for frozen, previous in zip(plan['cases'], prior['age_cases']):
        if frozen['rewards'] != frozen_rewards(previous, frozen['protocol']['reward_horizons_ms']):
            raise ValueError('Frozen follow-up changed inherited calibration weights/gates')
        old_case = next(c for c in library['age_cases'] if c['max_age_ms'] == frozen['max_age_ms'])
        if frozen['initial_version'] not in old_case['versions']:
            raise ValueError('Frozen follow-up changed prior initial model')
    checks = ReplayCheckpoints(checkpoint_dir, plan['plan_sha256'], detail)
    extension = checks.load('weekly-extension')
    if extension is None:
        with tempfile.TemporaryDirectory(prefix='fmato-weekly-build-') as root:
            path = Path(root)/'plan.json'; path.write_text(json.dumps(plan['extension_library_plan']))
            extension = build_library(path)
        checks.save('weekly-extension', extension)
    if extension['plan'] != plan['extension_library_plan']:
        raise ValueError('Restored extension library differs')
    cases = []
    with threadpool_limits(limits=1):
        for frozen in plan['cases']:
            case = deepcopy(frozen); age = case['max_age_ms']
            new = next(c for c in extension['age_cases'] if c['max_age_ms'] == age)
            case['versions'] = [case['initial_version']] + new['versions']
            reader = PreparedDatasetReader(case['dataset']['path'], calendar=calendar)
            if reader.provenance != case['dataset']:
                raise ValueError('Frozen follow-up dataset changed')
            training = training_visibility_audit(reader, case, extension['plan']['config'])
            daily = []
            for day in plan['config']['evaluation_sessions']:
                prediction = prediction_day_audit(reader, day, case, case['protocol']['reward_horizons_ms'])
                key = f'age-{age}-frozen-{day}'; full = checks.load(key); restored = full is not None
                if full is None:
                    full = dict(session_id=day, prediction_audit=prediction,
                        results=replay_day(reader, day, case, plan, detail=detail))
                if full['session_id'] != day or full['prediction_audit'] != prediction:
                    raise ValueError('Restored follow-up prediction/date differs')
                audits = audit_day(full['results'], case, plan)
                if restored and full['dynamic_audits'] != audits:
                    raise ValueError('Restored follow-up dynamic audits differ')
                full['dynamic_audits'] = audits
                if not restored:
                    checks.save(key, full)
                payload_sha = fingerprint(dict(payload=full))
                daily.append(dict(session_id=day, prediction_audit=prediction, dynamic_audits=audits,
                    full_day_payload_sha256=payload_sha,
                    results=[{k: r[k] for k in ROW_FIELDS if k in r} for r in full['results']]))
                del full
                print(f'完成/恢复 {age}ms 冻结复核 {day}：27项状态', flush=True)
            cases.append(dict(max_age_ms=age, rewards=case['rewards'], versions=case['versions'],
                training_visibility_audits=training, daily=daily, statistics=summarize(daily, plan['strategies'])))
    if (code_hashes() != plan['code_sha256'] or any(sha256_file(s['path']) != s['sha256']
            for s in (plan['prior_evidence'], plan['prior_result'], plan['config_source']))):
        raise ValueError('Source changed during frozen follow-up')
    return dict(schema_version=1, result_kind='frozen_followup_ablation_recheck', plan=plan,
        plan_sha256=plan['plan_sha256'], environment=environment_versions(), instrument=INSTRUMENT_CONFIG['CME_ES'],
        checkpoint_binding=checks.binding, figure_environment=dict(matplotlib=importlib.metadata.version('matplotlib')),
        age_cases=cases, selected_age_ms=None, selected_expert_scope=None,
        holdout_claim=plan['config']['holdout_claim'], account_policy='independent_daily_accounts_and_feedback',
        limits=['冻结后复核不称未触碰测试；七日不等于论文三个月长期实验。',
            '最短尺度赋权仍共同成熟七尺度；仅检验赋权，不等价单尺度独立成熟。',
            '原ME聚合/筛选、双ARS、有限IRL和1/2session历史等工程假设保留。',
            '完整期间账本在日检查点，结果保存其内容SHA与摘要，不是连续跨日资金。'])


def render_ablation(result):
    """全部27策略的日/累计收益从同一JSON输出，阻断显示未定义，不画现金零。"""
    lines = ['# 收尾3/4：七日冻结复核与赋权消融', '',
        f"计划 `{result['plan_sha256']}`。日期：{result['plan']['config']['evaluation_sessions']}。", '',
        '旧奖励不重学；单尺度仍共同成熟七尺度。三年龄/两来源/双ARS全部保留。', '',
        '| 年龄ms | 策略 | 状态 | 净利USD | 毛利USD | 摩擦USD | 成熟期间 | 可评分历史窗 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in result['age_cases']:
        for r in case['statistics']:
            lines.append(f"| {case['max_age_ms']} | {r['strategy']} | {r['run_status']} | "
                f"{r['net_pnl_usd']} | {r['gross_pnl_usd']} | {r['friction_usd']} | "
                f"{r.get('mature_periods',0)} | {r.get('scored_windows',0)} |")
    lines += ['', '## 逐日净利', '', '| 年龄ms | 日期 | 策略 | 状态 | 净利USD |', '| ---: | --- | --- | --- | ---: |']
    for case in result['age_cases']:
        for day in case['daily']:
            for r in day['results']:
                net = r['net_pnl_usd'] if r.get('terminal_position_liquidated') else None
                lines.append(f"| {case['max_age_ms']} | {day['session_id']} | {r['strategy']} | {r['run_status']} | {net} |")
    lines += ['', '## 限制', ''] + ['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'


def plot_ablation(result, output_dir):
    """标准matplotlib输出可共享PNG：全27项逐日热图及预声明UCB日累计曲线。

    灰格表示未执行/收益未定义，不替换成现金0；红蓝以零为中心统一美元尺度。
    曲线是独立日账户净利的算术累计，不是复利资金曲线或实盘净值。曲线只
    展示预声明UCB消融和三参照，热图/同源表格保留全部双ARS与失败状态。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    root = Path(output_dir)
    paths = [root/'daily_net_usd.png', root/'cumulative_daily_net_usd.png']
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite frozen-ablation figures')
    names = result['plan']['strategies']; days = result['plan']['config']['evaluation_sessions']
    matrices = [np.array([[next(r['net_pnl_usd'] if r.get('terminal_position_liquidated') else None
        for r in day['results'] if r['strategy'] == name)
        for day in c['daily']] for name in names], dtype=float) for c in result['age_cases']]
    bound = max(1., max(float(np.nanmax(np.abs(m))) for m in matrices))
    cmap = plt.get_cmap('RdBu').copy(); cmap.set_bad('#bbbbbb')
    fig, axes = plt.subplots(1, 3, figsize=(18, 12), sharey=True)
    for ax, matrix, case in zip(axes, matrices, result['age_cases']):
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap,
            norm=TwoSlopeNorm(vmin=-bound, vcenter=0., vmax=bound))
        ax.set_title(f"Quote age <= {case['max_age_ms']} ms")
        ax.set_xticks(range(len(days)), [d[5:] for d in days], rotation=45)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for i, j in zip(*np.where(~np.isfinite(matrix))):
            ax.text(j, i, 'NA', ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=axes, label='Daily net USD; grey = blocked/undefined', shrink=.7)
    fig.suptitle('Frozen recheck: all 27 states, no missing state replaced by zero')
    fig.savefig(paths[0], dpi=150, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for col, case in enumerate(result['age_cases']):
        for row, kind in enumerate(('ME', 'OE')):
            ax = axes[row, col]; blocked = []
            selected = [f'{kind}-{g}-UCB' for g in ('equal', 'single_shortest', *SCOPES)] + names[-3:]
            for name in selected:
                values = [next(r['net_pnl_usd'] if r.get('terminal_position_liquidated') else None
                    for r in d['results'] if r['strategy'] == name) for d in case['daily']]
                if any(v is None for v in values):
                    blocked.append(name); continue
                ax.plot(range(len(days)), np.cumsum(values), marker='o', markersize=3, label=name)
            ax.axhline(0., color='black', linewidth=.6)
            ax.set_title(f"{kind}, quote age <= {case['max_age_ms']} ms")
            ax.set_xticks(range(len(days)), [d[5:] for d in days], rotation=45)
            ax.set_ylabel('Sum of independent daily net USD')
            ax.grid(alpha=.2); ax.legend(fontsize=7)
            ax.text(.02, .02, 'Blocked/undefined: '+(', '.join(blocked) or 'none'), transform=ax.transAxes, fontsize=7)
    fig.tight_layout(); fig.savefig(paths[1], dpi=150, bbox_inches='tight'); plt.close(fig)
    return paths
