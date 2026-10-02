"""校准 OE 学习→冻结→UCB/ARS 联动，原18个等权/静态控制完整保留。

物理第4页 Algorithm 1 与第5页 Algorithm 2/3 的接口按时间隔离连接。
有限候选专家和完整最大间隔求解仍是工程近似；现金并非当前模型动作。
求解成功、专家可表示、动作集一致和实际可执行分别记录，失败不补现金收益。
"""
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.ars_experiment import code_hashes as ars_code_hashes, freeze_ars_experiment, replay_ars_day
from src.artifacts import pretty_json
from src.config import BASE_DIR, INSTRUMENT_CONFIG
from src.history_ars import HistoryARSSelector, HistoryWindowBacktester
from src.library_oe import experiment_code_hashes as oe_code_hashes, freeze_library_oe, load_library, replay_candidates
from src.period_experiment import code_hashes as period_code_hashes
from src.period_ucb import PeriodOESelector
from src.session_experiment import fingerprint, read_day, summarize_days
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import sha256_file
from src.snapshot_irl import learn_calibration_reward, pooled_policy_statistics
from src.sum_only_reward import FrozenSumOnlyReward, learn_sum_only
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library


LEARNED_STRATEGIES = ['Learned-OE-UCB', 'Learned-OE-ARS-recent', 'Learned-OE-ARS-matured']


def code_hashes():
    """包含旧控制及新学习/联动源码；不修改旧库绑定或旧结果。"""
    return ars_code_hashes() | {p: sha256_file(BASE_DIR/p) for p in
        ('src/sum_only_reward.py', 'src/learned_oe_experiment.py', 'run_learned_oe.py')}


def freeze_learned_experiment(config_path, library_path):
    """评估前声明唯一学习口径、动作门控与全部年龄；沿用旧18个控制。

显式要求专家属于原模型库且每个库动作都有校准成熟订单，是保守的项目门控，
不是论文规定。不能为了取得可用结果，改选次优专家或静默加入现金动作。
"""
    path = Path(config_path).resolve(); config = json.loads(path.read_text())
    required = {'schema_version', 'purpose', 'ars_protocol_file', 'learner', 'execution_gate'}
    if (set(config)-required-{'notes'} or required-set(config) or config['schema_version'] != 1
            or config['purpose'] != 'development_learned_OE_selection'
            or config['learner'] != 'sum_only_full_finite_max_margin_min_L1_tie_break'
            or config['execution_gate'] != 'expert_in_library_all_library_OE_observed'):
        raise ValueError('Unsupported learned OE development protocol')
    ars_path = path.parent/config['ars_protocol_file']
    ars = freeze_ars_experiment(ars_path, library_path)
    ars_config = json.loads(ars_path.read_text())
    period_path = ars_path.parent/ars_config['period_protocol_file']
    period_config = json.loads(period_path.read_text())
    oe = freeze_library_oe(period_path.parent/period_config['oe_protocol_file'], library_path)
    base = ars['base_period_plan']
    if oe['cases'] != base['cases'] or oe['library_source'] != base['library_source']:
        raise ValueError('Calibration and execution data/library bindings differ')
    plan = dict(schema_version=1, plan_kind='frozen_learned_OE_selection', config=config,
        config_source=dict(path=str(path), sha256=sha256_file(path)), base_ars_plan=ars,
        base_oe_plan=oe, strategies=ars['strategies']+LEARNED_STRATEGIES,
        code_sha256=code_hashes(), age_selection='none_all_ages_retained', holdout_claim='none_development_dates',
        stage_roles=dict(calibration='expert_and_reward_only', validation='frozen_reward_diagnostics',
                         test='frozen_development_no_retuning'),
        expert_rule='maximum_settled_calibration_net_USD', learned_weights_update_policy='never_after_calibration',
        online_actions='unchanged_weekly_library_no_cash_or_reference_threshold_arm')
    plan['plan_sha256'] = fingerprint(plan)
    return plan


def execution_gate(fit, policies):
    """禁止求解权重自动成为交易权重：逐项公开失败原因，不用未来收益判资格。"""
    library = [p['policy_id'] for p in policies if p['source'] == 'weekly_library']
    expert = fit['expert']['selected_policy_id'] if fit['expert'] else None
    reasons = []
    if fit['status'] != 'fitted': reasons.append('reward_not_fitted')
    if expert not in library: reasons.append('expert_outside_online_library')
    missing = [p for p in library if p not in fit['eligible_policy_ids']]
    if missing: reasons.append('online_library_calibration_OE_incomplete')
    return dict(usable_for_execution=not reasons, reasons=reasons,
        expert_policy_id=expert, missing_online_policy_ids=missing,
        action_space_changed=False, reward_uniquely_identified=False)


def replay_learned_day(reader, day, case, protocol, plan, weights, *, detail=False):
    """同一冻结权重同时进入真实期间反馈和独立历史重回测，不复用等权评分。

每个账户每日重启；模型按既有因果周计划更新，权重仅在校准后冻结。固定
期间、历史窗口、预测可见性、执行费用与旧控制一致。未来价格只用于成熟
反馈；ARS 历史回放仍严格截断到当下，保留最近与平移成熟两种工程口径。
"""
    ars = plan['base_ars_plan']; base = ars['base_period_plan']
    frame = read_day(reader, day)
    matrix, versions = predict_library(case['versions'], frame, reader.report['interval_ms'])
    quotes = frame[SNAPSHOT_SCHEMA.names+['session_id','segment_id','mid_price','feature_valid']].copy()
    valid = frame.feature_valid.to_numpy(); quotes['feature_valid'] = valid & np.isfinite(matrix).all(axis=1)
    models = [SimpleNamespace(name=m) for m in base['model_ids']]; results = []
    for name in LEARNED_STRATEGIES:
        reward = FrozenSumOnlyReward(protocol['reward_horizons_ms'], weights)
        if name == 'Learned-OE-UCB':
            selector = PeriodOESelector(name, models, period_ms=base['config']['selection_period_ms'],
                c=base['config']['exploration_c_price'])
        else:
            alignment = name.removeprefix('Learned-OE-ARS-')
            history = HistoryWindowBacktester(frame, case['versions'], reader.calendar, reward,
                interval_ms=reader.report['interval_ms'], latency_ms=protocol['latency_ms'],
                holding_review_ms=protocol['holding_review_ms'], threshold=protocol['threshold'])
            selector = HistoryARSSelector(name, models, history, period_ms=base['config']['selection_period_ms'],
                window_ms=ars['config']['history_window_ms'], alignment=alignment,
                tie_tolerance=ars['config']['tie_tolerance_reward'])
        engine = TimeExecutionEngine(reward, interval_ms=reader.report['interval_ms'], calendar=reader.calendar,
            latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
            threshold=protocol['threshold'], force_replay_end=False)
        result = engine.run_backtest(selector, quotes, matrix, detail=detail, model_version_ids=versions)
        result.update(run_status='executed', reward_source='frozen_calibration_sum_only',
            reward_weights=list(weights), current_prediction_unavailable_rows=int((valid & ~quotes.feature_valid.to_numpy()).sum()))
        if name != 'Learned-OE-UCB':
            result.pop('exploration_c_price')
            histories = [e for p in result['period_selections'] for e in p['evaluations']
                         if e['source'] == 'independent_history_replay']
            result.update(history_alignment=alignment, history_window_ms=ars['config']['history_window_ms'],
                history_status_counts=dict(Counter(e['status'] for e in histories)),
                scored_history_windows=sum(e['score'] is not None for e in histories),
                cold_start_periods=sum(p['cold_start'] for p in result['period_selections']))
        results.append(result)
    return results


def blocked_result(name, gate):
    """未执行策略没有成交/利润观察；None 必须与已运行的现金零收益区分。"""
    return dict(strategy=name, run_status='blocked', blocking_reasons=gate['reasons'],
        total_fills=None, matured_order_count=None, gross_pnl_usd=None, friction_usd=None,
        net_pnl_usd=None, pnl_aggregation_defined=False)


def run_learned_experiment(plan_path, *, detail=False):
    """每个年龄先重算真实校准账本并冻结学习，再进入验证/测试；失败也输出21项。"""
    plan = json.loads(Path(plan_path).read_text()); ars = plan.get('base_ars_plan', {})
    base = ars.get('base_period_plan', {}); oe = plan.get('base_oe_plan', {})
    for value, kind, hashes in ((plan, 'frozen_learned_OE_selection', code_hashes()),
            (ars, 'frozen_history_OE_ARS', ars_code_hashes()),
            (base, 'frozen_period_OE_UCB', period_code_hashes()),
            (oe, 'frozen_library_time_OE_IRL', oe_code_hashes())):
        if (value.get('plan_kind') != kind or value.get('schema_version') != 1
                or value.get('plan_sha256') != fingerprint(value) or value.get('code_sha256') != hashes):
            raise ValueError('Frozen learned OE plan/source integrity check failed')
    source = base['library_source']
    if sha256_file(source['path']) != source['sha256']: raise ValueError('Library artifact changed')
    library, calendar = load_library(source['path']); cases = []
    if library['plan']['config'] != base['library_configuration']: raise ValueError('Library configuration changed')
    with threadpool_limits(limits=1):
        for frozen in base['cases']:
            binding = frozen['session_binding']; protocol = binding['protocol']
            if binding['plan_sha256'] != fingerprint(binding): raise ValueError('Session binding changed')
            reader = PreparedDatasetReader(binding['dataset']['path'], calendar=calendar,
                allow_partial=protocol['allow_partial'], include_degraded=protocol['include_degraded'])
            if reader.provenance != binding['dataset']: raise ValueError('Prepared dataset changed')
            case = next(c for c in library['age_cases'] if c['max_age_ms'] == frozen['max_age_ms'])
            daily, audits = replay_candidates(reader, protocol['sessions']['calibration'], case,
                oe['policies'], protocol, detail=detail)
            stats = pooled_policy_statistics(daily, oe['policies'], protocol['reward_horizons_ms'])
            fits = {name: learn_calibration_reward(stats, protocol['reward_horizons_ms'], oe['irl_config'], name)
                    for name in oe['irl_config']['weight_constraints']}
            reference = fits.get('signed_box', fits['simplex'])
            fit = learn_sum_only(stats, protocol['reward_horizons_ms'], oe['irl_config'], reference)
            gate = execution_gate(fit, oe['policies']); fits['sum_only'] = fit
            # 校准摘要单独绑定身份。验证/测试的预测、成交和收益不参与学习函数输入。
            calibration_sha = fingerprint(dict(statistics=stats, reward_fits=fits, gate=gate))
            print(f"{frozen['max_age_ms']}ms 校准：{fit['status']}；在线门控：{gate['reasons']}", flush=True)
            phases = dict(calibration=dict(daily=daily, statistics=stats, horizon_audits=audits))
            for phase in base['config']['stages']:
                daily = []
                for day in protocol['sessions'][phase]:
                    row = replay_ars_day(reader, day, case, protocol, ars, detail=detail)
                    row['results'] += (replay_learned_day(reader, day, case, protocol, plan, fit['weights'], detail=detail)
                        if gate['usable_for_execution'] else [blocked_result(n, gate) for n in LEARNED_STRATEGIES])
                    daily.append(row)
                statistics = []
                for strategy in plan['strategies']:
                    if strategy in LEARNED_STRATEGIES and not gate['usable_for_execution']:
                        total = blocked_result(strategy, gate)
                    else:
                        total = summarize_days(daily, strategy) | dict(run_status='executed')
                        for key in ('history_status_counts', 'period_status_counts'):
                            counts = Counter()
                            for day in daily: counts.update(next(r for r in day['results'] if r['strategy'] == strategy).get(key, {}))
                            total[key] = dict(counts)
                    statistics.append(total)
                phases[phase] = dict(daily=daily, statistics=statistics,
                    calibration_sha256=calibration_sha, active_reward_weights=fit['weights'] if gate['usable_for_execution'] else None)
                print(f"完成 {frozen['max_age_ms']}ms {phase}：18个旧控制与3个学习策略状态", flush=True)
            cases.append(dict(max_age_ms=frozen['max_age_ms'], reward_fits=fits, execution_gate=gate,
                calibration_sha256=calibration_sha, active_reward_weights=fit['weights'] if gate['usable_for_execution'] else None,
                phases=phases))
    if plan['code_sha256'] != code_hashes() or sha256_file(source['path']) != source['sha256']:
        raise ValueError('Source/library changed during experiment')
    return dict(schema_version=1, result_kind='learned_OE_selection_development', plan=plan,
        plan_sha256=plan['plan_sha256'], age_cases=cases, selected_age_ms=None,
        instrument=INSTRUMENT_CONFIG['CME_ES'], environment=dict(python=platform.python_version(),
        **{n: importlib.metadata.version(n) for n in ('numpy','pandas','pyarrow','scikit-learn','scipy','threadpoolctl')}),
        limits=['有限完整候选最大间隔替代论文未公开的生产参数优化器；最小L1仅为最优面上的工程消歧。',
                '专家规则与成熟订单子集沿用旧校准，现金零向量仍是优化约定；不等于论文完整期间校准。',
                '在线动作保留原模型库；专家不属于库或库候选缺校准OE时阻断，不换专家、不补现金。',
                '校准一次冻结奖励，周版本按因果计划变化；未实现跨周重新校准奖励或每日反馈迁移。',
                '大正负权没有裁剪或归一化；C=1价格点沿用控制，探索尺度可能变化且未按测试调参。',
                '最近/平移成熟双ARS、固定期间与执行边界仍为已披露工程口径；负结果完整保留。',
                '真实日期全部用于开发；仍缺时间ME学习、真实跨周、正式全量方案与长期复现。'])


def render_learned_experiment(result):
    """同一JSON报告学习与执行门控、所有策略收益和数值诊断；阻断不显示0。"""
    lines = ['# OE 奖励学习与在线选择联动', '', f"计划 `{result['plan_sha256']}`；开发对照。", '',
        '| 年龄 ms | 仅等式拟合 | 专家 | 最优间隔 | 可执行 | 阻断原因 |',
        '| --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        fit = case['reward_fits']['sum_only']; gate = case['execution_gate']
        lines.append(f"| {case['max_age_ms']} | {fit['status']} | {gate['expert_policy_id']} | "
            f"{(fit['diagnostics'] or {}).get('optimal_margin')} | {gate['usable_for_execution']} | {gate['reasons']} |")
    lines += ['', '| 年龄 ms | 阶段 | 策略 | 状态 | 成交 | 毛利 USD | 成本 USD | 净利 USD |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in result['age_cases']:
        for phase in ('validation', 'test'):
            for r in case['phases'][phase]['statistics']:
                lines.append(f"| {case['max_age_ms']} | {phase} | {r['strategy']} | {r['run_status']} | "
                    f"{r['total_fills']} | {r['gross_pnl_usd']} | {r['friction_usd']} | {r['net_pnl_usd']} |")
    lines += ['', '## 校准拟合与门控诊断', '']
    for case in result['age_cases']:
        lines += [f"### {case['max_age_ms']}ms", '', '```json',
            pretty_json(dict(reward_fits=case['reward_fits'], execution_gate=case['execution_gate'])), '```', '']
    lines += ['## 限制', ''] + ['- '+s for s in result['limits']]
    return '\n'.join(lines)+'\n'
