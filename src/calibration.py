"""OE 历史校准：完整静态候选、净利专家资格与在线动作门控。

物理第4页 Algorithm 1 未公开专家筛选和生产优化器。本项目从已平仓的
库内候选按净利取首个并列最优，再检查完整成熟向量；不为拟合换次优专家。
现金与三个固定门槛参照仍在有限优化矩阵中，在线动作只保留12个库模型。
"""
from types import SimpleNamespace
import numpy as np
from src.irl_reward import IRLRewardLearner
from src.model_selector import FlatSelector, SingleModelSelector
from src.session_experiment import audit_horizons, fingerprint, read_day, summarize_days
from src.snapshot_backtest import FEATURE_COLUMNS
from src.time_execution import TimeExecutionEngine
from src.timed_snapshots import SNAPSHOT_SCHEMA
from src.weekly_library import predict_library, predict_model

def policy_specs(config, threshold):
    """现金优先的预声明顺序用于消歧，门槛乘数不从测试段选优。"""
    return [dict(policy_id='cash', cash=True, threshold=threshold)] + [
        dict(policy_id=f'Ridge-threshold-x{m:g}', cash=False, threshold=threshold * m)
        for m in config['threshold_multipliers']]

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

def qualify_calibration(statistics, horizons, config, *, expert_policy_ids):
    """只消费校准期 P×H 成熟订单特征，保留无数据、无信息和求解失败结果。

    专家先在所有已平仓候选中按净盈亏选择；若其奖励不可观测，不偷换成次优
    专家。现金是零向量约定；至少需要一个可观测非现金策略才允许拟合。
    求解收敛、专家可表示、正利润专家和权重唯一识别是不同概念。
    expert_policy_ids 仅限制专家来源，不删减用于求解的候选矩阵。
    库内来源属于工程设定，并列仍按 statistics 的冻结
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
    base = dict(weight_constraint='sum_only', expert=expert, excluded_policies=excluded,
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
    return base | dict(status='qualified')

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

def blocked_result(name, gate):
    """未执行策略没有成交/利润观察；None 必须与已运行的现金零收益区分。"""
    return dict(strategy=name, run_status='blocked', blocking_reasons=gate['reasons'],
        total_fills=None, matured_order_count=None, gross_pnl_usd=None, friction_usd=None,
        net_pnl_usd=None, pnl_aggregation_defined=False)

def replay_candidates(reader, days, library_case, specs, protocol, *, detail=False):
    """一日一次生成 N×K 可见预测，每个候选独立相同成本回放。

    库的模型 ID 表示冻结的算法/特征/历史规则，参数可按周更新；不因验证净利
    改选。失败/未预热模型只在当下不可预测时禁用，不按未来标签有效性删决策。
    日账户重置、队列不跨日；缺尾未平仓保留并使汇总净利未定义。
    """
    daily, audits = [], []
    models = library_case['versions'][0]['models']
    indices = {m['model_id']: i for i, m in enumerate(models)}
    reference = library_case['fixed_reference'] | dict(features=FEATURE_COLUMNS, family='Ridge')
    fixed_id = 'fixed-' + fingerprint(reference)
    reward = IRLRewardLearner(horizons=protocol['reward_horizons_ms'], definition='paper_price_difference')
    for day in days:
        frame = read_day(reader, day)
        audit = audit_horizons(frame, reader.calendar, protocol['reward_horizons_ms'])
        audits.append(audit)
        matrix, ids = predict_library(library_case['versions'], frame, reader.report['interval_ms'])
        reference_preds = np.full(len(frame), np.nan)
        valid = frame.feature_valid.to_numpy()
        if reference['status'] == 'fitted' and valid.any():
            reference_preds[valid] = predict_model(reference, frame.loc[valid, FEATURE_COLUMNS])
        results = []
        for spec in specs:
            predictions = (np.zeros(len(frame)) if spec['cash'] else
                           reference_preds if spec['source'] == 'fixed_reference' else matrix[:, indices[spec['model_id']]])
            versions = ids if spec['source'] == 'weekly_library' else np.full(len(frame), 'cash' if spec['cash'] else fixed_id, dtype=object)
            quotes = frame[SNAPSHOT_SCHEMA.names + ['session_id', 'segment_id', 'mid_price', 'feature_valid']].copy()
            # 只增加当时模型可预测性条件；未来标签/未来成熟情况不在掩码中。
            quotes['feature_valid'] = valid & np.isfinite(predictions)
            model_name = reference['name'] if spec['source'] == 'fixed_reference' else spec['model_id']
            selector = (FlatSelector if spec['cash'] else SingleModelSelector)(spec['policy_id'], [SimpleNamespace(name=model_name)])
            selector.reward_type = 'OE'
            engine = TimeExecutionEngine(reward, interval_ms=reader.report['interval_ms'], calendar=reader.calendar,
                latency_ms=protocol['latency_ms'], holding_review_ms=protocol['holding_review_ms'],
                threshold=spec['threshold'], force_replay_end=False)
            result = engine.run_backtest(selector, quotes, predictions[:, None], detail=detail, model_version_ids=versions)
            result['session_boundary'] = dict(policy='independent_daily_accounts_no_feedback_carry',
                unexecuted_intents_not_forwarded=result['unexecuted_intents_at_end'],
                unmatured_OE_not_forwarded=result['unmatured_fill_rewards_at_end'],
                terminal_position=result['terminal_position'],
                unsettled_net_pnl_cannot_aggregate=not result['terminal_position_liquidated'])
            results.append(result)
        daily.append(dict(session_id=day, input_feature_valid_rows=int(valid.sum()),
            library_version_ids=sorted(set(ids) - {None}), results=results))
    return daily, audits
