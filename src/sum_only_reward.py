"""仅保留 Eq.(3) 权重和约束的有限策略最大间隔近似。

论文物理第4页 Algorithm 1 没有公开生产参数优化器。这里一次枚举完整 P×H
候选矩阵，代替交替策略搜索；专家自身也在 oracle 中，因此最优间隔不大于0。
这消除了仅对子集求解、允许任意正负权时可能出现的无界问题。第二阶段在
相同最优间隔上最小化 L1，仅作可重复消歧，不声称权重被唯一识别。
"""
import numpy as np
from scipy.optimize import linprog

from src.irl_reward import IRLRewardLearner
from src.calibration import best_with_ties


def fit_sum_only(expectations, expert_idx, *, tolerance=1e-8):
    """输入 P×H 成熟订单均值（价格点），输出权重和独立复核的求解诊断。

第一阶段 max margin，约束 D_j·w>=margin、sum(w)=1，D=mu_E-mu_j。
含专家的零差行将 margin 封顶0，不引入额外盒约束或数据决定的范数预算。
第二阶段固定最优 margin，最小化 sum(u)，-u<=w<=u；权重不再归一化，
以免改变原解与间隔。复核任一阶段失败都拒绝导出可用权重。
"""
    mu = np.asarray(expectations, float)
    if (mu.ndim != 2 or min(mu.shape) < 1 or not np.isfinite(mu).all()
            or type(expert_idx) is not int or not 0 <= expert_idx < len(mu)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Invalid finite policy matrix/expert/tolerance')
    h = mu.shape[1]; differences = mu[expert_idx] - mu
    a = np.c_[-differences, np.ones(len(mu))]; e = np.array([np.r_[np.ones(h), 0.]])
    c = np.r_[np.zeros(h), -1.]
    first = linprog(c, A_ub=a, b_ub=np.zeros(len(mu)), A_eq=e, b_eq=[1.],
                    bounds=[(None, None)] * (h + 1), method='highs')
    if not first.success:
        raise RuntimeError('Full finite-policy maximum margin failed: ' + first.message)
    # 对偶乘子符号按 scipy 最小化 LP：不等式乘子<=0，自由变量驻点残差为零。
    y = first.ineqlin.marginals; z = first.eqlin.marginals
    primary = dict(primal_violation=float(max(0., np.max(a @ first.x),
        np.max(np.abs(e @ first.x - 1.)))),
        stationarity_residual=float(np.max(np.abs(c - a.T @ y - e.T @ z))),
        dual_sign_violation=float(max(0., np.max(y))),
        duality_gap=float(abs(first.fun - z[0])))
    if max(primary.values()) > tolerance:
        raise RuntimeError('Maximum-margin primal/dual verification failed')
    margin = float(first.x[-1])
    # 这里固定同一个最优目标值，不放松 tau 以人为制造较小范数或成功专家。
    second_a = np.r_[np.c_[-differences, np.zeros_like(differences)],
                     np.c_[np.eye(h), -np.eye(h)], np.c_[-np.eye(h), -np.eye(h)]]
    second_b = np.r_[np.full(len(mu), -margin), np.zeros(2*h)]
    second_e = np.array([np.r_[np.ones(h), np.zeros(h)]])
    second_c = np.r_[np.zeros(h), np.ones(h)]
    second = linprog(second_c, A_ub=second_a, b_ub=second_b,
                     A_eq=second_e, b_eq=[1.], bounds=[(None, None)]*h+[(0., None)]*h,
                     method='highs')
    if not second.success:
        raise RuntimeError('Minimum-L1 optimal-margin tie break failed: ' + second.message)
    weights = second.x[:h]; scores = mu @ weights
    gap = float(scores[expert_idx] - np.max(scores))
    y2 = second.ineqlin.marginals; z2 = second.eqlin.marginals
    secondary = dict(primal_violation=float(max(0., np.max(second_a @ second.x - second_b),
        np.max(np.abs(second_e @ second.x - 1.)), -np.min(second.x[h:]))),
        stationarity_residual=float(np.max(np.abs(second_c - second_a.T @ y2
            - second_e.T @ z2 - second.lower.marginals))),
        dual_sign_violation=float(max(0., np.max(y2), -np.min(second.lower.marginals))),
        duality_gap=float(abs(second.fun - (second_b @ y2 + z2[0]))),
        oracle_margin_gap=float(abs(gap - margin)))
    if max(secondary.values()) > tolerance:
        raise RuntimeError('Optimal-margin tie break primal/dual verification failed')
    return weights.tolist(), dict(converged=True, expert_representable=gap >= -tolerance,
        optimal_margin=margin, expert_minus_oracle=gap, primary_verification=primary,
        secondary_verification=secondary, weight_l1=float(np.abs(weights).sum()),
        maximum_absolute_weight=float(np.max(np.abs(weights))),
        oracle_includes_expert=True, reward_uniquely_identified=False,
        approximation='complete_finite_policy_max_margin_then_minimum_L1')


def learn_sum_only(statistics, horizons, config, reference_fit):
    """复用既有校准专家/资格规则，完全不读取验证或测试；保留所有阻断原因。

reference_fit 是 qualify_calibration 返回的专家、资格和前置检查。
不再运行已退出主线的盒约束拟合；成熟资格、并列顺序和完整矩阵保持原规则。
"""
    keys = ('expert', 'excluded_policies', 'eligible_policy_ids', 'expert_is_cash',
            'profitable_trading_expert', 'equal_weight_selected_policy',
            'cash_zero_vector_is_optimization_convention',
            'minimum_order_count_is_computability_not_statistical_sufficiency')
    base = {k: reference_fit[k] for k in keys if k in reference_fit}
    base.update(weight_constraint='sum_only', weights=None, diagnostics=None, selected_policy=None)
    if reference_fit['status'].startswith('blocked_'):
        return base | dict(status=reference_fit['status'])
    rows = [next(r for r in statistics if r['policy_id'] == p) for p in base['eligible_policy_ids']]
    mu = np.array([np.zeros(len(horizons)) if r['cash'] else r['order_feature_expectation'] for r in rows])
    expert_idx = base['eligible_policy_ids'].index(base['expert']['selected_policy_id'])
    try:
        weights, diagnostic = fit_sum_only(mu, expert_idx, tolerance=config['solver_tolerance'])
    except RuntimeError as error:
        return base | dict(status='solver_failed', error=str(error))
    response = best_with_ties(rows, mu @ weights, config['tie_tolerance_reward'])
    status = 'fitted' if diagnostic['expert_representable'] else 'fitted_expert_not_representable'
    return base | dict(status=status, weights=weights, diagnostics=diagnostic,
        candidate_response=response, selected_policy=response if status == 'fitted' else None,
        calibration_policy_scores=dict(zip(base['eligible_policy_ids'], map(float, mu @ weights))),
        reward_uniquely_identified=False)


class FrozenSumOnlyReward(IRLRewardLearner):
    """显式的新奖励类型：原始价格差、不扣成本、允许任意有限正负权且和为1。

此对象
只消费已冻结的校准权重；在线交易期间不能重新调用有界权重学习接口。
"""
    def __init__(self, horizons, weights):
        super().__init__(horizons=horizons, definition='paper_price_difference')
        value = np.asarray(weights, float)
        if (value.shape != (len(horizons),) or not np.isfinite(value).all()
                or abs(value.sum()-1.) > 1e-8):
            raise ValueError('Frozen Eq.(3) weights must be finite and sum to one')
        self.weights = value.copy(); self.weight_constraint = 'sum_only'

    def fit_reward_weights(self, *args, **kwargs):
        """阻止验证/测试期间重学，也不能误调用父类的额外盒约束。"""
        raise RuntimeError('Frozen calibration reward cannot be fitted online')
