"""只用历史校准结果审计 Eq.(3) 的线性可表示性，不发布新交易权重。

原文只要求 sum(w)=1。这里分别检查项目 simplex、signed_box 和仅等式
三种集合，区分附加约束造成的失败与当前特征/专家本身无法分离。
审计是有限策略几何检查，不代替 Algorithm 1 的参数优化或在线策略学习。
"""
import numpy as np
from scipy.optimize import linprog

from src.snapshot_irl import learn_calibration_reward

CONSTRAINTS = ('simplex', 'signed_box', 'sum_only')


def inequality_system(differences, names, constraint, reward_tolerance):
    """构造 A w <= b。D_j=mu_expert-mu_j，每行要求 D_j·w >= -tau。

    tau 是价格点容差，不能按美元费用改写。有限上下界也转成显式行，
    从而不可行证明可以说明究竟使用了哪些候选或额外边界。
    """
    h = differences.shape[1]
    rows = list(-differences);bounds = [reward_tolerance] * len(rows)
    labels = [dict(kind='expert_vs_policy', policy_id=name) for name in names]
    if constraint != 'sum_only':
        lower = 0. if constraint == 'simplex' else -1.
        for i, basis in enumerate(np.eye(h)):
            rows.extend([basis, -basis]);bounds.extend([1., -lower])
            labels.extend([dict(kind='weight_upper', horizon_index=i, value=1.),
                           dict(kind='weight_lower', horizon_index=i, value=lower)])
    return np.asarray(rows).reshape(-1, h), np.asarray(bounds), labels


def separation_audit(expectations, expert_index, policy_ids, constraint, *,
                     reward_tolerance=1e-8, verification_tolerance=1e-9,
                     solver_feasibility_tolerance=1e-9):
    """返回可直接重算的可行权重或不可行证书，求解状态不作为单独结论。

    输入 mu 为 P×H 原始价格差均值。可行问题最小化 sum(|w_i|)，避免仅等式
    集合上的最大间隔无界；最小 L1 只是诊断见证的工程消歧，不是原文目标。
    不可行时再求 Farkas 证书：y>=0，A.T@y+z*1=0，b@y+z=-1。
    若存在 w 满足 A w<=b、1·w=1，则左式给出 0<=-1 的矛盾。
    所有证书都复核残差；数值未通过就保留未验证，不能宣称已证明失败。
    """
    mu = np.asarray(expectations, dtype=float)
    tolerances = (reward_tolerance, verification_tolerance, solver_feasibility_tolerance)
    if (mu.ndim != 2 or min(mu.shape) < 1 or not np.isfinite(mu).all()
            or type(expert_index) is not int or not 0 <= expert_index < len(mu)
            or len(policy_ids) != len(mu) or len(set(policy_ids)) != len(mu)
            or any(not isinstance(name, str) or not name for name in policy_ids)
            or constraint not in CONSTRAINTS
            or any(type(t) not in (int, float) or not np.isfinite(t) or t <= 0 for t in tolerances)
            or not 1e-10 <= solver_feasibility_tolerance <= verification_tolerance):
        raise ValueError('Finite P×H matrix, expert identity and positive declared tolerances required')
    h = mu.shape[1]
    others = [i for i in range(len(mu)) if i != expert_index]
    differences = mu[expert_index] - mu[others]
    A, b, labels = inequality_system(differences, [policy_ids[i] for i in others], constraint, reward_tolerance)
    options = dict(primal_feasibility_tolerance=solver_feasibility_tolerance,
                   dual_feasibility_tolerance=solver_feasibility_tolerance)
    # x=[w,a]，a_i>=|w_i|；a 非负。sum(w)=1 是唯一共同的奖励权约束。
    absolute = np.vstack([np.c_[np.eye(h), -np.eye(h)], np.c_[-np.eye(h), -np.eye(h)]])
    rows = np.vstack([np.c_[A, np.zeros_like(A)], absolute])
    result = linprog(np.r_[np.zeros(h), np.ones(h)], A_ub=rows,
        b_ub=np.r_[b, np.zeros(2*h)], A_eq=[np.r_[np.ones(h), np.zeros(h)]], b_eq=[1.],
        bounds=[(None, None)]*h+[(0., None)]*h, method='highs', options=options)
    base = dict(constraint=constraint, expert_policy_id=policy_ids[expert_index],
        policy_ids=list(policy_ids), expectations=mu.tolist(), inequality_rows=labels,
        reward_tolerance_price=reward_tolerance, verification_tolerance=verification_tolerance,
        solver_feasibility_tolerance=solver_feasibility_tolerance,
        solver_status=int(result.status), solver_message=result.message,
        representable=None, witness=None, infeasibility_certificate=None,
        usable_for_execution=False, uniqueness_assessed=False,
        witness_objective='minimum_L1_feasibility_diagnostic_not_paper_IRL_objective')
    if result.success:
        if np.shape(result.x) != (2*h,) or not np.isfinite(result.x).all():
            return base | dict(status='numerically_unverified')
        w = result.x[:h];scores = mu @ w
        residual = max(0., float(np.max(A @ w-b))) if len(A) else 0.
        equality = abs(float(w.sum())-1.)
        valid = residual <= verification_tolerance and equality <= verification_tolerance
        witness = dict(weights=w.tolist(), weight_sum=float(w.sum()), L1=float(np.abs(w).sum()),
            max_abs_weight=float(np.abs(w).max()), policy_scores=dict(zip(policy_ids, map(float, scores))),
            expert_minus_best=float(scores[expert_index]-scores.max()),
            maximum_inequality_violation=residual, equality_residual=equality, verified=bool(valid))
        return base | dict(status='representable' if valid else 'numerically_unverified',
                           representable=True if valid else None, witness=witness)
    if result.status != 2:
        return base | dict(status='solver_failed')
    # 不用求解器一句 infeasible 充当研究证据。显式证书包括候选差值与盒边界。
    n = len(A)
    proof = linprog(np.r_[np.ones(n), 0.],
        A_eq=np.vstack([np.c_[A.T, np.ones(h)], np.r_[b, 1.]]),
        b_eq=np.r_[np.zeros(h), -1.], bounds=[(0., None)]*n+[(None, None)],
        method='highs', options=options)
    if not proof.success:
        return base | dict(status='infeasibility_unverified',
                           certificate_solver_status=int(proof.status), certificate_solver_message=proof.message)
    if np.shape(proof.x) != (n+1,) or not np.isfinite(proof.x).all():
        return base | dict(status='infeasibility_unverified')
    y, z = proof.x[:-1], float(proof.x[-1])
    stationarity = float(np.max(np.abs(A.T @ y+z)))
    contradiction = float(b @ y+z)
    nonnegative = max(0., float(-y.min()))
    valid = (stationarity <= verification_tolerance and nonnegative <= verification_tolerance
             and abs(contradiction+1.) <= verification_tolerance)
    certificate = dict(multipliers=y.tolist(), equality_multiplier=z,
        stationarity_residual=stationarity, nonnegative_violation=nonnegative,
        contradiction_rhs=contradiction, verified=bool(valid),
        proof='A.T@y+z*ones=0; y>=0; b@y+z=-1 contradicts Aw<=b and sum(w)=1')
    return base | dict(status='not_representable' if valid else 'infeasibility_unverified',
                       representable=False if valid else None, infeasibility_certificate=certificate)


def audit_calibration(statistics, horizons, irl_config, audit_config):
    """保持旧专家/资格/容差规则，三个约束使用完全相同的校准候选矩阵。

    不把未观测专家替换成次优交易策略；现金零向量是旧优化约定。全部同向或
    全零向量可几何匹配但缺少信息，旧 IRL 阻断仍保留。这里不读评价日数据。
    """
    fits = {name: learn_calibration_reward(statistics, horizons, irl_config, name)
            for name in ('simplex', 'signed_box')}
    original = fits['simplex']
    base = dict(expert=original['expert'], eligible_policy_ids=original['eligible_policy_ids'],
        excluded_policies=original['excluded_policies'], legacy_fits=fits,
        cash_zero_vector_is_optimization_convention=True, audits={}, usable_for_execution=False,
        original_expert_rule='maximum_settled_calibration_net_USD_no_replacement',
        feature_source='complete_matured_order_subset_not_all_period_orders')
    blocked = original['status'].startswith('blocked_') and original['status'] != 'blocked_uninformative_expectations'
    if blocked:
        return base | dict(status=original['status'], diagnosis='insufficient_observation_not_geometric_failure')
    rows = [next(r for r in statistics if r['policy_id'] == name) for name in base['eligible_policy_ids']]
    mu = [np.zeros(len(horizons)).tolist() if r['cash'] else r['order_feature_expectation'] for r in rows]
    expert = base['eligible_policy_ids'].index(base['expert']['selected_policy_id'])
    audits = {name: separation_audit(mu, expert, base['eligible_policy_ids'], name,
        reward_tolerance=irl_config['solver_tolerance'],
        verification_tolerance=audit_config['verification_tolerance'],
        solver_feasibility_tolerance=audit_config['solver_feasibility_tolerance']) for name in CONSTRAINTS}
    possible = [audits[name]['representable'] for name in CONSTRAINTS]
    diagnosis = ('numerical_or_solver_failure' if any(p is None for p in possible) else
        'uninformative_expectations' if original['status'] == 'blocked_uninformative_expectations' else
        'not_representable_even_with_sum_only' if not possible[-1] else
        'signed_box_bound_excludes_witness' if not possible[1] else
        'nonnegative_constraint_excludes_witness' if not possible[0] else
        'representable_in_all_declared_sets')
    return base | dict(status='audited', audits=audits, diagnosis=diagnosis)
