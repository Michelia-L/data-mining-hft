"""多尺度奖励学习：论文 Algorithm 1 的有限可执行策略近似。

一般强化学习给定奖励后寻找好策略；逆强化学习则观察专家行为，尝试反推出
能解释专家的奖励。本项目先在校准段选出净盈亏最好的可执行策略作为专家，
再交替求奖励权重与候选库中的最优响应。专家也可能是不交易的现金策略。
这里用有限策略库代替论文未公开的生产参数优化器，不声称奖励被唯一识别，
也不把线性规划收敛等同于学到了可盈利的专家行为。"""
import numpy as np
from scipy.optimize import linprog
from src.config import HORIZON_EVENTS


class IRLRewardLearner:
    """保存奖励定义、可选归一化参数、权重约束及学习诊断。

    价格差模式与归一化收益率模式显式分开；成本扣除默认关闭。
    ME 与 OE 分别创建实例，以免两种评价口径共享或覆盖权重。"""
    def __init__(self, horizons=HORIZON_EVENTS, weights=None, scales=None,
                 definition="normalized_return", deduct_cost=False, weight_constraint="simplex"):
        """初始化 H 个时域的权重与尺度。

        weights=None 时使用等权；simplex 要求非负，signed_box 允许负值；归一化为和为 1。
        scales 必须逐项为正；默认全 1 适合手工设定，正式实验调用 fit_scales 拟合。"""
        # Eq.(2) 是价格差；归一化收益率是工程变体，必须显式记录，不能混为同一单位。
        if definition not in ('paper_price_difference', 'normalized_return'):
            raise ValueError('Unknown reward definition')
        if weight_constraint not in ('simplex', 'signed_box'):
            raise ValueError('Unknown weight constraint')
        self.definition, self.deduct_cost = definition, deduct_cost
        self.weight_constraint = weight_constraint
        self.horizons = list(horizons)
        self.weights = np.ones(len(horizons)) / len(horizons) if weights is None else np.asarray(weights, float)
        self.scales = np.ones(len(horizons)) if scales is None else np.asarray(scales, float)
        if (self.weights.shape != (len(horizons),) or self.scales.shape != self.weights.shape
                or not np.isfinite(self.weights).all() or not np.isfinite(self.scales).all()
                or (weight_constraint == "simplex" and (self.weights < 0).any())
                or (self.scales <= 0).any() or self.weights.sum() <= 0):
            raise ValueError('Invalid reward weights/scales')
        self.weights = self.weights / self.weights.sum()
        if weight_constraint == 'signed_box' and (np.abs(self.weights) > 1. + 1e-12).any():
            raise ValueError('Normalized signed weights must lie in [-1, 1]')
        self.diagnostics = {}

    def fit_scales(self, train_df):
        """仅用已清除跨边界标签的训练段估计每个时域的收益幅度。

        取绝对收益的 99% 分位数，使较长时域不会仅因波动幅度大而占据优势。
        下限 1e-8 防止全零行情导致除零；这里的尺度是幅度校准，不是概率或显著性。"""
        if self.definition == 'paper_price_difference':
            self.scales = np.ones(len(self.horizons))
            return self.scales
        returns = train_df[[f'future_ret_{h}' for h in self.horizons]].to_numpy()
        if not np.isfinite(returns).all():
            raise ValueError('Training labels must be finite and purged')
        self.scales = np.maximum(np.quantile(np.abs(returns), .99, axis=0), 1e-8)
        return self.scales

    def fit_reward_weights(self, expectations, expert_idx, policy_names=None, tolerance=1e-8, max_iter=50):
        """通过最大间隔线性规划与有限策略最优响应，迭代拟合奖励权重。

        参数 expectations 是 P×H 矩阵：P 个候选策略、H 个时域的平均特征期望；
        expert_idx 是校准段选出的专家行号，policy_names 用于在报告中解释该行号。
        每轮解 max_w min_j w·(mu_expert-mu_j)，sum(w)=1；符号由 weight_constraint 决定。
        随后在全部候选策略中选 argmax(mu_j·w)，将这个最强对手加入约束再求解。
        当已有约束的最大间隔上界与完整策略库的实际间隔相差不超过 tolerance 时停止。
        返回权重；diagnostics 另记录求解收敛与专家可表示性，两者必须分开解读。"""
        mu = np.asarray(expectations, float)
        if mu.ndim != 2 or mu.shape[1] != len(self.horizons) or not np.isfinite(mu).all():
            raise ValueError('Invalid policy feature expectations')
        expert = mu[expert_idx]
        # 从等权策略混合的期望开始，避免随机初始化影响可重复性。
        visited = [mu.mean(axis=0)]
        history, converged = [], False
        for iteration in range(max_iter):
            differences = expert - np.asarray(visited)
            # 决策变量为 [w_1,...,w_H,margin]。linprog 默认最小化，因此目标取 -margin。
            # 每个约束 -difference·w + margin <= 0，即 margin 不得超过专家领先幅度。
            # A_eq 保留论文权重和约束；bounds 是本项目单纯形或有界负权约束。
            result = linprog(np.r_[np.zeros(len(self.horizons)), -1.],
                             A_ub=np.c_[-differences, np.ones(len(visited))],
                             b_ub=np.zeros(len(visited)),
                             A_eq=np.array([np.r_[np.ones(len(self.horizons)), 0.]]),
                             b_eq=[1.], bounds=([(0., 1.)] if self.weight_constraint == "simplex" else [(-1., 1.)]) * len(self.horizons) + [(None, None)],
                             method='highs')
            if not result.success:
                raise RuntimeError(f'Reward separation failed: {result.message}')
            # 仅在非负模式清理微小负数；有符号模式保留负权，再确保和为 1。
            self.weights = (np.maximum(result.x[:-1], 0.) if self.weight_constraint == "simplex"
                            else result.x[:-1].copy())
            self.weights /= self.weights.sum()
            # 这里的 oracle 只是遍历有限策略库的精确最优响应，不是偷看未来的交易者。
            scores = mu @ self.weights
            best_idx = int(np.argmax(scores))
            margin = float(result.x[-1])
            oracle_gap = float(expert @ self.weights - scores[best_idx])
            history.append(dict(iteration=iteration, margin=margin, oracle_policy=best_idx,
                                expert_minus_oracle=oracle_gap, weights=self.weights.tolist()))
            # 约束子集的最优间隔是上界，完整库的实际间隔是当前权重的下界。
            # 两者相等才说明无需再加约束；即使它们均为负，也可能正常收敛。
            if margin - oracle_gap <= tolerance:
                converged = True
                break
            visited.append(mu[best_idx])
        # 专家可表示要求其得分不低于库内最优策略；负间隔应如实暴露，不能宣传成成功匹配。
        self.diagnostics = dict(method='finite_policy_apprenticeship', converged=converged,
                                expert_index=int(expert_idx), policy_names=policy_names,
                                weight_constraint=self.weight_constraint, definition=self.definition,
                                deduct_cost=self.deduct_cost,
                                expert_representable=bool(history[-1]['expert_minus_oracle'] >= -tolerance),
                                final_margin=history[-1]['margin'],
                                expectations=mu.tolist(), history=history,
                                note='Finite executable policy library; reward is not uniquely identified.')
        return self.weights

    def features(self, future_returns, side=1, cost_ratio=0., reference_price=None):
        """从相对收益构造多尺度特征；支持批量 N×H 输入。

        paper_price_difference 还原 Eq.(2) 的绝对价格差，不归一化、不裁剪。
        side 对空头取反是项目的方向扩展，论文未完整规定卖单符号处理。
        默认 PaperOE 不扣成本；deduct_cost=True 才是明确命名的 NetOE 变体。
        账本在所有模式下均独立扣实际费用，不受奖励定义影响。
        """
        value = side * np.asarray(future_returns) - (cost_ratio if self.deduct_cost else 0.)
        if self.definition == 'paper_price_difference':
            if reference_price is None:
                raise ValueError('Price-difference reward requires reference_price')
            return value * reference_price
        return value / self.scales

    def features_from_prices(self, reference_price, future_prices, side=1, cost_ratio=0.):
        """直接按价格计算，避免 Eq.(2) 先除再乘引入不必要的舍入误差。"""
        future_prices = np.asarray(future_prices)
        if self.definition == 'paper_price_difference':
            return side * (future_prices - reference_price) - (
                cost_ratio * reference_price if self.deduct_cost else 0.)
        return self.features(future_prices / reference_price - 1., side, cost_ratio)

    def score(self, features):
        """Eq.(3) 线性加权；仅归一化变体裁剪到 [-1,1]。

        价格差模式的探索系数带价格单位，不能把跨品种相同 c 当成同强度探索。
        sum(w)=1 是论文明确约束；非负单纯形和 signed_box 的 [-1,1] 边界都是
        项目为有限策略求解增加的约束，后者仅用于敏感性检查。
        """
        value = float(np.dot(self.weights, features))
        return float(np.clip(value, -1., 1.)) if self.definition == 'normalized_return' else value
