"""Finite-policy apprenticeship approximation to paper Algorithm 1.

The expert is an executable policy selected on calibration PnL, never an oracle
that observes future test prices. A separation LP and a finite policy-improvement
oracle alternate until the separation bound matches the oracle. This is not the unpublished
production optimizer, nor evidence that the recovered reward is unique.
"""
import numpy as np
from scipy.optimize import linprog
from src.config import HORIZONS


class IRLRewardLearner:
    def __init__(self, horizons=HORIZONS, weights=None, scales=None):
        self.horizons = list(horizons)
        self.weights = np.ones(len(horizons)) / len(horizons) if weights is None else np.asarray(weights, float)
        self.scales = np.ones(len(horizons)) if scales is None else np.asarray(scales, float)
        if (self.weights.shape != (len(horizons),) or self.scales.shape != self.weights.shape
                or not np.isfinite(self.weights).all() or not np.isfinite(self.scales).all()
                or (self.weights < 0).any() or (self.scales <= 0).any() or self.weights.sum() <= 0):
            raise ValueError('Invalid reward weights/scales')
        self.weights = self.weights / self.weights.sum()
        self.diagnostics = {}

    def fit_scales(self, train_df):
        returns = train_df[[f'future_ret_{h}' for h in self.horizons]].to_numpy()
        if not np.isfinite(returns).all():
            raise ValueError('Training labels must be finite and purged')
        self.scales = np.maximum(np.quantile(np.abs(returns), .99, axis=0), 1e-8)
        return self.scales

    def fit_reward_weights(self, expectations, expert_idx, policy_names=None, tolerance=1e-8, max_iter=50):
        """Expectations are per-event averages of normalized signal/order features.

        At each iteration re-solve max_w min_j w.(mu_expert - mu_j), then
        improve the policy by maximizing w.mu over the complete finite library.
        Starting from a uniform policy mixture makes initialization deterministic.
        """
        mu = np.asarray(expectations, float)
        if mu.ndim != 2 or mu.shape[1] != len(self.horizons) or not np.isfinite(mu).all():
            raise ValueError('Invalid policy feature expectations')
        expert = mu[expert_idx]
        visited = [mu.mean(axis=0)]
        history, converged = [], False
        for iteration in range(max_iter):
            differences = expert - np.asarray(visited)
            # Variables [w_1, ..., w_K, margin]; maximize margin on the simplex.
            result = linprog(np.r_[np.zeros(len(self.horizons)), -1.],
                             A_ub=np.c_[-differences, np.ones(len(visited))],
                             b_ub=np.zeros(len(visited)),
                             A_eq=np.array([np.r_[np.ones(len(self.horizons)), 0.]]),
                             b_eq=[1.], bounds=[(0., 1.)] * len(self.horizons) + [(None, None)],
                             method='highs')
            if not result.success:
                raise RuntimeError(f'Reward separation failed: {result.message}')
            self.weights = np.maximum(result.x[:-1], 0.)
            self.weights /= self.weights.sum()
            scores = mu @ self.weights
            best_idx = int(np.argmax(scores))
            margin = float(result.x[-1])
            oracle_gap = float(expert @ self.weights - scores[best_idx])
            history.append(dict(iteration=iteration, margin=margin, oracle_policy=best_idx,
                                expert_minus_oracle=oracle_gap, weights=self.weights.tolist()))
            # Restricted max-margin upper bound agrees with the full-library oracle.
            if margin - oracle_gap <= tolerance:
                converged = True
                break
            visited.append(mu[best_idx])
        self.diagnostics = dict(method='finite_policy_apprenticeship', converged=converged,
                                expert_index=int(expert_idx), policy_names=policy_names,
                                expert_representable=bool(history[-1]['expert_minus_oracle'] >= -tolerance),
                                final_margin=history[-1]['margin'],
                                expectations=mu.tolist(), history=history,
                                note='Finite executable policy library; reward is not uniquely identified.')
        return self.weights

    def features(self, future_returns, side=1, cost_ratio=0.):
        return (side * np.asarray(future_returns) - cost_ratio) / self.scales

    def score(self, features):
        # Bounded feedback gives UCB exploration and exploitation compatible units.
        return float(np.clip(np.dot(self.weights, features), -1., 1.))
