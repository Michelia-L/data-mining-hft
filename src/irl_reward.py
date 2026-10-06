"""论文 Eq.(2)–(3) 的原始中间价差与线性评分，单位为价格点。

时间单位由真实时间引擎解释为毫秒；价格最小跳动、网格、前瞻与持仓分开。
成本只进美元账本，不在奖励中扣除。卖单方向取反是项目的明示工程扩展。
"""
import numpy as np

class IRLRewardLearner:
    """保存已声明的七尺度与等权评分；学习拟合由 sum_only_reward 负责。"""
    def __init__(self, horizons, *, definition='paper_price_difference'):
        """horizons 为递增正整数毫秒；等权是人工对照，不声称经过 IRL。"""
        if (definition != 'paper_price_difference' or not horizons
                or any(type(h) is not int or h <= 0 for h in horizons)
                or list(horizons) != sorted(set(horizons))):
            raise ValueError('Need ordered positive horizons and raw price differences')
        self.horizons = list(horizons)
        self.definition = definition
        self.deduct_cost = False
        self.weights = np.ones(len(horizons)) / len(horizons)
        self.weight_constraint = 'equal'

    def features_from_prices(self, reference_price, future_prices, side=1, cost_ratio=0.):
        """Eq.(2)：方向×(未来中间价−起点中间价)，不归一化、不裁剪。"""
        return side * (np.asarray(future_prices) - reference_price)

    def score(self, features):
        """Eq.(3)：w·u；完整向量只能在最长前瞻期成熟后评分。"""
        return float(np.dot(self.weights, features))
