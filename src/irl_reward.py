"""
逆强化学习 (IRL) 多尺度奖励优化模块 (irl_reward.py)
对应论文第 3.3 节与 Algorithm 1：
通过特征期望匹配 (Feature Expectation Matching) 求解多尺度周期最优权重向量 w*，
将短周期 (10 ticks)、中周期 (30 ticks) 与长周期 (90 ticks) 融合成兼顾灵敏度与抗噪性的综合奖励函数。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from src.config import HORIZONS


class IRLRewardLearner:
    """
    逆强化学习多尺度奖励学习器
    对应论文公式 (3) 与 (4)：
        E(O) = sum_{k} w_k * E_{T_k}, T = [10, 30, 90]
        sum w_k = 1, w_k >= 0
    """
    def __init__(self, horizons: List[int] = HORIZONS):
        self.horizons = horizons
        self.num_scales = len(horizons)
        # 初始化权重：默认等权 [1/3, 1/3, 1/3]
        self.weights = np.ones(self.num_scales) / self.num_scales
        self.expert_expectation = None
        self.random_expectation = None

    def fit_reward_weights(self, train_df: pd.DataFrame, max_iter: int = 50, lr: float = 0.05) -> np.ndarray:
        """
        基于训练集回测轨迹，通过特征期望匹配优化多尺度权重 w* (Algorithm 1)

        参数:
            train_df (pd.DataFrame): 包含各尺度 future_ret_{h} 的训练集
            max_iter (int): 迭代优化轮数
            lr (float): 学习率

        返回:
            np.ndarray: 归一化后的最优权重向量 w*
        """
        print(f"[IRLReward] 开始执行逆强化学习 (IRL) 奖励函数优化 (尺度: {self.horizons})...")
        
        # 1. 计算各尺度的未来期望矩阵 (N, 3)
        ret_cols = [f'future_ret_{h}' for h in self.horizons]
        ret_matrix = train_df[ret_cols].values
        
        # 2. 构建专家策略 (Expert Policy)：选取未来趋势显著且一致的高置信样本
        # 专家策略：在多尺度收益均显著正向或反向时才开仓的方向准确策略
        composite_trend = np.mean(ret_matrix, axis=1)
        expert_mask = np.abs(composite_trend) > np.percentile(np.abs(composite_trend), 85)
        
        if np.sum(expert_mask) < 50:
            expert_mask = np.ones(len(train_df), dtype=bool)
            
        expert_signals = np.sign(composite_trend[expert_mask])
        # 专家策略特征期望 u_E = E[action * future_returns]
        self.expert_expectation = np.mean(
            ret_matrix[expert_mask] * expert_signals[:, None], axis=0
        )
        
        # 3. 随机/基线策略特征期望 u_0
        random_signals = np.random.choice([-1.0, 1.0], size=len(train_df))
        self.random_expectation = np.mean(
            ret_matrix * random_signals[:, None], axis=0
        )
        
        print(f"  - 专家策略多尺度特征期望 u_E: {np.round(self.expert_expectation * 1e4, 2)} (bps)")
        print(f"  - 基线随机策略特征期望 u_R: {np.round(self.random_expectation * 1e4, 2)} (bps)")
        
        # 4. 指数梯度投影优化 (Exponentiated Gradient / Softmax IRL)
        # 求解使得专家策略相对于基线策略在多尺度回报差额最大的 w*，且满足非负与归一化约束
        log_w = np.zeros(self.num_scales)
        delta_u = self.expert_expectation - self.random_expectation
        
        # 保证差分具有辨识度
        delta_u = np.maximum(delta_u, 1e-6)
        
        for iteration in range(max_iter):
            # Softmax 投影确保 sum(w) = 1 且 w_i > 0
            w = np.exp(log_w) / np.sum(np.exp(log_w))
            # 目标梯度：使得权重更偏向于专家区分度更强、信号噪声比更优的尺度
            gradient = delta_u / (np.linalg.norm(delta_u) + 1e-8)
            log_w += lr * gradient
            
        self.weights = np.exp(log_w) / np.sum(np.exp(log_w))
        
        # 格式化打印最终学得的多尺度权重
        weight_str = ", ".join([f"{h}t: {w:.3f}" for h, w in zip(self.horizons, self.weights)])
        print(f"[IRLReward] IRL 最优权重求解完成: [{weight_str}]")
        return self.weights

    def compute_model_expectation(self, pred: float, future_rets: np.ndarray) -> float:
        """
        计算模型预测期望 (Model Prediction Expectation, ME)
        对应论文第 3.3.1 节：评估模型信号方向与未来多尺度实际价格变动的一致性

        参数:
            pred (float): 模型预测收益率 (符号代表方向)
            future_rets (np.ndarray): 实际各尺度的未来收益率 [R_10, R_30, R_90]

        返回:
            float: ME 奖励得分
        """
        direction = np.sign(pred)
        # 多尺度加权期望: sum(w_k * (direction * R_k))
        weighted_ret = np.sum(self.weights * (direction * future_rets))
        return float(weighted_ret)

    def compute_order_traded_expectation(
        self,
        order_side: int,
        future_rets: np.ndarray,
        rel_spread: float,
        cost_ratio: float = 0.0001
    ) -> float:
        """
        计算订单成交期望 (Order Traded Expectation, OE)
        对应论文第 3.3.2 节：结合实际交易撮合成本（点差与手续费）后的实盘期望

        参数:
            order_side (int): 开仓方向 (+1 买入, -1 卖出, 0 不操作)
            future_rets (np.ndarray): 实际各尺度的未来收益率 [R_10, R_30, R_90]
            rel_spread (float): 当前盘口相对买卖价差
            cost_ratio (float): 单边手续费率折算

        返回:
            float: OE 奖励得分
        """
        if order_side == 0:
            return 0.0
            
        # 毛收益加权期望
        gross_expectation = np.sum(self.weights * (order_side * future_rets))
        # 扣除半点差滑点与手续费成本
        friction = (rel_spread * 0.5) + cost_ratio
        net_expectation = gross_expectation - friction
        return float(net_expectation)
