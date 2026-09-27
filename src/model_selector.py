"""
动态模型在线自适应选择器模块 (model_selector.py)
对应论文第 3.4 节与 Algorithm 2, Algorithm 3：
实现基于多臂老虎机 (Multi-Armed Bandit) 的 UCB 算法与滑动窗口平均奖励 (ARS) 算法，
支持 4 种论文核心 FMATO 变体与 3 个基线策略。
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from collections import deque
from src.config import RL_CONFIG
from src.model_library import LightModelBase
from src.irl_reward import IRLRewardLearner


class BaseSelector(ABC):
    """
    在线模型选择器抽象基类
    """
    def __init__(self, name: str, models: List[LightModelBase], reward_type: str = "ME"):
        """
        参数:
            name (str): 策略名称
            models (List[LightModelBase]): 动作空间中的轻量模型库
            reward_type (str): 奖励类型 ('ME': 模型预测期望, 'OE': 订单成交期望)
        """
        self.name = name
        self.models = models
        self.k = len(models)
        self.reward_type = reward_type
        self.action_history = []
        self.reward_history = []

    @abstractmethod
    def select_model(self, step: int) -> int:
        """
        在当前 tick 选择一个模型索引 (0 到 K-1)
        """
        pass

    @abstractmethod
    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        """
        根据反馈的奖励更新选择器状态
        """
        pass

    def get_selection_distribution(self) -> Dict[str, float]:
        """
        统计各模型被选中的次数比例
        """
        if not self.action_history:
            return {m.name: 0.0 for m in self.models}
        counts = np.bincount(self.action_history, minlength=self.k)
        total = len(self.action_history)
        return {self.models[i].name: float(counts[i] / total) for i in range(self.k)}


class UCBSelector(BaseSelector):
    """
    UCB 在线模型选择器 (Algorithm 2)
    Q(a) = W(a) + C * sqrt(2 * ln(N) / n(a))
    在充分利用高期望模型的同时，对探索较少的模型施加不确定性溢价，适应非平稳高频市场。
    """
    def __init__(self, name: str, models: List[LightModelBase], reward_type: str = "ME", c: float = RL_CONFIG["ucb_c"]):
        super().__init__(name, models, reward_type)
        self.c = c
        self.counts = np.zeros(self.k, dtype=int)
        self.sum_rewards = np.zeros(self.k, dtype=float)
        self.total_steps = 0

    def select_model(self, step: int) -> int:
        self.total_steps += 1
        # 冷启动阶段：若存在从未被尝试的模型，优先探索
        for idx in range(self.k):
            if self.counts[idx] == 0:
                self.action_history.append(idx)
                return idx

        # 计算各模型的平均收益 W(a) 与置信区间探索奖励
        avg_rewards = self.sum_rewards / self.counts
        bonus = self.c * np.sqrt(2.0 * np.log(self.total_steps) / self.counts)
        ucb_scores = avg_rewards + bonus

        # 选取 UCB 分数最高的模型
        chosen_idx = int(np.argmax(ucb_scores))
        self.action_history.append(chosen_idx)
        return chosen_idx

    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        self.counts[chosen_idx] += 1
        self.sum_rewards[chosen_idx] += reward
        self.reward_history.append(reward)


class ARSSelector(BaseSelector):
    """
    平均奖励选择器 (ARS, Algorithm 3)
    利用滑动窗口 (Sliding Window) 持续回放评估所有候选模型的近期平均奖励，贪心选择当前最优臂。
    """
    def __init__(self, name: str, models: List[LightModelBase], reward_type: str = "ME", window_size: int = RL_CONFIG["ars_window"]):
        super().__init__(name, models, reward_type)
        self.window_size = window_size
        # 记录每个模型在滑动窗口内的近期奖励队列
        self.reward_queues = [deque(maxlen=window_size) for _ in range(self.k)]

    def select_model(self, step: int) -> int:
        # 冷启动阶段：轮流采样
        if step < self.k:
            chosen_idx = step % self.k
            self.action_history.append(chosen_idx)
            return chosen_idx

        # 计算窗口内各模型的平均奖励得分
        avg_scores = [
            np.mean(q) if len(q) > 0 else 0.0
            for q in self.reward_queues
        ]
        chosen_idx = int(np.argmax(avg_scores))
        self.action_history.append(chosen_idx)
        return chosen_idx

    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        # 论文特色：回测子系统可并行对未选中模型进行影子回测，更新全量窗口奖励
        if all_rewards is not None and len(all_rewards) == self.k:
            for i in range(self.k):
                self.reward_queues[i].append(all_rewards[i])
        else:
            self.reward_queues[chosen_idx].append(reward)
            
        self.reward_history.append(reward)


class SingleModelSelector(BaseSelector):
    """
    基线 1：固定单模型策略 (始终使用指定的固定最优模型，如 Ridge 或 GBDT)
    """
    def __init__(self, name: str, models: List[LightModelBase], fixed_idx: int = 0):
        super().__init__(name, models, reward_type="ME")
        self.fixed_idx = fixed_idx

    def select_model(self, step: int) -> int:
        self.action_history.append(self.fixed_idx)
        return self.fixed_idx

    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        self.reward_history.append(reward)


class EnsembleSelector(BaseSelector):
    """
    基线 2：静态等权集成基线 (在预测层融合全部模型)
    """
    def __init__(self, name: str, models: List[LightModelBase]):
        super().__init__(name, models, reward_type="ME")

    def select_model(self, step: int) -> int:
        # 虚拟选择，实际在回测执行中取全模型预测平均
        self.action_history.append(0)
        return 0

    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        self.reward_history.append(reward)


class RandomSelector(BaseSelector):
    """
    基线 3：随机动作基线 (每个 tick 随机选择模型，检验强化学习自适应有效性)
    """
    def __init__(self, name: str, models: List[LightModelBase]):
        super().__init__(name, models, reward_type="ME")

    def select_model(self, step: int) -> int:
        idx = int(np.random.randint(0, self.k))
        self.action_history.append(idx)
        return idx

    def update_reward(self, chosen_idx: int, reward: float, all_rewards: Optional[np.ndarray] = None) -> None:
        self.reward_history.append(reward)


def build_all_selectors(models: List[LightModelBase]) -> List[BaseSelector]:
    """
    构建论文对比实验中的 7 个策略选择器：
    - 4 个 FMATO 核心变体：
      1. FMATO-ME-UCBS (模型期望 + UCB)
      2. FMATO-OE-UCBS (成交期望 + UCB)
      3. FMATO-ME-ARS  (模型期望 + ARS)
      4. FMATO-OE-ARS  (成交期望 + ARS)
    - 3 个对照基准：
      5. Baseline-Single (单一最佳基模型)
      6. Baseline-Ensemble (静态等权集成)
      7. Baseline-Random (随机动作基准)
    """
    return [
        UCBSelector("FMATO-ME-UCBS", models, reward_type="ME"),
        UCBSelector("FMATO-OE-UCBS", models, reward_type="OE"),
        ARSSelector("FMATO-ME-ARS", models, reward_type="ME"),
        ARSSelector("FMATO-OE-ARS", models, reward_type="OE"),
        SingleModelSelector("Baseline-Single-HistGBDT", models, fixed_idx=3), # 选择 GBDT 作为单模型代表
        EnsembleSelector("Baseline-Static-Ensemble", models),
        RandomSelector("Baseline-Random", models),
    ]
