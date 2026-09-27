"""
轻量模型库模块 (model_library.py)
对应论文第 3.2 节：构建多样化的轻量级模型作为强化学习的动作空间 (Action Space)。
涵盖线性回归、分类判别、浅层决策树、梯度提升树及启发式动量规则。
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


class LightModelBase(ABC):
    """
    轻量模型抽象基类
    统一规范各模型的训练、预测及推理耗时统计接口
    """
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.latency_us = 0.0 # 单次推理延迟 (微秒 us)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """模型拟合训练"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """收益率连续预测"""
        pass

    def evaluate_latency(self, X_sample: np.ndarray) -> float:
        """测算单样本推理耗时 (微秒)"""
        start = time.perf_counter()
        _ = self.predict(X_sample)
        duration = time.perf_counter() - start
        self.latency_us = (duration / len(X_sample)) * 1e6
        return self.latency_us


class RidgeModel(LightModelBase):
    """
    1. Ridge 岭回归模型
    通过 L2 正则化控制权重幅度，擅长捕捉特征与未来收益之间的线性趋势关系。
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__("Ridge_Linear")
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class LogisticDirectionModel(LightModelBase):
    """
    2. 逻辑回归方向判别模型
    将收益预测转化为多空分类任务，输出带有置信度偏向的期望值。
    """
    def __init__(self, C: float = 1.0):
        super().__init__("Logistic_Direction")
        self.model = LogisticRegression(C=C, max_iter=200, random_state=42)
        self.std_scale = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 将连续收益二值化为涨跌方向标签: 1 (涨) 与 0 (跌)
        y_binary = (y > 0).astype(int)
        self.model.fit(X, y_binary)
        self.std_scale = float(np.std(y)) if np.std(y) > 0 else 0.001
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 输出净胜率偏移量乘以历史收益波动尺度
        proba = self.model.predict_proba(X)
        prob_up = proba[:, 1]
        # (prob_up - 0.5) 映射到期望收益幅度
        return (prob_up - 0.5) * 2.0 * self.std_scale


class DecisionTreeModel(LightModelBase):
    """
    3. 浅层决策树模型 (Decision Tree)
    论文重点提及的基模型之一，通过分段树状判定捕获非线性分位数与市场状态突变。
    """
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 50):
        super().__init__("Decision_Tree")
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class HistGBDTModel(LightModelBase):
    """
    4. 直方图梯度提升树模型 (HistGBDT)
    主流现代高频数据挖掘中高效的非线性集成模型，擅长挖掘高阶交叉特征。
    """
    def __init__(self, max_iter: int = 30, max_depth: int = 3):
        super().__init__("Hist_GBDT")
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class MomentumRuleModel(LightModelBase):
    """
    5. 启发式盘口动量规则模型
    基准模型：直接结合短期订单簿失衡度 (OBI) 与滞后动量发出信号，无需梯度拟合。
    """
    def __init__(self):
        super().__init__("Rule_Momentum")
        self.scale = 0.0005

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 仅根据样本方差估计输出尺度
        self.scale = float(np.std(y)) if len(y) > 0 else 0.0005
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 特征索引定义 (参考 data_loader):
        # index 2: obi_l1 (一级失衡)
        # index 6: ret_lag_5 (5-tick 动量)
        obi = X[:, 2] if X.shape[1] > 2 else np.zeros(len(X))
        mom = X[:, 6] if X.shape[1] > 6 else np.zeros(len(X))
        combined = 0.6 * obi + 0.4 * mom
        return np.clip(combined * self.scale, -3 * self.scale, 3 * self.scale)


def build_model_library() -> List[LightModelBase]:
    """
    实例化轻量模型库
    包含 5 个互补的模型作为动作空间
    """
    return [
        RidgeModel(alpha=5.0),
        LogisticDirectionModel(C=1.0),
        DecisionTreeModel(max_depth=4, min_samples_leaf=50),
        HistGBDTModel(max_iter=30, max_depth=3),
        MomentumRuleModel(),
    ]


def train_model_library(models: List[LightModelBase], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
    """
    批量训练模型库中的所有模型，并记录各模型的推理耗时
    """
    print(f"[ModelLibrary] 开始训练模型库 (包含 {len(models)} 个轻量候选模型)...")
    latencies = {}
    for model in models:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0
        latency = model.evaluate_latency(X_train[:1000])
        latencies[model.name] = latency
        print(f"  - [{model.name}] 训练完成 (耗时: {fit_time:.2f}s, 推理延迟: {latency:.2f} us/tick)")
    return latencies
