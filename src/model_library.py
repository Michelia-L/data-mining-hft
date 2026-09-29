"""
轻量模型库模块 (model_library.py)
受论文第 3.2 节启发，但当前只是相同特征、相同训练期上的异构算法库。
尚未实现论文通过不同特征集/历史时期构造市场分布候选、每周更新的模型库。
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
        """分别测量单条预测延迟与批量均摊耗时，单位均为微秒。

        先预热 10 次，再逐条预测 200 次，保存 P50（中位数）和 P95（较慢尾部）。
        随后一次预测整批样本并除以批量大小，得到吞吐口径的每行均摊耗时。
        批量向量化可以摊薄函数调用开销，所以该值不能冒充在线逐条推理延迟。
        这里只计模型 predict，不包含特征工程、模型选择、撮合或网络传输。"""
        one = X_sample[:1]
        for _ in range(10):
            self.predict(one)
        samples = []
        for _ in range(200):
            start = time.perf_counter_ns()
            self.predict(one)
            samples.append((time.perf_counter_ns() - start) / 1000)
        start = time.perf_counter_ns()
        self.predict(X_sample)
        batch_us = (time.perf_counter_ns() - start) / 1000 / len(X_sample)
        self.latency_us = float(np.median(samples))
        self.latency = dict(single_p50_us=self.latency_us,
                            single_p95_us=float(np.percentile(samples, 95)),
                            batch_us_per_row=batch_us, repeats=200)
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
    显式区分跌、平、涨，用训练集各类别平均收益校准期望值。
    """
    def __init__(self, C: float = 1.0):
        super().__init__("Logistic_Direction")
        self.model = LogisticRegression(C=C, max_iter=200, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 先分方向，再校准幅度；输出仍是连续收益率，便于统一比较开仓阈值。
        """将训练收益映射为跌(-1)、平(0)、涨(+1)，并计算每类的训练平均收益。

        高频中平盘样本常很多，不能直接并入“跌”。若训练段只有一个类别，
        逻辑回归无法拟合，则保存常数收益预测；整个处理都不使用测试标签。"""
        labels = np.sign(y).astype(int)
        self.class_returns = {int(c): float(y[labels == c].mean()) for c in np.unique(labels)}
        self.constant = float(y.mean()) if len(self.class_returns) == 1 else None
        if self.constant is None:
            self.model.fit(X, labels)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """用类别概率乘以训练类别平均收益，转换为与其他模型一致的收益率预测。

        predict_proba 的列顺序由 model.classes_ 决定，必须按它取对应收益，
        不能假定概率矩阵总是有三个固定位置的类别。"""
        if self.constant is not None:
            return np.full(len(X), self.constant)
        values = np.array([self.class_returns[int(c)] for c in self.model.classes_])
        return self.model.predict_proba(X) @ values

    def predict_class(self, X):
        """输出分类器概率最大类别；不能用期望收益的正负冒充分类器 argmax。

        例如平盘概率最大，但较小上涨概率乘上较大涨幅后，期望收益仍可为正。
        单类别训练的回退与连续预测保持同一训练类别。
        """
        if self.constant is not None:
            return np.full(len(X), next(iter(self.class_returns)), dtype=int)
        return self.model.predict(X)


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
            # 禁用模型内部自动验证划分，由外部时间切分统一管理训练与验证。
            early_stopping=False,
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
        # 仅根据训练收益的标准差估计输出尺度，不学习未来测试段的波动率。
        self.scale = float(np.std(y)) if len(y) > 0 else 0.0005
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 特征索引定义 (参考 data_loader):
        # index 2: obi_l1 (一级失衡)
        # index 6: ret_lag_5 (5 事件动量)
        obi = X[:, 2] if X.shape[1] > 2 else np.zeros(len(X))
        mom = X[:, 6] if X.shape[1] > 6 else np.zeros(len(X))
        # X 已按训练统计量标准化，这里的 obi/mom 是标准化数值，不是原始 OBI/收益。
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
        print(f"  - [{model.name}] 训练完成 (耗时: {fit_time:.2f}s, 推理延迟: {latency:.2f} us/event)")
    return latencies
