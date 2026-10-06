"""模型选择的共享接口与现金、固定候选、均值集成三个参照。"""
import numpy as np

class BaseSelector:
    """所有选择器共享的接口与行为开关。

    needs_shadow 表示需要评估未被选中的模型；ensemble 表示取预测均值；
    flat 表示现金策略。具体成交行为由执行引擎读取这些开关后处理。"""
    needs_shadow = False
    ensemble = False
    flat = False

    def __init__(self, name, models, reward_type='ME'):
        """登记候选模型、奖励类型以及动作/奖励历史；模型顺序必须与预测矩阵列一致。"""
        if not models or reward_type not in ('ME', 'OE'):
            raise ValueError('Nonempty model library and ME/OE reward required')
        self.name, self.models, self.reward_type = name, models, reward_type
        self.k = len(models)
        self.action_history = []
        self.reward_history = []

    def record(self, idx):
        """记录本事件选中的模型索引，并返回 Python int，方便后续索引与序列化。"""
        self.action_history.append(int(idx))
        return int(idx)

    def observe(self, idx, reward, step):
        """接收一条已经成熟的奖励，校验有限性并保存历史。

        idx 指产生该奖励的模型，step 指现在观察到奖励的事件步，不是原始决策步。
        子类在调用此方法后，再更新自己的统计量。"""
        if not np.isfinite(reward):
            raise ValueError('Nonfinite reward')
        self.reward_history.append(float(reward))

    def get_selection_distribution(self):
        """统计各模型被选中的频率，用于诊断是否学到偏好或近似均匀轮换。

        静态集成和现金没有真实的单模型选择，返回空字典而非虚构第一个模型占比。"""
        if self.ensemble or self.flat:
            return {}  # 集成和现金没有“被选中模型”，不生成误导性的选择分布。
        counts = np.bincount(self.action_history, minlength=self.k)
        return {m.name: float(c / max(1, len(self.action_history)))
                for m, c in zip(self.models, counts)}

class SingleModelSelector(BaseSelector):
    """固定模型基线：整个测试区间始终使用同一个预测模型。"""
    def __init__(self, name, models, fixed_idx=0):
        """fixed_idx 为模型库中的列号；课程固定候选在运行前声明，不能根据后段收益挑选。"""
        super().__init__(name, models)
        self.fixed_idx = fixed_idx

    def select_model(self, step):
        """记录并返回固定索引，不根据测试奖励重新挑选模型。"""
        return self.record(self.fixed_idx)

class EnsembleSelector(SingleModelSelector):
    """静态等权集成：引擎对所有模型预测求均值，继承的索引仅用于统一接口。"""
    ensemble = True

class FlatSelector(SingleModelSelector):
    """现金基线：引擎始终给出零交易意图，用来检验交易是否优于不交易。"""
    flat = True
