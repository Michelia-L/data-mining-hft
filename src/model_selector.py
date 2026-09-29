"""在线模型选择器：把轻量预测模型看作多臂老虎机的不同动作。

select_model(step) 只根据已观察奖励作决策；observe(idx,reward,step)
由执行引擎在奖励成熟后调用。选择器不访问行情未来标签，也不负责下单记账。
归一化奖励处于 [-1,1]，价格差奖励不裁剪。以下两类均是事件级工程变体，
不等价于论文按固定期间更新的 Algorithm 2 或历史区间重回测的 Algorithm 3。"""
from collections import deque
import numpy as np
from src.config import RL_CONFIG, SEED


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


class CausalEventUCBSelector(BaseSelector):
    """事件级 UCB 变体（不是论文 Algorithm 2 的固定期间评价）：在利用已有高奖励模型与探索少选模型之间权衡。

    分数为平均成熟奖励 + c*sqrt(2*log(1+总选择数)/该模型选择数)。
    反馈延迟时，“已选择次数”和“已观察奖励次数”不同，必须分开统计。"""
    def __init__(self, name, models, reward_type='ME', c=RL_CONFIG['ucb_c']):
        """初始化探索系数及两类计数；c 由验证段选择，不能使用测试表现调参。"""
        super().__init__(name, models, reward_type)
        self.c = c
        self.counts = np.zeros(self.k, dtype=int)  # 已作出的选择次数，包含仍在等待反馈的动作。
        self.feedback_counts = np.zeros(self.k, dtype=int)
        self.sum_rewards = np.zeros(self.k)

    def select_model(self, step):
        """选择未尝试过的模型，或选择 UCB 得分最大的模型。

        做决定就立即增加 counts；即使奖励尚未到期，也不会一直把第一个模型当成
        “从未选过”。没有成熟反馈时暂用均值 0，探索项仍按真实选择次数计算。"""
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited):
            idx = int(unvisited[0])
        else:
            means = self.sum_rewards / np.maximum(self.feedback_counts, 1)
            bonus = self.c * np.sqrt(2 * np.log(1 + self.counts.sum()) / self.counts)
            idx = int(np.argmax(means + bonus))
        self.counts[idx] += 1
        return self.record(idx)

    def observe(self, idx, reward, step):
        """按奖励归属更新反馈数量和奖励总和；引擎将同一决策产生的多笔成交先取平均，无成交则反馈零，
        因而每个已成熟决策恰有一次反馈。末尾未成熟决策只保留在选择计数中。"""
        super().observe(idx, reward, step)
        self.feedback_counts[idx] += 1
        self.sum_rewards[idx] += reward


class CausalShadowARSSelector(BaseSelector):
    """因果影子账户 ARS 变体，不是论文 Algorithm 3 的历史窗口重新回测。

    账户从回放起点连续运行；window_events=300 表示奖励到达后的事件窗口，
    不是论文举例的过去 30 分钟。不能用不等间隔事件近似声称固定经济时间。

    窗口按奖励到达的事件时间计，而不是每个模型各保留固定条数；因此少交易
    模型很久以前的奖励也会过期，不会永久支配选择。"""
    needs_shadow = True

    def __init__(self, name, models, reward_type='ME', window_events=RL_CONFIG['ars_window_events']):
        """每个模型维护一个 (观察事件步, 奖励) 队列及其运行和，便于快速增删。"""
        super().__init__(name, models, reward_type)
        self.window_events = window_events
        self.queues = [deque() for _ in models]
        self.sums = np.zeros(self.k)

    def observe(self, idx, reward, step):
        """把新成熟奖励加入对应模型队尾，同时更新运行和。"""
        super().observe(idx, reward, step)
        self.queues[idx].append((step, reward))
        self.sums[idx] += reward

    def select_model(self, step):
        """清理时间窗外反馈，再贪心选择窗口平均奖励最高者。

        全部模型都没有可用反馈时按步数轮换；某个模型窗口为空时其分数为 0。
        这是明确的冷启动约定，不能把“没有观测”解释成已证明该模型没有风险。"""
        for i, queue in enumerate(self.queues):
            while queue and queue[0][0] <= step - self.window_events:
                self.sums[i] -= queue.popleft()[1]
        # 所有奖励仍在等待期，或已有奖励全部过期时，轮换冷启动。
        if not any(self.queues):
            return self.record(step % self.k)
        means = [self.sums[i] / len(q) if q else 0. for i, q in enumerate(self.queues)]
        return self.record(np.argmax(means))


class SingleModelSelector(BaseSelector):
    """固定模型基线：整个测试区间始终使用同一个预测模型。"""
    def __init__(self, name, models, fixed_idx=0):
        """fixed_idx 为模型库中的列号；验证最佳基线的列号由验证段确定。"""
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


class RoundRobinSelector(BaseSelector):
    """固定轮换基线：检验动态选择能否优于不学习、只按顺序轮换模型。"""
    def select_model(self, step):
        """按事件步取模轮流选择，整个序列不依赖奖励。"""
        return self.record(step % self.k)


class RandomSelector(BaseSelector):
    """随机模型基线：随机的是模型索引，不是直接生成随机买卖方向。"""
    def __init__(self, name, models, seed=SEED):
        """使用实例独立的随机数生成器，避免其他代码调用全局随机函数改变实验。"""
        super().__init__(name, models)
        self.rng = np.random.default_rng(seed)

    def select_model(self, step):
        """均匀随机选择一个模型并登记动作历史。"""
        return self.record(self.rng.integers(self.k))


def build_all_selectors(models, c_by_reward=None, seed=SEED):
    """构建四个显式命名的工程变体组合和公共基线，每次调用都返回全新的选择器实例。

    ME/OE 各配 UCB 和 ARS；此外包含每个固定模型、集成、随机、轮换与现金。
    验证最佳模型、额外随机种子和奖励权重消融由实验入口继续添加。"""
    c_by_reward = c_by_reward or {}
    selectors = []
    for reward in ('ME', 'OE'):
        selectors += [CausalEventUCBSelector(f'Variant-{reward}-EventUCB', models, reward,
                                  c=c_by_reward.get(reward, RL_CONFIG['ucb_c'])),
                      CausalShadowARSSelector(f'Variant-{reward}-CausalShadowARS', models, reward)]
    selectors += [SingleModelSelector(f'Baseline-Single-{m.name}', models, i)
                  for i, m in enumerate(models)]
    selectors += [EnsembleSelector('Baseline-Static-Ensemble', models),
                  RandomSelector('Baseline-Random', models, seed),
                  RoundRobinSelector('Baseline-RoundRobin', models),
                  FlatSelector('Baseline-Cash', models)]
    return selectors
