"""Delayed-feedback selectors. Rewards have a common bounded [-1, 1] scale."""
from collections import deque
import numpy as np
from src.config import RL_CONFIG, SEED


class BaseSelector:
    needs_shadow = False
    ensemble = False
    flat = False

    def __init__(self, name, models, reward_type='ME'):
        if not models or reward_type not in ('ME', 'OE'):
            raise ValueError('Nonempty model library and ME/OE reward required')
        self.name, self.models, self.reward_type = name, models, reward_type
        self.k = len(models)
        self.action_history = []
        self.reward_history = []

    def record(self, idx):
        self.action_history.append(int(idx))
        return int(idx)

    def observe(self, idx, reward, step):
        if not np.isfinite(reward):
            raise ValueError('Nonfinite reward')
        self.reward_history.append(float(reward))

    def get_selection_distribution(self):
        if self.ensemble or self.flat:
            return {}  # There is no selected model for a static blend / cash policy.
        counts = np.bincount(self.action_history, minlength=self.k)
        return {m.name: float(c / max(1, len(self.action_history)))
                for m, c in zip(self.models, counts)}


class UCBSelector(BaseSelector):
    def __init__(self, name, models, reward_type='ME', c=RL_CONFIG['ucb_c']):
        super().__init__(name, models, reward_type)
        self.c = c
        self.counts = np.zeros(self.k, dtype=int)  # Decisions, including pending feedback.
        self.feedback_counts = np.zeros(self.k, dtype=int)
        self.sum_rewards = np.zeros(self.k)

    def select_model(self, step):
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
        super().observe(idx, reward, step)
        self.feedback_counts[idx] += 1
        self.sum_rewards[idx] += reward


class ARSSelector(BaseSelector):
    needs_shadow = True

    def __init__(self, name, models, reward_type='ME', window_size=RL_CONFIG['ars_window']):
        super().__init__(name, models, reward_type)
        self.window_size = window_size
        self.queues = [deque() for _ in models]
        self.sums = np.zeros(self.k)

    def observe(self, idx, reward, step):
        super().observe(idx, reward, step)
        self.queues[idx].append((step, reward))
        self.sums[idx] += reward

    def select_model(self, step):
        for i, queue in enumerate(self.queues):
            while queue and queue[0][0] <= step - self.window_size:
                self.sums[i] -= queue.popleft()[1]
        # No observations yet: rotate, including while all rewards are pending.
        if not any(self.queues):
            return self.record(step % self.k)
        means = [self.sums[i] / len(q) if q else 0. for i, q in enumerate(self.queues)]
        return self.record(np.argmax(means))


class SingleModelSelector(BaseSelector):
    def __init__(self, name, models, fixed_idx=0):
        super().__init__(name, models)
        self.fixed_idx = fixed_idx

    def select_model(self, step):
        return self.record(self.fixed_idx)


class EnsembleSelector(SingleModelSelector):
    ensemble = True


class FlatSelector(SingleModelSelector):
    flat = True


class RoundRobinSelector(BaseSelector):
    def select_model(self, step):
        return self.record(step % self.k)


class RandomSelector(BaseSelector):
    def __init__(self, name, models, seed=SEED):
        super().__init__(name, models)
        self.rng = np.random.default_rng(seed)

    def select_model(self, step):
        return self.record(self.rng.integers(self.k))


def build_all_selectors(models, c_by_reward=None, seed=SEED):
    c_by_reward = c_by_reward or {}
    selectors = []
    for reward in ('ME', 'OE'):
        selectors += [UCBSelector(f'FMATO-{reward}-UCBS', models, reward,
                                  c=c_by_reward.get(reward, RL_CONFIG['ucb_c'])),
                      ARSSelector(f'FMATO-{reward}-ARS', models, reward)]
    selectors += [SingleModelSelector(f'Baseline-Single-{m.name}', models, i)
                  for i, m in enumerate(models)]
    selectors += [EnsembleSelector('Baseline-Static-Ensemble', models),
                  RandomSelector('Baseline-Random', models, seed),
                  RoundRobinSelector('Baseline-RoundRobin', models),
                  FlatSelector('Baseline-Cash', models)]
    return selectors
