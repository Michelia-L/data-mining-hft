"""论文第 5 页 Eq.(5)–(6)/Algorithm 2 的固定期间 OE 选择协议。

原文未公开延迟、无订单、版本切换与 W 更新细节。本模块明确采用：
按 session 开盘锚定墙钟期间，每期间锁定模型；n 为实际选择期间数，
W 为完整可评价期间 OE 均值的运行平均。订单必须全部成熟才反馈，缺失不补零。
因此是带明确工程假设的 Algorithm 2 实现，不是论文生产系统的等价恢复。
"""
from collections import Counter
import heapq

import numpy as np

from src.model_selector import BaseSelector


class PeriodOESelector(BaseSelector):
    """固定期间 UCB 或同期间轮换对照，不接受逐订单 observe 更新。

    每个数值库版本有独立 W/n/反馈次数，旧版本反馈可更新旧统计，但不能
    污染新参数。无成熟反馈的 W 暂为初始化值 0，并另报 feedback_counts。
    """
    def __init__(self, name, models, *, period_ms=300000, c=1., mode='ucb'):
        super().__init__(name, models, reward_type='OE')
        if (type(period_ms) is not int or period_ms <= 0 or type(c) not in (int, float)
                or not np.isfinite(c) or c < 0 or mode not in ('ucb', 'round_robin')):
            raise ValueError('Positive millisecond period and finite nonnegative price-unit C required')
        self.period_ms, self.c, self.mode = period_ms, float(c), mode
        self.states, self.choices, self.period_selections = {}, {}, []

    def state(self, version):
        """版本首次可见时初始化；保留旧统计用于审计，绝不迁移为新统计。"""
        if version not in self.states:
            self.states[version] = dict(visits=np.zeros(self.k, dtype=int),
                feedback_counts=np.zeros(self.k, dtype=int), sums=np.zeros(self.k))
        return self.states[version]

    def select_period(self, session, version, period, clock_ns):
        """在一个墙钟期间首次有效决策时选一次，此后逐 tick 返回同一模型。

        Eq.(6)：W + C*sqrt(2*ln(N)/n)，N=sum(n)。未访臂视为 +∞，按
        声明顺序消歧；全部已访时 N>=K>=1，不用旧事件变体的 ln(1+N)。
        n 在选择时增加，奖励长延迟也不会把同一模型永远当作从未访问。
        """
        key = (session, version, period)
        if key not in self.choices:
            state = self.state(version)
            visits, counts = state['visits'], state['feedback_counts']
            means = state['sums'] / np.maximum(counts, 1)
            unvisited = np.flatnonzero(visits == 0)
            if len(unvisited):
                tied = unvisited.tolist()
                scores = [None if n == 0 else float(w + self.c * np.sqrt(2*np.log(max(1, visits.sum()))/n))
                          for n, w in zip(visits, means)]
            else:
                scores = (means + self.c*np.sqrt(2*np.log(visits.sum())/visits)).tolist()
                tied = np.flatnonzero(np.asarray(scores) == max(scores)).tolist()
            idx = int(visits.sum() % self.k) if self.mode == 'round_robin' else tied[0]
            if self.mode == 'round_robin':
                # 轮换不消费奖励或 UCB 得分，不把它的选择伪装成得分最大化。
                scores, tied = None, []
            self.choices[key] = idx
            self.period_selections.append(dict(session_id=session, model_version_id=version,
                period_index=int(period), selected_at_ns=int(clock_ns), owner=idx,
                visits_before=visits.tolist(), feedback_counts_before=counts.tolist(),
                means_before=means.tolist(), scores_before=scores, tied_best_owners=tied,
                unvisited_score='positive_infinity' if self.mode == 'ucb' else 'not_applicable', mode=self.mode))
            visits[idx] += 1
        return self.record(self.choices[key])

    def observe(self, idx, reward, step):
        """拒绝 per-order 回调，避免悄悄变回 CausalEventUCB 协议。"""
        raise ValueError('Period selector accepts complete period feedback only')

    def observe_period(self, version, owner, reward, *, context=None):
        """W 为每个完整期间 Eq.(5) 均值的等权运行平均；不按成交数量加权期间。

        n 和反馈次数分开：无订单、坏标签或仍在等待的已选择期间也占一次探索，
        但不会伪造一次零奖励。C 和 W 均用价格差单位，无裁剪或额外归一化。
        """
        if not np.isfinite(reward) or not 0 <= owner < self.k:
            raise ValueError('Finite period reward and valid owner required')
        BaseSelector.observe(self, owner, reward, 0)
        state = self.state(version)
        state['feedback_counts'][owner] += 1
        state['sums'][owner] += reward

    def summary(self):
        """以普通 JSON 数值保存访问数、反馈数和 W，零反馈均值显式未定义。"""
        return {version: dict(visits=s['visits'].tolist(), feedback_counts=s['feedback_counts'].tolist(),
            observed_period_means=[float(total/n) if n else None for total,n in zip(s['sums'],s['feedback_counts'])])
            for version,s in self.states.items()}


class PeriodOrderBook:
    """按实际成交时刻分桶，完整期间全部订单解决后才送一次 Eq.(5) 反馈。

    桶键为 session/数值版本/期间/原仓位模型。退出沿用原开仓归属，所以一个
    墙钟期间可能有多个模型桶；它们仍只来自真实账户成交，没有影子反馈。
    不因换模型强制平仓。原文未公开此归属细节，沿用项目可核对账本约定。
    """
    def __init__(self, selector, calendar, longest_ns):
        self.selector, self.calendar, self.longest_ns = selector, calendar, int(longest_ns)
        self.buckets, self.due, self.completed = {}, [], []

    def bounds(self, session, clock_ns):
        """[start,end) 墙钟期间锚定交易日开盘，休市和缺格不压缩成连续 tick。"""
        info = self.calendar.sessions[session]
        start, close = info['open'].value, info['close'].value
        if not start <= clock_ns < close:
            raise ValueError('Period decisions/fills must be inside session boundaries')
        width = self.selector.period_ms * 1000000
        index = int((clock_ns-start)//width)
        return index, start+index*width, min(start+(index+1)*width, close)

    def bucket(self, session, version, owner, clock_ns):
        """只有实际选择或成交才创建桶；跳过的空网格不产生虚假探索次数。"""
        period,start,end = self.bounds(session,clock_ns)
        key = (session,version,period,int(owner))
        if key not in self.buckets:
            self.buckets[key] = dict(session_id=session, model_version_id=version, period_index=period,
                owner=int(owner), start_ns=int(start), end_ns=int(end), selected=False,
                orders=0, pending=0, statuses=Counter(), reward_sum=0., last_origin_ns=None)
            heapq.heappush(self.due,(int(end),key))
        return key

    def choose(self, session, version, clock_ns):
        """仅在当前有效决策时分配期间访问；在缺口后根据实际墙钟另起期间。"""
        period,_,_ = self.bounds(session,clock_ns)
        owner = self.selector.select_period(session,version,period,clock_ns)
        self.buckets[self.bucket(session,version,owner,clock_ns)]['selected'] = True
        return owner

    def add_order(self, session, version, owner, clock_ns):
        """登记真实成交和未解决标签；返回不可变桶键供成熟队列保留归属。"""
        key=self.bucket(session,version,owner,clock_ns);bucket=self.buckets[key]
        bucket['orders'] += 1;bucket['pending'] += 1;bucket['last_origin_ns']=int(clock_ns)
        return key

    def resolve(self, key, status, score=None):
        """成熟/缺格都解决一单；坏奖励不能按成熟子集分母替代完整期间订单数。"""
        bucket=self.buckets[key]
        if status not in ('matured','missing_target','crossed_gap_or_session') or bucket['pending'] <= 0:
            raise ValueError('Invalid period order resolution')
        if status=='matured' and (score is None or not np.isfinite(score)):
            raise ValueError('Finite matured score required')
        bucket['pending'] -= 1;bucket['statuses'][status] += 1
        if status=='matured':bucket['reward_sum'] += score

    def complete(self, key, now, *, terminal=False):
        """未结束、未成熟、无订单或不完整期间保留 None，不隐式训练选择器。"""
        bucket=self.buckets.pop(key)
        status = ('incomplete_period' if now < bucket['end_ns'] else
                  'pending_orders' if bucket['pending'] else 'no_orders' if not bucket['orders'] else
                  'incomplete_orders' if bucket['statuses'].get('matured',0)!=bucket['orders'] else 'matured')
        reward = bucket['reward_sum']/bucket['orders'] if status=='matured' else None
        if reward is not None:
            # UCB 保持原数值；ARS 需要真实观察时刻和期间身份，不能把旧反馈冒充当前期间。
            self.selector.observe_period(bucket['model_version_id'],bucket['owner'],reward,
                context=dict(session_id=bucket['session_id'],period_index=bucket['period_index'],
                             observed_at_ns=int(now),last_origin_ns=bucket['last_origin_ns']))
        self.completed.append(bucket | dict(statuses=dict(bucket['statuses']), status=status,
            reward=reward, observed_at_ns=int(now), terminal_audit=terminal,
            updated_selector=reward is not None))

    def advance(self, now):
        """执行器先解决到期订单，再推进期间；堆仅处理到期桶，避免逐 tick 全表扫描。"""
        while self.due and self.due[0][0] <= now:
            due,key=heapq.heappop(self.due)
            if key not in self.buckets:continue
            bucket=self.buckets[key]
            if bucket['pending']:
                deadline=bucket['last_origin_ns']+self.longest_ns
                if deadline<=now:raise RuntimeError('Engine must resolve due orders before closing periods')
                heapq.heappush(self.due,(deadline,key))
            else:self.complete(key,now)

    def finish(self, now):
        """不向未来排空队列；只审计末端未结束或未成熟期间，不伪造尾部价格。"""
        self.advance(now)
        for key in list(self.buckets):self.complete(key,now,terminal=True)
        return self.completed
