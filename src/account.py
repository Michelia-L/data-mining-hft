"""期货美元账本：毛盈亏与点差、滑点、手续费分别记账。

Account 接收调用方的单调时钟；真实时间引擎传 UTC 纳秒，复核期也用纳秒。
价格点×合约点值×张数为美元；同向续持不增加成交，退出归原持仓模型。
"""
from src.config import QUANTITY

class Account:
    """单个固定张数期货账户，同时最多持有一个方向的仓位。

    position 为 +1/-1/0，分别表示多头、空头、空仓；quantity 才是合约张数。
    realized_gross 记录已平仓的中间价毛盈亏，costs 记录实际成交成本；
    trades 是完整开平仓记录，fills 是每次买或卖成交记录，两者数量不同。"""
    def __init__(self, config, holding_period, quantity=QUANTITY):
        """建立空仓账户。holding_period 与 step 使用调用方相同的时钟单位，turnover 是累计成交名义金额而非利润。"""
        self.config, self.holding_period, self.quantity = config, holding_period, quantity
        self.position = 0
        # 连续持仓的开仓时刻与最近一次“持仓复核”时刻分开记录：
        # entry_step 用于整笔交易持有期/盈亏，last_review_step 只控制下一次允许重新评估的时点。
        self.last_review_step = None
        self.realized_gross = 0.
        self.costs = 0.
        self.trades, self.fills = [], []
        self.turnover = 0.

    def fill(self, step, side, owner, mid, bid, ask, opening):
        """模拟一笔主动成交并立即记成本，返回供 OE 延迟评价使用的订单记录。

        side=+1 买、-1 卖；owner 是产生这笔仓位的模型索引；opening 区分开仓与退出。
        买入用卖一加不利滑点，卖出用买一减不利滑点。成交价与中间价的差额换算
        为美元后加单边手续费；cost_ratio 再将该美元成本除以成交时的中间价名义金额。
        这里仅记录成交，不自行修改持仓方向，持仓由 advance/close 管理。"""
        execution = (ask if side > 0 else bid) + side * self.config['slippage_ticks'] * self.config['tick_size']
        # 点值乘以张数：价格差 × multiplier 才是美元盈亏。
        multiplier = self.config['multiplier'] * self.quantity
        # 卖出成交价低于 mid，价格差为负；再乘 side=-1 后仍是正成本。
        cost = side * (execution - mid) * multiplier + self.config['commission_per_order'] * self.quantity
        order = dict(step=step, side=side, owner=int(owner), mid=float(mid), price=float(execution),
                     cost_usd=float(cost), cost_ratio=float(cost / (mid * multiplier)), opening=opening)
        self.costs += cost
        self.turnover += abs(execution * multiplier)
        self.fills.append(order)
        return order

    def close(self, step, mid, bid, ask):
        """平掉当前持仓，把一笔完整交易的毛利、双边成本和净利记入账本。

        退出方向与持仓方向相反，但退出的奖励归属原开仓模型，不归属当时碰巧
        被选择的新模型。成本已在 fill 时扣入账户，这里只汇总到交易明细，避免重复扣款。"""
        fill = self.fill(step, -self.position, self.owner, mid, bid, ask, False)
        # 多头涨价盈利、空头跌价盈利；毛利按中间价计，点差/滑点已计入成本。
        gross = self.position * (mid - self.entry_mid) * self.config['multiplier'] * self.quantity
        costs = self.entry_cost + fill['cost_usd']
        self.realized_gross += gross
        self.trades.append(dict(entry_step=self.entry_step, exit_step=step, direction=self.position,
                                owner=self.owner, entry_price=self.entry_price, exit_price=fill['price'],
                                gross_pnl_usd=float(gross), net_pnl_usd=float(gross - costs),
                                friction_usd=float(costs), holding_events=step - self.entry_step))
        self.position = 0
        self.last_review_step = None
        return fill

    def advance(self, step, desired, owner, mid, bid, ask, terminal=False):
        """用当前报价执行先前产生的交易意图，返回本事件真实发生的成交列表。

        holding_period 不再是强制平仓期限，而是两次持仓复核之间的最短事件间隔。
        在复核点之前忽略新的方向变化；到复核点时，同向信号继续持有并把下次复核
        顺延一个 holding_period，反向或零信号则只平仓。本事件不会立即重新开仓，
        避免“超时平仓后同方向/反方向立刻入场”造成没有经济暴露变化的重复摩擦。
        终点仍立即强平；空仓时的非零信号可正常开仓。"""
        fills = []
        closed_on_review = False
        if self.position:
            if terminal:
                fills.append(self.close(step, mid, bid, ask))
            elif step - self.last_review_step >= self.holding_period:
                if desired == self.position:
                    # 信号未改变：延续同一笔交易，只刷新下一次允许复核的时点。
                    self.last_review_step = step
                else:
                    # 反向或零信号：在复核点结束当前仓位，但不在同一事件立刻重开。
                    fills.append(self.close(step, mid, bid, ask))
                    closed_on_review = True
        if not terminal and not self.position and desired and not closed_on_review:
            order = self.fill(step, desired, owner, mid, bid, ask, True)
            fills.append(order)
            self.position, self.owner = desired, int(owner)
            self.entry_step, self.entry_mid = step, mid
            self.entry_price, self.entry_cost = order['price'], order['cost_usd']
            self.last_review_step = step
        return fills

    def gross_equity_change(self, mid):
        """返回已实现毛盈亏加当前持仓浮动毛盈亏，单位为美元。

        用中间价盯市可以让尚未平仓的亏损进入回撤计算；调用者再扣累计实际成本。"""
        unrealized = (self.position * (mid - self.entry_mid) * self.config['multiplier'] * self.quantity
                      if self.position else 0.)
        return self.realized_gross + unrealized
