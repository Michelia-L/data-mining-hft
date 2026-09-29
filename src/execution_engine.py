"""因果事件回放：下一事件成交、美元账本与多尺度延迟奖励。

本模块把“选模型”“下单成交”“观察奖励”分开处理。比如第 0 步产生预测，
默认第 1 步按当时盘口成交；ME 最早第 90 步反馈，OE 则从成交步起等待
90 步，最早第 91 步反馈。不能因为历史文件包含后续价格就提前更新选择器。
Account 管理资金与持仓；ExecutionEngine 管理事件时序、独立影子账户和统计。"""
from collections import deque
import numpy as np
from src.config import RL_CONFIG, INSTRUMENT_CONFIG, INITIAL_CAPITAL, QUANTITY


class Account:
    """单个固定张数期货账户，同时最多持有一个方向的仓位。

    position 为 +1/-1/0，分别表示多头、空头、空仓；quantity 才是合约张数。
    realized_gross 记录已平仓的中间价毛盈亏，costs 记录实际成交成本；
    trades 是完整开平仓记录，fills 是每次买或卖成交记录，两者数量不同。"""
    def __init__(self, config, holding_period, quantity=QUANTITY):
        """建立空仓账户。holding_period 按事件数计，turnover 是累计成交名义金额而非利润。"""
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


class ExecutionEngine:
    """在一段已按时间排序的行情上回放一个策略，不训练模型、不访问未来标签。"""
    def __init__(self, instrument_key, irl_learner, threshold=None, initial_capital=INITIAL_CAPITAL,
                 quantity=QUANTITY, latency_events=RL_CONFIG['latency_events'],
                 holding_period=RL_CONFIG['holding_period']):
        """设置品种、奖励学习器和执行假设。

        threshold 是预测相对收益门槛；initial_capital 是本窗口起始资金；
        quantity 是固定张数。latency_events 至少为 1，强制让信号在更晚事件执行。
        holding_period 是两次持仓复核之间的最短事件间隔，不是固定墙钟时间；同向信号
        在复核点续持，反向/零信号才平仓。这里仍不模拟挂单排队或保证金。"""
        self.config = INSTRUMENT_CONFIG[instrument_key]
        self.irl_learner = irl_learner
        self.threshold = self.config['trade_threshold'] if threshold is None else threshold
        self.initial_capital, self.quantity = initial_capital, quantity
        self.latency_events, self.holding_period = latency_events, holding_period
        if latency_events < 1 or holding_period < 1 or initial_capital <= 0 or quantity <= 0 or self.threshold < 0:
            raise ValueError('Invalid execution settings; signals must execute on a later event')

    def run_backtest(self, selector, test_df, X_test=None, all_model_preds=None, detail=False):
        """返回策略指标、抽样资金曲线；detail=True 时额外返回逐笔审计信息。

        输入 test_df 包含当前盘口和时间；all_model_preds 的形状为 N×K，
        每列对应 selector.models 中一个已冻结模型，且每行特征本身必须因果。
        X_test 为接口保留参数，本实现直接消费预先计算好的预测，不再重复推理。

        每个事件依次执行旧意图 → 结算已成熟奖励 → 逐事件盯市 → 产生新意图。
        ME 从信号步计时，OE 从每次实际买卖成交步计时（包括退出）。ARS 的 OE
        使用各模型独立影子账户，避免把未发生的交易或共享仓位当成独立策略业绩。
        OE 默认不扣成本，NetOE 变体可扣该次单边成本；资金账本始终扣完整开平仓成本。
        末尾尚未成熟的奖励不补写进选择器，也不因此丢弃末尾行情或期末平仓。"""
        preds = np.asarray(all_model_preds, float)
        n = len(test_df)
        if n < 2 or preds.shape != (n, selector.k) or not np.isfinite(preds).all():
            raise ValueError('Finite predictions aligned to at least two quotes are required')
        mid = test_df.mid_price.to_numpy(dtype=float)
        bid = test_df.bid_px_00.to_numpy(dtype=float)
        ask = test_df.ask_px_00.to_numpy(dtype=float)
        if (not np.isfinite(np.c_[mid, bid, ask]).all() or (bid <= 0).any()
                or (ask < bid).any() or not test_df.ts_event.is_monotonic_increasing):
            raise ValueError('Invalid or unordered quotes')
        hs = np.asarray(self.irl_learner.horizons, int)
        horizon = int(hs.max())
        account = Account(self.config, self.holding_period, self.quantity)
        shadows = ([Account(self.config, self.holding_period, self.quantity) for _ in range(selector.k)]
                   if selector.needs_shadow and selector.reward_type == 'OE' else [])
        # 实际账户与影子账户的成交分开排队，不能混淆奖励归属或重复更新。
        live_pending, shadow_pending = deque(), deque()
        # EventUCB 的一次拉臂对应一次决策，而不是一笔成交。将该决策触发的全部
        # 成交绑在一起等待成熟；没有成交也有一条零反馈，避免“探索了却永不观测”。
        decision_pending = deque()
        # intents 保存每步目标方向，owners 保存当时的模型，以便延迟执行/延迟反馈。
        intents, owners = [], []
        feature_sum = np.zeros(len(hs))
        matured_orders = 0
        gross_values, net_values = [], []
        reward_observations = []

        def mature(queue, t, update, collect=False):
            """取出当前事件 t 已具备全部未来观察价格的成交，计算并分配奖励。

            queue 按成交时间先进先出；所有成交等待同样的最长时域，因此检查队首即可。
            update 控制是否反馈给选择器；collect 控制是否累积本账户的特征期望。
            即使正在 ME 回放，也收集本账户 OE 特征，用于后续校准段比较不同奖励定义。"""
            nonlocal matured_orders
            while queue and queue[0]['step'] + horizon <= t:
                order = queue.popleft()
                start = order['step']
                # 已由队首条件保证 start+max(hs)<=t，此处每个索引都不超过当前事件。
                features = self.irl_learner.features_from_prices(
                    mid[start], mid[start + hs], order['side'], order['cost_ratio'])
                if collect:
                    feature_sum[:] += features
                    matured_orders += 1
                if update:
                    reward = self.irl_learner.score(features)
                    selector.observe(order['owner'], reward, t)
                    if detail:
                        reward_observations.append(dict(origin_step=start, observed_step=t,
                                                        owner=order['owner'], reward=reward, kind='OE'))

        for t in range(n):
            # 第一步：执行旧决策。t=0 时还没有历史意图，默认空仓。
            source = t - self.latency_events
            terminal = t == n - 1
            desired = intents[source] if source >= 0 else 0
            owner = owners[source] if source >= 0 else 0
            fills = account.advance(t, desired, owner, mid[t], bid[t], ask[t], terminal)
            live_pending.extend(fills)
            if source >= 0 and not terminal and selector.reward_type == 'OE' and not shadows:
                decision_pending.append((t, source, owner, fills))
            # 每个影子账户只跟随自己的模型，独立承担相同的延迟、持仓限制和成交成本。
            for i, shadow in enumerate(shadows):
                signal = preds[source, i] if source >= 0 else 0.
                direction = int(signal > self.threshold) - int(signal < -self.threshold)
                shadow_pending.extend(shadow.advance(t, direction, i, mid[t], bid[t], ask[t], terminal))

            # 第二步：只将现在已经到期的奖励送入选择器；然后才允许本事件的新选择。
            mature(live_pending, t, False, collect=True)
            mature(shadow_pending, t, bool(shadows))
            while decision_pending and decision_pending[0][0] + horizon <= t:
                executed, origin, decision_owner, decision_fills = decision_pending.popleft()
                # 这里归属触发成交的决策模型；Account 中平仓盈亏仍归属原开仓者。
                # 不把奖励归属与资金归属混用，也不把每事件零奖励声称为 Eq.(5)。
                vectors = [self.irl_learner.features_from_prices(
                    mid[executed], mid[executed + hs], f['side'], f['cost_ratio'])
                    for f in decision_fills]
                reward = self.irl_learner.score(np.mean(vectors, axis=0)) if vectors else 0.
                selector.observe(decision_owner, reward, t)
                if detail:
                    reward_observations.append(dict(origin_step=origin, executed_step=executed,
                        observed_step=t, owner=decision_owner, reward=reward,
                        fill_count=len(decision_fills), kind='OE-event'))
            if selector.reward_type == 'ME' and t >= horizon:
                start = t - horizon
                # ARS 可评价各模型的历史信号；UCB 只评价当时实际选中的模型。
                indices = range(selector.k) if selector.needs_shadow else [owners[start]]
                for idx in indices:
                    signal = (0. if selector.flat else float(preds[start].mean()) if selector.ensemble
                              else preds[start, idx])
                    direction = int(signal > self.threshold) - int(signal < -self.threshold)
                    # ME 使用与执行相同的信号门槛。影子 ARS 平均有效信号；EventUCB
                    # 每决策一次观测，因此非交易信号反馈零。这是两个不同的平均口径。
                    if selector.needs_shadow and direction == 0:
                        continue
                    reward = self.irl_learner.score(self.irl_learner.features_from_prices(
                        mid[start], mid[start + hs], direction))
                    selector.observe(idx, reward, t)
                    if detail:
                        reward_observations.append(dict(origin_step=start, observed_step=t,
                                                        owner=int(idx), reward=reward, kind='ME'))

            # 第三步：每条行情都盯市。账户成本已包含开仓费用，即使尚未平仓也必须扣除。
            gross = account.gross_equity_change(mid[t])
            gross_values.append(gross)
            net_values.append(gross - account.costs)
            # 第四步：用成熟反馈选模型，新意图只能在后续事件成交；末事件不再发新意图。
            if not terminal:
                idx = selector.select_model(t)
                signal = 0. if selector.flat else float(preds[t].mean()) if selector.ensemble else preds[t, idx]
                intents.append(int(signal > self.threshold) - int(signal < -self.threshold))
                owners.append(idx)

        gross_array, net_array = np.asarray(gross_values), np.asarray(net_values)
        # 将初始资金点也纳入峰值，确保第一笔手续费立即造成的回撤不会被遗漏。
        equity = np.r_[self.initial_capital, self.initial_capital + net_array]
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        trade_nets = np.array([x['net_pnl_usd'] for x in account.trades])
        wins, losses = trade_nets[trade_nets > 0], trade_nets[trade_nets < 0]
        # 风险指标已经用全部事件算完；以下仅为展示抽样，并显式保留首尾点。
        points = sorted(set([0, n - 1] + list(range(0, n, max(1, n // 300)))))
        # *_usd 为美元，*_return 以初始资金为分母；max_drawdown 以历史资金峰值为分母。
        # 短区间无法构造独立日收益，夏普返回 None，不假定每天固定交易次数来年化。
        # Eq.(5) 分母是成熟订单数 M，绝不是行情事件数。M=0 没有可估计均值；
        # 为让现金策略参加有限策略优化，约定零向量，同时记录 defined=False。
        # 每事件暴露另列，不能再以 order_feature_expectation 名称混用。
        result = dict(
            strategy=selector.name, total_trades=len(account.trades), total_fills=len(account.fills),
            initial_capital=self.initial_capital, quantity=self.quantity,
            gross_pnl_usd=float(gross_array[-1]), net_pnl_usd=float(net_array[-1]),
            friction_usd=float(account.costs), gross_return=float(gross_array[-1] / self.initial_capital),
            net_return=float(net_array[-1] / self.initial_capital),
            total_friction=float(account.costs / self.initial_capital),
            max_drawdown=float(drawdowns.max()), max_drawdown_usd=float((peaks - equity).max()),
            sharpe_ratio=None, sharpe_reason='Insufficient independent daily observations; no annualization.',
            # 没有完整交易时分母为零，胜率未定义；JSON 用 null，展示层显示 N/A。
            win_rate=float(len(wins) / len(trade_nets)) if len(trade_nets) else None,
            pl_ratio=float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
            avg_trade_net_usd=float(trade_nets.mean()) if len(trade_nets) else 0.,
            turnover_notional_usd=float(account.turnover),
            action_distribution=selector.get_selection_distribution(),
            gross_curve=(gross_array[points] / self.initial_capital).tolist(),
            net_curve=(net_array[points] / self.initial_capital).tolist(),
            timestamps=[str(test_df.ts_event.iloc[t]) for t in points], curve_steps=points,
            terminal_position=account.position, matured_order_count=matured_orders,
            unmatured_fill_rewards_at_end=len(live_pending),
            order_feature_expectation=(feature_sum / matured_orders if matured_orders else feature_sum).tolist(),
            order_feature_expectation_defined=bool(matured_orders),
            event_feature_exposure=(feature_sum / n).tolist(),
            reward_definition=self.irl_learner.definition,
            reward_cost_mode='NetOE' if self.irl_learner.deduct_cost else 'PaperOE-no-cost',
            reward_horizon_unit='events',
            selector_window_events=getattr(selector, 'window_events', None),
            decision_count=len(selector.action_history),
            feedback_counts=(selector.feedback_counts.tolist() if hasattr(selector, 'feedback_counts') else None),
            observed_rewards=len(selector.reward_history),
        )
        if detail:
            result.update(trades=account.trades, fills=account.fills,
                          actions=selector.action_history, reward_observations=reward_observations,
                          equity_usd=(self.initial_capital + net_array).tolist())
        return result
