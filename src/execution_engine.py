"""Causal event replay with next-event fills, dollar accounting and delayed rewards."""
from collections import deque
import numpy as np
from src.config import RL_CONFIG, INSTRUMENT_CONFIG, INITIAL_CAPITAL, QUANTITY


class Account:
    """One fixed-size futures position. Costs are charged once per actual fill."""
    def __init__(self, config, holding_period, quantity=QUANTITY):
        self.config, self.holding_period, self.quantity = config, holding_period, quantity
        self.position = 0
        self.realized_gross = 0.
        self.costs = 0.
        self.trades, self.fills = [], []
        self.turnover = 0.

    def fill(self, step, side, owner, mid, bid, ask, opening):
        execution = (ask if side > 0 else bid) + side * self.config['slippage_ticks'] * self.config['tick_size']
        multiplier = self.config['multiplier'] * self.quantity
        cost = side * (execution - mid) * multiplier + self.config['commission_per_order'] * self.quantity
        order = dict(step=step, side=side, owner=int(owner), mid=float(mid), price=float(execution),
                     cost_usd=float(cost), cost_ratio=float(cost / (mid * multiplier)), opening=opening)
        self.costs += cost
        self.turnover += abs(execution * multiplier)
        self.fills.append(order)
        return order

    def close(self, step, mid, bid, ask):
        fill = self.fill(step, -self.position, self.owner, mid, bid, ask, False)
        gross = self.position * (mid - self.entry_mid) * self.config['multiplier'] * self.quantity
        costs = self.entry_cost + fill['cost_usd']
        self.realized_gross += gross
        self.trades.append(dict(entry_step=self.entry_step, exit_step=step, direction=self.position,
                                owner=self.owner, entry_price=self.entry_price, exit_price=fill['price'],
                                gross_pnl_usd=float(gross), net_pnl_usd=float(gross - costs),
                                friction_usd=float(costs), holding_events=step - self.entry_step))
        self.position = 0
        return fill

    def advance(self, step, desired, owner, mid, bid, ask, terminal=False):
        fills = []
        if self.position and (terminal or step - self.entry_step >= self.holding_period
                              or (desired and desired != self.position)):
            fills.append(self.close(step, mid, bid, ask))
        if not terminal and not self.position and desired:
            order = self.fill(step, desired, owner, mid, bid, ask, True)
            fills.append(order)
            self.position, self.owner = desired, int(owner)
            self.entry_step, self.entry_mid = step, mid
            self.entry_price, self.entry_cost = order['price'], order['cost_usd']
        return fills

    def gross_equity_change(self, mid):
        unrealized = (self.position * (mid - self.entry_mid) * self.config['multiplier'] * self.quantity
                      if self.position else 0.)
        return self.realized_gross + unrealized


class ExecutionEngine:
    def __init__(self, instrument_key, irl_learner, threshold=None, initial_capital=INITIAL_CAPITAL,
                 quantity=QUANTITY, latency_events=RL_CONFIG['latency_events'],
                 holding_period=RL_CONFIG['holding_period']):
        self.config = INSTRUMENT_CONFIG[instrument_key]
        self.irl_learner = irl_learner
        self.threshold = self.config['trade_threshold'] if threshold is None else threshold
        self.initial_capital, self.quantity = initial_capital, quantity
        self.latency_events, self.holding_period = latency_events, holding_period
        if latency_events < 1 or holding_period < 1 or initial_capital <= 0 or quantity <= 0 or self.threshold < 0:
            raise ValueError('Invalid execution settings; signals must execute on a later event')

    def run_backtest(self, selector, test_df, X_test=None, all_model_preds=None, detail=False):
        """No label columns are consumed. All reward endpoints satisfy endpoint <= t.

        OE observes each actual buy/sell fill (including exits), attributed to its
        policy owner. ARS uses independent shadow accounts for these observations.
        OE measures future mark-to-mid value minus that fill's costs; account net
        PnL separately includes both entry and exit costs. Unmatured terminal
        observations never update a selector.
        """
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
        live_pending, shadow_pending = deque(), deque()
        intents, owners = [], []
        feature_sum = np.zeros(len(hs))
        matured_orders = 0
        gross_values, net_values = [], []
        reward_observations = []

        def mature(queue, t, update, collect=False):
            nonlocal matured_orders
            while queue and queue[0]['step'] + horizon <= t:
                order = queue.popleft()
                start = order['step']
                returns = mid[start + hs] / mid[start] - 1.
                features = self.irl_learner.features(returns, order['side'], order['cost_ratio'])
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
            # First execute a decision made at an earlier event, using current quotes.
            source = t - self.latency_events
            terminal = t == n - 1
            desired = intents[source] if source >= 0 else 0
            owner = owners[source] if source >= 0 else 0
            fills = account.advance(t, desired, owner, mid[t], bid[t], ask[t], terminal)
            live_pending.extend(fills)
            for i, shadow in enumerate(shadows):
                signal = preds[source, i] if source >= 0 else 0.
                direction = int(signal > self.threshold) - int(signal < -self.threshold)
                shadow_pending.extend(shadow.advance(t, direction, i, mid[t], bid[t], ask[t], terminal))

            # Mature previously issued observations before choosing the next action.
            mature(live_pending, t, selector.reward_type == 'OE' and not shadows, collect=True)
            mature(shadow_pending, t, bool(shadows))
            if selector.reward_type == 'ME' and t >= horizon:
                start = t - horizon
                returns = mid[start + hs] / mid[start] - 1.
                indices = range(selector.k) if selector.needs_shadow else [owners[start]]
                for idx in indices:
                    signal = (0. if selector.flat else float(preds[start].mean()) if selector.ensemble
                              else preds[start, idx])
                    reward = self.irl_learner.score(self.irl_learner.features(returns, np.sign(signal)))
                    selector.observe(idx, reward, t)
                    if detail:
                        reward_observations.append(dict(origin_step=start, observed_step=t,
                                                        owner=int(idx), reward=reward, kind='ME'))

            gross = account.gross_equity_change(mid[t])
            gross_values.append(gross)
            net_values.append(gross - account.costs)
            if not terminal:
                idx = selector.select_model(t)
                signal = 0. if selector.flat else float(preds[t].mean()) if selector.ensemble else preds[t, idx]
                intents.append(int(signal > self.threshold) - int(signal < -self.threshold))
                owners.append(idx)

        gross_array, net_array = np.asarray(gross_values), np.asarray(net_values)
        equity = np.r_[self.initial_capital, self.initial_capital + net_array]
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        trade_nets = np.array([x['net_pnl_usd'] for x in account.trades])
        wins, losses = trade_nets[trade_nets > 0], trade_nets[trade_nets < 0]
        # Downsample only presentation; risk metrics above use every mark.
        points = sorted(set([0, n - 1] + list(range(0, n, max(1, n // 300)))))
        result = dict(
            strategy=selector.name, total_trades=len(account.trades), total_fills=len(account.fills),
            initial_capital=self.initial_capital, quantity=self.quantity,
            gross_pnl_usd=float(gross_array[-1]), net_pnl_usd=float(net_array[-1]),
            friction_usd=float(account.costs), gross_return=float(gross_array[-1] / self.initial_capital),
            net_return=float(net_array[-1] / self.initial_capital),
            total_friction=float(account.costs / self.initial_capital),
            max_drawdown=float(drawdowns.max()), max_drawdown_usd=float((peaks - equity).max()),
            sharpe_ratio=None, sharpe_reason='Insufficient independent daily observations; no annualization.',
            win_rate=float(len(wins) / len(trade_nets)) if len(trade_nets) else 0.,
            pl_ratio=float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None,
            avg_trade_net_usd=float(trade_nets.mean()) if len(trade_nets) else 0.,
            turnover_notional_usd=float(account.turnover),
            action_distribution=selector.get_selection_distribution(),
            gross_curve=(gross_array[points] / self.initial_capital).tolist(),
            net_curve=(net_array[points] / self.initial_capital).tolist(),
            timestamps=[str(test_df.ts_event.iloc[t]) for t in points], curve_steps=points,
            terminal_position=account.position, matured_order_count=matured_orders,
            unobserved_terminal_fills=len(live_pending),
            order_feature_expectation=(feature_sum / n).tolist(),
            observed_rewards=len(selector.reward_history),
        )
        if detail:
            result.update(trades=account.trades, fills=account.fills,
                          actions=selector.action_history, reward_observations=reward_observations,
                          equity_usd=(self.initial_capital + net_array).tolist())
        return result
