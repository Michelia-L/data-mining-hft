"""Execution/backtest engine for BTC 1-second bars.

No bid/ask quotes are present in the input, so execution cost is an explicit
flat per-side assumption instead of a claimed observed spread.  Gross return is
therefore the most data-grounded execution metric; fixed-cost net returns and a
sensitivity table are reported separately.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

HORIZONS = [10, 30, 90]

class BTCExecutionEngine:
    def __init__(self, irl_learner, trade_threshold: float = 8e-5,
                 holding_period: int = 30, cost_bps_per_side: float = 3.0):
        self.irl_learner = irl_learner
        self.threshold = float(trade_threshold)
        self.holding_period = int(holding_period)
        self.cost_ratio = float(cost_bps_per_side) * 1e-4
        self.cost_bps_per_side = float(cost_bps_per_side)

    def run_backtest(self, selector, test_df, X_test, all_model_preds) -> Dict[str, Any]:
        n_steps = len(test_df)
        k_models = selector.k
        prices = test_df["reference_price"].to_numpy()
        future_rets_matrix = test_df[[f"future_ret_{h}" for h in HORIZONS]].to_numpy()

        current_pos = 0
        entry_price = 0.0
        entry_step = 0
        trades = []
        gross_cum_ret = 0.0
        net_cum_ret = 0.0
        gross_curve = [0.0]
        net_curve = [0.0]
        timestamps = [str(test_df["ts_event"].iloc[0])]

        feedback_delay = max(HORIZONS)
        chosen_history = []

        for t in range(n_steps - max(HORIZONS)):
            # Causal online feedback: the reward for decision (t-90) only
            # becomes observable now, when its longest horizon has elapsed.
            if t >= feedback_delay:
                r_idx = t - feedback_delay
                prev_chosen = chosen_history[r_idx]
                prev_pred = float(np.mean(all_model_preds[r_idx, :])) if selector.name == "Baseline-Static-Ensemble" else float(all_model_preds[r_idx, prev_chosen])
                prev_future_rets = future_rets_matrix[r_idx]
                prev_desired = 1 if prev_pred > self.threshold else (-1 if prev_pred < -self.threshold else 0)

                if selector.reward_type == "ME":
                    delayed_reward = self.irl_learner.compute_model_expectation(prev_pred, prev_future_rets)
                else:
                    delayed_reward = self.irl_learner.compute_order_traded_expectation(
                        prev_desired, prev_future_rets, rel_spread=0.0, cost_ratio=self.cost_ratio
                    )

                delayed_all_rewards = np.zeros(k_models)
                for m_i in range(k_models):
                    p_i = float(all_model_preds[r_idx, m_i])
                    if selector.reward_type == "ME":
                        delayed_all_rewards[m_i] = self.irl_learner.compute_model_expectation(p_i, prev_future_rets)
                    else:
                        d_i = 1 if p_i > self.threshold else (-1 if p_i < -self.threshold else 0)
                        delayed_all_rewards[m_i] = self.irl_learner.compute_order_traded_expectation(
                            d_i, prev_future_rets, rel_spread=0.0, cost_ratio=self.cost_ratio
                        )
                selector.update_reward(prev_chosen, delayed_reward, delayed_all_rewards)

            chosen_idx = selector.select_model(t)
            chosen_history.append(chosen_idx)
            pred_signal = float(np.mean(all_model_preds[t, :])) if selector.name == "Baseline-Static-Ensemble" else float(all_model_preds[t, chosen_idx])

            desired_pos = 1 if pred_signal > self.threshold else (-1 if pred_signal < -self.threshold else 0)
            holding_time = t - entry_step
            need_close = current_pos != 0 and (
                holding_time >= self.holding_period or
                (desired_pos != 0 and desired_pos != current_pos)
            )

            if need_close:
                exit_price = prices[t]
                raw_ret = ((exit_price - entry_price) / entry_price) if current_pos == 1 else ((entry_price - exit_price) / entry_price)
                friction = 2.0 * self.cost_ratio
                trade_net = raw_ret - friction
                gross_cum_ret += raw_ret
                net_cum_ret += trade_net
                trades.append({
                    "entry_step": int(entry_step), "exit_step": int(t),
                    "direction": int(current_pos), "gross_pnl": float(raw_ret),
                    "net_pnl": float(trade_net), "friction": float(friction),
                    "holding_seconds": int(holding_time),
                })
                current_pos = 0

            if current_pos == 0 and desired_pos != 0:
                current_pos = desired_pos
                entry_price = prices[t]
                entry_step = t

            if t % 1000 == 0:
                gross_curve.append(float(gross_cum_ret)); net_curve.append(float(net_cum_ret))
                timestamps.append(str(test_df["ts_event"].iloc[t]))

        total_trades = len(trades)
        trade_nets = np.array([x["net_pnl"] for x in trades], dtype=float)
        trade_grosses = np.array([x["gross_pnl"] for x in trades], dtype=float)

        if total_trades:
            wins = trade_nets[trade_nets > 0]; losses = trade_nets[trade_nets <= 0]
            win_rate = float(len(wins) / total_trades)
            avg_win = float(wins.mean()) if len(wins) else 0.0
            avg_loss = abs(float(losses.mean())) if len(losses) else 1e-12
            pl_ratio = float(avg_win / avg_loss)
            curve = np.asarray(net_curve)
            mdd = float(np.max(np.maximum.accumulate(curve) - curve))
            # This is deliberately not labelled annualized Sharpe; 1-day data
            # does not support a defensible annualization assumption.
            trade_sharpe = float(trade_nets.mean() / (trade_nets.std() + 1e-12))
        else:
            win_rate = pl_ratio = mdd = trade_sharpe = 0.0

        sensitivity = {}
        gross_total = float(trade_grosses.sum()) if total_trades else 0.0
        for bps in (0.0, 1.0, 3.0, 5.0, 10.0):
            sensitivity[f"{bps:g}_bps_per_side"] = round(gross_total - total_trades * 2.0 * bps * 1e-4, 6)

        return {
            "strategy": selector.name,
            "total_trades": int(total_trades),
            "win_rate": round(win_rate, 4),
            "pl_ratio": round(pl_ratio, 3),
            "gross_return": round(float(gross_cum_ret), 6),
            "net_return_assumed_cost": round(float(net_cum_ret), 6),
            "assumed_cost_bps_per_side": self.cost_bps_per_side,
            "total_assumed_friction": round(float(total_trades * 2.0 * self.cost_ratio), 6),
            "max_drawdown_assumed_cost": round(mdd, 6),
            "trade_sharpe_nonannualized": round(trade_sharpe, 4),
            "cost_sensitivity_net_return": sensitivity,
            "action_distribution": selector.get_selection_distribution(),
            "gross_curve": gross_curve,
            "net_curve_assumed_cost": net_curve,
            "timestamps": timestamps,
        }
