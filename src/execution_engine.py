"""
高频交易模拟执行与回测引擎模块 (execution_engine.py)
模拟高频挂单/吃单撮合逻辑，真实扣减买卖半点差滑点与交易所手续费，
计算毛收益、净收益、最大回撤、夏普比率、胜率等核心量化指标。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.config import RL_CONFIG, INSTRUMENT_CONFIG, HORIZONS
from src.model_selector import BaseSelector
from src.irl_reward import IRLRewardLearner


class ExecutionEngine:
    """
    高频交易执行回测引擎
    """
    def __init__(self, instrument_key: str, irl_learner: IRLRewardLearner):
        self.instrument_key = instrument_key
        self.config = INSTRUMENT_CONFIG.get(instrument_key, INSTRUMENT_CONFIG["CME_ES"])
        self.irl_learner = irl_learner
        self.threshold = self.config.get("trade_threshold", RL_CONFIG["trade_threshold"])
        self.holding_period = RL_CONFIG["holding_period"]

    def run_backtest(
        self,
        selector: BaseSelector,
        test_df: pd.DataFrame,
        X_test: np.ndarray,
        all_model_preds: np.ndarray
    ) -> Dict[str, Any]:
        """
        在时序测试集上运行完整的在线选择与高频模拟回测

        参数:
            selector (BaseSelector): 策略模型选择器 (如 FMATO-ME-UCBS)
            test_df (pd.DataFrame): 测试集行情数据 (包含买卖一档、中间价及多尺度标签)
            X_test (np.ndarray): 测试集特征矩阵
            all_model_preds (np.ndarray): 所有候选模型在测试集上的预测矩阵 (N, K)

        返回:
            Dict[str, Any]: 包含各项量化指标与净值曲线的详尽结果字典
        """
        n_steps = len(test_df)
        k_models = selector.k

        mid_prices = test_df['mid_price'].values
        spreads = test_df['spread'].values
        rel_spreads = test_df['rel_spread'].values

        # 提取多尺度真实未来收益率矩阵用于反馈奖励
        future_rets_matrix = test_df[[f'future_ret_{h}' for h in HORIZONS]].values

        # 交易与持仓状态记录
        current_pos = 0        # 当前持仓: +1 多头, -1 空头, 0 空仓
        entry_price = 0.0      # 开仓成交中间价
        entry_step = 0         # 开仓时步
        entry_friction = 0.0   # 开仓滑点与手续费成本

        # 逐笔交易收益与时序净值追踪
        trades = []            # 完成平仓的逐笔交易明细
        gross_cum_ret = 0.0    # 累计毛收益
        net_cum_ret = 0.0      # 累计净收益
        
        gross_curve = [0.0]    # 毛收益净值曲线
        net_curve = [0.0]      # 净收益净值曲线
        timestamps = [str(test_df['ts_event'].iloc[0])]

        # 单边手续费折算比例
        cost_ratio = self.config["commission_per_order"] / (mid_prices[0] * self.config["multiplier"])

        # 逐 tick 模拟撮合推进
        for t in range(n_steps - max(HORIZONS)):
            # 1. 选择器做出动作决策 (选择模型)
            chosen_idx = selector.select_model(t)

            # 2. 获取预测信号
            if selector.name == "Baseline-Static-Ensemble":
                pred_signal = float(np.mean(all_model_preds[t, :]))
            else:
                pred_signal = float(all_model_preds[t, chosen_idx])

            # 3. 产生交易意图方向
            desired_pos = 0
            if pred_signal > self.threshold:
                desired_pos = 1
            elif pred_signal < -self.threshold:
                desired_pos = -1

            # 4. 持仓超时或收到反向信号时平仓
            holding_time = t - entry_step
            need_close = (current_pos != 0) and (
                (holding_time >= self.holding_period) or 
                (desired_pos != 0 and desired_pos != current_pos)
            )

            if need_close:
                # 平仓结算: 扣除卖方半点差与手续费
                exit_price = mid_prices[t]
                exit_friction = (rel_spreads[t] * 0.5) + cost_ratio
                
                # 计算单笔毛收益率与净收益率
                raw_ret = (exit_price - entry_price) / entry_price if current_pos == 1 else (entry_price - exit_price) / entry_price
                trade_gross = raw_ret
                trade_friction = entry_friction + exit_friction
                trade_net = trade_gross - trade_friction

                gross_cum_ret += trade_gross
                net_cum_ret += trade_net

                trades.append({
                    "entry_step": entry_step,
                    "exit_step": t,
                    "direction": current_pos,
                    "gross_pnl": trade_gross,
                    "net_pnl": trade_net,
                    "friction": trade_friction,
                    "holding_ticks": holding_time,
                })
                current_pos = 0

            # 5. 若当前空仓且有有效交易信号，执行开仓
            if current_pos == 0 and desired_pos != 0:
                current_pos = desired_pos
                entry_price = mid_prices[t]
                entry_step = t
                entry_friction = (rel_spreads[t] * 0.5) + cost_ratio

            # 6. 计算强化学习反馈奖励并更新选择器
            curr_future_rets = future_rets_matrix[t]
            
            # 计算当前选定模型的即时奖励
            if selector.reward_type == "ME":
                # 模型预测期望奖励
                r_t = self.irl_learner.compute_model_expectation(pred_signal, curr_future_rets)
            else:
                # 订单成交期望奖励
                r_t = self.irl_learner.compute_order_traded_expectation(
                    desired_pos, curr_future_rets, rel_spreads[t], cost_ratio
                )

            # 同时为 ARS 影子回测计算所有模型的即时奖励
            all_r_t = np.zeros(k_models)
            for m_i in range(k_models):
                p_i = float(all_model_preds[t, m_i])
                if selector.reward_type == "ME":
                    all_r_t[m_i] = self.irl_learner.compute_model_expectation(p_i, curr_future_rets)
                else:
                    d_i = 1 if p_i > self.threshold else (-1 if p_i < -self.threshold else 0)
                    all_r_t[m_i] = self.irl_learner.compute_order_traded_expectation(
                        d_i, curr_future_rets, rel_spreads[t], cost_ratio
                    )

            # 更新在线选择器
            selector.update_reward(chosen_idx, r_t, all_r_t)

            # 定期采样净值曲线点 (每 50 ticks 记录一个点，避免前端图表数据过载)
            if t % 50 == 0:
                gross_curve.append(float(gross_cum_ret))
                net_curve.append(float(net_cum_ret))
                timestamps.append(str(test_df['ts_event'].iloc[t]))

        # 统计回测核心指标
        total_trades = len(trades)
        if total_trades > 0:
            trade_nets = [tr["net_pnl"] for tr in trades]
            trade_grosses = [tr["gross_pnl"] for tr in trades]
            
            win_trades = [p for p in trade_nets if p > 0]
            loss_trades = [p for p in trade_nets if p <= 0]
            
            win_rate = len(win_trades) / total_trades
            avg_win = float(np.mean(win_trades)) if win_trades else 0.0
            avg_loss = abs(float(np.mean(loss_trades))) if loss_trades else 1e-6
            pl_ratio = avg_win / avg_loss
            
            # 最大回撤 (MDD) 计算
            cum_arr = np.array(net_curve)
            running_max = np.maximum.accumulate(cum_arr)
            drawdowns = running_max - cum_arr
            mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # 年化/步长夏普比率 (基于逐笔净收益)
            std_net = float(np.std(trade_nets))
            sharpe = (float(np.mean(trade_nets)) / (std_net + 1e-8)) * np.sqrt(252 * 1000)
            
            total_friction = sum(tr["friction"] for tr in trades)
        else:
            win_rate = 0.0
            pl_ratio = 0.0
            mdd = 0.0
            sharpe = 0.0
            total_friction = 0.0

        return {
            "strategy": selector.name,
            "total_trades": total_trades,
            "win_rate": round(float(win_rate), 4),
            "pl_ratio": round(float(pl_ratio), 2),
            "gross_return": round(float(gross_cum_ret), 4),
            "net_return": round(float(net_cum_ret), 4),
            "total_friction": round(float(total_friction), 4),
            "max_drawdown": round(float(mdd), 4),
            "sharpe_ratio": round(float(sharpe), 2),
            "action_distribution": selector.get_selection_distribution(),
            "gross_curve": gross_curve,
            "net_curve": net_curve,
            "timestamps": timestamps,
        }
