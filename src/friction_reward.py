"""将实际成交摩擦内化为 OE 奖励的单项扩展，所有奖励仍以价格点计。

论文物理第4页 Eq.(2)–(4) 使用成交后的中间价差，第5页 Eq.(5)–(6)
把期间均值交给 UCB。本扩展仅改为 w·u−c：c 是该次真实成交的单边
美元摩擦除以合约点值和张数，已经在成交时可知，不预估未来退出成本。
校准权重、七尺度、期间分母与 UCB 探索参数均与原版共用。
"""
import numpy as np

from src.sum_only_reward import FrozenSumOnlyReward


FRICTION_EXPERIMENT = dict(
    name='actual_fill_friction_reward_increment',
    control='OE-online_library-UCB',
    treatment='OE-friction-online_library-UCB',
    formula='dot(shared_frozen_weights, raw_OE) - fill_cost_usd / (point_value * quantity)',
    reward_unit='price_points', cost_coefficient=1.0,
    cost_scope='actual_single_fill_spread_slippage_commission',
    cost_visible_at='fill_time', cost_deducted_once_per_fill=True,
    weight_source='shared_original_raw_OE_calibration_no_refit',
    unchanged='models_predictions_gate_horizons_periods_UCB_execution_dates',
    holdout_claim='none_all_dates_used_for_development',
)


class FrozenFrictionOEReward(FrozenSumOnlyReward):
    """共用原校准 H 维权重；仅在完整成熟 OE 的最终标量中减单边成本。

    直接减标量保证任意正负权时仍恰好扣一次，不把美元和价格点相减。
    奖励是扣本次成交摩擦的价格评分，不是整笔开平仓交易的净收益。
    """
    def __init__(self, horizons, weights):
        super().__init__(horizons, weights)
        self.definition = 'friction_adjusted_order_price_difference'
        self.deduct_cost = True

    @staticmethod
    def cost_in_price_points(cost_usd, point_value, quantity):
        """成交美元成本 / (美元每点每张 × 张数)；输入均须有限、单位明确。"""
        if (not np.isfinite([cost_usd, point_value, quantity]).all()
                or cost_usd < 0 or point_value <= 0 or quantity <= 0):
            raise ValueError('Need nonnegative fill cost and positive point value/quantity')
        return float(cost_usd / (point_value * quantity))

    def score(self, features, *, cost_price):
        """features 为完整 H 维原始 OE；成本虽已知，仍等待全部尺度成熟后反馈。"""
        values = np.asarray(features, float)
        if (values.shape != self.weights.shape or not np.isfinite(values).all()
                or not np.isfinite(cost_price) or cost_price < 0):
            raise ValueError('Complete finite OE and actual nonnegative fill cost required')
        return float(np.dot(self.weights, values) - cost_price)
