"""验证时间执行的因果性质，而非仅检查输出字段或镜像实现。"""
import unittest

import numpy as np
import pandas as pd

from src.irl_reward import IRLRewardLearner
from src.model_selector import (CausalEventUCBSelector, FlatSelector, SingleModelSelector)
from src.snapshot_dataset import SessionCalendar, build_session_dataset
from src.time_execution import TimeExecutionEngine


BASE = pd.Timestamp('2025-09-22T00:00:00Z')


class Model:
    """冻结预测由测试直接提供，排除训练随机性对时间性质的干扰。"""
    name = 'frozen'


def quote(stamp, price):
    """构造边界前 100ms 到达的完整五档盘口，价格与序号不携带未来信息。"""
    received = stamp - pd.Timedelta(milliseconds=100)
    result = dict(ts_event=stamp, source_ts_recv=received, source_ts_event=received,
        source_flags=128, age_ms=100., sequence=1, instrument_id=294973, symbol='ESZ5')
    for i in range(5):
        result.update({f'bid_px_{i:02d}': price - .25 * i,
                       f'ask_px_{i:02d}': price + .25 * (i + 1),
                       f'bid_sz_{i:02d}': 10, f'ask_sz_{i:02d}': 12})
    return result


def prepared(stamps=None, prices=None):
    """使用真实数据准备方法预热特征；缺口后不能伪造已有 30 个历史网格。"""
    stamps = list(pd.date_range(BASE, periods=100, freq='500ms')) if stamps is None else list(stamps)
    prices = [100. + .25 * i for i in range(len(stamps))] if prices is None else prices
    calendar = SessionCalendar()
    raw = pd.DataFrame([quote(t, p) for t, p in zip(stamps, prices)])
    assigned, _, _ = calendar.assign(raw)
    frames = [build_session_dataset(part, calendar, 500, (1000, 3000))[0]
              for _, part in assigned.groupby('session_id', sort=False)]
    return pd.concat(frames, ignore_index=True)


class TimeExecutionTests(unittest.TestCase):
    """检查实际可见信息、毫秒延迟和美元账本守恒，所有样本为合成小窗口。"""
    def run_replay(self, frame, *, reward_type='ME', predictions=None, cash=False, **settings):
        """每次创建全新固定选择器，避免上一回放残留反馈。"""
        selector = (FlatSelector if cash else SingleModelSelector)('fixed', [Model()])
        selector.reward_type = reward_type
        reward = IRLRewardLearner(horizons=[1000, 3000], definition='paper_price_difference')
        engine = TimeExecutionEngine(reward, **settings)
        preds = np.full((len(frame), 1), .01) if predictions is None else predictions
        return engine.run_backtest(selector, frame, preds, detail=True)

    def test_non_grid_latency_executes_after_deadline_not_next_row(self):
        """750ms 延迟在半秒网格至少等待两行，不能被静默换成一行延迟。"""
        result = self.run_replay(prepared(), latency_ms=750)
        first = result['fills'][0]
        self.assertEqual(pd.Timestamp(first['intent_time']), BASE + pd.Timedelta(seconds=15))
        self.assertEqual(pd.Timestamp(first['ts_event']), BASE + pd.Timedelta(seconds=16))
        self.assertGreaterEqual(pd.Timestamp(first['ts_event']), pd.Timestamp(first['execution_due_time']))

    def test_holding_review_is_elapsed_time_and_costs_are_conserved(self):
        """方向已反转仍需等待 5 秒复核；毛利减全部成本等于净利。"""
        frame = prepared()
        preds = np.full((len(frame), 1), .01)
        preds[35:] = -.01
        result = self.run_replay(frame, predictions=preds, holding_review_ms=5000)
        first = result['trades'][0]
        self.assertEqual(first['holding_ms'], 5000.)
        self.assertEqual(pd.Timestamp(first['exit_time']), BASE + pd.Timedelta(seconds=20.5))
        self.assertAlmostEqual(result['gross_pnl_usd'] - result['friction_usd'], result['net_pnl_usd'])
        self.assertAlmostEqual(sum(t['net_pnl_usd'] for t in result['trades']), result['net_pnl_usd'])
        self.assertAlmostEqual(sum(f['cost_usd'] for f in result['fills']), result['friction_usd'])
        self.assertEqual(result['terminal_position'], 0)
        self.assertNotIn('holding_events', first)

    def test_me_oe_wait_from_distinct_origin_and_use_exact_prices(self):
        """ME 从预测计时，OE 从成交计时；完整向量等待最长 3 秒，价格差可手算。"""
        frame = prepared()
        me, oe = self.run_replay(frame), self.run_replay(frame, reward_type='OE')
        self.assertEqual(pd.Timestamp(me['reward_observations'][0]['observed_time']), BASE + pd.Timedelta(seconds=18))
        self.assertEqual(pd.Timestamp(oe['reward_observations'][0]['observed_time']), BASE + pd.Timedelta(seconds=18.5))
        for result in (me, oe):
            for observation in result['reward_observations']:
                self.assertGreaterEqual(pd.Timestamp(observation['observed_time']), pd.Timestamp(observation['due_time']))
                self.assertAlmostEqual(observation['reward'], 1.)
        self.assertEqual(oe['unmatured_fill_rewards_at_end'], 1)  # 期末退出尚未到期，不能补写奖励。

    def test_future_labels_never_gate_decisions_or_fills(self):
        """抹掉全部离线标签和 learning_ready，回放仍应逐项一致。"""
        frame = prepared()
        original = self.run_replay(frame)
        for name in frame.columns:
            if name.startswith('future_'):
                frame[name] = np.nan
            if name.startswith('label_valid_') or name == 'learning_ready':
                frame[name] = False
        self.assertEqual(original, self.run_replay(frame))

    def test_changing_future_prices_does_not_change_past_feedback(self):
        """改变 25 秒后的价格，25 秒之前的决策、成交、成熟反馈均不变。"""
        stamps = list(pd.date_range(BASE, periods=100, freq='500ms'))
        prices = np.arange(100) * .25 + 100.
        first = self.run_replay(prepared(stamps, prices))
        prices[51:] += 100.
        second = self.run_replay(prepared(stamps, prices))
        cutoff = BASE + pd.Timedelta(seconds=25)
        for key, clock in [('decisions', 'ts_event'), ('fills', 'ts_event'), ('reward_observations', 'observed_time')]:
            self.assertEqual([r for r in first[key] if pd.Timestamp(r[clock]) <= cutoff],
                             [r for r in second[key] if pd.Timestamp(r[clock]) <= cutoff])

    def test_gap_cancels_old_intents_and_exits_at_first_arriving_quote(self):
        """缺口必须在看到下一条行情时退出，损失计入账本，不在缺口前有利价格强平。"""
        stamps = list(pd.date_range(BASE, periods=100, freq='500ms'))
        del stamps[35:39]
        prices = np.array([100. + (t - BASE).total_seconds() * .5 for t in stamps])
        prices[35:] -= 20.
        result = self.run_replay(prepared(stamps, prices), latency_ms=750)
        exit_fill = next(f for f in result['fills'] if f['execution_reason'] == 'gap')
        self.assertEqual(pd.Timestamp(exit_fill['ts_event']), BASE + pd.Timedelta(seconds=19.5))
        self.assertLess(result['trades'][0]['gross_pnl_usd'], 0)
        self.assertGreater(result['cancelled_intents']['gap'], 0)
        self.assertTrue(all(pd.Timestamp(f['ts_event']) >= BASE + pd.Timedelta(seconds=34.5)
            for f in result['fills'][2:]))

    def test_missing_reward_target_is_not_replaced_with_later_row_or_zero(self):
        """16 秒目标格缺失时，15 秒预测不借用 16.5 秒价格，也不送一条假零反馈。"""
        stamps = list(pd.date_range(BASE, periods=100, freq='500ms'))
        del stamps[32]
        result = self.run_replay(prepared(stamps))
        origin = (BASE + pd.Timedelta(seconds=15)).isoformat()
        self.assertFalse(any(o['origin_time'] == origin for o in result['reward_observations']))
        self.assertGreater(result['reward_status']['ME'].get('missing_target', 0), 0)
        self.assertEqual(sum(result['reward_status']['ME'].values()), result['decision_count'])
        self.assertEqual(sum(result['reward_status']['OE'].values()), result['total_fills'])

    def test_scheduled_break_exit_uses_known_calendar_not_future_rows(self):
        """日历已知暂停前退出；close 快照不能被当成可成交行情。"""
        before = list(pd.date_range('2025-09-22T20:14:35Z', periods=51, freq='500ms'))
        after = list(pd.date_range('2025-09-22T20:30:00.500Z', periods=80, freq='500ms'))
        result = self.run_replay(prepared(before + after))
        exit_fill = next(f for f in result['fills'] if f['execution_reason'] == 'scheduled_exit')
        self.assertEqual(pd.Timestamp(exit_fill['ts_event']), pd.Timestamp('2025-09-22T20:14:59.500Z'))
        self.assertFalse(any(pd.Timestamp(f['ts_event']) == pd.Timestamp('2025-09-22T20:15:00Z')
                             for f in result['fills']))

    def test_missing_scheduled_exit_quote_does_not_backdate_liquidation(self):
        """暂停前退出格未提供时，承担持仓价格风险直到暂停后的首条可交易行情。"""
        before = list(pd.date_range('2025-09-22T20:14:35Z', periods=49, freq='500ms'))
        after = list(pd.date_range('2025-09-22T20:30:00.500Z', periods=80, freq='500ms'))
        result = self.run_replay(prepared(before + after))
        fill = next(f for f in result['fills'] if f['execution_reason'] == 'scheduled_break')
        self.assertEqual(pd.Timestamp(fill['ts_event']), after[0])
        self.assertNotIn('scheduled_exit', result['risk_exits'])

    def test_session_transition_does_not_compress_overnight_into_one_step(self):
        """跨 session 撤销旧意图并按新到行情退出，不能沿用上一日的待执行预测。"""
        before = list(pd.date_range(BASE, periods=80, freq='500ms'))
        after = list(pd.date_range('2025-09-22T22:00:00.500Z', periods=80, freq='500ms'))
        result = self.run_replay(prepared(before + after))
        fill = next(f for f in result['fills'] if f['execution_reason'] == 'session_transition')
        self.assertEqual(pd.Timestamp(fill['ts_event']), after[0])
        self.assertGreater(result['trades'][0]['holding_ms'], 3600000)

    def test_no_tradable_terminal_quote_keeps_position_and_unrealized_pnl(self):
        """退出格缺失且最后一行是关闭边界时，不能伪造成交或隐藏未平仓风险。"""
        stamps = list(pd.date_range('2025-09-22T20:14:35Z', periods=51, freq='500ms'))
        del stamps[-2]
        result = self.run_replay(prepared(stamps))
        self.assertEqual(result['terminal_position'], 1)
        self.assertFalse(result['terminal_position_liquidated'])
        self.assertEqual(result['total_fills'], 1)
        self.assertEqual(result['total_trades'], 0)
        self.assertGreater(result['gross_pnl_usd'], 0)

    def test_cash_and_rejected_event_protocols(self):
        """现金保持零成本零盈亏；拒绝直接移植事件选择器与非法时间配置。"""
        frame = prepared()
        result = self.run_replay(frame, cash=True)
        self.assertEqual((result['total_fills'], result['net_pnl_usd']), (0, 0.))
        self.assertFalse(result['order_feature_expectation_defined'])
        reward = IRLRewardLearner(horizons=[1000, 3000], definition='paper_price_difference')
        with self.assertRaisesRegex(ValueError, 'static'):
            TimeExecutionEngine(reward).run_backtest(CausalEventUCBSelector('event', [Model()]), frame,
                                                    np.ones((len(frame), 1)))
        for settings in [{'latency_ms': 0}, {'holding_review_ms': -1}, {'threshold': float('nan')}]:
            with self.assertRaises(ValueError):
                TimeExecutionEngine(reward, **settings)
        late = frame.copy()
        late.loc[0, 'source_ts_recv'] = late.loc[0, 'ts_event']
        with self.assertRaisesRegex(ValueError, 'arrived'):
            self.run_replay(late)


if __name__ == '__main__':
    unittest.main()
