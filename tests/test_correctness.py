"""关键正确性回归测试：用可手算的合成行情约束因果时序、资金账本与学习行为。

测试不要求策略盈利，而要求它遵守真实可用信息与确定的会计恒等式。
建议学习时按“构造行情 → 运行回放 → 阅读断言”理解每个案例：
因果性检查历史不受未来变化影响，记账检查资金守恒，数据检查标签隔离，
学习检查延迟冷启动、时间窗过期和奖励匹配边界。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from src.config import HORIZON_EVENTS
from src.data_loader import extract_microstructure_features, prepare_train_test_split, load_and_preprocess_events
from src.execution_engine import ExecutionEngine
from src.irl_reward import IRLRewardLearner
from src.model_library import LogisticDirectionModel
from src.model_selector import (CausalShadowARSSelector, CausalEventUCBSelector, SingleModelSelector,
                                EnsembleSelector, RandomSelector)

MODELS = [SimpleNamespace(name='long'), SimpleNamespace(name='short')]


def quotes(mid):
    """按给定中间价序列构造逐秒五档盘口，返回引擎与特征工程可直接读取的表。

    买卖一分别位于 mid±0.125，因此点差固定为 0.25；深档每档再远离 0.25。
    全部样本用同一合约和固定数量，排除无关噪声，使预期成交成本可以手算。"""
    mid = np.asarray(mid, float)
    frame = pd.DataFrame(dict(ts_event=pd.date_range('2025-01-01', periods=len(mid), freq='s', tz='UTC'),
                              sequence=np.arange(len(mid)), instrument_id=1, symbol='SYNTHETIC', mid_price=mid))
    for i in range(5):
        frame[f'bid_px_{i:02d}'] = mid - .125 - i * .25
        frame[f'ask_px_{i:02d}'] = mid + .125 + i * .25
        frame[f'bid_sz_{i:02d}'] = 5
        frame[f'ask_sz_{i:02d}'] = 5
    return frame


def replay(frame, selector=None, preds=None, **kwargs):
    """用固定的两列多/空预测运行 ES 合约回放，默认开启详细审计输出。

    kwargs 可覆盖持仓、数量等假设；奖励尺度固定为 0.01，让测试只检查
    时序与记账本身，不受训练数据或拟合过程影响。"""
    selector = selector or SingleModelSelector('fixed', MODELS)
    preds = np.tile([.001, -.001], (len(frame), 1)) if preds is None else preds
    engine = ExecutionEngine('CME_ES', IRLRewardLearner(scales=[.01] * 3), **kwargs)
    return engine.run_backtest(selector, frame, all_model_preds=preds, detail=True)


class CausalityTests(unittest.TestCase):
    """因果性：更晚价格和离线未来标签不能改变此前已作出的决策及成交。"""
    def test_future_prices_cannot_change_past_actions_or_fills(self):
        """保持前 110 步相同，仅令之后价格向相反方向变化；四种选择组合的历史必须一致。"""
        for cls in (CausalShadowARSSelector, CausalEventUCBSelector):
            for reward in ('ME', 'OE'):
                with self.subTest(selector=cls.__name__, reward=reward):
                    left = quotes(np.r_[np.ones(110) * 100, np.ones(130) * 102])
                    right = quotes(np.r_[np.ones(110) * 100, np.ones(130) * 98])
                    a = replay(left, cls('test', MODELS, reward))
                    b = replay(right, cls('test', MODELS, reward))
                    self.assertEqual(a['actions'][:110], b['actions'][:110])
                    self.assertEqual([f for f in a['fills'] if f['step'] < 110],
                                     [f for f in b['fills'] if f['step'] < 110])

    def test_future_label_columns_are_never_read(self):
        """给行情附加全为无穷大的未来标签；若引擎只读实时行情，两次输出仍应完全一致。"""
        frame = quotes(np.linspace(100, 101, 210))
        a = replay(frame, CausalShadowARSSelector('test', MODELS))
        for h in HORIZON_EVENTS:
            frame[f'future_ret_{h}'] = np.inf
        b = replay(frame, CausalShadowARSSelector('test', MODELS))
        self.assertEqual(a, b)

    def test_feedback_is_delayed_and_owned_by_originating_action(self):
        """ME 反馈必须恰好延迟 90 步，并归属产生信号时选中的模型。"""
        result = replay(quotes(np.linspace(100, 101, 230)), CausalEventUCBSelector('test', MODELS))
        self.assertTrue(result['reward_observations'])
        for event in result['reward_observations']:
            self.assertEqual(event['observed_step'], event['origin_step'] + 90)
            self.assertEqual(event['owner'], result['actions'][event['origin_step']])

    def test_oe_event_feedback_includes_no_fill_decisions(self):
        """每个成熟决策恰好一条观测：成交均值或无成交零值，计数不能混用。"""
        selector = CausalEventUCBSelector('test', MODELS, 'OE')
        result = replay(quotes(np.ones(230) * 100), selector,
                        preds=np.ones((230, 2)) * .001, holding_period=1000)
        observations = result['reward_observations']
        self.assertEqual(len(observations), 230 - 1 - 90)
        self.assertEqual(sum(o['fill_count'] for o in observations), 1)
        self.assertEqual(int(selector.feedback_counts.sum()), len(observations))
        for event in observations:
            self.assertEqual(event['observed_step'], event['origin_step'] + 91)
            self.assertEqual(event['owner'], result['actions'][event['origin_step']])
            if not event['fill_count']:
                self.assertEqual(event['reward'], 0.)

    def test_no_trade_model_receives_zero_ucb_observations(self):
        """始终不下单的模型仍会得到到期零值，而不是永远没有反馈却不断增加探索数。"""
        selector = CausalEventUCBSelector('test', MODELS, 'OE')
        result = replay(quotes(np.ones(230) * 100), selector, preds=np.zeros((230, 2)))
        self.assertEqual(result['total_fills'], 0)
        self.assertTrue((selector.feedback_counts > 0).all())
        self.assertEqual(selector.feedback_counts.sum(), 139)
        np.testing.assert_array_equal(selector.sum_rewards, [0, 0])
        self.assertFalse(result['order_feature_expectation_defined'])
        self.assertEqual(result['order_feature_expectation'], [0., 0., 0.])

    def test_event_oe_groups_reversal_fills_and_assigns_decision_owner(self):
        """反转一次会平仓再开仓，但只反馈一次均值，归属触发反转的决策模型。"""
        selector = RandomSelector('test', MODELS, seed=7)
        selector.reward_type = 'OE'
        frame = quotes(np.linspace(100, 103, 230))
        result = replay(frame, selector)
        learner = IRLRewardLearner(scales=[.01] * 3)
        saw_reversal = False
        mid = frame.mid_price.to_numpy()
        for event in result['reward_observations']:
            step = event['executed_step']
            fills = [f for f in result['fills'] if f['step'] == step]
            self.assertEqual(len(fills), event['fill_count'])
            saw_reversal |= len(fills) == 2
            vectors = [learner.features_from_prices(mid[step], mid[step + np.array(HORIZON_EVENTS)],
                                                    f['side'], f['cost_ratio']) for f in fills]
            expected = learner.score(np.mean(vectors, axis=0)) if vectors else 0.
            self.assertAlmostEqual(event['reward'], expected)
            self.assertEqual(event['owner'], result['actions'][event['origin_step']])
        self.assertTrue(saw_reversal)

    def test_oe_shadow_accounts_include_unselected_models(self):
        """检查 ARS 的 OE 能获得两个独立影子模型的反馈，不只观察实际所选模型。"""
        result = replay(quotes(np.linspace(100, 102, 230)), CausalShadowARSSelector('test', MODELS, 'OE'))
        self.assertEqual({o['owner'] for o in result['reward_observations']}, {0, 1})

    def test_next_event_execution_uses_later_quote(self):
        """价格逐步跳涨时，第一笔必须在第 1 步按 101.25 成交，而不能使用第 0 步旧价。"""
        result = replay(quotes([100, 101, 102, 103]), holding_period=100)
        self.assertEqual(result['fills'][0]['step'], 1)
        self.assertEqual(result['fills'][0]['price'], 101.25)
        self.assertEqual(result['terminal_position'], 0)
        self.assertEqual(result['fills'][-1]['step'], 3)


class AccountingTests(unittest.TestCase):
    """账户正确性：成交、完整交易、资金曲线三个层次必须能相互对账。"""
    def test_round_trip_exact_costs_and_terminal_liquidation(self):
        """平价行情毛利为零；双边成本应为 27.5 美元，且终点必须完成平仓并进入曲线。"""
        result = replay(quotes(np.ones(10) * 100), holding_period=100)
        self.assertEqual(result['total_trades'], 1)
        self.assertEqual(result['total_fills'], 2)
        # 每边：(半点差 0.125 + 滑点 0.5×0.25)×点值 50 + 手续费 1.25 = 13.75。
        # 一次开仓加一次平仓，所以总成本为 2×13.75=27.5。
        self.assertAlmostEqual(result['friction_usd'], 27.5)
        self.assertAlmostEqual(result['net_pnl_usd'], -27.5)
        self.assertEqual(result['gross_pnl_usd'], 0)
        self.assertAlmostEqual(result['net_return'], -27.5 / 100000)
        self.assertEqual(result['net_curve'][-1], result['net_return'])
        self.assertIsNone(result['sharpe_ratio'])

    def test_quantity_scales_dollars_not_price_return(self):
        """三张合约的美元成本和盈亏都应是一张的三倍，验证乘数与张数使用一致。"""
        a = replay(quotes([100, 100, 101, 102]), quantity=1)
        b = replay(quotes([100, 100, 101, 102]), quantity=3)
        self.assertAlmostEqual(b['net_pnl_usd'], 3 * a['net_pnl_usd'])
        self.assertAlmostEqual(b['friction_usd'], 3 * a['friction_usd'])

    def test_short_and_long_symmetric_pnl(self):
        """对称的上涨多头与下跌空头应获得相同盈亏，避免空头方向符号写反。"""
        a = replay(quotes([100, 100, 101, 102]))
        b = replay(quotes([100, 100, 99, 98]), SingleModelSelector('short', MODELS, 1))
        self.assertAlmostEqual(a['net_pnl_usd'], b['net_pnl_usd'])
        self.assertAlmostEqual(a['gross_pnl_usd'], 100.)

    def test_drawdown_includes_open_position_loss_between_samples(self):
        """价格先跌后回到原点，最终毛利虽为零，中途浮亏仍必须进入最大回撤。"""
        frame = quotes([100, 100, 95, 100, 100])
        result = replay(frame, holding_period=100)
        self.assertGreater(result['max_drawdown_usd'], 250.)
        self.assertGreater(result['max_drawdown_usd'], -result['net_pnl_usd'])

    def test_ledger_reconciles_on_reversals(self):
        """频繁反向切仓时，逐笔净利之和等于最终净利，逐成交成本之和等于总成本。"""
        frame = quotes(100 + np.sin(np.arange(250) / 8))
        predictions = np.tile([.001, -.001], (250, 1))
        result = replay(frame, RandomSelector('random', MODELS), predictions)
        self.assertAlmostEqual(sum(t['net_pnl_usd'] for t in result['trades']), result['net_pnl_usd'])
        self.assertAlmostEqual(sum(f['cost_usd'] for f in result['fills']), result['friction_usd'])
        self.assertEqual(result['total_fills'], 2 * result['total_trades'])
        json.dumps(result, allow_nan=False)

    def test_no_trade_is_zero_and_ensemble_has_no_fake_selection(self):
        """相反预测等权抵消后不应交易，也不应把集成伪装成全程选择第一个模型。"""
        result = replay(quotes(np.ones(110) * 100), EnsembleSelector('ensemble', MODELS))
        self.assertEqual(result['net_pnl_usd'], 0.)
        self.assertEqual(result['action_distribution'], {})


class DataTests(unittest.TestCase):
    """数据边界：标签不能跨段，尾部行情不能丢，同时间事件与合约不能混淆。"""
    def test_purged_training_is_invariant_to_later_prices(self):
        """只改训练边界后的价格，清除跨段标签后训练特征与目标都应保持不变。"""
        raw = quotes(100 + np.sin(np.arange(4000) / 50))
        frame, features = extract_microstructure_features(raw.copy())
        a = prepare_train_test_split(frame, features)
        raw.loc[2000:, ['bid_px_00', 'ask_px_00']] += 10
        other, features = extract_microstructure_features(raw)
        b = prepare_train_test_split(other, features)
        np.testing.assert_array_equal(a['y_train'], b['y_train'])
        np.testing.assert_array_equal(a['X_train'], b['X_train'])
        self.assertEqual(len(a['test_df']), 800)
        self.assertEqual(int(a['test_df'].target_ret.isna().sum()), 30)

    def test_all_partition_label_endpoints_precede_next_partition(self):
        """逐个核对前三段最晚标签的行号终点严格早于下一段，并且训练标签均有限。"""
        frame, features = extract_microstructure_features(quotes(np.linspace(100, 101, 4000)))
        split = prepare_train_test_split(frame, features)
        for a, b in [('train', 'calibration'), ('calibration', 'validation'), ('validation', 'test')]:
            endpoint = split[f'{a}_df'].index[-1] + 90
            self.assertLess(endpoint, split[f'{b}_df'].index[0])
            self.assertTrue(np.isfinite(split[f'y_{a}']).all())

    def test_loader_orders_ties_and_rejects_mixed_instruments(self):
        """制造同时间且乱序的事件检查序号排序；再混入第二合约，必须明确拒绝。"""
        raw = quotes(np.ones(10) * 100)
        raw.ts_event = raw.ts_event.iloc[0]
        raw = raw.iloc[::-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quotes.parquet'
            raw.to_parquet(path)
            result = load_and_preprocess_events(str(path), 10)
            self.assertEqual(result.sequence.tolist(), list(range(10)))
            raw.loc[0, 'instrument_id'] = 2
            raw.to_parquet(path)
            with self.assertRaises(ValueError):
                load_and_preprocess_events(str(path), 10)

    def test_loader_bounded_offset_and_empty_window(self):
        """检查原始行偏移只取请求范围，超出文件尾部时明确报错，而不是返回伪造空实验。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quotes.parquet'
            quotes(np.ones(20) * 100).to_parquet(path)
            self.assertEqual(load_and_preprocess_events(str(path), 5, 10).sequence.tolist(), list(range(10, 15)))
            with self.assertRaises(ValueError):
                load_and_preprocess_events(str(path), 5, 30)


class LearningTests(unittest.TestCase):
    """选择与奖励学习：检查已知正确的简单情形及容易混淆的边界条件。"""
    def test_delayed_ucb_cold_start_does_not_stick_to_first_arm(self):
        """不提供任何反馈时，UCB 仍应按已选择计数探索两个模型，不被延迟卡在第一臂。"""
        selector = CausalEventUCBSelector('ucb', MODELS)
        self.assertEqual([selector.select_model(t) for t in range(4)], [0, 1, 0, 1])

    def test_scaled_ucb_learns_better_arm(self):
        """持续给一个模型正奖励、另一个负奖励，验证探索项不会压过已知明显的收益差。"""
        selector = CausalEventUCBSelector('ucb', MODELS, c=.1)
        for t in range(300):
            arm = selector.select_model(t)
            selector.observe(arm, .8 if arm == 1 else -.8, t)
        self.assertGreater(selector.counts[1], 290)

    def test_ars_expires_stale_observations_by_event_time(self):
        """停止新反馈后继续推进事件步，旧奖励也必须按时间过期，不能永久保留。"""
        selector = CausalShadowARSSelector('ars', MODELS, window_events=3)
        selector.observe(1, .9, 0)
        self.assertEqual(selector.select_model(1), 1)
        selector.select_model(4)
        self.assertFalse(any(selector.queues))

    def test_random_seed_is_local_and_reproducible(self):
        """两个同种子实例应产生相同动作，即使外部修改了 NumPy 全局随机种子。"""
        a, b = RandomSelector('a', MODELS, 7), RandomSelector('b', MODELS, 7)
        np.random.seed(999)
        self.assertEqual([a.select_model(t) for t in range(100)], [b.select_model(t) for t in range(100)])

    def test_reward_learning_updates_policy_and_checks_oracle_gap(self):
        """检查奖励学习确实加入最强策略约束并迭代，而非重复一个固定梯度。"""
        learner = IRLRewardLearner(horizons=[10, 30])
        weights = learner.fit_reward_weights([[0, 0], [1, 0], [0, 1]], 1)
        self.assertTrue(learner.diagnostics['converged'])
        self.assertGreaterEqual(len(learner.diagnostics['history']), 2)
        self.assertAlmostEqual(weights.sum(), 1.)
        final = learner.diagnostics['history'][-1]
        self.assertLessEqual(final['margin'] - final['expert_minus_oracle'], 1e-8)

    def test_reward_scale_fit_is_training_only_and_bounded(self):
        """用已知训练收益拟合尺度，再给极端输入，最终标量奖励仍应被限制在 [-1,1]。"""
        learner = IRLRewardLearner()
        frame = pd.DataFrame({f'future_ret_{h}': [.001, -.001] for h in HORIZON_EVENTS})
        learner.fit_scales(frame)
        self.assertEqual(learner.score(learner.features([100, 100, 100])), 1.)
        self.assertEqual(learner.score(learner.features([-100, -100, -100])), -1.)

    def test_solver_convergence_does_not_claim_infeasible_expert_is_matched(self):
        """专家在所有尺度都被其他策略支配时，求解可收敛，但必须报告专家不可表示。"""
        learner = IRLRewardLearner(horizons=[10, 30])
        learner.fit_reward_weights([[0, 0], [1, 2], [2, 1]], 0)
        self.assertTrue(learner.diagnostics['converged'])
        self.assertFalse(learner.diagnostics['expert_representable'])
        self.assertLess(learner.diagnostics['final_margin'], 0.)

    def test_logistic_handles_flat_class_and_single_class_data(self):
        """先检查全平盘常数回退，再确认混合样本显式包含跌、平、涨三个类别。"""
        X = np.arange(60).reshape(20, 3)
        model = LogisticDirectionModel()
        model.fit(X, np.zeros(20))
        np.testing.assert_array_equal(model.predict(X), np.zeros(20))
        y = np.tile([-.01, 0, 0, .01], 5)
        model.fit(X, y)
        self.assertEqual(set(model.model.classes_), {-1, 0, 1})
        self.assertTrue(np.isfinite(model.predict(X)).all())


if __name__ == '__main__':
    unittest.main()
