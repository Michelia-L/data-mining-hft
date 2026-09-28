"""Regression tests for causal replay, accounting, and experiment boundaries."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from src.config import HORIZONS
from src.data_loader import extract_microstructure_features, prepare_train_test_split, load_and_preprocess_ticks
from src.execution_engine import ExecutionEngine
from src.irl_reward import IRLRewardLearner
from src.model_library import LogisticDirectionModel
from src.model_selector import (ARSSelector, UCBSelector, SingleModelSelector,
                                EnsembleSelector, RandomSelector)

MODELS = [SimpleNamespace(name='long'), SimpleNamespace(name='short')]


def quotes(mid):
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
    selector = selector or SingleModelSelector('fixed', MODELS)
    preds = np.tile([.001, -.001], (len(frame), 1)) if preds is None else preds
    engine = ExecutionEngine('CME_ES', IRLRewardLearner(scales=[.01] * 3), **kwargs)
    return engine.run_backtest(selector, frame, all_model_preds=preds, detail=True)


class CausalityTests(unittest.TestCase):
    def test_future_prices_cannot_change_past_actions_or_fills(self):
        for cls in (ARSSelector, UCBSelector):
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
        frame = quotes(np.linspace(100, 101, 210))
        a = replay(frame, ARSSelector('test', MODELS))
        for h in HORIZONS:
            frame[f'future_ret_{h}'] = np.inf
        b = replay(frame, ARSSelector('test', MODELS))
        self.assertEqual(a, b)

    def test_feedback_is_delayed_and_owned_by_originating_action(self):
        result = replay(quotes(np.linspace(100, 101, 230)), UCBSelector('test', MODELS))
        self.assertTrue(result['reward_observations'])
        for event in result['reward_observations']:
            self.assertEqual(event['observed_step'], event['origin_step'] + 90)
            self.assertEqual(event['owner'], result['actions'][event['origin_step']])

    def test_oe_feedback_is_only_created_by_actual_fills(self):
        selector = UCBSelector('test', MODELS, 'OE')
        result = replay(quotes(np.ones(230) * 100), selector,
                        preds=np.ones((230, 2)) * .001, holding_period=1000)
        pairs = {(f['step'], f['owner']) for f in result['fills']}
        self.assertEqual(len(result['reward_observations']), 1)
        for event in result['reward_observations']:
            self.assertIn((event['origin_step'], event['owner']), pairs)
            self.assertEqual(event['observed_step'], event['origin_step'] + 90)

    def test_oe_shadow_accounts_include_unselected_models(self):
        result = replay(quotes(np.linspace(100, 102, 230)), ARSSelector('test', MODELS, 'OE'))
        self.assertEqual({o['owner'] for o in result['reward_observations']}, {0, 1})

    def test_next_event_execution_uses_later_quote(self):
        result = replay(quotes([100, 101, 102, 103]), holding_period=100)
        self.assertEqual(result['fills'][0]['step'], 1)
        self.assertEqual(result['fills'][0]['price'], 101.25)
        self.assertEqual(result['terminal_position'], 0)
        self.assertEqual(result['fills'][-1]['step'], 3)


class AccountingTests(unittest.TestCase):
    def test_round_trip_exact_costs_and_terminal_liquidation(self):
        result = replay(quotes(np.ones(10) * 100), holding_period=100)
        self.assertEqual(result['total_trades'], 1)
        self.assertEqual(result['total_fills'], 2)
        self.assertAlmostEqual(result['friction_usd'], 27.5)
        self.assertAlmostEqual(result['net_pnl_usd'], -27.5)
        self.assertEqual(result['gross_pnl_usd'], 0)
        self.assertAlmostEqual(result['net_return'], -27.5 / 100000)
        self.assertEqual(result['net_curve'][-1], result['net_return'])
        self.assertIsNone(result['sharpe_ratio'])

    def test_quantity_scales_dollars_not_price_return(self):
        a = replay(quotes([100, 100, 101, 102]), quantity=1)
        b = replay(quotes([100, 100, 101, 102]), quantity=3)
        self.assertAlmostEqual(b['net_pnl_usd'], 3 * a['net_pnl_usd'])
        self.assertAlmostEqual(b['friction_usd'], 3 * a['friction_usd'])

    def test_short_and_long_symmetric_pnl(self):
        a = replay(quotes([100, 100, 101, 102]))
        b = replay(quotes([100, 100, 99, 98]), SingleModelSelector('short', MODELS, 1))
        self.assertAlmostEqual(a['net_pnl_usd'], b['net_pnl_usd'])
        self.assertAlmostEqual(a['gross_pnl_usd'], 100.)

    def test_drawdown_includes_open_position_loss_between_samples(self):
        frame = quotes([100, 100, 95, 100, 100])
        result = replay(frame, holding_period=100)
        self.assertGreater(result['max_drawdown_usd'], 250.)
        self.assertGreater(result['max_drawdown_usd'], -result['net_pnl_usd'])

    def test_ledger_reconciles_on_reversals(self):
        frame = quotes(100 + np.sin(np.arange(250) / 8))
        predictions = np.tile([.001, -.001], (250, 1))
        result = replay(frame, RandomSelector('random', MODELS), predictions)
        self.assertAlmostEqual(sum(t['net_pnl_usd'] for t in result['trades']), result['net_pnl_usd'])
        self.assertAlmostEqual(sum(f['cost_usd'] for f in result['fills']), result['friction_usd'])
        self.assertEqual(result['total_fills'], 2 * result['total_trades'])
        json.dumps(result, allow_nan=False)

    def test_no_trade_is_zero_and_ensemble_has_no_fake_selection(self):
        result = replay(quotes(np.ones(110) * 100), EnsembleSelector('ensemble', MODELS))
        self.assertEqual(result['net_pnl_usd'], 0.)
        self.assertEqual(result['action_distribution'], {})


class DataTests(unittest.TestCase):
    def test_purged_training_is_invariant_to_later_prices(self):
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
        frame, features = extract_microstructure_features(quotes(np.linspace(100, 101, 4000)))
        split = prepare_train_test_split(frame, features)
        for a, b in [('train', 'calibration'), ('calibration', 'validation'), ('validation', 'test')]:
            endpoint = split[f'{a}_df'].index[-1] + 90
            self.assertLess(endpoint, split[f'{b}_df'].index[0])
            self.assertTrue(np.isfinite(split[f'y_{a}']).all())

    def test_loader_orders_ties_and_rejects_mixed_instruments(self):
        raw = quotes(np.ones(10) * 100)
        raw.ts_event = raw.ts_event.iloc[0]
        raw = raw.iloc[::-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quotes.parquet'
            raw.to_parquet(path)
            result = load_and_preprocess_ticks(str(path), 10)
            self.assertEqual(result.sequence.tolist(), list(range(10)))
            raw.loc[0, 'instrument_id'] = 2
            raw.to_parquet(path)
            with self.assertRaises(ValueError):
                load_and_preprocess_ticks(str(path), 10)

    def test_loader_bounded_offset_and_empty_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'quotes.parquet'
            quotes(np.ones(20) * 100).to_parquet(path)
            self.assertEqual(load_and_preprocess_ticks(str(path), 5, 10).sequence.tolist(), list(range(10, 15)))
            with self.assertRaises(ValueError):
                load_and_preprocess_ticks(str(path), 5, 30)


class LearningTests(unittest.TestCase):
    def test_delayed_ucb_cold_start_does_not_stick_to_first_arm(self):
        selector = UCBSelector('ucb', MODELS)
        self.assertEqual([selector.select_model(t) for t in range(4)], [0, 1, 0, 1])

    def test_scaled_ucb_learns_better_arm(self):
        selector = UCBSelector('ucb', MODELS, c=.1)
        for t in range(300):
            arm = selector.select_model(t)
            selector.observe(arm, .8 if arm == 1 else -.8, t)
        self.assertGreater(selector.counts[1], 290)

    def test_ars_expires_stale_observations_by_event_time(self):
        selector = ARSSelector('ars', MODELS, window_size=3)
        selector.observe(1, .9, 0)
        self.assertEqual(selector.select_model(1), 1)
        selector.select_model(4)
        self.assertFalse(any(selector.queues))

    def test_random_seed_is_local_and_reproducible(self):
        a, b = RandomSelector('a', MODELS, 7), RandomSelector('b', MODELS, 7)
        np.random.seed(999)
        self.assertEqual([a.select_model(t) for t in range(100)], [b.select_model(t) for t in range(100)])

    def test_reward_learning_updates_policy_and_checks_oracle_gap(self):
        learner = IRLRewardLearner(horizons=[10, 30])
        weights = learner.fit_reward_weights([[0, 0], [1, 0], [0, 1]], 1)
        self.assertTrue(learner.diagnostics['converged'])
        self.assertGreaterEqual(len(learner.diagnostics['history']), 2)
        self.assertAlmostEqual(weights.sum(), 1.)
        final = learner.diagnostics['history'][-1]
        self.assertLessEqual(final['margin'] - final['expert_minus_oracle'], 1e-8)

    def test_reward_scale_fit_is_training_only_and_bounded(self):
        learner = IRLRewardLearner()
        frame = pd.DataFrame({f'future_ret_{h}': [.001, -.001] for h in HORIZONS})
        learner.fit_scales(frame)
        self.assertEqual(learner.score(learner.features([100, 100, 100])), 1.)
        self.assertEqual(learner.score(learner.features([-100, -100, -100])), -1.)

    def test_solver_convergence_does_not_claim_infeasible_expert_is_matched(self):
        learner = IRLRewardLearner(horizons=[10, 30])
        learner.fit_reward_weights([[0, 0], [1, 2], [2, 1]], 0)
        self.assertTrue(learner.diagnostics['converged'])
        self.assertFalse(learner.diagnostics['expert_representable'])
        self.assertLess(learner.diagnostics['final_margin'], 0.)

    def test_logistic_handles_flat_class_and_single_class_data(self):
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
