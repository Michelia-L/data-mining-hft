"""论文定义对齐测试：因果正确不等于复现正确，公式与统计口径需单独约束。

这里不要求复现论文收益数字，而用可手算的价格、订单、门槛和并列候选检验定义。
论文 Algorithm 3 的历史重回测尚未实现，因此不伪造对应测试或宣称已验证它。
"""
import unittest
from types import SimpleNamespace
import numpy as np
from test_correctness import quotes, MODELS
from src.execution_engine import ExecutionEngine
from src.irl_reward import IRLRewardLearner
from src.model_selector import SingleModelSelector, CausalShadowARSSelector
from src.model_library import LogisticDirectionModel
from run_experiments import gated_me_expectation, select_with_ties, evaluate_models


class PaperDefinitionTests(unittest.TestCase):
    """逐项保护 Eq.(2)/(3)/(5)、事件单位、门槛化 ME 和评价口径。"""

    def test_eq2_price_difference_is_not_normalized_or_clipped(self):
        """100 到 102/104/106 的等权价格奖励应为 4，而不是收益率或裁剪后的 1。"""
        learner = IRLRewardLearner(definition='paper_price_difference')
        features = learner.features_from_prices(100, [102, 104, 106], cost_ratio=.2)
        np.testing.assert_array_equal(features, [2, 4, 6])
        self.assertEqual(learner.score(features), 4.)
        net = IRLRewardLearner(definition='paper_price_difference', deduct_cost=True)
        np.testing.assert_array_equal(net.features_from_prices(100, [102, 104, 106], cost_ratio=.01), [1, 3, 5])

    def test_eq5_order_average_invariant_to_order_count(self):
        """平价净奖励每笔都为负单边费用；交易次数不同不能改变每笔平均值。

        同时核对每事件暴露确实会变化，确保测试能抓住旧版除以事件数的错误。
        """
        frame = quotes(np.ones(230) * 100)
        learner = IRLRewardLearner(definition='paper_price_difference', deduct_cost=True)
        results = [ExecutionEngine('CME_ES', learner, holding_period=period).run_backtest(
            SingleModelSelector('test', MODELS), frame,
            all_model_preds=np.ones((230, 2)) * .001) for period in [10, 1000]]
        self.assertNotEqual(results[0]['matured_order_count'], results[1]['matured_order_count'])
        np.testing.assert_allclose(results[0]['order_feature_expectation'], [-.275] * 3)
        np.testing.assert_allclose(results[0]['order_feature_expectation'], results[1]['order_feature_expectation'])
        self.assertNotEqual(results[0]['event_feature_exposure'], results[1]['event_feature_exposure'])

    def test_reward_cost_setting_does_not_change_fixed_policy_ledger(self):
        """同一固定策略仅切换 PaperOE/NetOE，奖励改变，真实资金成本不变。"""
        frame = quotes(np.ones(230) * 100)
        output = [ExecutionEngine('CME_ES', IRLRewardLearner(deduct_cost=deduct)).run_backtest(
            SingleModelSelector('test', MODELS), frame,
            all_model_preds=np.ones((230, 2)) * .001) for deduct in [False, True]]
        self.assertEqual(output[0]['net_pnl_usd'], output[1]['net_pnl_usd'])
        self.assertNotEqual(output[0]['order_feature_expectation'], output[1]['order_feature_expectation'])

    def test_me_gate_matches_strict_execution_threshold(self):
        """只有严格超过门槛的信号进入 ME 均值；很大的未交易预测后验收益不得混入。"""
        learner = IRLRewardLearner(definition='paper_price_difference')
        mean, count = gated_me_expectation(learner, np.array([100., 100., 100.]),
            np.array([[101., 102., 103.], [999., 999., 999.], [900., 900., 900.]]),
            np.array([.02, .01, 0.]), .01)
        np.testing.assert_array_equal(mean, [1., 2., 3.])
        self.assertEqual(count, 1)
        selector = CausalShadowARSSelector('test', MODELS, 'ME')
        result = ExecutionEngine('CME_ES', learner, threshold=.01).run_backtest(selector,
            quotes(np.linspace(100, 105, 230)), all_model_preds=np.full((230, 2), .01))
        self.assertEqual(result['observed_rewards'], 0)
        self.assertEqual(result['total_fills'], 0)

    def test_ties_are_reported_and_break_rule_is_deterministic(self):
        """全零交易常产生并列，结果必须列出候选及固定顺序，而非声称唯一最佳。"""
        choice = select_with_ties([0., 0., -1.], ['a', 'b', 'c'])
        self.assertEqual(choice['tied_best_candidates'], ['a', 'b'])
        self.assertEqual(choice['selected'], 'a')
        self.assertEqual(choice['tie_break_rule'], 'first_in_declared_candidate_order')

    def test_shadow_window_is_events_not_seconds(self):
        """300 事件窗口不能被当作 30 分钟；参数与产物明确携带单位。"""
        selector = CausalShadowARSSelector('test', MODELS, window_events=3)
        selector.observe(1, 1., 0)
        selector.select_model(3)
        self.assertFalse(any(selector.queues))
        frame = quotes(np.ones(110) * 100)
        result = ExecutionEngine('CME_ES', IRLRewardLearner()).run_backtest(
            selector, frame, all_model_preds=np.zeros((110, 2)))
        self.assertEqual(result['reward_horizon_unit'], 'events')
        self.assertEqual(result['selector_window_events'], 3)

    def test_signed_weights_still_sum_to_one(self):
        """Eq.(3) 保留和为 1；放宽非负约束后能表示单纯形不能表示的专家。"""
        # 三维才能在每项不超过 1 的同时保留负权重且总和为 1。
        signed = IRLRewardLearner(weight_constraint='signed_box')
        signed.fit_reward_weights([[0, 0, 0], [2, 0, 0], [0, -1, -1]], 0)
        self.assertAlmostEqual(signed.weights.sum(), 1.)
        self.assertTrue((signed.weights < 0).any())

    def test_metrics_distinguish_raw_sign_and_trading_signal(self):
        """小正预测算原始上涨，但未达门槛时交易信号是零；覆盖率必须同时报告。"""
        model = SimpleNamespace(name='m', latency_us=0., latency={})
        metrics, _ = evaluate_models([model], np.array([[.001], [.1]]),
                                    np.array([0., .2]), np.array([0., .1]), .01)
        self.assertEqual(metrics[0]['raw_sign_accuracy'], 50.)
        self.assertEqual(metrics[0]['thresholded_signal_accuracy'], 100.)
        self.assertEqual(metrics[0]['signal_coverage'], .5)
        self.assertEqual(metrics[0]['active_signal_accuracy'], 100.)

    def test_logistic_expected_return_can_differ_from_argmax_class(self):
        """平盘概率最大仍可能期望上涨，两个接口要保留该区别。"""
        model = LogisticDirectionModel()
        model.constant = None
        model.class_returns = {-1: -.01, 0: 0., 1: .1}
        model.model = SimpleNamespace(classes_=np.array([-1, 0, 1]),
            predict_proba=lambda X: np.tile([.1, .6, .3], (len(X), 1)),
            predict=lambda X: np.zeros(len(X), dtype=int))
        X = np.ones((2, 1))
        self.assertTrue((model.predict(X) > 0).all())
        np.testing.assert_array_equal(model.predict_class(X), [0, 0])
