"""完整流程的可重复性与展示一致性测试。

用一段固定种子的合成行情替换文件读取，再真实执行模型训练、奖励学习、
验证调参、各策略回放和报告生成。与单元测试不同，这里检查模块连接是否正确，
不依赖大体积真实行情文件，便于组员和 GitHub CI 快速复查。"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
from test_correctness import quotes
from run_experiments import run_window, aggregate
from generate_dashboard import render_dashboard
from generate_report import render_report
from src.artifacts import pretty_json


def without_timings(result):
    """复制结果并只剔除耗时字段，再进行确定性比较。

    时间会受系统调度和硬件影响，不能要求两次逐位一致；收益、权重、调参结果
    和曲线则必须一致。使用深拷贝，避免清理比较字段时修改原始实验结果。"""
    result = deepcopy(result)
    result.pop('elapsed_seconds')
    for model in result['model_eval']:
        model.pop('latency')
        model.pop('latency_us')
    return result


class PipelineTests(unittest.TestCase):
    """在同一组合成行情上运行两遍流水线，复用结果检查算法和输出层的一致性。"""
    @classmethod
    def setUpClass(cls):
        """准备全类共用的两份实验结果，避免每个断言都重复训练。

        mock 仅替换行情入口；模型、奖励求解、验证和执行仍调用真实实现。
        每次提供 raw.copy()，防止特征工程原地添加列影响下一次运行。"""
        rng = np.random.default_rng(17)
        raw = quotes(100 + np.cumsum(rng.choice([-.125, 0., .125], size=3000)))
        with threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
            with patch('run_experiments.load_and_preprocess_ticks', side_effect=lambda *args: raw.copy()):
                cls.first = run_window('CME_ES', 'synthetic', 0, 3000, 42)
                cls.second = run_window('CME_ES', 'synthetic', 0, 3000, 42)
        cls.summary = dict(schema_version=2, metadata=dict(seed=42, windows=1, rows=3000, python='3.12', threads=1),
                           experiments={'CME_ES': dict(source_file='synthetic', source_rows=3000,
                                                      windows=[cls.first], aggregate=aggregate([cls.first]))})

    def test_two_runs_have_identical_financial_results_and_parameters(self):
        """排除真实耗时差异后，两次完整结果必须相等，而非只比较最终盈亏一个数。"""
        self.assertEqual(without_timings(self.first), without_timings(self.second))

    def test_all_strategies_reconcile_and_all_ablations_present(self):
        """检查基线和八个奖励消融齐全、期末空仓、毛利减成本等于净利，且 JSON 无非法数值。"""
        names = {s['strategy'] for s in self.first['strategies']}
        self.assertIn('Baseline-ValidationBest', names)
        self.assertIn('Baseline-RoundRobin', names)
        self.assertIn('Baseline-Cash', names)
        self.assertEqual(len([s for s in names if s.startswith('Ablation-')]), 8)
        for s in self.first['strategies']:
            self.assertEqual(s['terminal_position'], 0)
            self.assertAlmostEqual(s['gross_pnl_usd'] - s['friction_usd'], s['net_pnl_usd'])
            self.assertAlmostEqual(s['net_return'], s['net_curve'][-1])
        json.dumps(self.summary, allow_nan=False)

    def test_report_and_dashboard_derive_values_from_results(self):
        """检查报告使用输入中的真实收益、保留限制说明，并保证看板无需外部 CDN。"""
        report = render_report(self.summary)
        first = self.first['strategies'][0]
        self.assertIn(f"{first['net_pnl_usd']:.2f}", report)
        self.assertIn('旧版含前视偏差的收益结论已撤回', report)
        page = render_dashboard(self.summary)
        self.assertNotIn('https://', page)  # 离线打开不能依赖远端图表脚本。
        self.assertIn('const DATA = ', page)
        self.assertIn('N/A', page)

    def test_dashboard_escapes_script_termination_in_embedded_data(self):
        """向数据注入结束脚本标签，确认生成 HTML 时被转义，不能成为新的脚本节点。"""
        summary = deepcopy(self.summary)
        summary['untrusted'] = '</script><script>alert(1)</script>'
        page = render_dashboard(summary)
        self.assertNotIn('</script><script>alert', page)
        self.assertIn('\\u003c/script>', page)

    def test_result_serialization_preserves_numbers_and_rejects_nan(self):
        """紧凑排版后的 JSON 应可无损读回；无穷/非数异常不能悄悄进入实验文件。"""
        text = pretty_json(self.summary)
        self.assertEqual(json.loads(text), self.summary)
        with self.assertRaises(ValueError):
            pretty_json({'value': float('nan')})


if __name__ == '__main__':
    unittest.main()
