"""End-to-end reproducibility and result-presentation checks on synthetic data."""
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
    result = deepcopy(result)
    result.pop('elapsed_seconds')
    for model in result['model_eval']:
        model.pop('latency')
        model.pop('latency_us')
    return result


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        self.assertEqual(without_timings(self.first), without_timings(self.second))

    def test_all_strategies_reconcile_and_all_ablations_present(self):
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
        report = render_report(self.summary)
        first = self.first['strategies'][0]
        self.assertIn(f"{first['net_pnl_usd']:.2f}", report)
        self.assertIn('旧版含前视偏差的收益结论已撤回', report)
        page = render_dashboard(self.summary)
        self.assertNotIn('https://', page)  # No CDN needed to open offline.
        self.assertIn('const DATA = ', page)
        self.assertIn('N/A', page)

    def test_dashboard_escapes_script_termination_in_embedded_data(self):
        summary = deepcopy(self.summary)
        summary['untrusted'] = '</script><script>alert(1)</script>'
        page = render_dashboard(summary)
        self.assertNotIn('</script><script>alert', page)
        self.assertIn('\\u003c/script>', page)

    def test_result_serialization_preserves_numbers_and_rejects_nan(self):
        text = pretty_json(self.summary)
        self.assertEqual(json.loads(text), self.summary)
        with self.assertRaises(ValueError):
            pretty_json({'value': float('nan')})


if __name__ == '__main__':
    unittest.main()
