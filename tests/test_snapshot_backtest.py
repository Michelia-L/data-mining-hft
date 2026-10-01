"""合成快照到训练、时间回放、JSON 与中文报告的联动验证。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from run_snapshot_backtest import main
from src.snapshot_backtest import read_prepared_window, run_snapshot_experiment, training_samples
from src.snapshot_dataset import prepare_snapshot_dataset
from src.timed_snapshots import SNAPSHOT_SCHEMA
from test_time_execution import BASE, quote


class SnapshotBacktestTests(unittest.TestCase):
    """从数据准备的真实格式开始，不靠手写质量报告绕过来源校验。"""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rows = [quote(BASE + pd.Timedelta(milliseconds=500 * i), 100. + .25 * i)
                for i in range(600) if i != 330]
        metadata = dict(sampling_clock='ts_recv', interval_ms='500', max_age_ms='500',
            source_file='synthetic.raw', source_sha256='a' * 64, source_date_utc='2025-09-22',
            partial_prefix='false', source_condition='available')
        table = pa.Table.from_pylist(rows, schema=SNAPSHOT_SCHEMA.with_metadata(
            {key.encode(): value.encode() for key, value in metadata.items()}))
        raw = self.root / 'raw.parquet'
        pq.write_table(table, raw)
        self.dataset = self.root / 'dataset'
        prepare_snapshot_dataset([raw], self.dataset, horizons_ms=(1000, 3000))
        self.bounds = dict(train_start=BASE.isoformat(),
            train_end=(BASE + pd.Timedelta(seconds=120)).isoformat(),
            test_start=(BASE + pd.Timedelta(seconds=120)).isoformat(),
            test_end=(BASE + pd.Timedelta(seconds=240)).isoformat(), prediction_horizon_ms=1000)

    def test_train_only_scaling_boundary_purge_and_replay_rows(self):
        """训练末端按最长尺度隔离，标准化只拟合训练；缺口前后所有行情都进入回放。"""
        result = run_snapshot_experiment(self.dataset, **self.bounds, detail=True)
        frame, _, _ = read_prepared_window(self.dataset, self.bounds['train_start'], self.bounds['test_end'])
        X, _, keep = training_samples(frame, self.bounds['train_start'], self.bounds['train_end'], 1000, 3000)
        self.assertLess(frame.loc[keep, 'ts_event'].max() + pd.Timedelta(seconds=3),
                        pd.Timestamp(self.bounds['train_end']))
        np.testing.assert_allclose(result['model']['scaler_mean'], X.mean(axis=0))
        self.assertEqual(result['test_rows'], 239)
        self.assertEqual(result['results'][0]['replay_rows'], 239)
        self.assertEqual(result['results'][1]['net_pnl_usd'], 0.)
        self.assertLess(result['test_feature_valid_rows'], result['test_rows'])
        self.assertEqual(len(result['dataset']['parquet_sha256']), 64)
        self.assertIn('src/time_execution.py', result['code_sha256'])
        json.dumps(result, allow_nan=False)

    def test_poisoning_test_labels_does_not_change_model_or_replay(self):
        """修改测试区未来标签与有效标记，模型参数、决策、成交和反馈仍相同。"""
        first = run_snapshot_experiment(self.dataset, **self.bounds, detail=True)
        path = self.dataset / 'snapshots.parquet'
        table = pq.read_table(path)
        frame = table.to_pandas()
        mask = frame.ts_event >= pd.Timestamp(self.bounds['test_start'])
        for h in (1000, 3000):
            frame.loc[mask, f'label_valid_{h}ms'] = False
            frame.loc[mask, f'future_return_{h}ms'] = np.nan
        frame.loc[mask, 'learning_ready'] = False
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
            table.schema.metadata), path)
        second = run_snapshot_experiment(self.dataset, **self.bounds, detail=True)
        self.assertEqual(first['model'], second['model'])
        self.assertEqual(first['results'], second['results'])

    def test_cli_json_report_no_overwrite_and_reproducibility(self):
        """同样参数得到相同冻结参数和结果；新格式生成自己的报告并拒绝覆盖。"""
        args = ['--dataset-dir', str(self.dataset), '--output-dir', str(self.root / 'output'),
                '--prediction-horizon-ms', '1000', '--detail']
        for name in ('train_start', 'train_end', 'test_start', 'test_end'):
            args += ['--' + name.replace('_', '-'), self.bounds[name]]
        first = main(args)
        self.assertEqual(first, run_snapshot_experiment(self.dataset, **self.bounds, detail=True))
        self.assertEqual(first, json.loads((self.root / 'output/result.json').read_text()))
        self.assertIn('净盈亏 USD', (self.root / 'output/report.md').read_text())
        with self.assertRaises(FileExistsError):
            main(args)

    def test_source_quality_calendar_and_time_split_guards(self):
        """拒绝无时区/交叉切分/来源矛盾；前缀与降级标记须明确允许并保留。"""
        for overrides in [{'train_end': self.bounds['test_end']}, {'train_start': '2025-09-22'},
                          {'prediction_horizon_ms': 1500}]:
            with self.assertRaises(ValueError):
                run_snapshot_experiment(self.dataset, **(self.bounds | overrides))
        quality = self.dataset / 'quality.json'
        original = json.loads(quality.read_text())
        changed = deepcopy(original)
        changed['degraded_input'] = True
        changed['inputs'][0]['metadata']['source_condition'] = 'degraded'
        quality.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'include-degraded'):
            run_snapshot_experiment(self.dataset, **self.bounds)
        result = run_snapshot_experiment(self.dataset, **self.bounds, include_degraded=True)
        self.assertTrue(result['dataset']['degraded_input'])
        changed = deepcopy(original)
        changed['partial_input'] = True
        changed['inputs'][0]['metadata']['partial_prefix'] = 'true'
        quality.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'allow-partial'):
            run_snapshot_experiment(self.dataset, **self.bounds)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            run_snapshot_experiment(self.dataset, **self.bounds, allow_partial=True)


if __name__ == '__main__':
    unittest.main()
