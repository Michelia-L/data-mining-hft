"""新数据入口的回归测试：临时 Parquet 即可在 GitHub CI 验证，不依赖私有行情。

除了选择日期，还要验证质量状态、索引与文件不一致时会失败，以及最终保存的
结果能追溯到实际使用的文件。端到端用真实训练与回放，避免只测试参数转发。
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from generate_dashboard import render_dashboard
from generate_report import render_report
from run_experiments import file_hash, main
from src.data_catalog import select_daily_source
from test_correctness import quotes


class DailySourceTests(unittest.TestCase):
    """每个用例创建隔离的数据目录，故意破坏索引不会影响仓库的真实行情。"""

    def setUp(self):
        """生成单合约、3000 条有效报价，索引格式与本地 Databento 下载索引一致。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'daily.parquet'
        rng = np.random.default_rng(42)
        frame = quotes(6700 + np.cumsum(rng.choice([-.125, 0., .125], size=3000)))
        frame['symbol'] = 'ESZ5'
        frame['ts_event'] = pd.date_range('2025-09-22', periods=len(frame), freq='s', tz='UTC')
        frame.to_parquet(self.path, index=False)
        self.index_path = self.root / 'index.json'
        self.catalog = dict(dataset='GLBX.MDP3', schema='mbp-10', symbol='ESZ5', stype_in='raw_symbol',
                            files=[dict(date='2025-09-22', parquet='daily.parquet', rows=3000,
                                        condition='available', parquet_bytes=self.path.stat().st_size)])
        self.save_index()

    def save_index(self):
        """将当前用例修改后的目录信息写盘，模拟真实调用者提供的索引。"""
        self.index_path.write_text(json.dumps(self.catalog), encoding='utf-8')

    def select(self, **kwargs):
        """指定测试数据根目录，验证相对路径解析不依赖进程当前工作目录。"""
        return select_daily_source(self.index_path, '2025-09-22', data_root=self.root, **kwargs)

    def test_exact_date_selection_does_not_open_other_files(self):
        """只读取请求日期；其他日期即使未下载，也不能阻止当前文件的检查。"""
        self.catalog['files'].insert(0, dict(date='2025-09-21', parquet='absent.parquet'))
        self.save_index()
        source = self.select()
        self.assertEqual(source['source_file'], str(self.path))
        self.assertEqual(source['file_date_utc'], '2025-09-22')
        self.assertEqual(source['source_rows'], 3000)

    def test_degraded_requires_opt_in_and_keeps_quality_label(self):
        """显式允许降级数据后仍保留 degraded，不将授权误解释为数据恢复正常。"""
        self.catalog['files'][0]['condition'] = 'degraded'
        self.save_index()
        with self.assertRaisesRegex(ValueError, 'degraded'):
            self.select()
        self.assertEqual(self.select(include_degraded=True)['condition'], 'degraded')

    def test_missing_duplicate_or_invalid_date_fails(self):
        """无文件、重复日期与非标准日期都应失败，不能退回旧数据或默认选择第一条。"""
        for value in ('2025-09-23', '2025-02-30', '20250922'):
            with self.subTest(date=value), self.assertRaises(ValueError):
                select_daily_source(self.index_path, value, data_root=self.root)
        self.catalog['files'] *= 2
        self.save_index()
        with self.assertRaisesRegex(ValueError, 'found 2'):
            self.select()

    def test_wrong_instrument_or_inconsistent_metadata_fails(self):
        """覆盖最危险的静默错误：错品种套用 ES 费用，或读取与清单不一致的文件。"""
        for field, value in [('rows', 3001), ('parquet_bytes', 1), ('condition', 'unknown'),
                             ('parquet', 'missing.parquet')]:
            original = self.catalog['files'][0][field]
            self.catalog['files'][0][field] = value
            self.save_index()
            with self.subTest(field=field), self.assertRaises((ValueError, OSError)):
                self.select(include_degraded=True)
            self.catalog['files'][0][field] = original
        self.catalog['symbol'] = 'NQZ5'
        self.save_index()
        with self.assertRaisesRegex(ValueError, 'ESZ5'):
            self.select()

    def test_cli_requires_complete_daily_arguments_and_output(self):
        """命令不完整时在训练前终止，尤其不能自动覆盖原有正式实验结果。"""
        for args in (['--data-date', '2025-09-22'], ['--include-degraded'],
                     ['--data-index', str(self.index_path)],
                     ['--data-index', str(self.index_path), '--data-date', '2025-09-22']):
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)

    def test_daily_cli_runs_only_selected_es_and_preserves_provenance(self):
        """贯通索引、真实 Parquet、训练回放、JSON 和展示，确认实际没有混入旧品种。"""
        # 使用绝对路径也应兼容；本测试用 degraded 验证显式允许和展示贯通。
        self.catalog['files'][0].update(parquet=str(self.path), condition='degraded')
        self.save_index()
        output = self.root / 'result.json'
        with redirect_stdout(io.StringIO()):
            main(['--data-index', str(self.index_path), '--data-date', '2025-09-22',
                  '--include-degraded', '--rows', '3000', '--windows', '1', '--output', str(output)])
        summary = json.loads(output.read_text())
        self.assertEqual(set(summary['experiments']), {'CME_ES'})
        experiment = summary['experiments']['CME_ES']
        self.assertEqual(experiment['source_sha256'], file_hash(self.path))
        self.assertEqual(experiment['windows'][0]['symbol'], 'ESZ5')
        self.assertEqual(len(experiment['windows']), 1)
        source = summary['metadata']['daily_source']
        self.assertEqual(source['index_sha256'], file_hash(self.index_path))
        self.assertEqual(source['condition'], 'degraded')
        self.assertEqual(source['file_date_utc'], '2025-09-22')
        self.assertIn('数据质量 `degraded`', render_report(summary))
        self.assertIn('尚未跨日训练', render_dashboard(summary))


if __name__ == '__main__':
    unittest.main()
