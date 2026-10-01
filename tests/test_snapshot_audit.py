"""核对缺格诊断的守恒、因果边界和预声明对照；所有消息都是合成数据。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from run_snapshot_audit import main
from src.snapshot_audit import (GridAudit, add_age_bins, compare_audits, freeze_audit,
                               load_plan, run_age_audit, validate_config)
from src.snapshot_dataset import SessionCalendar
from src.timed_snapshots import iter_timed_snapshots
from test_timed_snapshots import BASE, quote


class GridDiagnosticTests(unittest.TestCase):
    """小消息序列可手算每个跳过边界；诊断不能改变原来的快照。"""
    def test_states_are_exclusive_and_observer_preserves_sampling(self):
        """过龄、事件未完成、坏盘口标记、盘口非法和坏接收时间须分开记录。"""
        invalid = quote(490)
        invalid['bid_px_00'] = -1.
        cases = [('stale_completed_book', [quote(100), quote(1100)]),
                 ('incomplete_event', [quote(100), quote(490, flags=0), quote(900), quote(1100)]),
                 ('bad_book_flag', [quote(100), quote(490, flags=132), quote(900), quote(1100)]),
                 ('invalid_completed_book', [quote(100), invalid, quote(900), quote(1100)]),
                 ('unreliable_receive_state', [quote(100), quote(490, flags=136), quote(900), quote(1100)])]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.parquet'
            for expected, messages in cases:
                with self.subTest(expected=expected):
                    pd.DataFrame(messages).to_parquet(path, index=False)
                    original = list(iter_timed_snapshots(path))
                    observed, stats = [], {}
                    measured = list(iter_timed_snapshots(path, batch_size=1, statistics=stats,
                        grid_observer=lambda *args: observed.append(args)))
                    self.assertEqual(measured, original)
                    self.assertEqual(stats['skipped_by_reason'], {expected: 1})
                    self.assertEqual(sum(stats['skipped_by_reason'].values()), stats['skipped_intervals'])
                    self.assertEqual(sum((end - start) // 500000000 for start, end, *_ in observed),
                                     stats['snapshots'] + stats['skipped_intervals'])
                    self.assertEqual(next(row[2] for row in observed if row[2] != 'emitted'), expected)

    def test_long_gap_is_one_range_and_age_bins_match_hand_count(self):
        """长空档诊断不逐格填充，不把静默直接叫作供应商丢包；分箱含上界。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.parquet'
            pd.DataFrame([quote(100), quote(60000)]).to_parquet(path, index=False)
            ranges, stats = [], {}
            result = list(iter_timed_snapshots(path, max_age_ms=2000, statistics=stats,
                          grid_observer=lambda *args: ranges.append(args)))
            self.assertEqual(len(result), 4)
            self.assertEqual(len(ranges), 5)
            self.assertEqual(stats['skipped_intervals'], 116)
            from collections import Counter
            counter = Counter()
            for start, end, _, _, reference in ranges:
                add_age_bins(counter, start // 500000000, end // 500000000, 500000000, reference)
            self.assertEqual(dict(counter), {'le_500ms': 1, '500_to_1000ms': 1,
                '1000_to_2000ms': 2, 'gt_2000ms': 116})

    def test_calendar_excludes_break_and_preserves_unknown_tail(self):
        """诊断只数计划交易网格；暂停前旧报价不能算暂停后有效快照。"""
        calendar = SessionCalendar()
        audit = GridAudit(calendar, ['2025-09-22'], 500)
        start = pd.Timestamp('2025-09-22T20:14:59.500Z').value
        end = pd.Timestamp('2025-09-22T20:30:00.500Z').value
        audit.observe(start, end, 'stale_completed_book', start-1000000000, start-1000000000)
        audit.observe(end, end+500000000, 'emitted', end-1000000000, end-1000000000)
        row = audit.result()[0]
        self.assertEqual(row['resolved_grid_points'], 3)  # 暂停前两格，恢复后一格。
        self.assertEqual(row['grid_reasons']['stale_completed_book'], 2)
        self.assertEqual(row['grid_reasons']['emitted_preopen_source'], 1)
        self.assertEqual(row['grid_reasons']['emitted'], 0)
        self.assertEqual(sum(row['grid_reasons'].values()) + row['not_resolved_by_stream'],
                         row['expected_grid_points'])
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            audit.observe(start, end, 'emitted', start-1, start-1)

    def test_future_prices_and_bad_clock_do_not_rewrite_prior_diagnostic(self):
        """未来价格不能改变过去分类；不可信接收时钟不能推进诊断边界。"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.parquet'
            def collect(tail):
                messages = [quote(100), quote(900), quote(1100)] + tail
                pd.DataFrame(messages).to_parquet(path, index=False)
                ranges = []
                list(iter_timed_snapshots(path, grid_observer=lambda *args: ranges.append(args)))
                return ranges
            first = collect([quote(1400), quote(1600)])
            second = collect([quote(1400, bid=999), quote(1600, bid=999)])
            self.assertEqual([r for r in first if r[1] <= BASE.value+1000000000],
                             [r for r in second if r[1] <= BASE.value+1000000000])
            bad = collect([quote(30000, flags=136), quote(1400), quote(1600)])
            # 范围右端排他；最后覆盖 1.5 秒网格，因此终点可恰为 2 秒。
            self.assertLessEqual(max(r[1] for r in bad), BASE.value+2000000000)


class SnapshotAgeAuditTests(unittest.TestCase):
    """66 分钟合成接收流既有严格基线失效，也有更长年龄的完整长尺度标签。"""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calendar = SessionCalendar()
        self.config = dict(schema_version=1, purpose='data_quality_diagnostic',
            session_dates=['2025-09-22'], source_dates_utc=['2025-09-21', '2025-09-22'],
            interval_ms=500, max_age_candidates_ms=[500, 1000],
            horizons_ms=[1000, 3000, 3660000], include_degraded=False)
        self.config_path = self.root / 'config.json'
        self.config_path.write_text(json.dumps(self.config))
        files = []
        for day, offset, times in [('2025-09-21', pd.Timedelta(days=-1, hours=22), [100, 900, 1100]),
                                    ('2025-09-22', pd.Timedelta(hours=14), range(100, 3960101, 750))]:
            records = []
            for i, t in enumerate(times):
                record = quote(t)
                record['sequence'] = i
                for clock in ('ts_event', 'ts_recv'):
                    record[clock] += offset
                records.append(record)
            path = self.root / (day + '.parquet')
            pd.DataFrame(records).to_parquet(path, index=False)
            files.append(dict(date=day, rows=len(records), condition='available',
                              parquet=str(path), parquet_bytes=path.stat().st_size))
        self.index = self.root / 'index.json'
        self.index.write_text(json.dumps(dict(dataset='GLBX.MDP3', schema='mbp-10',
            symbol='ESZ5', stype_in='raw_symbol', files=files)))

    def freeze(self):
        main(['freeze', '--config', str(self.config_path), '--data-index', str(self.index),
              '--output-dir', str(self.root / 'frozen')])
        return self.root / 'frozen/plan.json'

    def test_end_to_end_frozen_comparison_and_default_baseline_retained(self):
        """全候选一致来源并原子保存；年龄扩大使标签可用，但不自动选默认参数。"""
        plan = self.freeze()
        paths, results = [], []
        for age in (500, 1000):
            output = self.root / str(age)
            results.append(main(['run', '--plan', str(plan), '--max-age-ms', str(age),
                                 '--output-dir', str(output)]))
            paths.append(output / 'result.json')
            self.assertEqual(results[-1], json.loads(paths[-1].read_text()))
            quality = json.loads((output / 'prepared/quality.json').read_text())
            self.assertTrue(all(Path(r['path']).exists() for r in quality['inputs']))
            row = results[-1]['session_grids'][0]
            self.assertEqual(row['grid_reasons']['emitted'], results[-1]['horizon_audits'][0]['observed_rows'])
            self.assertEqual(row['resolved_grid_points']+row['not_resolved_by_stream'], row['expected_grid_points'])
            self.assertFalse(results[-1]['default_age_changed'])
        self.assertEqual(results[0]['horizon_audits'][0]['all_horizons_feature_valid_rows'], 0)
        self.assertGreater(results[1]['horizon_audits'][0]['all_horizons_feature_valid_rows'], 0)
        combined = main(['compare', '--plan', str(plan), '--results', *map(str, paths),
                         '--output-dir', str(self.root / 'comparison')])
        self.assertIsNone(combined['selected_age_ms'])
        with self.assertRaisesRegex(ValueError, 'every frozen candidate'):
            compare_audits(plan, paths[:1])
        with self.assertRaises(FileExistsError):
            run_age_audit(plan, 500, self.root / '500')
        quality_path = self.root / '500/prepared/quality.json'
        original_quality = quality_path.read_text()
        quality_path.write_text(original_quality + ' ')
        with self.assertRaisesRegex(ValueError, 'prepared files changed'):
            compare_audits(plan, paths)
        quality_path.write_text(original_quality)
        changed = deepcopy(results[0])
        changed['horizon_audits'][0]['observed_rows'] += 1
        paths[0].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'different plans'):
            compare_audits(plan, paths)

    def test_configuration_and_frozen_input_guards(self):
        """拒绝丢掉严格基线、缺前夜分区或临时增加候选；变更源码需新冻结。"""
        for change in [dict(max_age_candidates_ms=[1000]), dict(source_dates_utc=['2025-09-22']),
                       dict(include_degraded=1), dict(horizons_ms=[1001])]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_config(self.config | change, self.calendar)
        plan = self.freeze()
        with self.assertRaisesRegex(ValueError, 'not declared'):
            run_age_audit(plan, 2000, self.root / 'undeclared')
        with patch('src.snapshot_audit.audit_code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                load_plan(plan)
        raw = self.root / '2025-09-22.parquet'
        frame = pd.read_parquet(raw)
        frame.loc[0, 'bid_px_00'] = 99.
        frame.to_parquet(raw, index=False)
        with self.assertRaisesRegex(ValueError, 'Raw source changed'):
            run_age_audit(plan, 500, self.root / 'changed')
        self.assertFalse((self.root / 'changed').exists())

    def test_degraded_source_requires_explicit_permission(self):
        """诊断也不能把供应商降级日期静默当正常数据；许可冻结并保留在来源中。"""
        index = json.loads(self.index.read_text())
        index['files'][0]['condition'] = 'degraded'
        self.index.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'degraded'):
            freeze_audit(self.config_path, self.index)
        self.config['include_degraded'] = True
        self.config_path.write_text(json.dumps(self.config))
        plan = freeze_audit(self.config_path, self.index)
        self.assertEqual(plan['sources'][0]['condition'], 'degraded')


if __name__ == '__main__':
    unittest.main()
