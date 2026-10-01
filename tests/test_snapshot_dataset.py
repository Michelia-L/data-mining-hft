"""用合成定时快照验证真实时间标签、session 隔离、缺口与文件流式拼接。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_snapshot_dataset import main
from src.snapshot_dataset import (SessionCalendar, build_session_dataset,
                                  inspect_snapshot, prepare_snapshot_dataset)
from src.timed_snapshots import SNAPSHOT_SCHEMA


BASE = pd.Timestamp('2025-09-22T00:00:00Z')


def snapshot(stamp, bid=100., source=None):
    """网格边界前 100 ms 收到一个完整盘口；不用原始事件数定义时间。"""
    stamp = pd.Timestamp(stamp)
    received = stamp - pd.Timedelta(milliseconds=100) if source is None else pd.Timestamp(source)
    row = dict(ts_event=stamp, source_ts_recv=received, source_ts_event=received,
               source_flags=128, age_ms=(stamp - received).total_seconds() * 1000,
               sequence=1, instrument_id=294973, symbol='ESZ5')
    for level in range(5):
        row[f'bid_px_{level:02d}'] = bid - level * .25
        row[f'ask_px_{level:02d}'] = bid + (level + 1) * .25
        row[f'bid_sz_{level:02d}'] = 10
        row[f'ask_sz_{level:02d}'] = 12
    return row


def regular_rows(count=100):
    """价格按每秒 0.5 点缓慢上升，价差标签可以直接手算。"""
    return [snapshot(BASE + pd.Timedelta(milliseconds=500 * i), 100. + .25 * i)
            for i in range(count)]


class SnapshotDatasetTests(unittest.TestCase):
    """临时文件隔离所有产物；不依赖大型真实行情即可验证关键因果性质。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calendar = SessionCalendar()

    def write_snapshot(self, name, rows, **overrides):
        metadata = dict(sampling_clock='ts_recv', interval_ms='500', max_age_ms='500',
                        source_file=name + '.raw', source_sha256='a' * 64,
                        source_date_utc='2025-09-22', source_condition='available', partial_prefix='false')
        metadata.update(overrides)
        schema = SNAPSHOT_SCHEMA.with_metadata({k.encode(): v.encode() for k, v in metadata.items()})
        path = self.root / name
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        return path

    def build(self, rows, horizons=(1000, 5000), boundaries=()):
        frame, _, _ = self.calendar.assign(pd.DataFrame(rows))
        frame = frame.loc[frame.session_id.ne('')]
        return build_session_dataset(frame, self.calendar, 500, horizons, boundaries)

    def test_calendar_handles_night_session_dst_and_thanksgiving_trade_date(self):
        """11/3 夏令时结束后开盘 UTC 推后一小时；感恩节早盘与次日归同一 trade date。"""
        self.assertEqual(str(self.calendar.sessions['2025-10-31']['open']), '2025-10-30 22:00:00+00:00')
        self.assertEqual(str(self.calendar.sessions['2025-11-03']['open']), '2025-11-02 23:00:00+00:00')
        self.assertNotIn('2025-11-27', self.calendar.sessions)
        stamps = ['2025-09-21T22:00:00.500Z', '2025-09-22T00:00:00Z',
                  '2025-11-27T17:59:59.500Z', '2025-11-27T18:00:00.500Z',
                  '2025-11-27T23:00:00.500Z', '2025-11-28T18:15:00Z',
                  '2025-11-28T18:15:00.500Z']
        frame, outside, old_source = self.calendar.assign(pd.DataFrame([snapshot(t) for t in stamps]))
        self.assertEqual(frame.session_id.tolist(),
                         ['2025-09-22', '2025-09-22', '2025-11-28', '', '2025-11-28', '2025-11-28', ''])
        self.assertEqual(outside, 2)
        self.assertEqual(old_source, 0)

    def test_pause_reopen_cannot_use_pre_pause_quote_and_coverage_is_bounded(self):
        """暂停期间剔除；重开后的首格也不能拿暂停前的旧报价计算标签。"""
        rows = [snapshot('2025-09-22T20:15:00Z'), snapshot('2025-09-22T20:20:00Z'),
                snapshot('2025-09-22T20:30:00.500Z', source='2025-09-22T20:14:00Z')]
        assigned, outside, old = self.calendar.assign(pd.DataFrame(rows))
        self.assertEqual(assigned.session_id.tolist(), ['2025-09-22', '', ''])
        self.assertEqual((outside, old), (1, 1))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.calendar.assign(pd.DataFrame([snapshot('2026-01-01T00:00:00Z')]))

    def test_exact_price_difference_and_interior_gap_not_row_shift(self):
        """缺一格后 5 秒端点仍存在，但穿过缺口的标签必须失效；更远行不能补位。"""
        complete, _, _ = self.build(regular_rows())
        self.assertAlmostEqual(complete.future_price_diff_5000ms.iloc[0], 2.5)
        self.assertAlmostEqual(complete.future_return_5000ms.iloc[0], 2.5 / 100.125)
        rows = regular_rows()
        del rows[4]
        frame, _, report = self.build(rows)
        self.assertEqual(frame.label_reason_5000ms.iloc[0], 'gap')
        self.assertTrue(np.isnan(frame.future_price_diff_5000ms.iloc[0]))
        self.assertEqual(frame.label_reason_1000ms.iloc[2], 'missing_target')
        self.assertEqual(frame.label_end_5000ms.iloc[0], BASE + pd.Timedelta(seconds=5))
        self.assertEqual(report['internal_missing_grid_points'], 1)

    def test_feature_warmup_restarts_and_future_changes_do_not_change_features(self):
        """连续块需要 30 格历史；缺口后重新预热，修改未来价格不能改写过去特征。"""
        rows = regular_rows(130)
        del rows[60]
        frame, features, _ = self.build(rows)
        self.assertFalse(frame.feature_valid.iloc[29])
        self.assertTrue(frame.feature_valid.iloc[30])
        self.assertFalse(frame.feature_valid.iloc[60])
        self.assertTrue(frame.feature_valid.iloc[90])
        changed = deepcopy(rows)
        for row in changed[50:]:
            for name in row:
                if '_px_' in name:
                    row[name] += 50
        other, _, _ = self.build(changed)
        pd.testing.assert_frame_equal(frame[features].iloc[:50], other[features].iloc[:50])

    def test_split_boundary_equality_is_purged_and_labels_do_not_cross_close(self):
        """到期恰在下一切分起点也要清除；收盘之后的价格不能作为本 session 标签。"""
        frame, _, _ = self.build(regular_rows(), boundaries=['2025-09-22T00:00:05Z'])
        self.assertEqual(frame.label_reason_5000ms.iloc[0], 'split_boundary')
        self.assertFalse(frame.label_valid_5000ms.iloc[0])
        self.assertTrue(frame.label_valid_5000ms.iloc[10])
        rows = [snapshot('2025-09-22T20:59:59Z'), snapshot('2025-09-22T20:59:59.500Z'),
                snapshot('2025-09-22T21:00:00Z')]
        tail, _, _ = self.build(rows, horizons=(1000,))
        self.assertTrue(tail.label_valid_1000ms.iloc[0])
        self.assertEqual(tail.label_reason_1000ms.iloc[1], 'session_boundary')

    def test_mid_session_break_is_not_a_continuous_tick_or_label(self):
        """同一交易日的暂停两侧端点存在，也不能穿越暂停评价未来收益。"""
        rows = [snapshot('2025-09-22T20:14:59.500Z'), snapshot('2025-09-22T20:15:00Z'),
                snapshot('2025-09-22T20:30:00.500Z')]
        frame, _, _ = self.build(rows, horizons=(1000, 901000))
        self.assertEqual(frame.label_reason_901000ms.iloc[0], 'scheduled_break')
        self.assertEqual(frame.continuous_block.tolist(), [1, 1, 2])

    def test_streaming_batches_and_utc_file_partition_do_not_change_dataset(self):
        """同一 session 在午夜分成两个 UTC 文件，标签与预热必须与一次读入一致。"""
        rows = [snapshot(BASE - pd.Timedelta(seconds=20) + pd.Timedelta(milliseconds=500 * i),
                         100. + .25 * i) for i in range(140)]
        whole = self.write_snapshot('whole.parquet', rows)
        first = self.write_snapshot('first.parquet', rows[:40], source_date_utc='2025-09-21')
        second = self.write_snapshot('second.parquet', rows[40:])
        r1 = prepare_snapshot_dataset([whole], self.root / 'whole', horizons_ms=(1000, 5000), batch_size=19)
        r2 = prepare_snapshot_dataset([first, second], self.root / 'parts', horizons_ms=(1000, 5000), batch_size=7)
        d1, d2 = pq.read_table(self.root / 'whole/snapshots.parquet').to_pandas(), pq.read_table(self.root / 'parts/snapshots.parquet').to_pandas()
        pd.testing.assert_frame_equal(d1.drop(columns='snapshot_input_id'), d2.drop(columns='snapshot_input_id'))
        self.assertEqual(r1['sessions'], r2['sessions'])
        self.assertEqual(len(r2['sessions']), 1)
        self.assertFalse(r2['sessions'][0]['coverage_complete'])
        self.assertEqual(json.loads((self.root / 'parts/quality.json').read_text()), r2)
        self.assertIn('2025-09-22', (self.root / 'parts/quality.md').read_text())
        with self.assertRaises(FileExistsError):
            prepare_snapshot_dataset([whole], self.root / 'whole')

    def test_metadata_partial_degraded_and_bad_clock_are_rejected(self):
        """默认拒绝预览、降级及缺少来源的输入；明确放行后仍保留质量标志。"""
        partial = self.write_snapshot('partial.parquet', regular_rows(), partial_prefix='true')
        with self.assertRaisesRegex(ValueError, 'allow-partial'):
            inspect_snapshot(partial)
        degraded = self.write_snapshot('degraded.parquet', regular_rows(), source_condition='degraded')
        with self.assertRaisesRegex(ValueError, 'include-degraded'):
            inspect_snapshot(degraded)
        report = prepare_snapshot_dataset([partial], self.root / 'preview', allow_partial=True)
        self.assertTrue(report['partial_input'])
        rows = regular_rows()
        rows[0]['source_ts_recv'] = rows[0]['ts_event']
        bad = self.write_snapshot('bad.parquet', rows)
        with self.assertRaisesRegex(ValueError, 'causality'):
            prepare_snapshot_dataset([bad], self.root / 'bad')
        self.assertFalse((self.root / 'bad').exists())
        raw = self.root / 'raw.parquet'
        pd.DataFrame(regular_rows()).to_parquet(raw, index=False)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            inspect_snapshot(raw)

    def test_duplicate_order_mixed_grid_and_invalid_horizons_fail_without_outputs(self):
        """重复输入不能悄悄增加订单数；混合网格、非整数网格前瞻和倒序直接失败。"""
        first = self.write_snapshot('a.parquet', regular_rows())
        second = self.write_snapshot('b.parquet', regular_rows())
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            prepare_snapshot_dataset([first, second], self.root / 'duplicates')
        self.assertFalse((self.root / 'duplicates').exists())
        mixed = self.write_snapshot('mixed.parquet', regular_rows(), interval_ms='250')
        with self.assertRaisesRegex(ValueError, 'settings'):
            prepare_snapshot_dataset([first, mixed], self.root / 'mixed')
        for horizons in [(750,), (5000, 1000), ()]:
            with self.assertRaisesRegex(ValueError, 'Horizons'):
                prepare_snapshot_dataset([first], self.root / 'horizons', horizons_ms=horizons)

    def test_cli_and_missing_whole_session_are_reported(self):
        """两段输入间整天未提供数据，报告仍列出缺失 session；CLI 可直接消费快照。"""
        rows = regular_rows()
        following = [snapshot(r['ts_event'] + pd.Timedelta(days=2)) for r in regular_rows()]
        path = self.write_snapshot('two.parquet', rows + following)
        output = self.root / 'cli'
        main(['--snapshots', str(path), '--output-dir', str(output), '--horizons-ms', '1000'])
        report = json.loads((output / 'quality.json').read_text())
        empty = next(row for row in report['sessions'] if row['session_id'] == '2025-09-23')
        self.assertEqual(empty['observed_snapshots'], 0)
        self.assertEqual(empty['missing_grid_points'], empty['expected_grid_points'])

    def test_session_transition_cannot_reuse_previous_day_feature_or_label(self):
        """旧 session 的末行不能引用次日报价；新 session 特征须独立预热。"""
        before = [snapshot('2025-09-22T20:59:59.500Z'), snapshot('2025-09-22T21:00:00Z')]
        after = [snapshot('2025-09-22T22:00:00.500Z'), snapshot('2025-09-22T22:00:01Z')]
        path = self.write_snapshot('transition.parquet', before + after)
        prepare_snapshot_dataset([path], self.root / 'transition', horizons_ms=(500,))
        frame = pq.read_table(self.root / 'transition/snapshots.parquet').to_pandas()
        self.assertEqual(frame.session_id.tolist(), ['2025-09-22'] * 2 + ['2025-09-23'] * 2)
        self.assertEqual(frame.label_reason_500ms.iloc[1], 'session_boundary')
        self.assertTrue(np.isnan(frame.ret_lag_1.iloc[2]))
        self.assertFalse(frame.feature_valid.iloc[2])

    def test_mixed_arrow_clock_precision_preserves_exact_nanosecond_horizons(self):
        """us/ns 导出的时间列应统一为 ns，不能造成千倍前瞻期或跨 session 写出失败。"""
        first = self.write_snapshot('us.parquet', regular_rows())
        table = pq.read_table(first)
        for name in ('ts_event', 'source_ts_event', 'source_ts_recv'):
            index = table.schema.get_field_index(name)
            field = pa.field(name, pa.timestamp('us', tz='UTC'))
            table = table.set_column(index, field, table.column(index).cast(field.type))
        pq.write_table(table, first)
        second = self.write_snapshot('ns.parquet',
            [snapshot(r['ts_event'] + pd.Timedelta(days=1)) for r in regular_rows()])
        prepare_snapshot_dataset([first, second], self.root / 'precision', horizons_ms=(1000,))
        output = pq.read_table(self.root / 'precision/snapshots.parquet')
        self.assertEqual(output.schema.field('ts_event').type, pa.timestamp('ns', tz='UTC'))
        frame = output.to_pandas()
        self.assertEqual(frame.label_end_1000ms.iloc[0] - frame.ts_event.iloc[0], pd.Timedelta(seconds=1))

    def test_fractional_flags_cannot_be_silently_cast_to_a_reliable_event(self):
        """格式错误的浮点 flags 不能截断成 128，再被当作可靠完成事件接受。"""
        path = self.write_snapshot('fractional.parquet', regular_rows())
        table = pq.read_table(path)
        index = table.schema.get_field_index('source_flags')
        table = table.set_column(index, pa.field('source_flags', pa.float64()),
                                 pa.array([128.5] * table.num_rows))
        pq.write_table(table, path)
        with self.assertRaisesRegex(ValueError, 'column type'):
            inspect_snapshot(path)


if __name__ == '__main__':
    unittest.main()
