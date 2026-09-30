"""以手工可核对的接收顺序证明定时快照不偷看未来或使用未完成盘口。"""
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow.parquet as pq

from src.timed_snapshots import iter_timed_snapshots, write_timed_snapshots


BASE = pd.Timestamp('2025-09-22T00:00:00Z')


def quote(recv_ms, bid=100., flags=128, event_ms=None, symbol='ESZ5'):
    """用两个时钟构造五档盘口；event_ms 可模拟较晚才收到的旧交易所事件。"""
    record = dict(ts_recv=BASE + pd.Timedelta(milliseconds=recv_ms),
                  ts_event=BASE + pd.Timedelta(milliseconds=recv_ms if event_ms is None else event_ms),
                  sequence=int(recv_ms * 1000), instrument_id=294973, symbol=symbol, flags=flags)
    for level in range(5):
        record[f'bid_px_{level:02d}'] = bid - level * .25
        record[f'ask_px_{level:02d}'] = bid + (level + 1) * .25
        record[f'bid_sz_{level:02d}'] = 10
        record[f'ask_sz_{level:02d}'] = 12
    return record


class TimedSnapshotTests(unittest.TestCase):
    """每项都只写临时小 Parquet，不能把几百万行真实消息作为 CI 依赖。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'raw.parquet'

    def write_source(self, records):
        pd.DataFrame(records).to_parquet(self.source, index=False)

    def sample(self, **kwargs):
        return list(iter_timed_snapshots(self.source, **kwargs))

    def test_incomplete_event_and_late_message_cannot_change_prior_snapshot(self):
        """0.5 秒边界遇到未完成事件时跳过；迟到消息即使 ts_event 很早也不回写历史。"""
        prefix = [quote(100, bid=100, flags=160, event_ms=90),
                  quote(490, bid=105, flags=0, event_ms=480),
                  quote(800, bid=101, flags=128, event_ms=450),
                  quote(1490, bid=102)]
        self.write_source(prefix)
        first = self.sample()
        self.assertEqual([x['ts_event'] for x in first],
                         [BASE + pd.Timedelta(seconds=1)])
        self.assertEqual(first[0]['bid_px_00'], 101)
        self.assertEqual(first[0]['source_ts_event'], BASE + pd.Timedelta(milliseconds=450))
        self.assertEqual(first[0]['source_ts_recv'], BASE + pd.Timedelta(milliseconds=800))
        self.assertEqual(first[0]['source_flags'], 128)
        # 只有收到下一条消息之后，才能输出之前已跨过的 1.5 秒网格。
        self.write_source(prefix + [quote(3100, bid=200, event_ms=200)])
        second = self.sample()
        self.assertEqual(second[:len(first)], first)
        self.assertEqual([x['ts_event'] for x in second],
                         [BASE + pd.Timedelta(seconds=1), BASE + pd.Timedelta(milliseconds=1500)])
        self.assertEqual([x['bid_px_00'] for x in second], [101, 102])
        self.assertLess(second[1]['source_ts_recv'], second[1]['ts_event'])

    def test_boundary_message_is_only_available_at_next_grid(self):
        """恰在 B 到达的报价不能用于 B；年龄等于上限时仍可用于下一网格。"""
        self.write_source([quote(100, bid=100), quote(500, bid=101),
                           quote(1000, bid=102), quote(1501, bid=103)])
        result = self.sample()
        self.assertEqual([x['bid_px_00'] for x in result], [100, 101, 102])
        self.assertEqual([x['age_ms'] for x in result], [400., 500., 500.])

    def test_stale_gap_skips_grid_and_is_batch_size_invariant(self):
        """长空档不能前填成几千个假 tick；跨 Arrow 批次结果也应相同。"""
        self.write_source([quote(100, bid=100), quote(60000, bid=101),
                           quote(60400, bid=102), quote(61000, bid=103)])
        stats = {}
        snapshots = self.sample(batch_size=1, statistics=stats)
        self.assertEqual(snapshots, self.sample(batch_size=3))
        self.assertEqual([x['ts_event'] for x in snapshots],
                         [BASE + pd.Timedelta(milliseconds=500),
                          BASE + pd.Timedelta(milliseconds=60500)])
        # 1.0～60.0 秒的 119 个网格过期；61.0 秒又因 60.4 秒报价
        # 已超过默认 0.5 秒年龄上限而跳过，合计 120 个空缺。
        self.assertEqual(stats['skipped_intervals'], 120)

    def test_bad_clock_bad_book_and_wrong_contract_fail_closed(self):
        """异常完成事件使旧报价失效；错合约不能套用 ES 的成本和 tick 设置。"""
        self.write_source([quote(100, bid=100), quote(600, bid=101, flags=136),
                           quote(1100, bid=102, flags=132), quote(1400, bid=103),
                           quote(1600, bid=104)])
        snapshots = self.sample()
        self.assertEqual([x['ts_event'] for x in snapshots],
                         [BASE + pd.Timedelta(milliseconds=500),
                          BASE + pd.Timedelta(milliseconds=1500)])
        self.assertEqual([x['bid_px_00'] for x in snapshots], [100, 103])
        self.write_source([quote(100, symbol='NQZ5'), quote(1000)])
        with self.assertRaisesRegex(ValueError, 'ESZ5'):
            self.sample()

    def test_reverse_receive_order_is_rejected(self):
        """不能为了排序而看完未来才决定当下快照；接收乱序直接报错。"""
        self.write_source([quote(600), quote(100)])
        with self.assertRaisesRegex(ValueError, 'backwards'):
            self.sample(batch_size=1)

    def test_writer_preserves_provenance_and_marks_partial_prefix(self):
        """输出保留原消息时间、网格设置、源哈希和部分数据标志，可供下游审计。"""
        self.write_source([quote(100), quote(900), quote(1100)])
        output = self.root / 'snapshots.parquet'
        stats = write_timed_snapshots(self.source, output, max_records=2,
                                      source_date='2025-09-22', condition='available', batch_size=1)
        table = pq.read_table(output)
        metadata = table.schema.metadata
        self.assertEqual(stats['source_records'], 2)
        self.assertEqual(stats['snapshots'], 1)
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(metadata[b'partial_prefix'], b'true')
        self.assertEqual(metadata[b'sampling_clock'], b'ts_recv')
        self.assertEqual(metadata[b'interval_ms'], b'500')
        self.assertEqual(metadata[b'source_date_utc'], b'2025-09-22')
        self.assertEqual(len(metadata[b'source_sha256']), 64)
        with self.assertRaises(FileExistsError):
            write_timed_snapshots(self.source, output)


if __name__ == '__main__':
    unittest.main()
