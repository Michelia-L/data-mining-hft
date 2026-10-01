"""用真实 prepared 格式验证完整日协议、冻结约束与跨日训练的数学性质。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_session_experiment import main
from src.session_experiment import (PHASES, audit_horizons, fit_streaming_ridge,
    freeze_protocol, read_day, run_frozen_protocol, summarize_days, validate_protocol)
from src.snapshot_backtest import PreparedDatasetReader, training_samples
from src.snapshot_dataset import SessionCalendar, prepare_snapshot_dataset
from src.timed_snapshots import SNAPSHOT_SCHEMA
from test_time_execution import quote


class SessionExperimentTests(unittest.TestCase):
    """一分钟合成网格让超过一小时的前瞻测试保持小规模，不冒充真实回测。"""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calendar = SessionCalendar()
        self.days = ['2025-09-29', '2025-09-30', '2025-10-01', '2025-10-02', '2025-10-03']
        self.protocol = dict(schema_version=1, purpose='development_validation',
            sessions=dict(train=self.days[:2], calibration=self.days[2:3],
                          validation=self.days[3:4], test=self.days[4:]),
            prediction_horizon_ms=60000, reward_horizons_ms=[60000, 180000, 3660000],
            latency_ms=60000, holding_review_ms=180000, threshold=.000015,
            allow_partial=False, include_degraded=False)
        self.config = self.root / 'protocol.json'
        self.config.write_text(json.dumps(self.protocol))
        rows = []
        for d, day in enumerate(self.days):
            opening = self.calendar.sessions[day]['open'] + pd.Timedelta(minutes=1)
            stamps = [opening] + list(pd.date_range(day + 'T14:00:00Z', day + 'T21:00:00Z', freq='min'))
            rows += [quote(t, 100. + d + .25 * (i % 23)) for i, t in enumerate(stamps)]
        self.raw_rows = pd.DataFrame(rows)
        self.dataset = self.prepare(self.raw_rows, 'original')

    def prepare(self, rows, name):
        """每个 UTC 分区携带完整来源标记，由正式入口处理休市、缺格和特征。"""
        paths = []
        source = self.root / name
        source.mkdir()
        for day, part in rows.groupby(rows.ts_event.dt.date.astype(str), sort=True):
            metadata = dict(sampling_clock='ts_recv', interval_ms='60000', max_age_ms='60000',
                source_file='synthetic.raw', source_sha256='a' * 64, source_date_utc=day,
                partial_prefix='false', source_condition='available')
            table = pa.Table.from_pandas(part[SNAPSHOT_SCHEMA.names], schema=SNAPSHOT_SCHEMA,
                preserve_index=False).replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
            path = source / (day + '.parquet')
            pq.write_table(table, path, row_group_size=100)
            paths.append(path)
        output = source / 'prepared'
        prepare_snapshot_dataset(paths, output, horizons_ms=(60000, 180000))
        return output

    def frozen(self, dataset=None, name='plan'):
        """先保存冻结计划，运行时只有该计划作为实验参数来源。"""
        plan = main(['freeze', '--protocol', str(self.config), '--dataset-dir', str(dataset or self.dataset),
                     '--output-dir', str(self.root / name)])
        return plan, self.root / name / 'plan.json'

    def test_cli_four_phases_full_quotes_and_reproducible_outputs(self):
        """回放包含未预热行情，无交易现金不被丢弃；同一冻结计划可重现。"""
        plan, path = self.frozen()
        result = main(['run', '--plan', str(path), '--output-dir', str(self.root / 'result')])
        self.assertEqual(result, run_frozen_protocol(path))
        self.assertEqual(result, json.loads((self.root / 'result/result.json').read_text()))
        self.assertEqual(list(result['phases']), list(PHASES[1:]))
        self.assertEqual(len(result['horizon_audits']), 5)
        self.assertEqual(plan['stage_roles']['calibration'], 'diagnostics_reserved_for_IRL')
        for phase in PHASES[1:]:
            day = result['phases'][phase]['daily'][0]
            audit = next(a for a in result['horizon_audits'] if a['session_id'] == day['session_id'])
            self.assertGreater(audit['all_horizons_feature_valid_rows'], 0)
            self.assertLess(audit['feature_valid_rows'], audit['observed_rows'])
            for row in day['results']:
                self.assertEqual(row['replay_rows'], audit['observed_rows'])
                self.assertTrue(row['terminal_position_liquidated'])
                self.assertFalse(row['force_replay_end'])
            self.assertEqual(day['results'][1]['net_pnl_usd'], 0.)
        self.assertIn('校准/验证仅诊断', (self.root / 'result/report.md').read_text())
        with self.assertRaises(FileExistsError):
            main(['run', '--plan', str(path), '--output-dir', str(self.root / 'result')])

    def test_protocol_rejects_overlap_omitted_days_and_implicit_units(self):
        """不许按收益删去交易日，不接受跨段交叉、非网格前瞻或隐式质量许可。"""
        mutations = [dict(sessions=dict(train=self.days[:2], calibration=self.days[1:2],
                                       validation=self.days[3:4], test=self.days[4:])),
                     dict(sessions=dict(train=self.days[:1], calibration=self.days[2:3],
                                       validation=self.days[3:4], test=self.days[4:])),
                     dict(reward_horizons_ms=[60000, 60001]), dict(allow_partial=1),
                     dict(threshold=float('nan')), dict(prediction_horizon_ms=True)]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_protocol(self.protocol | change, self.calendar, 60000)

    def test_full_session_needs_previous_utc_partition(self):
        """仅有白天快照仍不是完整 session 来源；冻结检查必须发现缺少夜盘分区。"""
        quality = self.dataset / 'quality.json'
        report = json.loads(quality.read_text())
        report['inputs'] = [r for r in report['inputs'] if r['metadata']['source_date_utc'] != '2025-09-28']
        quality.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'missing UTC source partitions'):
            freeze_protocol(self.config, self.dataset)

    def test_frozen_plan_binds_configuration_source_and_data(self):
        """参数被改、代码被换、行情文件被改都不能复用原冻结实验身份。"""
        plan, path = self.frozen()
        changed = deepcopy(plan)
        changed['protocol']['threshold'] *= 2
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            run_frozen_protocol(path)
        path.write_text(json.dumps(plan))
        with patch('src.session_experiment.code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                run_frozen_protocol(path)
        parquet = self.dataset / 'snapshots.parquet'
        table = pq.read_table(parquet)
        pq.write_table(table, parquet, compression='none')
        with self.assertRaisesRegex(ValueError, 'data changed'):
            run_frozen_protocol(path)

    def test_two_pass_ridge_matches_batch_objective(self):
        """流式统计量必须解同一个带截距 Ridge 目标，而不是更换算法近似训练。"""
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        scaler, model, metadata = fit_streaming_ridge(reader, self.days[:2], 60000, 3660000)
        chunks = [training_samples(read_day(reader, day), self.calendar.sessions[day]['open'],
            self.calendar.sessions[day]['close'] + pd.Timedelta(nanoseconds=1), 60000, 3660000)
            for day in self.days[:2]]
        X, y = np.concatenate([r[0] for r in chunks]), np.concatenate([r[1] for r in chunks])
        batch_scaler = StandardScaler().fit(X)
        batch = Ridge(alpha=1.).fit(batch_scaler.transform(X), y)
        self.assertEqual(metadata['training_rows'], len(y))
        np.testing.assert_allclose(scaler.mean_, batch_scaler.mean_, atol=1e-10)
        np.testing.assert_allclose(scaler.scale_, batch_scaler.scale_, atol=1e-10)
        np.testing.assert_allclose(model.predict(scaler.transform(X)), batch.predict(batch_scaler.transform(X)),
                                   atol=1e-10)
        np.testing.assert_allclose(model.model.coef_, batch.coef_, atol=1e-10)

    def test_future_day_prices_do_not_change_training(self):
        """校准/验证/测试价格可改变诊断收益，不能反向改变训练参数与标准化。"""
        _, first_path = self.frozen()
        first = run_frozen_protocol(first_path)
        rows = self.raw_rows.copy()
        mask = rows.ts_event >= pd.Timestamp('2025-09-30T22:00:00Z')
        for name in rows:
            if name.startswith(('bid_px_', 'ask_px_')):
                rows.loc[mask, name] += 100.
        alternative = self.prepare(rows, 'changed-future')
        _, second_path = self.frozen(alternative, 'second-plan')
        second = run_frozen_protocol(second_path)
        self.assertEqual(first['model'], second['model'])

    def test_long_horizon_counts_gap_break_and_session_boundary(self):
        """超过一小时的目标不能越过缺格、暂停和收盘；全部失效原因须守恒。"""
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        frame = read_day(reader, self.days[0])
        audit = audit_horizons(frame, self.calendar, [60000, 180000, 3660000])
        for label in audit['labels'].values():
            self.assertEqual(label['valid_rows'] + sum(label['invalid_reasons'].values()), len(frame))
        long = audit['labels']['3660000']
        self.assertGreater(long['invalid_reasons']['scheduled_break'], 0)
        self.assertGreater(long['invalid_reasons']['session_boundary'], 0)
        self.assertLessEqual(audit['all_horizons_feature_valid_rows'], long['feature_and_label_valid_rows'])
        # 直接从原始行情去掉一格并重新准备，不能沿用原 continuous_block 隐藏缺口。
        gap_rows = self.raw_rows[self.raw_rows.ts_event != pd.Timestamp('2025-09-29T17:00:00Z')]
        gap_reader = PreparedDatasetReader(self.prepare(gap_rows, 'gap'), calendar=self.calendar)
        gap = audit_horizons(read_day(gap_reader, self.days[0]), self.calendar, [3660000])
        self.assertGreater(gap['labels']['3660000']['invalid_reasons'].get('gap', 0), 0)
        self.assertLess(gap['all_horizons_feature_valid_rows'], audit['all_horizons_feature_valid_rows'])

    def test_missing_tail_prevents_realized_pnl_aggregation(self):
        """日内未平仓盯市不能混成已实现总收益，成本与缺尾日期仍须报告。"""
        rows = [dict(session_id='2025-10-01', results=[dict(strategy='fixed', total_trades=0,
            total_fills=1, terminal_position_liquidated=False, gross_pnl_usd=5.,
            friction_usd=10., net_pnl_usd=-5., matured_order_count=0,
            unmatured_fill_rewards_at_end=1)])]
        summary = summarize_days(rows, 'fixed')
        self.assertFalse(summary['pnl_aggregation_defined'])
        self.assertIsNone(summary['net_pnl_usd'])
        self.assertEqual(summary['unsettled_sessions'], ['2025-10-01'])
        self.assertEqual(summary['friction_usd'], 10.)

    def test_no_feature_day_remains_in_protocol_and_cash_comparison(self):
        """不能因当天预热不足而删掉无交易日；两个基线都须保留零收益记录。"""
        rows = self.raw_rows.copy()
        opening, closing = [self.calendar.sessions[self.days[4]][k] for k in ('open', 'close')]
        mask = (rows.ts_event > opening) & (rows.ts_event <= closing)
        rows = pd.concat([rows[~mask], rows[mask].tail(2)], ignore_index=True).sort_values('ts_event')
        sparse = self.prepare(rows, 'no-features')
        _, path = self.frozen(sparse)
        result = run_frozen_protocol(path)
        audit = result['horizon_audits'][-1]
        self.assertEqual((audit['observed_rows'], audit['feature_valid_rows']), (2, 0))
        for row in result['phases']['test']['daily'][0]['results']:
            self.assertEqual((row['replay_rows'], row['total_trades'], row['net_pnl_usd']), (2, 0, 0.))

    def test_reader_detects_changes_during_multi_pass_training(self):
        """已校验读者不能在第二遍无声读取被替换的行情或质量文件。"""
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        day = self.days[0]
        self.assertEqual(len(read_day(reader, day)), reader.report['sessions'][0]['observed_snapshots'])
        quality = self.dataset / 'quality.json'
        quality.write_text(quality.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'changed'):
            read_day(reader, day)


if __name__ == '__main__':
    unittest.main()
