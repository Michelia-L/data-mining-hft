"""验证 Table 2 工程特征、按周可见性、历史隔离及可序列化模型的数学性质。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from run_weekly_library import main
from src.session_experiment import fingerprint, fit_streaming_ridge, read_day
from src.snapshot_backtest import PreparedDatasetReader
from src.snapshot_dataset import SessionCalendar, build_session_dataset
from src.weekly_library import (ALL_LIBRARY_FEATURES, FEATURE_GROUPS, build_library, fit_history,
    freeze_library, history_samples, library_features, predict_library, predict_model,
    validate_library_config, weekly_schedule)
import test_session_experiment as session_fixture
from test_time_execution import BASE, quote


def config_for_test():
    """一分钟合成网格与小抽样上限仅用于验证，不冒充论文或真实回测参数。"""
    return dict(schema_version=1, purpose='development_weekly_light_library',
        history_start_session='2025-09-29', bootstrap_session='2025-10-01', through_session='2025-10-06',
        evaluation_sessions=['2025-10-01', '2025-10-02', '2025-10-03'], history_session_counts=[1, 2],
        age_candidates_ms=[60000], prediction_horizon_ms=60000, purge_ms=3660000,
        ridge_alpha=1., tree_max_depth=3, tree_min_samples_leaf=2, tree_sample_limit=64, seed=42,
        allow_partial=False, include_degraded=False)


def feature_frame(stamps):
    """总挂单深度每格增加一合约，价格每格增加 0.25，便于手算差分与单位。"""
    calendar = SessionCalendar()
    raw = pd.DataFrame([quote(t, 100. + i * .25) for i, t in enumerate(stamps)])
    raw['bid_sz_00'] += np.arange(len(raw), dtype=np.uint32)
    assigned, _, _ = calendar.assign(raw)
    return build_session_dataset(assigned, calendar, 500, (5000,))[0]


class LibraryFeatureTests(unittest.TestCase):
    """直接从盘口手算特征，确认右对齐窗口及缺格后重启。"""
    def test_depth_units_cross_features_and_group_separation(self):
        """10 格总体方差为 8.25，五格深度速度为 2 合约/秒，交叉量不混单位。"""
        frame = feature_frame(pd.date_range(BASE, periods=80, freq='500ms'))
        features = library_features(frame, 500)
        self.assertEqual(features.loc[9, 'displayed_depth'], 119.)
        self.assertEqual(features.loc[9, 'depth_diff_1'], 1.)
        self.assertAlmostEqual(features.loc[9, 'depth_ma_10'], 114.5)
        self.assertAlmostEqual(features.loc[9, 'depth_var_10'], 8.25)
        self.assertAlmostEqual(features.loc[9, 'depth_speed_5'], 2.)
        self.assertAlmostEqual(features.loc[9, 'price_diff_1_x_depth'], .25 * 119)
        self.assertAlmostEqual(features.loc[9, 'price_diff_1_div_depth'], .25 / 119)
        self.assertEqual(len(ALL_LIBRARY_FEATURES), 19)
        groups = list(FEATURE_GROUPS.values())
        self.assertFalse(set(groups[0]) & set(groups[1]))
        self.assertFalse(set(groups[1]) & set(groups[2]))

    def test_future_changes_cannot_change_past_features(self):
        """改未来价格/挂单量并重算原特征，不能影响前 50 格任何新旧特征。"""
        frame = feature_frame(pd.date_range(BASE, periods=80, freq='500ms'))
        original = library_features(frame, 500)
        raw = frame.copy()
        raw.loc[50:, 'bid_sz_00'] += 1000
        for name in raw:
            if name.startswith(('bid_px_', 'ask_px_')):
                raw.loc[50:, name] += 100.
        changed = build_session_dataset(raw, SessionCalendar(), 500, (5000,))[0]
        pd.testing.assert_frame_equal(original.iloc[:50], library_features(changed, 500).iloc[:50])

    def test_gap_resets_rolling_depth_and_price_history(self):
        """缺一分钟行情后首格没有前值，10 格均值须重新预热，不跨块借历史。"""
        stamps = list(pd.date_range(BASE, periods=40, freq='500ms'))
        stamps += list(pd.date_range(BASE + pd.Timedelta(minutes=2), periods=40, freq='500ms'))
        features = library_features(feature_frame(stamps), 500)
        self.assertTrue(np.isnan(features.loc[40, 'depth_diff_1']))
        self.assertTrue(features.loc[40:48, 'depth_ma_10'].isna().all())
        self.assertTrue(np.isfinite(features.loc[49, 'depth_ma_10']))
        self.assertTrue(np.isnan(features.loc[40, 'price_diff_1_x_depth']))


class WeeklyLibraryTests(unittest.TestCase):
    """真实 prepared 合成夹具包含完整源分区；不继承或重复其现有测试。"""
    prepare = session_fixture.SessionExperimentTests.prepare

    def setUp(self):
        session_fixture.SessionExperimentTests.setUp(self)
        self.library_config = config_for_test()
        self.config_path = self.root / 'library-config.json'
        self.config_path.write_text(json.dumps(self.library_config))

    def frozen(self, dataset=None, name='library-plan'):
        plan = main(['freeze', '--config', str(self.config_path), '--dataset-dirs', str(dataset or self.dataset),
                     '--output-dir', str(self.root / name)])
        return plan, self.root / name / 'plan.json'

    def test_weekly_schedule_uses_trade_date_open_and_past_sessions(self):
        """10-06 的可用时间是 10-05 UTC 夜盘；周更新使用最近已结束 session。"""
        schedule = weekly_schedule(self.library_config, self.calendar)
        self.assertEqual([r['update_session'] for r in schedule], ['2025-10-01', '2025-10-06'])
        self.assertEqual(schedule[1]['available_at_utc'], '2025-10-05T22:00:00+00:00')
        self.assertEqual(schedule[0]['training_sessions']['2'], self.days[:2])
        self.assertEqual(schedule[1]['training_sessions']['2'], self.days[-2:])
        # 模拟周一休市，更新应在该周第一个实际交易日，不是固定星期一。
        holiday = SessionCalendar()
        del holiday.sessions['2025-10-06']
        schedule = weekly_schedule(self.library_config | {'through_session': '2025-10-07'}, holiday)
        self.assertEqual(schedule[-1]['update_session'], '2025-10-07')

    def test_streaming_group_ridge_matches_batch_objective(self):
        """每个特征组两遍充分统计量应与同一历史的批量标准化 Ridge 一致。"""
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        parts = [history_samples(reader, d, self.library_config) for d in self.days[:2]]
        x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        features = pd.DataFrame(x, columns=ALL_LIBRARY_FEATURES)
        models = fit_history(reader, self.days[:2], self.library_config, self.calendar.sessions[self.days[2]]['open'])
        for model in models:
            if model['family'] != 'Ridge':
                continue
            subset = features[model['features']]
            scaler = StandardScaler().fit(subset.to_numpy())
            batch = Ridge(alpha=1.).fit(scaler.transform(subset.to_numpy()), y)
            np.testing.assert_allclose(model['scaler_mean'], scaler.mean_, atol=1e-10)
            np.testing.assert_allclose(predict_model(model, features), batch.predict(scaler.transform(subset.to_numpy())), atol=1e-10)
            self.assertEqual(model['training_rows'], len(y))

    def test_tree_portable_parameters_match_sklearn_and_sample_bound(self):
        """JSON 树与 sklearn 走同一 float32 分支；样本上限不取文件前缀。"""
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        x, y, times = history_samples(reader, self.days[0], self.library_config)
        models = fit_history(reader, self.days[:1], self.library_config, self.calendar.sessions[self.days[2]]['open'])
        keys = np.random.default_rng(42).random(len(y))
        selected = np.argsort(keys)[:64]
        features = pd.DataFrame(x, columns=ALL_LIBRARY_FEATURES)
        self.assertFalse(np.array_equal(np.sort(selected), np.arange(64)))
        for model in models:
            if model['family'] != 'DecisionTree':
                continue
            z = (features[model['features']].to_numpy() - model['scaler_mean']) / model['scaler_scale']
            tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2, random_state=42).fit(z[selected], y[selected])
            restored = json.loads(json.dumps(model))
            np.testing.assert_array_equal(predict_model(restored, features), tree.predict(z))
            self.assertEqual(restored['tree_sample_rows'], 64)
            self.assertLessEqual(max(restored['tree']['n_node_samples']), 64)

    def test_cli_versions_diagnostics_reproducibility_and_fixed_reference(self):
        """两周版本和全部 12 模型可复现；固定 11 特征对照保留原数学目标。"""
        plan, path = self.frozen()
        result = main(['build', '--plan', str(path), '--output-dir', str(self.root / 'library-result')])
        self.assertEqual(result, build_library(path))
        self.assertEqual(result, json.loads((self.root / 'library-result/library.json').read_text()))
        self.assertIsNone(result['selected_age_ms'])
        case = result['age_cases'][0]
        self.assertEqual(len(case['versions']), 2)
        for version in case['versions']:
            self.assertEqual(len(version['models']), 12)
            self.assertTrue(all(m['status'] == 'fitted' for m in version['models']))
        first_id = case['versions'][0]['version_sha256']
        for diagnostic in case['prediction_diagnostics']:
            self.assertEqual(diagnostic['version_ids'], [first_id])
            self.assertEqual(len(diagnostic['candidates']), 13)
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        _, _, reference = fit_streaming_ridge(reader, self.days[:2], 60000, 3660000)
        self.assertEqual(case['fixed_reference'], reference | {'status': 'fitted'})
        self.assertIn('预测诊断', (self.root / 'library-result/report.md').read_text())
        with self.assertRaises(FileExistsError):
            main(['build', '--plan', str(path), '--output-dir', str(self.root / 'library-result')])

    def test_future_changes_affect_only_later_eligible_versions(self):
        """改 10-02/03 行情不能影响 10-01 模型，但允许它成为 10-06 已知历史。"""
        _, first = self.frozen()
        original = build_library(first)['age_cases'][0]
        rows = self.raw_rows.copy()
        mask = rows.ts_event >= pd.Timestamp('2025-10-01T22:00:00Z')
        for name in rows:
            if name.startswith(('bid_px_', 'ask_px_')):
                rows.loc[mask, name] += 100.
        rows.loc[mask, 'bid_sz_00'] *= 2
        other = self.prepare(rows, 'future-library-quotes')
        _, second = self.frozen(other, 'future-plan')
        changed = build_library(second)['age_cases'][0]
        self.assertEqual(original['versions'][0], changed['versions'][0])
        self.assertEqual(original['fixed_reference'], changed['fixed_reference'])
        self.assertNotEqual(original['versions'][1], changed['versions'][1])

    def test_future_versions_cannot_predict_early_and_grid_is_bound(self):
        """建好的未来版本不能提前使用，启动前为 NaN；声明不同网格须拒绝。"""
        _, path = self.frozen()
        versions = build_library(path)['age_cases'][0]['versions']
        reader = PreparedDatasetReader(self.dataset, calendar=self.calendar)
        before = read_day(reader, self.days[0])
        predictions, ids = predict_library(versions, before, 60000)
        self.assertTrue(np.isnan(predictions).all())
        self.assertTrue(all(v is None for v in ids))
        current = read_day(reader, self.days[2])
        predictions, ids = predict_library(versions, current, 60000)
        self.assertEqual(set(ids), {versions[0]['version_sha256']})
        self.assertTrue(np.isfinite(predictions[current.feature_valid]).all())
        # 构造下一周输入，仅验证路由时刻，不把移位行情当真实市场结果。
        future = current.copy()
        future['ts_event'] += pd.Timedelta(days=7)
        _, ids = predict_library(versions, future, 60000)
        self.assertEqual(set(ids), {versions[1]['version_sha256']})
        with self.assertRaisesRegex(ValueError, 'grid'):
            predict_library(versions, current, 500)
        tampered = deepcopy(versions);tampered[0]['models'][0]['coefficients'][0] += 1
        with self.assertRaisesRegex(ValueError, 'integrity'):
            predict_library(tampered, current, 60000)
        aged = current.copy();aged['age_ms'] = 60001.
        with self.assertRaisesRegex(ValueError, 'age exceeds'):
            predict_library(versions, aged, 60000)

    def test_cached_future_labels_do_not_filter_features_or_training(self):
        """抹去全部离线标签缓存仍能从实价重算标签，不能改变模型版本和预测。"""
        _, first = self.frozen()
        original = build_library(first)['age_cases'][0]
        other = self.prepare(self.raw_rows, 'library-censored-cache')
        path = other / 'snapshots.parquet'
        table = pq.read_table(path);frame = table.to_pandas()
        for name in frame:
            if name.startswith('label_valid_'):
                frame[name] = False
            elif name.startswith(('future_return_', 'future_price_diff_')):
                frame[name] = np.nan
        frame['learning_ready'] = False
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(table.schema.metadata), path)
        _, second = self.frozen(other, 'library-censored-plan')
        self.assertEqual(original, build_library(second)['age_cases'][0])

    def test_zero_samples_are_preserved_and_incomplete_history_is_rejected(self):
        """超长 purge 没有训练样本时保留全部候选失败状态，缺历史不偷换窗口。"""
        self.library_config['purge_ms'] = 86400000
        self.config_path.write_text(json.dumps(self.library_config))
        _, path = self.frozen()
        case = build_library(path)['age_cases'][0]
        for version in case['versions']:
            self.assertTrue(all(m['status'] == 'insufficient_samples' for m in version['models']))
        self.assertEqual(case['fixed_reference']['status'], 'insufficient_samples')
        self.assertTrue(all(row['rmse_relative_return'] is None for day in case['prediction_diagnostics'] for row in day['candidates']))
        with self.assertRaisesRegex(ValueError, 'Not enough complete'):
            weekly_schedule(config_for_test() | {'history_session_counts': [1, 3]}, self.calendar)

    def test_frozen_config_source_and_data_guards(self):
        """缺模型维度、隐式质量许可、参数/源码/数据变更都不得复用冻结身份。"""
        for change in [dict(history_session_counts=[2]), dict(tree_sample_limit=1),
                       dict(allow_partial=1), dict(purge_ms=1), dict(ridge_alpha=float('inf'))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_library_config(config_for_test() | change, self.calendar)
        with self.assertRaisesRegex(ValueError, 'every declared age'):
            freeze_library(self.config_path, [self.dataset, self.dataset])
        plan, path = self.frozen()
        changed = deepcopy(plan);changed['config']['seed'] += 1
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            build_library(path)
        path.write_text(json.dumps(plan))
        with patch('src.weekly_library.library_code_hashes', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                build_library(path)
        quality = self.dataset / 'quality.json';quality.write_text(quality.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            build_library(path)


if __name__ == '__main__':
    unittest.main()
