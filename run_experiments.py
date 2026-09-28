"""Deterministic, purged four-way experiments across disjoint event windows."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from src.config import (BASE_DIR, DATASET_PATHS, SAMPLE_TICKS, HORIZONS, SEED,
                        INITIAL_CAPITAL, QUANTITY, INSTRUMENT_CONFIG, RL_CONFIG,
                        UCB_CANDIDATES, SPLIT_RATIOS, SUMMARY_JSON_PATH)
from src.data_loader import (dataset_row_count, load_and_preprocess_ticks,
                             extract_microstructure_features, prepare_train_test_split)
from src.model_library import build_model_library, train_model_library
from src.irl_reward import IRLRewardLearner
from src.model_selector import (build_all_selectors, UCBSelector, ARSSelector,
                                SingleModelSelector, EnsembleSelector, FlatSelector, RandomSelector)
from src.execution_engine import ExecutionEngine
from src.artifacts import pretty_json


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def evaluate_models(models, predictions, y, train_y):
    valid = np.isfinite(y)
    truth = np.sign(y[valid]).astype(int)
    majority = int(np.unique(np.sign(train_y), return_counts=True)[0][
        np.argmax(np.unique(np.sign(train_y), return_counts=True)[1])])
    rows = []
    for i, model in enumerate(models):
        p = predictions[valid, i]
        direction = np.sign(p).astype(int)
        nonzero = truth != 0
        rows.append(dict(name=model.name, mse=float(np.mean((p - y[valid]) ** 2)),
                         direction_accuracy=float(np.mean(direction == truth) * 100),
                         balanced_accuracy=float(balanced_accuracy_score(truth, direction) * 100),
                         nonzero_direction_accuracy=float(np.mean(direction[nonzero] == truth[nonzero]) * 100)
                         if nonzero.any() else None,
                         confusion_matrix=confusion_matrix(truth, direction, labels=[-1, 0, 1]).tolist(),
                         latency_us=model.latency_us, latency=model.latency))
    baselines = dict(evaluated_rows=int(valid.sum()), unlabeled_tail=int((~valid).sum()),
                     class_order=['down', 'flat', 'up'],
                     class_proportions={str(c): float(np.mean(truth == c)) for c in [-1, 0, 1]},
                     zero_prediction_mse=float(np.mean(y[valid] ** 2)),
                     zero_direction_accuracy=float(np.mean(truth == 0) * 100),
                     train_majority_class=majority,
                     train_majority_accuracy=float(np.mean(truth == majority) * 100))
    return rows, baselines


def static_selectors(models):
    return ([SingleModelSelector(f'Baseline-Single-{m.name}', models, i) for i, m in enumerate(models)]
            + [EnsembleSelector('Baseline-Static-Ensemble', models), FlatSelector('Baseline-Cash', models)])


def run_window(key, path, offset, rows, seed):
    start = time.perf_counter()
    df, features = extract_microstructure_features(load_and_preprocess_ticks(path, rows, offset))
    split = prepare_train_test_split(df, features)
    models = build_model_library()
    train_model_library(models, split['X_train'], split['y_train'])
    predictions = {part: np.column_stack([m.predict(split[f'X_{part}']) for m in models])
                   for part in ('calibration', 'validation', 'test')}
    base_reward = IRLRewardLearner()
    base_reward.fit_scales(split['train_df'])
    base_threshold = INSTRUMENT_CONFIG[key]['trade_threshold']

    # Build executable expert candidates on calibration only, with next-event fills.
    calibration = split['calibration_df']
    cal_engine = ExecutionEngine(key, base_reward, threshold=base_threshold)
    cal_selectors = static_selectors(models)
    calibration_results = [cal_engine.run_backtest(s, calibration,
                           all_model_preds=predictions['calibration']) for s in cal_selectors]
    expert_idx = int(np.argmax([r['net_pnl_usd'] for r in calibration_results]))
    names = [s.name for s in cal_selectors]
    rewards = {}
    cal_mid = calibration.mid_price.to_numpy()
    count = len(calibration) - max(HORIZONS)
    future = np.column_stack([cal_mid[h:h + count] / cal_mid[:count] - 1. for h in HORIZONS])
    mu_me = []
    for i, selector in enumerate(cal_selectors):
        signal = (np.zeros(count) if selector.flat else predictions['calibration'][:count].mean(axis=1)
                  if selector.ensemble else predictions['calibration'][:count, i])
        mu_me.append(np.mean(np.sign(signal)[:, None] * future / base_reward.scales, axis=0))
    mu_oe = [r['order_feature_expectation'] for r in calibration_results]
    for reward_type, expectations in [('ME', mu_me), ('OE', mu_oe)]:
        learner = IRLRewardLearner(scales=base_reward.scales)
        learner.fit_reward_weights(expectations, expert_idx, names)
        rewards[reward_type] = learner

    # Common threshold selected by static ensemble, so all methods use the same gate.
    validation = split['validation_df']
    tuning = []
    for threshold in [base_threshold * x for x in (1, 2, 4)]:
        engine = ExecutionEngine(key, rewards['ME'], threshold=threshold)
        result = engine.run_backtest(EnsembleSelector('validation', models), validation,
                                     all_model_preds=predictions['validation'])
        tuning.append(dict(threshold=threshold, net_pnl_usd=result['net_pnl_usd']))
    threshold = max(tuning, key=lambda x: x['net_pnl_usd'])['threshold']
    c_by_reward, c_trials = {}, {}
    for reward_type in ('ME', 'OE'):
        trials = []
        for c in UCB_CANDIDATES:
            engine = ExecutionEngine(key, rewards[reward_type], threshold=threshold)
            result = engine.run_backtest(UCBSelector('validation', models, reward_type, c), validation,
                                         all_model_preds=predictions['validation'])
            trials.append(dict(c=c, net_pnl_usd=result['net_pnl_usd']))
        c_by_reward[reward_type] = max(trials, key=lambda x: x['net_pnl_usd'])['c']
        c_trials[reward_type] = trials
    val_fixed = []
    for i, model in enumerate(models):
        result = ExecutionEngine(key, rewards['ME'], threshold=threshold).run_backtest(
            SingleModelSelector('validation', models, i), validation, all_model_preds=predictions['validation'])
        val_fixed.append(result['net_pnl_usd'])
    best_fixed = int(np.argmax(val_fixed))

    test = split['test_df']
    selectors = build_all_selectors(models, c_by_reward, seed)
    selectors += [SingleModelSelector('Baseline-ValidationBest', models, best_fixed)]
    selectors += [RandomSelector(f'Baseline-Random-seed-{s}', models, s) for s in (seed + 1, seed + 2)]
    results = []
    for selector in selectors:
        result = ExecutionEngine(key, rewards[selector.reward_type], threshold=threshold).run_backtest(
            selector, test, all_model_preds=predictions['test'])
        results.append(result)
    # Same ARS, gate, data, fill model and reward scaling; only weights differ.
    for reward_type in ('ME', 'OE'):
        for name, weights in [('Equal', np.ones(len(HORIZONS)))] + [
                (f'Single-{h}', np.eye(len(HORIZONS))[i]) for i, h in enumerate(HORIZONS)]:
            learner = IRLRewardLearner(weights=weights, scales=base_reward.scales)
            selector = ARSSelector(f'Ablation-{reward_type}-ARS-{name}', models, reward_type)
            results.append(ExecutionEngine(key, learner, threshold=threshold).run_backtest(
                selector, test, all_model_preds=predictions['test']))
    model_eval, prediction_baselines = evaluate_models(models, predictions['test'],
                                                       split['y_test'], split['y_train'])
    partitions = {}
    for part in ('train', 'calibration', 'validation', 'test'):
        frame = split[f'{part}_df']
        partitions[part] = dict(rows=len(frame), start=str(frame.ts_event.iloc[0]),
                                end=str(frame.ts_event.iloc[-1]),
                                duration_seconds=float((frame.ts_event.iloc[-1] - frame.ts_event.iloc[0]).total_seconds()))
    event_seconds = (df.ts_event.shift(-max(HORIZONS)) - df.ts_event).dt.total_seconds().dropna()
    snapshot = test.iloc[0]
    l2_snapshot = dict(timestamp=str(snapshot.ts_event),
                       bids=[dict(price=float(snapshot[f'bid_px_{i:02d}']),
                                  size=float(snapshot[f'bid_sz_{i:02d}'])) for i in range(5)],
                       asks=[dict(price=float(snapshot[f'ask_px_{i:02d}']),
                                  size=float(snapshot[f'ask_sz_{i:02d}'])) for i in range(5)])
    return dict(instrument=key, source_offset=offset, requested_rows=rows, sample_size=len(df), seed=seed,
                l2_snapshot=l2_snapshot,
                symbol=str(df.symbol.iloc[0]), train_size=len(split['train_df']), test_size=len(test),
                partitions=partitions, purge_events=split['purge_events'], features=features,
                normalization=dict(mean=split['norm_mean'], std=split['norm_std']),
                horizon_90_seconds_quantiles={str(q): float(event_seconds.quantile(q)) for q in (.1, .5, .9)},
                model_eval=model_eval, prediction_baselines=prediction_baselines,
                reward_scales=base_reward.scales.tolist(),
                rewards={k: dict(weights=v.weights.tolist(), diagnostics=v.diagnostics) for k, v in rewards.items()},
                calibration_policies=[dict(strategy=r['strategy'], net_pnl_usd=r['net_pnl_usd'])
                                      for r in calibration_results],
                tuning=dict(threshold=threshold, threshold_trials=tuning, c_by_reward=c_by_reward,
                            c_trials=c_trials, validation_best_model=models[best_fixed].name,
                            validation_fixed_pnl=val_fixed),
                strategies=results, elapsed_seconds=time.perf_counter() - start)


def aggregate(windows):
    names = [r['strategy'] for r in windows[0]['strategies']]
    output = []
    for name in names:
        rows = [next(r for r in w['strategies'] if r['strategy'] == name) for w in windows]
        pnl = [r['net_pnl_usd'] for r in rows]
        output.append(dict(strategy=name, windows=len(rows), net_pnl_usd_sum=float(sum(pnl)),
                           net_pnl_usd_mean=float(np.mean(pnl)), net_pnl_usd_min=float(min(pnl)),
                           net_pnl_usd_max=float(max(pnl)), total_trades=sum(r['total_trades'] for r in rows)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=SAMPLE_TICKS, help='Raw events per independent window')
    parser.add_argument('--windows', type=int, default=3)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--output', default=SUMMARY_JSON_PATH)
    args = parser.parse_args()
    if args.rows < 2000 or args.windows < 1:
        parser.error('--rows >= 2000 and --windows >= 1 are required')
    source_files = [BASE_DIR / 'run_experiments.py'] + sorted((BASE_DIR / 'src').glob('*.py'))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    summary = dict(schema_version=2, timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'), horizons=HORIZONS,
                   metadata=dict(seed=args.seed, rows=args.rows, windows=args.windows,
                                 python=platform.python_version(), platform=platform.platform(),
                                 packages={p: importlib.metadata.version(p) for p in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'pyarrow', 'threadpoolctl']},
                                 source_sha256={str(p.relative_to(BASE_DIR)): file_hash(p) for p in source_files},
                                 git_base_revision=revision, threads=1, initial_capital=INITIAL_CAPITAL,
                                 quantity=QUANTITY, split_ratios=SPLIT_RATIOS, execution=RL_CONFIG,
                                 instruments=INSTRUMENT_CONFIG, ucb_candidates=UCB_CANDIDATES,
                                 scope='Disjoint windows within supplied files; not independent trading days.'),
                   experiments={})
    with threadpool_limits(limits=1):
        for key, path in DATASET_PATHS.items():
            if not Path(path).exists():
                raise FileNotFoundError(path)
            total = dataset_row_count(path)
            if total < args.rows * args.windows:
                raise ValueError(f'{key}: insufficient rows for nonoverlapping windows')
            offsets = np.linspace(0, total - args.rows, args.windows, dtype=int).tolist()
            windows = []
            for i, offset in enumerate(offsets):
                print(f'{key}: window {i + 1}/{args.windows}, offset={offset}', flush=True)
                windows.append(run_window(key, path, offset, args.rows, args.seed))
            summary['experiments'][key] = dict(source_file=Path(path).name, source_sha256=file_hash(path),
                                               source_rows=total, windows=windows, aggregate=aggregate(windows))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete successful run; strict JSON forbids NaN/Infinity.
    temporary = out.with_suffix('.tmp')
    temporary.write_text(pretty_json(summary) + '\n')
    temporary.replace(out)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
