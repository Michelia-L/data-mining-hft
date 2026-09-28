"""Run the BTCUSDT 1-second FMATO capability-bounded replication."""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

from src.btc_data_loader import load_btc_1s, validate_1s_grid, extract_btc_features, prepare_train_test_split, HORIZONS
from src.model_library import build_model_library, train_model_library
from src.irl_reward import IRLRewardLearner
from src.model_selector import build_all_selectors
from src.btc_execution_engine import BTCExecutionEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/BTCUSDT-1s-2024-03-01.part*.csv")
    p.add_argument("--output", default="results/btc_experiment_summary.json")
    p.add_argument("--cost-bps-per-side", type=float, default=3.0,
                   help="Explicit fixed execution-cost assumption; source bars have no bid/ask spread.")
    p.add_argument("--trade-threshold", type=float, default=8e-5)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    started = time.perf_counter()

    raw = load_btc_1s(args.data)
    quality = validate_1s_grid(raw)
    df, feature_cols = extract_btc_features(raw)
    split = prepare_train_test_split(df, feature_cols)
    train_df, test_df = split["train_df"], split["test_df"]
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    models = build_model_library()
    latencies = train_model_library(models, X_train, y_train)
    all_preds = np.zeros((len(test_df), len(models)))
    model_eval = []
    for idx, model in enumerate(models):
        preds = model.predict(X_test)
        all_preds[:, idx] = preds
        mse = float(np.mean((preds - y_test) ** 2))
        dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))
        signal_rate = float(np.mean(np.abs(preds) > args.trade_threshold))
        model_eval.append({
            "name": model.name,
            "mse": mse,
            "direction_accuracy": dir_acc,
            "signal_rate_at_threshold": signal_rate,
            "latency_us_per_row": float(latencies[model.name]),
        })

    irl = IRLRewardLearner(horizons=HORIZONS)
    weights = irl.fit_reward_weights(train_df)

    strategies = []
    engine = BTCExecutionEngine(
        irl_learner=irl,
        trade_threshold=args.trade_threshold,
        holding_period=30,
        cost_bps_per_side=args.cost_bps_per_side,
    )
    for selector in build_all_selectors(models):
        res = engine.run_backtest(selector, test_df, X_test, all_preds)
        compact_res = {
            k: v for k, v in res.items()
            if k not in {"gross_curve", "net_curve_assumed_cost", "timestamps"}
        }
        strategies.append(compact_res)
        print(f"{res['strategy']:<28} trades={res['total_trades']:4d} gross={res['gross_return']*100:8.3f}% net@{args.cost_bps_per_side:g}bps/side={res['net_return_assumed_cost']*100:8.3f}%")

    out = {
        "experiment": "BTCUSDT 1-second FMATO capability-bounded replication",
        "source_data": os.path.basename(args.data),
        "data_semantics": "Headerless Binance-style 1-second OHLCV/trade-flow klines; no L2 order book or observed bid/ask spread.",
        "random_seed": 42,
        "horizons_seconds": HORIZONS,
        "target": "30-second forward close-to-close return",
        "train_test_split": "chronological 70/30; scaler fit on train only",
        "execution_cost_model": {
            "type": "fixed assumed cost because bid/ask is unavailable",
            "default_bps_per_side": args.cost_bps_per_side,
            "observed_spread_used": False,
            "sensitivity_bps_per_side": [0, 1, 3, 5, 10],
        },
        "trade_threshold": args.trade_threshold,
        "data_quality": quality,
        "post_feature_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": feature_cols,
        "model_eval": model_eval,
        "irl_weights": {f"{h}_seconds": float(w) for h, w in zip(HORIZONS, weights)},
        "strategies": strategies,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "No L2/MBP order book: original OBI, microprice, depth and observed spread cannot be reproduced.",
            "Trade imbalance from taker-buy volume is a replacement feature, not the paper's order-book imbalance.",
            "Execution uses close as the available reference price and an explicit fixed-cost scenario, not queue-aware fills.",
            "Only one day of BTCUSDT data is included, so cross-day regime robustness and annualized risk statistics are not claimed.",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
