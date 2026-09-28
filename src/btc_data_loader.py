"""BTCUSDT 1-second kline loader for the FMATO core-method replication.

The source file is Binance-style, headerless 12-column kline data.  Because
there is no L2 order book, this module deliberately substitutes trade-flow and
bar microstructure features for OBI/spread/microprice features.  It never
fabricates bid/ask quotes.
"""
from __future__ import annotations
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

BTC_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "num_trades", "taker_buy_base_volume",
    "taker_buy_quote_volume", "ignore",
]
HORIZONS = [10, 30, 90]
TRAIN_RATIO = 0.70


def load_btc_1s(file_path: str) -> pd.DataFrame:
    paths = sorted(glob.glob(file_path)) if any(ch in file_path for ch in "*?[") else [file_path]
    if not paths:
        raise FileNotFoundError(f"No BTC data files matched: {file_path}")
    frames = [pd.read_csv(path, header=None, names=BTC_COLUMNS) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [c for c in BTC_COLUMNS if c != "ignore"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["open_time", "open", "high", "low", "close", "volume"], inplace=True)
    df["ts_event"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset=["open_time"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def validate_1s_grid(df: pd.DataFrame) -> Dict:
    diffs = df["open_time"].diff().dropna()
    return {
        "rows": int(len(df)),
        "start_utc": str(df["ts_event"].iloc[0]),
        "end_utc": str(df["ts_event"].iloc[-1]),
        "unique_timestamps": int(df["open_time"].nunique()),
        "one_second_intervals": int((diffs == 1000).sum()),
        "non_one_second_intervals": int((diffs != 1000).sum()),
        "zero_volume_bars": int((df["volume"] == 0).sum()),
        "missing_values": int(df.isna().sum().sum()),
    }


def extract_btc_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    eps = 1e-12

    # Keep the first feature positions compatible with the repository's
    # MomentumRuleModel: index 2 = imbalance proxy; index 6 = 5-second return.
    df["range_pct"] = (df["high"] - df["low"]) / (df["open"] + eps)
    df["candle_body"] = (df["close"] - df["open"]) / (df["open"] + eps)

    buy_ratio = np.where(
        df["volume"].to_numpy() > 0,
        df["taker_buy_base_volume"].to_numpy() / (df["volume"].to_numpy() + eps),
        0.5,
    )
    df["taker_buy_ratio"] = np.clip(buy_ratio, 0.0, 1.0)
    df["trade_imbalance"] = 2.0 * df["taker_buy_ratio"] - 1.0

    df["log_volume"] = np.log1p(df["volume"].clip(lower=0))
    df["ret_lag_1"] = df["close"].pct_change(1)
    df["ret_lag_5"] = df["close"].pct_change(5)
    df["ret_lag_10"] = df["close"].pct_change(10)
    df["ret_lag_30"] = df["close"].pct_change(30)
    df["vol_10"] = df["close"].pct_change().rolling(10).std()
    df["vol_30"] = df["close"].pct_change().rolling(30).std()
    df["log_num_trades"] = np.log1p(df["num_trades"].clip(lower=0))
    df["volume_change_5"] = np.log1p(df["volume"].clip(lower=0)).diff(5)

    # Close is used only as the available 1-second reference price.  We do not
    # call it a real mid-price and do not derive a synthetic bid/ask spread.
    df["reference_price"] = df["close"]

    features = [
        "range_pct",          # 0
        "candle_body",        # 1
        "trade_imbalance",    # 2 -- momentum rule compatibility
        "taker_buy_ratio",    # 3
        "log_volume",         # 4
        "ret_lag_1",          # 5
        "ret_lag_5",          # 6 -- momentum rule compatibility
        "ret_lag_10",         # 7
        "ret_lag_30",         # 8
        "vol_10",             # 9
        "vol_30",             # 10
        "log_num_trades",     # 11
        "volume_change_5",    # 12
    ]

    for h in HORIZONS:
        df[f"future_ret_{h}"] = df["close"].shift(-h) / (df["close"] + eps) - 1.0

    df["target_ret"] = df["future_ret_30"]
    df["target_dir"] = np.sign(df["target_ret"])

    required = features + [f"future_ret_{h}" for h in HORIZONS]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, features


def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    train_size = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0.0, 1.0)
    X_train = ((train_df[feature_cols] - mean) / std).to_numpy()
    X_test = ((test_df[feature_cols] - mean) / std).to_numpy()

    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df["target_ret"].to_numpy(),
        "y_test": test_df["target_ret"].to_numpy(),
        "norm_mean": {k: float(v) for k, v in mean.items()},
        "norm_std": {k: float(v) for k, v in std.items()},
    }
