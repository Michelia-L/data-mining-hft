# BTCUSDT 1-second FMATO capability-bounded replication

## 1. Scope

This experiment applies the repository's FMATO core workflow to the supplied `BTCUSDT-1s-2024-03-01.csv` within the information actually present in the data.

It is **not** a strict reproduction of the paper's L2/order-book experiment. The BTC file contains one-second OHLCV and trade-flow aggregates, but no bid/ask quotes, order-book depth, queue position, or observed spread. The implementation therefore preserves the multi-model + IRL multi-horizon reward + UCB/ARS online model-selection structure while replacing unavailable order-book features with observable bar/trade-flow features.

## 2. Data

- Files: `data/BTCUSDT-1s-2024-03-01.part01.csv` ... `part09.csv` (lossless sequential split of the supplied CSV; concatenating the parts reproduces the original 86,400 rows)
- Format: headerless Binance-style 12-column 1-second kline records
- UTC coverage: 2024-03-01 00:00:00 through 23:59:59
- Raw rows: 86,400
- SHA-256 of the original file and concatenated nine-part dataset: `7b5fb5e555446887284c2d73c6831b0bf21426df473c660c1300416861d1b3b1`
- Unique timestamps: 86,400
- One-second intervals: 86,399 / 86,399
- Missing values: 0
- Zero-volume bars: 933

The loader assigns the standard columns:

`open_time, open, high, low, close, volume, close_time, quote_volume, num_trades, taker_buy_base_volume, taker_buy_quote_volume, ignore`.

## 3. Feature mapping

Unavailable paper/order-book inputs are not fabricated. In particular, this BTC experiment does not claim to reproduce OBI, microprice, L2 depth, or real bid/ask spread.

Instead it uses 13 observable features:

1. one-second high-low range / open;
2. candle body return;
3. taker trade imbalance = `2 * taker_buy_base_volume / volume - 1`;
4. taker buy ratio;
5. log volume;
6. 1-second lagged return;
7. 5-second lagged return;
8. 10-second lagged return;
9. 30-second lagged return;
10. 10-second rolling volatility;
11. 30-second rolling volatility;
12. log trade count;
13. five-second log-volume change.

The multi-scale horizons are translated from 10/30/90 ticks to **10/30/90 seconds**, because the source grid is exactly one second.

The supervised target is the 30-second forward close-to-close return. Data is split chronologically 70/30. Feature normalization is fit only on the training segment and then applied to the test segment.

After feature construction and removal of rolling/forward-label boundary rows:

- usable rows: 86,280;
- training rows: 60,395;
- test rows: 25,885.

## 4. Model library

The existing repository model library is reused:

- Ridge linear regression;
- logistic direction model;
- shallow decision tree;
- histogram GBDT;
- momentum/imbalance rule.

Test-set results from the committed run:

| Model | Direction accuracy | MSE | Signal rate at 8e-5 threshold |
|---|---:|---:|---:|
| Ridge_Linear | 55.12% | 2.728e-7 | 11.11% |
| Logistic_Direction | 54.99% | 2.734e-7 | 9.83% |
| Decision_Tree | 54.53% | 2.765e-7 | 15.16% |
| Hist_GBDT | 55.22% | 2.752e-7 | 1.93% |
| Rule_Momentum | 53.96% | 3.911e-7 | 89.99% |

Inference latency values are also stored in the JSON, but they are machine/runtime dependent and should not be treated as stable scientific results.

## 5. IRL multi-scale reward

With random seed 42, the learned weights are:

| Horizon | Weight |
|---|---:|
| 10 seconds | 11.66% |
| 30 seconds | 22.75% |
| 90 seconds | 65.59% |

The fitted reward therefore emphasizes the longest available horizon in this one-day BTC sample.

## 6. Causal online feedback

A small but important implementation detail differs from the repository's original minimal backtest. In this BTC path, an action taken at time `t` is **not** immediately rewarded using future labels. Its multi-horizon reward is released only after 90 seconds, when the longest 90-second outcome has actually become observable.

This prevents the online UCB/ARS selector from using future information to choose the next model.

## 7. Execution assumptions

The source data has no real bid/ask spread. Therefore:

- `close` is used as the available reference price;
- no synthetic observed spread is invented;
- net-return figures use an explicit fixed cost assumption;
- the default run assumes **3 bps per side**;
- each completed round trip therefore pays 6 bps;
- the result JSON also reports 0/1/3/5/10 bps-per-side sensitivity.

These are scenario assumptions, not claims about a specific Binance fee tier or realized market spread.

Returns below are sums of completed-trade returns, matching the style of the existing minimal project; they are not a compounded capital curve.

## 8. Backtest result

| Strategy | Trades | Gross return | Net @ 3 bps/side | Break-even cost/side |
|---|---:|---:|---:|---:|
| FMATO-ME-UCBS | 2,046 | +9.91% | -112.85% | 0.24 bp |
| FMATO-OE-UCBS | 2,101 | +8.32% | -117.74% | 0.20 bp |
| FMATO-ME-ARS | 3,071 | +11.38% | -172.88% | 0.19 bp |
| FMATO-OE-ARS | 578 | +4.99% | -29.69% | 0.43 bp |
| Baseline-Single-HistGBDT | 124 | +0.97% | -6.47% | 0.39 bp |
| Baseline-Static-Ensemble | 2,628 | +21.78% | -135.90% | 0.41 bp |
| Baseline-Random | 2,306 | +11.98% | -126.38% | 0.26 bp |

The important result is not the large arithmetic gross-return number by itself. The strategies trade thousands of times in a single day, so even extremely small per-side friction overwhelms the gross edge. For example, at only **1 bp per side**, every tested strategy in the committed run is already net negative.

This sample therefore supports a limited conclusion: the feature/model stack contains some short-horizon directional information, but the tested high-turnover decision rules are nowhere near robust enough to claim executable profitability with realistic nonzero friction.

## 9. Reproduce

From the repository root:

```bash
pip install -r requirements.txt
python run_btc_experiment.py
```

Optional explicit cost assumption:

```bash
python run_btc_experiment.py --cost-bps-per-side 1
```

Default output:

```text
results/btc_experiment_summary.json
```

## 10. Limitations

1. One day is too short to establish regime robustness or generalization across days/months.
2. No L2 order book means the paper's OBI, microprice, depth, real spread, queue position and passive-fill mechanics cannot be reproduced.
3. Taker-volume imbalance is a substitute feature, not order-book imbalance.
4. The close-price execution reference is a bar-level approximation; no claim of exchange-grade fill simulation is made.
5. Fixed bps cost scenarios are sensitivity assumptions, not observed trading costs.
6. The experiment does not report annualized Sharpe from one day of data; the JSON stores a non-annualized per-trade mean/std statistic instead.
7. The model-selection algorithms and IRL implementation remain a minimal methodological reproduction rather than a full reconstruction of the paper's proprietary engineering system.
