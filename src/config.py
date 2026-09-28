"""Reproducible experiment defaults. Ticks below mean ordered book events, not seconds."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATHS = {
    key: str(next((BASE_DIR / f'{stem}{ext}' for ext in ('.parquet', '.csv')
                   if (BASE_DIR / f'{stem}{ext}').exists()), BASE_DIR / f'{stem}.parquet'))
    for key, stem in {
        'CME_ES': 'databento_glbx.mdp3_mbp_10',
        'ICE_BRENT': 'databento_ifeu.impact_mbp_10',
    }.items()
}
SEED = 42
SAMPLE_TICKS = 60000
HORIZONS = [10, 30, 90]
SPLIT_RATIOS = (0.50, 0.15, 0.15, 0.20)
INITIAL_CAPITAL = 100000.0
QUANTITY = 1
INSTRUMENT_CONFIG = {
    'CME_ES': dict(name='CME E-mini S&P 500 Futures (ES)', tick_size=0.25,
                   multiplier=50.0, commission_per_order=1.25,
                   slippage_ticks=0.5, trade_threshold=0.000015),
    'ICE_BRENT': dict(name='ICE Brent Crude Oil Futures (BRN)', tick_size=0.01,
                      multiplier=1000.0, commission_per_order=1.50,
                      slippage_ticks=0.5, trade_threshold=0.00006),
}
RL_CONFIG = dict(ucb_c=0.1, ars_window=300, holding_period=30, latency_events=1)
UCB_CANDIDATES = (0.01, 0.1, 0.8)
RESULTS_DIR = str(BASE_DIR / 'results')
SUMMARY_JSON_PATH = str(BASE_DIR / 'results' / 'experiment_summary.json')
