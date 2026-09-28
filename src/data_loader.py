"""Bounded, instrument-safe input and purged chronological partitions."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, List, Dict
from src.config import SAMPLE_TICKS, SPLIT_RATIOS, HORIZONS


def dataset_row_count(path):
    if str(path).endswith('.parquet'):
        return pq.ParquetFile(path).metadata.num_rows
    with open(path, 'rb') as stream:
        return sum(1 for _ in stream) - 1


def load_and_preprocess_ticks(file_path, nrows=SAMPLE_TICKS, offset=0):
    if offset < 0 or (nrows is not None and nrows <= 0):
        raise ValueError('offset must be nonnegative and nrows must be positive')
    columns = ['ts_event', 'sequence', 'instrument_id', 'symbol'] + [
        f'{field}_{i:02d}' for i in range(5)
        for field in ['bid_px', 'ask_px', 'bid_sz', 'ask_sz']]
    chunks, seen, remaining = [], 0, nrows
    if str(file_path).endswith('.parquet'):
        source = (b.to_pandas() for b in pq.ParquetFile(file_path).iter_batches(
            batch_size=65536, columns=columns))
    else:
        source = pd.read_csv(file_path, usecols=columns, chunksize=65536)
    for chunk in source:
        end = seen + len(chunk)
        if end <= offset:
            seen = end
            continue
        chunk = chunk.iloc[max(0, offset - seen):]
        if remaining is not None:
            chunk = chunk.iloc[:remaining]
            remaining -= len(chunk)
        chunks.append(chunk)
        seen = end
        if remaining == 0:
            break
    if not chunks:
        raise ValueError('Requested data window is empty')
    df = pd.concat(chunks, ignore_index=True)
    if df.instrument_id.nunique(dropna=False) != 1 or df.symbol.nunique(dropna=False) != 1:
        raise ValueError('Select one instrument before computing event horizons')
    numeric = [c for c in columns if c.startswith(('bid_', 'ask_'))]
    valid = np.isfinite(df[numeric]).all(axis=1)
    valid &= (df.bid_px_00 > 0) & (df.ask_px_00 >= df.bid_px_00)
    valid &= (df[[c for c in numeric if '_sz_' in c]] >= 0).all(axis=1)
    valid &= (df.bid_sz_00 > 0) & (df.ask_sz_00 > 0)
    df = df.loc[valid].copy()
    df['ts_event'] = pd.to_datetime(df.ts_event, utc=True, errors='raise')
    # Keep exchange sequence and source order for tied timestamps.
    df.sort_values(['ts_event', 'sequence'], kind='stable', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if df.empty:
        raise ValueError('No valid two-sided quotes')
    return df


def extract_microstructure_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建高频订单簿微观结构特征。

    包含四大类特征：
    1. 价格与点差特征 (Mid-price, Spread, Relative Spread)
    2. 买卖深度失衡度 (Order Book Imbalance, OBI)
    3. 微观加权价格与偏差 (Microprice Deviation)
    4. 滞后收益率与时序动量、波动率 (Momentum & Volatility)

    参数:
        df (pd.DataFrame): 原始盘口数据

    返回:
        Tuple[pd.DataFrame, List[str]]: 添加特征后的数据框及特征名称列表
    """
    eps = 1e-8
    features = []
    
    # 1. 基础价格与价差
    df['mid_price'] = (df['ask_px_00'] + df['bid_px_00']) / 2.0
    df['spread'] = df['ask_px_00'] - df['bid_px_00']
    df['rel_spread'] = df['spread'] / (df['mid_price'] + eps)
    features.extend(['spread', 'rel_spread'])
    
    # 2. 一级盘口失衡度 (Level 1 Order Book Imbalance)
    # 取值范围 [-1, 1]，正值表示买方力量强，负值表示卖方力量强
    df['obi_l1'] = (df['bid_sz_00'] - df['ask_sz_00']) / (df['bid_sz_00'] + df['ask_sz_00'] + eps)
    features.append('obi_l1')
    
    # 3. 前5档深度加权失衡度 (Multi-level OBI)
    # 越浅层权重越高，符合高频市场微观结构直觉
    weights = [1.0, 0.5, 0.33, 0.25, 0.2]
    w_bid_sz = sum(df[f'bid_sz_0{i}'] * w for i, w in enumerate(weights))
    w_ask_sz = sum(df[f'ask_sz_0{i}'] * w for i, w in enumerate(weights))
    df['obi_multi'] = (w_bid_sz - w_ask_sz) / (w_bid_sz + w_ask_sz + eps)
    features.append('obi_multi')
    
    # 4. 微观加权价格 (Micro-price) 及其相对中间价的偏离度
    df['micro_price'] = (
        df['ask_px_00'] * df['bid_sz_00'] + df['bid_px_00'] * df['ask_sz_00']
    ) / (df['bid_sz_00'] + df['ask_sz_00'] + eps)
    df['micro_dev'] = (df['micro_price'] - df['mid_price']) / (df['spread'] + eps)
    features.append('micro_dev')
    
    # 5. 滞后多周期动量 (Momentum / Past Returns)
    for lag in [1, 5, 10, 30]:
        col_name = f'ret_lag_{lag}'
        df[col_name] = df['mid_price'].pct_change(lag).fillna(0.0)
        features.append(col_name)
        
    # 6. 滚动微观波动率 (Rolling Volatility)
    for window in [10, 30]:
        col_name = f'vol_{window}'
        df[col_name] = df['mid_price'].pct_change().rolling(window).std().fillna(0.0)
        features.append(col_name)
        
    # 7. 构建论文核心：多尺度未来收益率期望标签 (Future Multi-horizon Returns)
    # E_10, E_30, E_90：未来 10、30、90 ticks 的价格相对变动
    for h in HORIZONS:
        col_name = f'future_ret_{h}'
        df[col_name] = (df['mid_price'].shift(-h) - df['mid_price']) / (df['mid_price'] + eps)
        
    # 主预测目标：未来 30-tick 价格变动方向与大小
    df['target_ret'] = df[f'future_ret_{HORIZONS[1]}']
    df['target_dir'] = np.sign(df['target_ret'])
    
    # 仅清理特征异常；保留没有未来标签的尾部行情用于执行和期末平仓
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[DataLoader] 特征工程构建完成，有效特征数: {len(features)}, 样本数: {len(df)}")
    return df, features


def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """Fit -> reward calibration -> hyperparameter validation -> untouched test.

    A row is usable for fitting only when every label endpoint is strictly before
    the next partition. Test retains its unlabeled tail for execution/liquidation.
    """
    n = len(df)
    bounds = [0] + [int(n * x) for x in np.cumsum(SPLIT_RATIOS)[:-1]] + [n]
    gap = max(HORIZONS)
    partitions = {}
    for i, name in enumerate(('train', 'calibration', 'validation', 'test')):
        start, stop = bounds[i], bounds[i + 1]
        if name != 'test':
            stop -= gap
            # Purge timestamp ties too: test must start strictly after label maturity.
            while stop > start and df.ts_event.iloc[stop - 1 + gap] >= df.ts_event.iloc[bounds[i + 1]]:
                stop -= 1
        if stop - start < 100:
            raise ValueError('Window too short for four purged partitions (need at least 2000 events)')
        partitions[name] = df.iloc[start:stop].copy()
    train = partitions['train']
    mean = train[feature_cols].mean()
    std = train[feature_cols].std().replace(0., 1.).fillna(1.)
    output = dict(feature_cols=feature_cols, norm_mean=mean.to_dict(), norm_std=std.to_dict(),
                  purge_events=gap)
    for name, frame in partitions.items():
        output[f'{name}_df'] = frame
        output[f'X_{name}'] = ((frame[feature_cols] - mean) / std).to_numpy(dtype=float)
        output[f'y_{name}'] = frame.target_ret.to_numpy()
    return output
