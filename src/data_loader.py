"""按窗口读取订单簿、构建因果特征，并完成带标签隔离的四段时间切分。

输入是单合约 MBP 数据；模型特征只使用当前或过去行情。future_ret_* 是
离线监督标签，允许用于训练/评估，却不能被在线执行引擎直接读取。
同一份表里同时存在特征和标签，不意味着它们在实盘时具有相同的可用时间。"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from typing import Tuple, List, Dict
from src.config import SAMPLE_TICKS, SPLIT_RATIOS, HORIZONS


def dataset_row_count(path):
    """取得原始行数，为均匀选取不重叠窗口提供边界。

    Parquet 直接读元数据，不解码整表；CSV 逐行计数并减去一行表头。
    CSV 计数假定行情文件每条记录占一个物理行，不用于含多行文本字段的任意 CSV。"""
    if str(path).endswith('.parquet'):
        return pq.ParquetFile(path).metadata.num_rows
    with open(path, 'rb') as stream:
        return sum(1 for _ in stream) - 1


def load_and_preprocess_ticks(file_path, nrows=SAMPLE_TICKS, offset=0):
    """读取从 offset 开始的 nrows 条原始记录，清洗后返回按事件排序的 DataFrame。

    offset、nrows 都按原始文件行计数；nrows=None 表示读到文件结束。
    分批读取限制峰值内存，但访问靠后的窗口仍会遍历前面的批次。
    只接受一个合约，以免相邻行属于不同商品、进而产生虚假的价格跳变。
    保留前五档价量，以及后续审计需要的时间、交易所序号和合约标识。"""
    if offset < 0 or (nrows is not None and nrows <= 0):
        raise ValueError('offset must be nonnegative and nrows must be positive')
    columns = ['ts_event', 'sequence', 'instrument_id', 'symbol'] + [
        f'{field}_{i:02d}' for i in range(5)
        for field in ['bid_px', 'ask_px', 'bid_sz', 'ask_sz']]
    # seen 表示已经跨过的原始行数，remaining 表示尚需收集的行数。
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
        # 窗口起点可能落在某个批次中间，只截取该批次的有效后半段。
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
    # 排除缺失/无穷价量、买卖倒挂及没有双边有效数量的报价。
    valid = np.isfinite(df[numeric]).all(axis=1)
    valid &= (df.bid_px_00 > 0) & (df.ask_px_00 >= df.bid_px_00)
    valid &= (df[[c for c in numeric if '_sz_' in c]] >= 0).all(axis=1)
    valid &= (df.bid_sz_00 > 0) & (df.ask_sz_00 > 0)
    df = df.loc[valid].copy()
    df['ts_event'] = pd.to_datetime(df.ts_event, utc=True, errors='raise')
    # 时间相同则按交易所 sequence 排序；键也相同则保留原始相对次序。
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
    """将行情顺序划分为训练、奖励校准、参数验证、测试四段。

    例如下一段从第 b 行开始，最长标签看未来 90 行，那么前段只有满足
    i+90<b 的样本才能用于学习；否则标签已引用了下一段的价格。
    同一时间戳的边界也继续向前清除，确保标签到期时间严格早于下一段起点。
    测试段保留没有未来标签的尾部行情，因为它仍然需要执行交易和期末平仓。
    返回各段 DataFrame、标准化特征矩阵 X、目标 y 和仅从训练段估计的均值/标准差。"""
    n = len(df)
    bounds = [0] + [int(n * x) for x in np.cumsum(SPLIT_RATIOS)[:-1]] + [n]
    gap = max(HORIZONS)
    partitions = {}
    for i, name in enumerate(('train', 'calibration', 'validation', 'test')):
        start, stop = bounds[i], bounds[i + 1]
        if name != 'test':
            stop -= gap
            # 行号隔离后再核对时间戳：一批同时间事件也不能跨越标签到期边界。
            while stop > start and df.ts_event.iloc[stop - 1 + gap] >= df.ts_event.iloc[bounds[i + 1]]:
                stop -= 1
        if stop - start < 100:
            raise ValueError('Window too short for four purged partitions (need at least 2000 events)')
        partitions[name] = df.iloc[start:stop].copy()
    train = partitions['train']
    mean = train[feature_cols].mean()
    # 常量特征的标准差设为 1，标准化后为 0，避免除零；后面各段复用同一参数。
    std = train[feature_cols].std().replace(0., 1.).fillna(1.)
    output = dict(feature_cols=feature_cols, norm_mean=mean.to_dict(), norm_std=std.to_dict(),
                  purge_events=gap)
    for name, frame in partitions.items():
        output[f'{name}_df'] = frame
        output[f'X_{name}'] = ((frame[feature_cols] - mean) / std).to_numpy(dtype=float)
        output[f'y_{name}'] = frame.target_ret.to_numpy()
    return output
