"""
高频数据加载与微观结构特征工程模块 (data_loader.py)
负责从 MBP-10 高频订单簿数据中提取价格、订单流不平衡 (OBI)、深度加权微观价格与多尺度时序特征。
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from src.config import SAMPLE_TICKS, TRAIN_RATIO, HORIZONS


def load_and_preprocess_ticks(file_path: str, nrows: int = SAMPLE_TICKS) -> pd.DataFrame:
    """
    加载 MBP-10 订单簿数据并执行初步清洗。

    参数:
        file_path (str): CSV 文件路径
        nrows (int): 加载的连续事件行数

    返回:
        pd.DataFrame: 包含基础行情字段的数据框
    """
    print(f"[DataLoader] 正在读取数据: {file_path} (采样前 {nrows} 行)...")
    
    # 仅加载核心所需字段，节省内存与加载时间
    usecols = [
        'ts_event', 'action', 'side', 'price', 'size',
        'bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00',
        'bid_px_01', 'ask_px_01', 'bid_sz_01', 'ask_sz_01',
        'bid_px_02', 'ask_px_02', 'bid_sz_02', 'ask_sz_02',
        'bid_px_03', 'ask_px_03', 'bid_sz_03', 'ask_sz_03',
        'bid_px_04', 'ask_px_04', 'bid_sz_04', 'ask_sz_04'
    ]
    
    df = pd.read_csv(file_path, nrows=nrows, usecols=usecols)
    
    # 过滤掉买一价或卖一价缺失、倒挂的异常事件
    valid_mask = (
        (df['bid_px_00'] > 0) & 
        (df['ask_px_00'] > 0) & 
        (df['ask_px_00'] >= df['bid_px_00'])
    )
    df = df[valid_mask].copy()
    
    # 解析时间戳并建立时序索引
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df.sort_values('ts_event', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[DataLoader] 数据读取完成，有效事件样本数: {len(df)}")
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
    df['target_dir'] = np.sign(df['target_ret']).fillna(0.0)
    
    # 清理由于 rolling 和 shift 产生的首尾空值
    df.dropna(subset=features + [f'future_ret_{h}' for h in HORIZONS], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"[DataLoader] 特征工程构建完成，有效特征数: {len(features)}, 样本数: {len(df)}")
    return df, features


def prepare_train_test_split(df: pd.DataFrame, feature_cols: List[str]) -> Dict:
    """
    按时间序列严格切分训练集与测试集 (禁止时序信息泄露)。

    参数:
        df (pd.DataFrame): 包含特征与目标的数据框
        feature_cols (List[str]): 特征列名列表

    返回:
        Dict: 包含训练集与测试集特征矩阵与时序元数据的字典
    """
    n = len(df)
    train_size = int(n * TRAIN_RATIO)
    
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    # 特征均值与方差标准化 (严格使用训练集的均值与方差，避免前视偏差)
    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std().replace(0.0, 1.0)
    
    X_train = ((train_df[feature_cols] - mean) / std).values
    X_test = ((test_df[feature_cols] - mean) / std).values
    
    y_train = train_df['target_ret'].values
    y_test = test_df['target_ret'].values
    
    print(f"[DataLoader] 时序划分完成: 训练集样本数={len(train_df)}, 测试集样本数={len(test_df)}")
    
    return {
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "norm_mean": mean.to_dict(),
        "norm_std": std.to_dict(),
    }
