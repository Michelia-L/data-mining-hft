"""
全局配置模块 (config.py)
定义高频订单簿数据路径、采样规模、特征参数、交易摩擦与强化学习超参数。
"""

import os

# 项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集文件路径
DATASET_PATHS = {
    # CME Globex 标普500 E-mini 期货 (ESZ5) MBP-10 数据
    "CME_ES": os.path.join(BASE_DIR, "databento_glbx.mdp3_mbp_10.csv"),
    # ICE Futures Europe 布伦特原油期货 (BRN) MBP-10 数据
    "ICE_BRENT": os.path.join(BASE_DIR, "databento_ifeu.impact_mbp_10.csv"),
}

# 实验采样设置 (兼顾代表性、内存安全与执行速度)
# 从连续活跃交易时段抽取样本行数
SAMPLE_TICKS = 60000

# 训练集与测试集划分比例 (前70%训练/校准，后30%时序外外推测试)
TRAIN_RATIO = 0.70

# 论文核心：多尺度时域时钟周期 (Ticks)
# 对应论文中的多尺度期望步长：短周期 10 ticks, 中周期 30 ticks, 长周期 90 ticks
HORIZONS = [10, 30, 90]

# 品种合约规格与交易摩擦参数
INSTRUMENT_CONFIG = {
    "CME_ES": {
        "name": "CME E-mini S&P 500 Futures (ES)",
        "tick_size": 0.25,          # 最小跳动单位
        "multiplier": 50.0,         # 合约乘数 (点值)
        "commission_per_order": 1.25, # 单边手续费 (美元)
        "slippage_ticks": 0.5,      # 平均滑点 (跳)
        "trade_threshold": 0.000015, # 适应高点位标普期货的开仓相对收益阈值 (~0.4 tick)
    },
    "ICE_BRENT": {
        "name": "ICE Brent Crude Oil Futures (BRN)",
        "tick_size": 0.01,          # 最小跳动单位
        "multiplier": 1000.0,       # 合约乘数 (点值)
        "commission_per_order": 1.50, # 单边手续费 (美元)
        "slippage_ticks": 0.5,      # 平均滑点 (跳)
        "trade_threshold": 0.00006,  # 适应原油期货的开仓相对收益阈值 (~0.4 tick)
    }
}

# 在线模型选择强化学习超参数
RL_CONFIG = {
    "ucb_c": 0.8,               # UCB 算法置信上限探索常数 C
    "ars_window": 100,          # ARS (平均奖励选择) 算法的滑动评估窗口长度 (ticks)
    "trade_threshold": 0.00008, # 模型预测收益率开仓阈值 (高于此阈值发出买/卖信号)
    "holding_period": 30,       # 默认持仓周期 (ticks)
}

# 结果保存路径
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR, "experiment_summary.json")
