"""集中定义数据位置、时间尺度、交易单位和实验默认参数。

阅读顺序建议：本文件 → data_loader → model_library → irl_reward →
model_selector → execution_engine，最后看 run_experiments 的完整流程。
这里的事件步是一条排序后的订单簿行情，不是固定秒数；
tick_size 则是最小报价跳动，两者含义不同。金额统一按美元记账。"""
from pathlib import Path

# 用文件自身的位置定位项目，避免从不同工作目录启动时找不到数据。
BASE_DIR = Path(__file__).resolve().parent.parent
# 优先选已存在的 Parquet，其次 CSV；都不存在时保留预期路径，由入口明确报错。
DATASET_PATHS = {
    key: str(next((BASE_DIR / f'{stem}{ext}' for ext in ('.parquet', '.csv')
                   if (BASE_DIR / f'{stem}{ext}').exists()), BASE_DIR / f'{stem}.parquet'))
    for key, stem in {
        'CME_ES': 'databento_glbx.mdp3_mbp_10',
        'ICE_BRENT': 'databento_ifeu.impact_mbp_10',
    }.items()
}
# 根种子用于随机策略；三个随机基线分别使用 seed、seed+1、seed+2。
SEED = 42
# 每个独立实验窗口先读取的原始行数；清洗后有效事件可能略少。
SAMPLE_EVENTS = 60000
# 多尺度收益标签与奖励的前瞻事件数；在线反馈必须等待最长尺度到期。
HORIZON_EVENTS = [10, 30, 90]
# 顺序为训练、奖励校准、参数验证、测试；前三段还要扣除尾部隔离区。
SPLIT_RATIOS = (0.50, 0.15, 0.15, 0.20)
# 每个窗口重新以这笔资金开始；收益率 = 美元盈亏 / 初始资金。
INITIAL_CAPITAL = 100000.0
# 固定合约张数，不按账户净值复利调整，也未模拟保证金约束。
QUANTITY = 1
# multiplier：价格变动 1 点时每张合约的美元盈亏；commission_per_order：每张单边手续费。
# slippage_ticks：在买卖一报价之外增加的不利滑点；不要与盘口点差重复混淆。
# trade_threshold：预测相对收益的开仓门槛，是验证段候选门槛的基值。
INSTRUMENT_CONFIG = {
    'CME_ES': dict(name='CME E-mini S&P 500 Futures (ES)', tick_size=0.25,
                   multiplier=50.0, commission_per_order=1.25,
                   slippage_ticks=0.5, trade_threshold=0.000015),
    'ICE_BRENT': dict(name='ICE Brent Crude Oil Futures (BRN)', tick_size=0.01,
                      multiplier=1000.0, commission_per_order=1.50,
                      slippage_ticks=0.5, trade_threshold=0.00006),
}
# ucb_c 控制探索；ars_window_events 按奖励到达的事件步计时；holding_period 是两次持仓复核的最短事件间隔。
# latency_events=1 表示本步决策下一条行情执行，不代表固定一秒网络延迟。
RL_CONFIG = dict(ucb_c=0.1, ars_window_events=300, holding_period=30, latency_events=1)
# 只在验证段比较这些探索系数，不能根据测试净收益回头选择。
UCB_CANDIDATES = (0.01, 0.1, 0.8)
RESULTS_DIR = str(BASE_DIR / 'results')
SUMMARY_JSON_PATH = str(BASE_DIR / 'results' / 'experiment_summary.json')
