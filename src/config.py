"""课程版的项目位置与固定美元账本单位；时间与模型参数见唯一实验配置。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INITIAL_CAPITAL = 100000.0
QUANTITY = 1
# ES 每张价格变动1点=50美元；最小报价跳动0.25点。
# 单边手续费1.25美元，在买卖一外增加0.5跳不利滑点；均为项目执行假设。
INSTRUMENT_CONFIG = {
    'CME_ES': dict(name='CME E-mini S&P 500 Futures (ES)', tick_size=0.25,
                   multiplier=50.0, commission_per_order=1.25,
                   slippage_ticks=0.5, trade_threshold=0.000015),
}
