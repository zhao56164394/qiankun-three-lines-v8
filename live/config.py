# -*- coding: utf-8 -*-
"""
乾坤三线 v8.0 — 实盘配置

共用策略参数从 backtest_bt.config 导入。
此文件只定义实盘特有的参数（路径、交易时间、QMT连接等）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 共用策略参数 — 统一来源
# ============================================================
from backtest_bt.config import (  # noqa: E402
    POOL_THRESHOLD, FILTER_TREND_AT_BUY_MAX, FILTER_RETAIL_RECOVERY_MAX,
    MAX_POSITIONS, DAILY_BUY_LIMIT, POSITION_MODE,
    SKIP_HEXAGRAMS, INNER_SELL_METHOD,
    CRAZY_TREND_THRESHOLD, CRAZY_MF_THRESHOLD,
    STALL_DAYS, TRAIL_PCT, TREND_CAP,
    TREND_FORCE_SELL, TREND_HIGH_ZONE, TREND_MID_ZONE, TREND_BUY_ABOVE,
)

# 等级过滤 (仅实盘使用，回测已废弃)
CRAZY_ALLOWED = {'A+', 'A', 'B+', 'B', 'B-', 'D'}
NORMAL_ALLOWED = {'A+'}

MIN_512_SAMPLES = 3

# 兼容旧变量名
CRAZY_ALLOWED_GRADES = CRAZY_ALLOWED
NORMAL_ALLOWED_GRADES = NORMAL_ALLOWED
TREND_TRIGGER = TREND_BUY_ABOVE
TREND_NO_SELL_BELOW = TREND_MID_ZONE
TREND_CROSS_89 = TREND_HIGH_ZONE
TREND_WAVE_END = TREND_FORCE_SELL

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LAYER_DIR = os.path.join(PROJECT_ROOT, 'data_layer')
DATA_DIR = os.path.join(DATA_LAYER_DIR, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'live', 'logs')
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, 'live', 'snapshots')

# ============================================================
# 实盘特有参数
# ============================================================
MIN_BUY_AMOUNT = 1000       # 单笔最小买入金额(元)

# 牛卖参数
BULL_SELL_CROSS89_COUNT = 2  # 牛卖: 穿89几次后卖出

# ============================================================
# 卖出时间控制
# ============================================================
SELL_REALTIME_CHECK = '145500'  # 14:55 实时三线卖出检查
SELL_CHECK_START = '145600'
SELL_FORCE_TIME = '145500'

# ============================================================
# 交易时间控制
# ============================================================
SELECT_TIME = '091500'       # 09:15 盘前选股
BUY_START_TIME = '093100'    # 开盘后1分钟开始买入
BUY_END_TIME = '100000'      # 10:00 之后不再买入
SELL_START_TIME = '093000'
SELL_END_TIME = '150000'
DATA_UPDATE_TIME = '150500'  # 15:05 收盘后更新数据

# ============================================================
# miniQMT 连接参数
# ============================================================
QMT_ACCOUNT = '8885686909'
QMT_ACCOUNT_TYPE = 'STOCK'
QMT_PATH = r'D:\国金证券QMT交易端\userdata_mini'
ORDER_TIMEOUT = 180
USE_LIMIT_PRICE = False

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_MAX_DAYS = 90
