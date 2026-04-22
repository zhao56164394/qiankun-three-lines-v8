# -*- coding: utf-8 -*-
"""
乾坤三线 v1.0 — 信号引擎

从回测代码(backtest_capital.py)抽取核心买卖信号逻辑，
供实盘系统调用。

核心职责:
  1. 管理候选池 (散户线 < -400 入池)
  2. 生成买入信号 (首次双升 + 趋势>11)
  3. 判断卖出信号 (日线级别: <50不卖 / 50-89双降 / 穿89卖)
  4. 确定卖法 (牛卖 vs 熊卖)
"""
import os
import sys
import numpy as np
import pandas as pd

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.config import (
    POOL_THRESHOLD, TREND_TRIGGER, INNER_SELL_METHOD,
    TREND_NO_SELL_BELOW, TREND_CROSS_89, TREND_WAVE_END,
    BULL_SELL_CROSS89_COUNT, DATA_DIR,
    CRAZY_TREND_THRESHOLD, CRAZY_MF_THRESHOLD,
    STALL_DAYS, TRAIL_PCT, TREND_CAP,
    MIN_512_SAMPLES,
    FILTER_TREND_AT_BUY_MAX, FILTER_RETAIL_RECOVERY_MAX,
)
from bagua_engine import encode_yao
from strategy.indicator import calc_trend_line, calc_retail_line


# ============================================================
# 候选池管理
# ============================================================
class StockPool:
    """
    个股候选池 — 追踪散户线 < -400 的股票

    状态机:
      未入池 → 散户线 < -400 → 入池(记录池底散户线)
      入池中 → 首次双升(趋势↑散户↑, 趋势>11) → 买入信号 → 进入波段
      波段中 → 趋势线 < 11 → 波段结束 → 重置
    """

    def __init__(self):
        # 状态: {code: {pooled, pool_retail, in_wave, buy_date, sell_method}}
        self.states = {}

    def get_state(self, code):
        """获取或初始化个股状态"""
        if code not in self.states:
            self.states[code] = {
                'pooled': False,
                'pool_retail': 0,
                'in_wave': False,
                'buy_date': None,
                'sell_method': None,
            }
        return self.states[code]

    def update(self, code, trend_today, trend_yest, retail_today, retail_yest):
        """
        更新单只股票状态，返回买入信号(如有)

        Args:
            code: 股票代码
            trend_today/yest: 今日/昨日趋势线
            retail_today/yest: 今日/昨日散户线

        Returns:
            dict or None: 买入信号 {code, pool_retail} 或 None
        """
        state = self.get_state(code)

        # 波段中: 检查是否波段结束
        if state['in_wave']:
            if not np.isnan(trend_today) and trend_today < TREND_WAVE_END:
                state['in_wave'] = False
                state['pooled'] = False
                state['pool_retail'] = 0
                state['buy_date'] = None
                state['sell_method'] = None
            return None

        # 未入池: 检查是否应该入池
        if not state['pooled']:
            if not np.isnan(retail_today) and retail_today < POOL_THRESHOLD:
                state['pooled'] = True
                state['pool_retail'] = retail_today
            # 入池当天也要更新pool_retail但不产生信号
            if state['pooled'] and not np.isnan(retail_today):
                state['pool_retail'] = min(state['pool_retail'], retail_today)
            return None

        # 已入池: 更新池底散户线
        if not np.isnan(retail_today):
            state['pool_retail'] = min(state['pool_retail'], retail_today)

        # 检查数据有效性
        if (np.isnan(trend_today) or np.isnan(trend_yest) or
                np.isnan(retail_today) or np.isnan(retail_yest)):
            return None

        # 首次双升 + 趋势>11 → 买入信号
        if (retail_today > retail_yest and
                trend_today > trend_yest and
                trend_today > TREND_TRIGGER):

            # ---- v1.1 过滤 (对齐回测 backtest_capital.py) ----
            skip = False

            # 过滤1: 买点趋势线过高 → 跳过 (已涨太多)
            if (FILTER_TREND_AT_BUY_MAX is not None and
                    FILTER_TREND_AT_BUY_MAX > 0 and
                    trend_today > FILTER_TREND_AT_BUY_MAX):
                skip = True

            # 过滤2: 散户线回升幅度过大 → 跳过 (底部已过)
            if (FILTER_RETAIL_RECOVERY_MAX is not None and
                    FILTER_RETAIL_RECOVERY_MAX > 0):
                recovery = retail_today - state['pool_retail']
                if recovery > FILTER_RETAIL_RECOVERY_MAX:
                    skip = True

            if skip:
                state['in_wave'] = True  # 与回测一致: 跳过后进入波段
                return None

            signal = {
                'code': code,
                'pool_retail': state['pool_retail'],
            }
            state['in_wave'] = True
            return signal

        return None

    def mark_bought(self, code, buy_date, sell_method):
        """标记已买入"""
        state = self.get_state(code)
        state['buy_date'] = buy_date
        state['sell_method'] = sell_method

    def reset(self):
        """完全重置池"""
        self.states.clear()

    def get_pool_summary(self):
        """获取候选池摘要"""
        pooled = [c for c, s in self.states.items() if s['pooled'] and not s['in_wave']]
        in_wave = [c for c, s in self.states.items() if s['in_wave']]
        return {
            'pooled_count': len(pooled),
            'in_wave_count': len(in_wave),
            'pooled_stocks': pooled,
        }


# ============================================================
# 买入信号生成
# ============================================================
def generate_buy_signals(stock_pool, stock_data_today, stock_data_yest):
    """
    扫描全市场，生成今日买入信号

    Args:
        stock_pool: StockPool 实例
        stock_data_today: dict {code: {trend, retail, open, close, ...}}
        stock_data_yest: dict {code: {trend, retail, open, close, ...}}

    Returns:
        list: 买入信号列表，按 pool_retail 从低到高排序
              [{code, pool_retail}, ...]
    """
    signals = []

    for code in stock_data_today:
        today = stock_data_today[code]
        yest = stock_data_yest.get(code)
        if yest is None:
            continue

        trend_today = today.get('trend', np.nan)
        trend_yest = yest.get('trend', np.nan)
        retail_today = today.get('retail', np.nan)
        retail_yest = yest.get('retail', np.nan)

        signal = stock_pool.update(code, trend_today, trend_yest,
                                   retail_today, retail_yest)
        if signal is not None:
            signal['open'] = today.get('open', np.nan)
            signal['close'] = today.get('close', np.nan)
            signal['year_gua'] = today.get('year_gua', '')
            signal['month_gua'] = today.get('month_gua', '')
            signal['day_gua'] = today.get('day_gua', '')
            signals.append(signal)

    # 按池底散户线排序 (越低越优先)
    signals.sort(key=lambda x: x['pool_retail'])
    return signals


# ============================================================
# 卖出信号判断 (日线级别)
# ============================================================
class SellTracker:
    """
    卖出条件追踪器 — 每只持仓股一个实例

    追踪:
      - 趋势线历史最高值 (running_max)
      - 是否曾达到89 (reached_89)
      - 穿89次数 (cross_89_count)
    """

    def __init__(self, code, sell_method='bear'):
        self.code = code
        self.sell_method = sell_method
        self.running_max = 0      # 持仓期间趋势线最高值
        self.reached_89 = False   # 是否曾到达89
        self.cross_89_count = 0   # 下穿89次数
        self.prev_trend = None    # 前一日趋势线
        self.prev_retail = None   # 前一日散户线

    def check_sell(self, trend, retail, trend_yest=None, retail_yest=None):
        """
        检查是否应该卖出

        判断顺序 (对齐修正后逻辑):
          0. trend < 11 → 波段结束，强制卖出
          1. running_max < 50 → 不卖
          2. 50 ≤ running_max < 89 → 双降(trend↓ 且 retail↓)卖出
          3. running_max ≥ 89 → 先查双降，再查下穿89

        Args:
            trend: 当日趋势线
            retail: 当日散户线
            trend_yest: 昨日趋势线 (如果为None，用self.prev_trend)
            retail_yest: 昨日散户线 (如果为None，用self.prev_retail)

        Returns:
            dict or None: 卖出信号 或 None(继续持有)
        """
        if np.isnan(trend) or np.isnan(retail):
            return None

        # 用前一日数据
        prev_t = trend_yest if trend_yest is not None else self.prev_trend
        prev_r = retail_yest if retail_yest is not None else self.prev_retail

        # 更新追踪状态
        self.running_max = max(self.running_max, trend)

        # === 规则0: 趋势线 < 11 → 波段结束，强制清仓 ===
        if trend < TREND_WAVE_END:
            self._update_prev(trend, retail)
            return self._sell_signal('波段结束(<11)', trend, retail)

        # === 需要前日数据才能判断 ===
        if prev_t is None or prev_r is None:
            self._update_prev(trend, retail)
            return None

        if np.isnan(prev_t) or np.isnan(prev_r):
            self._update_prev(trend, retail)
            return None

        # === 规则1: 趋势线从未到过50 → 不卖 ===
        if self.running_max < TREND_NO_SELL_BELOW:
            self._update_prev(trend, retail)
            return None

        # === 规则2: 先判断双降 (50~89区间) ===
        # 当 running_max ≥ 50 且趋势线当前在50~89之间，双降卖出
        if (TREND_NO_SELL_BELOW <= trend < TREND_CROSS_89 and
                trend < prev_t and retail < prev_r):
            self._update_prev(trend, retail)
            return self._sell_signal('双降(50-89)', trend, retail)

        # === 规则3: 再判断下穿89 ===
        if self.running_max >= TREND_CROSS_89:
            self.reached_89 = True
            # 检查下穿89
            if trend < TREND_CROSS_89 and prev_t >= TREND_CROSS_89:
                self.cross_89_count += 1

                if self.sell_method == 'bear':
                    # 熊卖: 首次下穿89即卖
                    self._update_prev(trend, retail)
                    return self._sell_signal('首穿89(熊卖)', trend, retail)
                else:
                    # 牛卖: 第N次下穿89才卖
                    if self.cross_89_count >= BULL_SELL_CROSS89_COUNT:
                        self._update_prev(trend, retail)
                        return self._sell_signal(
                            f'第{self.cross_89_count}次穿89(牛卖)', trend, retail)

        self._update_prev(trend, retail)
        return None

    def _sell_signal(self, reason, trend, retail):
        return {
            'action': 'sell',
            'reason': reason,
            'code': self.code,
            'trend': trend,
            'retail': retail,
            'running_max': self.running_max,
            'cross_89_count': self.cross_89_count,
        }

    def _update_prev(self, trend, retail):
        self.prev_trend = trend
        self.prev_retail = retail


# ============================================================
# 停滞止损追踪器 (疯狂模式)
# ============================================================
class StallSellTracker:
    """
    疯狂模式卖出追踪器 — 停滞止损

    卖出条件 (任一触发):
      1. 价格从高点回撤 >= trail_pct% → 卖出
      2. trend连续stall_days天不创新高 且 trend峰值 < trend_cap → 卖出
      3. trend < 11 → 强制退出
    """

    def __init__(self, code, buy_price,
                 stall_days=STALL_DAYS, trail_pct=TRAIL_PCT, trend_cap=TREND_CAP):
        self.code = code
        self.sell_method = 'stall'
        self.buy_price = buy_price
        self.stall_days = stall_days
        self.trail_pct = trail_pct
        self.trend_cap = trend_cap

        self.price_peak = buy_price     # 持仓期间价格最高值
        self.trend_peak = 0             # 持仓期间趋势线最高值
        self.stall_count = 0            # trend连续未创新高天数

    def check_sell(self, trend, retail, close_price,
                   trend_yest=None, retail_yest=None):
        """
        检查是否应该卖出

        Args:
            trend: 当日趋势线
            retail: 当日散户线
            close_price: 当日收盘价
            trend_yest: 昨日趋势线 (未使用，保持接口一致)
            retail_yest: 昨日散户线 (未使用，保持接口一致)

        Returns:
            dict or None: 卖出信号 或 None
        """
        if np.isnan(trend) or np.isnan(close_price):
            return None

        # 规则3: trend < 11 → 强制退出
        if trend < TREND_WAVE_END:
            return self._sell_signal('波段结束(<11)', trend, retail)

        # 更新价格峰值
        self.price_peak = max(self.price_peak, close_price)

        # 规则1: 价格回撤止损
        drawdown = (self.price_peak - close_price) / self.price_peak * 100
        if drawdown >= self.trail_pct:
            return self._sell_signal(
                f'回撤止损({drawdown:.1f}%>={self.trail_pct}%)', trend, retail)

        # 规则2: trend停滞
        if not np.isnan(trend):
            if trend > self.trend_peak:
                self.trend_peak = trend
                self.stall_count = 0
            else:
                self.stall_count += 1
                if self.stall_count >= self.stall_days and self.trend_peak < self.trend_cap:
                    return self._sell_signal(
                        f'停滞卖出(stall{self.stall_count}天,peak{self.trend_peak:.0f}<{self.trend_cap})',
                        trend, retail)

        return None

    def _sell_signal(self, reason, trend, retail):
        return {
            'action': 'sell',
            'reason': reason,
            'code': self.code,
            'trend': trend,
            'retail': retail,
            'running_max': self.trend_peak,
            'price_peak': self.price_peak,
            'cross_89_count': 0,
        }


# ============================================================
# 六爻卦编码 (用于空仓过滤)
# ============================================================
def calc_hexagram(zz1000_entry):
    """
    计算当日六爻卦 (大中小象卦体系)

    内卦(小象卦): 日线趋势线≥50 + 5日变化>0 + chg1_chg1>0
    外卦(大象卦): 直接从预计算数据读取 (基于月线)

    Args:
        zz1000_entry: dict, 包含 trend, chg5, accel, year_gua 等

    Returns:
        dict: {inner, outer, hex_code} 或 None
    """
    # 内卦: 小象卦 (日线级别)
    inner = encode_yao(
        zz1000_entry.get('trend'),
        zz1000_entry.get('chg5'),
        zz1000_entry.get('accel')
    )

    # 外卦: 直接使用预计算的年卦 (基于月线数据)
    outer = zz1000_entry.get('year_gua', '')
    if outer and len(outer) == 3 and outer != '000' or outer == '000':
        # year_gua 已经是三位二进制编码
        pass
    else:
        outer = None

    if inner and outer:
        return {
            'inner': inner,
            'outer': outer,
            'hex_code': inner + outer,
        }
    return None


def determine_sell_method(inner_code):
    """
    根据内卦确定卖法 (常规模式)

    Args:
        inner_code: 三位二进制字符串 (如 '101')

    Returns:
        str: 'bull' 或 'bear'
    """
    return INNER_SELL_METHOD.get(inner_code, 'bear')


# ============================================================
# 模式判断
# ============================================================
def is_crazy_mode(zz1000_entry):
    """
    判断当前是否疯狂模式

    条件: 中证1000 trend < 45 且 main_force > 0

    Args:
        zz1000_entry: dict, 包含 trend, main_force

    Returns:
        bool
    """
    trend = zz1000_entry.get('trend')
    mf = zz1000_entry.get('main_force')
    if trend is None or mf is None:
        return False
    try:
        return float(trend) < CRAZY_TREND_THRESHOLD and float(mf) > CRAZY_MF_THRESHOLD
    except (ValueError, TypeError):
        return False


# ============================================================
# 512卦象分级
# ============================================================
def to_yinyang(code):
    """三位卦码→阴/阳"""
    return '阳' if str(code).zfill(3) in ['111', '011', '101'] else '阴'


def load_stock_events_df():
    """加载个股段首事件表"""
    path = os.path.join(DATA_DIR, 'stock_seg_events.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['event_date'] = df['event_date'].astype(str)
    df['avail_date'] = df['avail_date'].astype(str)
    for col in ['year_gua', 'month_gua', 'day_gua']:
        df[col] = df[col].astype(str).str.zfill(3)
    return df


def build_512_snapshot(stock_events_df, as_of_date):
    """
    构建截至as_of_date的512卦象超额收益快照

    Args:
        stock_events_df: 事件表DataFrame
        as_of_date: 截止日期字符串 'YYYY-MM-DD'

    Returns:
        dict: {combo_key: mean_excess_ret}
    """
    if stock_events_df is None:
        return {}

    # 只取avail_date <= as_of_date的事件
    valid = stock_events_df[stock_events_df['avail_date'] <= as_of_date]
    if len(valid) == 0:
        return {}

    combos = (valid['year_gua'] + '_' + valid['month_gua'] + '_' + valid['day_gua']).values
    excess_rets = valid['excess_ret'].values

    combo_rets = {}
    for c, r in zip(combos, excess_rets):
        if not np.isnan(r):
            combo_rets.setdefault(c, []).append(r)

    snap = {}
    for c, rets in combo_rets.items():
        if len(rets) >= MIN_512_SAMPLES:
            snap[c] = np.mean(rets)
    return snap


def grade_signal(year_yy, combo_pred):
    """
    信号分级 (与backtest_capital.py完全一致)

    Args:
        year_yy: '阴' 或 '阳'
        combo_pred: 512卦象超额收益预测值

    Returns:
        (grade, description): 如 ('A+', '年阴+512强超额(>3%)')
    """
    if year_yy == '阳':
        if combo_pred is not None and not (isinstance(combo_pred, float) and np.isnan(combo_pred)):
            if combo_pred > 3:
                return 'C', '年阳但512超额>3%'
        return 'F', '年阳禁止'
    if combo_pred is None or (isinstance(combo_pred, float) and pd.isna(combo_pred)):
        return 'B', '年阴+无512历史'
    if combo_pred > 3:
        return 'A+', '年阴+512强超额(>3%)'
    elif combo_pred > 1:
        return 'A', '年阴+512超额(1~3%)'
    elif combo_pred > 0:
        return 'B+', '年阴+512微超额(0~1%)'
    elif combo_pred > -2:
        return 'B-', '年阴+512微跑输'
    else:
        return 'D', '年阴+512强跑输(<-2%)'


# ============================================================
# 日线数据加载 (实盘用，从预计算CSV读取)
# ============================================================
def load_stock_daily(code):
    """
    加载单只个股的日线数据(含预计算三线指标)

    Args:
        code: 股票代码如 '000001'

    Returns:
        pd.DataFrame 或 None
    """
    fpath = os.path.join(DATA_DIR, 'stocks', f'{code}.csv')
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath, encoding='utf-8-sig')
        return df
    except Exception:
        return None


def load_all_stock_latest(n_days=2):
    """
    加载所有个股最近N天的数据

    Args:
        n_days: 读取最近几天的数据

    Returns:
        dict: {code: DataFrame(最近N天)}
    """
    stocks_dir = os.path.join(DATA_DIR, 'stocks')
    if not os.path.exists(stocks_dir):
        return {}

    result = {}
    for fname in os.listdir(stocks_dir):
        if not fname.endswith('.csv'):
            continue
        code = fname.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(stocks_dir, fname),
                             encoding='utf-8-sig',
                             usecols=['date', 'open', 'close', 'trend', 'retail',
                                      'year_gua', 'month_gua', 'day_gua'])
            if len(df) >= n_days:
                result[code] = df.tail(n_days).reset_index(drop=True)
        except Exception:
            continue

    return result


def get_stock_data_by_date(all_data, date_str):
    """
    从加载的数据中提取指定日期的数据

    Args:
        all_data: load_all_stock_latest() 返回值
        date_str: 日期字符串如 '2026-04-03'

    Returns:
        dict: {code: {trend, retail, open, close}}
    """
    result = {}
    for code, df in all_data.items():
        match = df[df['date'] == date_str]
        if len(match) > 0:
            row = match.iloc[-1]
            result[code] = {
                'trend': row['trend'],
                'retail': row['retail'],
                'open': row['open'],
                'close': row['close'],
                'year_gua': str(row.get('year_gua', '')).zfill(3) if pd.notna(row.get('year_gua')) else '',
                'month_gua': str(row.get('month_gua', '')).zfill(3) if pd.notna(row.get('month_gua')) else '',
                'day_gua': str(row.get('day_gua', '')).zfill(3) if pd.notna(row.get('day_gua')) else '',
            }
    return result


# ============================================================
# 中证1000卦象数据 (用于六爻卦)
# ============================================================
def load_zz1000_latest():
    """
    加载中证1000最新数据，计算六爻卦所需指标

    大中小象卦体系:
      内卦(小象卦): 日线趋势线≥50 + 5日变化>0 + chg1_chg1>0
      外卦(大象卦): 直接从CSV的year_gua列读取 (月线计算，已预存)

    Returns:
        dict: {date_str: {trend, chg5, accel, main_force, year_gua, month_gua}}
    """
    zz_path = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    if not os.path.exists(zz_path):
        return {}

    df = pd.read_csv(zz_path, encoding='utf-8-sig')
    n = len(df)
    trend = df['trend'].values.astype(float)

    # 速度: 5日变化 — 用于小象卦
    chg5 = np.full(n, np.nan)
    for i in range(5, n):
        if not np.isnan(trend[i]) and not np.isnan(trend[i-5]):
            chg5[i] = trend[i] - trend[i-5]

    # 加速度: chg1的1日变化 — 用于小象卦
    chg1 = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(trend[i]) and not np.isnan(trend[i-1]):
            chg1[i] = trend[i] - trend[i-1]
    accel = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(chg1[i]) and not np.isnan(chg1[i-1]):
            accel[i] = chg1[i] - chg1[i-1]

    # 只返回最近30天 (实盘不需要全部历史)
    result = {}
    start = max(0, n - 30)
    for i in range(start, n):
        dt = str(df.loc[i, 'date'])
        # 大象卦/中象卦直接从CSV读取 (已由prepare/update脚本预计算)
        yg = str(df.loc[i, 'year_gua']).strip() if pd.notna(df.loc[i, 'year_gua']) else ''
        mg = str(df.loc[i, 'month_gua']).strip() if pd.notna(df.loc[i, 'month_gua']) else ''
        result[dt] = {
            'trend': trend[i] if not np.isnan(trend[i]) else None,
            'chg5': chg5[i] if not np.isnan(chg5[i]) else None,
            'accel': accel[i] if not np.isnan(accel[i]) else None,
            'main_force': df.loc[i, 'main_force'] if not pd.isna(df.loc[i, 'main_force']) else None,
            'year_gua': yg,
            'month_gua': mg,
        }

    return result


# ============================================================
# 实时三线计算 (盘中用实时价近似收盘价)
# ============================================================
def calc_realtime_indicators(code, realtime_price,
                             today_open=None, today_high=None, today_low=None):
    """
    用实时价格拼接历史日线，计算今日的趋势线和散户线

    Args:
        code: 股票代码
        realtime_price: 当前实时价格 (作为今日close)
        today_open: 今日开盘价 (可选，默认用realtime_price)
        today_high: 今日最高价 (可选，默认用max(open, realtime_price))
        today_low: 今日最低价 (可选，默认用min(open, realtime_price))

    Returns:
        dict: {trend_today, trend_yest, retail_today, retail_yest} 或 None
    """
    df = load_stock_daily(code)
    if df is None or len(df) < 60:
        return None

    # 取最近120天历史 (足够计算55日HHV/LLV + EMA预热)
    df = df.tail(120).copy().reset_index(drop=True)

    # 确保必要列存在
    for col in ['close', 'high', 'low']:
        if col not in df.columns:
            return None

    # 构造今日数据行
    _open = today_open if today_open else realtime_price
    _high = today_high if today_high else max(_open, realtime_price)
    _low = today_low if today_low else min(_open, realtime_price)

    today_row = pd.DataFrame([{
        'close': realtime_price,
        'high': _high,
        'low': _low,
    }])
    df_ext = pd.concat([df, today_row], ignore_index=True)

    closes = df_ext['close'].values.astype(float)
    highs = df_ext['high'].values.astype(float)
    lows = df_ext['low'].values.astype(float)

    trend_arr = calc_trend_line(closes, highs, lows)
    retail_arr = calc_retail_line(closes)

    n = len(trend_arr)
    if n < 2:
        return None

    trend_today = trend_arr[n - 1]
    trend_yest = trend_arr[n - 2]
    retail_today = retail_arr[n - 1]
    retail_yest = retail_arr[n - 2]

    if np.isnan(trend_today) or np.isnan(retail_today):
        return None

    return {
        'trend_today': float(trend_today),
        'trend_yest': float(trend_yest),
        'retail_today': float(retail_today),
        'retail_yest': float(retail_yest),
    }
