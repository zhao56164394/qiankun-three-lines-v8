# -*- coding: utf-8 -*-
"""
联合策略回测 — 疯狂模式 + 常规模式
20万初始资金，2015-2026

买入过滤 (v1.1):
  - 买点趋势线 ≤ 20 (过滤高位信号)
  - 散户线回升幅度 ≤ 500 (过滤回升过久的信号)

疯狂模式:
  触发: 中证1000 trend<45 且 main_force>0
  等级: S1全等级(A+/A/B+/B/B-/D)
  仓位: 5仓, 每日限买1笔
  卖法: 停滞止损(stall=15天, trail=15%, cap=30)

常规模式:
  等级: 仅A+
  仓位: 5仓, 每日限买1笔
  卖法: inner卦卖法(根据中证内卦选bear/bull)
"""
import sys
import io
import os
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer.foundation_data import load_macro_bagua, load_market_bagua

sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)


# ============================================================
# 常量配置
# ============================================================
INNER_SELL_METHOD = {
    '111': 'bear', '110': 'bear', '101': 'bull', '100': 'bull',
    '011': 'bear', '010': 'bull', '001': 'bull', '000': 'bull',
}
SKIP_HEXAGRAMS = {
    '101010', '001100', '010101', '111001', '111010', '000101',
    '100110', '011011', '110011', '101001', '110000', '011100',
    '001111', '100000', '100010', '111110', '101110', '110001',
}
POOL_THRESHOLD = -400
FILTER_TREND_AT_BUY_MAX = 20      # 买点趋势线不超过20
FILTER_RETAIL_RECOVERY_MAX = 500   # 散户线回升幅度不超过500
INIT_CAPITAL = 200000
YEAR_START = '2015-01-01'
YEAR_END = '2026-04-01'

MIN_512_SAMPLES = 3
BIG_CYCLE_ALLOWED_MACRO_GUAS = {'001', '011', '101', '111'}


def parse_macro_guas(value):
    """解析命令行传入的大周期白名单。"""
    if value is None:
        return set(BIG_CYCLE_ALLOWED_MACRO_GUAS)
    text = str(value).strip()
    if text == '':
        return set()
    parts = [p.strip() for p in text.replace('，', ',').split(',')]
    return {str(p).zfill(3) for p in parts if p.strip()}


def load_big_cycle_context():
    """加载大周期/市场卦上下文，按日期映射。"""
    macro_df = load_macro_bagua().copy()
    market_df = load_market_bagua().copy()

    macro_df['date'] = macro_df['date'].astype(str)
    market_df['date'] = market_df['date'].astype(str)
    macro_df['gua_code'] = macro_df['gua_code'].astype(str).str.zfill(3)
    market_df['gua_code'] = market_df['gua_code'].astype(str).str.zfill(3)

    macro_ctx = {
        row['date']: {
            'macro_gua': row['gua_code'],
            'macro_gua_name': row.get('gua_name', ''),
        }
        for _, row in macro_df[['date', 'gua_code', 'gua_name']].drop_duplicates('date').iterrows()
    }
    market_ctx = {
        row['date']: {
            'market_gua': row['gua_code'],
            'market_gua_name': row.get('gua_name', ''),
        }
        for _, row in market_df[['date', 'gua_code', 'gua_name']].drop_duplicates('date').iterrows()
    }

    all_dates = sorted(set(macro_ctx) | set(market_ctx))
    return {
        dt: {
            'macro_gua': macro_ctx.get(dt, {}).get('macro_gua', ''),
            'macro_gua_name': macro_ctx.get(dt, {}).get('macro_gua_name', ''),
            'market_gua': market_ctx.get(dt, {}).get('market_gua', ''),
            'market_gua_name': market_ctx.get(dt, {}).get('market_gua_name', ''),
        }
        for dt in all_dates
    }


def allow_macro_gua(macro_gua, allowed_macro_guas=None):
    allowed = allowed_macro_guas if allowed_macro_guas is not None else BIG_CYCLE_ALLOWED_MACRO_GUAS
    return bool(macro_gua) and macro_gua in allowed


def build_context_stats(trades):
    def _group_stats(group):
        if not group:
            return {'trade_count': 0, 'win_rate': 0, 'avg_ret': 0, 'profit': 0}
        profits = [t.get('profit', 0) for t in group]
        rets = [t.get('ret_pct', 0) for t in group]
        wins = sum(1 for p in profits if p > 0)
        return {
            'trade_count': len(group),
            'win_rate': wins / len(group) * 100,
            'avg_ret': float(np.mean(rets)) if rets else 0,
            'profit': float(np.sum(profits)) if profits else 0,
        }

    by_macro = {}
    by_macro_market = {}
    for t in trades:
        macro_gua = str(t.get('macro_gua', '') or '')
        market_gua = str(t.get('market_gua', '') or '')
        by_macro.setdefault(macro_gua, []).append(t)
        by_macro_market.setdefault(f'{macro_gua}|{market_gua}', []).append(t)

    return {
        'by_macro_gua': {k: _group_stats(v) for k, v in sorted(by_macro.items())},
        'by_macro_market_gua': {k: _group_stats(v) for k, v in sorted(by_macro_market.items())},
    }


def summarize_signal_context(sig_df):
    if sig_df is None or len(sig_df) == 0:
        return {
            'signal_count': 0,
            'macro_gua_counts': {},
            'market_gua_counts': {},
        }
    macro_counts = sig_df['macro_gua'].fillna('').astype(str).value_counts().sort_index().to_dict() if 'macro_gua' in sig_df.columns else {}
    market_counts = sig_df['market_gua'].fillna('').astype(str).value_counts().sort_index().to_dict() if 'market_gua' in sig_df.columns else {}
    return {
        'signal_count': int(len(sig_df)),
        'macro_gua_counts': {str(k): int(v) for k, v in macro_counts.items()},
        'market_gua_counts': {str(k): int(v) for k, v in market_counts.items()},
    }


def to_yinyang(code):
    return '阳' if str(code).zfill(3) in ['111', '011', '101', '110'] else '阴'


# ============================================================
# 数据加载（带pickle缓存）
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_layer/data')


def _cache_path(name):
    return os.path.join(DATA_DIR, f'_cache_{name}.pkl')


def _load_cached(name, source_files, build_fn):
    """通用缓存加载：如果pkl存在且比源文件新，直接读pkl；否则重建并保存。
    pickle反序列化失败（如numpy版本变更）时自动重建。"""
    pkl = _cache_path(name)
    if os.path.exists(pkl):
        pkl_mtime = os.path.getmtime(pkl)
        src_mtime = max(os.path.getmtime(f) for f in source_files if os.path.exists(f))
        if pkl_mtime > src_mtime:
            try:
                with open(pkl, 'rb') as f:
                    return pickle.load(f)
            except (ModuleNotFoundError, ImportError, pickle.UnpicklingError, Exception) as e:
                print(f"  ⚠ 缓存 {name} 反序列化失败({type(e).__name__}), 自动重建...")
                os.remove(pkl)
    data = build_fn()
    with open(pkl, 'wb') as f:
        pickle.dump(data, f, protocol=5)
    return data


def _fmt_gua(val):
    """格式化卦码：去小数点、补零到3位"""
    s = str(val).strip() if pd.notna(val) else ''
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(3) if s else ''


def _build_zz1000():
    path = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 统一日期格式
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    n = len(df)
    trend = df['trend'].values.astype(float)
    zz = {}
    for i in range(n):
        dt = df.loc[i, 'date']
        gua = _fmt_gua(df.loc[i, 'gua'])
        zz[dt] = {
            'trend': trend[i] if not np.isnan(trend[i]) else None,
            'main_force': df.loc[i, 'main_force'] if not pd.isna(df.loc[i, 'main_force']) else None,
            'gua': gua,
        }
    return zz


def load_zz1000():
    """加载中证1000（6字段dict, 用于卦象编码）"""
    src = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    return _load_cached('zz1000', [src], _build_zz1000)


def _build_zz1000_full():
    path = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 统一日期格式
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    df['gua'] = df['gua'].astype(str).str.zfill(3)
    df['trend_ma10'] = df['trend'].rolling(10).mean()
    df['gua_yy'] = df['gua'].apply(lambda x: to_yinyang(str(x)))
    return df


def load_zz1000_full():
    """加载中证1000全部字段（DataFrame, 用于疯狂模式触发判断）"""
    src = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    return _load_cached('zz1000_full', [src], _build_zz1000_full)


def _build_stocks():
    stock_dir = os.path.join(DATA_DIR, 'stocks')
    stock_data = {}
    for fname in os.listdir(stock_dir):
        if not fname.endswith('.csv'):
            continue
        code = fname.replace('.csv', '')
        fpath = os.path.join(stock_dir, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig',
                             usecols=['date', 'open', 'close', 'trend', 'retail',
                                      'gua'])
            # 统一日期格式为 YYYY-MM-DD (兼容 YYYYMMDD 和 YYYY-MM-DD 混合)
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            stock_data[code] = df
        except:
            continue
    return stock_data


def load_stocks():
    """加载个股数据"""
    stock_dir = os.path.join(DATA_DIR, 'stocks')
    pkl = _cache_path('stocks')
    # stocks目录特殊处理：检查目录下任一文件是否比pkl新
    if os.path.exists(pkl):
        pkl_mtime = os.path.getmtime(pkl)
        # 抽样检查前10个文件的mtime，有任何一个比pkl新就重建
        csvs = [os.path.join(stock_dir, f) for f in os.listdir(stock_dir)[:10]
                if f.endswith('.csv')]
        if csvs and all(os.path.getmtime(f) < pkl_mtime for f in csvs):
            with open(pkl, 'rb') as f:
                return pickle.load(f)
    data = _build_stocks()
    with open(pkl, 'wb') as f:
        pickle.dump(data, f, protocol=5)
    return data


def _build_stock_events():
    path = os.path.join(DATA_DIR, 'stock_seg_events.csv')
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['event_date'] = df['event_date'].astype(str)
    df['avail_date'] = df['avail_date'].astype(str)
    # 兼容新旧格式 + 处理浮点数gua值
    if 'gua' in df.columns:
        df['gua'] = df['gua'].astype(str).str.split('.').str[0].str.zfill(3)
    elif 'year_gua' in df.columns:
        df['gua'] = df['year_gua'].astype(str).str.split('.').str[0].str.zfill(3)
    # zz_gua 列
    if 'zz_gua' in df.columns:
        df['zz_gua'] = df['zz_gua'].astype(str).str.split('.').str[0].str.zfill(3)
    return df


def load_stock_events():
    """加载个股段首事件表"""
    src = os.path.join(DATA_DIR, 'stock_seg_events.csv')
    return _load_cached('stock_events', [src], _build_stock_events)


# ============================================================
# 512卦象分级
# ============================================================
def build_daily_512_snapshot(stock_events, signal_dates):
    """为每个信号日构建个股卦象快照（无未来函数）

    新系统: 使用个股 gua 单列作为 combo key
    """
    stock_events = stock_events.sort_values('avail_date').reset_index(drop=True)
    signal_dates = sorted(set(signal_dates))
    combos = stock_events['gua'].values
    avail_dates = stock_events['avail_date'].values
    excess_rets = stock_events['excess_ret'].values
    snapshots = {}
    evt_ptr = 0
    n_events = len(stock_events)
    combo_rets = {}
    for dt in signal_dates:
        while evt_ptr < n_events and avail_dates[evt_ptr] <= dt:
            c = combos[evt_ptr]
            r = excess_rets[evt_ptr]
            if not np.isnan(r):
                combo_rets.setdefault(c, []).append(r)
            evt_ptr += 1
        snap = {}
        for c, rets in combo_rets.items():
            if len(rets) >= MIN_512_SAMPLES:
                snap[c] = np.mean(rets)
        snapshots[dt] = snap
    return snapshots


def build_512_rolling_pred(sig_df, min_hist=3):
    """为信号DataFrame添加combo_pred列（512卦象预测）"""
    stock_events = load_stock_events()
    signal_dates = sig_df['signal_date'].tolist()
    snapshots = build_daily_512_snapshot(stock_events, signal_dates)
    combo_preds = []
    for _, row in sig_df.iterrows():
        snap = snapshots.get(row['signal_date'], {})
        combo = row.get('combo', '')
        combo_preds.append(snap.get(combo, np.nan))
    sig_df['combo_pred'] = combo_preds
    return sig_df


def grade_signal(gua_yy, combo_pred):
    """信号分级: 返回 (等级, 说明)"""
    if gua_yy == '阳':
        if not np.isnan(combo_pred) if isinstance(combo_pred, float) else combo_pred is not None:
            if combo_pred > 3:
                return 'C', '阳卦但超额>3%'
        return 'F', '阳卦禁止'
    if pd.isna(combo_pred):
        return 'B', '阴卦+无历史'
    if combo_pred > 3:
        return 'A+', '阴卦+强超额(>3%)'
    elif combo_pred > 1:
        return 'A', '阴卦+超额(1~3%)'
    elif combo_pred > 0:
        return 'B+', '阴卦+微超额(0~1%)'
    elif combo_pred > -2:
        return 'B-', '阴卦+微跑输'
    else:
        return 'D', '阴卦+强跑输(<-2%)'


# ============================================================
# 交叉表分级系统 (方案B)
# ============================================================
def build_cross_table_snapshot(stock_events, signal_dates, min_samples=5):
    """为每个信号日构建 (zz_gua, stk_gua) 交叉表快照（无未来函数）

    返回: {date: {(zz_gua, stk_gua): mean_excess_ret, ...}, ...}
    """
    stock_events = stock_events.sort_values('avail_date').reset_index(drop=True)
    signal_dates = sorted(set(signal_dates))

    stk_guas = stock_events['gua'].values
    stk_guas_zz = stock_events['zz_gua'].values
    avail_dates = stock_events['avail_date'].values
    excess_rets = stock_events['excess_ret'].values

    snapshots = {}
    evt_ptr = 0
    n_events = len(stock_events)
    # 累积每个 (zz_gua, stk_gua) 组合的收益列表
    cross_rets = {}

    for dt in signal_dates:
        while evt_ptr < n_events and avail_dates[evt_ptr] <= dt:
            zg = stk_guas_zz[evt_ptr]
            sg = stk_guas[evt_ptr]
            r = excess_rets[evt_ptr]
            if (pd.notna(r) and pd.notna(zg) and pd.notna(sg)
                    and str(zg) not in ('nan', '', 'None')
                    and str(sg) not in ('nan', '', 'None')):
                key = (str(zg), str(sg))
                cross_rets.setdefault(key, []).append(float(r))
            evt_ptr += 1
        snap = {}
        for key, rets in cross_rets.items():
            if len(rets) >= min_samples:
                snap[key] = np.mean(rets)
        snapshots[dt] = snap

    return snapshots


def build_cross_table_rolling_pred(sig_df, min_samples=5):
    """为信号DataFrame添加 cross_pred 列（交叉表预测）"""
    stock_events = load_stock_events()
    signal_dates = sig_df['signal_date'].tolist()
    snapshots = build_cross_table_snapshot(stock_events, signal_dates, min_samples)

    cross_preds = []
    for _, row in sig_df.iterrows():
        snap = snapshots.get(row['signal_date'], {})
        key = (str(row.get('zz_gua', '')), str(row.get('stk_gua', '')))
        cross_preds.append(snap.get(key, np.nan))
    sig_df['cross_pred'] = cross_preds
    return sig_df


def grade_cross_table(cross_pred):
    """交叉表分级: 基于 (zz_gua x stk_gua) 历史超额收益

    返回 (等级, 说明)
    等级体系:
      A+: 强超额 (>3%)
      A:  超额 (1~3%)
      B+: 微超额 (0~1%)
      B:  无历史数据
      B-: 微跑输 (-2~0%)
      D:  强跑输 (<-2%)
    """
    if pd.isna(cross_pred):
        return 'B', '无历史数据'
    if cross_pred > 3:
        return 'A+', f'强超额({cross_pred:+.1f}%)'
    elif cross_pred > 1:
        return 'A', f'超额({cross_pred:+.1f}%)'
    elif cross_pred > 0:
        return 'B+', f'微超额({cross_pred:+.1f}%)'
    elif cross_pred > -2:
        return 'B-', f'微跑输({cross_pred:+.1f}%)'
    else:
        return 'D', f'强跑输({cross_pred:+.1f}%)'


# ============================================================
# 卖出函数
# ============================================================
def calc_sell_bear(sd, buy_idx):
    """熊卖: 先判断50~89双降, 再判断首穿89"""
    trend = sd['trend'].values; retail = sd['retail'].values
    closes = sd['close'].values; n = len(closes)
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if trend[k] < 11: end_idx = k; break
    running_max = trend[buy_idx]
    for k in range(buy_idx + 1, end_idx + 1):
        if np.isnan(trend[k]) or np.isnan(retail[k]): continue
        running_max = max(running_max, trend[k])
        if k == 0: continue
        if np.isnan(trend[k-1]) or np.isnan(retail[k-1]): continue
        # 先判断双降 (50~89区间)
        if running_max >= 50 and trend[k] < 89:
            if trend[k] < trend[k-1] and retail[k] < retail[k-1]:
                return (closes[k] / buy_price - 1) * 100, k
        # 再判断首穿89
        if running_max >= 89 and trend[k] < 89 and trend[k-1] >= 89:
            return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_bull(sd, buy_idx):
    """牛卖: 第二次穿89"""
    trend = sd['trend'].values; retail = sd['retail'].values
    closes = sd['close'].values; n = len(closes)
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if trend[k] < 11: end_idx = k; break
    running_max = trend[buy_idx]
    cross_89_count = 0
    for k in range(buy_idx + 1, end_idx + 1):
        if np.isnan(trend[k]) or np.isnan(retail[k]): continue
        running_max = max(running_max, trend[k])
        if k == 0: continue
        if np.isnan(trend[k-1]): continue
        if running_max >= 89 and trend[k] < 89 and trend[k-1] >= 89:
            cross_89_count += 1
            if cross_89_count >= 2:
                return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_trailing(sd, buy_idx, trail_pct=15):
    """移动止损: 从最高点回撤trail_pct%就卖"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if trend[k] < 11: end_idx = k; break
    peak_price = buy_price
    for k in range(buy_idx + 1, end_idx + 1):
        peak_price = max(peak_price, closes[k])
        drawdown = (peak_price - closes[k]) / peak_price * 100
        if drawdown >= trail_pct:
            return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_stall(sd, buy_idx, stall_days=15, trail_pct=15, trend_cap=30):
    """停滞止损: 连续stall_days天不创新高且trend<trend_cap就卖"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    trend_peak = trend[buy_idx] if not np.isnan(trend[buy_idx]) else 0
    stall_count = 0
    price_peak = buy_price
    for k in range(buy_idx + 1, end_idx + 1):
        price_peak = max(price_peak, closes[k])
        dd = (price_peak - closes[k]) / price_peak * 100
        if dd >= trail_pct:
            return (closes[k] / buy_price - 1) * 100, k
        if not np.isnan(trend[k]):
            if trend[k] > trend_peak:
                trend_peak = trend[k]
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= stall_days and trend_peak < trend_cap:
                    return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_target(sd, buy_idx, target_pct=20):
    """目标止盈: 涨到target_pct%就卖"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    for k in range(buy_idx + 1, end_idx + 1):
        ret = (closes[k] / buy_price - 1) * 100
        if ret >= target_pct:
            return ret, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_time(sd, buy_idx, max_days=30):
    """时间止损: 最多持仓max_days天"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    sell_idx = min(buy_idx + max_days, end_idx)
    if sell_idx >= n:
        sell_idx = n - 1
    return (closes[sell_idx] / buy_price - 1) * 100, sell_idx


def calc_sell_trend_break(sd, buy_idx, trend_floor=50):
    """趋势线跌破: trend跌破trend_floor就卖"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    reached_above = False
    for k in range(buy_idx + 1, end_idx + 1):
        if not np.isnan(trend[k]):
            if trend[k] >= trend_floor:
                reached_above = True
            if reached_above and trend[k] < trend_floor:
                return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_trailing_var(sd, buy_idx, trail_pct=10):
    """移动止损变体: 可调trail_pct参数"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    peak_price = buy_price
    for k in range(buy_idx + 1, end_idx + 1):
        peak_price = max(peak_price, closes[k])
        drawdown = (peak_price - closes[k]) / peak_price * 100
        if drawdown >= trail_pct:
            return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


def calc_sell_target_trail(sd, buy_idx, target_pct=20, trail_pct=10):
    """目标+移动止损组合: 先看目标止盈, 未到则trailing保护"""
    closes = sd['close'].values; n = len(closes)
    trend = sd['trend'].values
    buy_price = closes[buy_idx]
    end_idx = n - 1
    for k in range(buy_idx + 1, n):
        if not np.isnan(trend[k]) and trend[k] < 11:
            end_idx = k; break
    peak_price = buy_price
    for k in range(buy_idx + 1, end_idx + 1):
        peak_price = max(peak_price, closes[k])
        ret = (closes[k] / buy_price - 1) * 100
        # 先检查目标止盈
        if ret >= target_pct:
            return ret, k
        # 再检查移动止损
        drawdown = (peak_price - closes[k]) / peak_price * 100
        if drawdown >= trail_pct:
            return (closes[k] / buy_price - 1) * 100, k
    return (closes[end_idx] / buy_price - 1) * 100, end_idx


# ============================================================
# 信号扫描（支持多种卖法）
# ============================================================
def scan_signals(stock_data, zz1000, sell_method_name='inner', sell_fn=None,
                 filter_trend_at_buy_max=FILTER_TREND_AT_BUY_MAX,
                 filter_retail_recovery_max=FILTER_RETAIL_RECOVERY_MAX,
                 big_cycle_context=None, allowed_macro_guas=None):
    """扫描买入信号

    sell_method_name: 'inner'(内卦), 'trailing_15'(移动止损), 'stall'(停滞止损)
    sell_fn: 自定义卖出函数 (sd, buy_idx) -> (ret%, sell_idx)

    过滤条件（v1.1新增，默认启用）:
      filter_trend_at_buy_max: 买点趋势线不超过此值（默认20），设None关闭
      filter_retail_recovery_max: 散户线回升幅度不超过此值（默认500），设None关闭
    """
    daily_hex = {}
    for dt, zz in zz1000.items():
        # 象卦: 直接从预计算的 gua 列读取
        zz_gua = zz.get('gua', '')
        if zz_gua and len(zz_gua) == 3:
            daily_hex[dt] = {'zz_gua': zz_gua}
        else:
            daily_hex[dt] = None

    all_signals = []
    filtered_counts = {'trend_at_buy': 0, 'retail_recovery': 0, 'macro_gua': 0}

    for code, df in stock_data.items():
        if len(df) < 35:
            continue
        dates = df['date'].values
        trend = df['trend'].values
        retail = df['retail'].values
        closes = df['close'].values
        opens = df['open'].values

        pooled = False; pool_retail = 0; in_wave = False
        pool_start_idx = None

        for i in range(1, len(df)):
            if in_wave:
                if not np.isnan(trend[i]) and trend[i] < 11:
                    in_wave = False; pooled = False; pool_retail = 0
                    pool_start_idx = None
                continue
            if not pooled:
                if not np.isnan(retail[i]) and retail[i] < POOL_THRESHOLD:
                    pooled = True; pool_retail = retail[i]
                    pool_start_idx = i
                if pooled and not np.isnan(retail[i]):
                    pool_retail = min(pool_retail, retail[i])
                continue
            if not np.isnan(retail[i]):
                pool_retail = min(pool_retail, retail[i])
            if np.isnan(trend[i]) or np.isnan(trend[i-1]): continue
            if np.isnan(retail[i]) or np.isnan(retail[i-1]): continue

            # 双升信号
            if retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11:
                signal_date = dates[i]
                dt_str = str(signal_date)

                # ---- 上下文过滤 (v1.1 + 大周期) ----
                skip_signal = False
                context = (big_cycle_context or {}).get(dt_str, {})
                macro_gua = context.get('macro_gua', '')
                macro_gua_name = context.get('macro_gua_name', '')
                market_gua = context.get('market_gua', '')
                market_gua_name = context.get('market_gua_name', '')
                big_cycle_ok = allow_macro_gua(macro_gua, allowed_macro_guas)

                if not big_cycle_ok:
                    filtered_counts['macro_gua'] += 1
                    skip_signal = True

                # 过滤1: 买点趋势线过高
                if filter_trend_at_buy_max is not None and trend[i] > filter_trend_at_buy_max:
                    filtered_counts['trend_at_buy'] += 1
                    skip_signal = True

                # 过滤2: 散户线回升幅度过大
                if filter_retail_recovery_max is not None:
                    start = pool_start_idx if pool_start_idx else max(0, i - 60)
                    retail_slice = retail[start:i + 1]
                    valid_retail = retail_slice[~np.isnan(retail_slice)]
                    retail_min_val = float(np.min(valid_retail)) if len(valid_retail) > 0 else retail[i]
                    retail_recovery = retail[i] - retail_min_val
                    if retail_recovery > filter_retail_recovery_max:
                        filtered_counts['retail_recovery'] += 1
                        skip_signal = True

                if skip_signal:
                    in_wave = True
                    continue

                # ---- 原始逻辑继续 ----
                hex_info = daily_hex.get(dt_str)
                zz_gua = hex_info['zz_gua'] if hex_info else ''

                # 根据卖法选择
                if sell_fn is not None:
                    _, sell_idx = sell_fn(df, i)
                elif sell_method_name == 'inner':
                    # 新系统: 使用象卦决定卖法
                    sm = INNER_SELL_METHOD.get(zz_gua, 'bear')
                    if sm == 'bull':
                        _, sell_idx = calc_sell_bull(df, i)
                    else:
                        _, sell_idx = calc_sell_bear(df, i)
                elif sell_method_name == 'trailing_15':
                    _, sell_idx = calc_sell_trailing(df, i, trail_pct=15)
                elif sell_method_name == 'stall':
                    _, sell_idx = calc_sell_stall(df, i)
                else:
                    _, sell_idx = calc_sell_bear(df, i)

                next_idx = i + 1
                if next_idx >= len(df):
                    in_wave = True; continue
                buy_date = dates[next_idx]
                buy_price = opens[next_idx]
                sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                sell_price = closes[sell_idx]
                hold_days = sell_idx - next_idx

                if buy_price <= 0 or np.isnan(buy_price) or hold_days <= 0:
                    in_wave = True; continue

                actual_ret = (sell_price / buy_price - 1) * 100
                stk_gua = _fmt_gua(df['gua'].iloc[i])

                all_signals.append({
                    'code': code,
                    'signal_date': dt_str,
                    'buy_date': str(buy_date),
                    'sell_date': str(sell_date),
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'actual_ret': actual_ret,
                    'hold_days': hold_days,
                    'pool_retail': pool_retail,
                    'is_skip': False,
                    'hex_code': '',
                    'combo': stk_gua,
                    'zz_gua': zz_gua,
                    'stk_gua': stk_gua,
                    'gua_yy': to_yinyang(stk_gua),
                    'macro_gua': macro_gua,
                    'macro_gua_name': macro_gua_name,
                    'market_gua': market_gua,
                    'market_gua_name': market_gua_name,
                    'big_cycle_ok': big_cycle_ok,
                })
                in_wave = True

    return pd.DataFrame(all_signals).sort_values('signal_date').reset_index(drop=True)


# ============================================================
# 联合策略模拟引擎
# ============================================================
def simulate_hybrid(sig_crazy, sig_normal, zz_df, trigger_fn,
                    crazy_allowed, crazy_max_pos, crazy_daily_limit,
                    normal_allowed, normal_max_pos, normal_daily_limit,
                    init_capital=None):
    """联合策略模拟: 疯狂模式 + 常规模式"""
    zz_indexed = zz_df.set_index('date')

    crazy_by_date = {}
    for _, row in sig_crazy.iterrows():
        crazy_by_date.setdefault(row['signal_date'], []).append(row)
    normal_by_date = {}
    for _, row in sig_normal.iterrows():
        normal_by_date.setdefault(row['signal_date'], []).append(row)

    all_dates = sorted(set(
        sig_crazy['signal_date'].tolist() + sig_crazy['sell_date'].tolist() +
        sig_normal['signal_date'].tolist() + sig_normal['sell_date'].tolist()))

    capital = init_capital or INIT_CAPITAL
    positions = []
    trade_log = []
    daily_equity = []

    for dt in all_dates:
        # 1. 卖出到期持仓
        new_pos = []
        for pos in positions:
            if pos['sell_date'] <= dt:
                profit = (pos['sell_price'] / pos['buy_price'] - 1) * pos['cost']
                capital += pos['cost'] + profit
                trade_log.append({
                    'code': pos['code'], 'buy_date': pos['buy_date'],
                    'sell_date': pos['sell_date'], 'cost': pos['cost'],
                    'profit': profit, 'buy_price': pos['buy_price'],
                    'sell_price': pos['sell_price'],
                    'ret_pct': (pos['sell_price'] / pos['buy_price'] - 1) * 100,
                    'hold_days': pos['hold_days'], 'grade': pos['grade'],
                    'mode': pos.get('mode', '?'),
                    'macro_gua': pos.get('macro_gua', ''),
                    'market_gua': pos.get('market_gua', ''),
                    'big_cycle_ok': pos.get('big_cycle_ok', False),
                })
            else:
                new_pos.append(pos)
        positions = new_pos

        # 2. 判断当天模式
        is_crazy = False
        if dt in zz_indexed.index:
            try:
                is_crazy = trigger_fn(zz_indexed.loc[dt])
            except:
                pass

        if is_crazy:
            candidates = crazy_by_date.get(dt, [])
            allowed = crazy_allowed
            max_pos = crazy_max_pos
            daily_limit = crazy_daily_limit
            mode = 'crazy'
        else:
            candidates = normal_by_date.get(dt, [])
            allowed = normal_allowed
            max_pos = normal_max_pos
            daily_limit = normal_daily_limit
            mode = 'normal'

        # 3. 过滤和买入
        if candidates:
            filtered = [c for c in candidates if not c['is_skip'] and c['grade'] in allowed]
            filtered.sort(key=lambda x: x['pool_retail'])
            slots = max_pos - len(positions)
            can_buy = min(slots, daily_limit, len(filtered))
            if can_buy > 0 and capital > 1000:
                total_eq = capital + sum(p['cost'] for p in positions)
                per_slot = total_eq / max_pos
                per_buy = min(per_slot, capital / can_buy)
                for i in range(can_buy):
                    cost = min(per_buy, capital)
                    if cost < 1000:
                        break
                    c = filtered[i]
                    capital -= cost
                    positions.append({
                        'code': c['code'], 'buy_date': c['buy_date'],
                        'sell_date': c['sell_date'], 'buy_price': c['buy_price'],
                        'sell_price': c['sell_price'], 'cost': cost,
                        'hold_days': c['hold_days'], 'grade': c['grade'],
                        'mode': mode,
                        'macro_gua': c.get('macro_gua', ''),
                        'market_gua': c.get('market_gua', ''),
                        'big_cycle_ok': c.get('big_cycle_ok', False),
                    })

        # 4. 记录净值
        hold_val = sum(p['cost'] for p in positions)
        daily_equity.append({
            'date': dt, 'cash': capital, 'hold_value': hold_val,
            'total_equity': capital + hold_val, 'n_positions': len(positions),
        })

    # 清仓
    for pos in positions:
        profit = (pos['sell_price'] / pos['buy_price'] - 1) * pos['cost']
        capital += pos['cost'] + profit
        trade_log.append({
            'code': pos['code'], 'buy_date': pos['buy_date'],
            'sell_date': pos['sell_date'], 'cost': pos['cost'],
            'profit': profit, 'buy_price': pos['buy_price'],
            'sell_price': pos['sell_price'],
            'ret_pct': (pos['sell_price'] / pos['buy_price'] - 1) * 100,
            'hold_days': pos['hold_days'], 'grade': pos['grade'],
            'mode': pos.get('mode', '?'),
            'macro_gua': pos.get('macro_gua', ''),
            'market_gua': pos.get('market_gua', ''),
            'big_cycle_ok': pos.get('big_cycle_ok', False),
        })

    _init = init_capital or INIT_CAPITAL
    return {
        'final_capital': capital, 'init_capital': _init,
        'total_return': (capital / _init - 1) * 100,
        'trade_log': trade_log, 'daily_equity': daily_equity,
    }


# ============================================================
# 统计
# ============================================================
def calc_stats(result, label=''):
    trades = result['trade_log']
    if not trades:
        return {'label': label, 'final_capital': result['init_capital'],
                'total_return': 0, 'trade_count': 0, 'win_rate': 0,
                'avg_ret': 0, 'max_dd': 0, 'avg_hold': 0, 'avg_pos': 0,
                'yearly': {}, 'context_stats': build_context_stats([])}
    rets = [t['ret_pct'] for t in trades]
    wins = [t for t in trades if t['profit'] > 0]
    yearly = {}
    for t in trades:
        y = t['buy_date'][:4]
        yearly.setdefault(y, {'profit': 0, 'count': 0, 'wins': 0})
        yearly[y]['profit'] += t['profit']
        yearly[y]['count'] += 1
        if t['profit'] > 0:
            yearly[y]['wins'] += 1
    eq = result['daily_equity']
    peak = result['init_capital']
    max_dd = 0
    for e in eq:
        if e['total_equity'] > peak:
            peak = e['total_equity']
        dd = (peak - e['total_equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd
    avg_pos = np.mean([e['n_positions'] for e in eq]) if eq else 0
    return {
        'label': label,
        'final_capital': result['final_capital'],
        'total_return': result['total_return'],
        'trade_count': len(trades),
        'avg_ret': np.mean(rets),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_hold': np.mean([t['hold_days'] for t in trades]),
        'max_dd': max_dd,
        'avg_pos': avg_pos,
        'yearly': yearly,
        'context_stats': build_context_stats(trades),
    }


# ============================================================
# 主流程
# ============================================================
def run(start_date=None, end_date=None, init_capital=None,
        filter_trend=None, filter_retail=None, macro_guas=None):
    year_start = start_date or YEAR_START
    year_end = end_date or YEAR_END
    capital = init_capital or INIT_CAPITAL
    ft = filter_trend if filter_trend is not None else FILTER_TREND_AT_BUY_MAX
    fr = filter_retail if filter_retail is not None else FILTER_RETAIL_RECOVERY_MAX
    allowed_macro_guas = parse_macro_guas(macro_guas)

    print("=" * 100)
    print("  联合策略回测: 疯狂(stall15+cap30) + 常规(A+_inner卖法)")
    print(f"  区间: {year_start} ~ {year_end}  初始资金: {capital:,}")
    print(f"  大周期白名单: {','.join(sorted(allowed_macro_guas)) if allowed_macro_guas else '空'}")
    print("=" * 100)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    zz_df = load_zz1000_full()
    zz1000 = load_zz1000()
    stock_data = load_stocks()
    big_cycle_context = load_big_cycle_context()
    print(f"  个股: {len(stock_data)} 只")
    print(f"  大周期上下文: {len(big_cycle_context)} 个交易日")

    # 疯狂模式触发条件
    trigger = lambda r: r.get('trend', 99) < 45 and r.get('main_force', -999) > 0
    crazy_allowed = {'A+', 'A', 'B+', 'B', 'B-', 'D'}
    normal_allowed = {'A+'}  # S4_年阴+超额>3%

    # 2. 疯狂模式信号 (stall15+cap30)
    print("\n[2] 扫描疯狂模式信号 (stall15+cap30)...")
    sell_stall = lambda sd, i: calc_sell_stall(sd, i, stall_days=15, trail_pct=15, trend_cap=30)
    sig_crazy = scan_signals(stock_data, zz1000, sell_fn=sell_stall,
                             filter_trend_at_buy_max=ft,
                             filter_retail_recovery_max=fr,
                             big_cycle_context=big_cycle_context,
                             allowed_macro_guas=allowed_macro_guas)
    sig_crazy = sig_crazy[(sig_crazy['signal_date'] >= year_start) &
                          (sig_crazy['signal_date'] < year_end)].reset_index(drop=True)
    crazy_context_summary = summarize_signal_context(sig_crazy)
    sig_crazy = build_512_rolling_pred(sig_crazy, min_hist=3)
    sig_crazy['grade'] = [grade_signal(r['gua_yy'], r['combo_pred'])[0]
                          for _, r in sig_crazy.iterrows()]
    print(f"  疯狂信号: {len(sig_crazy)}")

    # 3. 常规模式信号 (inner卖法)
    print("\n[3] 扫描常规模式信号 (inner卖法)...")
    sig_normal = scan_signals(stock_data, zz1000, sell_method_name='inner',
                              filter_trend_at_buy_max=ft,
                              filter_retail_recovery_max=fr,
                              big_cycle_context=big_cycle_context,
                              allowed_macro_guas=allowed_macro_guas)
    sig_normal = sig_normal[(sig_normal['signal_date'] >= year_start) &
                            (sig_normal['signal_date'] < year_end)].reset_index(drop=True)
    normal_context_summary = summarize_signal_context(sig_normal)
    sig_normal = build_512_rolling_pred(sig_normal, min_hist=3)
    sig_normal['grade'] = [grade_signal(r['gua_yy'], r['combo_pred'])[0]
                           for _, r in sig_normal.iterrows()]
    print(f"  常规信号: {len(sig_normal)}")

    print("\n[3b] 大周期过滤后信号分布...")
    print(f"  疯狂模式信号数: {crazy_context_summary['signal_count']}")
    print(f"  常规模式信号数: {normal_context_summary['signal_count']}")
    print(f"  疯狂模式 macro 分布: {crazy_context_summary['macro_gua_counts']}")
    print(f"  常规模式 macro 分布: {normal_context_summary['macro_gua_counts']}")
    print("\n[4] 联合策略模拟...")
    result = simulate_hybrid(
        sig_crazy, sig_normal, zz_df, trigger,
        crazy_allowed=crazy_allowed, crazy_max_pos=5, crazy_daily_limit=1,
        normal_allowed=normal_allowed, normal_max_pos=5, normal_daily_limit=1,
        init_capital=capital,
    )
    stats = calc_stats(result, '联合策略')
    trades = result['trade_log']
    crazy_trades = [t for t in trades if t.get('mode') == 'crazy']
    normal_trades = [t for t in trades if t.get('mode') == 'normal']

    # 基准: 纯疯狂(常规空仓)
    print("[4b] 基准: 纯疯狂(常规空仓)...")
    result_base = simulate_hybrid(
        sig_crazy, sig_normal, zz_df, trigger,
        crazy_allowed=crazy_allowed, crazy_max_pos=5, crazy_daily_limit=1,
        normal_allowed=set(), normal_max_pos=5, normal_daily_limit=1,
        init_capital=capital,
    )
    stats_base = calc_stats(result_base, '纯疯狂')

    # ============================================================
    # 输出报告
    # ============================================================
    print("\n" + "=" * 100)
    print("  Part1: 策略总览")
    print("=" * 100)

    print(f"\n  联合策略:")
    print(f"    买入过滤: 趋势线≤{ft} + 散户线回升≤{fr}")
    print(f"    疯狂: stall15+cap30 | S1全等级 | 5仓1日 | trigger: trend<45且mf>0")
    print(f"    常规: inner卖法 | A+等级 | 5仓1日")
    print(f"    初始资金: {capital:,}")
    print(f"    回测区间: {year_start} ~ {year_end}")

    print(f"\n  {'指标':<16} {'联合策略':>14} {'纯疯狂(空仓)':>14} {'差额':>14}")
    print("  " + "-" * 60)
    print(f"  {'终值':<16} {stats['final_capital']:>13,.0f} {stats_base['final_capital']:>13,.0f} "
          f"{stats['final_capital']-stats_base['final_capital']:>+13,.0f}")
    print(f"  {'收益%':<16} {stats['total_return']:>12.1f}% {stats_base['total_return']:>12.1f}% "
          f"{stats['total_return']-stats_base['total_return']:>+12.1f}%")
    print(f"  {'最大回撤%':<14} {stats['max_dd']:>12.1f}% {stats_base['max_dd']:>12.1f}%")
    print(f"  {'交易笔数':<16} {stats['trade_count']:>14} {stats_base['trade_count']:>14}")
    print(f"  {'胜率%':<16} {stats['win_rate']:>12.1f}% {stats_base['win_rate']:>12.1f}%")
    print(f"  {'疯狂笔数':<16} {len(crazy_trades):>14} {stats_base['trade_count']:>14}")
    print(f"  {'常规笔数':<16} {len(normal_trades):>14} {'0':>14}")

    crazy_profit = sum(t['profit'] for t in crazy_trades)
    normal_profit = sum(t['profit'] for t in normal_trades)
    print(f"\n  疯狂利润: {crazy_profit:>12,.0f}")
    print(f"  常规利润: {normal_profit:>12,.0f}")
    print(f"  合计利润: {crazy_profit+normal_profit:>12,.0f}")

    # Part2: 年度明细
    print("\n" + "=" * 100)
    print("  Part2: 年度明细 (分疯狂/常规)")
    print("=" * 100)

    yearly = {}
    for t in trades:
        y = t['buy_date'][:4]
        yearly.setdefault(y, {'c_profit': 0, 'c_count': 0, 'c_wins': 0,
                               'n_profit': 0, 'n_count': 0, 'n_wins': 0})
        if t.get('mode') == 'crazy':
            yearly[y]['c_profit'] += t['profit']
            yearly[y]['c_count'] += 1
            if t['profit'] > 0: yearly[y]['c_wins'] += 1
        else:
            yearly[y]['n_profit'] += t['profit']
            yearly[y]['n_count'] += 1
            if t['profit'] > 0: yearly[y]['n_wins'] += 1

    yearly_base = {}
    for t in result_base['trade_log']:
        y = t['buy_date'][:4]
        yearly_base.setdefault(y, {'profit': 0, 'count': 0})
        yearly_base[y]['profit'] += t['profit']
        yearly_base[y]['count'] += 1

    print(f"\n  {'年份':<6} {'疯狂利润':>12} {'疯狂笔':>5} {'疯狂胜率':>7}  "
          f"{'常规利润':>12} {'常规笔':>5} {'常规胜率':>7}  "
          f"{'年合计':>12} {'基准利润':>12} {'增量':>12}")
    print("  " + "-" * 115)

    all_years = sorted(yearly.keys())
    for y in all_years:
        v = yearly[y]
        b = yearly_base.get(y, {'profit': 0, 'count': 0})
        cwr = v['c_wins'] / v['c_count'] * 100 if v['c_count'] > 0 else 0
        nwr = v['n_wins'] / v['n_count'] * 100 if v['n_count'] > 0 else 0
        total = v['c_profit'] + v['n_profit']
        print(f"  {y:<6} {v['c_profit']:>11,.0f} {v['c_count']:>5} {cwr:>6.1f}%  "
              f"{v['n_profit']:>11,.0f} {v['n_count']:>5} {nwr:>6.1f}%  "
              f"{total:>11,.0f} {b['profit']:>11,.0f} {total-b['profit']:>+11,.0f}")

    # Part3: 最终参数总结
    print("\n" + "=" * 100)
    print("  最终参数总结")
    print("=" * 100)
    print(f"""
  买入过滤 (v1.1):
    买点趋势线 ≤ {FILTER_TREND_AT_BUY_MAX}
    散户线回升幅度 ≤ {FILTER_RETAIL_RECOVERY_MAX}

  疯狂模式:
    触发条件: 中证1000 trend<45 且 main_force>0
    等级过滤: S1全等级(A+/A/B+/B/B-/D)
    仓位: 5仓, 每日限买1笔
    卖法: 停滞止损(stall=15天, trail=15%, cap=30)

  常规模式:
    等级过滤: A+
    仓位: 5仓, 每日限买1笔
    卖法: inner(根据中证内卦选bear/bull)

  回测结果:
    初始资金: {capital:,}
    终值: {stats['final_capital']:,.0f}
    收益: {stats['total_return']:.1f}%
    最大回撤: {stats['max_dd']:.1f}%
    交易: {stats['trade_count']}笔 (疯狂{len(crazy_trades)} + 常规{len(normal_trades)})
    胜率: {stats['win_rate']:.1f}%
    vs 纯疯狂: +{stats['total_return']-stats_base['total_return']:.0f}pp
""")
    print("=" * 100)

    # === 导出 JSON 供 dashboard 使用 ===
    eq = result['daily_equity']
    # 统一日期格式为 YYYY-MM-DD (numpy datetime64 → str 可能产生无横杠格式)
    def _fmt_date(d):
        s = str(d)[:10]
        if len(s) == 8 and s.isdigit():  # 20250606 → 2025-06-06
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s

    for e in eq:
        e['date'] = _fmt_date(e['date'])
    for t in result['trade_log']:
        for k in ('buy_date', 'sell_date'):
            if k in t:
                t[k] = _fmt_date(t[k])

    peak = capital
    max_dd, max_dd_date = 0, ''
    for e in eq:
        if e['total_equity'] > peak:
            peak = e['total_equity']
        dd = (peak - e['total_equity']) / peak * 100
        if dd > max_dd:
            max_dd, max_dd_date = dd, e['date']

    out = {
        'meta': {
            'init_capital': capital,
            'final_capital': round(stats['final_capital'], 2),
            'total_return': round(stats['total_return'], 2),
            'trade_count': stats['trade_count'],
            'label': '联合策略',
            'win_rate': round(stats['win_rate'], 2),
            'avg_ret': round(stats['avg_ret'], 2),
            'avg_hold': round(stats['avg_hold'], 2),
            'max_dd': round(max_dd, 2),
            'max_dd_date': max_dd_date,
            'big_cycle_filter': {
                'allowed_macro_guas': sorted(allowed_macro_guas),
            },
        },
        'daily_equity': eq,
        'trade_log': result['trade_log'],
        'yearly': stats['yearly'],
        'context_stats': stats['context_stats'],
        'signal_context': {
            'crazy': crazy_context_summary,
            'normal': normal_context_summary,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), 'data_layer', 'data', 'backtest_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"\n  已导出: {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='联合策略回测')
    parser.add_argument('--start', type=str, default=None,
                        help=f'回测开始日期 (默认: {YEAR_START})')
    parser.add_argument('--end', type=str, default=None,
                        help=f'回测结束日期 (默认: {YEAR_END})')
    parser.add_argument('--capital', type=int, default=None,
                        help=f'初始资金 (默认: {INIT_CAPITAL})')
    parser.add_argument('--filter-trend', type=int, default=None,
                        help=f'买点趋势线上限 (默认: {FILTER_TREND_AT_BUY_MAX})')
    parser.add_argument('--filter-retail', type=int, default=None,
                        help=f'散户线回升上限 (默认: {FILTER_RETAIL_RECOVERY_MAX})')
    parser.add_argument('--macro-guas', type=str, default=None,
                        help='允许做多的大周期卦白名单，逗号分隔，如 001,011,101,111；传空字符串表示全部关闭')
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, init_capital=args.capital,
        filter_trend=args.filter_trend, filter_retail=args.filter_retail,
        macro_guas=args.macro_guas)
