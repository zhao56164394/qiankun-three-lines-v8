# -*- coding: utf-8 -*-
"""
backtest_capital.py — 回测的数据加载与基础算子

作为共享基础库:
  - 加载中证1000 / 个股 / 股票事件 / 大周期上下文
  - 提供全部卖出函数 (calc_sell_bear / bull / stall / trailing / ...)
  - scan_signals 通用信号扫描
  - 信号统计辅助 (calc_stats / summarize_signal_context)

本文件已不再承担 crazy/normal 联合策略回测的主流程 —— 八卦分治回测 (backtest_8gua.py)
取代了旧的 simulate_hybrid 流水线。保留本文件仅作为算子层被其他模块 import。
"""
import sys
import io
import os
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_layer.foundation_data import load_market_bagua

sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)


# ============================================================
# 常量配置 — 统一从 backtest_bt.config 导入
# ============================================================
from backtest_bt.config import (
    INIT_CAPITAL, YEAR_START, YEAR_END, DATA_DIR,
)



def load_big_cycle_context():
    """加载市场卦上下文，按日期映射。"""
    market_df = load_market_bagua().copy()
    market_df['date'] = market_df['date'].astype(str)
    market_df['gua_code'] = market_df['gua_code'].astype(str).str.zfill(3)

    return {
        row['date']: {
            'ren_gua': row['gua_code'],
            'ren_gua_name': row.get('gua_name', ''),
        }
        for _, row in market_df[['date', 'gua_code', 'gua_name']].drop_duplicates('date').iterrows()
    }



def summarize_signal_context(sig_df):
    if sig_df is None or len(sig_df) == 0:
        return {
            'signal_count': 0,
            'ren_gua_counts': {},
        }
    ren_counts = sig_df['ren_gua'].fillna('').astype(str).value_counts().sort_index().to_dict() if 'ren_gua' in sig_df.columns else {}
    return {
        'signal_count': int(len(sig_df)),
        'ren_gua_counts': {str(k): int(v) for k, v in ren_counts.items()},
    }


# ============================================================
# 数据加载（带pickle缓存）
# ============================================================


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


def _read_zz1000_df():
    """读 zz1000：Parquet 优先，CSV 兜底。"""
    pq_path = os.path.join(DATA_DIR, 'zz1000_daily.parquet')
    csv_path = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    return df


def _zz1000_source_files():
    """zz1000 缓存依赖的源文件（任一存在即可）。"""
    return [
        os.path.join(DATA_DIR, 'zz1000_daily.parquet'),
        os.path.join(DATA_DIR, 'zz1000_daily.csv'),
    ]


def _build_zz1000():
    df = _read_zz1000_df()
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
    return _load_cached('zz1000', _zz1000_source_files(), _build_zz1000)


def _build_zz1000_full():
    df = _read_zz1000_df()
    df['gua'] = df['gua'].astype(str).str.zfill(3)
    df['trend_ma10'] = df['trend'].rolling(10).mean()
    return df


def load_zz1000_full():
    """加载中证1000全部字段（DataFrame, 用于疯狂模式触发判断）"""
    return _load_cached('zz1000_full', _zz1000_source_files(), _build_zz1000_full)


def _build_stocks():
    """优先用 stocks.parquet 单文件加载；缺失则 fallback 到 5102 个 CSV 循环。"""
    pq_path = os.path.join(DATA_DIR, 'stocks.parquet')
    if os.path.exists(pq_path):
        cols = ['code', 'date', 'open', 'close', 'trend', 'retail', 'gua']
        df = pd.read_parquet(pq_path, columns=cols)
        df['code'] = df['code'].astype(str).str.zfill(6)
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
        stock_data = {code: g.drop(columns='code').reset_index(drop=True)
                      for code, g in df.groupby('code', sort=False)}
        return stock_data

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
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            stock_data[code] = df
        except:
            continue
    return stock_data


def load_stocks():
    """加载个股数据：优先用 stocks.parquet 的 mtime，否则抽样检查 stocks/ 目录。"""
    pq_path = os.path.join(DATA_DIR, 'stocks.parquet')
    stock_dir = os.path.join(DATA_DIR, 'stocks')
    pkl = _cache_path('stocks')

    if os.path.exists(pkl):
        pkl_mtime = os.path.getmtime(pkl)
        if os.path.exists(pq_path):
            if os.path.getmtime(pq_path) < pkl_mtime:
                with open(pkl, 'rb') as f:
                    return pickle.load(f)
        elif os.path.exists(stock_dir):
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
    pq_path = os.path.join(DATA_DIR, 'stock_seg_events.parquet')
    csv_path = os.path.join(DATA_DIR, 'stock_seg_events.csv')
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['event_date'] = df['event_date'].astype(str)
    df['avail_date'] = df['avail_date'].astype(str)
    if 'gua' in df.columns:
        df['gua'] = df['gua'].astype(str).str.split('.').str[0].str.zfill(3)
    elif 'year_gua' in df.columns:
        df['gua'] = df['year_gua'].astype(str).str.split('.').str[0].str.zfill(3)
    if 'zz_gua' in df.columns:
        df['zz_gua'] = df['zz_gua'].astype(str).str.split('.').str[0].str.zfill(3)
        df.rename(columns={'zz_gua': 'tian_gua'}, inplace=True)
    return df


def load_stock_events():
    """加载个股段首事件表"""
    return _load_cached('stock_events', [
        os.path.join(DATA_DIR, 'stock_seg_events.parquet'),
        os.path.join(DATA_DIR, 'stock_seg_events.csv'),
    ], _build_stock_events)


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


