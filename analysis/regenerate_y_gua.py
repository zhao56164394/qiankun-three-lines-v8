# -*- coding: utf-8 -*-
"""用指定窗口长度重新生成 multi_scale_gua_daily.parquet

仅修改年卦窗口 (月聚合下), 月卦/日卦保持原样.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'data_layer'))

from strategy.indicator import _tdx_sma, _tdx_ema
from prepare_multi_scale_gua import (
    compute_scale_per_day, apply_v10_rules,
    GUA_NAMES,
)

YEAR_WINDOW = int(os.environ.get('YEAR_WINDOW', 12))
print(f'Y_WINDOW = {YEAR_WINDOW} 月\n')


def compute_with_window(df_daily, freq, window):
    """compute_scale_per_day 的窗口可调版"""
    df = df_daily.copy().sort_values('date').reset_index(drop=True)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['period'] = df['date_dt'].dt.to_period(freq).astype(str)
    df['hi_run'] = df.groupby('period')['high'].cummax()
    df['lo_run'] = df.groupby('period')['low'].cummin()
    periods_ordered = df['period'].drop_duplicates().tolist()
    p2idx = {p: i for i, p in enumerate(periods_ordered)}
    df['p_idx'] = df['period'].map(p2idx)

    p_stats = df.groupby('p_idx', sort=True).agg(
        hi=('high', 'max'), lo=('low', 'min'), close=('close', 'last'))
    all_hi = p_stats['hi'].values
    all_lo = p_stats['lo'].values
    all_close = p_stats['close'].values

    hhv = pd.Series(all_hi).rolling(window, min_periods=1).max().values
    llv = pd.Series(all_lo).rolling(window, min_periods=1).min().values
    denom = hhv - llv
    rsv_w = np.where(denom > 0, (all_close - llv) / denom * 100, 50.0)
    sma1_w = _tdx_sma(rsv_w, 5, 1)
    sma2_w = _tdx_sma(sma1_w, 3, 1)
    v11_w = 3 * sma1_w - 2 * sma2_w
    trend_w = _tdx_ema(v11_w, 3)

    ma7 = pd.Series(all_close).rolling(7, min_periods=1).mean().values
    raw_mf = np.where(ma7 > 0, (all_close - ma7) / ma7 * 480, 0.0)
    mf_inner_w = _tdx_ema(raw_mf, 2)

    n = len(df)
    trend_arr = np.full(n, np.nan)
    mf_arr = np.full(n, np.nan)
    p_idx_arr = df['p_idx'].values
    hi_run_arr = df['hi_run'].values
    lo_run_arr = df['lo_run'].values
    close_arr = df['close'].values

    for i in range(n):
        w = int(p_idx_arr[i])
        p_hi, p_lo, p_cl = hi_run_arr[i], lo_run_arr[i], close_arr[i]
        start = max(0, w - (window-1))
        completed_lo = all_lo[start:w] if w > start else np.array([])
        completed_hi = all_hi[start:w] if w > start else np.array([])
        llv_val = min(completed_lo.min() if len(completed_lo) > 0 else p_lo, p_lo)
        hhv_val = max(completed_hi.max() if len(completed_hi) > 0 else p_hi, p_hi)
        rsv_val = ((p_cl - llv_val) / (hhv_val - llv_val) * 100) if hhv_val > llv_val else 50.0

        if w == 0:
            s1 = rsv_val; s2 = s1
            v11 = 3*s1 - 2*s2; tr_val = v11
        else:
            s1_p, s2_p, tr_p = sma1_w[w-1], sma2_w[w-1], trend_w[w-1]
            s1 = (4*s1_p + rsv_val)/5
            s2 = (2*s2_p + s1)/3
            v11 = 3*s1 - 2*s2
            tr_val = 0.5*v11 + 0.5*tr_p
        trend_arr[i] = tr_val

        take = min(6, w)
        ma7_sum = all_close[w-take:w].sum() + p_cl
        ma7_val = ma7_sum / (take+1)
        raw_mf_val = ((p_cl - ma7_val) / ma7_val * 480) if ma7_val > 0 else 0.0
        if w == 0:
            mi = raw_mf_val
        else:
            mi_p = mf_inner_w[w-1]
            mi = (1.0/3)*mi_p + (2.0/3)*raw_mf_val
        mf_arr[i] = mi * 5
    return trend_arr, mf_arr


# 读源
src = os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv')
df = pd.read_csv(src, encoding='utf-8-sig')
print(f'读取 {len(df)} 条')

# 日卦 (现成 trend + main_force)
d_trend = df['trend'].astype(float).values
d_mf = df['main_force'].astype(float).values
d_pos, d_spd, d_acc, d_gua = apply_v10_rules(d_trend, d_mf)

# 月卦 (周尺度, 保持原 55 周窗口)
print('计算月卦 (周, 55 周)...')
m_trend, m_mf = compute_scale_per_day(df, 'W-FRI')
m_pos, m_spd, m_acc, m_gua = apply_v10_rules(m_trend, m_mf)

# 年卦 (月尺度, 自定义窗口)
print(f'计算年卦 (月, {YEAR_WINDOW} 月)...')
y_trend, y_mf = compute_with_window(df, 'M', YEAR_WINDOW)
y_pos, y_spd, y_acc, y_gua = apply_v10_rules(y_trend, y_mf)

out = pd.DataFrame({
    'date': df['date'].values,
    'close': df['close'].values,
    'd_trend': d_trend, 'd_mf': d_mf, 'd_pos': d_pos, 'd_spd': d_spd, 'd_acc': d_acc, 'd_gua': d_gua,
    'm_trend': m_trend, 'm_mf': m_mf, 'm_pos': m_pos, 'm_spd': m_spd, 'm_acc': m_acc, 'm_gua': m_gua,
    'y_trend': y_trend, 'y_mf': y_mf, 'y_pos': y_pos, 'y_spd': y_spd, 'y_acc': y_acc, 'y_gua': y_gua,
})
out['d_name'] = out['d_gua'].map(GUA_NAMES).fillna('')
out['m_name'] = out['m_gua'].map(GUA_NAMES).fillna('')
out['y_name'] = out['y_gua'].map(GUA_NAMES).fillna('')

dst = os.path.join(ROOT, 'data_layer', 'data', 'foundation', f'multi_scale_gua_daily.parquet')
out.to_parquet(dst, index=False)
print(f'保存 {dst} (年卦窗口 {YEAR_WINDOW} 月)')

# 简要确认
chg = (out['y_gua'] != out['y_gua'].shift()).astype(int).sum()
print(f'  y_gua 切换 {chg} 次')
