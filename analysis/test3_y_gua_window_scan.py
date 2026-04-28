# -*- coding: utf-8 -*-
"""实验 1b: 年卦窗口长度扫描 (在 freq='M' 月聚合下)

55 月窗口让年卦严重滞后. 试 12/24/36/55 月看是否命中率提升.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'data_layer'))

from strategy.indicator import _tdx_sma, _tdx_ema
from prepare_multi_scale_gua import apply_v10_rules


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


# 数据
df = pd.read_csv(os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv'), encoding='utf-8-sig')
df['date'] = df['date'].astype(str)
print(f'数据: {len(df)} 条')

bull_periods = [
    ('2014-07-01', '2015-06-12', '14-15杠杆牛'),
    ('2019-01-04', '2019-04-19', '19Q1反弹'),
    ('2020-04-01', '2021-02-18', '20疫后小牛'),
    ('2024-09-24', '2025-12-31', '24-25政策牛'),
]
bear_periods = [
    ('2015-06-15', '2016-01-28', '15股灾+熔断'),
    ('2018-02-01', '2018-12-28', '18贸战熊'),
    ('2021-02-22', '2024-02-05', '21-24深熊'),
]


def evaluate(window):
    print(f'\n--- 窗口 = {window} 月 ---')
    y_trend, y_mf = compute_with_window(df, 'M', window)
    _, _, _, y_gua = apply_v10_rules(y_trend, y_mf)
    g = pd.Series(y_gua, index=df['date'].values)
    g_valid = g[g != '']

    chg = (g_valid != g_valid.shift()).astype(int)
    if len(chg) > 0: chg.iloc[0] = 0
    n_switch = int(chg.sum())
    avg_seg = len(g_valid) / max(n_switch+1, 1)
    print(f'  y_gua 切换 {n_switch} 次, 平均 {avg_seg:.1f} 天/段')

    # 牛市 / 熊市 阳阴占比
    print(f'  {"牛市":<26} {"日数":>6} {"阳卦%":>7}')
    bull_yang = 0; bull_total = 0
    for s, e, name in bull_periods:
        sub = g_valid[(g_valid.index>=s) & (g_valid.index<=e)]
        if len(sub) == 0: continue
        pos = sub.str[0].astype(int)
        pct = (pos==1).sum() / len(sub) * 100
        bull_yang += (pos==1).sum(); bull_total += len(sub)
        print(f'  {name:<26} {len(sub):>6} {pct:>6.1f}%')
    print(f'  {"牛市加权":<26} {bull_total:>6} {bull_yang/max(bull_total,1)*100:>6.1f}%')

    print(f'  {"熊市":<26} {"日数":>6} {"阴卦%":>7}')
    bear_yin = 0; bear_total = 0
    for s, e, name in bear_periods:
        sub = g_valid[(g_valid.index>=s) & (g_valid.index<=e)]
        if len(sub) == 0: continue
        pos = sub.str[0].astype(int)
        pct = (pos==0).sum() / len(sub) * 100
        bear_yin += (pos==0).sum(); bear_total += len(sub)
        print(f'  {name:<26} {len(sub):>6} {pct:>6.1f}%')
    print(f'  {"熊市加权":<26} {bear_total:>6} {bear_yin/max(bear_total,1)*100:>6.1f}%')

    combined = (bull_yang + bear_yin) / max(bull_total + bear_total, 1) * 100
    print(f'  综合命中率: {combined:.1f}%')
    return {'window': window, 'switches': n_switch, 'avg_seg': avg_seg,
            'bull_hit%': bull_yang/max(bull_total,1)*100,
            'bear_hit%': bear_yin/max(bear_total,1)*100,
            'combined%': combined}


print('=== 实验 1b: 月线窗口长度扫描 ===')
results = []
for w in [12, 18, 24, 36, 55, 72]:
    r = evaluate(w)
    results.append(r)

print('\n' + '='*80)
print('汇总')
print('='*80)
print(f'  {"窗口":>5} {"切换":>5} {"段长":>6} {"牛%":>6} {"熊%":>6} {"综合%":>7}')
for r in results:
    mark = ' ←现行' if r['window']==55 else ''
    print(f'  {r["window"]:>3}月 {r["switches"]:>5} {r["avg_seg"]:>5.0f}天 {r["bull_hit%"]:>5.1f}% {r["bear_hit%"]:>5.1f}% {r["combined%"]:>6.1f}%{mark}')
