# -*- coding: utf-8 -*-
"""验证两个猜测:
1. 大涨过后下跌趋势的巽卦 = 假突破
   → 拆 巽日入场前的近期涨跌, 看主升率/期望差异

2. 巽转乾那天作为触发更好
   → 扫所有 "前一日 d_gua=011 + 当日 d_gua=111" 作为入场点
   → 对比直接巽日入场的 baseline / 主升率
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '111'

WINDOWS = [
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # ===== 猜测 1: 巽日入场前近期涨跌 =====
    print(f'\n=== 猜测 1: 巽日前 5d/10d/30d 涨跌跟 主升/假突破 关系 ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]; td = trend_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            # 入场前近期涨跌
            ret_5d = (cl[i] / cl[i-5] - 1) * 100 if i-5 >= 0 else float('nan')
            ret_10d = (cl[i] / cl[i-10] - 1) * 100 if i-10 >= 0 else float('nan')
            ret_30d = (cl[i] / cl[i-30] - 1) * 100 if i-30 >= 0 else float('nan')
            # trend 近期斜率
            td_5d_slope = float(td[i] - td[i-5]) if i-5 >= 0 else float('nan')
            td_10d_max = float(np.nanmax(td[max(0,i-10):i+1]))
            events.append({
                'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN, 'ret_30': ret_30,
                'ret_5d': ret_5d, 'ret_10d': ret_10d, 'ret_30d': ret_30d,
                'td_5d_slope': td_5d_slope, 'td_10d_max': td_10d_max,
                'td': float(td[i]),
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日: {len(df_e):,}')

    # 按 5d 涨幅分桶
    print(f'\n  按"前 5 日涨跌"分桶:')
    print(f'  {"桶":<22} {"n":>8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8} {"加权":>8}')
    bins = [(-100, -10, '< -10%'), (-10, -5, '-10~-5'), (-5, 0, '-5~0'),
            (0, 5, '0~5%'), (5, 10, '5~10%'), (10, 100, '> 10%')]
    for lo, hi, label in bins:
        sub = df_e[(df_e['ret_5d'] >= lo) & (df_e['ret_5d'] < hi)]
        if len(sub) == 0: continue
        zsl = sub[sub['is_zsl']]
        fake = sub[~sub['is_zsl']]
        zsl_ret = zsl['ret_30'].mean() if len(zsl) > 0 else float('nan')
        fake_ret = fake['ret_30'].mean() if len(fake) > 0 else float('nan')
        avg = sub['ret_30'].mean()
        zsl_rate = sub['is_zsl'].mean() * 100
        print(f'  {label:<22} {len(sub):>8,} {zsl_rate:>7.1f} {zsl_ret:>+8.2f} {fake_ret:>+7.2f} {avg:>+7.2f}')

    print(f'\n  按"前 10 日涨跌"分桶:')
    print(f'  {"桶":<22} {"n":>8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8} {"加权":>8}')
    bins10 = [(-100, -15, '< -15%'), (-15, -5, '-15~-5'), (-5, 0, '-5~0'),
              (0, 5, '0~5%'), (5, 15, '5~15%'), (15, 100, '> 15%')]
    for lo, hi, label in bins10:
        sub = df_e[(df_e['ret_10d'] >= lo) & (df_e['ret_10d'] < hi)]
        if len(sub) == 0: continue
        zsl = sub[sub['is_zsl']]
        fake = sub[~sub['is_zsl']]
        zsl_ret = zsl['ret_30'].mean() if len(zsl) > 0 else float('nan')
        fake_ret = fake['ret_30'].mean() if len(fake) > 0 else float('nan')
        avg = sub['ret_30'].mean()
        zsl_rate = sub['is_zsl'].mean() * 100
        print(f'  {label:<22} {len(sub):>8,} {zsl_rate:>7.1f} {zsl_ret:>+8.2f} {fake_ret:>+7.2f} {avg:>+7.2f}')

    print(f'\n  按"前 30 日涨跌"分桶:')
    print(f'  {"桶":<22} {"n":>8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8} {"加权":>8}')
    bins30 = [(-100, -20, '< -20%'), (-20, -10, '-20~-10'), (-10, 0, '-10~0'),
              (0, 10, '0~10%'), (10, 25, '10~25%'), (25, 100, '> 25%')]
    for lo, hi, label in bins30:
        sub = df_e[(df_e['ret_30d'] >= lo) & (df_e['ret_30d'] < hi)]
        if len(sub) == 0: continue
        zsl_rate = sub['is_zsl'].mean() * 100
        avg = sub['ret_30'].mean()
        print(f'  {label:<22} {len(sub):>8,} {zsl_rate:>7.1f} {"--":>9} {"--":>8} {avg:>+7.2f}')

    # ===== 猜测 2: 巽→乾切换那天作为触发 =====
    print(f'\n\n=== 猜测 2: "前一日巽 + 当日乾" 作为入场点 ===')
    print(f'  含义: 主升浪刚刚启动 (巽→乾切换), 第一根乾日入场')

    xun_to_qian = []
    pure_qian = []  # 对照: 已经在乾连续中, 第 N 日 (N≥2) 入场
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK + 1, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '111': continue  # 必须当日乾
            if gua[i-1] != '011': continue  # 前一日必须巽

            # 这个就是巽→乾切换日
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            xun_to_qian.append({
                'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN, 'ret_30': ret_30,
            })

    df_xq = pd.DataFrame(xun_to_qian)
    print(f'  巽→乾切换日 (在乾 regime 内): {len(df_xq):,}')
    print(f'  全期期望: {df_xq["ret_30"].mean():+.2f}%')
    print(f'  主升率 (起点 30 日内乾≥10): {df_xq["is_zsl"].mean()*100:.1f}%')

    df_xq['seg'] = ''
    for w in WINDOWS:
        df_xq.loc[(df_xq['date'] >= w[1]) & (df_xq['date'] < w[2]), 'seg'] = w[0]

    print(f'\n  walk-forward:')
    print(f'  {"seg":<14} {"n":>6} {"期望%":>7} {"主升率%":>8}')
    for w in WINDOWS:
        sub = df_xq[df_xq['seg'] == w[0]]
        if len(sub) < 50: continue
        ret = sub['ret_30'].mean()
        zsl = sub['is_zsl'].mean() * 100
        print(f'  {w[0]:<14} {len(sub):>6,} {ret:>+6.2f} {zsl:>7.1f}')

    # 主升 vs 假
    zsl_ret = df_xq[df_xq['is_zsl']]['ret_30'].mean()
    fake_ret = df_xq[~df_xq['is_zsl']]['ret_30'].mean()
    print(f'\n  主升 ({df_xq["is_zsl"].sum():,}): {zsl_ret:+.2f}%')
    print(f'  假  ({(~df_xq["is_zsl"]).sum():,}): {fake_ret:+.2f}%')

    # 跟巽日对比
    print(f'\n## 巽日 vs 巽→乾日 对比 (乾 regime)')
    print(f'  {"触发":<14} {"n":>8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8} {"加权":>8}')

    # 巽日数据
    df_xun = df_e
    zsl_xun = df_xun[df_xun['is_zsl']]['ret_30'].mean()
    fake_xun = df_xun[~df_xun['is_zsl']]['ret_30'].mean()
    print(f'  {"巽日":<14} {len(df_xun):>8,} {df_xun["is_zsl"].mean()*100:>7.1f} {zsl_xun:>+8.2f} {fake_xun:>+7.2f} {df_xun["ret_30"].mean():>+7.2f}')
    print(f'  {"巽→乾日":<14} {len(df_xq):>8,} {df_xq["is_zsl"].mean()*100:>7.1f} {zsl_ret:>+8.2f} {fake_ret:>+7.2f} {df_xq["ret_30"].mean():>+7.2f}')


if __name__ == '__main__':
    main()
