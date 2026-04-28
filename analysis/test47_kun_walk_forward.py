# -*- coding: utf-8 -*-
"""Step 24 — 坤 regime 指纹 walk-forward

按 7 段窗口分 (跟其他 walk-forward 一致), 在坤 regime 内验证 4 个 Top 指纹
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '000'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
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
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'd_gua', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d = df['d_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy(); stk_y = df['y_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy(); mkt_m = df['mkt_m'].to_numpy(); mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫坤 regime 巽日
    print(f'\n=== 扫坤 regime 巽日 ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = stk_d[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            wlo = i - LOOKBACK + 1
            events.append({
                'date': date_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d[gi], 'mkt_m': mkt_m[gi],
                'stk_m': stk_m[gi], 'stk_y': stk_y[gi],
                'mf_30d_min': mf[wlo:i+1].min(),
            })
    df_e = pd.DataFrame(events)
    print(f'  坤 regime 巽日: {len(df_e):,}, {time.time()-t1:.1f}s')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  打段后: {len(df_e):,}')

    # 各段 baseline (该段坤 regime 任意巽日均 ret)
    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = (len(seg), seg['ret_30'].mean() if len(seg) > 0 else 0)
    print(f'\n## 坤 regime × 7 段 baseline')
    for w in WINDOWS:
        n, b = seg_baselines[w[0]]
        print(f'  {w[0]:<14} n={n:>6,}  期望 {b:>+5.2f}%')

    # 验证候选指纹
    print(f'\n## 4 个候选指纹 walk-forward (坤 regime 内)')
    print(f'  {"指纹":<24} {"全 n":>6} {"全期望":>7}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 130)

    candidates = [
        ('mkt_d=011巽', df_e['mkt_d'] == '011', 7.26),
        ('mkt_m=010坎', df_e['mkt_m'] == '010', 6.76),
        ('mf_30d_min < -200', df_e['mf_30d_min'] < -200, 6.64),
        ('stk_m=010坎', df_e['stk_m'] == '010', 6.36),
    ]

    for label, mask, exp in candidates:
        sub = df_e[mask]
        n_full = len(sub); ret_full = sub['ret_30'].mean()
        print(f'  {label:<24} {n_full:>6,} {ret_full:>+6.2f}%', end='')
        n_pass = 0; n_fail = 0; n_low = 0
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 30:
                n_low += 1
                print(f' {len(seg):>4}|  -- ', end='')
                continue
            r = seg['ret_30'].mean()
            base_n, base_r = seg_baselines[w[0]]
            lift = r - base_r
            mark = '✅' if lift > 1 else ('❌' if lift < -1 else '○')
            if lift > 1: n_pass += 1
            elif lift < -1: n_fail += 1
            print(f' {len(seg):>4}|{r:>+4.0f}{mark}', end='')

        n_valid = 7 - n_low
        if n_valid >= 5 and n_pass >= 5 and n_fail <= 1:
            verdict = '★真稳定'
        elif n_valid >= 4 and n_pass >= 4:
            verdict = '○准稳'
        elif n_valid < 4:
            verdict = '段不足'
        elif n_fail >= 3:
            verdict = '✗反向'
        else:
            verdict = '— 杂'
        print(f'  {verdict:>8}')

    # 多项叠加
    print(f'\n## 多项叠加 (坤 regime 内)')
    combos = [
        ('mkt_d=巽 + mkt_m=坎', (df_e['mkt_d'] == '011') & (df_e['mkt_m'] == '010')),
        ('mkt_d=巽 + stk_m=坎', (df_e['mkt_d'] == '011') & (df_e['stk_m'] == '010')),
        ('mkt_m=坎 + stk_m=坎', (df_e['mkt_m'] == '010') & (df_e['stk_m'] == '010')),
        ('3 项: 大d=巽 + 大m=坎 + 股m=坎', (df_e['mkt_d'] == '011') & (df_e['mkt_m'] == '010') & (df_e['stk_m'] == '010')),
        ('3 项: mf深+大m坎+股m坎', (df_e['mf_30d_min'] < -200) & (df_e['mkt_m'] == '010') & (df_e['stk_m'] == '010')),
    ]
    base_full_ret = df_e['ret_30'].mean()
    base_full_zsl = (df_e['n_qian'] >= QIAN_RUN).mean() * 100
    print(f'  baseline 坤 regime: 期望 {base_full_ret:+.2f}%, 主升率 {base_full_zsl:.1f}% ({len(df_e):,})')
    print()
    print(f'  {"组合":<35} {"全 n":>6} {"全期望":>7} {"lift":>6}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 145)

    for label, mask in combos:
        sub = df_e[mask]
        if len(sub) < 100:
            print(f'  {label:<35} {len(sub):>6}  样本不足')
            continue
        ret_full = sub['ret_30'].mean()
        lift_full = ret_full - base_full_ret
        zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
        print(f'  {label:<35} {len(sub):>6,} {ret_full:>+6.2f}% {lift_full:>+5.2f}', end='')
        n_pass = 0; n_fail = 0; n_low = 0
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 20:
                n_low += 1
                print(f' {len(seg):>4}|  -- ', end='')
                continue
            r = seg['ret_30'].mean()
            base_n, base_r = seg_baselines[w[0]]
            lift = r - base_r
            mark = '✅' if lift > 1 else ('❌' if lift < -1 else '○')
            if lift > 1: n_pass += 1
            elif lift < -1: n_fail += 1
            print(f' {len(seg):>4}|{r:>+4.0f}{mark}', end='')
        n_valid = 7 - n_low
        if n_valid >= 4 and n_pass >= 4 and n_fail <= 1:
            verdict = '★真稳'
        elif n_valid < 4:
            verdict = '段不足'
        elif n_fail >= 3:
            verdict = '✗'
        else:
            verdict = '○'
        print(f'  {verdict:>5}  主升率 {zsl:.0f}%')


if __name__ == '__main__':
    main()
