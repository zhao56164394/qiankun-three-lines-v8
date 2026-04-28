# -*- coding: utf-8 -*-
"""Step 26 — 坤 regime 反向避雷扫描

只看大盘 y_gua=000 期间的巽日, 找跨期稳定差的单维条件.

设计:
  - 6 卦类 × 8 态 = 48 个单维过滤条件
  - 对每个条件, 在坤 regime 内 7 段算 ret_30
  - 跨期稳差: ≥4 段有效 + ≥3 段 lift < -1%
  - 得到坤 regime 的"避雷名单"
  - 用避雷过滤后, 看剩余事件 lift / walk-forward
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
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]

MIN_N_SEG = 100   # 段内最少样本
LIFT_FAIL = -1.0  # lift < -1% 算 fail


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

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
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
            })
    df_e = pd.DataFrame(events)
    print(f'  坤 regime 巽日: {len(df_e):,}, {time.time()-t1:.1f}s')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  打段后: {len(df_e):,}')

    # 段 baseline
    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = (len(seg), seg['ret_30'].mean() if len(seg) > 0 else 0)

    print(f'\n## 坤 regime × 7 段 baseline')
    for w in WINDOWS:
        n, b = seg_baselines[w[0]]
        print(f'  {w[0]:<14} n={n:>6,}  期望 {b:>+5.2f}%')

    # === 单维候选扫描 ===
    print(f'\n## 坤 regime 内 单维条件跨期表现 (找避雷)')
    print(f'  {"卦+态":<14} {"全 n":>6} {"全期望":>7}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 130)

    avoid_list = []
    good_list = []
    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 500: continue
            n_full = len(sub); ret_full = sub['ret_30'].mean()
            label = f'{label_short}={state}{GUA_NAMES[state]}'
            print(f'  {label:<14} {n_full:>6,} {ret_full:>+6.2f}%', end='')

            n_fail = 0; n_pass = 0; n_low = 0
            for w in WINDOWS:
                seg = sub[sub['seg'] == w[0]]
                if len(seg) < MIN_N_SEG:
                    n_low += 1
                    print(f' {len(seg):>4}|  -- ', end='')
                    continue
                ret = seg['ret_30'].mean()
                base_n, base_r = seg_baselines[w[0]]
                lift = ret - base_r
                if lift <= LIFT_FAIL:
                    n_fail += 1; mark = '❌'
                elif lift >= -LIFT_FAIL:
                    n_pass += 1; mark = '✅'
                else:
                    mark = '○'
                print(f' {len(seg):>4}|{ret:>+4.0f}{mark}', end='')

            n_valid = 7 - n_low
            if n_valid >= 4 and n_fail >= 3 and n_pass <= 1:
                verdict = '★避雷'
                avoid_list.append((col, state, label, n_full, ret_full, n_fail, n_valid))
            elif n_valid >= 4 and n_pass >= 3 and n_fail <= 1:
                verdict = '★好信号'
                good_list.append((col, state, label, n_full, ret_full, n_pass, n_valid))
            elif n_valid < 4:
                verdict = '段不足'
            else:
                verdict = '— 杂'
            print(f'  {verdict:>8}')

    # === 汇总 ★ 避雷 ===
    print(f'\n## ★ 坤 regime 跨期稳避雷条件')
    if not avoid_list:
        print('  无!')
    else:
        for col, state, label, n, ret, nf, nv in avoid_list:
            print(f'  {label:<14} {n:>6,} {ret:>+6.2f}% (fail {nf}/{nv})')

    print(f'\n## ★ 坤 regime 跨期稳好条件')
    if not good_list:
        print('  无!')
    else:
        for col, state, label, n, ret, np_, nv in good_list:
            print(f'  {label:<14} {n:>6,} {ret:>+6.2f}% (pass {np_}/{nv})')

    # === 用避雷过滤后效果 ===
    if avoid_list:
        print(f'\n## 用 ★ 避雷条件 union 过滤')
        avoid_mask = pd.Series(False, index=df_e.index)
        for col, state, _, _, _, _, _ in avoid_list:
            avoid_mask = avoid_mask | (df_e[col] == state)
        keep = df_e[~avoid_mask]
        base_full = df_e['ret_30'].mean()
        print(f'  剩 {len(keep):,} ({len(keep)/len(df_e)*100:.0f}%)')
        print(f'  期望: {keep["ret_30"].mean():+.2f}% (vs {base_full:+.2f}%, lift {keep["ret_30"].mean()-base_full:+.2f})')
        print(f'  主升率: {(keep["n_qian"]>=QIAN_RUN).mean()*100:.1f}%')

        # walk-forward
        print(f'\n  walk-forward:')
        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 50:
                print(f'    {w[0]:<14}  样本不足 ({len(seg_b)})')
                continue
            b = seg_b['ret_30'].mean(); k = seg_k['ret_30'].mean() if len(seg_k) > 0 else float('nan')
            diff = k - b
            mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
            print(f'    {w[0]:<14} 全 {b:>+5.2f}%, 避雷后 {k:>+5.2f}% ({len(seg_k):>5}), lift {diff:>+5.2f} {mark}')


if __name__ == '__main__':
    main()
