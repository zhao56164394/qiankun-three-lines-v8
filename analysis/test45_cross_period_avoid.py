# -*- coding: utf-8 -*-
"""Step 22 — 跨期稳定避雷扫描

思路: 不"选最强组合" (会切片), 而是"删跨期都差的"

对单维卦象每个态 (6 卦类 × 8 态 = 48 个候选 filter), 算:
  - 巽日中, 该态发生时, 7 段各段的期望 ret
  - 跨期稳差: ≥5 段有效 + ≥4 段 ret < baseline-2

判定 ★: 该态在多数段都拖累期望, 加进永久避雷名单
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
ZSL_THRESH = 10

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

MIN_N_SEG = 200    # 段内最少样本 (足够稳定地估期望)
LIFT_FAIL = -1.0   # 段内 ret 差于 baseline 多少算 fail


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

    # 扫巽日
    print(f'\n=== 扫巽日 ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            gi = s + i
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_y': mkt_y_arr[gi], 'mkt_m': mkt_m_arr[gi], 'mkt_d': mkt_d_arr[gi],
                'stk_y': stk_y_arr[gi], 'stk_m': stk_m_arr[gi],
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日: {len(df_e):,}, {time.time()-t1:.1f}s')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()

    # 各段 baseline
    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret_30'].mean() if len(seg) > 0 else 0
    print(f'\n## 段 baseline (该段所有巽日均期望)')
    for w in WINDOWS:
        print(f'  {w[0]:<14}  baseline {seg_baselines[w[0]]:>+5.2f}%')

    # === 6 维度 × 8 态 跨期稳定避雷扫描 ===
    print(f'\n## 跨期稳定避雷扫描 (≥{MIN_N_SEG} n/段, lift < baseline {LIFT_FAIL}%)')
    print(f'  {"卦+态":<14} {"全n":>6} {"全期望":>7}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 130)

    avoid_list = []
    for col, label_short in [('mkt_y', '大y'), ('mkt_m', '大m'), ('mkt_d', '大d'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 1000: continue
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
                base = seg_baselines[w[0]]
                lift = ret - base
                if lift <= LIFT_FAIL:
                    n_fail += 1; mark = '❌'
                elif lift >= -LIFT_FAIL:
                    n_pass += 1; mark = '✅'
                else:
                    mark = '○'
                print(f' {len(seg):>4}|{ret:>+4.0f}{mark}', end='')

            n_valid = 7 - n_low
            if n_valid >= 5 and n_fail >= 4 and n_pass <= 1:
                verdict = '★避雷'
                avoid_list.append((label, ret_full, n_full, n_fail, n_valid))
            elif n_valid >= 5 and n_pass >= 4 and n_fail <= 1:
                verdict = '★好信号'
            elif n_valid < 5:
                verdict = '段不足'
            else:
                verdict = '— 杂'
            print(f'  {verdict:>8}')

    # === 汇总 ★ 避雷 ===
    print(f'\n## ★ 跨期稳避雷条件 (≥5 段有效 + ≥4 段 lift<{LIFT_FAIL}%)')
    if not avoid_list:
        print('  无! 没有跨期稳差的单维条件')
    else:
        print(f'  {"卦+态":<14} {"全n":>6} {"全期望":>7} {"段fail/有效":>12}')
        for label, ret, n, n_f, n_v in avoid_list:
            print(f'  {label:<14} {n:>6,} {ret:>+6.2f}% {n_f}/{n_v}')

    # === 用 ★ 避雷条件组合, 看效果 ===
    if avoid_list:
        print(f'\n## 用 ★ 避雷组合: 满足任一即跳过')
        # 解析 label
        avoid_mask = pd.Series(False, index=df_e.index)
        for label, _, _, _, _ in avoid_list:
            # label 形如 "大y=011巽" → col='mkt_y', state='011'
            col_short, gua_part = label.split('=')
            state = gua_part[:3]
            col_map = {'大y': 'mkt_y', '大m': 'mkt_m', '大d': 'mkt_d', '股y': 'stk_y', '股m': 'stk_m'}
            col = col_map[col_short]
            avoid_mask = avoid_mask | (df_e[col] == state)

        keep = df_e[~avoid_mask]
        print(f'  剩 {len(keep):,} ({len(keep)/len(df_e)*100:.0f}% of 巽日)')
        print(f'  期望: {keep["ret_30"].mean():+.2f}% (vs 巽日 {df_e["ret_30"].mean():+.2f}%, lift {keep["ret_30"].mean()-df_e["ret_30"].mean():+.2f})')
        print(f'  主升率: {(keep["n_qian"]>=ZSL_THRESH).mean()*100:.1f}%')

        # walk-forward
        print(f'\n  walk-forward:')
        n_pass = 0
        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 100 or len(seg_k) < 100: continue
            b_ret = seg_b['ret_30'].mean(); k_ret = seg_k['ret_30'].mean()
            diff = k_ret - b_ret
            mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
            if diff > 0.5: n_pass += 1
            print(f'    {w[0]:<14} 巽日 {b_ret:>+5.2f}%, 避雷后 {k_ret:>+5.2f}%, lift {diff:>+5.2f} {mark}')
        print(f'  → {n_pass}/7 段 lift > +0.5%')


if __name__ == '__main__':
    main()
