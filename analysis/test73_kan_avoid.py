# -*- coding: utf-8 -*-
"""阶段 4: 坎 regime (mkt_y=010) 反向避雷扫描

样本: 坎 regime 巽日 100K, 跨 4 段:
  w1_2018:  -5.46% (熊底), n=16006
  w2_2019: +23.55% (反弹), n=9624
  w5_2022: +15.52% (短暂坎), n=754
  w6_2023_24: +2.54% (主战场), n=69836

避雷判定 (跨 4 段都 fail):
  - 段内 lift ≤ -1% 算 fail
  - ≥3 段 fail 且 ≤1 段 pass (lift ≥ +1%) → 真避雷
  - 段内最少 50 样本

注意: w5 仅 754 样本, MIN_N_SEG 不设太高
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '010'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

MIN_N_SEG = 50
LIFT_FAIL = -1.0
LIFT_PASS = 1.0


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

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫坎 regime 巽日事件 ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
            })
    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  四段内事件: {len(df_e):,}')

    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret_30'].mean() if len(seg) > 0 else 0
        print(f'  baseline {w[0]:<14} n={len(seg):>6}  期望 {seg_baselines[w[0]]:>+5.2f}%')

    # 单维候选扫描
    print(f'\n## 跨段稳定避雷扫描 (≥{MIN_N_SEG} n/段, lift ≤ {LIFT_FAIL}%)')
    avoid_strong = []   # ≥3 段 fail 且 ≤1 段 pass
    avoid_weak = []     # 2 段 fail 且 0 段 pass
    good_strong = []    # ≥3 段 pass
    good_weak = []      # 2 段 pass 且 0 段 fail

    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 200: continue
            n_full = len(sub); ret_full = sub['ret_30'].mean()
            label = f'{label_short}={state}{GUA_NAMES[state]}'

            seg_results = {}
            for w in WINDOWS:
                seg = sub[sub['seg'] == w[0]]
                if len(seg) < MIN_N_SEG:
                    seg_results[w[0]] = None
                    continue
                ret = seg['ret_30'].mean()
                lift = ret - seg_baselines[w[0]]
                seg_results[w[0]] = (len(seg), ret, lift)

            n_pass = sum(1 for r in seg_results.values() if r and r[2] >= LIFT_PASS)
            n_fail = sum(1 for r in seg_results.values() if r and r[2] <= LIFT_FAIL)
            n_valid = sum(1 for r in seg_results.values() if r is not None)

            if n_valid >= 3 and n_fail >= 3 and n_pass <= 1:
                avoid_strong.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid >= 2 and n_fail >= 2 and n_pass == 0:
                avoid_weak.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid >= 3 and n_pass >= 3 and n_fail == 0:
                good_strong.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid >= 2 and n_pass >= 2 and n_fail == 0:
                good_weak.append((col, state, label, n_full, ret_full, seg_results))

    def fmt_seg(seg_results):
        s = []
        for w in WINDOWS:
            r = seg_results.get(w[0])
            if r is None: s.append(f'{w[0]}=NA')
            else: s.append(f'{w[0]}=n{r[0]} lift{r[2]:+.2f}')
        return ' | '.join(s)

    print(f'\n## ★★ 强避雷 (≥3 段 fail, ≤1 段 pass)')
    if not avoid_strong: print('  无')
    for col, state, label, n, ret, seg in avoid_strong:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    print(f'\n## ★ 弱避雷 (2 段 fail, 0 段 pass)')
    if not avoid_weak: print('  无')
    for col, state, label, n, ret, seg in avoid_weak:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    print(f'\n## ★★ 强好规律 (≥3 段 pass, 0 段 fail)')
    if not good_strong: print('  无')
    for col, state, label, n, ret, seg in good_strong:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    print(f'\n## ★ 弱好规律 (2 段 pass, 0 段 fail)')
    if not good_weak: print('  无')
    for col, state, label, n, ret, seg in good_weak:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    # 验证强避雷效果
    if avoid_strong:
        print(f'\n## 强避雷 union 效果')
        avoid_mask = pd.Series(False, index=df_e.index)
        for col, state, _, _, _, _ in avoid_strong:
            avoid_mask = avoid_mask | (df_e[col] == state)
        keep = df_e[~avoid_mask]
        base = df_e['ret_30'].mean()
        kept_ret = keep['ret_30'].mean()
        kept_zsl = (keep['n_qian'] >= QIAN_RUN).mean() * 100
        print(f'  剩 {len(keep):,} ({len(keep)/len(df_e)*100:.0f}%)')
        print(f'  期望: {kept_ret:+.2f}% (vs {base:+.2f}%, lift {kept_ret-base:+.2f})')
        print(f'  主升率: {kept_zsl:.1f}%')

        print(f'\n  walk-forward:')
        n_pass = 0
        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 50: continue
            b = seg_b['ret_30'].mean(); k = seg_k['ret_30'].mean() if len(seg_k) > 0 else float('nan')
            diff = k - b
            mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
            if diff > 0.5: n_pass += 1
            print(f'    {w[0]:<14} 全 {b:>+5.2f}%, 避雷后 {k:>+5.2f}% ({len(seg_k):>5}), lift {diff:>+5.2f} {mark}')
        print(f'  → {n_pass}/4 段 lift > +0.5%')


if __name__ == '__main__':
    main()
