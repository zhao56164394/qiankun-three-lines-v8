# -*- coding: utf-8 -*-
"""阶段 4: 坤 regime + 入池版 反向避雷扫描

入场池: 在池 + 坤 regime + 巽日 (27,886 事件, 5 段, 全期 +5.07%)
扫维度: 大盘 d_gua/m_gua + 个股 m_gua/y_gua + mf > 100/<-100

判定 (≥4 段 fail = 强避雷):
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '000'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

MIN_N_SEG = 100
LIFT_FAIL = -1.0
LIFT_PASS = 1.0


def main():
    t0 = time.time()
    print('=== 阶段 4: 坤 regime + 入池 反向避雷 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
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
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)

    # mf_5d / sanhu_5d
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)
    print(f'  {len(df):,} 行 (主板)')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    cross_arr = df['cross_below_11'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫入池版坤+巽事件 ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi],
                    'ret_30': ret_30,
                    'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                    'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
                    'mf': mf_arr[gi], 'mf_5d': mf5_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                })
                in_pool = False  # 出池

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()

    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret_30'].mean() if len(seg) > 0 else 0
        print(f'  baseline {w[0]:<14} n={len(seg):>6} 期望 {seg_baselines[w[0]]:>+5.2f}%')

    print(f'\n## 跨 5 段 强避雷扫描 (≥4 段 fail, ≤1 段 pass)')
    avoid_strong = []
    avoid_weak = []
    good_strong = []
    good_weak = []

    candidates = []
    # 卦象维度
    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            candidates.append((col, '==', state, f'{label_short}={state}{GUA_NAMES[state]}'))
    # 数值维度
    for thr in [-100, -50, 50, 100]:
        if thr < 0:
            candidates.append(('mf', '<', thr, f'mf<{thr}'))
            candidates.append(('mf_5d', '<', thr, f'mf_5d<{thr}'))
            candidates.append(('sanhu_5d', '<', thr, f'sanhu_5d<{thr}'))
        else:
            candidates.append(('mf', '>', thr, f'mf>{thr}'))
            candidates.append(('mf_5d', '>', thr, f'mf_5d>{thr}'))

    for col, op, thr, label in candidates:
        if op == '==':
            sub = df_e[df_e[col] == thr]
        elif op == '<':
            sub = df_e[df_e[col] < thr]
        elif op == '>':
            sub = df_e[df_e[col] > thr]
        if len(sub) < 500: continue
        n_full = len(sub); ret_full = sub['ret_30'].mean()

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

        if n_valid >= 4 and n_fail >= 4 and n_pass <= 1:
            avoid_strong.append((col, op, thr, label, n_full, ret_full, seg_results))
        elif n_valid >= 3 and n_fail >= 3 and n_pass <= 1:
            avoid_weak.append((col, op, thr, label, n_full, ret_full, seg_results))
        elif n_valid >= 4 and n_pass >= 4 and n_fail == 0:
            good_strong.append((col, op, thr, label, n_full, ret_full, seg_results))
        elif n_valid >= 3 and n_pass >= 3 and n_fail <= 1:
            good_weak.append((col, op, thr, label, n_full, ret_full, seg_results))

    def fmt_seg(seg_results):
        out = []
        for w in WINDOWS:
            r = seg_results.get(w[0])
            if r is None: out.append(f'{w[0][:2]}=NA')
            else: out.append(f'{w[0][:2]}={r[2]:+.1f}')
        return ' '.join(out)

    print(f'\n## ★★ 强避雷 (≥4 段 fail)')
    if not avoid_strong: print('  无')
    for c in avoid_strong:
        print(f'  {c[3]:<14} 全n {c[4]:>5,} 全 {c[5]:>+5.2f}%  {fmt_seg(c[6])}')

    print(f'\n## ★ 弱避雷 (≥3 段 fail)')
    if not avoid_weak: print('  无')
    for c in avoid_weak:
        print(f'  {c[3]:<14} 全n {c[4]:>5,} 全 {c[5]:>+5.2f}%  {fmt_seg(c[6])}')

    print(f'\n## ★★ 强好规律 (≥4 段 pass, 0 段 fail)')
    if not good_strong: print('  无')
    for c in good_strong:
        print(f'  {c[3]:<14} 全n {c[4]:>5,} 全 {c[5]:>+5.2f}%  {fmt_seg(c[6])}')

    print(f'\n## ★ 弱好规律 (≥3 段 pass)')
    if not good_weak: print('  无')
    for c in good_weak:
        print(f'  {c[3]:<14} 全n {c[4]:>5,} 全 {c[5]:>+5.2f}%  {fmt_seg(c[6])}')

    # 验证强避雷 union 效果
    if avoid_strong:
        print(f'\n## 强避雷 union 效果')
        avoid_mask = pd.Series(False, index=df_e.index)
        for col, op, thr, _, _, _, _ in avoid_strong:
            if op == '==':
                avoid_mask = avoid_mask | (df_e[col] == thr)
            elif op == '<':
                avoid_mask = avoid_mask | (df_e[col] < thr)
            elif op == '>':
                avoid_mask = avoid_mask | (df_e[col] > thr)
        keep = df_e[~avoid_mask]
        base = df_e['ret_30'].mean()
        print(f'  剩 {len(keep):,} ({len(keep)/len(df_e)*100:.0f}%)')
        print(f'  期望: {keep["ret_30"].mean():+.2f}% (vs {base:+.2f}%, lift {keep["ret_30"].mean()-base:+.2f})')
        print(f'  胜率: {(keep["ret_30"]>0).mean()*100:.1f}%')

        n_pos = 0; n_seg = 0
        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 50: continue
            n_seg += 1
            b = seg_b['ret_30'].mean()
            k = seg_k['ret_30'].mean() if len(seg_k) > 0 else float('nan')
            diff = k - b
            mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
            if not np.isnan(k) and k > 0: n_pos += 1
            print(f'  {w[0]:<14} 全 {b:>+5.2f}% → {k:>+5.2f}% ({len(seg_k):>4}) lift {diff:>+5.2f} {mark}')
        print(f'  段稳 (避雷后): {n_pos}/{n_seg}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
