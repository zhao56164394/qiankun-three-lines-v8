# -*- coding: utf-8 -*-
"""Step 21 — 8 regime 最佳组合 walk-forward 验证

验证 Step 20 找到的 8 个 (mkt_y, mkt_m, stk_m) "最佳" 是不是切片福利

8 候选 (来自 Step 20):
  000坤 + (mkt_m=100震, stk_m=100震)  — IS +13.90%, 1675
  001艮 + (111乾, 000坤)              — IS +0.04%, 552
  010坎 + (011巽, 011巽)              — IS +27.48%, 466
  011巽 + (100震, 110兑)              — IS +3.54%, 683
  100震 + (100震, 100震)              — IS +10.51%, 1790
  101离 + (011巽, 111乾)              — IS +10.89%, 457
  110兑 + (000坤, 000坤)              — IS +12.07%, 1443
  111乾 + (010坎, 110兑)              — IS +12.50%, 486

每候选拆 7 段:
  - 段内 baseline (该 regime 该段巽日均期望)
  - 段内候选 期望
  - lift = 候选 - baseline
  - 段内候选 n

判定:
  ★ 真规律: ≥5 段 n>=20 + ≥4 段 lift>+2%
  ✗ 切片: 集中在 1-2 段
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

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]

CANDIDATES = [
    ('000', '100', '100', 13.90),
    ('001', '111', '000', 0.04),
    ('010', '011', '011', 27.48),
    ('011', '100', '110', 3.54),
    ('100', '100', '100', 10.51),
    ('101', '011', '111', 10.89),
    ('110', '000', '000', 12.07),
    ('111', '010', '110', 12.50),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g['stk_m'] = g['m_gua'].astype(str).str.zfill(3)
    g = g[['date', 'code', 'd_gua', 'stk_m']]

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'd_gua', 'mkt_m']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    gua_arr = df['d_gua'].to_numpy()
    stk_m = df['stk_m'].to_numpy()
    mkt_m = df['mkt_m'].to_numpy()
    mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫巽日, 仅记录必要字段
    print(f'\n=== 扫巽日 ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = gua_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            gi = s + i
            events.append({
                'date': date_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_y': mkt_y[gi], 'mkt_m': mkt_m[gi], 'stk_m': stk_m[gi],
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日: {len(df_e):,}, {time.time()-t1:.1f}s')

    # 打段
    df_e['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label
    df_e = df_e[df_e['seg'] != ''].copy()

    # === 8 候选 walk-forward ===
    print(f'\n## 8 regime 最佳组合 walk-forward')
    print(f'  {"组合 (regime|mkt_m|stk_m)":<28} {"全 IS":>7} {"全 n":>5}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 130)

    for y, mm, sm, is_exp in CANDIDATES:
        sub = df_e[(df_e['mkt_y'] == y) & (df_e['mkt_m'] == mm) & (df_e['stk_m'] == sm)]
        n_full = len(sub)
        ret_full = sub['ret_30'].mean() if n_full > 0 else 0
        label = f'{y}{GUA_NAMES[y]}|{mm}{GUA_NAMES[mm]}|{sm}{GUA_NAMES[sm]}'
        print(f'  {label:<28} {ret_full:>+6.2f}% {n_full:>5}', end='')

        # 该 regime 段内 baseline (用所有该 regime 的巽日)
        regime_sub = df_e[df_e['mkt_y'] == y]

        n_pass = 0; n_fail = 0; n_low = 0
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            n_seg = len(seg)
            if n_seg < 20:
                n_low += 1
                print(f' {n_seg:>3}|  -- ', end='')
                continue
            seg_ret = seg['ret_30'].mean()
            # 段内 regime baseline
            seg_base_sub = regime_sub[regime_sub['seg'] == w[0]]
            seg_base = seg_base_sub['ret_30'].mean() if len(seg_base_sub) > 0 else 0
            lift = seg_ret - seg_base
            mark = '✅' if lift > 2 else ('❌' if lift < -2 else '○')
            if lift > 2: n_pass += 1
            elif lift < -2: n_fail += 1
            print(f' {n_seg:>3}|{seg_ret:>+4.0f}{mark}', end='')

        n_valid = 7 - n_low
        if n_valid >= 5 and n_pass >= 5 and n_fail <= 1:
            verdict = '★真规律'
        elif n_valid >= 4 and n_pass >= 4:
            verdict = '○准稳'
        elif n_valid < 4:
            verdict = '段不足'
        elif n_fail >= 3:
            verdict = '✗反向'
        else:
            verdict = '— 杂'
        print(f'  {verdict:>8}')

    # === 简化版: 强 regime (4 个) baseline 跨期稳定性 ===
    print(f'\n\n## 简化: 4 强 regime baseline (该 regime 任意巽日) 跨期稳定性')
    print(f'  {"regime":<14}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"全均":>7}')
    print('  ' + '-' * 100)
    for y, label in [('000', '000坤 (4.10%)'), ('010', '010坎 (3.34%)'),
                      ('100', '100震 (1.16%)'), ('111', '111乾 (1.70%)')]:
        sub = df_e[df_e['mkt_y'] == y]
        print(f'  {label:<14}', end='')
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 30:
                print(f' {len(seg):>4}|  -- ', end='')
            else:
                print(f' {len(seg):>4}|{seg["ret_30"].mean():>+4.1f}', end='')
        print(f'  {sub["ret_30"].mean():>+5.2f}%')

    # === 看每个 regime 的"段内最佳 m 组合"是不是同一个 ===
    # 即: 在 7 段中, 各段最佳 (mkt_m, stk_m) 是不是稳定?
    print(f'\n\n## 每 regime × 每段 最佳 (mkt_m, stk_m) 看稳定性')
    for y in ['000', '010', '100', '111']:
        sub = df_e[df_e['mkt_y'] == y]
        if len(sub) < 1000: continue
        print(f'\n  ## regime y={y}{GUA_NAMES[y]}')
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 200:
                print(f'  {w[0]:<14}  样本不足 ({len(seg)})')
                continue
            # 段内最佳 (mkt_m, stk_m)
            grp = seg.groupby(['mkt_m', 'stk_m']).agg(n=('ret_30', 'size'), ret=('ret_30', 'mean'))
            grp = grp[grp['n'] >= 30].sort_values('ret', ascending=False)
            if len(grp) == 0:
                print(f'  {w[0]:<14}  无足够细分')
                continue
            top = grp.head(3)
            line = f'  {w[0]:<14} '
            for (mm, sm), r in top.iterrows():
                line += f'{mm}{GUA_NAMES[mm]}|{sm}{GUA_NAMES[sm]}({int(r["n"])},{r["ret"]:+.0f}%)  '
            print(line)


if __name__ == '__main__':
    main()
