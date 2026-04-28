# -*- coding: utf-8 -*-
"""验证: 乾 regime 里 8 个 stk_d 触发卦哪个最强?

主升浪起点前一日卦象分布 (从 test80):
  011巽 60.6%  ← 我们一直默认用这个
  101离 13.3%
  010坎 9.0%
  100震 5.0%
  110兑 4.7%
  000坤 4.4%
  001艮 2.9%

但占比高 ≠ 收益高! 占比高可能是因为巽日数量多, 不代表巽日入场后赢面大

本扫描: 在乾 regime 内, 各 stk_d 触发卦
  - 30 日固定收益 (baseline)
  - 主升率
  - 走势跨段稳定性
  - 不同卦 30 日期望对比
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

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

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
                        columns=['date', 'code', 'd_gua'])
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
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫乾 regime 8 个 stk_d 触发卦 ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            events.append({
                'date': date_arr[gi],
                'stk_d': stk_d_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
            })
    df_e = pd.DataFrame(events)
    print(f'  乾 regime 全部事件: {len(df_e):,}')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()

    print(f'\n## 8 个 stk_d 触发卦 baseline (乾 regime)')
    print(f'  {"卦":<10} {"n":>8} {"30日%":>7} {"主升率%":>8} {"段稳/可用段":>11}')
    rows = []
    for state in GUAS:
        sub = df_e[df_e['stk_d'] == state]
        if len(sub) == 0: continue
        n = len(sub); ret = sub['ret_30'].mean()
        zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100

        n_pos = 0; n_seg = 0
        seg_rets = {}
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 200:
                seg_rets[w[0]] = None
                continue
            sret = seg['ret_30'].mean()
            seg_rets[w[0]] = sret
            n_seg += 1
            if sret > 0: n_pos += 1
        label = f'{state}{GUA_NAMES[state]}'
        print(f'  {label:<10} {n:>8,} {ret:>+6.2f} {zsl:>7.1f} {n_pos}/{n_seg}')
        rows.append((state, n, ret, zsl, n_pos, n_seg, seg_rets))

    print(f'\n## 各卦详细段表现')
    print(f'  {"卦":<10} ', end='')
    for w in WINDOWS:
        print(f'{w[0]:>10} ', end='')
    print()
    for state, n, ret, zsl, n_pos, n_seg, seg_rets in rows:
        label = f'{state}{GUA_NAMES[state]}'
        print(f'  {label:<10} ', end='')
        for w in WINDOWS:
            r = seg_rets.get(w[0])
            if r is None: print(f'{"--":>10} ', end='')
            else: print(f'{r:>+9.2f} ', end='')
        print()

    # 各卦主升浪起点数量 + 主升率
    print(f'\n## 各 stk_d 卦的"乾连续 ≥10日"主升浪起点数 (在乾 regime 内)')
    runs = {}
    for state in GUAS:
        runs[state] = 0

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 5: continue
        gua = stk_d_arr[s:e]
        n = len(gua)
        i = 0
        while i < n:
            if gua[i] != '111':
                i += 1; continue
            j = i
            while j < n and gua[j] == '111':
                j += 1
            length = j - i
            gi = s + i
            if length >= QIAN_RUN and mkt_y_arr[gi] == REGIME_Y and i > 0:
                prev_gua = gua[i-1]
                if prev_gua in runs:
                    runs[prev_gua] += 1
            i = j

    print(f'  {"卦":<10} {"主升浪起点数":>13} {"贡献率":>8}')
    total_runs = sum(runs.values())
    for state in GUAS:
        n = runs.get(state, 0)
        ratio = n / total_runs * 100 if total_runs > 0 else 0
        label = f'{state}{GUA_NAMES[state]}'
        print(f'  {label:<10} {n:>13,} {ratio:>7.1f}%')


if __name__ == '__main__':
    main()
