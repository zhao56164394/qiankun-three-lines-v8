# -*- coding: utf-8 -*-
"""离 regime alpha 检查: 巽日 / 坤日 / 震日 哪个跟全市场 alpha 最强

离 regime 全市场 baseline +2.96% (6 段都有数据), alpha 空间不大
看 8 触发卦的 alpha 全期 + 各段
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '101'

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

    print(f'\n=== 离 regime alpha 检查 (跟全市场 30 日对比) ===')
    all_events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = e - s
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            all_events.append({
                'date': date_arr[gi], 'trigger': stk_d_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
            })

    df_all = pd.DataFrame(all_events)
    df_all['seg'] = ''
    for w in WINDOWS:
        df_all.loc[(df_all['date'] >= w[1]) & (df_all['date'] < w[2]), 'seg'] = w[0]
    df_all = df_all[df_all['seg'] != ''].copy()

    print(f'  离 regime 总事件: {len(df_all):,}')
    print(f'  全样本 baseline: {df_all["ret_30"].mean():+.2f}%')

    # 按触发卦 + 各段 alpha
    print(f'\n  按触发卦 各段 alpha (vs 全市场基准):')
    print(f'  {"卦":<8} {"全期":>7} {"alpha":>6}  {"w2":>6} {"w3":>6} {"w4":>6} {"w5":>6} {"w6":>6} {"w7":>6} {"段稳":>5}')
    seg_baselines = {}
    for w in WINDOWS:
        seg_baselines[w[0]] = df_all[df_all['seg'] == w[0]]['ret_30'].mean()

    for trig in GUAS:
        sub = df_all[df_all['trigger'] == trig]
        if len(sub) < 100: continue
        all_ret = sub['ret_30'].mean()
        alpha_all = all_ret - df_all['ret_30'].mean()

        alphas = []
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 50:
                alphas.append(float('nan'))
                continue
            stk = seg['ret_30'].mean()
            base = seg_baselines[w[0]]
            alphas.append(stk - base)
        n_pos = sum(1 for a in alphas if not np.isnan(a) and a > 0)
        n_seg = sum(1 for a in alphas if not np.isnan(a))
        seg_str = ' '.join(f'{a:>+5.2f}' if not np.isnan(a) else '   --' for a in alphas)
        print(f'  {trig}{GUA_NAMES[trig]:<6} {all_ret:>+6.2f} {alpha_all:>+6.2f}  {seg_str}  {n_pos}/{n_seg}')

    # 主升率 vs alpha
    print(f'\n  主升浪率/期望 (n_qian≥10) 跟 alpha 关系:')
    print(f'  {"卦":<8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8} {"全期 alpha":>10}')
    for trig in GUAS:
        sub = df_all[df_all['trigger'] == trig]
        if len(sub) < 100: continue
        zsl_rate = (sub['n_qian'] >= QIAN_RUN).mean() * 100
        zsl = sub[sub['n_qian'] >= QIAN_RUN]['ret_30'].mean() if (sub['n_qian'] >= QIAN_RUN).sum() > 0 else float('nan')
        fake = sub[sub['n_qian'] < QIAN_RUN]['ret_30'].mean()
        alpha_all = sub['ret_30'].mean() - df_all['ret_30'].mean()
        print(f'  {trig}{GUA_NAMES[trig]:<6} {zsl_rate:>7.1f} {zsl:>+8.2f} {fake:>+7.2f} {alpha_all:>+9.2f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
