# -*- coding: utf-8 -*-
"""兑 regime alpha 检查: 任何触发卦在兑 regime 是否比全市场强?

兑 regime 只有 w2_2019 和 w5_2022 两段:
  w2_2019 全市场涨, 兑 regime 任何卦都涨 — 不能算 alpha
  w5_2022 全市场跌, 看兑 regime 哪个触发卦"减亏"

如果都不"减亏", 那兑 regime 就是 case study, 不投产
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '110'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w2_2019', '2019-01-01', '2020-01-01'),
    ('w5_2022', '2022-01-01', '2023-01-01'),
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

    # 在兑 regime 内, 收集所有事件 (任何触发卦) + 按触发卦分组
    print(f'\n=== 兑 regime alpha 检查 ===')
    all_events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]
        n = e - s
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            all_events.append({
                'date': date_arr[gi], 'trigger': stk_d_arr[gi], 'ret_30': ret_30,
            })

    df_all = pd.DataFrame(all_events)
    df_all['seg'] = ''
    for w in WINDOWS:
        df_all.loc[(df_all['date'] >= w[1]) & (df_all['date'] < w[2]), 'seg'] = w[0]
    df_all = df_all[df_all['seg'] != ''].copy()

    print(f'  兑 regime 总事件: {len(df_all):,}')

    # 全市场 (任何卦) 在兑 regime 内的基准
    print(f'\n  全市场基准 (在兑 regime 内, 任何卦):')
    for w in WINDOWS:
        seg = df_all[df_all['seg'] == w[0]]
        ret = seg['ret_30'].mean()
        print(f'    {w[0]:<14} n={len(seg):>7,} 30日 {ret:>+5.2f}%')

    # 按触发卦分组
    print(f'\n  按触发卦, 跟全市场基准对比 (alpha):')
    print(f'  {"卦":<8} {"全期":>7} {"alpha全":>7}  {"w2基":>6} {"w2策":>6} {"alpha":>6}  {"w5基":>6} {"w5策":>6} {"alpha":>6}')
    for trig in GUAS:
        sub = df_all[df_all['trigger'] == trig]
        if len(sub) < 100: continue
        all_ret = sub['ret_30'].mean()
        alpha_all = all_ret - df_all['ret_30'].mean()

        seg_data = []
        for w in WINDOWS:
            base = df_all[df_all['seg'] == w[0]]['ret_30'].mean()
            stk = sub[sub['seg'] == w[0]]['ret_30'].mean() if len(sub[sub['seg'] == w[0]]) > 50 else float('nan')
            alpha = stk - base if not np.isnan(stk) else float('nan')
            seg_data.append((base, stk, alpha))

        w2 = seg_data[0]; w5 = seg_data[1]
        print(f'  {trig}{GUA_NAMES[trig]:<6} {all_ret:>+6.2f} {alpha_all:>+6.2f}  '
              f'{w2[0]:>+5.2f} {w2[1]:>+5.2f} {w2[2]:>+5.2f}  '
              f'{w5[0]:>+5.2f} {w5[1]:>+5.2f} {w5[2]:>+5.2f}')

    # 主升浪率 vs alpha
    print(f'\n  主升浪率 (n_qian≥10) 跟 alpha 关系:')
    print(f'  {"卦":<8} {"主升率%":>8} {"主升期望":>9} {"假期望":>8}')

    # 重新构建带 n_qian 的事件
    print('\n  正在统计主升浪事件...')
    detail_events = []
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
            detail_events.append({
                'date': date_arr[gi], 'trigger': stk_d_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
            })
    df_det = pd.DataFrame(detail_events)

    for trig in GUAS:
        sub = df_det[df_det['trigger'] == trig]
        if len(sub) < 100: continue
        zsl_rate = (sub['n_qian'] >= QIAN_RUN).mean() * 100
        zsl = sub[sub['n_qian'] >= QIAN_RUN]['ret_30'].mean() if (sub['n_qian'] >= QIAN_RUN).sum() > 0 else float('nan')
        fake = sub[sub['n_qian'] < QIAN_RUN]['ret_30'].mean()
        print(f'  {trig}{GUA_NAMES[trig]:<6} {zsl_rate:>7.1f} {zsl:>+8.2f} {fake:>+7.2f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
