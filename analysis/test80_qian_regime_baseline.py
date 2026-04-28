# -*- coding: utf-8 -*-
"""阶段 2: 乾 regime (mkt_y=111) baseline + 主升浪事件分布

乾 = 大盘三尺度全强 = 抱团/上涨期
skill 表 baseline +1.70%, 不达标但样本巨大
"""
import os, sys, io, time
import numpy as np
import pandas as pd
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '111'  # 乾
TRIGGER_GUA = '011'

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
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # ===== 乾 regime 巽日 baseline =====
    print(f'\n=== 乾 regime ({REGIME_Y}) 巽日 baseline ===')
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
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日事件: {len(df_e):,}')

    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()

    print(f'\n  全期期望: {df_e["ret_30"].mean():+.2f}%')
    print(f'  主升率: {(df_e["n_qian"]>=QIAN_RUN).mean()*100:.1f}%')
    print(f'\n  walk-forward 段详情:')
    print(f'  {"seg":<14} {"n":>8} {"期望%":>7} {"主升率%":>8}')
    n_pos = 0; n_seg = 0
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        if len(seg) < 50:
            print(f'  {w[0]:<14} {len(seg):>8,} {"--":>7} {"--":>8}')
            continue
        ret = seg['ret_30'].mean()
        zsl = (seg['n_qian'] >= QIAN_RUN).mean() * 100
        print(f'  {w[0]:<14} {len(seg):>8,} {ret:>+6.2f} {zsl:>7.1f}')
        n_seg += 1
        if ret > 0: n_pos += 1
    print(f'\n  段稳: {n_pos}/{n_seg}')

    # ===== 主升浪事件分布 =====
    print(f'\n=== 乾 regime 主升浪 (≥{QIAN_RUN}日 乾连续) 事件分布 ===')
    runs = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 5: continue
        gua = stk_d_arr[s:e]; cl = close_arr[s:e]
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
            if length >= QIAN_RUN and mkt_y_arr[gi] == REGIME_Y:
                prev = gua[i-1] if i > 0 else 'NULL'
                if i + 30 < n:
                    ret_30 = (cl[i+30] / cl[i] - 1) * 100
                else:
                    ret_30 = float('nan')
                runs.append({
                    'start': date_arr[gi], 'length': length,
                    'prev_gua': prev, 'ret_30': ret_30,
                })
            i = j
    df_r = pd.DataFrame(runs)
    print(f'  主升浪起点: {len(df_r):,}')
    if len(df_r) > 0:
        print(f'  平均长度: {df_r["length"].mean():.1f} 日')
        print(f'  起点 30 日均收益: {df_r["ret_30"].mean():+.2f}%')

        df_r['seg'] = ''
        for w in WINDOWS:
            df_r.loc[(df_r['start'] >= w[1]) & (df_r['start'] < w[2]), 'seg'] = w[0]

        print(f'\n  按段分布:')
        print(f'  {"seg":<14} {"事件":>6} {"30日%":>7}')
        for w in WINDOWS:
            seg = df_r[df_r['seg'] == w[0]]
            if len(seg) == 0:
                print(f'  {w[0]:<14} {0:>6} {"--":>7}')
                continue
            print(f'  {w[0]:<14} {len(seg):>6} {seg["ret_30"].mean():>+6.2f}')

        print(f'\n  起点前一日卦象分布:')
        prev_cnt = Counter(df_r['prev_gua'])
        for prev, cnt in prev_cnt.most_common():
            ratio = cnt / len(df_r) * 100
            label = f'{prev}{GUA_NAMES.get(prev, "?")}' if prev in GUA_NAMES else prev
            print(f'    {label:<10} {cnt:>5} ({ratio:.1f}%)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
