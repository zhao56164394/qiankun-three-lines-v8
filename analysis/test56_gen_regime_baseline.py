# -*- coding: utf-8 -*-
"""阶段 2: 艮 regime (mkt_y=001) baseline 扫描

输出:
  - 艮 regime 巽日 baseline 期望 (30 日)
  - 主升浪率
  - 7 段 walk-forward 稳定性
  - 样本量
  - 同时对比 8 regime 全景, 确认艮在哪一档

用于决策: 艮 regime 是否值得做主升浪研究
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
TRIGGER_GUA = '011'   # 巽日触发 (反转卦)

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

    # 扫所有巽日 (在所有 regime 下)
    print(f'\n=== 扫巽日 d_gua=011 (评估窗口 {EVAL_WIN} 日) ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            events.append({
                'date': date_arr[gi],
                'mkt_y': mkt_y_arr[gi],
                'n_qian': int(n_qian),
                'ret_30': ret_30,
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日事件: {len(df_e):,}')

    # 标段
    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()

    # 8 regime 全景
    print(f'\n## 8 regime 巽日全景 (n_qian>={QIAN_RUN}=主升浪)')
    print(f'  {"regime":<10} {"n":>8} {"期望%":>7} {"主升率%":>8} {"段稳":>6}')
    rows = []
    for r in GUAS:
        sub = df_e[df_e['mkt_y'] == r]
        if len(sub) == 0: continue
        n = len(sub); ret = sub['ret_30'].mean()
        zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
        # 段稳 = ret > 0 的段数 / 有数据的段数
        n_pos = 0; n_seg = 0
        seg_details = {}
        for w in WINDOWS:
            seg = sub[sub['seg'] == w[0]]
            if len(seg) < 50:
                seg_details[w[0]] = (len(seg), float('nan'))
                continue
            sret = seg['ret_30'].mean()
            seg_details[w[0]] = (len(seg), sret)
            n_seg += 1
            if sret > 0: n_pos += 1
        label = f'{r}{GUA_NAMES[r]}'
        print(f'  {label:<10} {n:>8,} {ret:>+6.2f} {zsl:>7.1f} {n_pos}/{n_seg}')
        rows.append((r, n, ret, zsl, n_pos, n_seg, seg_details))

    # 艮 regime 详细
    print(f'\n## 艮 regime (mkt_y=001) walk-forward 详细')
    sub_gen = df_e[df_e['mkt_y'] == '001']
    print(f'  总样本: {len(sub_gen):,}')
    if len(sub_gen) > 0:
        print(f'  全期期望: {sub_gen["ret_30"].mean():+.2f}%')
        print(f'  主升率: {(sub_gen["n_qian"]>=QIAN_RUN).mean()*100:.1f}%')
        print(f'\n  段详情:')
        print(f'  {"seg":<14} {"n":>8} {"期望%":>7} {"主升率%":>8}')
        for w in WINDOWS:
            seg = sub_gen[sub_gen['seg'] == w[0]]
            if len(seg) < 50:
                print(f'  {w[0]:<14} {len(seg):>8,} {"--":>7} {"--":>8}')
                continue
            ret = seg['ret_30'].mean()
            zsl = (seg['n_qian'] >= QIAN_RUN).mean() * 100
            print(f'  {w[0]:<14} {len(seg):>8,} {ret:>+6.2f} {zsl:>7.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
