# -*- coding: utf-8 -*-
"""阶段 3a: w4_2021 内艮 regime (mkt_y=001) 时间跨度

输出:
  - 艮 regime 在 2021 年具体哪些日期出现
  - 138 个主升浪起点的日期分布 (按月)
  - 用于决定 IS/OOS 时间拆分点
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print('=== 加载 ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        market[c] = market[c].astype(str).str.zfill(3)

    # w4_2021 期间艮 regime 日期范围
    m2021 = market[(market['date'] >= '2021-01-01') & (market['date'] < '2022-01-01')].copy()
    m2021_gen = m2021[m2021['y_gua'] == '001'].copy()
    print(f'\n## 2021 全年大盘交易日: {len(m2021):,}')
    print(f'## 其中艮 regime: {len(m2021_gen):,} 日')
    if len(m2021_gen) > 0:
        print(f'  日期: {m2021_gen.date.min()} ~ {m2021_gen.date.max()}')
        # 按月分布
        m2021_gen['month'] = m2021_gen['date'].str[:7]
        month_cnt = m2021_gen['month'].value_counts().sort_index()
        print(f'\n  按月日数:')
        for mo, cnt in month_cnt.items():
            print(f'    {mo}: {cnt} 日')

    # 主升浪 138 起点 (复用前面逻辑)
    print(f'\n=== 加载个股卦+收盘 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    market_y = market[['date', 'y_gua']].rename(columns={'y_gua': 'mkt_y'}).drop_duplicates('date')
    df = g.merge(p, on=['date', 'code'], how='inner').merge(market_y, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫艮 regime w4_2021 内的乾连续段 ≥10
    print(f'\n=== 艮 regime w4_2021 主升浪 (≥10日) 起点 ===')
    runs = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
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
            if length >= 10 and mkt_y_arr[gi] == '001':
                d = date_arr[gi]
                if '2021-01-01' <= d < '2022-01-01':
                    runs.append({
                        'code': code_arr[gi], 'start': d, 'end': date_arr[s+j-1],
                        'length': length,
                    })
            i = j
    df_r = pd.DataFrame(runs)
    print(f'  共 {len(df_r)} 个起点')
    if len(df_r) > 0:
        df_r['month'] = df_r['start'].str[:7]
        month_cnt = df_r['month'].value_counts().sort_index()
        print(f'\n  按月分布:')
        for mo, cnt in month_cnt.items():
            print(f'    {mo}: {cnt} 个')
        print(f'\n  起点日期范围: {df_r.start.min()} ~ {df_r.start.max()}')
        print(f'  长度分布: 中位 {df_r.length.median():.0f}, mean {df_r.length.mean():.1f}, max {df_r.length.max()}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
