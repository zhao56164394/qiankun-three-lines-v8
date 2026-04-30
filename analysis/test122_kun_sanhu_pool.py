# -*- coding: utf-8 -*-
"""坤 + 多种 散户线低点 入池条件对比

关键 v3 历史: 老 backtest_8gua.py 的入池阈值就是 散户线 < -250.
这次系统对比:

候选 (基于散户线 sanhu):
  A. sanhu < -100 (5 日均 < -100)
  B. sanhu < -150
  C. sanhu < -200
  D. sanhu < -250 (老 v3 标准)
  E. sanhu < -300
  F. sanhu < -400 (深度)
  G. sanhu < -500
  H. sanhu_5d < -100
  I. sanhu_5d < -200
  J. sanhu_5d < -300
  K. 30d 内 sanhu_5d < -250 (1 月内有过深度)

跟之前 trend<11 对比 + 跟无入池基线对比
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
REGIME_Y = '000'
TRIGGER_GUA = '011'

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]


def main():
    t0 = time.time()
    print('=== 坤 + 散户线低点 入池条件对比 ===\n')

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
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_30d_min'] = df.groupby('code', sort=False)['sanhu_5d'].transform(
        lambda s: s.rolling(30, min_periods=10).min())
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)

    # 散户低点条件 (状态型, 当日满足)
    for thr in [-100, -150, -200, -250, -300, -400, -500]:
        df[f'sanhu_below_{abs(thr)}'] = df['retail'] < thr
        df[f'sanhu_5d_below_{abs(thr)}'] = df['sanhu_5d'] < thr

    # 30d 内 sanhu_5d 触及深度
    for thr in [-250, -300, -400]:
        df[f'sanhu_30d_min_below_{abs(thr)}'] = df['sanhu_30d_min'] < thr

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

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    pool_conditions = {
        'A0. 无入池 (基线)': None,
        'A1. trend 下穿 11 (现行)': df['cross_below_11'].values,
        'B1. sanhu < -100': df['sanhu_below_100'].values,
        'B2. sanhu < -150': df['sanhu_below_150'].values,
        'B3. sanhu < -200': df['sanhu_below_200'].values,
        'B4. sanhu < -250 (老 v3)': df['sanhu_below_250'].values,
        'B5. sanhu < -300': df['sanhu_below_300'].values,
        'B6. sanhu < -400': df['sanhu_below_400'].values,
        'B7. sanhu < -500': df['sanhu_below_500'].values,
        'C1. sanhu_5d < -100': df['sanhu_5d_below_100'].values,
        'C2. sanhu_5d < -200': df['sanhu_5d_below_200'].values,
        'C3. sanhu_5d < -250': df['sanhu_5d_below_250'].values,
        'C4. sanhu_5d < -300': df['sanhu_5d_below_300'].values,
        'D1. 30d 内 sanhu_5d <-250': df['sanhu_30d_min_below_250'].values,
        'D2. 30d 内 sanhu_5d <-300': df['sanhu_30d_min_below_300'].values,
        'D3. 30d 内 sanhu_5d <-400': df['sanhu_30d_min_below_400'].values,
    }

    print(f'  {len(df):,} 行\n')
    print(f'  {"入池条件":<32} {"事件":>7} {"全 ret":>8} {"胜率":>6} {"段稳":>5} {"sc3 n":>6} {"sc3 ret":>9}')

    results = []
    for label, mask in pool_conditions.items():
        events = []

        if mask is None:
            for ci in range(len(code_starts)):
                s = code_starts[ci]; e = code_ends[ci]
                if e - s < LOOKBACK + EVAL_WIN + 5: continue
                n = e - s
                for i in range(LOOKBACK, n - EVAL_WIN):
                    gi = s + i
                    if mkt_y_arr[gi] != REGIME_Y: continue
                    if stk_d_arr[gi] != TRIGGER_GUA: continue
                    if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111': continue
                    score = 0
                    if mkt_m_arr[gi] == '100': score += 1
                    if mkt_d_arr[gi] == '011': score += 1
                    if mkt_m_arr[gi] == '010': score += 1
                    if stk_m_arr[gi] == '010': score += 1
                    if score < 2: continue
                    ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                    events.append({'date': date_arr[gi], 'score': score, 'ret_30': ret_30})
        else:
            for ci in range(len(code_starts)):
                s = code_starts[ci]; e = code_ends[ci]
                if e - s < LOOKBACK + EVAL_WIN + 5: continue
                n = e - s
                in_pool = False

                for i in range(LOOKBACK, n - EVAL_WIN):
                    gi = s + i
                    if mask[gi]:
                        in_pool = True

                    if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                        if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                            in_pool = False
                            continue
                        score = 0
                        if mkt_m_arr[gi] == '100': score += 1
                        if mkt_d_arr[gi] == '011': score += 1
                        if mkt_m_arr[gi] == '010': score += 1
                        if stk_m_arr[gi] == '010': score += 1
                        if score < 2:
                            in_pool = False
                            continue
                        ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                        events.append({'date': date_arr[gi], 'score': score, 'ret_30': ret_30})
                        in_pool = False

        df_e = pd.DataFrame(events)
        if len(df_e) == 0:
            print(f'  {label:<32} no events')
            continue
        df_e['seg'] = ''
        for w in WINDOWS:
            df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
        df_e = df_e[df_e['seg'] != ''].copy()

        avg = df_e['ret_30'].mean()
        win = (df_e['ret_30']>0).mean() * 100
        n_pos = 0; n_seg = 0
        for w in WINDOWS:
            sub = df_e[df_e['seg'] == w[0]]
            if len(sub) > 50:
                n_seg += 1
                if sub['ret_30'].mean() > 0: n_pos += 1
        sub3 = df_e[df_e['score']==3]
        s3_n = len(sub3)
        s3_ret = sub3['ret_30'].mean() if len(sub3) else 0

        results.append((label, len(df_e), avg, win, n_pos, n_seg, s3_n, s3_ret))
        print(f'  {label:<32} {len(df_e):>7,} {avg:>+7.2f}% {win:>5.1f}% '
              f'{n_pos}/{n_seg} {s3_n:>6,} {s3_ret:>+8.2f}%')

    print('\n=== 按全期 ret 排序 (前 5) ===')
    results.sort(key=lambda x: x[2], reverse=True)
    for r in results[:5]:
        print(f'  {r[0]:<32} 全 {r[2]:+.2f}% 胜 {r[3]:.1f}% n={r[1]:,} sc3 {r[7]:+.2f}%')

    print('\n=== 按 score=3 ret 排序 (高质量子集) ===')
    results.sort(key=lambda x: x[7], reverse=True)
    for r in results[:5]:
        print(f'  {r[0]:<32} sc3 {r[7]:+.2f}% n={r[6]} 全 {r[2]:+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
