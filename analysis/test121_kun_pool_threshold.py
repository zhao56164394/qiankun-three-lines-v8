# -*- coding: utf-8 -*-
"""坤 + 多种入池条件对比 (不限仓位, 事件级 ret_30)

对比候选入池条件:
  A. 不入池 (基线 v3)
  B. 下穿 11 (现行 v4)
  C. 下穿 5 (更深底部)
  D. 下穿 8
  E. 下穿 15
  F. 下穿 20
  G. 下穿 30
  H. 下穿 50 (中位以下都算)
  I. 5d trend 最低 < 11 (放宽: 5 日内有过下穿)
  J. 30d trend 最低 < 11 (1 月内有过下穿)
  K. trend < 30 (低位状态, 不要求穿过)
  L. trend < 50

每个条件后, 在坤 regime + 巽日 + 强避雷 + score>=2 上扫
评估:
  - 总事件数 (越多 = 信号丰富)
  - 全期 30 日 ret (越高越好)
  - 胜率
  - 跨段稳定性 (5/5)
  - score=3 子集表现
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
    print('=== 坤 + 多种入池条件对比 ===\n')

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
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)

    # 多种下穿阈值
    for thr in [5, 8, 11, 15, 20, 30, 50]:
        df[f'cross_{thr}'] = (df['t_prev'] >= thr) & (df['d_trend'] < thr)

    # trend 5d_min, 30d_min 最近 N 日是否曾下穿 11
    df['trend_5d_min'] = df.groupby('code', sort=False)['d_trend'].transform(
        lambda s: s.rolling(5, min_periods=3).min())
    df['trend_30d_min'] = df.groupby('code', sort=False)['d_trend'].transform(
        lambda s: s.rolling(30, min_periods=10).min())
    df['recently_5d_below_11'] = df['trend_5d_min'] < 11
    df['recently_30d_below_11'] = df['trend_30d_min'] < 11

    # 当前低位 (不要求穿过)
    df['t_below_30'] = df['d_trend'] < 30
    df['t_below_50'] = df['d_trend'] < 50

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 每个候选条件的 mask
    pool_conditions = {
        'A. 无入池 (基线 v3)': None,  # 不需要入池逻辑, 直接所有事件
        'B. 下穿 11 (v4 现行)': df['cross_11'].values,
        'C. 下穿 5': df['cross_5'].values,
        'D. 下穿 8': df['cross_8'].values,
        'E. 下穿 15': df['cross_15'].values,
        'F. 下穿 20': df['cross_20'].values,
        'G. 下穿 30': df['cross_30'].values,
        'H. 下穿 50': df['cross_50'].values,
        'I. 5d 内 trend<11': df['recently_5d_below_11'].values,  # 状态型
        'J. 30d 内 trend<11': df['recently_30d_below_11'].values,
        'K. 当前 trend<30': df['t_below_30'].values,
        'L. 当前 trend<50': df['t_below_50'].values,
    }

    print(f'  {len(df):,} 行\n')

    print(f'  {"入池条件":<28} {"事件":>7} {"全 ret":>8} {"胜率":>6}  {"段稳":>5} {"sc=3 n":>7} {"sc=3 ret":>9}')
    results = []

    for label, mask in pool_conditions.items():
        events = []

        if mask is None:
            # A 基线: 不入池, 所有 mkt_y=000 + stk_d=011 都是事件
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
                    events.append({
                        'date': date_arr[gi], 'score': score, 'ret_30': ret_30,
                    })
        else:
            # 状态型 (I, J, K, L) 不需要 in_pool 跟踪, 直接判断
            is_state_type = label.startswith(('I.', 'J.', 'K.', 'L.'))

            if is_state_type:
                # 状态型: 当日满足条件即可 (也实现"满足条件再触发巽日就出池")
                # 这里跟"下穿"逻辑保持一致: 第一次满足条件起入池, 直到出池
                # 但状态型本身就是逐日判断的, 所以用"当日满足"作为入池条件
                # 改为: 当日满足状态 + 出池时检查
                for ci in range(len(code_starts)):
                    s = code_starts[ci]; e = code_ends[ci]
                    if e - s < LOOKBACK + EVAL_WIN + 5: continue
                    n = e - s
                    in_pool = False

                    for i in range(LOOKBACK, n - EVAL_WIN):
                        gi = s + i
                        if mask[gi]:  # 当日满足状态 → 入池 (持续标记)
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
                            events.append({
                                'date': date_arr[gi], 'score': score, 'ret_30': ret_30,
                            })
                            in_pool = False
            else:
                # 下穿型 (B, C, D, E, F, G, H)
                for ci in range(len(code_starts)):
                    s = code_starts[ci]; e = code_ends[ci]
                    if e - s < LOOKBACK + EVAL_WIN + 5: continue
                    n = e - s
                    in_pool = False

                    for i in range(LOOKBACK, n - EVAL_WIN):
                        gi = s + i
                        if mask[gi]:  # 下穿
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
                            events.append({
                                'date': date_arr[gi], 'score': score, 'ret_30': ret_30,
                            })
                            in_pool = False

        df_e = pd.DataFrame(events)
        if len(df_e) == 0:
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
                if sub['ret_30'].mean() > 0:
                    n_pos += 1

        sub3 = df_e[df_e['score']==3]
        s3_n = len(sub3)
        s3_ret = sub3['ret_30'].mean() if len(sub3) else 0

        results.append((label, len(df_e), avg, win, n_pos, n_seg, s3_n, s3_ret))
        print(f'  {label:<28} {len(df_e):>7,} {avg:>+7.2f}% {win:>5.1f}% '
              f'{n_pos}/{n_seg:<3} {s3_n:>6,} {s3_ret:>+8.2f}%')

    # 按全期 ret 排序
    print('\n=== 按全期 ret 排序 (前 5) ===')
    results.sort(key=lambda x: x[2], reverse=True)
    for r in results[:5]:
        print(f'  {r[0]:<28} ret={r[2]:+.2f}% n={r[1]:,} 段{r[4]}/{r[5]} sc=3 ret={r[7]:+.2f}%')

    print('\n=== 按 score=3 ret 排序 (高质量子集表现) ===')
    results.sort(key=lambda x: x[7], reverse=True)
    for r in results[:5]:
        print(f'  {r[0]:<28} sc=3 ret={r[7]:+.2f}% n={r[6]} 全 ret={r[2]:+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
