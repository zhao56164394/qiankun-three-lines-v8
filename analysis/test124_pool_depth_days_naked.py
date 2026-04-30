# -*- coding: utf-8 -*-
"""坤 + sanhu<-250 入池 — 池深/池天 裸跑因子测试

去掉所有过滤 (避雷, score, stk_y/stk_m 黑名单), 只保留:
  入池: retail < -250 (第一次)
  触发: 大盘 y_gua=000 + 个股 d_gua=011 (任意巽日, 不挑 score)

目的: 看池深和池天这对因子单独的预测力, 不被 score 和避雷干扰

输出:
  1. 池深 6 档 × n / ret / 胜率
  2. 池天 6 档 × n / ret / 胜率
  3. 池深 × 池天 交叉
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
POOL_THR = -250


def main():
    t0 = time.time()
    print('=== 坤 + sanhu<-250 入池: 池深/池天 裸跑 (无避雷/score) ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y']).reset_index(drop=True)
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件 (裸跑, 仅 regime + 巽日)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf
        pool_min_sanhu5 = np.inf

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i

            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail_arr[gi]
                pool_min_sanhu5 = sh5_arr[gi] if not np.isnan(sh5_arr[gi]) else 0.0

            if in_pool:
                if retail_arr[gi] < pool_min_retail:
                    pool_min_retail = retail_arr[gi]
                if not np.isnan(sh5_arr[gi]) and sh5_arr[gi] < pool_min_sanhu5:
                    pool_min_sanhu5 = sh5_arr[gi]

            # 裸跑触发: 仅 regime + 巽日, 无避雷无 score
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                pool_days = i - pool_enter_i
                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'ret_30': ret_30,
                    'pool_days': pool_days,
                    'pool_min_retail': pool_min_retail,
                    'pool_min_sanhu5': pool_min_sanhu5,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,} (裸跑, 无 score 过滤)')
    print(f'  全集 ret = {df_e["ret_30"].mean():+.2f}%, 胜 {(df_e["ret_30"]>0).mean()*100:.1f}%')
    print(f'  池天: avg={df_e["pool_days"].mean():.1f} 中位={df_e["pool_days"].median():.0f} max={df_e["pool_days"].max()}')
    print(f'  池深: avg={df_e["pool_min_retail"].mean():.0f} 中位={df_e["pool_min_retail"].median():.0f} min={df_e["pool_min_retail"].min():.0f}')

    BASE_RET = df_e['ret_30'].mean()

    # ============ 1. 池深分箱 ============
    print('\n=== 池深分箱: pool_min_retail ===')
    bins_retail = [-np.inf, -1000, -700, -500, -400, -300, POOL_THR + 0.01]
    labels_retail = ['<-1000', '[-1000,-700)', '[-700,-500)', '[-500,-400)', '[-400,-300)', '[-300,-250)']
    df_e['depth_bin'] = pd.cut(df_e['pool_min_retail'], bins=bins_retail, labels=labels_retail, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"lift":>7}')
    for lab in labels_retail:
        sub = df_e[df_e['depth_bin'] == lab]
        if len(sub) == 0: continue
        avg = sub['ret_30'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["ret_30"].median():>+7.2f}% {avg-BASE_RET:>+6.2f}')

    # ============ 2. 池深分箱 (sanhu_5d) ============
    print('\n=== 池深分箱: pool_min_sanhu5 ===')
    bins_sh5 = [-np.inf, -800, -500, -350, -250, -150, np.inf]
    labels_sh5 = ['<-800', '[-800,-500)', '[-500,-350)', '[-350,-250)', '[-250,-150)', '>=-150']
    df_e['depth_sh5_bin'] = pd.cut(df_e['pool_min_sanhu5'], bins=bins_sh5, labels=labels_sh5, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"lift":>7}')
    for lab in labels_sh5:
        sub = df_e[df_e['depth_sh5_bin'] == lab]
        if len(sub) == 0: continue
        avg = sub['ret_30'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {avg-BASE_RET:>+6.2f}')

    # ============ 3. 池天分箱 ============
    print('\n=== 池天分箱: pool_days ===')
    bins_days = [0, 3, 7, 15, 30, 60, np.inf]
    labels_days = ['1-2', '3-6', '7-14', '15-29', '30-59', '60+']
    df_e['days_bin'] = pd.cut(df_e['pool_days'], bins=bins_days, labels=labels_days, right=False)
    print(f'  {"档":<10} {"n":>6} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"lift":>7}')
    for lab in labels_days:
        sub = df_e[df_e['days_bin'] == lab]
        if len(sub) == 0: continue
        avg = sub['ret_30'].mean()
        print(f'  {lab:<10} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["ret_30"].median():>+7.2f}% {avg-BASE_RET:>+6.2f}')

    # ============ 4. 池天细分 (1-29 死亡区里再细看) ============
    print('\n=== 池天细分 (前 30 天每 3 天一档) ===')
    fine_bins = list(range(0, 31, 3)) + [60, 120, 365, 9999]
    fine_labels = [f'[{fine_bins[i]},{fine_bins[i+1]})' for i in range(len(fine_bins)-1)]
    df_e['days_fine'] = pd.cut(df_e['pool_days'], bins=fine_bins, labels=fine_labels, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"lift":>7}')
    for lab in fine_labels:
        sub = df_e[df_e['days_fine'] == lab]
        if len(sub) < 30: continue
        avg = sub['ret_30'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {avg-BASE_RET:>+6.2f}')

    # ============ 5. 交叉 ============
    print('\n=== 池深 × 池天 交叉 (avg_ret%) ===')
    pivot = df_e.pivot_table(values='ret_30', index='depth_bin', columns='days_bin',
                                  aggfunc='mean', observed=True)
    print(pivot.round(2).to_string())

    print('\n=== 池深 × 池天 交叉 (n) ===')
    pivot_n = df_e.pivot_table(values='ret_30', index='depth_bin', columns='days_bin',
                                  aggfunc='count', observed=True)
    print(pivot_n.to_string())

    # ============ 6. 跨段稳定性 (池天 60+) ============
    print('\n=== 池天 60+ 跨段稳定性 ===')
    WINDOWS = [
        ('w1_2018',    '2018-01-01', '2019-01-01'),
        ('w2_2019',    '2019-01-01', '2020-01-01'),
        ('w4_2021',    '2021-01-01', '2022-01-01'),
        ('w5_2022',    '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    s60 = df_e[df_e['pool_days'] >= 60]
    print(f'  全集 (池天 60+): n={len(s60)} ret={s60["ret_30"].mean():+.2f}% '
          f'win {(s60["ret_30"]>0).mean()*100:.1f}%')
    for w in WINDOWS:
        sub = s60[(s60['date'] >= w[1]) & (s60['date'] < w[2])]
        if len(sub):
            print(f'    {w[0]:<12} n={len(sub):>4} ret={sub["ret_30"].mean():>+6.2f}% '
                  f'win {(sub["ret_30"]>0).mean()*100:>5.1f}%')

    # ============ 7. 跨段稳定性 (池天 7-29 死亡区) ============
    print('\n=== 池天 7-29 死亡区 跨段 ===')
    s_dead = df_e[(df_e['pool_days'] >= 7) & (df_e['pool_days'] < 30)]
    print(f'  全集: n={len(s_dead)} ret={s_dead["ret_30"].mean():+.2f}%')
    for w in WINDOWS:
        sub = s_dead[(s_dead['date'] >= w[1]) & (s_dead['date'] < w[2])]
        if len(sub):
            print(f'    {w[0]:<12} n={len(sub):>4} ret={sub["ret_30"].mean():>+6.2f}% '
                  f'win {(sub["ret_30"]>0).mean()*100:>5.1f}%')

    # ============ 8. 极深池 跨段 ============
    print('\n=== 池深 < -500 跨段 ===')
    s_deep = df_e[df_e['pool_min_retail'] < -500]
    print(f'  全集: n={len(s_deep)} ret={s_deep["ret_30"].mean():+.2f}% '
          f'win {(s_deep["ret_30"]>0).mean()*100:.1f}%')
    for w in WINDOWS:
        sub = s_deep[(s_deep['date'] >= w[1]) & (s_deep['date'] < w[2])]
        if len(sub):
            print(f'    {w[0]:<12} n={len(sub):>4} ret={sub["ret_30"].mean():>+6.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
