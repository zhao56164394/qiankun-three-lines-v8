# -*- coding: utf-8 -*-
"""池天 bug 修正 — 三种入池逻辑对比

test124 发现 "池天 120-365 天 +20.86%" 极可疑.
原因: pool_enter_i 只在第一次入池时记, 之后即使 retail 又跌到 -250 也不更新.
所以"挂着 1 年"的池天可能根本是错记的.

三种逻辑对比:
  V1 (test124 原版): 第一次入池就 lock enter_i, 不滚动. 退池靠触发.
  V2 (滚动入池): 每次 retail<-250 都更新 enter_i. 池天 = 距最近一次深跌的天数.
  V3 (超时退池): 入池后 60 日没触发就强退. 强退后下次 retail<-250 重新入池.

预期:
  如果 V1 的"120-365 金区"是 bug, V2/V3 里这部分应该消失或大幅变弱.
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
POOL_TIMEOUT = 60  # V3 超时


def scan(arrays, mode):
    """mode in {'V1', 'V2', 'V3'}"""
    code_starts, code_ends = arrays['starts'], arrays['ends']
    retail = arrays['retail']
    close = arrays['close']
    stk_d = arrays['stk_d']
    mkt_y = arrays['mkt_y']
    date = arrays['date']
    code = arrays['code']

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i

            below = retail[gi] < POOL_THR

            if mode == 'V1':
                # 原版: 第一次入池就 lock
                if not in_pool and below:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]

            elif mode == 'V2':
                # 滚动: 每次 retail<-250 都重置 enter_i
                if below:
                    in_pool = True
                    pool_enter_i = i  # 每次都更新
                    pool_min_retail = retail[gi]
                # 不更新 pool_min_retail 累积, 因为 enter 是滚动的

            elif mode == 'V3':
                # 超时退池
                if not in_pool and below:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                if in_pool:
                    if retail[gi] < pool_min_retail:
                        pool_min_retail = retail[gi]
                    if i - pool_enter_i > POOL_TIMEOUT:  # 超时
                        in_pool = False

            # 触发: regime + 巽日 (裸跑, 无 score 无避雷)
            if in_pool and mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                pool_days = i - pool_enter_i
                ret_30 = (close[gi+EVAL_WIN] / close[gi] - 1) * 100
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'ret_30': ret_30,
                    'pool_days': pool_days,
                    'pool_min_retail': pool_min_retail,
                })
                in_pool = False
    return pd.DataFrame(events)


def report(df_e, mode):
    print(f'\n=== {mode}: n={len(df_e):,} 全集 {df_e["ret_30"].mean():+.2f}% '
          f'胜 {(df_e["ret_30"]>0).mean()*100:.1f}% ===')
    print(f'  池天: avg={df_e["pool_days"].mean():.1f} 中位={df_e["pool_days"].median():.0f} '
          f'max={df_e["pool_days"].max()}')

    fine_bins = [0, 3, 6, 9, 15, 30, 60, 120, 365, 9999]
    fine_labels = ['[0,3)', '[3,6)', '[6,9)', '[9,15)', '[15,30)', '[30,60)',
                    '[60,120)', '[120,365)', '[365+)']
    df_e['db'] = pd.cut(df_e['pool_days'], bins=fine_bins, labels=fine_labels, right=False)
    base = df_e['ret_30'].mean()
    print(f'  {"档":<14} {"n":>6} {"avg":>8} {"win":>6} {"lift":>7}')
    for lab in fine_labels:
        sub = df_e[df_e['db'] == lab]
        if len(sub) < 10: continue
        avg = sub['ret_30'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+7.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>5.1f}% {avg-base:>+6.2f}')

    # 池深
    print(f'  池深 <-500: ', end='')
    sub = df_e[df_e['pool_min_retail'] < -500]
    if len(sub):
        print(f'n={len(sub)} ret={sub["ret_30"].mean():+.2f}% '
              f'win={(sub["ret_30"]>0).mean()*100:.1f}%')


def main():
    t0 = time.time()
    print('=== 池天 bug 修正: V1/V2/V3 对比 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for c in ['d_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

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
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'close': df['close'].to_numpy().astype(np.float64),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'stk_d': df['stk_d'].to_numpy(),
        'mkt_y': df['mkt_y'].to_numpy(),
        'starts': code_starts, 'ends': code_ends,
    }

    df_v1 = scan(arrays, 'V1'); report(df_v1, 'V1 原版 (第一次入池 lock)')
    df_v2 = scan(arrays, 'V2'); report(df_v2, 'V2 滚动入池 (每次 retail<-250 重置 enter_i)')
    df_v3 = scan(arrays, 'V3'); report(df_v3, 'V3 超时退池 (60 日强退)')

    # 同口径对比 [120,365) 金区
    print('\n=== 池天 [120,365) 金区在三版下的对比 ===')
    for mode, df_x in [('V1', df_v1), ('V2', df_v2), ('V3', df_v3)]:
        sub = df_x[(df_x['pool_days'] >= 120) & (df_x['pool_days'] < 365)]
        print(f'  {mode}: n={len(sub):>5,} ret={sub["ret_30"].mean():>+6.2f}% '
              f'win={(sub["ret_30"]>0).mean()*100:>5.1f}%')

    # 同口径对比 [60,120)
    print('\n=== 池天 [60,120) 在三版下的对比 ===')
    for mode, df_x in [('V1', df_v1), ('V2', df_v2), ('V3', df_v3)]:
        sub = df_x[(df_x['pool_days'] >= 60) & (df_x['pool_days'] < 120)]
        print(f'  {mode}: n={len(sub):>5,} ret={sub["ret_30"].mean():>+6.2f}% '
              f'win={(sub["ret_30"]>0).mean()*100:>5.1f}%')

    # 同口径对比 [3,30)
    print('\n=== 池天 [3,30) 在三版下的对比 ===')
    for mode, df_x in [('V1', df_v1), ('V2', df_v2), ('V3', df_v3)]:
        sub = df_x[(df_x['pool_days'] >= 3) & (df_x['pool_days'] < 30)]
        print(f'  {mode}: n={len(sub):>5,} ret={sub["ret_30"].mean():>+6.2f}% '
              f'win={(sub["ret_30"]>0).mean()*100:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
