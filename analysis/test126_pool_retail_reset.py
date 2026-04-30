# -*- coding: utf-8 -*-
"""池子的物理意义对比 — retail 持续低位才算池

V1: retail<-250 入池, 永不主动出池 (test124 原版)
V4: retail<-250 入池, retail>=0 主动出池
V5: retail<-250 入池, retail>=-100 主动出池 (更严)
V6: retail<-250 入池, sanhu_5d>=0 主动出池 (5 日均回正)

如果"金区 [120,365)"是真"散户线持续低位 1 年"
  → V4/V5/V6 这一档 n 应该大幅减少 (因为很多挂池其实早就 retail 转正了, 应该提前出池)
  → 但留下来的 [120,365) 应该 ret 仍然高 (真持续低位 = 真主力慢吸)

如果是"挂池 bug"
  → V4/V5/V6 之后 [120,365) 几乎清零
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


def scan(arrays, exit_thr_retail=None, exit_thr_sh5=None):
    """exit_thr_retail: retail >= 此值 出池 (None 不主动出)
       exit_thr_sh5: sanhu_5d >= 此值 出池"""
    code_starts, code_ends = arrays['starts'], arrays['ends']
    retail = arrays['retail']
    sh5 = arrays['sh5']
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

            # 入池
            if not in_pool and retail[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail[gi]

            # 池中累积深度
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]

            # 主动出池 (基于退出阈值)
            if in_pool:
                if exit_thr_retail is not None and retail[gi] >= exit_thr_retail:
                    in_pool = False
                    continue
                if exit_thr_sh5 is not None and not np.isnan(sh5[gi]) and sh5[gi] >= exit_thr_sh5:
                    in_pool = False
                    continue

            # 触发买入
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
    if len(df_e) == 0: return
    print(f'  池天 avg={df_e["pool_days"].mean():.1f} 中位={df_e["pool_days"].median():.0f} '
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


def main():
    t0 = time.time()
    print('=== 池主动出池机制对比 ===\n')

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
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
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
        'sh5': df['sanhu_5d'].to_numpy().astype(np.float64),
        'stk_d': df['stk_d'].to_numpy(),
        'mkt_y': df['mkt_y'].to_numpy(),
        'starts': code_starts, 'ends': code_ends,
    }

    df_v1 = scan(arrays); report(df_v1, 'V1 不主动出池 (原版)')
    df_v4 = scan(arrays, exit_thr_retail=0); report(df_v4, 'V4 retail>=0 出池')
    df_v5 = scan(arrays, exit_thr_retail=-100); report(df_v5, 'V5 retail>=-100 出池')
    df_v6 = scan(arrays, exit_thr_sh5=0); report(df_v6, 'V6 sanhu_5d>=0 出池')

    print('\n\n=== 对比 [120,365) 金区在四版下 ===')
    for mode, df_x in [('V1 (无出池)', df_v1), ('V4 (retail>=0)', df_v4),
                          ('V5 (retail>=-100)', df_v5), ('V6 (sh5>=0)', df_v6)]:
        sub = df_x[(df_x['pool_days'] >= 120) & (df_x['pool_days'] < 365)]
        if len(sub):
            print(f'  {mode:<22} n={len(sub):>5,} ret={sub["ret_30"].mean():>+6.2f}% '
                  f'win={(sub["ret_30"]>0).mean()*100:>5.1f}%')
        else:
            print(f'  {mode:<22} 0')

    print('\n=== 对比 全集 ret 在四版下 ===')
    for mode, df_x in [('V1', df_v1), ('V4', df_v4), ('V5', df_v5), ('V6', df_v6)]:
        print(f'  {mode}: n={len(df_x):,} ret={df_x["ret_30"].mean():+.2f}% '
              f'胜 {(df_x["ret_30"]>0).mean()*100:.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
