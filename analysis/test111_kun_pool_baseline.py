# -*- coding: utf-8 -*-
"""阶段 2: 坤 regime + 入池版 baseline

入池条件: 个股 d_trend 上一日 >= 11 且 当日 < 11
在池: 入池后直到出池 (满足"巽日+regime"即出池, 不要求 score 这里只看入场基础事件)

事件定义:
  - 在池 + 大盘 y_gua=000 (坤) + 个股 d_gua=011 (巽日) → 入场事件
  - 评估: 后 30 日收盘价 / 当日收盘价 - 1

对比:
  - v3 时代 baseline: 无入池, 仅 mkt_y=000 + 巽日 (102K 事件 +6.20%)
  - 入池 baseline: 看保留率 + 期望差异
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '000'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

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
    print('=== 阶段 2: 坤 regime + 入池 baseline ===')
    print(f'  入池: 个股 d_trend 上一日>=11, 当日<11')
    print(f'  在池: 入池后, 直到 满足 mkt_y=000 + stk_d=011 任一日 (出池)')
    print(f'  事件: 在池 + 坤 regime + 巽日 → 30 日 ret\n')

    print('=== 加载 ===')
    # 主板过滤
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())
    print(f'  主板: {len(main_codes):,} 只')

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行 (主板)')

    # 计算下穿 11 标记
    print('  计算下穿 11 标记...')
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)
    n_cross = df['cross_below_11'].sum()
    print(f'  历史下穿 11 事件总数 (主板): {n_cross:,}')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    cross_arr = df['cross_below_11'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('\n=== 计算事件 (按你设定的入池逻辑) ===')
    # 按 code 模拟池子状态
    # 入池: cross_below_11
    # 出池: 满足 mkt_y=000 + stk_d=011 (即基础事件触发)
    # 入场事件 = 出池触发的那天, 即 在池 + mkt_y=000 + stk_d=011

    pool_events = []  # 入池版事件 (出池那天)
    no_pool_events = []  # 无入池版事件 (兼容老 v3, 任何 mkt_y=000 + stk_d=011)

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5:
            continue
        n = e - s
        in_pool = False
        pool_enter_idx = -1

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            # 1. 处理入池
            if cross_arr[gi]:
                in_pool = True
                pool_enter_idx = i

            # 2. 检查基础事件 (regime + 巽日)
            is_basic = (mkt_y_arr[gi] == REGIME_Y) and (stk_d_arr[gi] == TRIGGER_GUA)

            if is_basic:
                ret_30 = (close_arr[gi + EVAL_WIN] / close_arr[gi] - 1) * 100
                event = {
                    'date': date_arr[gi],
                    'code': code_arr[gi],
                    'ret_30': ret_30,
                    'in_pool': in_pool,
                    'days_in_pool': i - pool_enter_idx if in_pool else -1,
                }
                no_pool_events.append(event)
                if in_pool:
                    pool_events.append(event)
                    # 出池
                    in_pool = False
                    pool_enter_idx = -1

    df_pool = pd.DataFrame(pool_events)
    df_nopool = pd.DataFrame(no_pool_events)
    print(f'  无入池事件 (老 v3): {len(df_nopool):,}')
    print(f'  入池事件 (新): {len(df_pool):,}')
    print(f'  保留率: {len(df_pool)/max(len(df_nopool),1)*100:.1f}%')

    # 加段
    for df_e in [df_pool, df_nopool]:
        df_e['seg'] = ''
        for w in WINDOWS:
            df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]

    df_pool = df_pool[df_pool['seg'] != ''].copy()
    df_nopool = df_nopool[df_nopool['seg'] != ''].copy()

    print(f'\n=== 整体 baseline 对比 ===')
    print(f'  {"指标":<14} {"无入池(v3)":>12} {"入池版":>12}')
    print(f'  {"事件数":<14} {len(df_nopool):>12,} {len(df_pool):>12,}')
    print(f'  {"全期 30日 ret":<14} {df_nopool["ret_30"].mean():>+11.2f}% {df_pool["ret_30"].mean():>+11.2f}%')
    print(f'  {"胜率":<14} {(df_nopool["ret_30"]>0).mean()*100:>11.1f}% {(df_pool["ret_30"]>0).mean()*100:>11.1f}%')

    print(f'\n=== walk-forward (按段对比) ===')
    print(f'  {"段":<14} {"无入池 n":>10} {"无入池 ret":>11} {"入池 n":>9} {"入池 ret":>10} {"差异":>8}')
    n_pos_pool = 0; n_seg = 0
    for w in WINDOWS:
        sub_n = df_nopool[df_nopool['seg'] == w[0]]
        sub_p = df_pool[df_pool['seg'] == w[0]]
        if len(sub_n) < 50 and len(sub_p) < 50:
            continue
        n_seg += 1
        ret_n = sub_n['ret_30'].mean() if len(sub_n) else float('nan')
        ret_p = sub_p['ret_30'].mean() if len(sub_p) else float('nan')
        diff = ret_p - ret_n if not np.isnan(ret_p) and not np.isnan(ret_n) else float('nan')
        if not np.isnan(ret_p) and ret_p > 0:
            n_pos_pool += 1
        print(f'  {w[0]:<14} {len(sub_n):>10,} {ret_n:>+10.2f}% '
              f'{len(sub_p):>9,} {ret_p:>+9.2f}% {diff:>+7.2f}')

    print(f'\n  入池版段稳: {n_pos_pool}/{n_seg}')

    # 按入池停留时间分析
    if len(df_pool):
        print(f'\n=== 入池停留时间分析 (从入池到触发的天数) ===')
        df_pool['days_bucket'] = pd.cut(df_pool['days_in_pool'],
                                          bins=[-1, 0, 5, 10, 20, 60, 9999],
                                          labels=['当日', '1-5日', '6-10日', '11-20日', '21-60日', '>60日'])
        g = df_pool.groupby('days_bucket', observed=True).agg(
            n=('ret_30', 'count'),
            win=('ret_30', lambda x: (x > 0).mean() * 100),
            ret=('ret_30', 'mean'),
        )
        print(g.round(2))

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
