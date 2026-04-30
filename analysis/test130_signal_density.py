# -*- coding: utf-8 -*-
"""F0 / F3 / F8 信号密度对比 — 看每天能不能均衡买入

核心问题:
  K=5 仓位上限, 但若 2582 个交易日只有 124 天有信号 (F8),
  仓位就被一两笔单股长期占用 -> 大量资金闲置
  这才是 F8 平均仓位 11.9% 的原因

输出:
  1. 每日信号数分布 (有几只可选)
  2. 信号天数 / 总交易日 比例
  3. 每年信号天数 (看是否有信号荒月)
  4. 信号天的间隔分布 (信号荒长度)
  5. 仓位实际占用率与各方案的资金效率
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
POOL_THR = -250
POOL_EXIT_RETAIL = 0


def main():
    t0 = time.time()
    print('=== F0 / F3 / F8 信号密度对比 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
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

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True; pool_enter_i = i; pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] < pool_min_retail:
                pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False; continue
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                events.append({'date':date_arr[gi], 'code':code_arr[gi],
                                'pool_days':i-pool_enter_i,
                                'pool_min_retail':pool_min_retail})
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  全集事件: {len(df_e):,}')

    schemes = {
        'F0 baseline':                  df_e,
        'F3 (days<6)':                  df_e[df_e['pool_days'] < 6],
        'F8 (retail<-300 + days<6)':   df_e[(df_e['pool_min_retail'] < -300) & (df_e['pool_days'] < 6)],
        'F3a (days<9)':                 df_e[df_e['pool_days'] < 9],
        'F8a (retail<-300 + days<9)':  df_e[(df_e['pool_min_retail'] < -300) & (df_e['pool_days'] < 9)],
    }

    all_dates = sorted(df['date'].unique())
    total_days = len(all_dates)
    print(f'  总交易日: {total_days}\n')

    # ============ 1. 每日信号数分布 ============
    print('=== 1. 每日信号数分布 ===\n')
    print(f'  {"方案":<32} {"事件":>6} {"信号天":>6} {"覆盖":>6} {"avg/天":>7} {"max":>4} {"信号 1只":>8} {"信号 2-5":>8} {"信号 6+":>7}')
    for label, df_x in schemes.items():
        n_evt = len(df_x)
        daily_count = df_x.groupby('date').size()
        n_sig = len(daily_count)
        cov = n_sig / total_days * 100
        avg_per_day = n_evt / n_sig if n_sig else 0
        n1 = (daily_count == 1).sum()
        n25 = ((daily_count >= 2) & (daily_count <= 5)).sum()
        n6 = (daily_count >= 6).sum()
        print(f'  {label:<32} {n_evt:>6,} {n_sig:>6} {cov:>5.1f}% {avg_per_day:>6.1f} '
              f'{daily_count.max():>4} {n1:>7} {n25:>7} {n6:>6}')

    # ============ 2. 每年信号天数 ============
    print('\n=== 2. 每年信号天数 ===\n')
    df_dates = pd.DataFrame({'date': all_dates})
    df_dates['year'] = pd.to_datetime(df_dates['date']).dt.year
    yearly_total = df_dates.groupby('year').size()
    print(f'  {"年":<6} {"交易日":>6}', end='')
    for label in schemes.keys():
        print(f' | {label[:14]:>14}', end='')
    print()
    for y in sorted(yearly_total.index):
        td = yearly_total[y]
        print(f'  {y:<6} {td:>6}', end='')
        for label, df_x in schemes.items():
            df_x_y = df_x[pd.to_datetime(df_x['date']).dt.year == y]
            n_sig = df_x_y['date'].nunique()
            cov = n_sig / td * 100
            print(f' | {n_sig:>3}({cov:>4.1f}%)', end='')
        print()

    # ============ 3. 信号荒长度 (相邻信号天间隔) ============
    print('\n=== 3. 信号荒长度 (相邻信号天交易日间隔) ===\n')
    print(f'  {"方案":<32} {"avg":>6} {"中位":>5} {"max":>5} {"P90":>5} {"P95":>5} {"间隔>30天":>9}')
    for label, df_x in schemes.items():
        sig_days = sorted(df_x['date'].unique())
        if len(sig_days) < 2:
            print(f'  {label:<32} 信号太少')
            continue
        # 用日期 index 算交易日距离
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        sig_idx = [date_to_idx[d] for d in sig_days if d in date_to_idx]
        gaps = np.diff(sig_idx)
        print(f'  {label:<32} {gaps.mean():>5.1f} {np.median(gaps):>5.0f} '
              f'{gaps.max():>5} {np.percentile(gaps, 90):>5.0f} {np.percentile(gaps, 95):>5.0f} '
              f'{(gaps > 30).sum():>9}')

    # ============ 4. 仓位均衡性: 假设 K=5, 持仓 30 天, 实际能买多少 ============
    print('\n=== 4. 仓位利用模拟 (K=5, 假设每笔持 30 天) ===\n')
    K = 5
    HOLD = 30
    print(f'  {"方案":<32} {"信号天":>6} {"实际买":>6} {"挤压率":>6} {"满仓天":>6} {"空仓天":>6} {"avg 仓位":>7}')
    for label, df_x in schemes.items():
        df_picks = df_x.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')

        n_sig = len(df_picks)
        # 模拟 K=5 持 30 天
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        positions = np.zeros(total_days, dtype=int)  # 每天持仓数
        actual_buys = 0
        squeezed = 0

        # 简单模拟: 按日处理 picks
        slot_release_day = []  # 每个 slot 何时释放
        slot_taken = 0
        for d in df_picks['date'].values:
            if d not in date_to_idx: continue
            di = date_to_idx[d]
            # 释放过期 slot
            slot_release_day = [r for r in slot_release_day if r > di]
            slot_taken = len(slot_release_day)
            if slot_taken < K:
                slot_release_day.append(di + HOLD)
                actual_buys += 1
            else:
                squeezed += 1

        # 计算每天的 positions
        slot_release_day2 = []
        for di in range(total_days):
            slot_release_day2 = [r for r in slot_release_day2 if r > di]
            positions[di] = len(slot_release_day2)
            d = all_dates[di]
            if d in df_picks['date'].values:
                if len(slot_release_day2) < K:
                    slot_release_day2.append(di + HOLD)

        full_days = (positions >= K).sum()
        empty_days = (positions == 0).sum()
        avg_pos = positions.mean()
        squeeze_rate = squeezed / n_sig * 100 if n_sig else 0
        print(f'  {label:<32} {n_sig:>6} {actual_buys:>6} {squeeze_rate:>5.1f}% '
              f'{full_days:>6} {empty_days:>6} {avg_pos:>6.2f}/{K}')

    # ============ 5. F8 与 F0/F3 错过的好信号 ============
    print('\n=== 5. F0 vs F8 信号天对齐 ===\n')
    f0_dates = set(df_e['date'].unique())
    f3_dates = set(df_e[df_e['pool_days'] < 6]['date'].unique())
    f8_dates = set(df_e[(df_e['pool_min_retail'] < -300) & (df_e['pool_days'] < 6)]['date'].unique())
    print(f'  F0 有信号天: {len(f0_dates)}')
    print(f'  F3 有信号天: {len(f3_dates)} (F0 的 {len(f3_dates)/len(f0_dates)*100:.1f}%)')
    print(f'  F8 有信号天: {len(f8_dates)} (F0 的 {len(f8_dates)/len(f0_dates)*100:.1f}%)')
    print(f'  F0 - F8 (F0 有但 F8 没的天数): {len(f0_dates - f8_dates)}')
    print(f'  F3 - F8 (F3 有但 F8 没的天数): {len(f3_dates - f8_dates)}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
