# -*- coding: utf-8 -*-
"""暴涨股切换 T3 — 检测到暴涨就切换到 T3 (3 次穿 89 终结)

逻辑:
  默认 T0 (仅 trend<11 终结, D6+U1 波段)
  入场后持续监测浮盈, 触发暴涨条件就 lock T3:
    - F1: 入场后 5 日内浮盈 > +30%
    - F2: 入场后 10 日内浮盈 > +50%
    - F3: 入场后 20 日内浮盈 > +100%
    - F4: 入场后 5 日内连续 2 个涨停 (10%+)
    - F5: trend 入场后 5 日内 > 89 (快速冲顶)

也测纯 T3, 看哪个组合最优
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan
        pool_min_retail = np.inf
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
                pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            if in_pool and mf_cross_up and retail_rising:
                events.append({'date': date[gi], 'code': code[gi],
                                'buy_idx_global': gi, 'pool_min_retail': pool_min_retail})
                in_pool = False
            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_with_switch(buy_idx, td, close, mf, retail, max_end, switch_mode):
    """切换逻辑:
       默认 T0 (无 89 穿越终结)
       检测到暴涨切换 T3 (3 次穿 89 终结)

       switch_mode:
         'none'    : 一直 T0
         'always'  : 一直 T3 (基线对比)
         'F1'      : 5d ret>30% 切 T3
         'F2'      : 10d ret>50% 切 T3
         'F3'      : 20d ret>100% 切 T3
         'F4'      : 5d 内 trend>=89 切 T3
         'F5'      : 5d 内浮盈 +50% (更严)
         'F6'      : 5d 内 mf > 500 (主力极强)
    """
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 0
    cross_89_count = 0
    in_t3 = (switch_mode == 'always')
    high_close = bp_first  # 浮盈高点
    switched_at = None

    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx

        # 检测切换
        if not in_t3:
            cur_ret = (close[k] / bp_first - 1) * 100
            high_close = max(high_close, close[k])
            high_ret = (high_close / bp_first - 1) * 100

            if switch_mode == 'F1' and days <= 5 and high_ret > 30:
                in_t3 = True
                switched_at = k
            elif switch_mode == 'F2' and days <= 10 and high_ret > 50:
                in_t3 = True
                switched_at = k
            elif switch_mode == 'F3' and days <= 20 and high_ret > 100:
                in_t3 = True
                switched_at = k
            elif switch_mode == 'F4' and days <= 5 and not np.isnan(td[k]) and td[k] >= 89:
                in_t3 = True
                switched_at = k
            elif switch_mode == 'F5' and days <= 5 and high_ret > 50:
                in_t3 = True
                switched_at = k
            elif switch_mode == 'F6' and days <= 5 and not np.isnan(mf[k]) and mf[k] > 500:
                in_t3 = True
                switched_at = k

        # trend<11 终结
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'td<11', (cum_mult-1)*100, legs, in_t3, switched_at

        # 89 穿越 (T3 模式才用)
        if k > 0 and not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] >= 89 and td[k] < 89:
                cross_89_count += 1
                if in_t3 and cross_89_count >= 3:
                    if holding:
                        cum_mult *= close[k] / cur_buy_price
                        legs += 1
                    return k, 'cross89_3', (cum_mult-1)*100, legs, in_t3, switched_at

        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0

        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]
                holding = True

    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
    return max_end, 'fc', (cum_mult-1)*100, legs, in_t3, switched_at


def main():
    t0 = time.time()
    print('=== T0 + 暴涨股切 T3 组合 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'mf': df['main_force'].to_numpy().astype(np.float64),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    mf_arr = arrays['mf']
    retail_arr = arrays['retail']

    df_e = find_signals(arrays)
    print(f'  入场信号: {len(df_e):,}')

    # 关键股测试
    cases = [
        ('顺丰', '002352', '2016-01-19'),
        ('九安', '002432', '2021-11-03'),
        ('神火', '000933', '2016-02-17'),
        ('科沃斯', '603486', '2020-04-07'),
        ('澄星', '600078', '2021-05-24'),
    ]
    print(f'\n=== 关键股测试 ===\n')
    print(f'  {"股":<10} {"日期":<12}', end='')
    for s in ['none', 'always(T3)', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
        print(f' {s:>11}', end='')
    print()
    for tag, code, dt in cases:
        sf_idx = None
        for _, ev in df_e.iterrows():
            if ev['code'] == code and ev['date'] == dt:
                sf_idx = int(ev['buy_idx_global'])
                break
        if sf_idx is None: continue
        ci = np.searchsorted(code_starts, sf_idx, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, sf_idx + MAX_TRACK)

        print(f'  {tag:<10} {dt:<12}', end='')
        for sm in ['none', 'always', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
            _, _, ret, _, in_t3, _ = simulate_with_switch(sf_idx, trend_arr, close_arr,
                                                              mf_arr, retail_arr, max_end, sm)
            mark = '*' if in_t3 and sm not in ('none', 'always') else ' '
            print(f' {ret:>+9.1f}%{mark}', end='')
        print()
    print('  (* 表示触发了切换)')

    # 全样本
    print(f'\n=== 全样本 ===\n')
    print(f'  {"模式":<14} {"avg_ret":>9} {"win%":>7} {"max":>9} {"min":>7} {"切换%":>6}')
    for sm in ['none', 'always', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']:
        rets = []; switches = 0
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e = code_ends[ci]
            max_end = min(e - 1, gi + MAX_TRACK)
            _, _, ret, _, in_t3, _ = simulate_with_switch(gi, trend_arr, close_arr,
                                                              mf_arr, retail_arr, max_end, sm)
            rets.append(ret)
            if in_t3 and sm != 'none' and sm != 'always':
                switches += 1
        avg = np.mean(rets); win = sum(1 for r in rets if r>0)/len(rets)*100
        switch_pct = switches/len(rets)*100 if sm not in ('none','always') else (0 if sm=='none' else 100)
        print(f'  {sm:<14} {avg:>+8.2f}% {win:>6.1f}% {max(rets):>+8.1f}% {min(rets):>+6.1f}% {switch_pct:>5.1f}%')

    # 跨段对比 (前 4 模式)
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    show_modes = ['none', 'F1', 'F2', 'F3', 'F4']
    print(f'\n=== 跨段 ===\n')
    print(f'  {"段":<14}', end='')
    for s in show_modes:
        print(f' {s:>10}', end='')
    print()

    all_rows = {sm: [] for sm in show_modes}
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        for sm in show_modes:
            _, _, ret, _, _, _ = simulate_with_switch(gi, trend_arr, close_arr,
                                                          mf_arr, retail_arr, max_end, sm)
            all_rows[sm].append({'date': ev['date'], 'ret': ret})

    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for sm in show_modes:
            df_x = pd.DataFrame(all_rows[sm])
            sub = df_x[(df_x['date'] >= w[1]) & (df_x['date'] < w[2])]
            print(f' {sub["ret"].mean():>+9.2f}%', end='')
        print()

    # F2 / F3 切换的股 vs 没切的
    print(f'\n=== F2 切换 vs 没切对比 ===')
    rets_switched = []; rets_not_switched = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        _, _, ret, _, in_t3, _ = simulate_with_switch(gi, trend_arr, close_arr,
                                                          mf_arr, retail_arr, max_end, 'F2')
        # 不切 T0 ret 多少
        _, _, ret_t0, _, _, _ = simulate_with_switch(gi, trend_arr, close_arr,
                                                          mf_arr, retail_arr, max_end, 'none')
        if in_t3:
            rets_switched.append((ret, ret_t0))
        else:
            rets_not_switched.append((ret, ret_t0))
    print(f'  切换组: n={len(rets_switched)}, F2 ret avg={np.mean([r[0] for r in rets_switched]):+.2f}%, '
          f'T0 ret avg={np.mean([r[1] for r in rets_switched]):+.2f}%')
    print(f'  不切组: n={len(rets_not_switched)}, F2 ret avg={np.mean([r[0] for r in rets_not_switched]):+.2f}%, '
          f'T0 ret avg={np.mean([r[1] for r in rets_not_switched]):+.2f}%')
    print(f'  切换组的差: {np.mean([r[0]-r[1] for r in rets_switched]):+.2f}% (T3 vs T0)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
