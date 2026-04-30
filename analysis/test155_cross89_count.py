# -*- coding: utf-8 -*-
"""加 trend 第 N 次下穿 89 终结条件

T0: 仅 trend<11
T1: trend<11 OR trend 第 1 次下穿 89
T2: trend<11 OR trend 第 2 次下穿 89 (= 现 baseline 思路)
T3: trend<11 OR trend 第 3 次下穿 89 (你的思路)
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


def simulate(buy_idx, td, close, mf, retail, max_end, end_mode):
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 0
    cross_89_count = 0
    end_threshold = {'T0': 999, 'T1': 1, 'T2': 2, 'T3': 3}[end_mode]

    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'td<11', (cum_mult-1)*100, legs, cross_89_count

        if k > 0 and not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] >= 89 and td[k] < 89:
                cross_89_count += 1
                if cross_89_count >= end_threshold:
                    if holding:
                        cum_mult *= close[k] / cur_buy_price
                        legs += 1
                    return k, f'cross89_{cross_89_count}', (cum_mult-1)*100, legs, cross_89_count

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
    return max_end, 'fc', (cum_mult-1)*100, legs, cross_89_count


def simulate_baseline(buy_idx, td, close, max_end):
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td<11', (close[k]/bp-1)*100
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close[k]/bp-1)*100
        if days >= 20:
            seg = td[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return k, 'ts20', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def main():
    t0 = time.time()
    print('=== T0/T1/T2/T3 终结条件对比 ===\n')

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

    df_e = find_signals(arrays)
    print(f'  入场信号: {len(df_e):,}')

    cases = [
        ('顺丰', '002352', '2016-01-19'),
        ('九安', '002432', '2021-11-03'),
        ('神火', '000933', '2016-02-17'),
        ('科沃斯', '603486', '2020-04-07'),
        ('澄星', '600078', '2021-05-24'),
    ]
    print(f'\n=== 关键股 ===\n')
    print(f'  {"股":<10} {"日期":<12}', end='')
    for t in ['T0', 'T1', 'T2', 'T3', 'base']:
        print(f' {t:>10}', end='')
    print(f' {"89穿":>5}')
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
        cross_total = 0
        for t_mode in ['T0', 'T1', 'T2', 'T3']:
            _, _, ret, _, cross = simulate(sf_idx, trend_arr, close_arr, arrays['mf'], arrays['retail'],
                                                  max_end, t_mode)
            print(f' {ret:>+9.1f}%', end='')
            cross_total = max(cross_total, cross)
        _, _, br = simulate_baseline(sf_idx, trend_arr, close_arr, max_end)
        print(f' {br:>+9.1f}% {cross_total:>5}')

    # 全样本
    print(f'\n=== 全样本 ===\n')
    print(f'  {"方案":<8} {"avg_ret":>9} {"win%":>7} {"max":>9} {"min":>7} {"avg_legs":>8}')
    all_rows = {}
    for t_mode in ['T0', 'T1', 'T2', 'T3']:
        rets = []; legs_list = []; rows = []
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e = code_ends[ci]
            max_end = min(e - 1, gi + MAX_TRACK)
            _, r, ret, legs, _ = simulate(gi, trend_arr, close_arr, arrays['mf'], arrays['retail'],
                                              max_end, t_mode)
            rets.append(ret); legs_list.append(legs)
            rows.append({'date': ev['date'], 'ret': ret, 'reason': r})
        all_rows[t_mode] = pd.DataFrame(rows)
        print(f'  {t_mode:<8} {np.mean(rets):>+8.2f}% {sum(1 for r in rets if r>0)/len(rets)*100:>6.1f}% '
              f'{max(rets):>+8.1f}% {min(rets):>+6.1f}% {np.mean(legs_list):>7.1f}')

    base_rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        _, _, br = simulate_baseline(gi, trend_arr, close_arr, max_end)
        base_rets.append(br)
    print(f'  {"base":<8} {np.mean(base_rets):>+8.2f}% '
          f'{sum(1 for r in base_rets if r>0)/len(base_rets)*100:>6.1f}%')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'\n=== 跨段 ===\n')
    print(f'  {"段":<14}', end='')
    for t in ['T0', 'T1', 'T2', 'T3', 'base']:
        print(f' {t:>10}', end='')
    print()
    base_df = pd.DataFrame([{'date':ev['date'],'ret':base_rets[i]} for i, (_, ev) in enumerate(df_e.iterrows())])
    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for t_mode in ['T0', 'T1', 'T2', 'T3']:
            sub = all_rows[t_mode][(all_rows[t_mode]['date'] >= w[1]) & (all_rows[t_mode]['date'] < w[2])]
            print(f' {sub["ret"].mean():>+9.2f}%', end='')
        sub_b = base_df[(base_df['date'] >= w[1]) & (base_df['date'] < w[2])]
        print(f' {sub_b["ret"].mean():>+9.2f}%')

    # 各 T 的 reason 分布
    for t_mode in ['T1', 'T2', 'T3']:
        print(f'\n=== {t_mode} reason 分布 ===')
        df_r = all_rows[t_mode]
        for r, cnt in df_r['reason'].value_counts().items():
            sub = df_r[df_r['reason']==r]
            print(f'  {r:<14} n={cnt:>4} ret={sub["ret"].mean():>+6.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
