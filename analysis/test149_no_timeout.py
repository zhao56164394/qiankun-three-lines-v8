# -*- coding: utf-8 -*-
"""去掉 60 天 timeout, 只用 trend<11 终结 — 看 D6+U1 vs baseline

之前 60 天 timeout 砍掉了真正的主升浪后半段
顺丰从 60 天的 +400% 还能继续涨到 +469% (9 月末)
真正终结应该是 trend<11

bull_2nd 单次也对应改 — 不限 60 天, 看 bull_2nd 触发或 trend<11 哪个先
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365  # 极限追踪 1 年
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
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            if in_pool and mf_cross_up and retail_rising:
                events.append({'date': date[gi], 'code': code[gi],
                                'buy_idx_global': gi})
                in_pool = False
            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_d6u1_no_timeout(buy_idx, td, close, mf, retail, max_end):
    """D6 卖 + U1 买, 终结条件: trend<11 (无 60d timeout)"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 0

    for k in range(buy_idx + 1, max_end + 1):
        # trend<11 强卖 (整段终结)
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'td<11', (cum_mult-1)*100, legs

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
    return max_end, 'fc', (cum_mult-1)*100, legs


def simulate_baseline_no_timeout(buy_idx, td, close, max_end):
    """bull_2nd / TS20 / trend<11 (无 60d timeout)"""
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        # trend<11 强卖
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
    print('=== 无 60d timeout, 仅 trend<11 终结 — D6+U1 vs baseline ===\n')

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

    rows = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)

        si_d, r_d, ret_d, legs_d = simulate_d6u1_no_timeout(gi, trend_arr, close_arr,
                                                                arrays['mf'], arrays['retail'], max_end)
        si_b, r_b, ret_b = simulate_baseline_no_timeout(gi, trend_arr, close_arr, max_end)
        rows.append({
            'date': ev['date'], 'code': ev['code'],
            'd6u1_ret': ret_d, 'd6u1_days': si_d - gi, 'd6u1_legs': legs_d, 'd6u1_reason': r_d,
            'base_ret': ret_b, 'base_days': si_b - gi, 'base_reason': r_b,
            'diff': ret_d - ret_b,
        })

    df_r = pd.DataFrame(rows)

    print(f'\n=== 总览 ===')
    print(f'  D6+U1: avg ret={df_r["d6u1_ret"].mean():+.2f}%, '
          f'win={(df_r["d6u1_ret"]>0).mean()*100:.1f}%, '
          f'avg days={df_r["d6u1_days"].mean():.1f}, avg legs={df_r["d6u1_legs"].mean():.1f}')
    print(f'  baseline: avg ret={df_r["base_ret"].mean():+.2f}%, '
          f'win={(df_r["base_ret"]>0).mean()*100:.1f}%, '
          f'avg days={df_r["base_days"].mean():.1f}')
    print(f'  diff: {df_r["diff"].mean():+.2f}%')
    print(f'  D6+U1 赢笔数: {(df_r["diff"]>0).mean()*100:.1f}%')

    # 神火/顺丰
    print(f'\n=== 神火/顺丰 ===')
    for code, dt in [('000933', '2016-02-17'), ('002352', '2016-01-19'), ('002432', '2021-11-03')]:
        sub = df_r[(df_r['code']==code) & (df_r['date']==dt)]
        if len(sub):
            r = sub.iloc[0]
            print(f'  {code} {dt}:')
            print(f'    D6+U1: {r["d6u1_ret"]:+.2f}% / {r["d6u1_days"]}d / {r["d6u1_legs"]} 腿 / {r["d6u1_reason"]}')
            print(f'    base : {r["base_ret"]:+.2f}% / {r["base_days"]}d / {r["base_reason"]}')

    # reason 分布
    print(f'\n=== D6+U1 reason 分布 ===')
    for r, cnt in df_r['d6u1_reason'].value_counts().items():
        sub = df_r[df_r['d6u1_reason']==r]
        print(f'  {r:<10} n={cnt:>5} ret={sub["d6u1_ret"].mean():>+6.2f}% days={sub["d6u1_days"].mean():.1f}')

    print(f'\n=== baseline reason 分布 ===')
    for r, cnt in df_r['base_reason'].value_counts().items():
        sub = df_r[df_r['base_reason']==r]
        print(f'  {r:<10} n={cnt:>5} ret={sub["base_ret"].mean():>+6.2f}% days={sub["base_days"].mean():.1f}')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'\n=== 跨段 D6+U1 vs baseline ===')
    print(f'  {"段":<14} {"n":>5} {"D6+U1":>8} {"base":>8} {"diff":>8} {"win%":>6}')
    for w in WINDOWS:
        sub = df_r[(df_r['date'] >= w[1]) & (df_r['date'] < w[2])]
        if len(sub) < 5: continue
        print(f'  {w[0]:<14} {len(sub):>5} {sub["d6u1_ret"].mean():>+7.2f}% '
              f'{sub["base_ret"].mean():>+7.2f}% {sub["diff"].mean():>+7.2f}% '
              f'{(sub["diff"]>0).mean()*100:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
