# -*- coding: utf-8 -*-
"""验证 D6 + U2 (mf 上升 AND retail>0)
对九安医疗 + 全样本 看效果

U1: mf 上升
U2: mf 上升 AND retail > 0
U3: mf 上升 AND trend > 50
U4: mf 上升 AND retail > 0 AND trend > 50
U5: mf 上升 AND mf > 上次卖出 mf
U6: mf 上升 AND mf > 0
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


def simulate(buy_idx, td, close, mf, retail, max_end, u_mode):
    """模拟 D6 + 不同 U_mode"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    last_sell_mf = mf[buy_idx]
    legs = 0

    for k in range(buy_idx + 1, max_end + 1):
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
                last_sell_mf = mf[k]
        else:
            # 不同 U_mode
            buy_now = False
            if u_mode == 'U1':
                buy_now = mf_c > 0
            elif u_mode == 'U2':  # mf 上升 AND retail>0
                buy_now = mf_c > 0 and retail[k] > 0
            elif u_mode == 'U3':  # mf 上升 AND trend>50
                buy_now = mf_c > 0 and (not np.isnan(td[k])) and td[k] > 50
            elif u_mode == 'U4':  # 联合
                buy_now = mf_c > 0 and retail[k] > 0 and (not np.isnan(td[k])) and td[k] > 50
            elif u_mode == 'U5':  # mf 上升 AND mf > 上次卖 mf
                buy_now = mf_c > 0 and mf[k] > last_sell_mf
            elif u_mode == 'U6':  # mf 上升 AND mf > 0
                buy_now = mf_c > 0 and mf[k] > 0
            elif u_mode == 'U7':  # mf 上升 AND retail>0 AND mf>50
                buy_now = mf_c > 0 and retail[k] > 0 and mf[k] > 50

            if buy_now:
                cur_buy_price = close[k]
                holding = True

    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
    return max_end, 'fc', (cum_mult-1)*100, legs


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
    print('=== U1-U7 对比 ===\n')

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

    # 单九安测试
    print(f'\n=== 九安医疗 002432 各 U 表现 ===')
    sf_idx = None
    for _, ev in df_e.iterrows():
        if ev['code'] == '002432' and ev['date'] == '2021-11-03':
            sf_idx = int(ev['buy_idx_global'])
            break
    if sf_idx:
        ci = np.searchsorted(code_starts, sf_idx, side='right') - 1
        e_idx = code_ends[ci]
        max_end = min(e_idx - 1, sf_idx + MAX_TRACK)
        for u in ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']:
            _, r, ret, legs = simulate(sf_idx, trend_arr, close_arr, mf_arr, retail_arr, max_end, u)
            print(f'  {u}: ret={ret:>+7.2f}%, {legs} 腿')
        _, _, br = simulate_baseline(sf_idx, trend_arr, close_arr, max_end)
        print(f'  baseline: ret={br:>+7.2f}%')

    # 全样本
    print(f'\n=== 全样本 U1-U7 ===\n')
    print(f'  {"方案":<8} {"avg_ret":>9} {"win%":>7} {"max":>8} {"min":>7} {"avg_legs":>8}')
    for u in ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']:
        rets = []; legs_list = []
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e_idx = code_ends[ci]
            max_end = min(e_idx - 1, gi + MAX_TRACK)
            _, _, ret, legs = simulate(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end, u)
            rets.append(ret); legs_list.append(legs)
        avg = np.mean(rets)
        win = sum(1 for r in rets if r>0) / len(rets) * 100
        mx = max(rets); mn = min(rets)
        legs = np.mean(legs_list)
        print(f'  {u:<8} {avg:>+8.2f}% {win:>6.1f}% {mx:>+7.1f}% {mn:>+6.1f}% {legs:>7.1f}')

    # baseline 对比
    base_rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e_idx = code_ends[ci]
        max_end = min(e_idx - 1, gi + MAX_TRACK)
        _, _, br = simulate_baseline(gi, trend_arr, close_arr, max_end)
        base_rets.append(br)
    print(f'\n  baseline: avg={np.mean(base_rets):+.2f}%, win={sum(1 for r in base_rets if r>0)/len(base_rets)*100:.1f}%')

    # 跨段 (取 top 3)
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'\n=== 跨段 ===\n')
    print(f'  {"段":<14}', end='')
    for u in ['U1', 'U2', 'U4', 'U7', 'baseline']:
        print(f' {u:>10}', end='')
    print()

    # 算各 U 跨段
    all_rows = {u: [] for u in ['U1', 'U2', 'U4', 'U7']}
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e_idx = code_ends[ci]
        max_end = min(e_idx - 1, gi + MAX_TRACK)
        for u in ['U1', 'U2', 'U4', 'U7']:
            _, _, ret, _ = simulate(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end, u)
            all_rows[u].append({'date': ev['date'], 'ret': ret})
    base_rows = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e_idx = code_ends[ci]
        max_end = min(e_idx - 1, gi + MAX_TRACK)
        _, _, br = simulate_baseline(gi, trend_arr, close_arr, max_end)
        base_rows.append({'date': ev['date'], 'ret': br})

    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for u in ['U1', 'U2', 'U4', 'U7']:
            df_x = pd.DataFrame(all_rows[u])
            sub = df_x[(df_x['date'] >= w[1]) & (df_x['date'] < w[2])]
            print(f' {sub["ret"].mean():>+9.2f}%', end='')
        sub_b = pd.DataFrame(base_rows)
        sub_b = sub_b[(sub_b['date'] >= w[1]) & (sub_b['date'] < w[2])]
        print(f' {sub_b["ret"].mean():>+9.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
