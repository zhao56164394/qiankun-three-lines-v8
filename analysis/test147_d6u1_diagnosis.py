# -*- coding: utf-8 -*-
"""D6+U1 vs baseline (bull_2nd) 详细对比 — 哪些股赢哪些股输

D6+U1 全样本 +7.03%, baseline +7.49%, 差 -0.46%
但顺丰 +500% vs +401% (+99%), 神火 +86% vs -59% (+145%)

诊断: 哪类股 D6+U1 赢, 哪类输? 能否结合用?
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
LOOKBACK = 30


def find_signals(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan
        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            if in_pool and mf_cross_up and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                })
                in_pool = False
            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_d6u1(buy_idx, td, close, mf, retail, max_end):
    """D6: 三线齐降卖, U1: mf 上升买"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 0
    legs_detail = []

    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx

        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                legs_detail.append(('sell-td<11', k, close[k]))
            return k, 'td<11', (cum_mult-1)*100, legs, legs_detail

        if days >= MAX_HOLD:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                legs_detail.append(('sell-timeout', k, close[k]))
            return k, 'timeout', (cum_mult-1)*100, legs, legs_detail

        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0

        if holding:
            # D6: 三线齐降
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                legs_detail.append(('sell-D6', k, close[k]))
                holding = False
        else:
            # U1: mf 上升
            if mf_c > 0:
                cur_buy_price = close[k]
                legs_detail.append(('buy-U1', k, close[k]))
                holding = True

    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
        legs_detail.append(('sell-fc', max_end, close[max_end]))
    return max_end, 'fc', (cum_mult-1)*100, legs, legs_detail


def simulate_baseline(buy_idx, td, close, max_end):
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close[k]/bp-1)*100
        if days >= 20:
            seg = td[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return k, 'ts20', (close[k]/bp-1)*100
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def main():
    t0 = time.time()
    print('=== D6+U1 vs baseline 对比 ===\n')

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

    # 对每笔 跑两种
    rows = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)

        _, _, ret_d6u1, legs, _ = simulate_d6u1(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)
        _, _, ret_base = simulate_baseline(gi, trend_arr, close_arr, max_end)

        rows.append({
            'date': ev['date'], 'code': ev['code'],
            'd6u1_ret': ret_d6u1, 'legs': legs,
            'base_ret': ret_base,
            'diff': ret_d6u1 - ret_base,
        })

    df_r = pd.DataFrame(rows)
    print(f'\n  D6+U1 全样本 avg: {df_r["d6u1_ret"].mean():+.2f}%')
    print(f'  baseline 全样本 avg: {df_r["base_ret"].mean():+.2f}%')
    print(f'  D6+U1 - baseline 差距: {df_r["diff"].mean():+.2f}%')

    # diff 分布
    print(f'\n=== D6+U1 vs baseline 单笔差距分布 ===')
    print(f'  D6+U1 赢的笔数 (diff>0): {(df_r["diff"]>0).sum()} ({(df_r["diff"]>0).mean()*100:.1f}%)')
    print(f'  D6+U1 输的笔数 (diff<0): {(df_r["diff"]<0).sum()} ({(df_r["diff"]<0).mean()*100:.1f}%)')
    print(f'  D6+U1 输的中位差距: {df_r[df_r["diff"]<0]["diff"].median():+.2f}%')
    print(f'  D6+U1 赢的中位差距: {df_r[df_r["diff"]>0]["diff"].median():+.2f}%')

    # 按 base_ret 分箱看 D6+U1 表现
    print(f'\n=== 按 baseline ret 分箱 看 D6+U1 表现 ===\n')
    bins = [-100, -30, -10, 0, 10, 30, 100, 9999]
    labels = ['<-30', '[-30,-10)', '[-10,0)', '[0,10)', '[10,30)', '[30,100)', '>=100']
    df_r['base_bin'] = pd.cut(df_r['base_ret'], bins=bins, labels=labels)

    print(f'  {"baseline 区间":<12} {"n":>6} {"base_avg":>9} {"d6u1_avg":>9} {"diff":>9}')
    for lab in labels:
        sub = df_r[df_r['base_bin'] == lab]
        if len(sub) < 10: continue
        print(f'  {lab:<12} {len(sub):>6} {sub["base_ret"].mean():>+8.2f}% '
              f'{sub["d6u1_ret"].mean():>+8.2f}% {sub["diff"].mean():>+8.2f}%')

    # 按 D6+U1 legs 分箱看 ret
    print(f'\n=== 按 D6+U1 legs 分箱 ===\n')
    print(f'  {"legs":<6} {"n":>6} {"d6u1 ret":>10} {"base ret":>10} {"diff":>9}')
    for legs in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        sub = df_r[df_r['legs'] == legs]
        if len(sub) < 5: continue
        print(f'  {legs:<6} {len(sub):>6} {sub["d6u1_ret"].mean():>+9.2f}% '
              f'{sub["base_ret"].mean():>+9.2f}% {sub["diff"].mean():>+8.2f}%')
    sub = df_r[df_r['legs'] >= 11]
    if len(sub) >= 5:
        print(f'  >=11   {len(sub):>6} {sub["d6u1_ret"].mean():>+9.2f}% '
              f'{sub["base_ret"].mean():>+9.2f}% {sub["diff"].mean():>+8.2f}%')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'\n=== 跨段 D6+U1 vs baseline ===\n')
    print(f'  {"段":<14} {"n":>5} {"D6+U1":>8} {"base":>8} {"diff":>8} {"D6+U1 win%":>11}')
    for w in WINDOWS:
        sub = df_r[(df_r['date'] >= w[1]) & (df_r['date'] < w[2])]
        if len(sub) < 5: continue
        print(f'  {w[0]:<14} {len(sub):>5} {sub["d6u1_ret"].mean():>+7.2f}% '
              f'{sub["base_ret"].mean():>+7.2f}% {sub["diff"].mean():>+7.2f}% '
              f'{(sub["d6u1_ret"]>sub["base_ret"]).mean()*100:>9.1f}%')

    # 最大输和最大赢的案例
    print(f'\n=== D6+U1 vs baseline 最大赢/输案例 ===\n')
    print('Top 5 D6+U1 赢:')
    top = df_r.sort_values('diff', ascending=False).head(5)
    for _, r in top.iterrows():
        print(f'  {r["code"]} {r["date"]}: D6+U1 {r["d6u1_ret"]:+.2f}% / base {r["base_ret"]:+.2f}% / diff +{r["diff"]:.2f}% ({r["legs"]} 腿)')
    print('\nTop 5 D6+U1 输:')
    bot = df_r.sort_values('diff').head(5)
    for _, r in bot.iterrows():
        print(f'  {r["code"]} {r["date"]}: D6+U1 {r["d6u1_ret"]:+.2f}% / base {r["base_ret"]:+.2f}% / diff {r["diff"]:.2f}% ({r["legs"]} 腿)')

    # 顺丰诊断
    print(f'\n=== 顺丰 002352 2016-01-19 D6+U1 详细 ===')
    sf_evt = df_r[(df_r['code']=='002352') & (df_r['date']=='2016-01-19')]
    if len(sf_evt):
        gi = int(df_e[(df_e['code']=='002352') & (df_e['date']=='2016-01-19')].iloc[0]['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)
        _, _, ret, legs, detail = simulate_d6u1(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)
        print(f'  最终: {ret:+.2f}%, {legs} 腿')
        for d in detail:
            d_idx = d[1]
            print(f'    {d[0]:<14} {arrays["date"][d_idx]} close={d[2]:.2f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
