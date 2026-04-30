# -*- coding: utf-8 -*-
"""对比入场条件 — 原 E2 (mf 上穿 50) vs 新 E2 (mf 上升 + trend>11)

原: retail<-250 池中 + mf 上穿 50 + retail 上升
新: retail<-250 池中 + mf 上升 + retail 上升 + trend>11

也输出几个暴涨股的案例供通达信复盘
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v1(arrays):
    """原 E2: mf 上穿 50"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
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
                                'buy_idx_global': gi, 'pool_min_retail': pool_min_retail,
                                'cur_mf': mf[gi], 'cur_trend': td[gi]})
                in_pool = False
            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def find_signals_v2(arrays):
    """新 E2: mf 上升 (chg>0) + trend>11 (无穿 50 要求)"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
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
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                events.append({'date': date[gi], 'code': code[gi],
                                'buy_idx_global': gi, 'pool_min_retail': pool_min_retail,
                                'cur_mf': mf[gi], 'cur_trend': td[gi]})
                in_pool = False
            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_d6u1(buy_idx, td, close, mf, retail, max_end):
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
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
        else:
            if mf_c > 0:
                cur_buy_price = close[k]
                holding = True
    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
    return max_end, 'fc', (cum_mult-1)*100, legs


def main():
    t0 = time.time()
    print('=== 入场条件对比: 原 E2 (上穿50) vs 新 E2 (上升+trend>11) ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board', 'name'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())
    code2name = dict(zip(uni['code'], uni['name']))

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
        'td': df['d_trend'].to_numpy().astype(np.float64),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = arrays['td']
    mf_arr = arrays['mf']
    retail_arr = arrays['retail']

    df_v1 = find_signals_v1(arrays)
    df_v2 = find_signals_v2(arrays)
    print(f'  v1 (mf 上穿 50):       {len(df_v1):,}')
    print(f'  v2 (mf 上升 + td>11): {len(df_v2):,}')

    # 入场时 mf/trend 分布
    print(f'\n=== 入场时 mf/trend 分布 ===')
    for label, df_x in [('v1', df_v1), ('v2', df_v2)]:
        print(f'  {label}: cur_mf avg={df_x["cur_mf"].mean():.0f} 中位={df_x["cur_mf"].median():.0f}')
        print(f'      cur_trend avg={df_x["cur_trend"].mean():.1f} 中位={df_x["cur_trend"].median():.1f}')

    # 跑两版本
    rows = {'v1': [], 'v2': []}
    for label, df_e in [('v1', df_v1), ('v2', df_v2)]:
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e = code_ends[ci]
            max_end = min(e - 1, gi + MAX_TRACK)
            _, _, ret, legs = simulate_d6u1(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)
            rows[label].append({'date': ev['date'], 'code': ev['code'], 'ret': ret, 'legs': legs,
                                  'cur_mf': ev['cur_mf'], 'cur_trend': ev['cur_trend'],
                                  'pool_min_retail': ev['pool_min_retail']})

    print(f'\n=== 总览 ===\n')
    print(f'  {"版本":<8} {"事件":>6} {"avg_ret":>9} {"win%":>7} {"max":>7} {"min":>7} {"avg_legs":>8}')
    for label in ['v1', 'v2']:
        df_x = pd.DataFrame(rows[label])
        avg = df_x['ret'].mean()
        win = (df_x['ret']>0).mean()*100
        mx = df_x['ret'].max()
        mn = df_x['ret'].min()
        legs = df_x['legs'].mean()
        print(f'  {label:<8} {len(df_x):>6,} {avg:>+8.2f}% {win:>6.1f}% {mx:>+6.1f}% {mn:>+6.1f}% {legs:>7.1f}')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'\n=== 跨段对比 ===')
    print(f'  {"段":<14} {"v1 n/ret":<22} {"v2 n/ret":<22}')
    df1 = pd.DataFrame(rows['v1'])
    df2 = pd.DataFrame(rows['v2'])
    for w in WINDOWS:
        s1 = df1[(df1['date'] >= w[1]) & (df1['date'] < w[2])]
        s2 = df2[(df2['date'] >= w[1]) & (df2['date'] < w[2])]
        print(f'  {w[0]:<14} {len(s1):>4} {s1["ret"].mean():>+6.2f}% {(s1["ret"]>0).mean()*100:>5.1f}% '
              f'  {len(s2):>4} {s2["ret"].mean():>+6.2f}% {(s2["ret"]>0).mean()*100:>5.1f}%')

    # ============ 暴涨股案例 ============
    print(f'\n\n=== 暴涨股案例 (v1 入场, ret>500%) 通达信复盘 ===\n')
    df_extreme = df1.sort_values('ret', ascending=False).head(20)
    print(f'{"代码":<8} {"名称":<10} {"入场日":<12} {"当日 mf":>7} {"当日 trend":>9} '
          f'{"波段后 ret%":>10} {"legs":>4}')
    for _, r in df_extreme.iterrows():
        name = code2name.get(r['code'], '?')
        print(f'{r["code"]:<8} {str(name)[:8]:<10} {r["date"]:<12} '
              f'{r["cur_mf"]:>+7.0f} {r["cur_trend"]:>+8.1f}  '
              f'{r["ret"]:>+9.1f}% {r["legs"]:>3}')

    # baseline 在这些上的表现
    print(f'\n=== 这 20 个 case 的 baseline 表现 ===')
    print(f'{"代码":<8} {"入场":<12} {"D6+U1":>10} {"baseline":>10} {"diff":>9}')
    for _, r in df_extreme.iterrows():
        gi = int(df_v1[(df_v1['code']==r['code']) & (df_v1['date']==r['date'])].iloc[0]['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        # baseline
        bp = close_arr[gi]
        cross_count = 0
        running_max = trend_arr[gi]
        base_ret = 0
        for k in range(gi + 1, max_end + 1):
            days = k - gi
            if not np.isnan(trend_arr[k]):
                running_max = max(running_max, trend_arr[k])
            if not np.isnan(trend_arr[k]) and trend_arr[k] < 11:
                base_ret = (close_arr[k]/bp-1)*100; break
            if running_max >= 89 and trend_arr[k] < 89 and trend_arr[k-1] >= 89:
                cross_count += 1
                if cross_count >= 2:
                    base_ret = (close_arr[k]/bp-1)*100; break
            if days >= 20:
                seg = trend_arr[gi:k+1]
                valid = seg[~np.isnan(seg)]
                if len(valid) > 0 and valid.max() < 89:
                    base_ret = (close_arr[k]/bp-1)*100; break
        else:
            base_ret = (close_arr[max_end]/bp-1)*100
        print(f'{r["code"]:<8} {r["date"]:<12} {r["ret"]:>+9.1f}% {base_ret:>+9.1f}% {r["ret"]-base_ret:>+8.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
