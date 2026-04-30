# -*- coding: utf-8 -*-
"""卖点优化 — 多层卖点对比

S0 (现有 baseline): bull_2nd / TS20 / 60d
S1: trend<11 + bull_2nd + 60d (加保底硬卖)
S2: trend 第 1 次下穿 89 + 浮盈高点回撤 1/3 + trend<11 + 60d
S3: trend 第 1 次下穿 89 + 浮盈回撤 1/2 + trend<11 + 60d
S4: bull_1st (第 1 次下穿 89, 简单粗暴) + 60d
S5: bull_1st AND 浮盈>+15% (短打), 否则等 bull_2nd
S6: trend<60 (持续衰减破位) + 60d

入场: retail<-250 + mf 上穿 50 + 巽日 (T1b 思路)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
LOOKBACK = 30


def find_signals(arrays):
    """T1b: retail<-250 池中 + mf 上穿 50 + 巽日, 5 日窗口"""
    code_starts = arrays['starts']; code_ends = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d = arrays['stk_d']; mkt_y = arrays['mkt_y']
    date = arrays['date']; code = arrays['code']

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_trigger_i = -999

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)

            if not in_pool and cur_below and not prev_below:
                in_pool = True

            if in_pool and mf_cross_up:
                last_trigger_i = i

            if in_pool and (i - last_trigger_i <= 5):
                if mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                    events.append({
                        'date': date[gi], 'code': code[gi],
                        'buy_idx_global': gi,
                        'pool_min_retail': retail[gi],  # 简化, 不深度算
                    })
                    in_pool = False

            last_mf = mf[gi]
            prev_below = cur_below

    return pd.DataFrame(events)


def sell_S0(buy_idx, td, close, max_end):
    """bull_2nd / TS20 / 60d"""
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


def sell_S1(buy_idx, td, close, max_end):
    """trend<11 + bull_2nd + 60d"""
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td_below_11', (close[k]/bp-1)*100
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close[k]/bp-1)*100
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_S2(buy_idx, td, close, max_end, retrace_frac=1/3):
    """trend 第1次下穿89 后 进入保盈模式
    保盈: 浮盈 >+15%, 从浮盈高点回撤 > retrace_frac → 卖
    + trend<11 + 60d
    """
    bp = close[buy_idx]
    crossed_89 = False  # 是否曾经 trend>=89
    descend_from_89 = False  # 是否第 1 次下穿 89 已经发生
    high_ret = 0  # 浮盈高点
    high_ret_idx = -1

    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        cur_ret = (close[k]/bp-1)*100

        if not np.isnan(td[k]) and td[k] >= 89:
            crossed_89 = True
        if crossed_89 and not np.isnan(td[k]) and td[k] < 89 and td[k-1] >= 89:
            descend_from_89 = True

        if cur_ret > high_ret:
            high_ret = cur_ret
            high_ret_idx = k

        # 强卖: trend<11
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td_below_11', cur_ret

        # 保盈卖: 进入保盈模式 + 浮盈 >+15% + 回撤大
        if descend_from_89 and high_ret > 15:
            retrace = (high_ret - cur_ret) / max(high_ret, 1)
            if retrace > retrace_frac:
                return k, f'protect_{int(retrace_frac*100)}', cur_ret

        if days >= MAX_HOLD:
            return k, 'timeout', cur_ret
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_S4(buy_idx, td, close, max_end):
    """bull_1st: trend 第 1 次下穿 89 就卖 + 60d"""
    bp = close[buy_idx]
    crossed_89 = False
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]) and td[k] >= 89:
            crossed_89 = True
        if crossed_89 and not np.isnan(td[k]) and td[k] < 89 and td[k-1] >= 89:
            return k, 'bull_1st', (close[k]/bp-1)*100
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_S5(buy_idx, td, close, max_end):
    """bull_1st AND 浮盈>+15%, 否则等 bull_2nd"""
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        cur_ret = (close[k]/bp-1)*100
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 1 and cur_ret >= 15:
                return k, 'bull_1st+15', cur_ret
            if cross_count >= 2:
                return k, 'bull_2nd', cur_ret
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td_below_11', cur_ret
        if days >= MAX_HOLD:
            return k, 'timeout', cur_ret
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_S6(buy_idx, td, close, max_end):
    """trend 持续 < 60 + 60d"""
    bp = close[buy_idx]
    crossed_89 = False
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]) and td[k] >= 89:
            crossed_89 = True
        if crossed_89 and not np.isnan(td[k]) and td[k] < 60:
            return k, 'td_below_60', (close[k]/bp-1)*100
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td_below_11', (close[k]/bp-1)*100
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def main():
    t0 = time.time()
    print('=== 卖点优化对比: 多层卖点 ===\n')

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

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'mf': df['main_force'].to_numpy().astype(np.float64),
        'stk_d': df['stk_d'].to_numpy(),
        'mkt_y': df['mkt_y'].to_numpy(),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    df_e = find_signals(arrays)
    print(f'  T1b 信号: {len(df_e):,}')

    # 6 种卖点模拟
    rows = {label: [] for label in ['S0', 'S1', 'S2_33', 'S2_50', 'S4', 'S5', 'S6']}
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)
        for label, fn in [
            ('S0', sell_S0),
            ('S1', sell_S1),
            ('S4', sell_S4),
            ('S5', sell_S5),
            ('S6', sell_S6),
        ]:
            si, r, ret = fn(gi, trend_arr, close_arr, max_end)
            rows[label].append({'date': ev['date'], 'code': ev['code'],
                                  'days': si - gi, 'reason': r, 'ret_pct': ret})
        # S2 with retrace_frac=1/3
        si, r, ret = sell_S2(gi, trend_arr, close_arr, max_end, 1/3)
        rows['S2_33'].append({'date': ev['date'], 'code': ev['code'],
                                'days': si - gi, 'reason': r, 'ret_pct': ret})
        # S2 with retrace_frac=1/2
        si, r, ret = sell_S2(gi, trend_arr, close_arr, max_end, 1/2)
        rows['S2_50'].append({'date': ev['date'], 'code': ev['code'],
                                'days': si - gi, 'reason': r, 'ret_pct': ret})

    print('\n=== 单事件级对比 ===\n')
    print(f'  {"模式":<28} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>5} {"max":>7} {"min":>7}')
    for label in ['S0', 'S1', 'S2_33', 'S2_50', 'S4', 'S5', 'S6']:
        df_x = pd.DataFrame(rows[label])
        avg = df_x['ret_pct'].mean()
        win = (df_x['ret_pct']>0).mean()*100
        med = df_x['ret_pct'].median()
        days = df_x['days'].mean()
        mx = df_x['ret_pct'].max()
        mn = df_x['ret_pct'].min()
        desc = {
            'S0': 'S0 现有 bull/TS20/60d',
            'S1': 'S1 + trend<11 强卖',
            'S2_33': 'S2 第1次穿89 + 回撤1/3',
            'S2_50': 'S2 第1次穿89 + 回撤1/2',
            'S4': 'S4 bull_1st 简单',
            'S5': 'S5 1st+15% / 2nd',
            'S6': 'S6 trend<60 + trend<11',
        }
        print(f'  {desc[label]:<28} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}% {days:>4.1f}d '
              f'{mx:>+6.1f}% {mn:>+6.1f}%')

    # 神火 / 顺丰 各模式表现
    print('\n=== 神火 vs 顺丰 各模式 ===\n')
    for code, dt in [('000933', '2016-02-17'), ('002352', '2016-01-19')]:
        print(f'  {code} {dt}:')
        for label in ['S0', 'S1', 'S2_33', 'S2_50', 'S4', 'S5', 'S6']:
            df_x = pd.DataFrame(rows[label])
            sub = df_x[(df_x['code'] == code) & (df_x['date'] == dt)]
            if len(sub):
                r = sub.iloc[0]
                print(f'    {label}: {r["ret_pct"]:>+7.2f}% / {r["days"]:>3}d / {r["reason"]}')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print('\n=== 跨段 avg_ret ===\n')
    print(f'  {"段":<12}', end='')
    for label in ['S0', 'S1', 'S2_33', 'S5', 'S4', 'S6']:
        print(f'{label:>8}', end='')
    print()
    for w in WINDOWS:
        print(f'  {w[0]:<12}', end='')
        for label in ['S0', 'S1', 'S2_33', 'S5', 'S4', 'S6']:
            df_x = pd.DataFrame(rows[label])
            sub = df_x[(df_x['date'] >= w[1]) & (df_x['date'] < w[2])]
            if len(sub):
                print(f'{sub["ret_pct"].mean():>+7.2f}%', end='')
            else:
                print(f'{"-":>8}', end='')
        print()

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
