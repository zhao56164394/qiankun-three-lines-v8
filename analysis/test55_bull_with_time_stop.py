# -*- coding: utf-8 -*-
"""Step 34 — V8 (第2次穿89) + 时间止损 hybrid 优化

V8 单独: +11.43% / 主升 +12.53% / 假亏 -16.90%
V4 单独: +6.96%  / 主升 +7.37%  / 假亏 -3.39%

V9 hybrid (尝试结合):
  主路: 第2次穿89 卖
  但若持仓 ≥ N 日 且 d_trend 从未到过 89 → 强卖 (假突破快出)
  N 试: 15, 20, 25, 30
"""
import os, sys, io, time
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
QIAN_RUN = 10
HARD_TIMEOUT = 60
REGIME_Y = '000'

WINDOWS = [
    ('w1_2018', '2018-01-01', '2019-01-01'),
    ('w2_2019', '2019-01-01', '2020-01-01'),
    ('w4_2021', '2021-01-01', '2022-01-01'),
    ('w5_2022', '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

AVOID = [
    ('mkt_d', '000'), ('mkt_d', '001'), ('mkt_d', '100'), ('mkt_d', '101'),
    ('stk_y', '001'), ('stk_y', '011'),
    ('stk_m', '101'), ('stk_m', '110'), ('stk_m', '111'),
]


def b2_bull(buy_idx, td, cl, end_idx):
    """V8 单独"""
    n = len(td); end = min(end_idx, n - 1)
    cross_count = 0; running_max = td[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        if np.isnan(td[k]): continue
        running_max = max(running_max, td[k])
        if k > 0 and not np.isnan(td[k-1]):
            if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
                cross_count += 1
                if cross_count == 2:
                    return k, 'bull_2nd'
    return end, 'timeout'


def b2_bull_with_time_stop(buy_idx, td, cl, end_idx, time_limit):
    """V9 — 第2次穿89 OR (持仓≥N 日 且 d_trend 从未到过 89 → 强卖)"""
    n = len(td); end = min(end_idx, n - 1)
    cross_count = 0; running_max = td[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if k > 0 and not np.isnan(td[k-1]):
            if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
                cross_count += 1
                if cross_count == 2:
                    return k, 'bull_2nd'
        # 时间止损
        if k - buy_idx >= time_limit:
            seg_max = np.nanmax(td[buy_idx:k+1])
            if seg_max < 89:
                return k, 'time_stop'
    return end, 'timeout'


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua','m_gua','y_gua']: g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua':'stk_d','m_gua':'stk_m','y_gua':'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date','d_gua','m_gua','y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date','mkt_d','mkt_m','mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date','code','close','main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date','code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','stk_d','mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:]!=code_arr[:-1]]
    code_starts = np.where(code_change)[0]; code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫 v2 买入
    avoid_arr_map = {'mkt_d':mkt_d_arr,'stk_y':stk_y_arr,'stk_m':stk_m_arr}
    buy_events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e-s < LOOKBACK + HARD_TIMEOUT + 5: continue
        gua = stk_d_arr[s:e]; n = len(gua)
        for i in range(LOOKBACK, n - HARD_TIMEOUT):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            avoid = False
            for col, val in AVOID:
                if avoid_arr_map[col][gi] == val: avoid = True; break
            if avoid: continue
            score = 0
            if mkt_m_arr[gi] == '100': score += 1
            if mkt_d_arr[gi] == '011': score += 1
            if mf_arr[gi] > 100: score += 1
            if stk_m_arr[gi] == '010': score += 1
            if score < 2: continue
            buy_events.append((gi, ci))
    print(f'\n  v2 买入: {len(buy_events):,}')

    # 模拟
    print(f'\n=== 模拟 ===')
    t1 = time.time()
    variants = [
        ('V8_bull', None, None),
        ('V9_bull_TS15', 'time_stop', 15),
        ('V9_bull_TS20', 'time_stop', 20),
        ('V9_bull_TS25', 'time_stop', 25),
        ('V9_bull_TS30', 'time_stop', 30),
        ('V9_bull_TS35', 'time_stop', 35),
    ]
    results = {v[0]:[] for v in variants}

    for gi, ci in buy_events:
        s = code_starts[ci]; e = code_ends[ci]
        local_buy = gi - s
        cl_seg = close_arr[s:e]; gua_seg = stk_d_arr[s:e]
        td_seg = trend_arr[s:e]
        max_end = min(local_buy + HARD_TIMEOUT, len(gua_seg)-1)
        buy_close = cl_seg[local_buy]; buy_date = date_arr[gi]
        n_qian_60 = (gua_seg[local_buy:max_end+1] == '111').sum()
        is_zsl = n_qian_60 >= QIAN_RUN

        for label, mode, param in variants:
            if mode is None:
                sl, ex = b2_bull(local_buy, td_seg, cl_seg, max_end)
            else:
                sl, ex = b2_bull_with_time_stop(local_buy, td_seg, cl_seg, max_end, param)
            results[label].append({
                'date': buy_date, 'is_zsl': is_zsl,
                'hold': sl-local_buy, 'ret': (cl_seg[sl]/buy_close-1)*100, 'exit': ex,
            })
    print(f'  完成 {time.time()-t1:.1f}s')

    print(f'\n## V8 + 时间止损变体')
    print(f'  {"变体":<16} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"持仓":>5} {"主升期望":>9} {"假期望":>8}')
    print('  ' + '-' * 70)
    for label, _, _ in variants:
        d = pd.DataFrame(results[label])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret']>0).mean()*100
        hold = d['hold'].mean()
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        print(f'  {label:<16} {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>4.1f} '
              f'{zsl["ret"].mean():>+7.2f}% {fake["ret"].mean():>+7.2f}%')

    print(f'\n## 退出类型分布')
    for label, _, _ in variants:
        d = pd.DataFrame(results[label])
        ex_dist = d['exit'].value_counts(normalize=True) * 100
        print(f'  {label:<16}  ', end='')
        for k, v in ex_dist.items():
            print(f'{k}={v:.0f}%  ', end='')
        print()

    print(f'\n## walk-forward')
    print(f'  {"段":<14}', end='')
    for label, _, _ in variants:
        print(f' {label:>13}', end='')
    print()
    print('  ' + '-' * 100)
    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for label, _, _ in variants:
            d = pd.DataFrame(results[label])
            seg = d[(d['date']>=w[1])&(d['date']<w[2])]
            if len(seg) < 30: print(f' {"--":>13}', end='')
            else: print(f' {seg["ret"].mean():>+12.2f}%', end='')
        print()


if __name__ == '__main__':
    main()
