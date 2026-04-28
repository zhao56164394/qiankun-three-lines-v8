# -*- coding: utf-8 -*-
"""Step 33 — M3 多变量优化扫描

baseline:
  M3 单独 = 上一日 d_trend>89, 今日 ≤89 卖出 (60 日超时)

优化变体 (8 种):
  V1 — M3 + 浮亏 -8% 止损 (任一触发)
  V2 — M3 + 浮亏 -10% 止损
  V3 — M3 + 时间止损 (持仓 30 日 d_trend 仍未到 89 → 强卖)
  V4 — M3 + 时间止损 (持仓 20 日)
  V5 — M3 + 假突破识别 (入场 5 日 d_trend 不创新高 → 卖)
  V6 — M3 + 假突破识别 (入场 10 日 d_gua 没出现过 111 → 卖)
  V7 — M3 + 主力转负 (累计 mf 转负 → 卖)
  V8 — 第 2 次穿 89 (B2 bull, 让赚钱的多跑)

对每个变体: 期望/胜率/主升期望/假期望/持仓 + walk-forward
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
QIAN_RUN = 10
HARD_TIMEOUT = 60
REGIME_Y = '000'

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

AVOID = [
    ('mkt_d', '000'), ('mkt_d', '001'), ('mkt_d', '100'), ('mkt_d', '101'),
    ('stk_y', '001'), ('stk_y', '011'),
    ('stk_m', '101'), ('stk_m', '110'), ('stk_m', '111'),
]


# ===== 卖点函数 =====

def m3_baseline(buy_idx, td, cl, end_idx):
    """V0 — M3 单独"""
    n = len(td); end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
    return end, 'timeout'


def m3_with_stop_loss(buy_idx, td, cl, end_idx, sl_pct):
    """V1/V2 — M3 OR 浮亏止损"""
    n = len(td); end = min(end_idx, n - 1)
    buy_close = cl[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        # 浮亏止损
        if (cl[k] / buy_close - 1) * 100 <= sl_pct:
            return k, 'stop_loss'
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
    return end, 'timeout'


def m3_with_time_stop(buy_idx, td, cl, end_idx, time_limit):
    """V3/V4 — M3 OR 时间止损 (持仓 N 日 d_trend 仍未到 89 → 强卖)"""
    n = len(td); end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
        # 时间止损: 持仓 ≥ time_limit 且 d_trend 未到 89
        if k - buy_idx >= time_limit:
            seg_max = np.nanmax(td[buy_idx:k+1])
            if seg_max < 89:
                return k, 'time_stop'
    return end, 'timeout'


def m3_with_no_new_high(buy_idx, td, cl, end_idx, lookback):
    """V5 — M3 OR 入场后 N 日 d_trend 不创新高 (=买入日 trend 仍是 N 日内最高) → 假突破"""
    n = len(td); end = min(end_idx, n - 1)
    buy_trend = td[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
        if k - buy_idx == lookback:
            seg_max = np.nanmax(td[buy_idx:k+1])
            if seg_max <= buy_trend:  # 没创新高
                return k, 'no_new_high'
    return end, 'timeout'


def m3_with_no_qian(buy_idx, gua, td, cl, end_idx, lookback):
    """V6 — M3 OR 入场后 N 日 d_gua 没出现过 111 → 假突破"""
    n = len(gua); end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
        if k - buy_idx == lookback:
            if not (gua[buy_idx:k+1] == '111').any():
                return k, 'no_qian'
    return end, 'timeout'


def m3_with_mf_neg(buy_idx, mf, td, cl, end_idx, mf_thresh):
    """V7 — M3 OR 累计 mf 转负 (5 日累计 < mf_thresh) → 主力撤"""
    n = len(td); end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        if not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                return k, 'cross89'
        if k - buy_idx >= 5:
            mf_5d = mf[k-4:k+1].sum()
            if mf_5d < mf_thresh:
                return k, 'mf_neg'
    return end, 'timeout'


def b2_bull(buy_idx, td, cl, end_idx):
    """V8 — 第 2 次下穿 89"""
    n = len(td); end = min(end_idx, n - 1)
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        if np.isnan(td[k]): continue
        running_max = max(running_max, td[k])
        if k > 0 and not np.isnan(td[k-1]):
            if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
                cross_count += 1
                if cross_count == 2:
                    return k, 'bull_2nd'
    return end, 'timeout'


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫 v2 买入
    print(f'\n=== 扫 v2 买入事件 ===')
    avoid_arr_map = {'mkt_d': mkt_d_arr, 'stk_y': stk_y_arr, 'stk_m': stk_m_arr}
    buy_events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + HARD_TIMEOUT + 5: continue
        gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - HARD_TIMEOUT):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            avoid = False
            for col, val in AVOID:
                if avoid_arr_map[col][gi] == val:
                    avoid = True; break
            if avoid: continue
            score = 0
            if mkt_m_arr[gi] == '100': score += 1
            if mkt_d_arr[gi] == '011': score += 1
            if mf_arr[gi] > 100: score += 1
            if stk_m_arr[gi] == '010': score += 1
            if score < 2: continue
            buy_events.append((gi, ci))
    print(f'  v2 买入: {len(buy_events):,}')

    # 模拟所有变体
    print(f'\n=== 模拟 8 种变体 ===')
    t1 = time.time()

    variants = [
        ('V0_baseline', lambda b, td, gua, mf, cl, end: m3_baseline(b, td, cl, end)),
        ('V1_SL_-8', lambda b, td, gua, mf, cl, end: m3_with_stop_loss(b, td, cl, end, -8.0)),
        ('V2_SL_-10', lambda b, td, gua, mf, cl, end: m3_with_stop_loss(b, td, cl, end, -10.0)),
        ('V3_TS_30', lambda b, td, gua, mf, cl, end: m3_with_time_stop(b, td, cl, end, 30)),
        ('V4_TS_20', lambda b, td, gua, mf, cl, end: m3_with_time_stop(b, td, cl, end, 20)),
        ('V5_NoNH_5', lambda b, td, gua, mf, cl, end: m3_with_no_new_high(b, td, cl, end, 5)),
        ('V6_NoQ_10', lambda b, td, gua, mf, cl, end: m3_with_no_qian(b, gua, td, cl, end, 10)),
        ('V7_MF_-100', lambda b, td, gua, mf, cl, end: m3_with_mf_neg(b, mf, td, cl, end, -100)),
        ('V8_bull_2nd', lambda b, td, gua, mf, cl, end: b2_bull(b, td, cl, end)),
    ]

    results = {v[0]: [] for v in variants}

    for gi, ci in buy_events:
        s = code_starts[ci]; e = code_ends[ci]
        local_buy = gi - s
        cl_seg = close_arr[s:e]
        gua_seg = stk_d_arr[s:e]
        td_seg = trend_arr[s:e]
        mf_seg = mf_arr[s:e]
        n_local = len(gua_seg)
        max_end = min(local_buy + HARD_TIMEOUT, n_local - 1)
        buy_close = cl_seg[local_buy]
        buy_date = date_arr[gi]
        n_qian_60 = (gua_seg[local_buy:max_end+1] == '111').sum()
        is_zsl = n_qian_60 >= QIAN_RUN

        for label, fn in variants:
            sl, ex = fn(local_buy, td_seg, gua_seg, mf_seg, cl_seg, max_end)
            results[label].append({
                'date': buy_date, 'is_zsl': is_zsl,
                'hold': sl - local_buy, 'ret': (cl_seg[sl] / buy_close - 1) * 100,
                'exit': ex,
            })

    print(f'  完成 {time.time()-t1:.1f}s')

    # 全样本对比
    print(f'\n## 8 变体对比 ({len(buy_events):,} v2 买入)')
    print(f'  {"变体":<14} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"持仓":>5} {"主升期望":>9} {"假期望":>8} {"主升n%":>7}')
    print('  ' + '-' * 80)
    for label, _ in variants:
        d = pd.DataFrame(results[label])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret'] > 0).mean() * 100
        hold = d['hold'].mean()
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        zsl_pct = len(zsl) / len(d) * 100
        print(f'  {label:<14} {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>4.1f} '
              f'{zsl["ret"].mean():>+7.2f}% {fake["ret"].mean():>+7.2f}% {zsl_pct:>5.1f}%')

    # 退出类型
    print(f'\n## 退出类型分布')
    for label, _ in variants:
        d = pd.DataFrame(results[label])
        ex_dist = d['exit'].value_counts(normalize=True) * 100
        print(f'  {label:<14}  ', end='')
        for k, v in ex_dist.items():
            print(f'{k}={v:.0f}%  ', end='')
        print()

    # walk-forward
    print(f'\n## walk-forward 各段期望%')
    print(f'  {"段":<14}', end='')
    for label, _ in variants:
        print(f' {label:>11}', end='')
    print()
    print('  ' + '-' * 130)
    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for label, _ in variants:
            d = pd.DataFrame(results[label])
            seg = d[(d['date'] >= w[1]) & (d['date'] < w[2])]
            if len(seg) < 30:
                print(f' {"--":>11}', end='')
            else:
                print(f' {seg["ret"].mean():>+10.2f}%', end='')
        print()

    # 关键: 主升 vs 假突破 拆解
    print(f'\n## 主升/假突破 拆解 (主升期望最高 + 假期望最浅 = 最佳)')
    print(f'  {"变体":<14} {"主升期望":>9} {"主升持仓":>9} {"假期望":>8} {"假持仓":>8}')
    print('  ' + '-' * 60)
    for label, _ in variants:
        d = pd.DataFrame(results[label])
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        print(f'  {label:<14} {zsl["ret"].mean():>+7.2f}% {zsl["hold"].mean():>7.1f} '
              f'{fake["ret"].mean():>+7.2f}% {fake["hold"].mean():>7.1f}')


if __name__ == '__main__':
    main()
