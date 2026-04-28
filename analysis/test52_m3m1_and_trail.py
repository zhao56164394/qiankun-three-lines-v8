# -*- coding: utf-8 -*-
"""Step 31 — M3 AND 移动止损 / M1 AND 移动止损, 多变量扫

机制 (双重确认 AND):
  M3∩SL: 必须 (下穿89 已触发) AND (从最高点回撤 ≥ trail_pct%) → 卖
  M1∩SL: 必须 (乾→其他 已触发) AND (从最高点回撤 ≥ trail_pct%) → 卖

trail_pct 扫: 5%, 8%, 10%, 12%, 15%, 20%, 25%

加 baseline 对照:
  B2 bull (项目现成牛卖, 第二次穿89)
  M3 单独 (只看下穿89)
  M1 单独 (只看乾→其他)
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

TRAIL_PCTS = [5, 8, 10, 12, 15, 20, 25]


def sell_m3_and_trail(buy_idx, td, cl, end_idx, buy_close, trail_pct):
    """下穿89 AND 从最高点回撤 ≥ trail_pct% — 两个都满足才卖"""
    n = len(td)
    end = min(end_idx, n - 1)
    has_cross89 = False  # 是否触发过下穿89
    max_close = buy_close
    for k in range(buy_idx + 1, end + 1):
        max_close = max(max_close, cl[k])
        # 下穿89 触发
        if k > 0 and not np.isnan(td[k]) and not np.isnan(td[k-1]):
            if td[k-1] > 89 and td[k] <= 89:
                has_cross89 = True
        # 移动止损触发
        drawdown_pct = (cl[k] / max_close - 1) * 100
        trail_triggered = drawdown_pct <= -trail_pct
        # 两个都触发才卖
        if has_cross89 and trail_triggered:
            return k, 'and_both'
    return end, 'timeout'


def sell_m1_and_trail(buy_idx, gua, cl, end_idx, buy_close, trail_pct):
    """乾→其他 AND 从最高点回撤 ≥ trail_pct% — 两个都满足才卖"""
    n = len(gua)
    end = min(end_idx, n - 1)
    has_qian_change = False
    max_close = buy_close
    for k in range(buy_idx + 1, end + 1):
        max_close = max(max_close, cl[k])
        # 乾切触发
        if gua[k-1] == '111' and gua[k] != '111':
            has_qian_change = True
        # 移动止损触发
        drawdown_pct = (cl[k] / max_close - 1) * 100
        trail_triggered = drawdown_pct <= -trail_pct
        # 两个都触发才卖
        if has_qian_change and trail_triggered:
            return k, 'and_both'
    return end, 'timeout'


def sell_b2_bull(buy_idx, td, cl, end_idx):
    """B2 bull baseline: 第二次下穿89"""
    n = len(td)
    end = min(end_idx, n - 1)
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
            buy_events.append((gi, ci, score))
    print(f'  v2 买入: {len(buy_events):,}')

    # 模拟
    print(f'\n=== 模拟 M3∩trail / M1∩trail × {len(TRAIL_PCTS)} trail_pcts ===')
    t1 = time.time()
    results = {}

    # B2 bull baseline
    results['B2_bull'] = []
    for trail in TRAIL_PCTS:
        results[f'M3_AND_{trail}'] = []
        results[f'M1_AND_{trail}'] = []

    for gi, ci, score in buy_events:
        s = code_starts[ci]; e = code_ends[ci]
        local_buy = gi - s
        cl_seg = close_arr[s:e]
        gua_seg = stk_d_arr[s:e]
        td_seg = trend_arr[s:e]
        n_local = len(gua_seg)
        max_end = min(local_buy + HARD_TIMEOUT, n_local - 1)
        buy_close = cl_seg[local_buy]
        buy_date = date_arr[gi]
        n_qian_60 = (gua_seg[local_buy:max_end+1] == '111').sum()
        is_zsl = n_qian_60 >= QIAN_RUN

        common = {'date': buy_date, 'is_zsl': is_zsl}

        # B2 bull
        sl, exit_t = sell_b2_bull(local_buy, td_seg, cl_seg, max_end)
        results['B2_bull'].append({**common, 'hold': sl - local_buy,
                                     'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

        for trail in TRAIL_PCTS:
            # M3 AND trail
            sl, exit_t = sell_m3_and_trail(local_buy, td_seg, cl_seg, max_end, buy_close, trail)
            results[f'M3_AND_{trail}'].append({**common, 'hold': sl - local_buy,
                                                  'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})
            # M1 AND trail
            sl, exit_t = sell_m1_and_trail(local_buy, gua_seg, cl_seg, max_end, buy_close, trail)
            results[f'M1_AND_{trail}'].append({**common, 'hold': sl - local_buy,
                                                  'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

    print(f'  完成 {time.time()-t1:.1f}s')

    # === 输出 M3 AND trail ===
    print(f'\n## M3 AND trail (下穿89 + 移动止损 双触发)')
    print(f'  {"trail":<8} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"均持仓":>6} {"主升期望":>9} {"假期望":>8} {"AND触发":>8}')
    print('  ' + '-' * 75)
    for trail in TRAIL_PCTS:
        d = pd.DataFrame(results[f'M3_AND_{trail}'])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret'] > 0).mean() * 100
        hold = d['hold'].mean()
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        and_pct = (d['exit'] == 'and_both').mean() * 100
        print(f'  trail={trail}%  {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>5.1f} '
              f'{zsl["ret"].mean():>+7.2f}% {fake["ret"].mean():>+7.2f}% {and_pct:>6.1f}%')

    # === M1 AND trail ===
    print(f'\n## M1 AND trail (乾→其他 + 移动止损 双触发)')
    print(f'  {"trail":<8} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"均持仓":>6} {"主升期望":>9} {"假期望":>8} {"AND触发":>8}')
    print('  ' + '-' * 75)
    for trail in TRAIL_PCTS:
        d = pd.DataFrame(results[f'M1_AND_{trail}'])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret'] > 0).mean() * 100
        hold = d['hold'].mean()
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        and_pct = (d['exit'] == 'and_both').mean() * 100
        print(f'  trail={trail}%  {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>5.1f} '
              f'{zsl["ret"].mean():>+7.2f}% {fake["ret"].mean():>+7.2f}% {and_pct:>6.1f}%')

    # B2 bull
    d = pd.DataFrame(results['B2_bull'])
    ret_m = d['ret'].mean(); ret_med = d['ret'].median()
    win = (d['ret'] > 0).mean() * 100
    hold = d['hold'].mean()
    zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
    print(f'\n## B2 bull (基准, 项目现成第二次穿89)')
    print(f'  期望 {ret_m:+.2f}%, 中位 {ret_med:+.2f}%, 胜率 {win:.1f}%, 均持仓 {hold:.1f} 日')
    print(f'  主升期望 {zsl["ret"].mean():+.2f}%, 假期望 {fake["ret"].mean():+.2f}%')

    # walk-forward Top 几个
    print(f'\n## walk-forward 各段期望% (B2 + M3_AND × 3 + M1_AND × 3)')
    cols = ['B2_bull', 'M3_AND_8', 'M3_AND_12', 'M3_AND_20', 'M1_AND_8', 'M1_AND_12', 'M1_AND_20']
    print(f'  {"段":<14}', end='')
    for c in cols:
        print(f' {c[:11]:>11}', end='')
    print()
    print('  ' + '-' * 100)
    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        for c in cols:
            d = pd.DataFrame(results[c])
            seg = d[(d['date'] >= w[1]) & (d['date'] < w[2])]
            if len(seg) < 30:
                print(f' {"--":>11}', end='')
            else:
                print(f' {seg["ret"].mean():>+10.2f}%', end='')
        print()


if __name__ == '__main__':
    main()
