# -*- coding: utf-8 -*-
"""Step 29 — 坤 regime v2 买入事件 × 6 种候选卖点机制扫描

设计:
  对每个 v2 买入事件 (避雷后 + score ≥2), 同时模拟 6 种卖点
  比较: 期望/胜率/持仓天数/主升浪保留率

6 种卖点:
  M1: 乾→其他 (一日乘乘) — 上日 d=111, 今日≠111
  M2: 乾→坤/坎/艮 (质变才卖) — 上日=111, 今日∈{000,001,010,011}
  M3: 趋势线下穿 89 — d_trend 上日>89, 今日≤89
  M4: 主力线 mf 转负 — mf 上日>0, 今日<-50
  M5: 散户线 sanhu 转正大量 — sanhu 上日<0, 今日>50
  M6: 30 日固定窗口 (baseline)

通用兜底: 持仓 60 日强卖
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
HARD_TIMEOUT = 60  # 通用兜底
REGIME_Y = '000'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

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
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # === 扫坤 regime + 避雷 + score ≥2 的买入事件 ===
    print(f'\n=== 扫 v2 买入事件 (避雷 + score≥2) ===')
    t1 = time.time()
    avoid_arr_map = {'mkt_d': mkt_d_arr, 'mkt_m': mkt_m_arr, 'mkt_y': mkt_y_arr,
                     'stk_d': stk_d_arr, 'stk_m': stk_m_arr, 'stk_y': stk_y_arr}

    buy_events = []  # (global_idx, code_seg)
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + HARD_TIMEOUT + 5: continue
        gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - HARD_TIMEOUT):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            # 避雷
            avoid = False
            for col, val in AVOID:
                if avoid_arr_map[col][gi] == val:
                    avoid = True; break
            if avoid: continue
            # score
            score = 0
            if mkt_m_arr[gi] == '100': score += 1
            if mkt_d_arr[gi] == '011': score += 1
            if mf_arr[gi] > 100: score += 1
            if stk_m_arr[gi] == '010': score += 1
            if score < 2: continue

            buy_events.append((gi, ci, score))

    print(f'  v2 买入事件: {len(buy_events):,}, {time.time()-t1:.1f}s')

    # === 对每个买点同时模拟 6 种卖点 ===
    print(f'\n=== 模拟 6 种卖点机制 ===')
    t2 = time.time()
    results = {f'M{i}': [] for i in range(1, 7)}

    for gi, ci, score in buy_events:
        s = code_starts[ci]; e = code_ends[ci]
        local_buy = gi - s
        cl_seg = close_arr[s:e]
        gua_seg = stk_d_arr[s:e]
        td_seg = trend_arr[s:e]
        mf_seg = mf_arr[s:e]
        sanhu_seg = sanhu_arr[s:e]
        n_local = len(gua_seg)

        max_end = min(local_buy + HARD_TIMEOUT, n_local - 1)
        buy_close = cl_seg[local_buy]
        buy_date = date_arr[gi]
        # 算 60 日内乾天数 (用于判断主升浪)
        n_qian_60 = (gua_seg[local_buy:max_end+1] == '111').sum()
        is_zsl = n_qian_60 >= QIAN_RUN

        # M1: 乾→其他 (一日乘乘)
        sell_local = None
        for k in range(local_buy + 1, max_end + 1):
            if gua_seg[k-1] == '111' and gua_seg[k] != '111':
                sell_local = k; break
        if sell_local is None: sell_local = max_end
        results['M1'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'qian_change' if sell_local < max_end else 'timeout',
        })

        # M2: 乾→坤/坎/艮 (质变才卖)
        sell_local = None
        for k in range(local_buy + 1, max_end + 1):
            if gua_seg[k-1] == '111' and gua_seg[k] in ['000', '001', '010', '011']:
                sell_local = k; break
        if sell_local is None: sell_local = max_end
        results['M2'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'qian_quality' if sell_local < max_end else 'timeout',
        })

        # M3: 下穿 89
        sell_local = None
        for k in range(local_buy + 1, max_end + 1):
            if td_seg[k-1] > 89 and td_seg[k] <= 89:
                sell_local = k; break
        if sell_local is None: sell_local = max_end
        results['M3'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'cross_89' if sell_local < max_end else 'timeout',
        })

        # M4: 主力 mf 转负
        sell_local = None
        for k in range(local_buy + 1, max_end + 1):
            if mf_seg[k-1] > 0 and mf_seg[k] < -50:
                sell_local = k; break
        if sell_local is None: sell_local = max_end
        results['M4'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'mf_neg' if sell_local < max_end else 'timeout',
        })

        # M5: 散户 sanhu 暴涨 (>50)
        sell_local = None
        for k in range(local_buy + 1, max_end + 1):
            if sanhu_seg[k-1] < 0 and sanhu_seg[k] > 50:
                sell_local = k; break
        if sell_local is None: sell_local = max_end
        results['M5'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'sanhu_surge' if sell_local < max_end else 'timeout',
        })

        # M6: 30 日固定 (baseline)
        sell_local = min(local_buy + 30, max_end)
        results['M6'].append({
            'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60),
            'hold': sell_local - local_buy, 'ret': (cl_seg[sell_local] / buy_close - 1) * 100,
            'exit': 'fixed_30',
        })

    print(f'  6 机制模拟完: {time.time()-t2:.1f}s')

    # === 对比 ===
    print(f'\n## 6 机制对比 (在 {len(buy_events):,} v2 买入事件上)')
    print(f'  {"机制":<6} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"均持仓":>6} {"最大%":>6} {"最小%":>7}')
    print('  ' + '-' * 60)
    for m in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        d = pd.DataFrame(results[m])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret'] > 0).mean() * 100
        hold = d['hold'].mean()
        print(f'  {m:<6} {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>5.1f} '
              f'{d["ret"].max():>+5.1f}% {d["ret"].min():>+6.1f}%')

    # 拆 主升浪 vs 假突破
    print(f'\n## 各机制在 主升浪股 (60日乾≥10) vs 假突破股 上的表现')
    print(f'  {"机制":<6} {"主升 期望":>9} {"主升 持仓":>9} {"假 期望":>8} {"假 持仓":>8}')
    print('  ' + '-' * 55)
    for m in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        d = pd.DataFrame(results[m])
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        print(f'  {m:<6} {zsl["ret"].mean():>+7.2f}% {zsl["hold"].mean():>7.1f} '
              f'{fake["ret"].mean():>+7.2f}% {fake["hold"].mean():>7.1f}')

    # 退出类型分布
    print(f'\n## 退出类型分布')
    for m in ['M1', 'M2', 'M3', 'M4', 'M5']:
        d = pd.DataFrame(results[m])
        exit_dist = d['exit'].value_counts(normalize=True) * 100
        print(f'  {m}: ', end='')
        for k, v in exit_dist.items():
            print(f'{k}={v:.0f}%  ', end='')
        print()

    # walk-forward
    print(f'\n## walk-forward 各段期望%')
    print(f'  {"段":<14}', end='')
    for m in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
        print(f' {m:>8}', end='')
    print()
    print('  ' + '-' * 80)
    for w in WINDOWS:
        # 每段
        print(f'  {w[0]:<14}', end='')
        for m in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']:
            d = pd.DataFrame(results[m])
            seg = d[(d['date'] >= w[1]) & (d['date'] < w[2])]
            if len(seg) < 30:
                print(f' {"--":>7}', end='')
            else:
                print(f' {seg["ret"].mean():>+6.2f}%', end='')
        print()


if __name__ == '__main__':
    main()
