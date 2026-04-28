# -*- coding: utf-8 -*-
"""Step 12 — 单卦分治 walk-forward

对 6 类卦 (大盘 y/m/d, 个股 y/m/d) 每个的 8 态各做 walk-forward:
  - 7 段窗口 (2018/2019/2020/2021/2022/2023-24/2025-26)
  - 每态在每段算: n, 期望 ret%, lift = ret - 段 baseline
  - 判定: ≥5 段有效 + ≥5 段 lift ≥+3% + ≤1 段 ≤-3% → ★ 真稳定

输出: 6 卦类 × 8 态 = 48 个分治单元, 列出 ★ 真稳定者 + 各段细节
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOW = 11
HIGH = 89
TIMEOUT = 250

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]
SEG_LABELS = [w[0] for w in WINDOWS]

MIN_N_SEG = 30
LIFT_PASS = 3.0   # 段内 lift 阈值
LIFT_FAIL = -3.0


def main():
    t0 = time.time()
    print('=== 加载数据 + 扫上穿11事件 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float32)
    mkt_y = df['mkt_y'].to_numpy(); mkt_m = df['mkt_m'].to_numpy(); mkt_d = df['mkt_d'].to_numpy()
    stk_y = df['y_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy(); stk_d = df['d_gua'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    events = []
    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; n = len(td)
        buy_mask = (td[:-1] < LOW) & (td[1:] >= LOW)
        buy_idx = np.where(buy_mask)[0] + 1
        sell_mask = (td[:-1] > HIGH) & (td[1:] <= HIGH)
        sell_idx = np.where(sell_mask)[0] + 1

        last_exit = -1
        for b in buy_idx:
            if b <= last_exit: continue
            global_b = s + b
            future_sells = sell_idx[sell_idx > b]
            if len(future_sells) == 0:
                hold = n - 1 - b
                if hold > TIMEOUT:
                    s_local = b + TIMEOUT
                    ret = cl[s_local] / cl[b] - 1
                    last_exit = s_local
                else:
                    continue
            else:
                s_local = future_sells[0]
                hold = s_local - b
                if hold > TIMEOUT:
                    s_local = b + TIMEOUT
                ret = cl[s_local] / cl[b] - 1
                last_exit = s_local

            events.append((date_arr[global_b], ret * 100,
                           mkt_y[global_b], mkt_m[global_b], mkt_d[global_b],
                           stk_y[global_b], stk_m[global_b], stk_d[global_b]))

    df_e = pd.DataFrame(events, columns=['date', 'ret', 'mkt_y', 'mkt_m', 'mkt_d',
                                          'stk_y', 'stk_m', 'stk_d'])
    print(f'  事件: {len(df_e):,}, {time.time()-t1:.1f}s')

    df_e['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label
    df_e = df_e[df_e['seg'] != ''].copy()

    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret'].mean() if len(seg) > 0 else 0

    print(f'\n## 段 baseline:')
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        print(f'  {w[0]:<14} n={len(seg):>6,}  期望 {seg_baselines[w[0]]:>+6.2f}%')

    # === 6 卦类 × 8 态 ×7 段 ===
    GUA_COLS = [
        ('mkt_y', '大盘 y_gua (regime)'),
        ('mkt_m', '大盘 m_gua (趋势)'),
        ('mkt_d', '大盘 d_gua (验证)'),
        ('stk_y', '个股 y_gua (选股)'),
        ('stk_m', '个股 m_gua (中波)'),
        ('stk_d', '个股 d_gua (timing)'),
    ]

    all_stable = []  # 全局 ★ 列表
    for col, label in GUA_COLS:
        print(f'\n\n{"="*100}')
        print(f'## {label} 单卦分治 walk-forward')
        print(f'{"="*100}')
        print(f'  {"态":<8} {"全n":>6} {"全期望":>8} {"vs全":>6}', end='')
        for w in WINDOWS:
            print(f' {w[0][:6]:>10}', end='')
        print(f' {"pass":>4} {"fail":>4} {"判定":>10}')
        print('  ' + '-' * 145)

        global_avg = df_e['ret'].mean()

        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 100:
                continue
            n_full = len(sub); ret_full = sub['ret'].mean()
            lift_full = ret_full - global_avg
            print(f'  {state}{GUA_NAMES[state]:<5} {n_full:>6,} {ret_full:>+7.2f}% {lift_full:>+5.1f}', end='')

            n_pass = 0; n_fail = 0; n_low = 0
            seg_lifts = []
            for w in WINDOWS:
                seg = sub[sub['seg'] == w[0]]
                if len(seg) < MIN_N_SEG:
                    n_low += 1
                    print(f' {len(seg):>4}|  -- ', end='')
                    continue
                r = seg['ret'].mean()
                base = seg_baselines[w[0]]
                lift = r - base
                seg_lifts.append(lift)
                if lift >= LIFT_PASS:
                    n_pass += 1; mark = '✅'
                elif lift <= LIFT_FAIL:
                    n_fail += 1; mark = '❌'
                else:
                    mark = '○'
                print(f' {len(seg):>4}|{r:>+4.0f}{mark}', end='')

            n_valid = 7 - n_low
            if n_valid >= 5 and n_pass >= 5 and n_fail <= 1:
                verdict = '★真稳定'
                all_stable.append((label, state, n_full, ret_full, n_valid, n_pass, n_fail, np.mean(seg_lifts)))
            elif n_valid >= 5 and n_pass >= 4 and n_fail <= 1:
                verdict = '○准稳定'
            elif n_valid >= 5 and n_fail >= 4:
                verdict = '✗稳负向'
            elif n_valid < 5:
                verdict = '段不足'
            else:
                verdict = '— 杂'
            print(f' {n_pass:>4} {n_fail:>4}  {verdict:>8}')

    # === 汇总 ★ ===
    print(f'\n\n{"="*100}')
    print(f'## ★ 真稳定 (≥5 段 lift ≥+3%, ≤1 段反向) 汇总')
    print(f'{"="*100}')
    if all_stable:
        print(f'  {"卦类":<22} {"态":<8} {"全n":>6} {"全期望":>8} {"段pass/fail":>12} {"均段lift":>9}')
        print('  ' + '-' * 75)
        for label, state, n, ret, n_v, n_p, n_f, ml in all_stable:
            print(f'  {label:<22} {state}{GUA_NAMES[state]:<6} {n:>6,} {ret:>+7.2f}% '
                  f'{n_p}+/{n_f}-       {ml:>+6.2f}%')
    else:
        print(f'  ★ 真稳定 = 0 个! 单卦分治找不到跨期 ≥5/7 段稳定的态')


if __name__ == '__main__':
    main()
