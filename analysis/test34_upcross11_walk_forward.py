# -*- coding: utf-8 -*-
"""Step 11 — 上穿 11 × 卦象 walk-forward 验证

7 段拆 (2018/2019/2020/2021/2022/2023-24/2025-26):
  - 单维 (大盘 y_gua) 8 态各段表现
  - 二维 (大盘 y × 大盘 m) IS Top/Bot 看跨段稳定性

判定:
  ★ 真稳定: ≥5/7 段 期望 ≥ 段 baseline + 3%, ≤1 段反向
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

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
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

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫所有事件 + 记录 date 和卦象
    events = []
    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30: continue
        td = trend_arr[s:e]
        cl = close_arr[s:e]
        n = len(td)
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
                    success = False
                else:
                    continue  # pending
            else:
                s_local = future_sells[0]
                hold = s_local - b
                if hold > TIMEOUT:
                    s_local = b + TIMEOUT
                    success = False
                else:
                    success = True
                ret = cl[s_local] / cl[b] - 1
                last_exit = s_local

            events.append((date_arr[global_b], success, ret * 100,
                           mkt_y[global_b], mkt_m[global_b], mkt_d[global_b]))

    df_e = pd.DataFrame(events, columns=['date', 'success', 'ret', 'mkt_y', 'mkt_m', 'mkt_d'])
    print(f'  事件: {len(df_e):,}, {time.time()-t1:.1f}s')

    # 打段
    df_e['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  打段后: {len(df_e):,}')

    # 各段 baseline (该段所有上穿11 事件的均期望)
    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret'].mean() if len(seg) > 0 else 0
    print(f'\n## 各段 baseline (任意上穿 11 事件 期望 ret%):')
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        print(f'  {w[0]:<14} n={len(seg):>6,}  期望 {seg_baselines[w[0]]:>+6.2f}%  '
              f'成功率 {seg["success"].mean()*100:>5.1f}%')

    # === 单维: 大盘 y_gua 7 段表现 ===
    print(f'\n## 单维 大盘 y_gua × 7 段 期望 ret%')
    print(f'  {"y_gua":<8} {"全n":>6} {"全期望":>8}', end='')
    for w in WINDOWS:
        print(f' {w[0][:6]:>10}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 130)

    pivot = df_e.groupby(['mkt_y', 'seg']).agg(n=('ret', 'size'), ret=('ret', 'mean')).reset_index()
    pivot_n = pivot.pivot(index='mkt_y', columns='seg', values='n').fillna(0)
    pivot_r = pivot.pivot(index='mkt_y', columns='seg', values='ret')

    for y_gua in sorted(df_e['mkt_y'].unique()):
        sub_full = df_e[df_e['mkt_y'] == y_gua]
        n_full = len(sub_full); ret_full = sub_full['ret'].mean()
        print(f'  {y_gua}{GUA_NAMES[y_gua]:<5} {n_full:>6,} {ret_full:>+7.2f}%', end='')
        n_pass = 0; n_fail = 0; n_low = 0
        for w in WINDOWS:
            n = int(pivot_n.loc[y_gua, w[0]]) if y_gua in pivot_n.index and w[0] in pivot_n.columns else 0
            if n < 30:
                n_low += 1
                print(f' {n:>4}|  -- ', end='')
                continue
            r = pivot_r.loc[y_gua, w[0]]
            base = seg_baselines[w[0]]
            lift = r - base
            mark = '✅' if lift >= 3 else ('❌' if lift <= -3 else '○')
            if lift >= 3: n_pass += 1
            elif lift <= -3: n_fail += 1
            print(f' {n:>4}|{r:>+4.0f}{mark}', end='')

        n_valid = 7 - n_low
        if n_valid >= 5 and n_pass >= 5 and n_fail <= 1:
            verdict = '★真稳定'
        elif n_pass >= 4:
            verdict = '○准稳定'
        elif n_fail >= 4:
            verdict = '✗反向'
        else:
            verdict = '— 杂'
        print(f'  {verdict:>8}')

    # === 二维 IS Top 5 (大盘 y × 大盘 m) walk-forward ===
    print(f'\n## 二维 IS Top 5 (大盘 y × 大盘 m) × 7 段 期望 ret%')
    grp2 = df_e.groupby(['mkt_y', 'mkt_m']).agg(n=('ret', 'size'), ret=('ret', 'mean'))
    grp2 = grp2[grp2['n'] >= 200].sort_values('ret', ascending=False)
    top5 = grp2.head(5).index.tolist()
    bot5 = grp2.tail(5).index.tolist()

    for label_set, candidates in [('IS Top 5', top5), ('IS Bot 5', bot5)]:
        print(f'\n  {label_set}:')
        print(f'  {"组合":<14} {"全n":>5} {"全期望":>8}', end='')
        for w in WINDOWS:
            print(f' {w[0][:6]:>10}', end='')
        print(f' {"判定":>10}')
        print('  ' + '-' * 130)
        for (y, m) in candidates:
            sub = df_e[(df_e['mkt_y'] == y) & (df_e['mkt_m'] == m)]
            arrow = f'{y}{GUA_NAMES[y]}|{m}{GUA_NAMES[m]}'
            print(f'  {arrow:<14} {len(sub):>5} {sub["ret"].mean():>+7.2f}%', end='')
            n_pass = 0; n_fail = 0; n_low = 0
            for w in WINDOWS:
                seg = sub[sub['seg'] == w[0]]
                if len(seg) < 30:
                    n_low += 1
                    print(f' {len(seg):>4}|  -- ', end='')
                    continue
                r = seg['ret'].mean()
                base = seg_baselines[w[0]]
                lift = r - base
                mark = '✅' if lift >= 3 else ('❌' if lift <= -3 else '○')
                if lift >= 3: n_pass += 1
                elif lift <= -3: n_fail += 1
                print(f' {len(seg):>4}|{r:>+4.0f}{mark}', end='')
            n_valid = 7 - n_low
            if n_valid >= 5 and n_pass >= 5 and n_fail <= 1: verdict = '★真稳'
            elif n_pass >= 4: verdict = '○准稳'
            elif n_fail >= 4: verdict = '✗反向'
            elif n_valid < 5: verdict = '段不足'
            else: verdict = '— 杂'
            print(f'  {verdict:>8}')


if __name__ == '__main__':
    main()
