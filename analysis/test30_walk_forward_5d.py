# -*- coding: utf-8 -*-
"""Step 6 — walk-forward 验证 Top 5 维组合

7 段拆 (2018/2019/2020/2021/2022/2023-24/2025-26):
  对 Top 10 5 维组合, 每段算 hit_rate + 段内 baseline
  看是否多段稳定 (≥5/7 ≥70%) 还是切片福利 (集中在 1-2 段)
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOLD = 60
THRESH = 0.05

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01', '2018 大熊'),
    ('w2_2019',    '2019-01-01', '2020-01-01', '2019 反弹'),
    ('w3_2020',    '2020-01-01', '2021-01-01', '2020 抱团'),
    ('w4_2021',    '2021-01-01', '2022-01-01', '2021 延续'),
    ('w5_2022',    '2022-01-01', '2023-01-01', '2022 杀跌'),
    ('w6_2023_24', '2023-01-01', '2025-01-01', '2023-24 震荡'),
    ('w7_2025_26', '2025-01-01', '2026-04-21', '2025-26 慢牛'),
]

# Top 候选 (来自 test29)
CANDIDATES = [
    # (mkt_y, stk_y, mkt_m, stk_d, mkt_d) — 期望 hit_rate
    ('010', '000', '010', '101', '101', 99.0),
    ('010', '010', '010', '000', '001', 98.4),
    ('010', '000', '010', '100', '001', 98.3),
    ('010', '000', '010', '000', '001', 98.3),
    ('010', '000', '010', '001', '001', 98.3),
    ('010', '000', '010', '101', '001', 97.7),
    ('010', '000', '010', '111', '101', 97.2),
    ('010', '010', '010', '101', '001', 96.9),
    ('010', '000', '010', '000', '101', 96.9),
    ('010', '000', '010', '010', '001', 95.7),
]


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua', 'm_gua', 'd_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y', 'mkt_m', 'mkt_d']].drop_duplicates('date')

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'y_gua', 'm_gua', 'd_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_y'] = g['y_gua'].astype(str).str.zfill(3)
    g['stk_m'] = g['m_gua'].astype(str).str.zfill(3)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)
    g = g[['date', 'code', 'stk_y', 'stk_m', 'stk_d']]

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    p = p.sort_values(['code', 'date']).reset_index(drop=True)
    p['c60'] = p.groupby('code', sort=False)['close'].shift(-HOLD)
    p = p.dropna(subset=['c60']).reset_index(drop=True)
    p['ret'] = p['c60'] / p['close'] - 1
    p['hit'] = (p['ret'] >= THRESH)

    p = p.merge(market, on='date', how='left').merge(g, on=['date', 'code'], how='left')
    p = p.dropna(subset=['mkt_y', 'mkt_m', 'mkt_d', 'stk_y', 'stk_m', 'stk_d']).reset_index(drop=True)
    print(f'  {len(p):,} 行, {time.time()-t0:.1f}s')

    base_full = p['hit'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={base_full:.1f}%')

    # 段内 baseline (每段 hit_rate)
    print(f'\n## 各段 baseline (任意股任意日, 60d +5%)')
    seg_baselines = {}
    for w_label, ws, we, desc in WINDOWS:
        sub = p[(p['date'] >= ws) & (p['date'] < we)]
        b = sub['hit'].mean() * 100
        seg_baselines[w_label] = b
        print(f'  {w_label:<14} ({desc}) {ws}~{we}  n={len(sub):>10,}  hit={b:>5.1f}%')

    # === 验证每个候选 ===
    print(f'\n## 候选 5 维组合 walk-forward 验证')
    print(f'  {"组合":<32} {"全n":>5} {"IS hit%":>8}', end='')
    for w_label, _, _, _ in WINDOWS:
        print(f' {w_label[:6]:>10}', end='')
    print(f' {"判定":>8}')
    print('  ' + '-' * 130)

    for cand in CANDIDATES:
        my, sy, mm, sd, md, exp_hit = cand
        sub = p[(p['mkt_y'] == my) & (p['stk_y'] == sy) & (p['mkt_m'] == mm)
                & (p['stk_d'] == sd) & (p['mkt_d'] == md)]
        n_full = len(sub)
        is_hit = sub['hit'].mean() * 100 if n_full > 0 else 0
        label = f'{my}{GUA_NAMES[my]}|{sy}{GUA_NAMES[sy]}|{mm}{GUA_NAMES[mm]}|{sd}{GUA_NAMES[sd]}|{md}{GUA_NAMES[md]}'
        print(f'  {label:<32} {n_full:>5} {is_hit:>7.1f}%', end='')

        # 各段
        n_pass = 0; n_fail = 0; n_low_n = 0
        seg_results = []
        for w_label, ws, we, _ in WINDOWS:
            seg = sub[(sub['date'] >= ws) & (sub['date'] < we)]
            n_seg = len(seg)
            if n_seg < 30:
                n_low_n += 1
                seg_results.append((n_seg, None))
                print(f' {n_seg:>4}|  -- ', end='')
                continue
            seg_hit = seg['hit'].mean() * 100
            base_seg = seg_baselines[w_label]
            seg_results.append((n_seg, seg_hit))
            # 判定: 段内 hit ≥ 70%
            mark = '✅' if seg_hit >= 70 else ('❌' if seg_hit < 50 else '○')
            if seg_hit >= 70: n_pass += 1
            elif seg_hit < 50: n_fail += 1
            print(f' {n_seg:>3}|{seg_hit:>4.0f}{mark}', end='')

        # 总判定
        n_valid = 7 - n_low_n
        if n_valid >= 5 and n_pass >= 5 and n_fail <= 1:
            verdict = '★真规律'
        elif n_valid >= 4 and n_pass >= 4:
            verdict = '○不稳'
        elif n_low_n >= 4:
            verdict = '段不足'
        else:
            verdict = '✗切片'
        print(f'  {verdict:>6}')

    # 输出每段 baseline 给一个对比眼
    print(f'\n  baseline 各段:        ', end='')
    for w_label, _, _, _ in WINDOWS:
        b = seg_baselines[w_label]
        print(f'         {b:>4.0f}', end='')


if __name__ == '__main__':
    main()
