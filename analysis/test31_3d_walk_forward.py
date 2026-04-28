# -*- coding: utf-8 -*-
"""Step 7 — 3 维组合的 walk-forward 跨段稳定性扫描

不按 IS hit_rate 排序 (会被切片福利欺骗), 按 "跨段稳定 lift" 排序:
  - 桶: (mkt_y × stk_y × mkt_m) 三维, 共 512 桶
  - 每桶在 7 段中算 (n, hit_rate, lift = hit - 段内 baseline)
  - 筛: 有效段 ≥5/7 + 其中 ≥5 段 lift ≥ +5%
  - 排: 按 (有效段数, 平均 lift) 综合
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
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]

MIN_N = 30        # 段内最少样本
MIN_LIFT = 5.0    # 段内 lift 阈值 (vs 段 baseline)
MIN_VALID_SEG = 5 # 至少几段有效
MIN_PASS_SEG = 5  # 至少几段 lift ≥ MIN_LIFT


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua', 'm_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y', 'mkt_m']].drop_duplicates('date')

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'y_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_y'] = g['y_gua'].astype(str).str.zfill(3)
    g = g[['date', 'code', 'stk_y']]

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
    p = p.dropna(subset=['mkt_y', 'mkt_m', 'stk_y']).reset_index(drop=True)
    print(f'  {len(p):,} 行, {time.time()-t0:.1f}s')

    # 给每行打段标签
    p['seg'] = ''
    for w_label, ws, we in WINDOWS:
        p.loc[(p['date'] >= ws) & (p['date'] < we), 'seg'] = w_label
    p = p[p['seg'] != ''].copy()
    print(f'  打段后: {len(p):,} 行')

    # 各段 baseline
    seg_baselines = {}
    for w_label, _, _ in WINDOWS:
        sub = p[p['seg'] == w_label]
        seg_baselines[w_label] = sub['hit'].mean() * 100 if len(sub) > 0 else 0
    print(f'\n## 段 baseline:')
    for w in WINDOWS:
        print(f'  {w[0]:<14} {seg_baselines[w[0]]:>5.1f}%')

    # 4 维 group: (mkt_y × stk_y × mkt_m × seg)
    print(f'\n## 扫 3 维 × 7 段 hit_rate ...')
    t1 = time.time()
    grp = p.groupby(['mkt_y', 'stk_y', 'mkt_m', 'seg']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp['hit'] *= 100
    grp = grp.reset_index()
    print(f'  group 完成: {len(grp):,} 行, {time.time()-t1:.1f}s')

    # 转 pivot: index = (my, sy, mm), cols = seg, vals = hit / n
    pivot_hit = grp.pivot_table(index=['mkt_y', 'stk_y', 'mkt_m'], columns='seg', values='hit')
    pivot_n = grp.pivot_table(index=['mkt_y', 'stk_y', 'mkt_m'], columns='seg', values='n', fill_value=0)

    # 计算每个 3 维组合的统计
    rows = []
    seg_cols = [w[0] for w in WINDOWS]
    for idx, hits in pivot_hit.iterrows():
        ns = pivot_n.loc[idx]
        n_valid = 0
        n_pass = 0
        n_fail = 0
        seg_lifts = []
        seg_hits_kept = []
        for seg in seg_cols:
            n_seg = int(ns.get(seg, 0))
            if n_seg < MIN_N:
                continue
            n_valid += 1
            h = hits.get(seg, np.nan)
            if pd.isna(h):
                continue
            lift = h - seg_baselines[seg]
            seg_lifts.append(lift)
            seg_hits_kept.append(h)
            if lift >= MIN_LIFT: n_pass += 1
            elif lift <= -MIN_LIFT: n_fail += 1
        if n_valid < MIN_VALID_SEG:
            continue
        my, sy, mm = idx
        rows.append({
            'mkt_y': my, 'stk_y': sy, 'mkt_m': mm,
            'n_valid': n_valid, 'n_pass': n_pass, 'n_fail': n_fail,
            'mean_lift': np.mean(seg_lifts) if seg_lifts else 0,
            'mean_hit': np.mean(seg_hits_kept) if seg_hits_kept else 0,
            'min_lift': np.min(seg_lifts) if seg_lifts else 0,
        })

    rdf = pd.DataFrame(rows)
    print(f'\n## 跨 ≥{MIN_VALID_SEG} 段有效的 3 维组合: {len(rdf)} 个')

    # 真规律: ≥5/7 段 lift ≥ +5%
    star = rdf[(rdf['n_pass'] >= MIN_PASS_SEG) & (rdf['n_fail'] <= 1)].sort_values('mean_lift', ascending=False)
    print(f'\n## ★ 真规律 (≥{MIN_PASS_SEG} 段 lift ≥+{MIN_LIFT}%, ≤1 段反向): {len(star)} 个')
    if len(star) > 0:
        print(f'  {"组合":<22} {"有效":>4} {"+5%":>4} {"-5%":>4} {"均lift":>7} {"均hit":>6} {"最差":>6}')
        print('  ' + '-' * 70)
        for _, r in star.head(20).iterrows():
            arrow = f'{r["mkt_y"]}{GUA_NAMES[r["mkt_y"]]}|{r["stk_y"]}{GUA_NAMES[r["stk_y"]]}|{r["mkt_m"]}{GUA_NAMES[r["mkt_m"]]}'
            print(f'  {arrow:<22} {int(r["n_valid"]):>4} {int(r["n_pass"]):>4} {int(r["n_fail"]):>4} '
                  f'{r["mean_lift"]:>+6.1f}% {r["mean_hit"]:>5.1f}% {r["min_lift"]:>+5.1f}')

        # 详细看 Top 5 组合的各段表现
        print(f'\n## Top 5 真规律的各段细节')
        print(f'  {"组合":<22}', end='')
        for w in WINDOWS:
            print(f' {w[0][:6]:>10}', end='')
        print()
        print('  ' + '-' * 100)
        for _, r in star.head(5).iterrows():
            arrow = f'{r["mkt_y"]}{GUA_NAMES[r["mkt_y"]]}|{r["stk_y"]}{GUA_NAMES[r["stk_y"]]}|{r["mkt_m"]}{GUA_NAMES[r["mkt_m"]]}'
            print(f'  {arrow:<22}', end='')
            for w_label, _, _ in WINDOWS:
                idx = (r['mkt_y'], r['stk_y'], r['mkt_m'])
                n = pivot_n.loc[idx].get(w_label, 0)
                h = pivot_hit.loc[idx].get(w_label, np.nan)
                if n < MIN_N or pd.isna(h):
                    print(f' {int(n):>3}|  -- ', end='')
                else:
                    base = seg_baselines[w_label]
                    lift = h - base
                    mark = '✅' if lift >= MIN_LIFT else ('❌' if lift <= -MIN_LIFT else '○')
                    print(f' {int(n):>3}|{h:>4.0f}{mark}', end='')
            print()
    else:
        print('  无! 3 维 control 不足以找到跨段稳定 ≥+5% lift 的组合')

    # 也看准 ★ (4-5 段 pass)
    near = rdf[(rdf['n_pass'] >= 4) & (rdf['n_pass'] < MIN_PASS_SEG) & (rdf['n_fail'] <= 1)].sort_values('mean_lift', ascending=False)
    print(f'\n## ○ 准真规律 (4 段 lift ≥+{MIN_LIFT}%): {len(near)} 个')
    for _, r in near.head(10).iterrows():
        arrow = f'{r["mkt_y"]}{GUA_NAMES[r["mkt_y"]]}|{r["stk_y"]}{GUA_NAMES[r["stk_y"]]}|{r["mkt_m"]}{GUA_NAMES[r["mkt_m"]]}'
        print(f'  {arrow:<22}  有效 {int(r["n_valid"])}/7  pass {int(r["n_pass"])}  均 lift {r["mean_lift"]:+.1f}%  均 hit {r["mean_hit"]:.1f}%')


if __name__ == '__main__':
    main()
