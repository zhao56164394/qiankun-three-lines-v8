# -*- coding: utf-8 -*-
"""Step 5 — 验证层 (大盘日卦) 验证

问: 大盘 d_gua 在前 4 层 (大盘y + 个股y + 大盘m + 个股d) 之上是否还有增量?

测试:
  5.1 大盘 d_gua 静态 8 态 hit_rate 跨度
  5.2 大盘 d_gua 变化 X→Y hit_rate 跨度
  5.3 4 维 control 下加 mkt_d 增量跨度
  5.4 最强 5 维组合 (大盘y + 个股y + 大盘m + 个股d + 大盘d) push 多少
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
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']


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
    print(f'  最终: {len(p):,} 行, {time.time()-t0:.1f}s')

    base_full = p['hit'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={base_full:.1f}%')

    # === 5.1 大盘 d_gua 静态 ===
    print(f'\n## 5.1 大盘 d_gua 静态 8 态')
    grp1 = p.groupby('mkt_d').agg(n=('hit', 'size'), hit=('hit', 'mean'), ret=('ret', 'mean'))
    grp1['hit'] *= 100; grp1['ret'] *= 100
    grp1['lift'] = grp1['hit'] - base_full
    print(f'  {"大盘d":<10} {"行数":>10} {"hit%":>6} {"lift":>6} {"均ret%":>7}')
    print('  ' + '-' * 50)
    for k, r in grp1.sort_index().iterrows():
        print(f'  {k} {GUA_NAMES.get(k, "?"):<6} {int(r["n"]):>10,} {r["hit"]:>5.1f}% '
              f'{r["lift"]:>+5.1f} {r["ret"]:>+6.2f}%')
    print(f'  → 跨度: {grp1["hit"].max() - grp1["hit"].min():.1f}%')

    # === 5.2 大盘 d_gua 变化 X→Y (按日期, 不按股) ===
    print(f'\n## 5.2 大盘 d_gua 变化 X→Y (全市场, 用大盘 d_gua 做触发)')
    # 算大盘的前一日 d_gua (只有 1 个时间序列, 全股共享)
    market_sorted = market.sort_values('date').reset_index(drop=True)
    market_sorted['mkt_d_prev'] = market_sorted['mkt_d'].shift(1)
    p_chg = p.merge(market_sorted[['date', 'mkt_d_prev']], on='date', how='left')
    p_chg = p_chg.dropna(subset=['mkt_d_prev'])
    p_chg = p_chg[p_chg['mkt_d'] != p_chg['mkt_d_prev']].copy()
    print(f'  全市场处于"大盘 d 变化日"的样本: {len(p_chg):,}')

    grp_chg = p_chg.groupby(['mkt_d_prev', 'mkt_d']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp_chg['hit'] *= 100; grp_chg['lift'] = grp_chg['hit'] - base_full
    grp_chg = grp_chg[grp_chg['n'] >= 1000].sort_values('hit', ascending=False)
    print(f'  Top 8 大盘 d X→Y:')
    for (f, t), r in grp_chg.head(8).iterrows():
        print(f'    {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]}  n={int(r["n"]):>7,}  hit={r["hit"]:>5.1f}%  lift={r["lift"]:>+5.1f}')
    print(f'  Bot 5 大盘 d X→Y:')
    for (f, t), r in grp_chg.tail(5).iterrows():
        print(f'    {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]}  n={int(r["n"]):>7,}  hit={r["hit"]:>5.1f}%  lift={r["lift"]:>+5.1f}')
    if len(grp_chg) > 0:
        print(f'  → 跨度: {grp_chg["hit"].max() - grp_chg["hit"].min():.1f}%')

    # === 5.3 4 维 control 下加 mkt_d 增量 ===
    print(f'\n## 5.3 4 维 control (大盘y × 个股y × 大盘m × 个股d) 下 大盘 d 增量')
    print(f'  {"大盘y":<5} {"个股y":<5} {"大盘m":<5} {"个股d":<5} {"n_total":>8} {"d最高":>7} {"d最低":>7} {"跨度":>5}')
    print('  ' + '-' * 70)
    n_strong = 0; n_total = 0
    sample_rows = []
    for k, sub in p.groupby(['mkt_y', 'stk_y', 'mkt_m', 'stk_d']):
        if len(sub) < 5000:
            continue
        n_total += 1
        grp_d = sub.groupby('mkt_d').agg(n=('hit', 'size'), hit=('hit', 'mean'))
        grp_d['hit'] *= 100
        valid = grp_d[grp_d['n'] >= 200]
        if len(valid) < 4:
            continue
        sp = valid['hit'].max() - valid['hit'].min()
        if sp >= 10: n_strong += 1
        sample_rows.append((*k, len(sub), valid['hit'].max(), valid['hit'].min(), sp))

    sample_rows.sort(key=lambda x: -x[7])
    for r in sample_rows[:10]:
        my, sy, mm, sd, n, mx, mn, sp = r
        mark = '★' if sp >= 10 else ('○' if sp >= 5 else '✗')
        print(f'  {my}{GUA_NAMES[my]:<2} {sy}{GUA_NAMES[sy]:<2} {mm}{GUA_NAMES[mm]:<2} {sd}{GUA_NAMES[sd]:<2} '
              f'{n:>8,} {mx:>6.1f}% {mn:>6.1f}% {sp:>5.1f} {mark}')
    print(f'\n  → ★ 跨度 ≥10% 的 4 维桶: {n_strong}/{n_total}')

    # === 5.4 最强 5 维组合 ===
    print(f'\n## 5.4 最强 5 维组合')
    grp5 = p.groupby(['mkt_y', 'stk_y', 'mkt_m', 'stk_d', 'mkt_d']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp5['hit'] *= 100
    grp5 = grp5[grp5['n'] >= 1000].sort_values('hit', ascending=False)
    print(f'  Top 10 5 维组合 (n≥1000):')
    print(f'  {"大盘y":<4} {"个股y":<4} {"大盘m":<4} {"个股d":<4} {"大盘d":<4} {"n":>6} {"hit%":>6}')
    print('  ' + '-' * 55)
    for (my, sy, mm, sd, md), r in grp5.head(10).iterrows():
        print(f'  {my}{GUA_NAMES[my]:<2} {sy}{GUA_NAMES[sy]:<2} {mm}{GUA_NAMES[mm]:<2} {sd}{GUA_NAMES[sd]:<2} '
              f'{md}{GUA_NAMES[md]:<2} {int(r["n"]):>6,} {r["hit"]:>5.1f}%')
    print(f'\n  Bot 5 5 维组合:')
    for (my, sy, mm, sd, md), r in grp5.tail(5).iterrows():
        print(f'  {my}{GUA_NAMES[my]:<2} {sy}{GUA_NAMES[sy]:<2} {mm}{GUA_NAMES[mm]:<2} {sd}{GUA_NAMES[sd]:<2} '
              f'{md}{GUA_NAMES[md]:<2} {int(r["n"]):>6,} {r["hit"]:>5.1f}%')

    # === 累计跨度汇总 ===
    print(f'\n## 5 层累计 hit_rate 跨度 (n≥1000 的桶)')
    print(f'  baseline     : 36.9%  跨度 -')
    print(f'  +大盘y      : 22.5%-44.1%  跨度 21.6%')
    print(f'  +个股y      : 19.2%-49.7%  跨度 30.5%')
    print(f'  +大盘m      : 22.5%-88.5%  跨度 66.0%')
    print(f'  +个股d      : 2.7%-90.0%   跨度 87.3%')
    if len(grp5) > 0:
        print(f'  +大盘d      : {grp5["hit"].min():.1f}%-{grp5["hit"].max():.1f}%  跨度 {grp5["hit"].max()-grp5["hit"].min():.1f}%')


if __name__ == '__main__':
    main()
