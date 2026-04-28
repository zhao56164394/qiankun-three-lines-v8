# -*- coding: utf-8 -*-
"""Step 4 — 个股日卦层验证

问: 个股日卦在前 3 层 (大盘 y + 个股 y + 大盘 m) 之上是否还有真增量?

测试:
  4.1 个股 d_gua 静态 8 态 hit_rate 跨度
  4.2 个股 d_gua 变化 X→Y 的全市场 hit_rate (动态)
  4.3 在 3 维 control 下 (大盘y + 个股y + 大盘m), 个股 d_gua 增量
  4.4 最强 4 维组合 (3 维 control + d_gua) push 多少

判定:
  ★ 加 d_gua 后还能 ≥ 10% 跨度 → 真增量
  ✗ 跨度小 → 日卦冗余, 建议砍
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
                              columns=['date', 'y_gua', 'm_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y', 'mkt_m']].drop_duplicates('date')

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
    p = p.dropna(subset=['mkt_y', 'mkt_m', 'stk_y', 'stk_m', 'stk_d']).reset_index(drop=True)
    print(f'  最终: {len(p):,} 行, {time.time()-t0:.1f}s')

    base_full = p['hit'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={base_full:.1f}%')

    # === 4.1 个股 d_gua 静态 8 态 hit_rate ===
    print(f'\n## 4.1 个股 d_gua 静态 8 态')
    grp1 = p.groupby('stk_d').agg(n=('hit', 'size'), hit=('hit', 'mean'), ret=('ret', 'mean'))
    grp1['hit'] *= 100; grp1['ret'] *= 100
    grp1['lift'] = grp1['hit'] - base_full
    print(f'  {"d_gua":<10} {"行数":>10} {"hit%":>6} {"lift":>6} {"均ret%":>7}')
    print('  ' + '-' * 50)
    for k, r in grp1.sort_index().iterrows():
        print(f'  {k} {GUA_NAMES.get(k, "?"):<6} {int(r["n"]):>10,} {r["hit"]:>5.1f}% '
              f'{r["lift"]:>+5.1f} {r["ret"]:>+6.2f}%')
    print(f'  → 跨度: {grp1["hit"].max() - grp1["hit"].min():.1f}%')

    # === 4.2 个股 d_gua 变化 X→Y 全市场 hit_rate ===
    print(f'\n## 4.2 个股 d_gua 变化 X→Y (全市场动态)')
    # 算每个 row 的"前一日 d_gua"
    p['prev_d'] = p.groupby('code', sort=False)['stk_d'].shift(1)
    p_chg = p.dropna(subset=['prev_d']).copy()
    p_chg['is_change'] = (p_chg['stk_d'] != p_chg['prev_d'])
    chg = p_chg[p_chg['is_change']].copy()
    print(f'  全市场变化事件: {len(chg):,}')

    # 只看 64 桶 (排除 from==to)
    grp_chg = chg.groupby(['prev_d', 'stk_d']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp_chg['hit'] *= 100
    grp_chg['lift'] = grp_chg['hit'] - base_full
    grp_chg = grp_chg[grp_chg['n'] >= 1000].sort_values('hit', ascending=False)
    print(f'  Top 8 X→Y (n≥1000):')
    for (f, t), r in grp_chg.head(8).iterrows():
        print(f'    {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]}  n={int(r["n"]):>7,}  hit={r["hit"]:>5.1f}%  lift={r["lift"]:>+5.1f}')
    print(f'  Bot 5 X→Y:')
    for (f, t), r in grp_chg.tail(5).iterrows():
        print(f'    {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]}  n={int(r["n"]):>7,}  hit={r["hit"]:>5.1f}%  lift={r["lift"]:>+5.1f}')
    sp = grp_chg['hit'].max() - grp_chg['hit'].min()
    print(f'  → X→Y 跨度: {sp:.1f}%')

    # === 4.3 在 3 维 control (大盘y + 个股y + 大盘m) 下, 个股 d_gua 增量 ===
    print(f'\n## 4.3 3 维 control (大盘y × 个股y × 大盘m) 下 个股 d_gua 增量跨度')
    print(f'  {"大盘y":<6} {"个股y":<6} {"大盘m":<6} {"n_total":>10} {"d最高":>8} {"d最低":>8} {"跨度":>5}')
    print('  ' + '-' * 70)
    n_strong = 0; n_total = 0
    sample_rows = []
    for (my, sy, mm), sub in p.groupby(['mkt_y', 'stk_y', 'mkt_m']):
        if len(sub) < 5000:
            continue
        n_total += 1
        grp_d = sub.groupby('stk_d').agg(n=('hit', 'size'), hit=('hit', 'mean'))
        grp_d['hit'] *= 100
        valid = grp_d[grp_d['n'] >= 200]
        if len(valid) < 4:
            continue
        sp = valid['hit'].max() - valid['hit'].min()
        if sp >= 10: n_strong += 1
        sample_rows.append((my, sy, mm, len(sub), valid['hit'].max(), valid['hit'].min(), sp))

    sample_rows.sort(key=lambda x: -x[6])
    for r in sample_rows[:10]:
        my, sy, mm, n, mx, mn, sp = r
        mark = '★' if sp >= 10 else ('○' if sp >= 5 else '✗')
        print(f'  {my}{GUA_NAMES[my]:<3} {sy}{GUA_NAMES[sy]:<3} {mm}{GUA_NAMES[mm]:<3} {n:>10,} '
              f'{mx:>7.1f}% {mn:>7.1f}% {sp:>5.1f} {mark}')
    print(f'\n  → ★ 跨度 ≥10% 的 3 维桶: {n_strong}/{n_total}')

    # === 4.4 最强 4 维组合 push 多少 ===
    print(f'\n## 4.4 最强 4 维组合 (大盘y × 个股y × 大盘m × 个股d)')
    grp4 = p.groupby(['mkt_y', 'stk_y', 'mkt_m', 'stk_d']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp4['hit'] *= 100
    # 至少 1000 样本
    grp4 = grp4[grp4['n'] >= 1000].sort_values('hit', ascending=False)
    print(f'  Top 10 4 维组合 (n≥1000):')
    print(f'  {"大盘y":<5} {"个股y":<5} {"大盘m":<5} {"个股d":<5} {"n":>7} {"hit%":>6}')
    print('  ' + '-' * 50)
    for (my, sy, mm, sd), r in grp4.head(10).iterrows():
        print(f'  {my}{GUA_NAMES[my]:<2} {sy}{GUA_NAMES[sy]:<2} {mm}{GUA_NAMES[mm]:<2} {sd}{GUA_NAMES[sd]:<2} '
              f'{int(r["n"]):>7,} {r["hit"]:>5.1f}%')
    print(f'\n  Bot 5 4 维组合:')
    for (my, sy, mm, sd), r in grp4.tail(5).iterrows():
        print(f'  {my}{GUA_NAMES[my]:<2} {sy}{GUA_NAMES[sy]:<2} {mm}{GUA_NAMES[mm]:<2} {sd}{GUA_NAMES[sd]:<2} '
              f'{int(r["n"]):>7,} {r["hit"]:>5.1f}%')


if __name__ == '__main__':
    main()
