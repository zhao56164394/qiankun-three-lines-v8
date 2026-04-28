# -*- coding: utf-8 -*-
"""Step 3 — 趋势层 (月卦) 验证

问: 月卦是 Step 1+2 之上的增量信号 还是冗余?

测试:
  3.1 单看大盘月卦 / 个股月卦 hit_rate 跨度
  3.2 双维 (大盘 m × 个股 m) 矩阵
  3.3 control 在最强 regime (大盘 y=010 + 个股 y=000) 下, 月卦能否再 push
  3.4 control 在最弱 regime (大盘 y=011 巽) 下, 月卦能否反转

判定:
  ★ 加月卦后跨度 ≥ 10% → 月卦有真增量
  ○ 5-10% → 弱
  ✗ < 5% → 月卦是冗余 (跟年卦相关), 砍掉
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
                        columns=['date', 'code', 'y_gua', 'm_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_y'] = g['y_gua'].astype(str).str.zfill(3)
    g['stk_m'] = g['m_gua'].astype(str).str.zfill(3)
    g = g[['date', 'code', 'stk_y', 'stk_m']]

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
    p = p.dropna(subset=['mkt_y', 'mkt_m', 'stk_y', 'stk_m']).reset_index(drop=True)
    print(f'  最终: {len(p):,} 行, {time.time()-t0:.1f}s')

    base_full = p['hit'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={base_full:.1f}%')

    # === 3.1 单维: 大盘 m / 个股 m ===
    print(f'\n## 3.1 单看月卦 hit_rate')
    for col, label in [('mkt_m', '大盘月卦'), ('stk_m', '个股月卦')]:
        grp = p.groupby(col).agg(n=('hit', 'size'), hit=('hit', 'mean'))
        grp['hit'] *= 100
        print(f'\n  {label}: {{', end='')
        for k, r in grp.sort_index().iterrows():
            print(f' {k}={r["hit"]:.1f}%', end=' ')
        print(f'}}, 跨度={grp["hit"].max() - grp["hit"].min():.1f}%')

    # === 3.2 双维 大盘 m × 个股 m ===
    print(f'\n## 3.2 大盘 m × 个股 m 矩阵')
    grp2 = p.groupby(['mkt_m', 'stk_m']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp2['hit'] *= 100
    pivot = grp2['hit'].unstack(fill_value=np.nan).reindex(index=GUAS, columns=GUAS)
    pivot_n = grp2['n'].unstack(fill_value=0).reindex(index=GUAS, columns=GUAS).fillna(0).astype(int)

    print(f'  {"":<6}', end='')
    for c in GUAS:
        print(f'{c}{GUA_NAMES[c]:>5}', end='')
    print(f'  | {"行min":>5} {"行max":>5} {"跨度":>5}')
    print('  ' + '-' * 90)
    row_spreads = []
    for r_g in GUAS:
        row = pivot.loc[r_g]
        mask = pivot_n.loc[r_g] >= 1000
        valid = row[mask].dropna()
        print(f'  {r_g}{GUA_NAMES[r_g]:<3}', end='')
        for c_g in GUAS:
            v = row.get(c_g, np.nan)
            n = pivot_n.loc[r_g, c_g]
            if np.isnan(v) or n < 1000:
                print(f'  {"--":>5}', end='')
            else:
                print(f'  {v:>4.1f}', end='')
        if len(valid) >= 3:
            sp = valid.max() - valid.min()
            row_spreads.append(sp)
            print(f'   {valid.min():>5.1f} {valid.max():>5.1f} {sp:>5.1f}')
        else:
            print()
    if row_spreads:
        print(f'\n  → 双维月卦 行内平均跨度: {np.mean(row_spreads):.1f}%')

    # === 3.3 control 最强 regime (大盘y=010 + 个股y=000) 下, 月卦增量 ===
    print(f'\n## 3.3 在最强 regime (大盘y=010坎 + 个股y=000坤, base 49.7%) 下加月卦')
    sub = p[(p['mkt_y'] == '010') & (p['stk_y'] == '000')]
    base_sub = sub['hit'].mean() * 100
    print(f'  子集 baseline: {len(sub):,} 行, hit={base_sub:.1f}%')
    grp_mm = sub.groupby(['mkt_m', 'stk_m']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp_mm['hit'] *= 100
    grp_mm = grp_mm[grp_mm['n'] >= 500].sort_values('hit', ascending=False)
    if len(grp_mm) >= 3:
        top5 = grp_mm.head(5)
        bot3 = grp_mm.tail(3)
        print(f'  最高 5 个月卦组合:')
        for (mm, sm), r in top5.iterrows():
            print(f'    大盘m={mm}{GUA_NAMES[mm]} 个股m={sm}{GUA_NAMES[sm]}  n={int(r["n"]):>5}  hit={r["hit"]:.1f}%  '
                  f'(vs base 49.7%, +{r["hit"]-49.7:+.1f})')
        print(f'  最低 3 个月卦组合:')
        for (mm, sm), r in bot3.iterrows():
            print(f'    大盘m={mm}{GUA_NAMES[mm]} 个股m={sm}{GUA_NAMES[sm]}  n={int(r["n"]):>5}  hit={r["hit"]:.1f}%  '
                  f'(vs base 49.7%, {r["hit"]-49.7:+.1f})')
        sp = grp_mm['hit'].max() - grp_mm['hit'].min()
        print(f'  → control 后 月卦双维跨度: {sp:.1f}%')
    else:
        print('  样本不足')

    # === 3.4 control 最弱 regime (大盘y=011巽 + 个股y=110兑) 下 ===
    print(f'\n## 3.4 在最弱 regime (大盘y=011巽 + 个股y=110兑, base 19.2%) 下加月卦')
    sub = p[(p['mkt_y'] == '011') & (p['stk_y'] == '110')]
    base_sub = sub['hit'].mean() * 100
    print(f'  子集 baseline: {len(sub):,} 行, hit={base_sub:.1f}%')
    if len(sub) < 1000:
        print('  样本不足')
    else:
        grp_mm = sub.groupby(['mkt_m', 'stk_m']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
        grp_mm['hit'] *= 100
        grp_mm = grp_mm[grp_mm['n'] >= 100].sort_values('hit', ascending=False)
        if len(grp_mm) >= 3:
            print(f'  Top 3 月卦组合 (能否反转):')
            for (mm, sm), r in grp_mm.head(3).iterrows():
                print(f'    大盘m={mm}{GUA_NAMES[mm]} 个股m={sm}{GUA_NAMES[sm]}  n={int(r["n"])}  hit={r["hit"]:.1f}%')
            sp = grp_mm['hit'].max() - grp_mm['hit'].min()
            print(f'  → 跨度 {sp:.1f}%, 最高 {grp_mm["hit"].max():.1f}% (能否突破 baseline 36.9%?)')

    # === 3.5 控住"大盘 y × 个股 y", 看月卦的"行内平均跨度" ===
    print(f'\n## 3.5 在每个 (大盘 y × 个股 y) 桶内, 大盘月卦的增量跨度')
    print(f'  (只看样本足的桶, 看月卦的边际贡献)')
    print(f'  {"大盘y":<8} {"个股y":<8} {"n_total":>10} {"月卦最高":>10} {"月卦最低":>10} {"跨度":>5}')
    print('  ' + '-' * 65)
    n_strong = 0; n_total = 0
    sample_rows = []
    for y_pair, sub in p.groupby(['mkt_y', 'stk_y']):
        if len(sub) < 5000:
            continue
        n_total += 1
        # 仅按大盘 m 拆 (8 桶)
        grp_m = sub.groupby('mkt_m').agg(n=('hit', 'size'), hit=('hit', 'mean'))
        grp_m['hit'] *= 100
        valid = grp_m[grp_m['n'] >= 200]
        if len(valid) < 3:
            continue
        sp = valid['hit'].max() - valid['hit'].min()
        if sp >= 10: n_strong += 1
        sample_rows.append((y_pair[0], y_pair[1], len(sub), valid['hit'].max(), valid['hit'].min(), sp))

    sample_rows.sort(key=lambda x: -x[5])
    for r in sample_rows[:10]:
        my, sy, n, mx, mn, sp = r
        mark = '★' if sp >= 10 else ('○' if sp >= 5 else '✗')
        print(f'  {my}{GUA_NAMES[my]:<5} {sy}{GUA_NAMES[sy]:<5} {n:>10,} {mx:>9.1f}% {mn:>9.1f}% {sp:>5.1f} {mark}')
    print(f'\n  → ★ 行内跨度 ≥10% 的桶: {n_strong}/{n_total}')


if __name__ == '__main__':
    main()
