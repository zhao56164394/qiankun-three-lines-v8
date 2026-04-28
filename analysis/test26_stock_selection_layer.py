# -*- coding: utf-8 -*-
"""Step 2 — 选股层 (个股年卦) 验证

问: 在已知大盘 y_gua 的前提下, 个股 y_gua 是否有增量选股信号?

方法:
  双维 group (mkt_y_gua × stk_y_gua) → 64 桶
  对每个 mkt_y_gua 行, 看个股 y_gua 8 态 hit_rate 跨度

判定 (按"增量"价值):
  ★ 行内跨度 ≥ 10%  → 个股 y_gua 是真增量信号
  ○ 5-10%          → 弱增量
  ✗ < 5%           → 个股年卦无增量, 砍掉
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
    print('=== 加载大盘 y_gua ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    print('=== 加载个股 y_gua ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'y_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_y'] = g['y_gua'].astype(str).str.zfill(3)
    g = g[['date', 'code', 'stk_y']]

    print('=== 加载全市场 close ===')
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    p = p.sort_values(['code', 'date']).reset_index(drop=True)
    p['c60'] = p.groupby('code', sort=False)['close'].shift(-HOLD)
    p = p.dropna(subset=['c60']).reset_index(drop=True)
    p['ret'] = p['c60'] / p['close'] - 1
    p['hit'] = (p['ret'] >= THRESH)
    print(f'  c60 计算后: {len(p):,} 行, {time.time()-t0:.1f}s')

    # merge
    p = p.merge(market, on='date', how='left').merge(g, on=['date', 'code'], how='left')
    p = p.dropna(subset=['mkt_y', 'stk_y']).reset_index(drop=True)
    print(f'  merge 完成: {len(p):,} 行, {time.time()-t0:.1f}s')

    base_full = p['hit'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={base_full:.1f}%')

    # === 一维: 个股 y_gua (不 control 大盘) ===
    print(f'\n## 维度 1: 仅按个股 y_gua 分桶')
    grp1 = p.groupby('stk_y').agg(n=('hit', 'size'), hit=('hit', 'mean'), ret=('ret', 'mean'))
    grp1['hit'] *= 100; grp1['ret'] *= 100
    grp1['lift'] = grp1['hit'] - base_full
    print(f'  {"个股 y_gua":<12} {"行数":>10} {"hit%":>6} {"lift":>6} {"均ret%":>7}')
    print('  ' + '-' * 55)
    for k, r in grp1.sort_index().iterrows():
        print(f'  {k} {GUA_NAMES.get(k, "?"):<10} {int(r["n"]):>10,} {r["hit"]:>5.1f}% '
              f'{r["lift"]:>+5.1f} {r["ret"]:>+6.2f}%')
    print(f'  → 跨度: {grp1["hit"].max() - grp1["hit"].min():.1f}%')

    # === 二维: 大盘 y_gua × 个股 y_gua → 64 桶 ===
    print(f'\n## 维度 2: 双维 (大盘 × 个股) hit_rate 矩阵')
    grp2 = p.groupby(['mkt_y', 'stk_y']).agg(n=('hit', 'size'), hit=('hit', 'mean'))
    grp2['hit'] *= 100
    pivot_hit = grp2['hit'].unstack(fill_value=np.nan)
    pivot_n = grp2['n'].unstack(fill_value=0)
    # reorder
    pivot_hit = pivot_hit.reindex(index=GUAS, columns=GUAS)
    pivot_n = pivot_n.reindex(index=GUAS, columns=GUAS).fillna(0).astype(int)

    # 打印 hit% 矩阵
    print(f'\n  hit_rate 矩阵 (行=大盘 y, 列=个股 y, 单位 %)')
    print(f'  {"":<6}', end='')
    for c in GUAS:
        print(f'{c}{GUA_NAMES[c]:>5}', end='')
    print(f'  | {"行min":>5} {"行max":>5} {"跨度":>5} {"行均":>5}')
    print('  ' + '-' * 95)
    for r_g in GUAS:
        row = pivot_hit.loc[r_g]
        valid = row.dropna()
        print(f'  {r_g}{GUA_NAMES[r_g]:<3}', end='')
        for c_g in GUAS:
            v = row.get(c_g, np.nan)
            n = pivot_n.loc[r_g, c_g]
            if np.isnan(v) or n < 100:
                print(f'  {"--":>5}', end='')
            else:
                print(f'  {v:>4.1f}', end='')
        if len(valid) > 0:
            print(f'   {valid.min():>5.1f} {valid.max():>5.1f} {valid.max()-valid.min():>5.1f} {valid.mean():>5.1f}')
        else:
            print()

    # 打印样本数矩阵 (千)
    print(f'\n  样本数矩阵 (万行)')
    print(f'  {"":<6}', end='')
    for c in GUAS:
        print(f'{c}{GUA_NAMES[c]:>5}', end='')
    print()
    print('  ' + '-' * 70)
    for r_g in GUAS:
        print(f'  {r_g}{GUA_NAMES[r_g]:<3}', end='')
        for c_g in GUAS:
            n = pivot_n.loc[r_g, c_g]
            print(f'  {n/1e4:>4.1f}', end='')
        print()

    # === 行内跨度统计 (核心判定) ===
    print(f'\n## 行内 (固定大盘 regime) 个股 y_gua 跨度')
    print(f'  {"大盘 y":<10} {"行内最佳":>14} {"行内最差":>14} {"跨度":>6} {"判定":>4}')
    print('  ' + '-' * 60)
    spreads = []
    for r_g in GUAS:
        row = pivot_hit.loc[r_g]
        # 只看样本足够的列
        mask = pivot_n.loc[r_g] >= 1000
        valid = row[mask].dropna()
        if len(valid) < 3:
            print(f'  {r_g}{GUA_NAMES[r_g]:<8} 行内有效列 < 3, 略')
            continue
        best_g = valid.idxmax(); worst_g = valid.idxmin()
        best_v = valid.max(); worst_v = valid.min()
        spread = best_v - worst_v
        spreads.append(spread)
        mark = '★' if spread >= 10 else ('○' if spread >= 5 else '✗')
        print(f'  {r_g}{GUA_NAMES[r_g]:<8} {best_g}{GUA_NAMES[best_g]} {best_v:>5.1f}%  '
              f'{worst_g}{GUA_NAMES[worst_g]} {worst_v:>5.1f}%  {spread:>5.1f}  {mark:>3}')

    if spreads:
        avg_spread = np.mean(spreads)
        print(f'\n  平均行内跨度: {avg_spread:.1f}%')
        if avg_spread >= 10:
            print('  ★ 选股层 (个股 y_gua) 强增量, 保留')
        elif avg_spread >= 5:
            print('  ○ 选股层中等增量, 辅助用')
        else:
            print('  ✗ 选股层无增量, 可砍')


if __name__ == '__main__':
    main()
