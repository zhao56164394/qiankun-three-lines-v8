# -*- coding: utf-8 -*-
"""Step 1 — Regime 层验证

问: 大盘 y_gua ∈ {000..111} 八态下, 全市场任意股任意日入场,
   60 日后 +5% 的 hit_rate 是否有显著差异?

判定:
  max - min ≥ 15%  → regime 层有效
  max - min < 5%   → regime 层无用, 砍掉
  中间             → 弱开关, 辅助用
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


def main():
    t0 = time.time()
    print('=== 加载大盘 y_gua ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y_gua'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y_gua']].drop_duplicates('date')
    print(f'  {len(market):,} 个交易日')

    print('=== 加载全市场 close ===')
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    p = p.sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  {len(p):,} 行, {time.time()-t0:.1f}s')

    # 计算 60d 后 close (按 code 分组 shift)
    p['c60'] = p.groupby('code', sort=False)['close'].shift(-HOLD)
    p = p.dropna(subset=['c60']).reset_index(drop=True)
    p['ret'] = p['c60'] / p['close'] - 1
    p['hit'] = (p['ret'] >= THRESH)
    print(f'  有效样本 (含 c60): {len(p):,}, {time.time()-t0:.1f}s')

    # 全市场 baseline
    full_hit = p['hit'].mean() * 100
    full_ret = p['ret'].mean() * 100
    print(f'\n## 全市场 baseline: hit_rate={full_hit:.1f}%, 均收益={full_ret:+.2f}%')

    # merge 大盘 y_gua
    p = p.merge(market, on='date', how='left')
    p = p.dropna(subset=['mkt_y_gua']).reset_index(drop=True)

    # 按 y_gua 聚合
    print(f'\n## 按 大盘 y_gua 状态分桶 (HOLD={HOLD}日, +{THRESH*100:.0f}% 阈值)\n')
    print(f'  {"大盘 y_gua":<14} {"行数":>10} {"占比":>5} {"hit%":>6} {"lift":>6} {"均ret%":>7}')
    print('  ' + '-' * 60)
    grp = p.groupby('mkt_y_gua').agg(n=('hit', 'size'), hit_rate=('hit', 'mean'), mean_ret=('ret', 'mean'))
    grp = grp.sort_index()
    grp['hit_rate'] *= 100
    grp['mean_ret'] *= 100
    grp['lift'] = grp['hit_rate'] - full_hit
    grp['pct'] = grp['n'] / grp['n'].sum() * 100
    for g, r in grp.iterrows():
        name = f'{g} {GUA_NAMES.get(g, "?")}'
        print(f'  {name:<14} {int(r["n"]):>10,} {r["pct"]:>4.1f}% {r["hit_rate"]:>5.1f}% '
              f'{r["lift"]:>+5.1f} {r["mean_ret"]:>+6.2f}%')

    spread = grp['hit_rate'].max() - grp['hit_rate'].min()
    print(f'\n  hit_rate 跨度: max - min = {spread:.1f} 个百分点')

    # 判定
    print(f'\n## 判定')
    if spread >= 15:
        print(f'  ★ regime 层强 (跨度 {spread:.1f}% ≥ 15%) — 大盘 y_gua 是有效开关')
        # 找出最佳/最差状态
        best = grp.sort_values('hit_rate', ascending=False).iloc[0]
        worst = grp.sort_values('hit_rate').iloc[0]
        print(f'    最佳: y_gua={best.name} ({GUA_NAMES.get(best.name, "?")}) hit={best["hit_rate"]:.1f}%')
        print(f'    最差: y_gua={worst.name} ({GUA_NAMES.get(worst.name, "?")}) hit={worst["hit_rate"]:.1f}%')
    elif spread < 5:
        print(f'  ✗ regime 层弱 (跨度 {spread:.1f}% < 5%) — 砍掉')
    else:
        print(f'  ○ regime 层中等 (跨度 {spread:.1f}%) — 辅助用')

    # 看时间分布: 牛/熊段长度
    print(f'\n## y_gua 时间分布 (用日历日, 不分股)')
    daily = p.drop_duplicates('date').sort_values('date')
    state_run = []
    cur = None; cnt = 0; start = None
    for _, r in daily.iterrows():
        s = r['mkt_y_gua']
        if s != cur:
            if cur is not None:
                state_run.append((cur, start, prev_d, cnt))
            cur = s; cnt = 1; start = r['date']
        else:
            cnt += 1
        prev_d = r['date']
    state_run.append((cur, start, prev_d, cnt))
    print(f'  共 {len(state_run)} 段, 各段时长 (前 15 段):')
    for st, sd, ed, n in state_run[:15]:
        print(f'    y_gua={st} {GUA_NAMES.get(st, "?")}  {sd} ~ {ed}  ({n} 日)')


if __name__ == '__main__':
    main()
