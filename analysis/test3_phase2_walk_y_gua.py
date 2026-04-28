# -*- coding: utf-8 -*-
"""按 y_gua 切片 v1 实验.

每窗口的 baseline / v1 实买记录, 按 当时的 y_gua 切片:
  - 哪些 y_gua 下 v1 alpha 稳定 ★?
  - 哪些 y_gua 下 v1 反向 ✗?

如果发现 "v1 只在 y_gua=000 时 ★" 之类的规律 → y_gua 是更好的分治依据.
"""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3')

# 加载 y_gua 映射
zz = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet'))
zz['date'] = zz['date'].astype(str)
zz['y_gua'] = zz['y_gua'].astype(str).str.zfill(3)
y_map = dict(zip(zz['date'], zz['y_gua']))

WINDOWS = [
    ('w1_2018',     '2018 -38%'),
    ('w2_2019',     '2019 +26%'),
    ('w3_2020',     '2020 +17%'),
    ('w4_2021',     '2021 +18%'),
    ('w5_2022',     '2022 -21%'),
    ('w6_2023_24',  '2023-24 震'),
    ('w7_2025_26',  '2025-26 牛'),
]
GUA_NAME = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}


def load_trades(label):
    path = os.path.join(ABL, f'{label}.json')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    trades = pd.DataFrame(d['trade_log'])
    if len(trades) == 0:
        return trades
    # 按买入日查 y_gua
    trades['buy_date'] = trades['buy_date'].astype(str)
    trades['y_gua'] = trades['buy_date'].map(y_map)
    trades['profit_wan'] = trades['profit'] / 10000
    return trades


# 1. 整体: 各 y_gua 下 baseline vs v1 累计利润
print('\n=== 实验: 按 y_gua 切片对比 baseline vs v1 (跨 7 窗口聚合) ===\n')
all_b = []
all_v = []
for w, _ in WINDOWS:
    tb = load_trades(f'phase2_walk_baseline_{w}')
    tv = load_trades(f'phase2_walk_v1_{w}')
    if tb is not None and len(tb): tb['window']=w; all_b.append(tb)
    if tv is not None and len(tv): tv['window']=w; all_v.append(tv)
all_b = pd.concat(all_b, ignore_index=True) if all_b else pd.DataFrame()
all_v = pd.concat(all_v, ignore_index=True) if all_v else pd.DataFrame()

print(f'baseline 总笔: {len(all_b)}, v1 总笔: {len(all_v)}\n')

# 按 y_gua 汇总
print(f'  {"y_gua":<8} {"卦名":<3} {"base 笔":>7} {"base 利万":>10} {"v1 笔":>7} {"v1 利万":>10} {"alpha":>9}')
print('  ' + '-' * 75)
for y in sorted(set(all_b['y_gua'].dropna()) | set(all_v['y_gua'].dropna())):
    bs = all_b[all_b['y_gua']==y]
    vs = all_v[all_v['y_gua']==y]
    b_p = bs['profit_wan'].sum() if len(bs) else 0
    v_p = vs['profit_wan'].sum() if len(vs) else 0
    diff = v_p - b_p
    name = GUA_NAME.get(y, '?')
    sign = '✅★' if diff > 5 else ('❌✗' if diff < -5 else '○')
    print(f'  {y:<8} {name:<3} {len(bs):>7} {b_p:>+10.1f} {len(vs):>7} {v_p:>+10.1f} {diff:>+8.1f}  {sign}')

# 2. 按 (窗口 × y_gua) 二维: 发现"v1 偏爱哪个 y_gua + 在哪个窗口"
print('\n\n=== 二维: (窗口 × y_gua) 下 v1 - baseline 利润差 (万) ===\n')
all_y = sorted(set(all_b['y_gua'].dropna()) | set(all_v['y_gua'].dropna()))
print(f'  {"窗口":<14} ' + ' '.join(f'{GUA_NAME.get(y,y):>6}' for y in all_y))
for w, desc in WINDOWS:
    row = [f'  {w:<14}']
    for y in all_y:
        b_p = all_b[(all_b['window']==w) & (all_b['y_gua']==y)]['profit_wan'].sum()
        v_p = all_v[(all_v['window']==w) & (all_v['y_gua']==y)]['profit_wan'].sum()
        diff = v_p - b_p
        if abs(diff) < 0.05:
            row.append(f'{".":>6}')
        else:
            row.append(f'{diff:>+6.1f}')
    print(' '.join(row))

# 3. 各 y_gua 在哪些窗口"出现" + alpha 走向
print('\n\n=== 各 y_gua 跨窗口稳定性: v1 alpha 在多窗口下的方向 ===\n')
print(f'  {"y_gua":<6} {"卦":<3} {"出现窗口":<12} {"v1 - base (万)"}')
for y in all_y:
    deltas = []
    for w, _ in WINDOWS:
        b_p = all_b[(all_b['window']==w) & (all_b['y_gua']==y)]['profit_wan'].sum()
        v_p = all_v[(all_v['window']==w) & (all_v['y_gua']==y)]['profit_wan'].sum()
        if abs(b_p)+abs(v_p) > 0.1:
            deltas.append((w, v_p - b_p))
    if not deltas: continue
    n_pos = sum(1 for _, d in deltas if d > 0.5)
    n_neg = sum(1 for _, d in deltas if d < -0.5)
    n_neutral = len(deltas) - n_pos - n_neg
    delta_str = ', '.join(f'{w[:5]}={d:+.1f}' for w, d in deltas)
    sign = '✅' if n_pos > n_neg + 1 else ('❌' if n_neg > n_pos + 1 else '○')
    print(f'  {y:<6} {GUA_NAME.get(y,"?"):<3} {len(deltas):>3}个窗口   {sign}  ({delta_str})')

with open(os.path.join(ABL, 'phase2_walk_y_gua_slice.json'), 'w', encoding='utf-8') as f:
    json.dump({'analyzed': True}, f)
print()
