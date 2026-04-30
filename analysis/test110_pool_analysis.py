# -*- coding: utf-8 -*-
"""分析 v7 买点之前是否曾下穿 11"""
import os, sys, io
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                     columns=['date', 'code', 'd_trend'])
df['date'] = df['date'].astype(str)
df['code'] = df['code'].astype(str).str.zfill(6)
df = df.sort_values(['code','date']).reset_index(drop=True)

df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
df['cross'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)
df['cross_60d'] = df.groupby('code', sort=False)['cross'].transform(
    lambda s: s.rolling(60, min_periods=1).sum())
df['cross_30d'] = df.groupby('code', sort=False)['cross'].transform(
    lambda s: s.rolling(30, min_periods=1).sum())

v7 = pd.read_csv(os.path.join(ROOT, 'data_layer/data/results/capital_trades_v7.csv'), encoding='utf-8-sig')
v7['code'] = v7['code'].astype(str).str.zfill(6)
m = v7[['buy_date','code']].rename(columns={'buy_date':'date'}).merge(
    df[['date','code','cross_60d','cross_30d','d_trend']], on=['date','code'], how='left')
v7['cross_60d'] = m['cross_60d'].values
v7['cross_30d'] = m['cross_30d'].values
v7['t_at_buy'] = m['d_trend'].values

print('v7 买点 60d 内下穿 11 次数分布:')
print(v7['cross_60d'].value_counts().sort_index().head(10))
print()
total = len(v7)
print(f'有 >=1 次下穿: {(v7["cross_60d"]>=1).sum()}/{total} ({(v7["cross_60d"]>=1).mean()*100:.1f}%)')
print(f'0 次下穿: {(v7["cross_60d"]==0).sum()}/{total}')

print()
print('v7 按 60d 内下穿次数分组:')
groups = [
    (0, 0, '0 (无)'),
    (1, 1, '1 次'),
    (2, 2, '2 次'),
    (3, 100, '>=3 次'),
]
for lo, hi, label in groups:
    if lo == hi:
        sub = v7[v7['cross_60d']==lo]
    else:
        sub = v7[(v7['cross_60d']>=lo)&(v7['cross_60d']<=hi)]
    if len(sub):
        print(f'  {label:<14}: n={len(sub):>3} win {(sub["ret_pct"]>0).mean()*100:>5.1f}% '
              f'avg {sub["ret_pct"].mean():>+6.2f}% sum {sub["profit"].sum():>+10,.0f}')

print()
print('v7 30d 内下穿次数:')
for lo, hi, label in groups:
    if lo == hi:
        sub = v7[v7['cross_30d']==lo]
    else:
        sub = v7[(v7['cross_30d']>=lo)&(v7['cross_30d']<=hi)]
    if len(sub):
        print(f'  {label:<14}: n={len(sub):>3} win {(sub["ret_pct"]>0).mean()*100:>5.1f}% '
              f'avg {sub["ret_pct"].mean():>+6.2f}% sum {sub["profit"].sum():>+10,.0f}')

# 看 v8 那 30 笔 (v7 没买的) 的特征
print()
print('=== 对比 v7 vs v8 (谁买的票更好) ===')
v8 = pd.read_csv(os.path.join(ROOT, 'data_layer/data/results/capital_trades_v8.csv'), encoding='utf-8-sig')
print(f'v7 总盈亏: {v7["profit"].sum():>+10,.0f} (n={len(v7)})')
print(f'v8 总盈亏: {v8["profit"].sum():>+10,.0f} (n={len(v8)})')

# v7 - v8 共同 buy_date+code, 看 v7 买而 v8 没买的部分
v7_keys = set(zip(v7['buy_date'], v7['code'].astype(str).str.zfill(6)))
v8['code'] = v8['code'].astype(str).str.zfill(6)
v8_keys = set(zip(v8['buy_date'], v8['code']))

shared_keys = v7_keys & v8_keys
v7_only_keys = v7_keys - v8_keys  # v7 买了 v8 没买 (可能因池子限制)
v8_only_keys = v8_keys - v7_keys  # v8 买了 v7 没买

v7_shared = v7[v7.apply(lambda r: (r['buy_date'], r['code']) in shared_keys, axis=1)]
v7_only = v7[v7.apply(lambda r: (r['buy_date'], r['code']) in v7_only_keys, axis=1)]

print()
print(f'共同买入 (v7 ∩ v8): n={len(v7_shared)} avg {v7_shared["ret_pct"].mean():>+6.2f}% sum {v7_shared["profit"].sum():>+10,.0f}')
print(f'v7 独有 (v8 因池漏掉): n={len(v7_only)} avg {v7_only["ret_pct"].mean():>+6.2f}% sum {v7_only["profit"].sum():>+10,.0f}')

if len(v7_only):
    print()
    print('v7 独有的买点 60d 内下穿次数 (这些就是"v8 因池子拒绝"的):')
    print(v7_only['cross_60d'].value_counts().sort_index().head(8))
