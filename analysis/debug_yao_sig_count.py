# -*- coding: utf-8 -*-
"""验证 baseline IS 中 2020 段 (w3) 实际命中 skip 集的 sig 数"""
import os, sys, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(ROOT, 'data_layer/data/ablation/test6_pool_depth/baseline_IS.json'),
          encoding='utf-8') as f:
    d = json.load(f)
sigs = pd.DataFrame(d['signal_detail'])
print(f'baseline IS sig: {len(sigs)}')

yg = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                     columns=['date', 'y_gua'])
yg['date'] = yg['date'].astype(str)
yg['y_gua'] = yg['y_gua'].astype(str).str.zfill(3)
yg = yg.drop_duplicates('date').sort_values('date').reset_index(drop=True)

# 计算 buy_date 用的 last_from + signal_date 用的 last_from
prev_y = None
last_from = None
last_from_map = {}
y_gua_map = {}
for _, row in yg.iterrows():
    d = row['date']
    if prev_y is not None and row['y_gua'] != prev_y:
        last_from = prev_y
    last_from_map[d] = last_from
    y_gua_map[d] = row['y_gua']
    prev_y = row['y_gua']

sigs['buy_date'] = sigs['buy_date'].astype(str)
sigs['signal_date'] = sigs['signal_date'].astype(str)
sigs['ct_buy'] = sigs['buy_date'].map(last_from_map).astype(str) + '->' + sigs['buy_date'].map(y_gua_map).astype(str)
sigs['ct_signal'] = sigs['signal_date'].map(last_from_map).astype(str) + '->' + sigs['signal_date'].map(y_gua_map).astype(str)

skip = {'111->101', '001->000'}

print('\n## 全 IS')
print(f'  按 buy_date 命中 skip: {sigs["ct_buy"].isin(skip).sum()}')
print(f'  按 signal_date 命中 skip: {sigs["ct_signal"].isin(skip).sum()}')

# 切 2020 段 (signal_date)
for label, ys, ye in [('w3 2020', '2020', '2021'), ('w2 2019', '2019', '2020'),
                      ('w1 2018', '2018', '2019')]:
    s_w = sigs[(sigs['signal_date'] >= ys) & (sigs['signal_date'] < ye)]
    print(f'\n## {label} (n={len(s_w)})')
    print(f'  按 buy_date 命中 skip: {s_w["ct_buy"].isin(skip).sum()}')
    print(f'  按 signal_date 命中 skip: {s_w["ct_signal"].isin(skip).sum()}')
    # 拆 type
    for ct in skip:
        n_buy = (s_w['ct_buy'] == ct).sum()
        n_sig = (s_w['ct_signal'] == ct).sum()
        print(f'    {ct}: by_buy={n_buy} / by_signal={n_sig}')
