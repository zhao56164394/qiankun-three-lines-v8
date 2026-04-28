# -*- coding: utf-8 -*-
"""验证 _from_map 计算 + change_type_skip 命中率"""
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ['STRATEGY_VERSION'] = 'test12yaomin'
import strategy_configs

cfg = strategy_configs.get_strategy()
skip_set = cfg['000'].get('change_type_skip', set())
print(f'cfg 000 change_type_skip: {skip_set}')

# 加载 gate_map (跟 backtest_y_gua 同样的 ms_df 来源)
import pandas as pd
yg = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                     columns=['date', 'y_gua', 'm_gua'])
yg['date'] = yg['date'].astype(str)
yg['y_gua'] = yg['y_gua'].astype(str).str.zfill(3)
yg['m_gua'] = yg['m_gua'].astype(str).str.zfill(3)
yg = yg.drop_duplicates('date').sort_values('date').reset_index(drop=True)
gate_map = {row['date']: (row['m_gua'], row['y_gua']) for _, row in yg.iterrows()}
print(f'gate_map 大小: {len(gate_map)}, 日期范围: {min(gate_map.keys())} ~ {max(gate_map.keys())}')

# 跟 backtest_y_gua 一样的 _from_map 计算逻辑
_from_map = {}
_last_y = None
_last_from = None
for _d in sorted(gate_map.keys()):
    _y = gate_map[_d][1]
    if _last_y is not None and _y != _last_y:
        _last_from = _last_y
    _from_map[_d] = _last_from
    _last_y = _y

print(f'_from_map 大小: {len(_from_map)}, _from_map 中 None 的天数: {sum(1 for v in _from_map.values() if v is None)}')

# 各时段统计 change_type hits
periods = [
    ('IS 2014-2022', '20140101', '20221231'),
    ('w1 2018', '20180101', '20190101'),
    ('w2 2019', '20190101', '20200101'),
    ('w3 2020', '20200101', '20210101'),
    ('w4 2021', '20210101', '20220101'),
    ('w5 2022', '20220101', '20230101'),
    ('w6 2023-24', '20230101', '20250101'),
    ('w7 2025-26', '20250101', '20260417'),
]

print(f'\n各时段 change_type 命中 skip 集 ({skip_set}) 的天数:')
print(f'  {"时段":<14} {"总天":>5} {"hit":>5} {"hit %":>7} {"具体 ct":<30}')
print('  ' + '-' * 70)
for label, start, end in periods:
    total = 0; hits = 0
    hit_cts = {}
    for d, (m, y) in gate_map.items():
        if d < start or d >= end:
            continue
        total += 1
        f = _from_map.get(d)
        if f is None:
            continue
        ct = f'{f}->{y}'
        if ct in skip_set:
            hits += 1
            hit_cts[ct] = hit_cts.get(ct, 0) + 1
    cts_str = ' '.join(f'{c}:{n}' for c, n in hit_cts.items())
    pct = hits/total*100 if total > 0 else 0
    print(f'  {label:<14} {total:>5} {hits:>5} {pct:>6.1f}% {cts_str:<30}')
