# -*- coding: utf-8 -*-
"""Step 4c: 买点 cross 切换的组合验证

发现 (Step 4 add-one):
  - 101 离: double_rise → cross  +1115 万
  - 000 坤: double_rise → cross  +261 万
  - 100 震: double_rise → cross  +194 万
  - 110 dui_thresh: 20 → 30      +38 万

但 add-one 是各自独立改, 实际同时改可能因资金挤压协同变化.
本实验做组合验证, 看哪种组合最优.
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 1466.9

cross_li   = {'101': {'li_buy_mode': 'cross', 'li_cross_threshold': 20}}
cross_kun  = {'000': {'kun_buy_mode': 'cross', 'kun_cross_threshold': 20}}
cross_zhen = {'100': {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 20}}
dui_30     = {'110': {'dui_cross_threshold': 30}}

scenarios = [
    ('only_li_cross',         {**cross_li}),
    ('li_kun',                {**cross_li, **cross_kun}),
    ('li_zhen',               {**cross_li, **cross_zhen}),
    ('li_kun_zhen',           {**cross_li, **cross_kun, **cross_zhen}),
    ('li_kun_zhen_dui30',     {**cross_li, **cross_kun, **cross_zhen, **dui_30}),
    ('all_three_no_dui',      {**cross_li, **cross_kun, **cross_zhen}),  # = li_kun_zhen
]

# 去重
seen = {}
for name, ps in scenarios:
    key = json.dumps(ps, sort_keys=True, default=str)
    if key not in seen:
        seen[key] = (name, ps)
unique = list(seen.values())

print(f'\nbaseline = {BASELINE} 万 (~3 min/run, 共 {len(unique)} 组合, ~{len(unique)*3} min)\n')
print(f'{"组合":<28} {"终值万":>9} {"vs base":>9}  判定')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for name, patches in unique:
    label = f'step4c_{name}'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, capture_output=True, encoding='utf-8', errors='replace',
        cwd=ROOT,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'{name:<28} FAIL ({elapsed:.0f}s)')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - BASELINE
    mark = '★★★' if diff > 500 else ('★★' if diff > 100 else ('★' if diff > 5 else ('✗' if diff < -5 else '○')))
    print(f'{name:<28} {v:>9.1f} {diff:>+9.1f}  {mark}  ({elapsed:.0f}s)')
