# -*- coding: utf-8 -*-
"""跑 v3 综合视角 patch — IS 和 OOS 各一次 + baseline 对比"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_v3_patches import V3_PATCHES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
OOS_START = '2023-01-01'


def run_one(label, patches, ys=None, ye=None):
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    if patches:
        write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test3'
    if patches:
        env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    if ys: env['BACKTEST_START'] = ys
    if ye: env['BACKTEST_END'] = ye
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(result_path):
        print(f'  [{label}] FAIL ({elapsed:.0f}s)')
        if proc.stderr: print(proc.stderr[-1500:])
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000, elapsed


print('\n=== v3 综合视角 IS/OOS ===\n')
runs = [
    ('phase2_baseline_IS',  None,        None,      IS_END),
    ('phase2_baseline_OOS', None,        OOS_START, None),
    ('phase2_v3_IS',        V3_PATCHES,  None,      IS_END),
    ('phase2_v3_OOS',       V3_PATCHES,  OOS_START, None),
]
results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

print('\n=== 总结 ===')
b_is = results['phase2_baseline_IS']
b_oos = results['phase2_baseline_OOS']
v3_is = results['phase2_v3_IS']
v3_oos = results['phase2_v3_OOS']

print(f'  baseline_IS:  {b_is:>9.1f}万')
print(f'  baseline_OOS: {b_oos:>9.1f}万')
print(f'  v3_IS:        {v3_is:>9.1f}万 (alpha {(v3_is-b_is)/b_is*100:+.1f}%)')
print(f'  v3_OOS:       {v3_oos:>9.1f}万 (alpha {(v3_oos-b_oos)/b_oos*100:+.1f}%)')

is_alpha = (v3_is - b_is) / b_is * 100
oos_alpha = (v3_oos - b_oos) / b_oos * 100
print(f'\n  IS  alpha: {is_alpha:+.1f}%')
print(f'  OOS alpha: {oos_alpha:+.1f}%')
if is_alpha != 0:
    decay = (is_alpha - oos_alpha) / abs(is_alpha) * 100
    print(f'  衰减率: {decay:+.1f}% (越低越好, <0 = OOS 反而更强)')

with open(os.path.join(ABLATION_DIR, 'phase2_v3_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
