# -*- coding: utf-8 -*-
"""跑 v1 / v2 IS/OOS — 复用现有 baseline 结果"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_v1_patches import V1_PATCHES
from analysis.test3_phase2_v2_patches import V2_PATCHES

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


def load_existing(label):
    p = os.path.join(ABLATION_DIR, f'{label}.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000


# 复用 baseline + v3
b_is = load_existing('phase2_baseline_IS')
b_oos = load_existing('phase2_baseline_OOS')
v3_is = load_existing('phase2_v3_IS')
v3_oos = load_existing('phase2_v3_OOS')
print(f'\n复用现有: baseline_IS={b_is}万, baseline_OOS={b_oos}万, v3_IS={v3_is}万, v3_OOS={v3_oos}万\n')

runs = [
    ('phase2_v1_IS',  V1_PATCHES, None,      IS_END),
    ('phase2_v1_OOS', V1_PATCHES, OOS_START, None),
    ('phase2_v2_IS',  V2_PATCHES, None,      IS_END),
    ('phase2_v2_OOS', V2_PATCHES, OOS_START, None),
]
results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

v1_is = results['phase2_v1_IS']
v1_oos = results['phase2_v1_OOS']
v2_is = results['phase2_v2_IS']
v2_oos = results['phase2_v2_OOS']

print('\n' + '='*70)
print('Phase 2 三版本过拟合对决')
print('='*70)
print(f'  {"版本":<10} {"IS":>9}  {"OOS":>9}  {"IS_alpha":>10}  {"OOS_alpha":>10}  衰减率')
print(f'  {"baseline":<10} {b_is:>9.1f}  {b_oos:>9.1f}      —          —')
for ver, vis, voos in [('v1', v1_is, v1_oos), ('v2', v2_is, v2_oos), ('v3', v3_is, v3_oos)]:
    if vis is None or voos is None: continue
    isa = (vis - b_is) / b_is * 100
    oosa = (voos - b_oos) / b_oos * 100
    if abs(isa) < 0.01:
        decay_str = 'N/A'
    else:
        decay = (isa - oosa) / abs(isa) * 100
        decay_str = f'{decay:+.0f}%'
    print(f'  {ver:<10} {vis:>9.1f}  {voos:>9.1f}  {isa:>+9.1f}%  {oosa:>+9.1f}%  {decay_str}')

with open(os.path.join(ABLATION_DIR, 'phase2_three_versions_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'baseline_IS': b_is, 'baseline_OOS': b_oos,
        'v1_IS': v1_is, 'v1_OOS': v1_oos,
        'v2_IS': v2_is, 'v2_OOS': v2_oos,
        'v3_IS': v3_is, 'v3_OOS': v3_oos,
    }, f, ensure_ascii=False, indent=2)
print(f'\n  落地: phase2_three_versions_summary.json')
