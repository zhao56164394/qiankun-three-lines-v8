# -*- coding: utf-8 -*-
"""三版本 cfg patch IS+OOS 对决 — 仅池深单维"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_depth_patches import V1_DEPTH_PATCHES, V2_DEPTH_PATCHES, V3_DEPTH_PATCHES

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


# 注: cfg 改了 110/111 买入模式 + 我们是裸跑起点, baseline 也要重跑
# 起点应是: 新 cfg + 解除 000/100/101 池深池天约束 (与 IS_naked_baseline 一致)
NAKED_BASE = {
    '000': {'pool_depth_tiers': None},
    '100': {'pool_days_min': None, 'pool_days_max': None},
    '101': {'pool_depth_tiers': None},
}

# 三版本要在裸跑起点上叠加各自的池深 patch (注意要保留 NAKED_BASE)
def merge(base_patches, ver_patches):
    out = {k: dict(v) for k, v in base_patches.items()}
    for gua, fields in ver_patches.items():
        if gua not in out:
            out[gua] = {}
        out[gua].update(fields)
    return out

V1 = merge(NAKED_BASE, V1_DEPTH_PATCHES)
V2 = merge(NAKED_BASE, V2_DEPTH_PATCHES)
V3 = merge(NAKED_BASE, V3_DEPTH_PATCHES)

runs = [
    ('phase2_depth_baseline_IS',  NAKED_BASE, None,      IS_END),
    ('phase2_depth_baseline_OOS', NAKED_BASE, OOS_START, None),
    ('phase2_depth_v1_IS',  V1, None,      IS_END),
    ('phase2_depth_v1_OOS', V1, OOS_START, None),
    ('phase2_depth_v2_IS',  V2, None,      IS_END),
    ('phase2_depth_v2_OOS', V2, OOS_START, None),
    ('phase2_depth_v3_IS',  V3, None,      IS_END),
    ('phase2_depth_v3_OOS', V3, OOS_START, None),
]
results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

b_is = results['phase2_depth_baseline_IS']
b_oos = results['phase2_depth_baseline_OOS']

print('\n' + '='*90)
print('Phase 2 单维池深 — 三版本过拟合对决')
print('='*90)
print(f'  {"":<22} {"IS":>9}  {"OOS":>9}  {"IS_α":>8} {"OOS_α":>8} {"衰减率":>10}')
print(f'  {"baseline":<22} {b_is:>9.1f}  {b_oos:>9.1f}     —       —          —')

def show(name, label):
    vis = results.get(f'phase2_depth_{label}_IS')
    voos = results.get(f'phase2_depth_{label}_OOS')
    if vis is None or voos is None: return
    isa = (vis - b_is)/b_is*100
    oosa = (voos - b_oos)/b_oos*100
    decay = (isa - oosa)/abs(isa)*100 if abs(isa)>0.01 else 0
    print(f'  {name:<22} {vis:>9.1f}  {voos:>9.1f}  {isa:>+7.1f}% {oosa:>+7.1f}% {decay:>+9.0f}%')

show('v1 单 sig (5 卦排)',  'v1')
show('v2 单 trd (6 卦排)',  'v2')
show('v3 综合 (2 卦排)',    'v3')

with open(os.path.join(ABLATION_DIR, 'phase2_depth_only_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\n  落地: phase2_depth_only_summary.json')
