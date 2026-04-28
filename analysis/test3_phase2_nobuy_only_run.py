# -*- coding: utf-8 -*-
"""诊断: 移除优先买区, 只保留不买区, 跑三版本 IS/OOS

判断假设: 优先买区机制是过拟合元凶
预期: 如果"仅不买区"版本 OOS 转正或衰减率显著降低 → 假设成立
      如果仍然过拟合 → 不买区也是过拟合, 池深池天因子整体无 OOS 价值
"""
import os, sys, json, subprocess, time, copy
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_v1_patches import V1_PATCHES
from analysis.test3_phase2_v2_patches import V2_PATCHES
from analysis.test3_phase2_v3_patches import V3_PATCHES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
OOS_START = '2023-01-01'


def strip_priority(patches):
    """深拷贝并移除所有 pool_priority_tiers 字段"""
    out = copy.deepcopy(patches)
    for gua, fields in out.items():
        fields.pop('pool_priority_tiers', None)
    return out


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


b_is = load_existing('phase2_baseline_IS')
b_oos = load_existing('phase2_baseline_OOS')
print(f'\n复用 baseline: IS={b_is:.1f}万, OOS={b_oos:.1f}万\n')

# 移除优先买区
v1_nb = strip_priority(V1_PATCHES)
v2_nb = strip_priority(V2_PATCHES)
v3_nb = strip_priority(V3_PATCHES)

runs = [
    ('phase2_v1_nobuy_IS',  v1_nb, None,      IS_END),
    ('phase2_v1_nobuy_OOS', v1_nb, OOS_START, None),
    ('phase2_v2_nobuy_IS',  v2_nb, None,      IS_END),
    ('phase2_v2_nobuy_OOS', v2_nb, OOS_START, None),
    ('phase2_v3_nobuy_IS',  v3_nb, None,      IS_END),
    ('phase2_v3_nobuy_OOS', v3_nb, OOS_START, None),
]
results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

# 加载之前 v1/v2/v3 (含优先买区) 结果
v1_is = load_existing('phase2_v1_IS')
v1_oos = load_existing('phase2_v1_OOS')
v2_is = load_existing('phase2_v2_IS')
v2_oos = load_existing('phase2_v2_OOS')
v3_is = load_existing('phase2_v3_IS')
v3_oos = load_existing('phase2_v3_OOS')

print('\n' + '='*90)
print('Phase 2 — 优先买区 vs 仅不买区 对照')
print('='*90)
print(f'  {"":<22} {"IS":>9} {"OOS":>9} {"IS_α":>8} {"OOS_α":>8} {"衰减率":>10}')
print(f'  {"baseline":<22} {b_is:>9.1f} {b_oos:>9.1f}     —       —          —')
print()

def show(name, vis, voos):
    if vis is None or voos is None:
        return
    isa = (vis - b_is)/b_is*100
    oosa = (voos - b_oos)/b_oos*100
    decay = (isa - oosa)/abs(isa)*100 if abs(isa)>0.01 else 0
    print(f'  {name:<22} {vis:>9.1f} {voos:>9.1f} {isa:>+7.1f}% {oosa:>+7.1f}% {decay:>+9.0f}%')

show('v1 (含优先买)', v1_is, v1_oos)
show('v1_nobuy (仅不买)', results['phase2_v1_nobuy_IS'], results['phase2_v1_nobuy_OOS'])
print()
show('v2 (含优先买)', v2_is, v2_oos)
show('v2_nobuy (仅不买)', results['phase2_v2_nobuy_IS'], results['phase2_v2_nobuy_OOS'])
print()
show('v3 (含优先买)', v3_is, v3_oos)
show('v3_nobuy (仅不买)', results['phase2_v3_nobuy_IS'], results['phase2_v3_nobuy_OOS'])

with open(os.path.join(ABLATION_DIR, 'phase2_nobuy_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({**results, 'v1_IS': v1_is, 'v1_OOS': v1_oos,
               'v2_IS': v2_is, 'v2_OOS': v2_oos,
               'v3_IS': v3_is, 'v3_OOS': v3_oos,
               'baseline_IS': b_is, 'baseline_OOS': b_oos}, f, ensure_ascii=False, indent=2)
print(f'\n  落地: phase2_nobuy_summary.json')
