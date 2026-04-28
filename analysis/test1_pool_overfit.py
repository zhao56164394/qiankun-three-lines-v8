# -*- coding: utf-8 -*-
"""测 test1 cfg 池深池天约束是否过拟合.

实验:
  test1_full      = test1 完整 cfg (含 000/100/101 的池深池天约束)
  test1_no_pool   = test1 但移除 000/100/101 的池深池天 (pool_depth_tiers=None, pool_days_min/max=None)

各跑 IS (2014-2022) + OOS (2023-2026), 共 4 次回测.

判定:
  alpha_IS  = (full_IS - no_pool_IS) / no_pool_IS
  alpha_OOS = (full_OOS - no_pool_OOS) / no_pool_OOS
  衰减率 = (alpha_IS - alpha_OOS) / |alpha_IS|

  alpha_OOS > 0 → 池深池天 OOS 上仍有效 (不过拟合)
  alpha_OOS < 0 → 过拟合
  衰减率小 → 规律稳健; 衰减率大 → 过拟合
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test1'  # 切到 test1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
OOS_START = '2023-01-01'

# 移除 test1 cfg 的池深池天约束
NO_POOL_PATCHES = {
    '000': {'pool_depth_tiers': None},
    '100': {'pool_days_min': None, 'pool_days_max': None},
    '101': {'pool_depth_tiers': None},
}


def run_one(label, patches, ys=None, ye=None):
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    if patches:
        write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test1'
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


print('\n=== test1 池深池天过拟合诊断 ===\n')
runs = [
    ('test1_pool_full_IS',     None,             None,      IS_END),
    ('test1_pool_full_OOS',    None,             OOS_START, None),
    ('test1_pool_no_pool_IS',  NO_POOL_PATCHES,  None,      IS_END),
    ('test1_pool_no_pool_OOS', NO_POOL_PATCHES,  OOS_START, None),
]
results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

full_is = results['test1_pool_full_IS']
full_oos = results['test1_pool_full_OOS']
np_is = results['test1_pool_no_pool_IS']
np_oos = results['test1_pool_no_pool_OOS']

print('\n' + '='*80)
print('test1 池深池天对决')
print('='*80)
print(f'  {"":<22} {"IS":>10} {"OOS":>10}')
print(f'  {"含池深池天 (full)":<22} {full_is:>10.1f}万 {full_oos:>10.1f}万')
print(f'  {"移除池深池天 (no)":<22} {np_is:>10.1f}万 {np_oos:>10.1f}万')

is_alpha = (full_is - np_is) / np_is * 100
oos_alpha = (full_oos - np_oos) / np_oos * 100
print(f'\n  IS  alpha (含 vs 移除): {is_alpha:+.1f}%')
print(f'  OOS alpha (含 vs 移除): {oos_alpha:+.1f}%')

if abs(is_alpha) > 0.01:
    decay = (is_alpha - oos_alpha) / abs(is_alpha) * 100
    print(f'  衰减率: {decay:+.0f}%')
print()
if oos_alpha > 5:
    print(f'  ✅ 池深池天 OOS 仍有效 (+{oos_alpha:.1f}%) — test1 这部分不过拟合')
elif oos_alpha < -5:
    print(f'  ❌ 池深池天 OOS 反向 ({oos_alpha:.1f}%) — test1 这部分严重过拟合')
else:
    print(f'  ○ OOS 中性 ({oos_alpha:+.1f}%) — 边缘有效')

with open(os.path.join(ABLATION_DIR, 'test1_pool_overfit.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
