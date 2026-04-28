# -*- coding: utf-8 -*-
"""Phase 2 池深/池天 LOO — 两版对决

A 版 (找规律): 6 个 sig CI 全负 cell, 不管 trd 兑现
  a1: 011 巽 ≤-400 极深        sig_n=1169 CI(-9.07,-6.03)
  a2: 101 离 (-300,-250] 浅    sig_n=51   CI(-10.47,-3.94)
  a3: 110 兑 (-350,-300] 中    sig_n=22   CI(-8.60,-1.25)
  a4: 000 坤 [11-30] 物极      sig_n=652  CI(-3.36,-0.22)
  a5: 011 巽 [11-30] 物极      sig_n=1103 CI(-5.02,-1.98)
  a6: 101 离 [4-10] 磨底       sig_n=52   CI(-10.18,-2.28)

B 版 (凑收益): 3 个 sig CI 全负 + trd 利<0 cell
  = {a2, a3, a6}

实验:
  baseline = 320.9万 (无任何排除)
  A_full / B_full / A_LOO_each / B_LOO_each
  共 1 + 1 + 6 + 1 + 3 = 12 次回测 (baseline 已存在, 实跑 11 次)

判定 (两版各自):
  full > baseline → 整体方向正确
  LOO_X > full → X 独立有害, 剔除
  LOO_X < full → X 独立有益, 保留

最终:
  对比 A_full vs B_full 在 IS 上的差距 (IS 阶段先各自落地)
  OOS 阶段再决出最终胜者 (留到所有 Phase 走完)
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
IS_BASELINE = 320.9

CAND_DESC = {
    'a1': '011 巽 ≤-400 极深',
    'a2': '101 离 (-300,-250] 浅',
    'a3': '110 兑 (-350,-300] 中',
    'a4': '000 坤 [11-30] 物极',
    'a5': '011 巽 [11-30] 物极',
    'a6': '101 离 [4-10] 磨底',
}
A_SET = {'a1','a2','a3','a4','a5','a6'}
B_SET = {'a2','a3','a6'}


def build_patches(active):
    """active: subset of {a1..a6}, 返回 {gua: {field: value}} patch dict"""
    patches = {}

    # 000 坤 (cfg 原 tier: days_exclude=[4,10])
    if 'a4' in active:
        # 排 [11-30] + 原 [4-10] = 只接受 [0-3]
        patches['000'] = {'pool_depth_tiers': [
            {'depth_max': None, 'days_min': 0, 'days_max': 3},
        ]}

    # 011 巽 (cfg 原: 无 tier)
    a1, a5 = 'a1' in active, 'a5' in active
    if a1 and a5:
        patches['011'] = {'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None, 'days_exclude': [11, 30]},
        ]}
    elif a1:
        patches['011'] = {'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ]}
    elif a5:
        patches['011'] = {'pool_depth_tiers': [
            {'depth_max': None, 'days_min': 0, 'days_max': None, 'days_exclude': [11, 30]},
        ]}

    # 101 离 (cfg 原 tier: 极深拒, (-350,-250] 0-15 天)
    a2, a6 = 'a2' in active, 'a6' in active
    if a2 and a6:
        patches['101'] = {'pool_depth_tiers': [
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': 15, 'days_exclude': [4, 10]},
        ]}
    elif a2:
        patches['101'] = {'pool_depth_tiers': [
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': 15},
        ]}
    elif a6:
        patches['101'] = {'pool_depth_tiers': [
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            {'depth_max': -250, 'days_min': 0, 'days_max': 15, 'days_exclude': [4, 10]},
        ]}

    # 110 兑 (cfg 原: 无 tier)
    if 'a3' in active:
        patches['110'] = {'pool_depth_tiers': [
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ]}

    return patches


def run_one(label, patches):
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test3'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    env['BACKTEST_END'] = IS_END
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(result_path):
        print(f'  [{label}] FAIL ({elapsed:.0f}s)')
        if proc.stderr: print(proc.stderr[-1000:])
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000, elapsed


# 实验列表
runs = []
runs.append(('B_full', B_SET))
for c in sorted(B_SET):
    runs.append((f'B_LOO_{c}', B_SET - {c}))
runs.append(('A_full', A_SET))
for c in sorted(A_SET):
    runs.append((f'A_LOO_{c}', A_SET - {c}))

print(f'\n=== Phase 2 LOO — 两版对决 ===')
print(f'IS baseline (test3 起点, 无 patch): {IS_BASELINE} 万')
print(f'共 {len(runs)} 次回测, 预计 ~{len(runs)*3} 分钟\n')

results = {}
for i, (label, active) in enumerate(runs, 1):
    patches = build_patches(active)
    full_label = f'phase2_loo_{label}'
    print(f'[{i:>2}/{len(runs)}] {label:<10} active={sorted(active)}')
    v, t = run_one(full_label, patches)
    results[label] = {'final_wan': v, 'active': sorted(active)}
    if v is not None:
        diff = v - IS_BASELINE
        print(f'           {v:>9.1f}万 (vs base {diff:+.1f}, {t:.0f}s)')

# 汇总
print('\n' + '='*80)
print('汇总')
print('='*80)

A_full = results['A_full']['final_wan']
B_full = results['B_full']['final_wan']
print(f'\n  baseline:        {IS_BASELINE:>9.1f}万')
print(f'  B_full (3排除):  {B_full:>9.1f}万 (vs base {B_full-IS_BASELINE:+.1f})')
print(f'  A_full (6排除):  {A_full:>9.1f}万 (vs base {A_full-IS_BASELINE:+.1f})')
print(f'  A vs B 差异:                            {A_full-B_full:+.1f}万')

print(f'\n--- B 版 LOO (3 候选) ---')
print(f'  {"name":<6} {"desc":<28} {"final":>9} {"vs full":>9}  判定')
for c in sorted(B_SET):
    v = results[f'B_LOO_{c}']['final_wan']
    if v is None: continue
    diff_full = v - B_full
    if diff_full > 5:
        verdict = '✗ 独立有害 (剔)'
    elif diff_full < -5:
        verdict = '★ 独立有益 (留)'
    else:
        verdict = '○ 协同中性'
    print(f'  {c:<6} {CAND_DESC[c]:<28} {v:>9.1f} {diff_full:>+9.1f}  {verdict}')

print(f'\n--- A 版 LOO (6 候选) ---')
print(f'  {"name":<6} {"desc":<28} {"final":>9} {"vs full":>9}  判定')
for c in sorted(A_SET):
    v = results[f'A_LOO_{c}']['final_wan']
    if v is None: continue
    diff_full = v - A_full
    if diff_full > 5:
        verdict = '✗ 独立有害 (剔)'
    elif diff_full < -5:
        verdict = '★ 独立有益 (留)'
    else:
        verdict = '○ 协同中性'
    print(f'  {c:<6} {CAND_DESC[c]:<28} {v:>9.1f} {diff_full:>+9.1f}  {verdict}')

# 落地
out = {
    'IS_baseline_wan': IS_BASELINE,
    'IS_end': IS_END,
    'A_set': sorted(A_SET),
    'B_set': sorted(B_SET),
    'results': results,
    'desc': CAND_DESC,
}
with open(os.path.join(ABLATION_DIR, 'phase2_loo_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'\n  落地: {os.path.join(ABLATION_DIR, "phase2_loo_summary.json")}')
