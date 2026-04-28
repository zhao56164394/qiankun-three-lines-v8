# -*- coding: utf-8 -*-
"""分解 test1 池深池天哪一条贡献 OOS alpha — 各 tier 独立 LOO

test1 含 3 处池深池天约束:
  000 坤: days_exclude=[4,10]  (磨底死区, 任何深度排 4-10 天)
  100 震: pool_days_min=1, pool_days_max=7  (仅接 1-7 天池天)
  101 离: 复杂 tier (深档永拒 + 中档 0-15 天)

策略: 对每个约束做"逐项 LOO" — 只移除该项, 看 IS/OOS 变化
"""
import os, sys, json, subprocess, time, copy
os.environ['STRATEGY_VERSION'] = 'test1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
OOS_START = '2023-01-01'

# 4 套 patch:
PATCHES = {
    'drop_kun_only':  {'000': {'pool_depth_tiers': None}},
    'drop_zhen_only': {'100': {'pool_days_min': None, 'pool_days_max': None}},
    'drop_li_only':   {'101': {'pool_depth_tiers': None}},
    # full = 全移除已经在 test1_pool_no_pool 跑过
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
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000, elapsed


def load(label):
    p = os.path.join(ABLATION_DIR, f'{label}.json')
    if not os.path.exists(p): return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)['meta']['final_capital']/10000


# 复用之前已跑的 baseline + drop_all
full_is = load('test1_pool_full_IS')
full_oos = load('test1_pool_full_OOS')
none_is = load('test1_pool_no_pool_IS')
none_oos = load('test1_pool_no_pool_OOS')
print(f'\n复用: full_IS={full_is:.1f}, full_OOS={full_oos:.1f}, none_IS={none_is:.1f}, none_OOS={none_oos:.1f}\n')

# 跑 3 个 LOO, IS+OOS 各一次
runs = []
for name, patches in PATCHES.items():
    runs.append((f'test1_{name}_IS',  patches, None,      IS_END))
    runs.append((f'test1_{name}_OOS', patches, OOS_START, None))

results = {}
for i, (label, patches, ys, ye) in enumerate(runs, 1):
    print(f'[{i}/{len(runs)}] {label}')
    v, t = run_one(label, patches, ys, ye)
    results[label] = v
    if v is not None:
        print(f'         {v:>9.1f}万 ({t:.0f}s)')

# 汇总
print('\n' + '='*90)
print('test1 池深池天约束逐项 LOO')
print('='*90)
print(f'  {"":<25} {"IS":>9}  {"OOS":>9}  {"IS_α(vs full)":>16}  {"OOS_α(vs full)":>16}')

# 基准: full = 含全部约束
def show(name, vis, voos):
    if vis is None or voos is None: return
    isd = (vis - full_is) / full_is * 100
    oosd = (voos - full_oos) / full_oos * 100
    print(f'  {name:<25} {vis:>9.1f} {voos:>9.1f}  {isd:>+15.1f}%  {oosd:>+15.1f}%')

show('test1 full (基准)', full_is, full_oos)
show('LOO 移除 000 坤', results['test1_drop_kun_only_IS'], results['test1_drop_kun_only_OOS'])
show('LOO 移除 100 震', results['test1_drop_zhen_only_IS'], results['test1_drop_zhen_only_OOS'])
show('LOO 移除 101 离', results['test1_drop_li_only_IS'], results['test1_drop_li_only_OOS'])
show('全移除 (none)',  none_is, none_oos)

print('\n判定:')
print('  LOO_X 的 IS/OOS 都 < full → X 双向独立有益, 真规律')
print('  LOO_X 的 IS < full 但 OOS > full → X 在 IS 上有益但 OOS 上拖累 → 过拟合')
print('  LOO_X 的 IS > full → X 协同有害, 应该撤')

with open(os.path.join(ABLATION_DIR, 'test1_pool_loo_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({**results, 'full_IS': full_is, 'full_OOS': full_oos,
               'none_IS': none_is, 'none_OOS': none_oos}, f, ensure_ascii=False, indent=2)
print(f'\n  落地: test1_pool_loo_summary.json')
