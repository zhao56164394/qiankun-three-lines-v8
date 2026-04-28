# -*- coding: utf-8 -*-
"""方案 A 实验: v1 池深规律仅在 y_gua ∈ {000, 111} 时生效

新字段 pool_depth_tiers_only_y_gua = {'000', '111'}
当今日 y_gua 不在该 set 时, _pool_depth_tier_ok 直接返回 ok=True (等价于裸跑该卦池深规则)
否则正常按 tier 检查.

然后在 7 个滚动窗口下跑 baseline + v1 (含 only_y_gua) = 14 次回测,
看 v1 alpha 是否在每个窗口都更稳定.

预期: 真规律应在 7/7 窗口都正向 (或至少 5/7 ★)
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_depth_patches import V1_DEPTH_PATCHES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOWS = [
    ('w1_2018',     '2018-01-01', '2019-01-01', '2018 大熊'),
    ('w2_2019',     '2019-01-01', '2020-01-01', '2019 反弹'),
    ('w3_2020',     '2020-01-01', '2021-01-01', '2020 抱团'),
    ('w4_2021',     '2021-01-01', '2022-01-01', '2021 延续'),
    ('w5_2022',     '2022-01-01', '2023-01-01', '2022 杀跌'),
    ('w6_2023_24',  '2023-01-01', '2025-01-01', '2023-24 震'),
    ('w7_2025_26',  '2025-01-01', '2026-04-21', '2025-26 牛'),
]

NAKED_BASE = {
    '000': {'pool_depth_tiers': None},
    '100': {'pool_days_min': None, 'pool_days_max': None},
    '101': {'pool_depth_tiers': None},
}


def add_only_y_gua(patches, y_gua_set):
    """在所有有 pool_depth_tiers 的卦上加 pool_depth_tiers_only_y_gua"""
    out = {k: dict(v) for k, v in patches.items()}
    for gua, fields in out.items():
        if 'pool_depth_tiers' in fields and fields['pool_depth_tiers'] is not None:
            fields['pool_depth_tiers_only_y_gua'] = set(y_gua_set)
        # active=False 的离卦也要条件化 → 改用 tier 表达
        if 'active' in fields and fields['active'] is False:
            # 离卦原 v1 是 active=False (整卦关), 现在改为: 仅在 {000,111} y_gua 时关
            # 用 pool_depth_tiers + only_y_gua 表达 "tier 不匹配任何 → 拒"
            del fields['active']
            fields['pool_depth_tiers'] = [
                {'depth_max': -10000, 'days_min': 99999, 'days_max': None},  # 永不接
            ]
            fields['pool_depth_tiers_only_y_gua'] = set(y_gua_set)
    return out


def merge(base, ver):
    out = {k: dict(v) for k, v in base.items()}
    for gua, fields in ver.items():
        if gua not in out: out[gua] = {}
        out[gua].update(fields)
    return out


# v1_y = v1 但仅在 y_gua ∈ {010, 111} 时生效 (12 月版 y_gua 重切片后的新激活集合)
V1_Y = merge(NAKED_BASE, add_only_y_gua(V1_DEPTH_PATCHES, {'010', '111'}))


def run_one(label, patches, ys, ye):
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
    env['BACKTEST_START'] = ys
    env['BACKTEST_END'] = ye
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


print(f'\n=== 方案 A 实验: v1 池深仅在 y_gua ∈ {{000, 111}} 时生效 ===\n')
print('对比 baseline / v1_full (无条件) / v1_y (条件激活)')
print(f'每窗口跑 v1_y = 1 次, 共 7 次. baseline 和 v1_full 复用之前结果.\n')

results = {}
for i, (wlabel, ys, ye, desc) in enumerate(WINDOWS, 1):
    label = f'phase2_walk_v1_y_{wlabel}'
    print(f'[{i}/{len(WINDOWS)}] {wlabel} ({desc})')
    v, t = run_one(label, V1_Y, ys, ye)
    if v is None:
        results[wlabel] = None
        continue
    # 复用 baseline + v1_full
    with open(os.path.join(ABLATION_DIR, f'phase2_walk_baseline_{wlabel}.json'), encoding='utf-8') as f:
        b = json.load(f)['meta']['final_capital']/10000
    with open(os.path.join(ABLATION_DIR, f'phase2_walk_v1_{wlabel}.json'), encoding='utf-8') as f:
        v_full = json.load(f)['meta']['final_capital']/10000
    a_full = (v_full - b) / b * 100
    a_y = (v - b) / b * 100
    results[wlabel] = {
        'desc': desc, 'baseline': b, 'v1_full': v_full, 'v1_y': v,
        'alpha_full': a_full, 'alpha_y': a_y,
    }
    print(f'  baseline={b:.1f}  v1_full={v_full:.1f} (a={a_full:+.1f}%)  v1_y={v:.1f} (a={a_y:+.1f}%)  ({t:.0f}s)')

# 汇总
print('\n' + '='*100)
print('方案 A: y_gua 条件化 v1 池深规律')
print('='*100)
print(f'  {"窗口":<10} {"OOS":<11} {"base":>7} {"v1_full":>9} {"v1_y":>7} {"α_full":>8} {"α_y":>8} {"提升":>7}')
total_full = 0; total_y = 0; n_better = 0
for w, r in results.items():
    if r is None:
        print(f'  {w:<10}  FAIL'); continue
    diff = r['alpha_y'] - r['alpha_full']
    if diff > 1: n_better += 1
    print(f'  {w:<10} {r["desc"]:<11} {r["baseline"]:>6.1f}万 {r["v1_full"]:>8.1f}万 {r["v1_y"]:>6.1f}万 {r["alpha_full"]:>+7.1f}% {r["alpha_y"]:>+7.1f}% {diff:>+6.1f}%')

print(f'\n  v1_y 更好的窗口数: {n_better}/{len(results)}')

with open(os.path.join(ABLATION_DIR, 'phase2_y_gua_conditional_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'  落地: phase2_y_gua_conditional_summary.json')
