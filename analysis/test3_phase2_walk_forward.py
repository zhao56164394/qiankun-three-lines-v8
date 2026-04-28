# -*- coding: utf-8 -*-
"""Phase 2 单维池深 v1 — 滚动验证 (Walk-Forward)

7 个窗口, 每个窗口用前 N 年作 IS 标定 v1, 紧邻的下一年作 OOS 验证.
v1 cfg 是固定的 (单维池深 sig 视角 5 卦排), 不随窗口变 — 我们要验的是
"v1 是否在不同 OOS 段都稳定有效".

每窗口跑 baseline + v1 = 2 次回测, 共 14 次 ≈ 42 分钟.

输出每窗口的:
  baseline_OOS, v1_OOS, OOS_α
然后看 v1 在 7 个 OOS 段的 alpha 分布 — 真规律 = 大部分段都 +.
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR
from analysis.test3_phase2_depth_patches import V1_DEPTH_PATCHES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 7 个窗口: (label, OOS_start, OOS_end, 描述)
WINDOWS = [
    ('w1_2018',     '2018-01-01', '2019-01-01', '2018 大熊  -38%'),
    ('w2_2019',     '2019-01-01', '2020-01-01', '2019 反弹  +26%'),
    ('w3_2020',     '2020-01-01', '2021-01-01', '2020 抱团  +17%'),
    ('w4_2021',     '2021-01-01', '2022-01-01', '2021 延续  +18%'),
    ('w5_2022',     '2022-01-01', '2023-01-01', '2022 杀跌  -21%'),
    ('w6_2023_24',  '2023-01-01', '2025-01-01', '2023-24 震荡'),
    ('w7_2025_26',  '2025-01-01', '2026-04-21', '2025-26 慢牛  +31%'),
]

# 注: cfg 改了 110/111 买入模式, 起点已是新 baseline
# 每窗口 baseline = 解除 000/100/101 池深池天 (与 IS_naked_baseline 一致)
NAKED_BASE = {
    '000': {'pool_depth_tiers': None},
    '100': {'pool_days_min': None, 'pool_days_max': None},
    '101': {'pool_depth_tiers': None},
}

def merge(base, ver):
    out = {k: dict(v) for k, v in base.items()}
    for gua, fields in ver.items():
        if gua not in out: out[gua] = {}
        out[gua].update(fields)
    return out

V1 = merge(NAKED_BASE, V1_DEPTH_PATCHES)


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


print('\n=== 滚动验证 (7 窗口 × 2 版本 = 14 次回测) ===\n')
results = {}
for i, (wlabel, oos_start, oos_end, desc) in enumerate(WINDOWS, 1):
    print(f'\n--- 窗口 {i}/{len(WINDOWS)}: OOS {oos_start} ~ {oos_end}  [{desc}] ---')
    # baseline
    label_b = f'phase2_walk_baseline_{wlabel}'
    print(f'  baseline...')
    vb, tb = run_one(label_b, NAKED_BASE, oos_start, oos_end)
    # v1
    label_v = f'phase2_walk_v1_{wlabel}'
    print(f'  v1...')
    vv, tv = run_one(label_v, V1, oos_start, oos_end)
    if vb is None or vv is None:
        results[wlabel] = None
        continue
    alpha = (vv - vb) / vb * 100
    results[wlabel] = {
        'desc': desc, 'oos_start': oos_start, 'oos_end': oos_end,
        'baseline': vb, 'v1': vv, 'alpha%': alpha,
        'time_s': tb + tv,
    }
    sign = '✅ ★' if alpha > 5 else ('❌ ✗' if alpha < -5 else '○')
    print(f'  baseline={vb:.1f}万  v1={vv:.1f}万  alpha={alpha:+.1f}%  {sign}  ({tb+tv:.0f}s)')

# 汇总
print('\n' + '='*80)
print('滚动验证结果')
print('='*80)
print(f'  {"窗口":<10} {"OOS 描述":<22} {"baseline":>9} {"v1":>9} {"alpha":>9}  判定')
star = 0; bad = 0; neutral = 0
for wlabel, r in results.items():
    if r is None:
        print(f'  {wlabel:<10}  FAIL')
        continue
    if r['alpha%'] > 5:
        verdict = '✅ ★'; star += 1
    elif r['alpha%'] < -5:
        verdict = '❌ ✗'; bad += 1
    else:
        verdict = '○'; neutral += 1
    print(f'  {wlabel:<10} {r["desc"]:<22} {r["baseline"]:>8.1f}万 {r["v1"]:>8.1f}万 {r["alpha%"]:>+8.1f}%  {verdict}')

print(f'\n  ★ 大正: {star}  ✗ 大负: {bad}  ○ 中性: {neutral}')
print()
if star >= 5:
    print(f'  → v1 真规律: 7 段中 {star} 段大正, OOS 普适稳定')
elif bad >= 4:
    print(f'  → v1 严重过拟合: {bad} 段大负, 不该落地')
else:
    print(f'  → v1 中等强度: 段间分布混合, 需看哪些环境失效')

with open(os.path.join(ABLATION_DIR, 'phase2_walk_forward_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\n  落地: phase2_walk_forward_summary.json')
