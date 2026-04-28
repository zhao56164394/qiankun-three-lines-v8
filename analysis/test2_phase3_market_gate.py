# -*- coding: utf-8 -*-
"""test2 Phase 3a: 单加大盘 (d, y_gua) gate 关火 - 针对性候选

策略: 只关 baseline 中 双视角都明确为负 的 9 个 (d, y) cell, 看是否 add-one 有效.
基于 test2 baseline 363 万扰动表分析得出.
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 363.1
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 9 个双视角差格 (来自 baseline 扰动表)
candidates = [
    {'name': 'kun_y101',  'gua': '000', 'patch': {'gate_disable_y_gua': {'101'}}},  # sig -16.94 trd -9.7
    {'name': 'gen_y011',  'gua': '001', 'patch': {'gate_disable_y_gua': {'011'}}},  # sig -6.13 trd 0
    {'name': 'gen_y101',  'gua': '001', 'patch': {'gate_disable_y_gua': {'101'}}},  # sig -16.75 trd +9.5(单笔不显著)
    {'name': 'kan_y101',  'gua': '010', 'patch': {'gate_disable_y_gua': {'101'}}},  # sig -13.91 trd -6.9
    {'name': 'xun_y101',  'gua': '011', 'patch': {'gate_disable_y_gua': {'101'}}},  # sig -6.36 trd -8.9
    {'name': 'zhen_y100', 'gua': '100', 'patch': {'gate_disable_y_gua': {'100'}}},  # sig -5.23 trd +7.8
    {'name': 'zhen_y101', 'gua': '100', 'patch': {'gate_disable_y_gua': {'101'}}},  # sig -7.43 trd +15.4
    {'name': 'li_y110',   'gua': '101', 'patch': {'gate_disable_y_gua': {'110'}}},  # sig -4.38 trd 0
    {'name': 'li_y111',   'gua': '101', 'patch': {'gate_disable_y_gua': {'111'}}},  # sig -7.67 trd -5.4
]

print(f'\nbaseline (test2) = {BASELINE} 万')
print(f'共 {len(candidates)} 候选 (来自双视角扰动表筛选), ~3 min/run\n')
print(f'  {"name":<14} {"final":>8} {"vs base":>9}  判定')

results = []
for c in candidates:
    label = f'phase3a_{c["name"]}'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches({c['gua']: c['patch']}, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test2'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    if proc.returncode != 0:
        print(f'  {c["name"]:<14} FAIL ({time.time()-t0:.0f}s)')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - BASELINE
    mark = '★★★' if diff > 50 else ('★★' if diff > 20 else ('★' if diff > 5 else ('✗' if diff < -5 else '○')))
    print(f'  {c["name"]:<14} {v:>8.1f} {diff:>+9.1f}  {mark}  ({time.time()-t0:.0f}s)')
    results.append({'name': c['name'], 'gua': c['gua'], 'patch': str(c['patch']), 'v': v, 'diff': diff})

print(f'\n汇总:')
print(f'  ★ 显著有益:')
for r in sorted([r for r in results if r['diff'] > 5], key=lambda x: -x['diff']):
    print(f'    {r["name"]}: {r["diff"]:+.1f}万')
print(f'  ✗ 有害:')
for r in sorted([r for r in results if r['diff'] < -5], key=lambda x: x['diff']):
    print(f'    {r["name"]}: {r["diff"]:+.1f}万')
print(f'  ○ 中性: {sum(1 for r in results if -5 <= r["diff"] <= 5)} 个')

with open(os.path.join(ABLATION_DIR, 'phase3a_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
