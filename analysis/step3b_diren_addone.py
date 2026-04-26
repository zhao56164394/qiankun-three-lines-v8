# -*- coding: utf-8 -*-
"""Step 3b: 反向消融 — 单加 di/ren 一条, 看是否提升 baseline=1216.8万

baseline = 无 di/ren (=1216.8 万)
single_i = 只加候选 i 一条
判定: single_i > baseline → 这条 di/ren 单独有正贡献 → 保留
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

candidates = [
    {'name': 'kun_exclude_ren_000_110','gua': '000', 'patch': {'kun_exclude_ren_gua': {'000','110'}}},
    {'name': 'kun_allow_di_110',      'gua': '000', 'patch': {'kun_allow_di_gua': {'110'}}},
    {'name': 'gen_allow_di_000_010',  'gua': '001', 'patch': {'gen_allow_di_gua': {'000','010'}}},
    {'name': 'xun_allow_di_010',      'gua': '011', 'patch': {'xun_allow_di_gua': {'010'}}},
    {'name': 'zhen_exclude_ren_001_011','gua': '100','patch': {'zhen_exclude_ren_gua': {'001','011'}}},
    {'name': 'dui_exclude_ren_100_110','gua': '110','patch': {'dui_exclude_ren_gua': {'100','110'}}},
    {'name': 'dui_allow_di_000_010_110','gua': '110','patch': {'dui_allow_di_gua': {'000','010','110'}}},
    {'name': 'qian_exclude_di_101_111','gua': '111','patch': {'qian_exclude_di_gua': {'101','111'}}},
]

baseline_v = 1216.8
print(f'\nbaseline (naked = 无 di/ren) = {baseline_v} 万\n')
print(f'{"候选":<32} {"终值万":>8} {"vs baseline":>10}')

for c in candidates:
    patches = {c['gua']: c['patch']}
    label = f'step3b_single_{c["name"]}'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), '..', 'backtest_8gua_naked.py')],
        env=env, capture_output=True, encoding='utf-8', errors='replace',
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'{c["name"]:<32} FAIL ({elapsed:.0f}s)')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - baseline_v
    mark = '★' if diff > 5 else ('✗' if diff < -5 else '○')
    print(f'{c["name"]:<32} {v:>8.1f} {diff:>+10.1f}  {mark}  ({elapsed:.0f}s)')
