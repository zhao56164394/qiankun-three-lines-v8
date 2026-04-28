# -*- coding: utf-8 -*-
"""stock_gate 联合 (stk_y, stk_m) 二维 add-one 验证
跑 4 个新联合 cell, 不与单维候选重叠.
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 4425.5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 个新联合 cell (基于 (d, stk_y, stk_m) 双视角扫描)
candidates = [
    {'name': 'kun_y101_m100', 'gua': '000', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_ym': {('101','100')}}},
    {'name': 'kun_y100_m000', 'gua': '000', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_ym': {('100','000')}}},
    {'name': 'gen_y111_m100', 'gua': '001', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_ym': {('111','100')}}},
    {'name': 'zhen_y011_m101','gua': '100', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_ym': {('011','101')}}},
]

print(f'\nbaseline {BASELINE} 万')
print(f'{"name":<22} {"final":>9} {"vs base":>9}  判定')

results = []
for c in candidates:
    label = f'stkgate_combo_{c["name"]}'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches({c['gua']: c['patch']}, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test1'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    env['SIM_MAX_POS'] = '3'
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    if proc.returncode != 0:
        print(f'  {c["name"]:<22} FAIL ({time.time()-t0:.0f}s)')
        continue
    if not os.path.exists(result_path):
        print(f'  {c["name"]:<22} no file')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - BASELINE
    mark = '★★★' if diff > 1000 else ('★★' if diff > 200 else ('★' if diff > 5 else ('✗' if diff < -5 else '○')))
    print(f'  {c["name"]:<22} {v:>9.1f} {diff:>+9.1f}  {mark} ({time.time()-t0:.0f}s)')
    # patch 转 str 避免 set JSON 错
    results.append({'name': c['name'], 'gua': c['gua'], 'patch_str': str(c['patch']), 'v': v, 'diff': diff})

with open(os.path.join(ABLATION_DIR, 'stkgate_combo_addone.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
