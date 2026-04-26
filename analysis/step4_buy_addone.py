# -*- coding: utf-8 -*-
"""Step 4: 买点择优 add-one 消融

baseline = 当前 cfg (1466.9万).
single_i = 在 baseline 基础上单改一个分支的买点设置, 看总收益变化.

包含:
  6 个分支模式切换 (double_rise <-> cross)
  111 qian threshold 阈值扫描 (30/40/50/70/80)
  110 dui threshold 阈值扫描 (10/30/40)
  011 xun double_rise threshold 扫描 (5/15/20) — xun_buy_param

判定: single_i > baseline +5万 = 改动有益; < baseline -5万 = 改动有害
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 1466.9

candidates = [
    # 模式切换
    {'name': 'kun_to_cross',     'gua': '000', 'patch': {'kun_buy_mode': 'cross', 'kun_cross_threshold': 20}},
    {'name': 'gen_to_cross',     'gua': '001', 'patch': {'gen_buy_mode': 'cross', 'gen_cross_threshold': 20}},
    {'name': 'xun_to_cross',     'gua': '011', 'patch': {'xun_buy': 'cross', 'xun_buy_param': 20}},
    {'name': 'zhen_to_cross',    'gua': '100', 'patch': {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 20}},
    {'name': 'li_to_cross',      'gua': '101', 'patch': {'li_buy_mode': 'cross', 'li_cross_threshold': 20}},
    {'name': 'dui_to_double',    'gua': '110', 'patch': {'dui_buy_mode': 'double_rise'}},
    # qian 阈值
    {'name': 'qian_thresh_30',   'gua': '111', 'patch': {'qian_cross_threshold': 30}},
    {'name': 'qian_thresh_40',   'gua': '111', 'patch': {'qian_cross_threshold': 40}},
    {'name': 'qian_thresh_50',   'gua': '111', 'patch': {'qian_cross_threshold': 50}},
    {'name': 'qian_thresh_70',   'gua': '111', 'patch': {'qian_cross_threshold': 70}},
    {'name': 'qian_thresh_80',   'gua': '111', 'patch': {'qian_cross_threshold': 80}},
    # dui 阈值
    {'name': 'dui_thresh_10',    'gua': '110', 'patch': {'dui_cross_threshold': 10}},
    {'name': 'dui_thresh_30',    'gua': '110', 'patch': {'dui_cross_threshold': 30}},
    {'name': 'dui_thresh_40',    'gua': '110', 'patch': {'dui_cross_threshold': 40}},
    # xun_buy_param (double_rise 下的 trend>X 阈值)
    {'name': 'xun_param_5',      'gua': '011', 'patch': {'xun_buy_param': 5}},
    {'name': 'xun_param_15',     'gua': '011', 'patch': {'xun_buy_param': 15}},
    {'name': 'xun_param_20',     'gua': '011', 'patch': {'xun_buy_param': 20}},
]

print(f'\nbaseline = {BASELINE} 万 (3.7 min/run, 共 {len(candidates)} 个候选, ~{len(candidates)*4} min)\n')
print(f'{"候选":<25} {"终值万":>9} {"vs base":>9}  判定')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for c in candidates:
    patches = {c['gua']: c['patch']}
    label = f'step4b_single_{c["name"]}'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, capture_output=True, encoding='utf-8', errors='replace',
        cwd=ROOT,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'{c["name"]:<25} FAIL ({elapsed:.0f}s)')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - BASELINE
    mark = '★' if diff > 5 else ('✗' if diff < -5 else '○')
    print(f'{c["name"]:<25} {v:>9.1f} {diff:>+9.1f}  {mark}  ({elapsed:.0f}s)')
