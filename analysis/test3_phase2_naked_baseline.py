# -*- coding: utf-8 -*-
"""跑一次"完全裸跑"IS baseline — 解除 000/100/101 的 tier / pool_days 约束.

目的: 暴露完整 4×4 (depth × days) 矩阵, 否则 cfg 起点会过滤掉部分 cell.
原 test3 cfg 已有约束:
  000 坤: pool_depth_tiers = days_exclude=[4,10]
  100 震: pool_days_min=1, pool_days_max=7
  101 离: pool_depth_tiers (复杂分档)
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'

# 解除 3 个卦的池深/天约束 → 完全裸跑
patches = {
    '000': {'pool_depth_tiers': None},
    '100': {'pool_days_min': None, 'pool_days_max': None},
    '101': {'pool_depth_tiers': None},
}

label = 'IS_naked_baseline'
patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
result_path = os.path.join(ABLATION_DIR, f'{label}.json')
write_patches(patches, patch_path)

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
env['STRATEGY_VERSION'] = 'test3'
env['ABLATION_PATCH_PATH'] = patch_path
env['ABLATION_RESULT_PATH'] = result_path
env['BACKTEST_END'] = IS_END

print(f'[run] {label} — 解除 000/100/101 池深池天约束')
t0 = time.time()
proc = subprocess.run(
    [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
    env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
)
elapsed = time.time() - t0
if proc.returncode != 0 or not os.path.exists(result_path):
    print(f'  FAIL ({elapsed:.0f}s)')
    if proc.stderr: print(proc.stderr[-1500:])
    sys.exit(1)
with open(result_path, encoding='utf-8') as f:
    d = json.load(f)
print(f'  终值 {d["meta"]["final_capital"]/10000:.1f}万 / {d["meta"]["trade_count"]} 笔  ({elapsed:.0f}s)')
print(f'  signal_detail: {len(d["signal_detail"])} 条')
print(f'  落地: {result_path}')
