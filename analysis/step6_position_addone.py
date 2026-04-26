# -*- coding: utf-8 -*-
"""Step 6: 仓位 / 每日限买 / 初始资金 add-one 消融

baseline = 当前 cfg (max_pos=5, daily_limit=1, capital=200000) = 2582 万 (Step 4 落地后)

考虑 Step 4 离卦换 cross 后信号变多, 资金挤压加剧, 该实验关键.
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 2582.0

# (env_dict, label)
candidates = [
    ({'SIM_MAX_POS': '3'},                  'pos_3'),
    ({'SIM_MAX_POS': '4'},                  'pos_4'),
    ({'SIM_MAX_POS': '6'},                  'pos_6'),
    ({'SIM_MAX_POS': '7'},                  'pos_7'),
    ({'SIM_MAX_POS': '8'},                  'pos_8'),
    ({'SIM_MAX_POS': '10'},                 'pos_10'),
    ({'SIM_DAILY_LIMIT': '2'},              'daily_2'),
    ({'SIM_DAILY_LIMIT': '3'},              'daily_3'),
    ({'SIM_MAX_POS': '6', 'SIM_DAILY_LIMIT': '2'},   'pos_6_daily_2'),
    ({'SIM_MAX_POS': '7', 'SIM_DAILY_LIMIT': '2'},   'pos_7_daily_2'),
    ({'SIM_MAX_POS': '8', 'SIM_DAILY_LIMIT': '2'},   'pos_8_daily_2'),
    ({'SIM_MAX_POS': '10', 'SIM_DAILY_LIMIT': '2'},  'pos_10_daily_2'),
]

print(f'\nbaseline = {BASELINE} 万 (~2 min/run, 共 {len(candidates)} 个候选)\n')
print(f'{"候选":<22} {"终值万":>9} {"vs base":>9}  判定')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for env_overrides, label in candidates:
    result_path = os.path.join(ABLATION_DIR, f'step6_{label}.json')
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['ABLATION_RESULT_PATH'] = result_path
    env.update(env_overrides)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, capture_output=True, encoding='utf-8', errors='replace',
        cwd=ROOT,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'{label:<22} FAIL ({elapsed:.0f}s)')
        continue
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    v = d['meta']['final_capital']/10000
    diff = v - BASELINE
    mark = '★★★' if diff > 500 else ('★★' if diff > 100 else ('★' if diff > 5 else ('✗' if diff < -5 else '○')))
    print(f'{label:<22} {v:>9.1f} {diff:>+9.1f}  {mark}  ({elapsed:.0f}s)')
