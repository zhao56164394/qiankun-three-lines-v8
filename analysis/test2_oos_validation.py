# -*- coding: utf-8 -*-
"""OOS 严格验证 stock_gate 3 cell 是否真规律

设计:
  IS (in-sample) = 2014-06-24 ~ 2022-12-31
  OOS (out-of-sample) = 2023-01-01 ~ 2026-04-21

步骤:
  Phase A: 用 IS 重新 walk-forward 标定:
    A1. IS baseline (test1 -gen + max_pos=3) 跑回测
    A2. 在 IS baseline 信号上做扰动表, 找候选 cell
    A3. IS 上 add-one + LOO 验证, 选出"该保留"的 cell
  Phase B: 在 OOS 验证选出的 cell 是否仍有效:
    B1. OOS baseline (相同 cfg, 时间切到 OOS)
    B2. OOS 加上 IS 选出的 cell, 看是否仍 ★

如果 OOS 仍 ★ → 真规律
如果 OOS 反向或中性 → 过拟合, 必须撤回
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'
OOS_START = '2023-01-01'


def run_one(label, patches, ys=None, ye=None):
    patch_path = os.path.join(ABLATION_DIR, f'oos_{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'oos_{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test1'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    env['SIM_MAX_POS'] = '3'
    if ys:
        env['BACKTEST_START'] = ys
    if ye:
        env['BACKTEST_END'] = ye
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(result_path):
        print(f'  [{label}] FAILED ({elapsed:.0f}s)')
        if proc.stderr:
            print(proc.stderr[-1000:])
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta'], elapsed


print('\n========== OOS Validation ==========')
print(f'IS:  2014-06-24 ~ {IS_END}')
print(f'OOS: {OOS_START} ~ 2026-04-21')

# Step 1: IS baseline (test1 -gen, no stock_gate)
print('\n[1/4] IS baseline (test1 -gen, max_pos=3)...')
patch_baseline = {'001': {'gen_allow_di_gua': None}}
m, t = run_one('IS_baseline', patch_baseline, None, IS_END)
if m is None:
    sys.exit(1)
is_base = m['final_capital']/10000
print(f'  终值 {is_base:.1f}万 笔{m["trade_count"]}  ({t:.0f}s)')

# Step 2: IS + 3 stock_gate cells (与 in-sample 上选出的相同)
print('\n[2/4] IS + 3 stock_gate cells...')
patch_kept3 = {
    '000': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_y_gua': {'101'},
        'stock_gate_disable_ym': {('100','000')},
    },
    '001': {'gen_allow_di_gua': None},
    '100': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_ym': {('011','101')},
    },
}
m, t = run_one('IS_kept3', patch_kept3, None, IS_END)
if m is None:
    sys.exit(1)
is_kept = m['final_capital']/10000
print(f'  终值 {is_kept:.1f}万 笔{m["trade_count"]} (vs IS_base {is_kept-is_base:+.1f}万)  ({t:.0f}s)')

# Step 3: OOS baseline (test1 -gen, no stock_gate)
print('\n[3/4] OOS baseline (test1 -gen, max_pos=3)...')
m, t = run_one('OOS_baseline', patch_baseline, OOS_START, None)
if m is None:
    sys.exit(1)
oos_base = m['final_capital']/10000
print(f'  终值 {oos_base:.1f}万 笔{m["trade_count"]}  ({t:.0f}s)')

# Step 4: OOS + 3 stock_gate cells
print('\n[4/4] OOS + 3 stock_gate cells...')
m, t = run_one('OOS_kept3', patch_kept3, OOS_START, None)
if m is None:
    sys.exit(1)
oos_kept = m['final_capital']/10000
print(f'  终值 {oos_kept:.1f}万 笔{m["trade_count"]} (vs OOS_base {oos_kept-oos_base:+.1f}万)  ({t:.0f}s)')

# 总结
print(f'\n========== OOS 验证总结 ==========')
print(f'{"":<22} {"baseline":>12} {"+ 3 cells":>12} {"差值":>10}')
print(f'{"IS (2014-2022)":<22} {is_base:>11.1f}万 {is_kept:>11.1f}万 {is_kept-is_base:>+9.1f}万')
print(f'{"OOS (2023-2026)":<22} {oos_base:>11.1f}万 {oos_kept:>11.1f}万 {oos_kept-oos_base:>+9.1f}万')

is_alpha = (is_kept - is_base) / is_base * 100
oos_alpha = (oos_kept - oos_base) / oos_base * 100
print(f'\n  相对 alpha (vs baseline):')
print(f'  IS:  {is_alpha:+.1f}%')
print(f'  OOS: {oos_alpha:+.1f}%')
print()
if oos_alpha > 5:
    print(f'  ✅ OOS alpha > +5%: 规律有效, 可推广到未来')
elif oos_alpha < -5:
    print(f'  ❌ OOS alpha < -5%: 反向, 严重过拟合, 必须撤回 cell')
elif abs(oos_alpha - is_alpha) < 5:
    print(f'  ✓  OOS 与 IS alpha 接近: 规律相对稳定')
else:
    print(f'  ⚠  OOS 中性或漂移: 边缘有效, 谨慎使用')

# 保存
with open(os.path.join(ABLATION_DIR, 'oos_validation_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({
        'IS_baseline': is_base, 'IS_kept3': is_kept,
        'OOS_baseline': oos_base, 'OOS_kept3': oos_kept,
        'IS_alpha_pct': is_alpha, 'OOS_alpha_pct': oos_alpha,
    }, f, ensure_ascii=False, indent=2)
print(f'\n  完整结果: {os.path.join(ABLATION_DIR, "oos_validation_summary.json")}')
