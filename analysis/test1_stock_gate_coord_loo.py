# -*- coding: utf-8 -*-
"""stock_gate 协同 LOO 验证

策略: 把所有 ★★ 以上的 stock_gate 候选一起加 (full), 然后逐个 leave-one-out,
看哪些是真的独立有益, 哪些是协同冲突.

注意嵌套关系:
  - 000 坤 stk_y=101 与 (stk_y=101, stk_m=100) 嵌套 → 优先选粗粒度 stk_y=101
  - 100 震 stk_m=101 与 (stk_y=011, stk_m=101) 嵌套 → 优先选粗粒度 stk_m=101
  - 110 兑 stk_y=000 与 stk_m=001 是非嵌套的 (维度不同, 但可能交集大)

候选 6 个 (去重 + 嵌套分组各取最强):
  dui_stky_000   (110 兑 stk_y=000)        +5343
  dui_stkm_001   (110 兑 stk_m=001)        +4676   [可能与上重叠]
  kun_stky_101   (000 坤 stk_y=101)        +3388   [优于嵌套子集 +2171]
  zhen_stky011_stkm101 (100 震 y=011 m=101) +3503  [优于父集 stk_m=101 +1913]
  kun_stky100_stkm000  (000 坤 y=100 m=000) +240
  gen_stky_111   (001 艮 stk_y=111)        +152
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 4425.5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 所有候选 patches (按分支聚合)
ALL_PATCHES = {
    '000': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_y_gua': {'101'},
        'stock_gate_disable_ym': {('100','000')},
    },
    '001': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_y_gua': {'111'},
    },
    '100': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_ym': {('011','101')},
    },
    '110': {
        'gen_allow_di_gua': None,
        'stock_gate_disable_y_gua': {'000'},
        'stock_gate_disable_m_gua': {'001'},
    },
}

# 为 LOO 设计候选名: 每个 patch 字段单独成为 1 个候选
candidates = [
    {'name': 'dui_y000',           'gua': '110', 'field': 'stock_gate_disable_y_gua', 'value': {'000'}},
    {'name': 'dui_m001',           'gua': '110', 'field': 'stock_gate_disable_m_gua', 'value': {'001'}},
    {'name': 'kun_y101',           'gua': '000', 'field': 'stock_gate_disable_y_gua', 'value': {'101'}},
    {'name': 'zhen_y011m101',      'gua': '100', 'field': 'stock_gate_disable_ym',   'value': {('011','101')}},
    {'name': 'kun_y100m000',       'gua': '000', 'field': 'stock_gate_disable_ym',   'value': {('100','000')}},
    {'name': 'gen_y111',           'gua': '001', 'field': 'stock_gate_disable_y_gua', 'value': {'111'}},
]


def make_patches(active_names):
    """根据激活的候选名生成 patches dict"""
    patches = {}
    # 所有 cfg 都需要 gen_allow_di_gua=None
    patches['001'] = {'gen_allow_di_gua': None}
    for c in candidates:
        if c['name'] not in active_names:
            continue
        gua = c['gua']
        if gua not in patches:
            patches[gua] = {}
        if gua == '001' and 'gen_allow_di_gua' not in patches[gua]:
            patches[gua]['gen_allow_di_gua'] = None
        patches[gua][c['field']] = c['value']
    return patches


def run_one(label, patches):
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
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
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(result_path):
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000, elapsed


print(f'\n=== Stock Gate Coordination LOO ===')
print(f'baseline {BASELINE}万, {len(candidates)} 候选')
print()

# Full
all_names = set(c['name'] for c in candidates)
full_v, full_t = run_one('stkgate_loo_full', make_patches(all_names))
if full_v is None:
    print('FULL FAILED')
    sys.exit(1)
diff_full = full_v - BASELINE
print(f'  FULL (全部 6 个加): {full_v:.1f}万 (vs base {diff_full:+.1f}万)  ({full_t:.0f}s)')
print()

# LOO 每次去掉 1 个
print(f'  --- LOO (敲掉 1 个其他全保留) ---')
print(f'  {"name":<22} {"final":>9} {"vs full":>9} {"vs base":>9}  判定')
results = []
for c in candidates:
    active = all_names - {c['name']}
    v, t = run_one(f'stkgate_loo_drop_{c["name"]}', make_patches(active))
    if v is None:
        print(f'  {c["name"]:<22} FAIL')
        continue
    diff_full = v - full_v
    diff_base = v - BASELINE
    if diff_full > 5:
        verdict = '✗ 该字段独立有害'  # 敲掉它整体反升
    elif diff_full < -5:
        verdict = '★ 该字段独立有益'  # 敲掉它整体下降
    else:
        verdict = '○ 边际中性'
    print(f'  {c["name"]:<22} {v:>9.1f} {diff_full:>+9.1f} {diff_base:>+9.1f}  {verdict} ({t:.0f}s)')
    results.append({'name': c['name'], 'v': v, 'diff_full': diff_full, 'diff_base': diff_base, 'verdict': verdict})

# 落地决定
print(f'\n=== 决定 ===')
keep = [r for r in results if r['verdict'].startswith('★') or r['verdict'].startswith('○')]
drop = [r for r in results if r['verdict'].startswith('✗')]
print(f'  保留 {len(keep)} 个: {[r["name"] for r in keep]}')
print(f'  敲掉 {len(drop)} 个: {[r["name"] for r in drop]}')

with open(os.path.join(ABLATION_DIR, 'stkgate_coord_loo_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({'baseline': BASELINE, 'full': full_v, 'loo': results}, f, ensure_ascii=False, indent=2)
