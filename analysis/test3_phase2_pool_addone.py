# -*- coding: utf-8 -*-
"""Phase 2 池深/池天 IS add-one 验证

在 IS baseline (320.9万/322笔) 上, 逐个加候选 cell 排除规则, 看 IS 终值变化.
全部决策都在 IS 上做, 不动 OOS.

候选 (来自 phase2_perturb 表, sig_n>=15 + sig_mean<-3 + 业务可解释):

depth 维:
  A1 排 011 巽 ≤-400 极深  (sig_n=1169, mean=-7.57)  暴跌中底爆=陷阱
  A2 排 101 离 (-300,-250] 浅 (sig_n=51, mean=-7.24) 高位浅回调=接最后一棒
  A3 排 010 坎 ≤-400 极深  (sig_n=64,  mean=-5.77)  暴跌中乏力反弹=死猫跳
  A4 排 110 兑 (-350,-300] 中 (sig_n=22, mean=-5.06) 牛末中回撤=顶部前奏

days 维:
  B1 排 101 离 [4-10] 磨底  (sig_n=52,  mean=-6.56)  高位长磨=出货征兆
  B2 排 011 巽 [11-30] 物极 (sig_n=1103, mean=-3.48) 久磨爆发=虚假
"""
import os, sys, json, subprocess, time
os.environ['STRATEGY_VERSION'] = 'test3'  # 必须在 import ablation 之前
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_END = '2023-01-01'

# 已知 IS baseline (test3 cfg + max_pos=5 + IS 切片)
IS_BASELINE = 320.9  # 万

# 6 个候选, 每个独立 patch; tier 写法保留卦原 cfg 的"架构性约束"
# 注: 101 离当前 cfg 已有 tier, 候选 A2/B1 是修改它; 其他卦无 tier, 候选直接添加
candidates = [
    {
        'name': 'A1_xun_deepest',
        'desc': '排 011 巽 ≤-400 极深',
        'gua': '011',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},  # 极深永不接受
                {'depth_max': None, 'days_min': 0, 'days_max': None},       # 其他全接
            ],
        },
    },
    {
        'name': 'A2_li_shallow',
        'desc': '排 101 离 (-300,-250] 浅',
        'gua': '101',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 99999, 'days_max': None},   # 极深永不接 (原 cfg)
                {'depth_max': -300, 'days_min': 0, 'days_max': 15},          # 中档 0-15 天
                # >-300 的浅档不写 tier, 自动拒绝
            ],
        },
    },
    {
        'name': 'A3_kan_deepest',
        'desc': '排 010 坎 ≤-400 极深',
        'gua': '010',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},
                {'depth_max': None, 'days_min': 0, 'days_max': None},
            ],
        },
    },
    {
        'name': 'A4_dui_mid',
        'desc': '排 110 兑 (-350,-300] 中',
        'gua': '110',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 0, 'days_max': None},        # 深 ≤-350 接受
                {'depth_max': -300, 'days_min': 99999, 'days_max': None},    # 中 (-350,-300] 拒
                {'depth_max': None, 'days_min': 0, 'days_max': None},        # 浅接受
            ],
        },
    },
    {
        'name': 'B1_li_4_10',
        'desc': '排 101 离 [4-10] 磨底',
        'gua': '101',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 99999, 'days_max': None},
                {'depth_max': -250, 'days_min': 0, 'days_max': 15, 'days_exclude': [4, 10]},
            ],
        },
    },
    {
        'name': 'B2_xun_11_30',
        'desc': '排 011 巽 [11-30] 物极',
        'gua': '011',
        'patch': {
            'pool_depth_tiers': [
                {'depth_max': None, 'days_min': 0, 'days_max': None, 'days_exclude': [11, 30]},
            ],
        },
    },
]


def run_one(label, patches, ys=None, ye=None):
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, f'{label}.json')
    write_patches(patches, patch_path)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['STRATEGY_VERSION'] = 'test3'
    env['ABLATION_PATCH_PATH'] = patch_path
    env['ABLATION_RESULT_PATH'] = result_path
    if ys: env['BACKTEST_START'] = ys
    if ye: env['BACKTEST_END'] = ye
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(result_path):
        print(f'  [{label}] FAIL ({elapsed:.0f}s)')
        if proc.stderr: print(proc.stderr[-1000:])
        return None, elapsed
    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']['final_capital']/10000, elapsed


print(f'\n=== Phase 2 Pool Add-one (IS only, 2014-2022) ===')
print(f'IS baseline: {IS_BASELINE}万 (test3 cfg, max_pos=5)\n')
print(f'  {"name":<22} {"desc":<32} {"final":>9} {"vs base":>9}  判定')

results = []
for c in candidates:
    label = f'phase2_addone_{c["name"]}'
    patches = {c['gua']: c['patch']}
    v, t = run_one(label, patches, None, IS_END)
    if v is None:
        results.append({'name': c['name'], 'desc': c['desc'], 'final': None, 'verdict': 'FAIL'})
        continue
    diff = v - IS_BASELINE
    if diff > 5:
        verdict = '★'
    elif diff < -5:
        verdict = '✗'
    else:
        verdict = '○'
    if diff > 50:
        verdict = '★★'
    if diff > 200:
        verdict = '★★★'
    print(f'  {c["name"]:<22} {c["desc"]:<32} {v:>9.1f} {diff:>+9.1f}  {verdict} ({t:.0f}s)')
    results.append({'name': c['name'], 'desc': c['desc'], 'gua': c['gua'],
                    'final_wan': v, 'diff_wan': diff, 'verdict': verdict})

print(f'\n=== 总结 ===')
star = [r for r in results if r.get('verdict', '').startswith('★')]
neutral = [r for r in results if r.get('verdict') == '○']
bad = [r for r in results if r.get('verdict') == '✗']
print(f'  ★ 独立有益 (留): {len(star)}')
for r in star: print(f'    - {r["name"]}: {r["desc"]} (+{r["diff_wan"]:.1f}万)')
print(f'  ○ 中性 (留, 阶段二再判): {len(neutral)}')
for r in neutral: print(f'    - {r["name"]}: {r["desc"]} ({r["diff_wan"]:+.1f}万)')
print(f'  ✗ 独立有害 (剔): {len(bad)}')
for r in bad: print(f'    - {r["name"]}: {r["desc"]} ({r["diff_wan"]:+.1f}万)')

with open(os.path.join(ABLATION_DIR, 'phase2_addone_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({'IS_baseline_wan': IS_BASELINE, 'IS_end': IS_END, 'results': results},
              f, ensure_ascii=False, indent=2)
print(f'\n  落地: {os.path.join(ABLATION_DIR, "phase2_addone_summary.json")}')
