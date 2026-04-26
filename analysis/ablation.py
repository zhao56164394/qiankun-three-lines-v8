# -*- coding: utf-8 -*-
"""消融实验工具 — 给定 cfg patch 集合, 跑 N+1 次回测验证各 patch 的独立有效性

工作流:
  Step 1: 列出候选 patch (例如: 5 个 ym gate 候选)
  Step 2: baseline 回测 (不加任何候选)
  Step 3: full 回测 (一次性加全部候选)
  Step 4: 逐个去掉一个 patch 跑 leave-one-out
          → 如果 LOO_i > full, 第 i 个 patch 该去掉 (独立有害)
          → 如果 LOO_i < full, 第 i 个 patch 该留 (独立有益)
  Step 5: 保留所有"该留"的 patch

通过环境变量 PATCH_PATH 传入 cfg patch 文件 (json 格式), 让 backtest_8gua_naked.py 应用.

直接调用方法:
  from analysis.ablation import run_ablation
  result = run_ablation(
      candidates=[
          {'name': 'A', 'gua': '110', 'patch': {'gate_disable_ym': {('111','111')}}},
          {'name': 'B', 'gua': '111', 'patch': {'gate_disable_ym': {('011','101')}}},
      ],
      tag='stage_xxx',
  )
"""
import copy, json, os, subprocess, sys, time
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT, 'data_layer', 'data')
# 消融数据按 STRATEGY_VERSION 分子目录, 默认 test1
_ABL_VERSION = os.environ.get('STRATEGY_VERSION', 'test1')
ABLATION_DIR = os.path.join(RESULT_DIR, 'ablation', _ABL_VERSION)
os.makedirs(ABLATION_DIR, exist_ok=True)


def _set_to_list(v):
    if isinstance(v, set):
        return sorted([list(x) if isinstance(x, tuple) else x for x in v],
                      key=lambda x: str(x))
    return v


def _serialize_patch(patch_dict):
    """把 patch_dict 中的 set 序列化, 让 JSON 能存"""
    out = {}
    for k, v in patch_dict.items():
        if isinstance(v, set):
            out[k] = {'__set__': True,
                      'items': [list(x) if isinstance(x, tuple) else x for x in v]}
        else:
            out[k] = v
    return out


def write_patches(patches_by_gua, path):
    """patches_by_gua: {gua: {field: value, ...}, ...}"""
    out = {g: _serialize_patch(p) for g, p in patches_by_gua.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def run_one(label, patches_by_gua):
    """跑一次回测, 返回 meta dict.
    patches_by_gua: {gua_code: {field: set/list/value, ...}}, None 或 {} 表示 baseline
    """
    fname = f'{label}.json'
    patch_path = os.path.join(ABLATION_DIR, f'{label}_patch.json')
    result_path = os.path.join(ABLATION_DIR, fname)

    if patches_by_gua:
        write_patches(patches_by_gua, patch_path)
        env_patch = patch_path
    else:
        env_patch = ''

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['ABLATION_PATCH_PATH'] = env_patch
    env['ABLATION_RESULT_PATH'] = result_path

    t0 = time.time()
    print(f'\n[{label}] 启动 (patches={list(patches_by_gua.keys()) if patches_by_gua else "baseline"})')
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'backtest_8gua_naked.py')],
        env=env, cwd=ROOT, capture_output=True,
        encoding='utf-8', errors='replace',
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f'[{label}] FAILED in {elapsed:.0f}s')
        print(proc.stderr[-2000:])
        raise RuntimeError(f'ablation run failed: {label}')
    print(f'[{label}] 完成 ({elapsed:.0f}s)')

    with open(result_path, encoding='utf-8') as f:
        d = json.load(f)
    return d['meta']


def run_ablation(candidates: List[Dict[str, Any]], tag: str):
    """
    candidates: [{'name':, 'gua':, 'patch': {field:val,...}}, ...]
       patch 在该 gua 上 update GUA_STRATEGY[gua]
       语义: patch 表示"这个候选启用时的字段值"
    tag: 实验标签

    流程:
      baseline = 全部 candidate 都不应用 (默认 cfg)
        → 注意: 如果想把 baseline 设为"敲掉所有候选", 需要在 candidate.patch
                里写"敲掉的状态", 而不是"启用的状态". 这种情况下 full 才是
                "全部敲掉", LOO 是"敲掉除 i 之外", baseline 是"全部启用".

    一般用法:
      candidate.patch = "把当前字段改成什么样"
      → full = 一次性应用所有候选 (e.g. 全部敲掉)
      → LOO_i = 应用除 i 之外的所有 (留 i 的原状, 敲掉其他)
      → baseline = 不应用任何候选 (=当前 cfg, 如全部启用)

    判定 (与 full 对比):
      LOO_i 比 full 显著好 → 候选 i 的 patch 有害 → 不该 patch (维持原状更好)
      LOO_i 比 full 显著差 → 候选 i 的 patch 有益 → 应该 patch
    """
    print(f'\n{"="*80}')
    print(f'  ABLATION 实验: {tag}')
    print(f'  候选数: {len(candidates)}')
    print(f'{"="*80}')
    for c in candidates:
        print(f"  - {c['name']:<20}  gua={c['gua']}  patch={c['patch']}")

    # baseline (空 patch)
    baseline = run_one(f'{tag}_baseline', {})

    # full (全部候选合并)
    full_patches = {}
    for c in candidates:
        g = c['gua']
        if g not in full_patches:
            full_patches[g] = {}
        for k, v in c['patch'].items():
            if k in full_patches[g] and isinstance(v, set):
                full_patches[g][k] = full_patches[g][k] | v
            else:
                full_patches[g][k] = copy.deepcopy(v) if isinstance(v, (set, list, dict)) else v
    full = run_one(f'{tag}_full', full_patches)

    # leave-one-out: 每次去掉一个候选
    loo = {}
    for i, c in enumerate(candidates):
        loo_patches = {}
        for j, c2 in enumerate(candidates):
            if i == j: continue
            g = c2['gua']
            if g not in loo_patches:
                loo_patches[g] = {}
            for k, v in c2['patch'].items():
                if k in loo_patches[g] and isinstance(v, set):
                    loo_patches[g][k] = loo_patches[g][k] | v
                else:
                    loo_patches[g][k] = copy.deepcopy(v) if isinstance(v, (set, list, dict)) else v
        loo[c['name']] = run_one(f'{tag}_loo_{c["name"]}', loo_patches)

    # 分析
    print(f'\n{"="*80}')
    print(f'  ABLATION 结果汇总: {tag}')
    print(f'{"="*80}')
    print(f'  baseline (当前cfg, 不应用任何 patch):  终值 {baseline["final_capital"]/10000:>8.1f}万  收益 {baseline["total_return"]:+8.2f}%  回撤 {baseline["max_dd"]:.2f}')
    print(f'  full (全部 patch 应用, 即"全部敲掉"):  终值 {full["final_capital"]/10000:>8.1f}万  收益 {full["total_return"]:+8.2f}%  回撤 {full["max_dd"]:.2f}')
    print(f'  full vs base:  {(full["final_capital"]-baseline["final_capital"])/10000:+8.1f}万 (整体敲掉影响)')
    print()
    print(f'  --- 逐个 leave-one-out (敲掉除 i 之外的所有, 即只保留字段 i) ---')
    print(f'  语义: LOO_i = "保留字段 i, 敲掉其他"')
    print(f'        LOO_i 比 full 高 → 保留字段 i 有益 (建议留)')
    print(f'        LOO_i 比 full 低 → 保留字段 i 有害 (建议敲)')
    print(f'  {"name":<28} {"LOO 终值万":>12} {"vs full":>10} {"vs base":>10} {"独立判定"}')

    analysis = []
    for c in candidates:
        m = loo[c['name']]
        loo_v = m['final_capital'] / 10000
        full_v = full['final_capital'] / 10000
        base_v = baseline['final_capital'] / 10000
        diff_full = loo_v - full_v
        diff_base = loo_v - base_v
        # 关键: LOO_i 表示"只保留 i, 敲掉其他".
        #       如果 LOO_i > full → 保留 i 比全敲好 → 字段 i 该保留
        #       如果 LOO_i < full → 保留 i 比全敲差 → 字段 i 该敲掉
        if diff_full > 5:  # >5万 才认为显著
            verdict = '★ 保留字段 i (独立有益)'
            keep = True
        elif diff_full < -5:
            verdict = '✗ 敲掉字段 i (独立有害)'
            keep = False
        else:
            verdict = '○ 边际中性 (建议保留, 但作用小)'
            keep = True
        analysis.append({
            'name': c['name'], 'gua': c['gua'], 'patch': str(c['patch']),
            'loo_final': loo_v, 'diff_full': diff_full, 'diff_base': diff_base,
            'verdict': verdict, 'keep': keep,
        })
        print(f'  {c["name"]:<28} {loo_v:>12.1f} {diff_full:>+10.1f} {diff_base:>+10.1f}  {verdict}')

    keep_set = [a for a in analysis if a['keep']]
    drop_set = [a for a in analysis if not a['keep']]
    print(f'\n  → 保留 {len(keep_set)} 个候选, 去掉 {len(drop_set)} 个候选')
    if drop_set:
        print(f'    去掉的: {[a["name"] for a in drop_set]}')

    summary = {
        'tag': tag,
        'baseline': baseline,
        'full': full,
        'loo': loo,
        'analysis': analysis,
    }
    summary_path = os.path.join(ABLATION_DIR, f'{tag}_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n  完整结果: {summary_path}')
    return summary
