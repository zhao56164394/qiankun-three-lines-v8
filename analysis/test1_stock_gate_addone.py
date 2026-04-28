# -*- coding: utf-8 -*-
"""stock_gate add-one 实验 — 在 test1 -gen + max_pos=3 的 baseline (4425万) 上,
单加每个 stock_gate cell 看效果, const 和 std 两版各跑.
"""
import os, sys, json, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import write_patches, ABLATION_DIR

BASELINE = 4425.5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 双视角差格候选
candidates = [
    # const 版差格
    {'name': 'const_kun_y101',   'gua': '000', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_y_gua': {'101'}}, 'data_ver': 'const'},
    {'name': 'const_gen_y111',   'gua': '001', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_y_gua': {'111'}}, 'data_ver': 'const'},
    {'name': 'const_dui_y000',   'gua': '110', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_y_gua': {'000'}}, 'data_ver': 'const'},
    # std 版差格
    {'name': 'std_dui_y000',     'gua': '110', 'patch': {'gen_allow_di_gua': None, 'stock_gate_disable_y_gua': {'000'}}, 'data_ver': 'std'},
]

# 注: gen_allow_di_gua 必须显式 patch 成 None, 否则 STRATEGY_TEST1 默认还有这个字段


def get_smg_path(version):
    if version == 'const':
        return os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily_const.parquet')
    return os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet')


def make_link(version):
    """主代码读 stock_multi_scale_gua_daily.parquet, 我们用复制切换"""
    target = get_smg_path(version)
    link = os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet')
    if not os.path.exists(target):
        raise FileNotFoundError(f'目标数据不存在: {target}')
    import shutil
    if os.path.exists(link) and os.path.samefile(link, target):
        return  # 已是目标版本
    if os.path.exists(link):
        os.remove(link)
    shutil.copy(target, link)


print(f'\nbaseline (test1 -gen + max_pos=3) = {BASELINE} 万')
print(f'  目标: 加 stock_gate 后总收益超过 6217 万 (test1 含 gen_allow_di_gua)')
print(f'  net win: stock_gate 该 cell 关掉后比 baseline 多赚多少, 与 +1791 万 (gen 的 alpha) 对比')
print()
print(f'{"name":<22} {"data":<6} {"final":>9} {"vs base":>9}  判定')

# 备份当前 stock_multi_scale_gua_daily (std 版)
import shutil
orig = os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet')
backup = orig + '.bak'
if not os.path.exists(backup):
    shutil.copy(orig, backup)

results = []
try:
    last_data_ver = None
    for c in candidates:
        # 切换数据源
        if c['data_ver'] != last_data_ver:
            make_link(c['data_ver'])
            last_data_ver = c['data_ver']

        label = f'stkgate_{c["name"]}'
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
            print(proc.stderr[-500:])
            continue
        if not os.path.exists(result_path):
            print(f'  {c["name"]:<22} no result file')
            continue
        with open(result_path, encoding='utf-8') as f:
            d = json.load(f)
        v = d['meta']['final_capital']/10000
        diff = v - BASELINE
        mark = '★★★' if diff > 500 else ('★★' if diff > 100 else ('★' if diff > 5 else ('✗' if diff < -5 else '○')))
        print(f'  {c["name"]:<22} {c["data_ver"]:<6} {v:>9.1f} {diff:>+9.1f}  {mark} ({time.time()-t0:.0f}s)')
        results.append({**c, 'v': v, 'diff': diff})
finally:
    # 还原 std 版
    shutil.copy(backup, orig)
    print(f'\n[还原] stock_multi_scale_gua_daily.parquet → std 版')

# 汇总
print(f'\n{"="*60}')
print(f'  对比 stock_gate vs gen_allow_di_gua (+1791万)')
print(f'{"="*60}')
const_max = max([r['diff'] for r in results if r.get('data_ver')=='const'], default=0)
std_max = max([r['diff'] for r in results if r.get('data_ver')=='std'], default=0)
print(f'  const 版最佳 add-one: {const_max:+.1f}万 (vs gen +1791 万 → {"赢" if const_max > 1791 else "输"})')
print(f'  std 版最佳 add-one:   {std_max:+.1f}万 (vs gen +1791 万 → {"赢" if std_max > 1791 else "输"})')

with open(os.path.join(ABLATION_DIR, 'stock_gate_addone_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
