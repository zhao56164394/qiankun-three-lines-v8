# -*- coding: utf-8 -*-
"""桶级 + 单维池深 + 单维池天 + 4×4 单格 LOO/add-one 消融

输入: test6 真裸 baseline 的 IS sig
LOO: 排 mask=True 的部分, baseline=池里全部, alpha = keep_mean - baseline_mean
add-one: 只留 mask=True 的部分, alpha = keep_mean - baseline_mean (与 LOO 不一样)
判定: alpha CI 下限 > 0 = 显著★; CI 上限 < 0 = 反向; 跨 0 = 灰区

只做 sig 视角, 不重跑回测 (LOO 数学切片).
按 strategy-ablation skill: 优先单维 > 多维; 信任 sig 大样本; LOO + add-one 双向必做
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPTH_BINS = [-np.inf, -400, -350, -300, -250]
DEPTH_LABELS = ['≤-400', '(-400,-350]', '(-350,-300]', '(-300,-250]']
DAYS_BINS = [-1, 3, 10, 30, 1e9]
DAYS_LABELS = ['[0-3]', '[4-10]', '[11-30]', '[31+]']
GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def load_data():
    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth', 'baseline_IS.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])

    p = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    yg = pd.read_parquet(p, columns=['date', 'y_gua'])
    yg['date'] = yg['date'].astype(str)
    yg['y_gua'] = yg['y_gua'].astype(str).str.zfill(3)
    ymap = dict(zip(yg['date'], yg['y_gua']))
    sigs['y_gua'] = sigs['buy_date'].astype(str).map(ymap)

    sigs['depth_b'] = pd.cut(sigs['pool_retail'], bins=DEPTH_BINS,
                             labels=DEPTH_LABELS, include_lowest=True)
    sigs['days_b'] = pd.cut(sigs['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)
    return sigs


def boot_alpha_ci(keep_arr, baseline_mean, n_boot=1000, seed=42):
    """alpha = keep_mean - baseline_mean, 对 keep 做 bootstrap"""
    if len(keep_arr) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    boots = rng.choice(keep_arr, size=(n_boot, len(keep_arr)), replace=True).mean(axis=1)
    alpha_boots = boots - baseline_mean
    return float(np.percentile(alpha_boots, 2.5)), float(np.percentile(alpha_boots, 97.5))


def loo_one(sig_pool, mask, label):
    """LOO: 排 mask=True 的部分"""
    n_drop = int(mask.sum())
    if n_drop < 5:
        return None
    baseline_mean = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[~mask, 'actual_ret'].values
    drop = sig_pool.loc[mask, 'actual_ret'].values
    if len(keep) < 30:
        return None
    lo, hi = boot_alpha_ci(keep, baseline_mean)
    return {
        'label': label, 'drop_n': n_drop, 'drop_mean': float(drop.mean()),
        'keep_n': len(keep), 'keep_mean': float(keep.mean()),
        'alpha': float(keep.mean() - baseline_mean),
        'ci_lo': lo, 'ci_hi': hi,
    }


def addone_one(sig_pool, mask, label):
    """add-one: 只保留 mask=True 的部分"""
    n_keep = int(mask.sum())
    if n_keep < 30:
        return None
    baseline_mean = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[mask, 'actual_ret'].values
    lo, hi = boot_alpha_ci(keep, baseline_mean)
    return {
        'label': label, 'keep_n': n_keep,
        'keep_mean': float(keep.mean()),
        'alpha': float(keep.mean() - baseline_mean),
        'ci_lo': lo, 'ci_hi': hi,
    }


def verdict_loo(c):
    if c['ci_lo'] is None:
        return '○ 样本不足'
    if c['ci_lo'] > 0:
        return '★ 显著该排'
    if c['ci_hi'] < 0:
        return '✗ 反向(该留)'
    return '○ 灰区'


def verdict_addone(c):
    if c['ci_lo'] is None:
        return '○ 样本不足'
    if c['ci_lo'] > 0:
        return '★ 显著该留'
    if c['ci_hi'] < 0:
        return '✗ 反向(该排)'
    return '○ 灰区'


def print_loo_table(results, title):
    print(f'\n### LOO {title}')
    valid = [r for r in results if r]
    if not valid:
        print('  (无有效候选)')
        return
    print(f'  {"候选":<28} {"drop_n":>6} {"drop%":>7} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>10}')
    print('  ' + '-' * 100)
    for r in sorted(valid, key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  排 {r['label']:<26} {r['drop_n']:>6} {r['drop_mean']:>+7.2f} "
              f"{r['keep_mean']:>+7.2f} {r['alpha']:>+7.2f} {ci:>16} {verdict_loo(r):>10}")


def print_addone_table(results, title):
    print(f'\n### add-one {title}')
    valid = [r for r in results if r]
    if not valid:
        print('  (无有效候选)')
        return
    print(f'  {"候选":<28} {"keep_n":>6} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>10}')
    print('  ' + '-' * 90)
    for r in sorted(valid, key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  留 {r['label']:<26} {r['keep_n']:>6} {r['keep_mean']:>+7.2f} "
              f"{r['alpha']:>+7.2f} {ci:>16} {verdict_addone(r):>10}")


def analyze_pool(s, pool_label, do_grid=False):
    bm = s['actual_ret'].mean()
    n = len(s)
    print(f'\n# {pool_label} (n={n}, baseline_mean={bm:+.2f}%)')
    if n < 50:
        print('  样本太小, 跳过')
        return

    # 单维池深
    loo_d, add_d = [], []
    for d in DEPTH_LABELS:
        mask = (s['depth_b'] == d).values
        loo_d.append(loo_one(s, mask, f'[{d}] 整行'))
        add_d.append(addone_one(s, mask, f'[{d}] 整行'))
    print_loo_table(loo_d, '— 单维池深')
    print_addone_table(add_d, '— 单维池深')

    # 单维池天
    loo_t, add_t = [], []
    for t in DAYS_LABELS:
        mask = (s['days_b'] == t).values
        loo_t.append(loo_one(s, mask, f'[{t}] 整列'))
        add_t.append(addone_one(s, mask, f'[{t}] 整列'))
    print_loo_table(loo_t, '— 单维池天')
    print_addone_table(add_t, '— 单维池天')

    if do_grid:
        loo_g = []
        for d in DEPTH_LABELS:
            for t in DAYS_LABELS:
                mask = ((s['depth_b'] == d) & (s['days_b'] == t)).values
                loo_g.append(loo_one(s, mask, f'[{d}×{t}]'))
        print_loo_table(loo_g, '— 4×4 单格 (仅 LOO, add-one 单格 n 太小)')


def main():
    sigs = load_data()
    bm = sigs['actual_ret'].mean()
    n = len(sigs)
    print(f'baseline: sig {n}, mean {bm:+.2f}%')

    # 桶级 LOO + add-one (8 桶)
    print('\n\n' + '=' * 100)
    print('# 桶级 (8 桶, 全量基线)')
    print('=' * 100)
    bucket_loo, bucket_add = [], []
    for y in ['000', '001', '010', '011', '100', '101', '110', '111']:
        mask = (sigs['y_gua'] == y).values
        if mask.sum() < 30:
            continue
        bucket_loo.append(loo_one(sigs, mask, f'{y} {GUA_NAMES[y]} 整桶'))
        bucket_add.append(addone_one(sigs, mask, f'{y} {GUA_NAMES[y]} 整桶'))
    print_loo_table(bucket_loo, '— 桶级')
    print_addone_table(bucket_add, '— 桶级')

    # 全量级单维 (跨桶)
    print('\n\n' + '=' * 100)
    print('# 全量级单维 (跨桶, 桶内分布混合)')
    print('=' * 100)
    analyze_pool(sigs, '全量', do_grid=False)

    # 大样本桶 (000/100/101/111) 桶内单维 + 4×4
    for y in ['000', '100', '101', '111']:
        sub = sigs[sigs['y_gua'] == y].copy()
        if len(sub) < 100:
            continue
        print('\n\n' + '=' * 100)
        print(f'# 桶 {y} {GUA_NAMES[y]} 桶内 (单维 + 4×4)')
        print('=' * 100)
        analyze_pool(sub, f'{y} {GUA_NAMES[y]}', do_grid=True)


if __name__ == '__main__':
    main()
