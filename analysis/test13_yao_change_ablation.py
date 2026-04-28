# -*- coding: utf-8 -*-
"""变爻 (from→to) LOO + add-one 双向消融

把 from→to 当成新的分治维度 (替代静态 y_gua).
LOO: 排掉某 from→to 后, 剩余 sig 全量的 mean 比 baseline 提升多少 (alpha)
add-one: 只留某 from→to 后的 mean 与 baseline 比较

判定: alpha 95% CI 完全不跨 0 = 显著 (★ / ✗)
双向自洽: LOO ★ 该排 + add-one ✗反向(该排) → 真规律
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def load_data():
    p = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    df_y = pd.read_parquet(p, columns=['date', 'y_gua'])
    df_y['date'] = df_y['date'].astype(str)
    df_y['y_gua'] = df_y['y_gua'].astype(str).str.zfill(3)
    df_y = df_y.drop_duplicates('date').sort_values('date').reset_index(drop=True)

    df_y['prev_y'] = df_y['y_gua'].shift(1)
    df_y['is_change'] = (df_y['y_gua'] != df_y['prev_y']) & df_y['prev_y'].notna()
    df_y['last_from'] = df_y['prev_y'].where(df_y['is_change']).ffill()
    df_lookup = df_y.set_index('date')[['y_gua', 'last_from']]

    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth', 'baseline_IS.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    sigs['buy_date'] = sigs['buy_date'].astype(str)
    sigs['y_gua'] = sigs['buy_date'].map(df_lookup['y_gua'])
    sigs['last_from'] = sigs['buy_date'].map(df_lookup['last_from'])
    sigs = sigs.dropna(subset=['last_from'])  # 去掉 IS 起点前的信号 (无 from)
    sigs['change_type'] = sigs['last_from'].astype(str) + '→' + sigs['y_gua'].astype(str)
    return sigs


def boot_ci(arr, baseline_mean, n_boot=1000, seed=42):
    if len(arr) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    alpha_boots = boots - baseline_mean
    return float(np.percentile(alpha_boots, 2.5)), float(np.percentile(alpha_boots, 97.5))


def loo_one(sig_pool, mask, label):
    n_drop = int(mask.sum())
    if n_drop < 30:
        return None
    bm = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[~mask, 'actual_ret'].values
    drop = sig_pool.loc[mask, 'actual_ret'].values
    if len(keep) < 30:
        return None
    lo, hi = boot_ci(keep, bm)
    return {'label': label, 'drop_n': n_drop, 'drop_mean': float(drop.mean()),
            'keep_n': len(keep), 'keep_mean': float(keep.mean()),
            'alpha': float(keep.mean() - bm), 'ci_lo': lo, 'ci_hi': hi}


def addone_one(sig_pool, mask, label, min_n=50):
    n = int(mask.sum())
    if n < min_n:
        return None
    bm = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[mask, 'actual_ret'].values
    lo, hi = boot_ci(keep, bm)
    return {'label': label, 'keep_n': n, 'keep_mean': float(keep.mean()),
            'alpha': float(keep.mean() - bm), 'ci_lo': lo, 'ci_hi': hi}


def verdict(c, mode='loo'):
    if c is None or c['ci_lo'] is None:
        return '○ 样本不足'
    if mode == 'loo':
        if c['ci_lo'] > 0:
            return '★ 显著该排'
        if c['ci_hi'] < 0:
            return '✗ 反向(该留)'
    else:
        if c['ci_lo'] > 0:
            return '★ 显著该留'
        if c['ci_hi'] < 0:
            return '✗ 反向(该排)'
    return '○ 灰区'


def main():
    sigs = load_data()
    bm = sigs['actual_ret'].mean()
    print(f'baseline (含变爻标记的 sig): {len(sigs)}, mean {bm:+.2f}%')

    cts = sigs['change_type'].value_counts()
    print(f'\n变爻类型: 总 {len(cts)} 种 (n>=30: {(cts >= 30).sum()})')

    # LOO + add-one
    loos, adds = [], []
    for ct in cts.index:
        if cts[ct] < 30:
            continue
        mask = (sigs['change_type'] == ct).values
        loos.append(loo_one(sigs, mask, ct))
        adds.append(addone_one(sigs, mask, ct))

    print('\n## LOO (排某 from→to 整桶)')
    print(f'  {"from→to":<14} {"drop_n":>6} {"drop%":>7} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>12}')
    print('  ' + '-' * 100)
    for r in sorted([r for r in loos if r], key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  排 {r['label']:<12} {r['drop_n']:>6} {r['drop_mean']:>+7.2f} "
              f"{r['keep_mean']:>+7.2f} {r['alpha']:>+7.2f} {ci:>16} {verdict(r,'loo'):>12}")

    print('\n## add-one (只留某 from→to)')
    print(f'  {"from→to":<14} {"keep_n":>6} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>12}')
    print('  ' + '-' * 90)
    for r in sorted([r for r in adds if r], key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  留 {r['label']:<12} {r['keep_n']:>6} {r['keep_mean']:>+7.2f} "
              f"{r['alpha']:>+7.2f} {ci:>16} {verdict(r,'addone'):>12}")

    # 双向 ★ 汇总
    by_loo = {r['label']: r for r in loos if r}
    by_add = {r['label']: r for r in adds if r}
    all_labels = sorted(set(by_loo.keys()) | set(by_add.keys()))

    print('\n## 双向汇总')
    print(f'  {"from→to":<14} {"LOO 判定":<14} {"add-one 判定":<14} {"双向自洽":>8} {"建议":<14}')
    print('  ' + '-' * 80)
    for ct in all_labels:
        loo_r = by_loo.get(ct)
        add_r = by_add.get(ct)
        loo_v = verdict(loo_r, 'loo')
        add_v = verdict(add_r, 'addone')

        # 双向自洽
        consistent_skip = (loo_v.startswith('★') and add_v.startswith('✗'))
        consistent_keep = (loo_v.startswith('✗') and add_v.startswith('★'))

        if consistent_skip:
            advice = '★★ 真该排'
        elif consistent_keep:
            advice = '★★ 真该留'
        elif loo_v.startswith('★') or add_v.startswith('✗'):
            advice = '○ 单向 该排?'
        elif loo_v.startswith('✗') or add_v.startswith('★'):
            advice = '○ 单向 该留?'
        else:
            advice = '○ 中性'

        print(f'  {ct:<14} {loo_v:<14} {add_v:<14} '
              f'{("✓" if (consistent_skip or consistent_keep) else " "):>8} {advice}')


if __name__ == '__main__':
    main()
