# -*- coding: utf-8 -*-
"""把 8 卦合并成 4 季 (熊 / 转牛 / 牛 / 转熊), 跑 LOO + add-one 消融

合并方案 (按 y_gua 命名的 regime 含义):
  熊_探底  = 000 (深熊)
  转牛     = 001 (底吸) + 010 (反弹乏力) + 011 (底爆发)
  牛_主升  = 111 (主升)
  转熊     = 100 (出货) + 101 (高位护盘) + 110 (牛末减仓)

输出: 4 季 sig 整体 + 4 季桶级 LOO/add-one + 各季桶内单维池深/池天 LOO/add-one
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

SEASON_MAP = {
    '000': '熊_探底',
    '001': '转牛', '010': '转牛', '011': '转牛',
    '111': '牛_主升',
    '100': '转熊', '101': '转熊', '110': '转熊',
}
SEASON_ORDER = ['熊_探底', '转牛', '牛_主升', '转熊']


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
    sigs['season'] = sigs['y_gua'].map(SEASON_MAP)

    sigs['depth_b'] = pd.cut(sigs['pool_retail'], bins=DEPTH_BINS,
                             labels=DEPTH_LABELS, include_lowest=True)
    sigs['days_b'] = pd.cut(sigs['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)
    return sigs


def boot_alpha_ci(keep_arr, baseline_mean, n_boot=1000, seed=42):
    if len(keep_arr) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    boots = rng.choice(keep_arr, size=(n_boot, len(keep_arr)), replace=True).mean(axis=1)
    alpha_boots = boots - baseline_mean
    return float(np.percentile(alpha_boots, 2.5)), float(np.percentile(alpha_boots, 97.5))


def loo_one(sig_pool, mask, label):
    n_drop = int(mask.sum())
    if n_drop < 5:
        return None
    bm = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[~mask, 'actual_ret'].values
    drop = sig_pool.loc[mask, 'actual_ret'].values
    if len(keep) < 30:
        return None
    lo, hi = boot_alpha_ci(keep, bm)
    return {'label': label, 'drop_n': n_drop, 'drop_mean': float(drop.mean()),
            'keep_n': len(keep), 'keep_mean': float(keep.mean()),
            'alpha': float(keep.mean() - bm), 'ci_lo': lo, 'ci_hi': hi}


def addone_one(sig_pool, mask, label):
    n_keep = int(mask.sum())
    if n_keep < 30:
        return None
    bm = sig_pool['actual_ret'].mean()
    keep = sig_pool.loc[mask, 'actual_ret'].values
    lo, hi = boot_alpha_ci(keep, bm)
    return {'label': label, 'keep_n': n_keep,
            'keep_mean': float(keep.mean()),
            'alpha': float(keep.mean() - bm), 'ci_lo': lo, 'ci_hi': hi}


def verdict_loo(c):
    if c['ci_lo'] is None: return '○ 样本不足'
    if c['ci_lo'] > 0: return '★ 显著该排'
    if c['ci_hi'] < 0: return '✗ 反向(该留)'
    return '○ 灰区'


def verdict_addone(c):
    if c['ci_lo'] is None: return '○ 样本不足'
    if c['ci_lo'] > 0: return '★ 显著该留'
    if c['ci_hi'] < 0: return '✗ 反向(该排)'
    return '○ 灰区'


def print_loo_table(results, title):
    print(f'\n### LOO {title}')
    valid = [r for r in results if r]
    if not valid:
        print('  (无有效候选)')
        return
    print(f'  {"候选":<24} {"drop_n":>6} {"drop%":>7} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>10}')
    print('  ' + '-' * 95)
    for r in sorted(valid, key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  排 {r['label']:<22} {r['drop_n']:>6} {r['drop_mean']:>+7.2f} "
              f"{r['keep_mean']:>+7.2f} {r['alpha']:>+7.2f} {ci:>16} {verdict_loo(r):>10}")


def print_addone_table(results, title):
    print(f'\n### add-one {title}')
    valid = [r for r in results if r]
    if not valid:
        print('  (无有效候选)')
        return
    print(f'  {"候选":<24} {"keep_n":>6} {"keep%":>7} {"alpha":>7} {"95%CI":>16} {"判定":>10}')
    print('  ' + '-' * 85)
    for r in sorted(valid, key=lambda x: -x['alpha']):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  留 {r['label']:<22} {r['keep_n']:>6} {r['keep_mean']:>+7.2f} "
              f"{r['alpha']:>+7.2f} {ci:>16} {verdict_addone(r):>10}")


def analyze_pool(s, pool_label):
    bm = s['actual_ret'].mean()
    n = len(s)
    print(f'\n# {pool_label} (n={n}, baseline_mean={bm:+.2f}%)')
    if n < 50:
        print('  样本太小, 跳过')
        return

    loo_d, add_d = [], []
    for d in DEPTH_LABELS:
        mask = (s['depth_b'] == d).values
        loo_d.append(loo_one(s, mask, f'[{d}] 整行'))
        add_d.append(addone_one(s, mask, f'[{d}] 整行'))
    print_loo_table(loo_d, '— 单维池深')
    print_addone_table(add_d, '— 单维池深')

    loo_t, add_t = [], []
    for t in DAYS_LABELS:
        mask = (s['days_b'] == t).values
        loo_t.append(loo_one(s, mask, f'[{t}] 整列'))
        add_t.append(addone_one(s, mask, f'[{t}] 整列'))
    print_loo_table(loo_t, '— 单维池天')
    print_addone_table(add_t, '— 单维池天')


def main():
    sigs = load_data()
    bm = sigs['actual_ret'].mean()
    n = len(sigs)
    print(f'baseline: sig {n}, mean {bm:+.2f}%')

    # 4 季样本分布
    print('\n=== 4 季 sig 分布 ===')
    print(f'{"季":<10} {"n":>6} {"mean%":>7} {"win%":>6}  {"含 8 卦":>20}')
    print('-' * 65)
    yg2season = SEASON_MAP
    season2yg = {}
    for y, s in yg2season.items():
        season2yg.setdefault(s, []).append(y)
    for s in SEASON_ORDER:
        sub = sigs[sigs['season'] == s]
        win = (sub['actual_ret'] > 0).mean() * 100
        ygs = ', '.join(season2yg[s])
        print(f'{s:<10} {len(sub):>6} {sub["actual_ret"].mean():>+7.2f} {win:>6.1f}  ({ygs})')

    # 4 季桶级 LOO + add-one
    print('\n\n' + '=' * 100)
    print('# 4 季桶级 (全量基线)')
    print('=' * 100)
    season_loo, season_add = [], []
    for s in SEASON_ORDER:
        mask = (sigs['season'] == s).values
        if mask.sum() < 30:
            continue
        season_loo.append(loo_one(sigs, mask, f'{s} 整季'))
        season_add.append(addone_one(sigs, mask, f'{s} 整季'))
    print_loo_table(season_loo, '— 4 季桶级')
    print_addone_table(season_add, '— 4 季桶级')

    # 各季桶内单维分析
    for s in SEASON_ORDER:
        sub = sigs[sigs['season'] == s].copy()
        if len(sub) < 100:
            print(f'\n## 季 {s} (n={len(sub)}) 样本太小, 跳过桶内分析')
            continue
        print('\n\n' + '=' * 100)
        print(f'# 季 {s} 桶内 (单维池深/池天)')
        print('=' * 100)
        analyze_pool(sub, f'季 {s}')


if __name__ == '__main__':
    main()
