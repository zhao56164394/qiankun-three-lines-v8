# -*- coding: utf-8 -*-
"""坤 regime 同日选股 — 排雷+排名扫描

用 test157 dump 的 v4 / v5 事件因子表, 扫每个因子:
  排雷: 找 lift 显著为负的桶 (跨段稳定差) → 硬过滤
  排名: 看排序键 (asc / desc) 下 top-K 平均 ret 提升

候选因子:
  数值: pool_min_retail / pool_days / pool_min_mf / cur_retail / cur_mf / cur_trend
        mf_5d / ret_5d / td_5d / mf_30d_min / mf_30d_mean / ret_30d_min / ret_30d_mean
  卦象: mkt_d / mkt_m / stk_m / stk_y

排雷判据: 全样本 lift < -2% AND ≥3 个"段" (按入场年) lift < -1%
排名判据: top-1 / top-3 / top-5 比 baseline (随机) 高
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def numeric_buckets(s, n=4):
    """把数值列按分位数切成 n 桶"""
    qs = s.quantile([i/n for i in range(n+1)]).to_numpy().copy()
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    # 处理重复边界
    qs = np.unique(qs)
    if len(qs) < 2:
        return pd.Series(['q1'] * len(s), index=s.index)
    labels = [f'q{i+1}' for i in range(len(qs)-1)]
    return pd.cut(s, bins=qs, labels=labels, include_lowest=True)


def scan_avoid_numeric(df, col, baseline_ret, n_buckets=4):
    """对一个数值列, 4 分位扫排雷"""
    buckets = numeric_buckets(df[col], n_buckets)
    results = []
    df['__year'] = df['date'].str[:4]
    years = sorted(df['__year'].unique())
    for q in [f'q{i+1}' for i in range(n_buckets)]:
        sub = df[buckets == q]
        if len(sub) < 50: continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct'] > 0).mean() * 100
        lift = avg - baseline_ret
        # 跨年稳定性
        year_lifts = []
        for y in years:
            ys = sub[sub['__year'] == y]
            yb = df[df['__year'] == y]
            if len(ys) >= 20 and len(yb) >= 50:
                year_lifts.append(ys['ret_pct'].mean() - yb['ret_pct'].mean())
        n_neg = sum(1 for l in year_lifts if l < -1)
        n_pos = sum(1 for l in year_lifts if l > 1)
        results.append({
            'col': col, 'bucket': q, 'n': len(sub),
            'avg': avg, 'win': win, 'lift': lift,
            'n_neg_yr': n_neg, 'n_pos_yr': n_pos,
            'year_lifts': year_lifts,
        })
    return results


def scan_avoid_categorical(df, col, baseline_ret):
    """卦象类按值扫"""
    df['__year'] = df['date'].str[:4]
    years = sorted(df['__year'].unique())
    results = []
    for v, sub in df.groupby(col):
        if len(sub) < 100 or v == '': continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct'] > 0).mean() * 100
        lift = avg - baseline_ret
        year_lifts = []
        for y in years:
            ys = sub[sub['__year'] == y]
            yb = df[df['__year'] == y]
            if len(ys) >= 20 and len(yb) >= 50:
                year_lifts.append(ys['ret_pct'].mean() - yb['ret_pct'].mean())
        n_neg = sum(1 for l in year_lifts if l < -1)
        n_pos = sum(1 for l in year_lifts if l > 1)
        results.append({
            'col': col, 'value': v, 'n': len(sub),
            'avg': avg, 'win': win, 'lift': lift,
            'n_neg_yr': n_neg, 'n_pos_yr': n_pos,
            'year_lifts': year_lifts,
        })
    return results


def scan_rank_topk(df, col, asc, K, baseline_ret):
    """每天按 col (asc/desc) 排序, 取 top-K, 算这些事件的 avg ret"""
    sub = df.sort_values(['date', col, 'code'], ascending=[True, asc, True])
    sub = sub.groupby('date').head(K)
    if len(sub) == 0: return None
    return {
        'col': col, 'asc': asc, 'K': K, 'n': len(sub),
        'avg': sub['ret_pct'].mean(),
        'win': (sub['ret_pct'] > 0).mean() * 100,
        'lift': sub['ret_pct'].mean() - baseline_ret,
    }


def main():
    t0 = time.time()
    print('=== test158: 因子排雷+排名扫描 ===\n')

    out_dir = os.path.join(ROOT, 'data_layer/data/results')

    NUMERIC_COLS = ['pool_min_retail', 'pool_days', 'pool_min_mf',
                    'cur_retail', 'cur_mf', 'cur_trend',
                    'mf_5d', 'ret_5d', 'td_5d',
                    'mf_30d_min', 'mf_30d_mean', 'ret_30d_min', 'ret_30d_mean',
                    'legs']
    CAT_COLS = ['mkt_d', 'mkt_m', 'stk_m', 'stk_y']

    for tag in ['v4', 'v5']:
        df = pd.read_parquet(os.path.join(out_dir, f'kun_event_factors_{tag}.parquet'))
        df = df.dropna(subset=['ret_pct']).reset_index(drop=True)
        baseline = df['ret_pct'].mean()
        baseline_win = (df['ret_pct'] > 0).mean() * 100
        print(f'\n{"="*70}')
        print(f'  [{tag}] n={len(df):,}, baseline avg={baseline:+.2f}%, win={baseline_win:.1f}%')
        print(f'{"="*70}')

        # ===== 数值列 4 桶扫描 =====
        print(f'\n--- 数值列 4 分位 排雷扫描 ---')
        print(f'  {"factor":<18} {"bucket":<6} {"n":>5} {"avg":>8} {"lift":>8} {"win":>6} '
              f'{"年正":>4}{"年负":>4}  年 lift')
        all_num = []
        for col in NUMERIC_COLS:
            res = scan_avoid_numeric(df, col, baseline)
            all_num.extend(res)
        # 按 lift 排
        all_num_sorted = sorted(all_num, key=lambda r: r['lift'])
        # 看最差 (排雷候选) 和最好 (排名候选)
        print(f'\n  -- 最差 8 桶 (排雷候选) --')
        for r in all_num_sorted[:8]:
            yl_str = ' '.join(f'{l:+.1f}' for l in r['year_lifts'])
            print(f'  {r["col"]:<18} {r["bucket"]:<6} {r["n"]:>5} {r["avg"]:>+7.2f}% '
                  f'{r["lift"]:>+7.2f}% {r["win"]:>5.1f}% {r["n_pos_yr"]:>4}{r["n_neg_yr"]:>4}  {yl_str}')
        print(f'\n  -- 最好 8 桶 (排名候选) --')
        for r in all_num_sorted[-8:][::-1]:
            yl_str = ' '.join(f'{l:+.1f}' for l in r['year_lifts'])
            print(f'  {r["col"]:<18} {r["bucket"]:<6} {r["n"]:>5} {r["avg"]:>+7.2f}% '
                  f'{r["lift"]:>+7.2f}% {r["win"]:>5.1f}% {r["n_pos_yr"]:>4}{r["n_neg_yr"]:>4}  {yl_str}')

        # ===== 卦象列扫描 =====
        print(f'\n--- 卦象列 排雷扫描 ---')
        all_cat = []
        for col in CAT_COLS:
            res = scan_avoid_categorical(df, col, baseline)
            all_cat.extend(res)
        all_cat_sorted = sorted(all_cat, key=lambda r: r['lift'])
        print(f'\n  -- 最差 6 (排雷候选) --')
        for r in all_cat_sorted[:6]:
            yl_str = ' '.join(f'{l:+.1f}' for l in r['year_lifts'])
            print(f'  {r["col"]:<8} {r["value"]:<5} n={r["n"]:>5} avg {r["avg"]:>+7.2f}% '
                  f'lift {r["lift"]:>+7.2f}% win {r["win"]:>5.1f}% {r["n_pos_yr"]:>4}{r["n_neg_yr"]:>4}  {yl_str}')
        print(f'\n  -- 最好 6 (排名候选) --')
        for r in all_cat_sorted[-6:][::-1]:
            yl_str = ' '.join(f'{l:+.1f}' for l in r['year_lifts'])
            print(f'  {r["col"]:<8} {r["value"]:<5} n={r["n"]:>5} avg {r["avg"]:>+7.2f}% '
                  f'lift {r["lift"]:>+7.2f}% win {r["win"]:>5.1f}% {r["n_pos_yr"]:>4}{r["n_neg_yr"]:>4}  {yl_str}')

        # ===== 排名 top-K =====
        print(f'\n--- 同日 top-K 排名扫描 ---')
        # 先看每日信号数量
        per_day = df.groupby('date').size()
        print(f'  每日信号: 平均 {per_day.mean():.1f}, 中位 {per_day.median():.0f}, '
              f'p75 {per_day.quantile(0.75):.0f}, max {per_day.max()}')
        print(f'  信号天数: {len(per_day)}, 多信号天 (>=2): {(per_day>=2).sum()}, '
              f'(>=5): {(per_day>=5).sum()}')

        print(f'\n  {"factor":<18} {"asc":<4} {"K":>3} {"n":>5} {"avg":>8} {"lift":>8} {"win":>6}')
        rank_results = []
        for col in NUMERIC_COLS:
            for asc in [True, False]:
                for K in [1, 3, 5]:
                    r = scan_rank_topk(df, col, asc, K, baseline)
                    if r: rank_results.append(r)
        rank_sorted = sorted(rank_results, key=lambda r: -r['lift'])
        for r in rank_sorted[:15]:
            print(f'  {r["col"]:<18} {str(r["asc"]):<5} {r["K"]:>3} {r["n"]:>5} '
                  f'{r["avg"]:>+7.2f}% {r["lift"]:>+7.2f}% {r["win"]:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
