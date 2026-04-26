# -*- coding: utf-8 -*-
"""离卦 (101) 第二轮深挖 · 三才底座视角

架构前提 (memory/feedback_gua_architecture.md):
  - 三才主路径: 天(tian)/地(di)/人(ren)
  - 离卦语境下 di_gua 已天然坍缩 (73% 在 000) → 不做 di 黑白名单
  - 重心放在 ren_gua 上, m_gua 仅作"环境观察"不进黑白名单

补齐前一轮的明显短板:
  1. 样本量置信区间 (bootstrap 95% CI) — 避免在 n<10 的桶下结论
  2. 时间稳定性 — 拆 2015-2018 / 2019-2021 / 2022-2025 三期
  3. ren_gua × 池深 / × 池天 二维 — 看 ren 的优势是否依赖入池条件
  4. ren_gua × m_gua 二维 — ren 在不同月卦环境下是否一致 (环境层视角)
  5. 持仓收益分布形态 — 看尾部 (P10/P25/P75/P90)
"""
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
MULTI = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def load_data():
    with open(NAKED, encoding='utf-8') as f:
        d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    trd = pd.DataFrame(d['trade_log'])
    sig = sig[~sig['is_skip']].copy()
    for c in ('di_gua', 'ren_gua', 'tian_gua'):
        sig[c] = sig[c].astype(str).str.zfill(3)
        if c in trd.columns:
            trd[c] = trd[c].astype(str).str.zfill(3)
    if 'gua' in trd.columns:
        trd['gua'] = trd['gua'].astype(str).str.zfill(3)

    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'m_gua': str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
    m_map = dict(zip(ms['date'], ms['m_gua'].fillna('').astype(str).str.zfill(3)))
    sig['m_gua'] = sig['signal_date'].astype(str).map(m_map).fillna('').astype(str)

    li_s = sig[sig['tian_gua'] == '101'].copy()
    li_s['signal_year'] = pd.to_datetime(li_s['signal_date']).dt.year

    li_t = trd[trd.get('gua', pd.Series(dtype=str)) == '101'].copy() if 'gua' in trd.columns else pd.DataFrame()
    return li_s, li_t


# ============= bootstrap CI =============
def bootstrap_ci(arr, n_iter=2000, alpha=0.05, seed=42):
    """返回 (mean, lo, hi)。n<5 返回 NaN。"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return (np.mean(arr) if len(arr) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_iter):
        means.append(arr[rng.integers(0, len(arr), len(arr))].mean())
    means = np.array(means)
    return (arr.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2))


# ============= 1. ren_gua 强度 + CI =============
def section_ren_with_ci(li_s):
    print('\n' + '=' * 100)
    print('  1. ren_gua 8 卦 · 全量收益 + 95% bootstrap CI')
    print('  (CI 不重叠 0 才算"显著"; n<10 不下结论, n<5 不算 CI)')
    print('=' * 100)
    print(f'  {"ren_gua":<10}  {"n":>5} {"均收":>8} {"95% CI":>22} {"中位":>8} {"胜率":>6} {"P25":>8} {"P75":>8}')
    print('  ' + '-' * 96)

    overall = li_s['actual_ret']
    o_mean, o_lo, o_hi = bootstrap_ci(overall.values)
    print(f'  {"全部":<10}  {len(overall):>5} {o_mean:+7.2f}% [{o_lo:+5.2f}, {o_hi:+5.2f}] '
          f'{overall.median():+7.2f}% {(overall>0).mean()*100:5.1f}% '
          f'{overall.quantile(.25):+7.2f}% {overall.quantile(.75):+7.2f}%')
    print()

    rows = []
    for g in GUA_ORDER:
        sub = li_s[li_s['ren_gua'] == g]['actual_ret']
        if len(sub) == 0:
            print(f'  {g} {GUA_NAME[g]:<6} {0:>5}     -                     -      -        -        -')
            continue
        m, lo, hi = bootstrap_ci(sub.values)
        sig_mark = ''
        if not np.isnan(lo):
            if lo > 0: sig_mark = '  ★ 全正区间'
            elif hi < 0: sig_mark = '  ✗ 全负区间'
        ci_str = f'[{lo:+5.2f}, {hi:+5.2f}]' if not np.isnan(lo) else '   (n太少)'
        rows.append((g, len(sub), m, lo, hi))
        print(f'  {g} {GUA_NAME[g]:<6} {len(sub):>5} {m:+7.2f}% {ci_str:>22} '
              f'{sub.median():+7.2f}% {(sub>0).mean()*100:5.1f}% '
              f'{sub.quantile(.25):+7.2f}% {sub.quantile(.75):+7.2f}%{sig_mark}')


# ============= 2. ren_gua × 时间 (年度切片) =============
def section_ren_time_stability(li_s):
    print('\n' + '=' * 100)
    print('  2. ren_gua 时间稳定性 · 三期切片 (2015-18 / 2019-21 / 2022-25)')
    print('  (符号一致 + 各期 n>=8 才算"稳定")')
    print('=' * 100)
    periods = [
        ('2015-18', (2015, 2018)),
        ('2019-21', (2019, 2021)),
        ('2022-25', (2022, 2025)),
    ]
    print(f'  {"ren_gua":<10}', end='')
    for name, _ in periods:
        print(f' | {name+" n":>6} {"均":>7} {"胜":>5}', end='')
    print(f' | {"判定":<14}')
    print('  ' + '-' * 96)

    for g in GUA_ORDER:
        sub_g = li_s[li_s['ren_gua'] == g]
        if len(sub_g) == 0: continue
        cells = []
        signs = []
        small_warn = False
        for name, (y0, y1) in periods:
            sp = sub_g[(sub_g['signal_year'] >= y0) & (sub_g['signal_year'] <= y1)]
            n = len(sp)
            if n == 0:
                cells.append(f' | {0:>6} {"-":>7} {"-":>5}'); signs.append(0); continue
            mean = sp['actual_ret'].mean(); win = (sp['actual_ret'] > 0).mean() * 100
            cells.append(f' | {n:>6} {mean:+6.2f}% {win:4.0f}%')
            signs.append(1 if mean > 0 else (-1 if mean < 0 else 0))
            if n < 8: small_warn = True
        # 判定
        pos_periods = sum(1 for s in signs if s > 0)
        neg_periods = sum(1 for s in signs if s < 0)
        if small_warn:
            verdict = '样本不足'
        elif pos_periods == 3:
            verdict = '★ 全正稳定'
        elif neg_periods == 3:
            verdict = '✗ 全负稳定'
        elif pos_periods >= 2:
            verdict = '○ 多数正'
        elif neg_periods >= 2:
            verdict = '○ 多数负'
        else:
            verdict = '? 不稳'
        print(f'  {g} {GUA_NAME[g]:<6}', end='')
        for c in cells: print(c, end='')
        print(f' | {verdict:<14}')


# ============= 3. ren_gua × 池深 / 池天 =============
def section_ren_pool_cross(li_s):
    print('\n' + '=' * 100)
    print('  3. ren_gua × 池深 (pool_retail) — 入池条件下 ren 优势是否一致')
    print('=' * 100)
    depth_buckets = [(-1e9, -500, '≤-500 极深'),
                     (-500, -350, '-500~-350 深'),
                     (-350, -300, '-350~-300 中'),
                     (-300, -250, '-300~-250 浅')]
    li_s = li_s.copy()
    li_s['depth_b'] = pd.NA
    for lo, hi, label in depth_buckets:
        mask = (li_s['pool_retail'] > lo) & (li_s['pool_retail'] <= hi)
        li_s.loc[mask, 'depth_b'] = label
    depth_order = [b[2] for b in depth_buckets]

    cnt = li_s.pivot_table(index='ren_gua', columns='depth_b', values='actual_ret',
                           aggfunc='count', fill_value=0).reindex(GUA_ORDER, fill_value=0)
    mean = li_s.pivot_table(index='ren_gua', columns='depth_b', values='actual_ret',
                            aggfunc='mean').reindex(GUA_ORDER)
    cnt = cnt.reindex(columns=depth_order, fill_value=0)
    mean = mean.reindex(columns=depth_order)

    print(f'  {"ren_gua":<10} ' + ' '.join(f'{c[:14]:<14}' for c in depth_order))
    for g in GUA_ORDER:
        cells = []
        for c in depth_order:
            n = int(cnt.loc[g, c]); m = mean.loc[g, c]
            if n < 5 or pd.isna(m):
                cells.append(f'  -    (n={n:>2})    ')
            else:
                cells.append(f'{m:+6.2f}% (n={n:>2})  ')
        print(f'  {g} {GUA_NAME[g]:<6} ' + ' '.join(cells))

    print('\n' + '=' * 100)
    print('  3b. ren_gua × 池天 (pool_days) — 入池后等待天数对 ren 分组的影响')
    print('=' * 100)
    days_buckets = [(0, 3, '0-3 天'), (4, 7, '4-7 天'),
                    (8, 15, '8-15 天'), (16, 30, '16-30 天'), (31, 9999, '31+ 天')]
    li_s['days_b'] = pd.NA
    for lo, hi, label in days_buckets:
        mask = (li_s['pool_days'] >= lo) & (li_s['pool_days'] <= hi)
        li_s.loc[mask, 'days_b'] = label
    days_order = [b[2] for b in days_buckets]

    cnt2 = li_s.pivot_table(index='ren_gua', columns='days_b', values='actual_ret',
                            aggfunc='count', fill_value=0).reindex(GUA_ORDER, fill_value=0)
    mean2 = li_s.pivot_table(index='ren_gua', columns='days_b', values='actual_ret',
                             aggfunc='mean').reindex(GUA_ORDER)
    cnt2 = cnt2.reindex(columns=days_order, fill_value=0)
    mean2 = mean2.reindex(columns=days_order)

    print(f'  {"ren_gua":<10} ' + ' '.join(f'{c:<13}' for c in days_order))
    for g in GUA_ORDER:
        cells = []
        for c in days_order:
            n = int(cnt2.loc[g, c]); m = mean2.loc[g, c]
            if n < 5 or pd.isna(m):
                cells.append(f'  - (n={n:>2})  ')
            else:
                cells.append(f'{m:+6.2f}% n={n:<2}')
        print(f'  {g} {GUA_NAME[g]:<6} ' + ' '.join(cells))


# ============= 4. ren_gua × m_gua (环境观察) =============
def section_ren_env(li_s):
    print('\n' + '=' * 100)
    print('  4. ren_gua × m_gua — ren 优势是否跨大盘环境一致 (环境层视角, 不做黑白名单)')
    print('=' * 100)
    cnt = li_s.pivot_table(index='ren_gua', columns='m_gua', values='actual_ret',
                           aggfunc='count', fill_value=0).reindex(GUA_ORDER, fill_value=0)
    mean = li_s.pivot_table(index='ren_gua', columns='m_gua', values='actual_ret',
                            aggfunc='mean').reindex(GUA_ORDER)
    cnt = cnt.reindex(columns=GUA_ORDER, fill_value=0)
    mean = mean.reindex(columns=GUA_ORDER)

    head = 'ren\\m'
    print(f'  {head:<10} ' + ' '.join(f'{g}{GUA_NAME[g]:<3}' for g in GUA_ORDER))
    for g in GUA_ORDER:
        cells = []
        for cg in GUA_ORDER:
            n = int(cnt.loc[g, cg]); m = mean.loc[g, cg]
            if n < 5 or pd.isna(m):
                cells.append('   -  ')
            else:
                cells.append(f'{m:+5.1f}')
        print(f'  {g} {GUA_NAME[g]:<6} ' + ' '.join(cells))
    print('\n  [count]')
    for g in GUA_ORDER:
        print(f'  {g} {GUA_NAME[g]:<6} ' + ' '.join(f'{int(cnt.loc[g, cg]):>5}' for cg in GUA_ORDER))


# ============= 5. ren_gua 持仓分布 (尾部形态) =============
def section_ren_distribution(li_s):
    print('\n' + '=' * 100)
    print('  5. ren_gua 收益分布形态 — 是否有"右长尾"驱动均值, 还是"分布稳健"')
    print('=' * 100)
    print(f'  {"ren_gua":<10} {"n":>5} {"P10":>8} {"P25":>8} {"中位":>8} {"P75":>8} {"P90":>8} {"max":>8} {"min":>8} {"均":>8} {"std":>7}')
    print('  ' + '-' * 100)
    for g in GUA_ORDER:
        sub = li_s[li_s['ren_gua'] == g]['actual_ret']
        if len(sub) < 5:
            print(f'  {g} {GUA_NAME[g]:<6} {len(sub):>5} (n太少)')
            continue
        print(f'  {g} {GUA_NAME[g]:<6} {len(sub):>5} '
              f'{sub.quantile(.10):+7.2f}% {sub.quantile(.25):+7.2f}% {sub.median():+7.2f}% '
              f'{sub.quantile(.75):+7.2f}% {sub.quantile(.90):+7.2f}% '
              f'{sub.max():+7.2f}% {sub.min():+7.2f}% {sub.mean():+7.2f}% {sub.std():6.2f}')


def main():
    print('\n  离卦 (101) 第二轮深挖 — 三才底座视角')
    print('  原则: di_gua 在离卦坍缩 → 不用; m_gua 是环境层 → 不进黑白名单; 重心 ren_gua')
    li_s, li_t = load_data()
    print(f'  全量信号 n={len(li_s)}  日期 {li_s["signal_date"].min()} ~ {li_s["signal_date"].max()}')
    print(f'  整体 actual_ret 均 {li_s["actual_ret"].mean():+.2f}%  胜率 {(li_s["actual_ret"]>0).mean()*100:.1f}%')

    section_ren_with_ci(li_s)
    section_ren_time_stability(li_s)
    section_ren_pool_cross(li_s)
    section_ren_env(li_s)
    section_ren_distribution(li_s)

    print('\n  ' + '=' * 60)
    print('  分析完成. 接下来交给人类: 看哪些 ren_gua 桶 (a) CI 显著 (b) 跨期稳 (c) 池深/池天/m_gua 不挑场景')


if __name__ == '__main__':
    main()
