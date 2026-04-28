# -*- coding: utf-8 -*-
"""Phase 3 坤桶研究 Step 2 — 因子区分力分析.

输入: 37648 个坤期上穿 11 信号 (kun_naked_t11_t89.json) + 标签 (success/fail)
逻辑: 对每个候选因子, 看 success 组 vs fail 组的分布, 找出"在哪些档位下成功率显著偏离基线 29.1%"
输出: 8 张因子对比表 + 区分力排序汇总

因子库 (8 个):
  类别: 大盘 d_gua, 大盘 m_gua, 个股 d_gua, 个股 m_gua, 个股 y_gua
  连续: 个股 retail, 个股 main_force, 大盘 zz1000 trend
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE = None  # 信号总成功率, 自动计算


def load_signals():
    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test4',
                     'kun_naked_t11_t89.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    df = pd.DataFrame(d['results'])
    df['entry_date'] = df['entry_date'].astype(str)
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def join_factors(sigs):
    """逐个数据源 join 信号当日的因子值."""
    # 1. 大盘卦
    mkt = pd.read_parquet(
        os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                     'multi_scale_gua_daily.parquet'),
        columns=['date', 'd_gua', 'm_gua', 'd_trend'])
    mkt['date'] = mkt['date'].astype(str)
    mkt['d_gua'] = mkt['d_gua'].astype(str).str.zfill(3)
    mkt['m_gua'] = mkt['m_gua'].astype(str).str.zfill(3)
    mkt = mkt.rename(columns={'d_gua': 'mkt_d_gua', 'm_gua': 'mkt_m_gua',
                              'd_trend': 'mkt_d_trend'})

    # 2. 个股卦 (逐股逐日)
    stk = pd.read_parquet(
        os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                     'stock_multi_scale_gua_daily.parquet'),
        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    stk['date'] = stk['date'].astype(str)
    stk['code'] = stk['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        stk[c] = stk[c].astype(str).str.zfill(3)
    stk = stk.rename(columns={'d_gua': 'stk_d_gua', 'm_gua': 'stk_m_gua',
                              'y_gua': 'stk_y_gua'})

    # 3. zz1000 trend (大盘)
    zz = pd.read_parquet(
        os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.parquet'),
        columns=['date', 'trend'])
    zz['date'] = pd.to_datetime(zz['date'], format='mixed').dt.strftime('%Y-%m-%d')
    zz = zz.rename(columns={'trend': 'zz1000_trend'})

    # 4. stocks: retail, main_force (信号当日)
    sk = pd.read_parquet(
        os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet'),
        columns=['code', 'date', 'retail', 'main_force'])
    sk['code'] = sk['code'].astype(str).str.zfill(6)
    sk['date'] = sk['date'].astype(str).str[:10]
    sk = sk.rename(columns={'retail': 'stk_retail', 'main_force': 'stk_mf'})

    # join
    df = sigs.copy()
    df = df.merge(mkt, left_on='entry_date', right_on='date', how='left').drop(columns='date')
    df = df.merge(stk, left_on=['entry_date', 'code'], right_on=['date', 'code'], how='left').drop(columns='date')
    df = df.merge(zz, left_on='entry_date', right_on='date', how='left').drop(columns='date')
    df = df.merge(sk, left_on=['entry_date', 'code'], right_on=['date', 'code'], how='left').drop(columns='date')
    return df


GUA_NAME = {'000':'坤(深熊)','001':'艮(吸筹)','010':'坎(乏力)','011':'巽(底爆)',
            '100':'震(出货)','101':'离(护盘)','110':'兑(末减)','111':'乾(疯牛)'}


def cat_table(df, factor, name):
    """类别因子分布对比"""
    print(f'\n{"=" * 80}')
    print(f'因子: {name} ({factor})')
    print('=' * 80)
    g = df.groupby(factor)['success'].agg(['count', 'sum'])
    g['rate'] = g['sum'] / g['count'] * 100
    g['bias'] = g['rate'] - BASELINE
    g = g.sort_values('bias', ascending=False)

    print(f'  {"值":<6} {"卦名":<10} {"n_total":>8} {"n_succ":>7} '
          f'{"成功率":>8} {"偏离基线":>10}')
    print('  ' + '-' * 58)
    for v, row in g.iterrows():
        n = int(row['count'])
        ns = int(row['sum'])
        rate = row['rate']
        bias = row['bias']
        sign = '★ 强提升' if bias > 5 else ('✗ 强抑制' if bias < -5 else
                ('+ 弱提升' if bias > 1.5 else ('- 弱抑制' if bias < -1.5 else '○ 中性')))
        nm = GUA_NAME.get(str(v), '') if 'gua' in factor else ''
        print(f'  {v!s:<6} {nm:<10} {n:>8d} {ns:>7d} '
              f'{rate:>7.1f}% {bias:>+8.1f}pp  {sign}')
    return g


def cont_table(df, factor, name, bins, labels):
    """连续因子分档对比"""
    print(f'\n{"=" * 80}')
    print(f'因子: {name} ({factor})')
    print('=' * 80)
    sub = df[df[factor].notna()].copy()
    sub['bucket'] = pd.cut(sub[factor], bins=bins, labels=labels, include_lowest=True)
    g = sub.groupby('bucket', observed=True)['success'].agg(['count', 'sum'])
    g['rate'] = g['sum'] / g['count'] * 100
    g['bias'] = g['rate'] - BASELINE

    print(f'  {"档位":<20} {"n_total":>8} {"n_succ":>7} '
          f'{"成功率":>8} {"偏离基线":>10}')
    print('  ' + '-' * 60)
    for v, row in g.iterrows():
        n = int(row['count']) if row['count'] > 0 else 0
        if n == 0:
            continue
        ns = int(row['sum'])
        rate = row['rate']
        bias = row['bias']
        sign = '★ 强提升' if bias > 5 else ('✗ 强抑制' if bias < -5 else
                ('+ 弱提升' if bias > 1.5 else ('- 弱抑制' if bias < -1.5 else '○ 中性')))
        print(f'  {v!s:<20} {n:>8d} {ns:>7d} {rate:>7.1f}% {bias:>+8.1f}pp  {sign}')

    print(f'\n  [分布对比] success n={int(sub["success"].sum())}, fail n={int((~sub["success"].astype(bool)).sum())}')
    print(f'  {"":>10} {"mean":>8} {"p10":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p90":>7}')
    for label, mask in [('成功组', sub['success']), ('失败组', ~sub['success'].astype(bool))]:
        a = sub.loc[mask, factor].values
        if len(a) == 0:
            continue
        qs = np.nanquantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
        print(f'  {label:>10} {np.nanmean(a):>+8.2f} '
              f'{qs[0]:>+7.2f} {qs[1]:>+7.2f} {qs[2]:>+7.2f} {qs[3]:>+7.2f} {qs[4]:>+7.2f}')
    return g


def main():
    global BASELINE
    sigs = load_signals()
    print(f'信号总数: {len(sigs)}, 成功: {int(sigs["success"].sum())}')
    BASELINE = sigs['success'].mean() * 100
    print(f'基线成功率: {BASELINE:.2f}%')

    df = join_factors(sigs)
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        print(f'\n[merge 缺失] {miss.to_dict()}')

    print('\n' + '#' * 80)
    print('# 类别因子: 大盘 / 个股 卦')
    print('#' * 80)
    cat_table(df, 'mkt_d_gua', '大盘 d_gua (信号当日)')
    cat_table(df, 'mkt_m_gua', '大盘 m_gua (信号当日)')
    cat_table(df, 'stk_d_gua', '个股 d_gua (信号当日)')
    cat_table(df, 'stk_m_gua', '个股 m_gua (信号当日)')
    cat_table(df, 'stk_y_gua', '个股 y_gua (信号当日)')

    print('\n' + '#' * 80)
    print('# 连续因子: retail / main_force / zz1000 trend')
    print('#' * 80)
    # 散户线 retail (个股池深)
    cont_table(df, 'stk_retail', '个股 retail 散户线 (=池深)',
               bins=[-np.inf, -400, -300, -200, -100, 0, 100, np.inf],
               labels=['(-inf,-400] 极深', '(-400,-300] 深', '(-300,-200] 中深',
                       '(-200,-100] 中浅', '(-100,0] 浅', '(0,100] 正零', '(100,+inf) 强正'])
    # 个股主力线
    cont_table(df, 'stk_mf', '个股 main_force 主力线',
               bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
               labels=['(-inf,-200]','(-200,-100]','(-100,-50]','(-50,0]',
                       '(0,50]','(50,100]','(100,200]','(200,+inf)'])
    # 大盘 zz1000 trend
    cont_table(df, 'zz1000_trend', '大盘 zz1000 趋势线值',
               bins=[-np.inf, 5, 15, 25, 35, 50, np.inf],
               labels=['<=5 极低','(5,15]','(15,25]','(25,35]','(35,50]','>50'])

    # 区分力汇总: 各因子最大|偏离|
    print('\n' + '#' * 80)
    print('# 区分力汇总 (按"最大偏离档"排序)')
    print('#' * 80)
    print(f'{"因子":<40} {"最强档":<25} {"该档成功率":>10} {"偏离":>8} {"该档样本":>10}')
    print('-' * 100)
    factors_summary = []
    for factor, label, is_cat, bins, lbls in [
        ('mkt_d_gua', '大盘 d_gua', True, None, None),
        ('mkt_m_gua', '大盘 m_gua', True, None, None),
        ('stk_d_gua', '个股 d_gua', True, None, None),
        ('stk_m_gua', '个股 m_gua', True, None, None),
        ('stk_y_gua', '个股 y_gua', True, None, None),
        ('stk_retail', '个股 retail', False,
         [-np.inf, -400, -300, -200, -100, 0, 100, np.inf],
         ['(-inf,-400]', '(-400,-300]', '(-300,-200]', '(-200,-100]', '(-100,0]', '(0,100]', '(100,+inf)']),
        ('stk_mf', '个股 main_force', False,
         [-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
         ['(-inf,-200]','(-200,-100]','(-100,-50]','(-50,0]','(0,50]','(50,100]','(100,200]','(200,+inf)']),
        ('zz1000_trend', '大盘 zz1000 trend', False,
         [-np.inf, 5, 15, 25, 35, 50, np.inf],
         ['<=5','(5,15]','(15,25]','(25,35]','(35,50]','>50']),
    ]:
        sub = df[df[factor].notna()].copy()
        if not is_cat:
            sub['bucket'] = pd.cut(sub[factor], bins=bins, labels=lbls, include_lowest=True)
            g = sub.groupby('bucket', observed=True)['success'].agg(['count', 'sum'])
        else:
            g = sub.groupby(factor)['success'].agg(['count', 'sum'])
        g['rate'] = g['sum'] / g['count'] * 100
        g['bias'] = g['rate'] - BASELINE
        # 仅看样本量 ≥ 200 的档
        g_valid = g[g['count'] >= 200]
        if len(g_valid) == 0:
            continue
        # 找"最大|偏离|"
        idx_max = g_valid['bias'].abs().idxmax()
        row = g_valid.loc[idx_max]
        bias = row['bias']
        n = int(row['count'])
        rate = row['rate']
        sign = '★' if bias > 5 else ('✗' if bias < -5 else ('+' if bias > 0 else '-'))
        factors_summary.append({
            'factor': label, 'best_bucket': str(idx_max),
            'rate': rate, 'bias': bias, 'n': n
        })
        print(f'{label:<40} {str(idx_max):<25} {rate:>9.1f}% {bias:>+7.1f}pp {n:>10d}  {sign}')

    print(f'\n基线成功率: {BASELINE:.2f}%')
    print('\n说明: 偏离 ≥ ±5pp 且样本 ≥ 200 的档位 = 有用过滤候选')

    # 落地
    out_path = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test4',
                            'kun_factor_discrimination.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline_pct': BASELINE,
            'n_signals': len(sigs),
            'factor_summary': factors_summary,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n落地: {out_path}')


if __name__ == '__main__':
    main()
