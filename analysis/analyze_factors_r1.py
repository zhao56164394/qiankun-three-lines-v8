# -*- coding: utf-8 -*-
"""四因子对收益的影响 · 第一轮全量扫描

数据源: data_layer/data/backtest_8gua_naked_result.json (裸跑 signal_detail, 14130 条)
辅助源: data_layer/data/foundation/multi_scale_gua_daily.csv (大盘 d/m/y 卦, 按日对齐)

因子:
  1. 大盘年卦 y_gua (月线构造, 宏观趋势)
  2. 大盘月卦 m_gua (周线构造, 中周期)
  3. 个股地卦 di_gua (象卦, 个股自身)
  4. 个股人卦 ren_gua (daily_bagua_sequence, 个股资金氛围)

每个因子独立分桶 (8 个卦码: 000~111), 统计 actual_ret:
  - n, mean, median, winrate, sum

视角:
  (A) 全量 — 裸跑综合所有 8 卦的 signal_detail
  (B) 分治后 — 按 tian_gua(=d_gua) 分组, 再看因子在每个分治卦内的分布 (交叉)

输出: 每个因子一张一维表 + 一张 tian_gua × 因子 二维热图 (均收益)
"""
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED_RESULT = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
MULTI_SCALE_PATH = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def load_signals():
    with open(NAKED_RESULT, encoding='utf-8') as f:
        d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    # 清洗 gua 列
    for col in ('tian_gua', 'di_gua', 'ren_gua'):
        sig[col] = sig[col].astype(str).str.zfill(3)
    sig = sig[~sig['is_skip']].copy()  # 剔除空仓卦被跳过的
    return sig


def load_market_scale():
    df = pd.read_csv(MULTI_SCALE_PATH, encoding='utf-8-sig',
                     dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    for col in ('d_gua', 'm_gua', 'y_gua'):
        df[col] = df[col].str.zfill(3)
    return df[['date', 'd_gua', 'm_gua', 'y_gua']]


def enrich_signals(sig, scale):
    sig = sig.merge(scale, left_on='signal_date', right_on='date', how='left')
    # d_gua 与 tian_gua 应一致, 做个 sanity
    mismatch = (sig['d_gua'] != sig['tian_gua']).sum()
    if mismatch > 0:
        print(f'  ⚠ d_gua ≠ tian_gua 有 {mismatch} 条 (可能 signal_date vs buy_date 差)')
    return sig


def agg_factor_1d(df, factor_col, ret_col='actual_ret'):
    rows = []
    for g in GUA_ORDER:
        sub = df[df[factor_col] == g]
        n = len(sub)
        if n == 0:
            rows.append({'gua': g, 'name': GUA_NAME[g], 'n': 0,
                         'mean': np.nan, 'median': np.nan, 'win': np.nan, 'sum': 0.0})
            continue
        rets = sub[ret_col].dropna()
        rows.append({
            'gua': g,
            'name': GUA_NAME[g],
            'n': n,
            'mean': rets.mean() if len(rets) else np.nan,
            'median': rets.median() if len(rets) else np.nan,
            'win': (rets > 0).mean() * 100 if len(rets) else np.nan,
            'sum': rets.sum() if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


def _fmt(v, kind='ret'):
    if pd.isna(v):
        return '  -'
    if kind == 'ret':
        return f'{v:+7.2f}%'
    if kind == 'pct':
        return f'{v:5.1f}%'
    if kind == 'int':
        return f'{int(v):>5}'
    if kind == 'sum':
        return f'{v:+10.1f}'
    return str(v)


def print_1d(title, df, factor_col):
    a = agg_factor_1d(df, factor_col)
    print('\n' + '=' * 82)
    print(f'  {title}  (N={len(df)})')
    print('=' * 82)
    hdr = f'  {"卦":<6} {"n":>6} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积收益":>10}'
    print(hdr)
    print('  ' + '-' * 78)
    for _, r in a.iterrows():
        star = ''
        if not pd.isna(r['mean']):
            if r['mean'] >= 2.0:
                star = ' ★★'  # 显著利好
            elif r['mean'] >= 0.5:
                star = ' ★'
            elif r['mean'] <= -2.0:
                star = ' ✗✗'  # 显著利空
            elif r['mean'] <= -0.5:
                star = ' ✗'
        print(f'  {r["gua"]} {r["name"]:<3} {_fmt(r["n"], "int")} {_fmt(r["mean"])} '
              f'{_fmt(r["median"])} {_fmt(r["win"], "pct")} {_fmt(r["sum"], "sum")}{star}')

    # 汇总: 最好/最差 bucket
    valid = a.dropna(subset=['mean'])
    if len(valid) > 0:
        best = valid.loc[valid['mean'].idxmax()]
        worst = valid.loc[valid['mean'].idxmin()]
        spread = best['mean'] - worst['mean']
        print(f'\n  最优: {best["gua"]} {best["name"]} 均收 {best["mean"]:+.2f}% (n={int(best["n"])})')
        print(f'  最差: {worst["gua"]} {worst["name"]} 均收 {worst["mean"]:+.2f}% (n={int(worst["n"])})')
        print(f'  极差 (spread): {spread:.2f} pp  ← 因子强度指标')


def print_2d(title, df, tian_col, factor_col):
    print('\n' + '=' * 110)
    print(f'  {title} — tian_gua(分治) × {factor_col} 二维均收益')
    print('=' * 110)
    mean = df.pivot_table(index=tian_col, columns=factor_col, values='actual_ret', aggfunc='mean')
    cnt = df.pivot_table(index=tian_col, columns=factor_col, values='actual_ret', aggfunc='count', fill_value=0)
    mean = mean.reindex(index=GUA_ORDER, columns=GUA_ORDER)
    cnt = cnt.reindex(index=GUA_ORDER, columns=GUA_ORDER, fill_value=0)

    hdr = '  tian\\' + factor_col.replace('_gua', '') + ' | ' + ' '.join(f'{g}{GUA_NAME[g]:<2}' for g in GUA_ORDER)
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for tg in GUA_ORDER:
        cells = []
        for fg in GUA_ORDER:
            m = mean.loc[tg, fg] if tg in mean.index and fg in mean.columns else np.nan
            n = int(cnt.loc[tg, fg]) if tg in cnt.index and fg in cnt.columns else 0
            if pd.isna(m) or n < 5:
                cells.append('   -   ')
            else:
                cells.append(f'{m:+6.1f}')
        print(f'  {tg} {GUA_NAME[tg]:<3} | ' + ' '.join(cells))
    print()
    print('  [样本数]')
    for tg in GUA_ORDER:
        cells = []
        for fg in GUA_ORDER:
            n = int(cnt.loc[tg, fg]) if tg in cnt.index and fg in cnt.columns else 0
            cells.append(f'{n:>6}')
        print(f'  {tg} {GUA_NAME[tg]:<3} | ' + ' '.join(cells))


def main():
    print('\n  四因子对收益的影响 · 第一轮')
    print(f'  数据源: {os.path.basename(NAKED_RESULT)}  +  {os.path.basename(MULTI_SCALE_PATH)}')

    sig = load_signals()
    scale = load_market_scale()
    sig = enrich_signals(sig, scale)
    print(f'  有效 signal: {len(sig):,} 条 (剔除 is_skip)')
    print(f'  日期范围: {sig["signal_date"].min()} ~ {sig["signal_date"].max()}')
    print(f'  actual_ret: 均 {sig["actual_ret"].mean():+.2f}%  中位 {sig["actual_ret"].median():+.2f}%  '
          f'胜率 {(sig["actual_ret"] > 0).mean() * 100:.1f}%')

    # 一维: 4 因子独立看
    print_1d('因子 1 · 大盘年卦 y_gua (月线构造 · 宏观趋势)', sig, 'y_gua')
    print_1d('因子 2 · 大盘月卦 m_gua (周线构造 · 中周期)', sig, 'm_gua')
    print_1d('因子 3 · 个股地卦 di_gua (象卦 · 个股自身)',   sig, 'di_gua')
    print_1d('因子 4 · 个股人卦 ren_gua (人气 · 资金氛围)', sig, 'ren_gua')

    # 二维: 分治 × 因子 (看因子是"全局信号"还是"局部信号")
    for fc in ('y_gua', 'm_gua', 'di_gua', 'ren_gua'):
        print_2d(fc, sig, 'tian_gua', fc)


if __name__ == '__main__':
    main()
