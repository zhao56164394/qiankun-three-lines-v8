# -*- coding: utf-8 -*-
"""m_gua × 地/人卦 二维交叉分析

背景: m_gua=010 确认为跨年稳定正向真因子 (+14.31%, 7 年 6 正).
      但 m_gua 只是个"市场中周期状态" -- 要想真正用起来, 必须看它和
      个股层的地/人卦组合后是否有协同放大效应.

分析目标:
  1. m_gua × di_gua (8x8) 均收热图: 哪些组合稳定强/弱?
  2. m_gua × ren_gua (8x8) 均收热图: 同上
  3. 对 top 10 强组合 / top 5 弱组合, 看跨年份稳定性 (避免 y_gua=101 教训)
  4. 导出 "(m_gua, di_gua/ren_gua) → 均收 + 年份稳定性" 表, 作为未来过滤规则候选
"""
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED_RESULT = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
MULTI = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def load_data():
    with open(NAKED_RESULT, encoding='utf-8') as f:
        d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    sig['signal_date'] = pd.to_datetime(sig['signal_date'])
    sig = sig[~sig['is_skip']].copy()
    sig['year'] = sig['signal_date'].dt.year
    for c in ('di_gua', 'ren_gua', 'tian_gua'):
        sig[c] = sig[c].astype(str).str.zfill(3)
    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'m_gua': str, 'y_gua': str})
    ms['date'] = pd.to_datetime(ms['date'])
    sig = sig.merge(ms[['date', 'm_gua', 'y_gua']], left_on='signal_date', right_on='date', how='left')
    return sig


def print_heatmap(title, sig, factor_col, min_n=20):
    print('\n' + '=' * 100)
    print(f'  {title}  (只标注 n ≥ {min_n})')
    print('=' * 100)

    cnt = sig.pivot_table(index='m_gua', columns=factor_col, values='actual_ret', aggfunc='count', fill_value=0)
    mean = sig.pivot_table(index='m_gua', columns=factor_col, values='actual_ret', aggfunc='mean')
    cnt = cnt.reindex(index=GUA_ORDER, columns=GUA_ORDER, fill_value=0)
    mean = mean.reindex(index=GUA_ORDER, columns=GUA_ORDER)

    print('\n  [均收 %] 行=m_gua (中周期)  列=' + factor_col)
    print('  m\\' + factor_col[:3] + '   | ' + ' '.join(f'{g}{GUA_NAME[g]:<2}' for g in GUA_ORDER))
    print('  ' + '-' * 80)
    for mg in GUA_ORDER:
        cells = []
        for fg in GUA_ORDER:
            m = mean.loc[mg, fg]
            n = int(cnt.loc[mg, fg])
            if pd.isna(m) or n < min_n:
                cells.append('   -  ')
            else:
                cells.append(f'{m:+6.1f}')
        print(f'  {mg} {GUA_NAME[mg]:<3} | ' + ' '.join(cells))

    print('\n  [样本数]')
    print('  m\\' + factor_col[:3] + '   | ' + ' '.join(f'{g}{GUA_NAME[g]:<2}' for g in GUA_ORDER))
    print('  ' + '-' * 80)
    for mg in GUA_ORDER:
        cells = []
        for fg in GUA_ORDER:
            n = int(cnt.loc[mg, fg])
            cells.append(f'{n:>6}')
        print(f'  {mg} {GUA_NAME[mg]:<3} | ' + ' '.join(cells))


def top_combos(sig, factor_col, min_n=30, top_n=15):
    """返回 (m_gua, factor) 组合均收 top N + bottom N"""
    agg = sig.groupby(['m_gua', factor_col]).agg(
        n=('actual_ret', 'count'),
        mean=('actual_ret', 'mean'),
        win=('actual_ret', lambda x: (x > 0).mean() * 100),
        sum=('actual_ret', 'sum'),
    ).reset_index()
    agg = agg[agg['n'] >= min_n].copy()
    agg_sorted_up = agg.sort_values('mean', ascending=False).head(top_n)
    agg_sorted_dn = agg.sort_values('mean', ascending=True).head(top_n)
    return agg_sorted_up, agg_sorted_dn


def year_stability(sig, m_gua, factor_col, factor_val, min_year_n=10):
    """一个组合的跨年份稳定性"""
    sub = sig[(sig['m_gua'] == m_gua) & (sig[factor_col] == factor_val)]
    if len(sub) == 0:
        return None
    by_year = sub.groupby('year').agg(
        n=('actual_ret', 'count'),
        mean=('actual_ret', 'mean'),
    )
    qual = by_year[by_year['n'] >= min_year_n]
    return {
        'n_total': len(sub),
        'n_years_qualified': len(qual),
        'pos_years': int((qual['mean'] > 0).sum()),
        'neg_years': int((qual['mean'] < 0).sum()),
        'max_year_share': (by_year['n'].max() / len(sub) * 100) if len(sub) else 0,
        'by_year': by_year,
    }


def print_top_combos(title, agg_df, sig, factor_col):
    print('\n' + '=' * 110)
    print(f'  {title}')
    print('=' * 110)
    print(f'  {"组合":<20} {"n":>6} {"均收":>8} {"胜率":>6} {"累积":>9} | {"合格年":>6} {"正":>3} {"负":>3} {"最大年份占比":>10}')
    print('  ' + '-' * 105)
    for _, r in agg_df.iterrows():
        m = r['m_gua']
        f = r[factor_col]
        stab = year_stability(sig, m, factor_col, f)
        stab_info = f'{stab["n_years_qualified"]:>6} {stab["pos_years"]:>3} {stab["neg_years"]:>3} {stab["max_year_share"]:>9.1f}%'
        combo = f'm={m}{GUA_NAME[m]} × {factor_col[:3]}={f}{GUA_NAME[f]}'
        print(f'  {combo:<20} {int(r["n"]):>6} {r["mean"]:>+7.2f}% {r["win"]:>5.1f}% {r["sum"]:>+9.1f} | {stab_info}')


def main():
    print('\n  m_gua × 地/人卦 二维交叉分析\n')
    sig = load_data()
    print(f'  有效 signal: {len(sig):,} 条')

    # ======== 热图 ========
    print_heatmap('m_gua × di_gua (地卦/象卦) 均收 %', sig, 'di_gua')
    print_heatmap('m_gua × ren_gua (人卦/人气) 均收 %', sig, 'ren_gua')

    # ======== Top 15 正向组合 ========
    print('\n\n【地卦协同】')
    up_di, dn_di = top_combos(sig, 'di_gua', min_n=30, top_n=15)
    print_top_combos('Top 15 正向 m_gua × di_gua  (n≥30, 均收降序)', up_di, sig, 'di_gua')
    print_top_combos('Top 15 负向 m_gua × di_gua  (n≥30, 均收升序)', dn_di, sig, 'di_gua')

    print('\n\n【人卦协同】')
    up_ren, dn_ren = top_combos(sig, 'ren_gua', min_n=30, top_n=15)
    print_top_combos('Top 15 正向 m_gua × ren_gua (n≥30, 均收降序)', up_ren, sig, 'ren_gua')
    print_top_combos('Top 15 负向 m_gua × ren_gua (n≥30, 均收升序)', dn_ren, sig, 'ren_gua')


if __name__ == '__main__':
    main()
