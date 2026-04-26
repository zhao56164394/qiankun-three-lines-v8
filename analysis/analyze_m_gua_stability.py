# -*- coding: utf-8 -*-
"""周卦 m_gua 8 桶跨年份稳定性验证

背景: 一维分析里 m_gua spread=19.55 pp (最强), 但我们已经见识过
      y_gua=101 的 "98% 样本集中 2015" 伪因子陷阱. 所以 m_gua 的 8 桶
      也必须按年份切片看, 避免被单个事件主导.

判据: 一个桶 "跨年份稳定", 至少满足:
  - 在 3+ 个不同年份都有 ≥10 条信号
  - 这些年份里均收方向一致 (都正 或 都负)
  - 年份加权平均不是单个年份主导 (最大年份占比 < 50%)
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


def main():
    with open(NAKED_RESULT, encoding='utf-8') as f:
        d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    sig['signal_date'] = pd.to_datetime(sig['signal_date'])
    sig = sig[~sig['is_skip']].copy()
    sig['year'] = sig['signal_date'].dt.year

    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'m_gua': str, 'y_gua': str, 'd_gua': str})
    ms['date'] = pd.to_datetime(ms['date'])
    sig = sig.merge(ms[['date', 'm_gua', 'y_gua']],
                    left_on='signal_date', right_on='date', how='left')

    print('\n  周卦 m_gua 8 桶跨年份稳定性')
    print(f'  有效 signal: {len(sig):,} 条')
    print(f'  年份范围: {sig["year"].min()} ~ {sig["year"].max()}')

    for g in GUA_ORDER:
        sub = sig[sig['m_gua'] == g]
        total_n = len(sub)
        if total_n == 0:
            print(f'\n  m_gua={g} {GUA_NAME[g]}: 无样本')
            continue

        total_mean = sub['actual_ret'].mean()
        total_win = (sub['actual_ret'] > 0).mean() * 100

        print('\n' + '=' * 90)
        print(f'  m_gua={g} {GUA_NAME[g]}  总样本 {total_n:,}  总均收 {total_mean:+.2f}%  总胜率 {total_win:.1f}%')
        print('=' * 90)
        by_year = sub.groupby('year').agg(
            n=('actual_ret', 'count'),
            mean=('actual_ret', 'mean'),
            win=('actual_ret', lambda x: (x > 0).mean() * 100),
            sum_ret=('actual_ret', 'sum'),
        )
        print(f'  {"年份":<6} {"n":>5} {"均收":>8} {"胜率":>6} {"累积":>9} {"占比":>6}')
        for y, r in by_year.iterrows():
            share = r['n'] / total_n * 100
            marker = ''
            if r['n'] >= 10 and not pd.isna(r['mean']):
                if r['mean'] >= 3.0:
                    marker = '  ★'
                elif r['mean'] <= -3.0:
                    marker = '  ✗'
            print(f'  {y:<6} {int(r["n"]):>5} {r["mean"]:>+7.2f}% {r["win"]:>5.1f}% {r["sum_ret"]:>+9.1f} {share:>5.1f}%{marker}')

        # 稳定性判据
        qualified = by_year[(by_year['n'] >= 10)]
        n_qualified_years = len(qualified)
        max_share = by_year['n'].max() / total_n * 100 if total_n > 0 else 0
        pos_years = (qualified['mean'] > 0).sum()
        neg_years = (qualified['mean'] < 0).sum()
        direction = '正' if pos_years > neg_years * 2 else ('负' if neg_years > pos_years * 2 else '混合')
        consistency = ''
        if n_qualified_years >= 3:
            if direction == '正' and total_mean > 2:
                consistency = '  → 稳定正向 ✓'
            elif direction == '负' and total_mean < -2:
                consistency = '  → 稳定负向 ✓ (可作拒买过滤)'
            else:
                consistency = '  → 方向混合或均收偏弱'
        else:
            consistency = f'  → 样本年份太少 ({n_qualified_years} 年 ≥10 条)'

        print(f'\n  [稳定性] ≥10 条信号的年份: {n_qualified_years}  |  最大年份占比: {max_share:.1f}%  |  方向: {direction} (正{pos_years}年/负{neg_years}年){consistency}')


if __name__ == '__main__':
    main()
