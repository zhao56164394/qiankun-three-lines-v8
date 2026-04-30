# -*- coding: utf-8 -*-
"""诊断: v1 资金回测 11 年只 +60.78% / 年化 4.56% / MDD 36.5%
   - 期望应有: 坤 v3 +10.84% × 笔, 坎 v3 +14.08% × 笔, 实测都不到 +9%

候选诊断维度:
  1. 资金占用 vs 信号到达: 满仓 66% — 错过多少高 score 信号?
  2. 同 regime score 排序: 高 score 实战表现, 低 score 拖累?
  3. 实测期望 vs 文档期望: 各 regime 偏离多少?
  4. 时序问题: bull 卖晚, 持仓久, 资金周转慢
  5. T+1 buy 问题: 文档按 close 入场, 这里 next_open 入场, 滑点多大?
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    df_trades = pd.read_csv(os.path.join(ROOT, '..', 'data_layer/data/results/capital_trades_v1.csv'),
                             encoding='utf-8-sig')
    df_nav = pd.read_csv(os.path.join(ROOT, '..', 'data_layer/data/results/capital_nav_v1.csv'),
                          encoding='utf-8-sig')

    INIT = 200_000

    # 总览
    print('=== 1. 总体 ===')
    final = df_nav['nav'].iloc[-1]
    print(f'  期末: {final:,.0f}')
    print(f'  总收益: {(final/INIT-1)*100:+.2f}%')
    print(f'  交易笔数: {len(df_trades)}')
    print(f'  胜率: {(df_trades["ret_pct"]>0).mean()*100:.1f}%')
    print(f'  均收益/笔: {df_trades["ret_pct"].mean():+.2f}%')

    # 按 regime + score 分解
    print('\n=== 2. 按 regime × score 拆解 ===')
    print(f'  {"regime":<10} {"score":>5} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"主升期":>9} {"假期":>9} {"max":>7} {"min":>7}')
    df_trades['is_zsl'] = df_trades['ret_pct'] >= 10  # 简化定义
    for r in df_trades['regime'].unique():
        for sc in sorted(df_trades['score'].unique()):
            sub = df_trades[(df_trades['regime'] == r) & (df_trades['score'] == sc)]
            if len(sub) == 0: continue
            zsl = sub[sub['is_zsl']]['ret_pct'].mean() if sub['is_zsl'].sum() > 0 else float('nan')
            fake = sub[~sub['is_zsl']]['ret_pct'].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
            print(f'  {r:<10} {sc:>5} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} {zsl:>+8.2f} {fake:>+7.2f} '
                  f'{sub["ret_pct"].max():>+6.1f} {sub["ret_pct"].min():>+6.1f}')

    # 文档期望 vs 实测对比
    print('\n=== 3. 文档期望 vs 实测 期望偏离 ===')
    doc_expectations = {
        '坤 v3': 10.84, '艮 v3': 16.37, '坎 v3': 14.08,
        '震 v1': 16.44, '离 v1': 6.45, '兑 v1': 4.31, '乾 v3': 4.89,
    }
    print(f'  {"regime":<10} {"文档":>7} {"实测":>7} {"偏离":>7} {"实测笔":>7}')
    for r, doc in doc_expectations.items():
        sub = df_trades[df_trades['regime'] == r]
        if len(sub) == 0: continue
        actual = sub['ret_pct'].mean()
        diff = actual - doc
        print(f'  {r:<10} {doc:>+6.2f} {actual:>+6.2f} {diff:>+6.2f} {len(sub):>7}')

    # 信号占用诊断: 每个 regime 有多少 buy day 在满仓 (即没买入机会)
    print('\n=== 4. 满仓占比分析 ===')
    df_nav['date'] = df_nav['date'].astype(str)
    df_trades['buy_date'] = df_trades['buy_date'].astype(str)
    full_pos_days = (df_nav['pos_count'] >= 10).sum()
    nav_total_days = len(df_nav)
    print(f'  总日数: {nav_total_days}')
    print(f'  满仓 (10) 日数: {full_pos_days} ({full_pos_days/nav_total_days*100:.1f}%)')
    print(f'  ≥9 日数: {(df_nav["pos_count"] >= 9).sum()} ({(df_nav["pos_count"] >= 9).sum()/nav_total_days*100:.1f}%)')

    # 满仓时还有多少 regime 信号被错过? 看 regime 分布
    print('\n=== 5. 各 regime 信号到达天数 vs 实际买入天数 ===')
    df_nav['mkt_y'] = df_nav['mkt_y'].astype(str).str.zfill(3)
    GUA_NAMES = {'000': '坤 v3', '001': '艮 v3', '010': '坎 v3', '011': '巽',
                 '100': '震 v1', '101': '离 v1', '110': '兑 v1', '111': '乾 v3'}
    print(f'  {"regime":<10} {"y_gua":>6} {"regime 日":>9} {"满仓日数":>9} {"满仓占比%":>9}')
    for y, name in GUA_NAMES.items():
        if name == '巽': continue
        regime_days = (df_nav['mkt_y'] == y).sum()
        full_in_regime = ((df_nav['mkt_y'] == y) & (df_nav['pos_count'] >= 10)).sum()
        if regime_days > 0:
            pct = full_in_regime / regime_days * 100
            print(f'  {name:<10} {y:>6} {regime_days:>9} {full_in_regime:>9} {pct:>8.1f}')

    # 持仓机会成本: 持仓时是哪个 regime, 卖出后切到哪个
    print('\n=== 6. 持仓 → 后续 regime 转换 ===')
    print('  (买入时 regime 不同于今日 regime 的天数 = 错过新 regime 信号)')
    df_trades['buy_y'] = df_trades['regime'].map({
        '坤 v3': '000', '艮 v3': '001', '坎 v3': '010',
        '震 v1': '100', '离 v1': '101', '兑 v1': '110', '乾 v3': '111',
    })

    # 看 sell_date → 之后 30 日 nav 变化
    print('\n=== 7. 时间分布: 资金哪些年涨/亏 ===')
    df_nav['year'] = pd.to_datetime(df_nav['date']).dt.year
    yearly_nav = df_nav.groupby('year').agg(start=('nav', 'first'), end=('nav', 'last'))
    yearly_nav['ret_pct'] = (yearly_nav['end'] / yearly_nav['start'] - 1) * 100
    yearly_nav['draw'] = ''
    for y in yearly_nav.index:
        sub = df_nav[df_nav['year'] == y]
        peak = sub['nav'].cummax()
        dd = ((sub['nav'] - peak) / peak * 100).min()
        yearly_nav.loc[y, 'draw'] = f'{dd:.1f}'
    print(yearly_nav.round(2))

    # 持仓数分布对收益的影响
    print('\n=== 8. 持仓数 vs 当日收益率 ===')
    df_nav_sorted = df_nav.sort_values('date').reset_index(drop=True)
    df_nav_sorted['ret'] = df_nav_sorted['nav'].pct_change() * 100
    print(f'  {"持仓数":<6} {"日数":>5} {"均日收益%":>9} {"年化%":>8}')
    for pc in sorted(df_nav_sorted['pos_count'].unique()):
        sub = df_nav_sorted[df_nav_sorted['pos_count'] == pc]
        avg = sub['ret'].mean()
        ann = (1 + avg/100) ** 252 - 1
        print(f'  {pc:<6} {len(sub):>5} {avg:>+8.3f} {ann*100:>+7.2f}')

    # 按 reason 看卖出
    print('\n=== 9. 按卖出原因 ===')
    print(f'  {"reason":<14} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"均持仓":>7}')
    for r in df_trades['reason'].unique():
        sub = df_trades[df_trades['reason'] == r]
        print(f'  {r:<14} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
              f'{sub["ret_pct"].mean():>+8.2f} {sub["days"].mean():>6.1f}')


if __name__ == '__main__':
    main()
