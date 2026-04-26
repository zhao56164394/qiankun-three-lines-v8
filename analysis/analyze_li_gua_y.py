# -*- coding: utf-8 -*-
"""离卦 (101) 第三轮 · 大盘年卦 y_gua 视角

核心假设 (用户提出):
  时间稳定性问题 (2019-21 整体偏负, 2022-25 整体偏正) 是不是因为没有用 y_gua?
  如果 y_gua 能清晰区分牛/熊/震荡, 那"时间不稳"就转化为"年卦环境不同"
  → 整套买卖在不同年卦下应该不一样

验证路径:
  A. 看 2015-2025 每年的 y_gua 分布 (是否真的有牛/熊/震荡的分割)
  B. 把 489 离卦信号按 y_gua 分桶, 看 n / 均值 / CI
  C. y_gua × signal_year 交叉, 验证 "2019-21 偏负" 是不是 "y_gua=某些特定值"
  D. y_gua × ren_gua 交叉, 看是否能进一步解释
"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
MULTI = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA_ORDER = ['000','001','010','011','100','101','110','111']
GUA_NAME = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}
GUA_MEANING = {'000':'深熊探底','001':'熊底异动','010':'反弹乏力','011':'底部爆发',
               '100':'崩盘加速','101':'下跌护盘','110':'牛末滞涨','111':'疯牛主升'}


def load():
    with open(NAKED, encoding='utf-8') as f: d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    sig = sig[~sig['is_skip']].copy()
    for c in ('di_gua','ren_gua','tian_gua'):
        sig[c] = sig[c].astype(str).str.zfill(3)
    ms = pd.read_csv(MULTI, encoding='utf-8-sig',
                     dtype={'d_gua':str,'m_gua':str,'y_gua':str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
    for c in ('d_gua','m_gua','y_gua'):
        ms[c] = ms[c].astype(str).str.zfill(3)
    sig = sig.merge(ms[['date','d_gua','m_gua','y_gua']],
                    left_on='signal_date', right_on='date', how='left')
    sig['signal_year'] = pd.to_datetime(sig['signal_date']).dt.year
    li = sig[sig['tian_gua']=='101'].copy()
    return li, ms


def bootstrap_ci(arr, n_iter=2000, alpha=0.05, seed=42):
    arr = np.asarray(arr, dtype=float); arr = arr[~np.isnan(arr)]
    if len(arr) < 5: return (np.mean(arr) if len(arr) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_iter)])
    return (arr.mean(), np.quantile(means, alpha/2), np.quantile(means, 1-alpha/2))


def section_a_y_gua_timeline(ms):
    print('\n' + '='*100)
    print('  A. y_gua 历年分布 — 看年卦是否真的随牛熊周期变化')
    print('='*100)
    ms = ms.copy()
    ms['year'] = pd.to_datetime(ms['date']).dt.year
    yearly = ms[ms['year'] >= 2014].pivot_table(
        index='year', columns='y_gua', aggfunc='size', fill_value=0)
    yearly = yearly.reindex(columns=GUA_ORDER, fill_value=0)
    print(f'  {"年份":<6}', end='')
    for g in GUA_ORDER: print(f'  {g}{GUA_NAME[g]:>3}', end='')
    print(f'  {"主导卦":>10} {"占比":>6}')
    for y, row in yearly.iterrows():
        total = row.sum()
        if total == 0: continue
        cells = []
        max_g = row.idxmax(); max_p = row.max()/total*100
        for g in GUA_ORDER:
            p = row[g]/total*100 if total else 0
            cells.append(f'{p:5.1f}%')
        print(f'  {y:<6}' + '  '.join(cells) + f'  {max_g}{GUA_NAME[max_g]:>3} {max_p:5.1f}%')


def section_b_y_gua_returns(li):
    print('\n' + '='*100)
    print('  B. 离卦信号在不同 y_gua 环境下的收益 + 95% CI')
    print('='*100)
    print(f'  {"y_gua":<14}     n     均收           95% CI         胜率   显著')
    print('  ' + '-'*92)
    for g in GUA_ORDER:
        sub = li[li['y_gua']==g]
        n = len(sub)
        if n == 0:
            print(f'  {g} {GUA_NAME[g]} ({GUA_MEANING[g]:<4})  {0:>5}      -                  -        -')
            continue
        m, lo, hi = bootstrap_ci(sub['actual_ret'].values)
        win = (sub['actual_ret']>0).mean()*100
        sig_mark = ''
        if not np.isnan(lo):
            if lo > 0: sig_mark = '★ 全正'
            elif hi < 0: sig_mark = '✗ 全负'
        ci_str = f'[{lo:+5.2f}, {hi:+5.2f}]' if not np.isnan(lo) else '   (n太少)'
        print(f'  {g} {GUA_NAME[g]} ({GUA_MEANING[g]:<4})  {n:>5}  {m:+7.2f}% {ci_str:>22}  {win:5.1f}%  {sig_mark}')


def section_c_year_x_y_gua(li):
    print('\n' + '='*100)
    print('  C. signal_year × y_gua — 验证 "2019-21 偏负" 是不是某些 y_gua 的集中')
    print('='*100)
    cnt = li.pivot_table(index='signal_year', columns='y_gua', values='actual_ret',
                         aggfunc='count', fill_value=0)
    mean = li.pivot_table(index='signal_year', columns='y_gua', values='actual_ret',
                          aggfunc='mean')
    cnt = cnt.reindex(columns=GUA_ORDER, fill_value=0)
    mean = mean.reindex(columns=GUA_ORDER)

    print(f'  {"year":<6}', end='')
    for g in GUA_ORDER: print(f' {g}{GUA_NAME[g]:>3}', end='')
    print(f' {"全":>5} {"年总n":>5} {"年均":>7}')
    print('  ' + '-'*100)
    for y in sorted(li['signal_year'].unique()):
        cells_n = []; cells_m = []
        for g in GUA_ORDER:
            n = int(cnt.loc[y, g]) if y in cnt.index else 0
            m = mean.loc[y, g] if y in mean.index else np.nan
            cells_n.append(n)
            if n < 3 or pd.isna(m):
                cells_m.append('   -  ')
            else:
                cells_m.append(f'{m:+5.1f}')
        total_n = sum(cells_n)
        year_mean = li[li['signal_year']==y]['actual_ret'].mean()
        # 用 mean 行
        print(f'  {y:<6}', end='')
        for g, m in zip(GUA_ORDER, cells_m): print(f' {m:>5}', end='')
        print(f' {"":>5} {total_n:>5} {year_mean:+6.2f}%')
        # n 行
        print(f'  {"":<6}', end='')
        for n in cells_n: print(f' n={n:<3}', end='')
        print()


def section_d_y_x_ren(li):
    print('\n' + '='*100)
    print('  D. y_gua × ren_gua — 在不同年卦下 ren_gua 是否表现不同')
    print('='*100)
    cnt = li.pivot_table(index='y_gua', columns='ren_gua', values='actual_ret',
                         aggfunc='count', fill_value=0).reindex(GUA_ORDER, fill_value=0)
    mean = li.pivot_table(index='y_gua', columns='ren_gua', values='actual_ret',
                          aggfunc='mean').reindex(GUA_ORDER)
    cnt = cnt.reindex(columns=GUA_ORDER, fill_value=0)
    mean = mean.reindex(columns=GUA_ORDER)
    head = 'y\\ren'
    print(f'  {head:<8} ' + ' '.join(f'{g}{GUA_NAME[g]:<3}' for g in GUA_ORDER))
    for g in GUA_ORDER:
        cells = []
        for cg in GUA_ORDER:
            n = int(cnt.loc[g, cg]); m = mean.loc[g, cg]
            if n < 5 or pd.isna(m):
                cells.append('   -  ')
            else:
                cells.append(f'{m:+5.1f}')
        print(f'  {g} {GUA_NAME[g]:<4} ' + ' '.join(cells))
    print('\n  [count]')
    for g in GUA_ORDER:
        print(f'  {g} {GUA_NAME[g]:<4} ' + ' '.join(f'{int(cnt.loc[g, cg]):>5}' for cg in GUA_ORDER))


def section_e_three_factor(li):
    print('\n' + '='*100)
    print('  E. 三因子联合 — y_gua + m_gua + ren_gua, 找出 "明确好/明确坏" 的组合')
    print('='*100)
    grp = li.groupby(['y_gua','m_gua','ren_gua'])['actual_ret'].agg(['count','mean']).reset_index()
    grp = grp[grp['count'] >= 5].sort_values('mean', ascending=False)
    print('  排名 Top 8 (n>=5)')
    print(f'  {"y":>4} {"m":>4} {"ren":>4}  {"n":>4}  {"均收":>8}')
    for _, r in grp.head(8).iterrows():
        print(f'  {r["y_gua"]:>4} {r["m_gua"]:>4} {r["ren_gua"]:>4}  {int(r["count"]):>4}  {r["mean"]:+7.2f}%')
    print('\n  排名 Bottom 8 (n>=5)')
    print(f'  {"y":>4} {"m":>4} {"ren":>4}  {"n":>4}  {"均收":>8}')
    for _, r in grp.tail(8).iterrows():
        print(f'  {r["y_gua"]:>4} {r["m_gua"]:>4} {r["ren_gua"]:>4}  {int(r["count"]):>4}  {r["mean"]:+7.2f}%')


def main():
    print('\n  离卦 (101) 第三轮 · 大盘年卦 y_gua 视角')
    print('  假设: 时间分布问题 = 年卦没接进来, 不是结构性限制')
    li, ms = load()
    print(f'  全量 n={len(li)}, 日期 {li["signal_date"].min()} ~ {li["signal_date"].max()}')

    section_a_y_gua_timeline(ms)
    section_b_y_gua_returns(li)
    section_c_year_x_y_gua(li)
    section_d_y_x_ren(li)
    section_e_three_factor(li)


if __name__ == '__main__':
    main()
