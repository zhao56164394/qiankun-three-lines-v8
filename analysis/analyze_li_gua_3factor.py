# -*- coding: utf-8 -*-
"""离卦 (101) 三因子双视角分析 · 大盘月卦 m_gua / 个股地卦 di_gua / 个股人卦 ren_gua

数据源:
  - data_layer/data/backtest_8gua_naked_result.json  (signal_detail / trade_log)
  - data_layer/data/foundation/multi_scale_gua_daily.csv  (大盘 d/m/y 卦, 按日对齐)

只保留 tian_gua=='101' (离卦) 的信号, 然后按三个因子分别分桶看收益:
  - 大盘月卦 m_gua : 中周期市场结构 (周线压缩成的卦)
  - 个股地卦 di_gua: 象卦, 个股自身价格三线状态
  - 个股人卦 ren_gua: 人气, 资金/情绪氛围

双视角:
  全量 (signal): 假设无资金约束都买入, actual_ret -> 因子的"理论 alpha"
  买入 (trade) : 资金约束 + 5 仓限制下实际买入 22 笔, ret_pct -> 因子在实战是否兑现

输出 4 部分:
  1. 三因子一维表 (全量 + 买入并排)
  2. 二维交叉: m_gua × ren_gua, m_gua × di_gua, di_gua × ren_gua (全量)
  3. 现有过滤 (li_exclude_ren_gua=['001'], li_allow_di_gua=['000']) 在双视角下的拆分
  4. 八卦理论解读 + 候选过滤组合的 backtest 增量
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

GUA = '101'
GUA_NAME_ZH = '离'

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}
GUA_MEANING = {
    '000': '深熊探底', '001': '熊底异动', '010': '反弹乏力', '011': '底部爆发',
    '100': '崩盘加速', '101': '下跌护盘', '110': '牛末滞涨', '111': '疯牛主升',
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
    trd['m_gua'] = trd['buy_date'].astype(str).map(m_map).fillna('').astype(str)

    li_s = sig[sig['tian_gua'] == GUA].copy()
    li_t = trd[trd['gua'] == GUA].copy() if 'gua' in trd.columns else trd[trd['tian_gua'] == GUA].copy()

    # trade -> signal 反查 di_gua/ren_gua (有时 trade_log 不带)
    if 'di_gua' not in li_t.columns or li_t['di_gua'].isna().any():
        key = li_s.set_index(['code', 'buy_date'])[['di_gua', 'ren_gua']]
        li_t = li_t.merge(key.reset_index(), on=['code', 'buy_date'], how='left', suffixes=('_old', ''))
    return li_s, li_t


def _fmt(v, kind='ret'):
    if pd.isna(v):
        return '   -  '
    if kind == 'ret':
        return f'{v:+6.2f}%'
    if kind == 'pct':
        return f'{v:5.1f}%'
    if kind == 'int':
        return f'{int(v):>5}'
    if kind == 'sum':
        return f'{v:+10.1f}'
    return str(v)


def _star(v):
    if pd.isna(v):
        return ''
    if v >= 8:
        return ' ★★★'
    if v >= 5:
        return ' ★★'
    if v >= 2:
        return ' ★'
    if v <= -5:
        return ' ✗✗'
    if v <= -2:
        return ' ✗'
    return ''


def agg(df, factor, ret_col):
    rows = []
    for g in GUA_ORDER:
        sub = df[df[factor] == g]
        rets = sub[ret_col].dropna()
        rows.append({
            'gua': g,
            'n': len(sub),
            'mean': rets.mean() if len(rets) else np.nan,
            'median': rets.median() if len(rets) else np.nan,
            'win': (rets > 0).mean() * 100 if len(rets) else np.nan,
            'sum': rets.sum() if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


def print_factor(title, sig, trd, factor):
    s = agg(sig, factor, 'actual_ret')
    t = agg(trd, factor, 'ret_pct')

    print('\n' + '=' * 118)
    print(f'  {title}  |  全量 N={len(sig)}  买入 N={len(trd)}')
    print('=' * 118)
    hdr = (f'  {"卦":<14} | {"全 n":>4} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>10}'
           f'  | {"买 n":>4} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>9}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for g in GUA_ORDER:
        rs = s[s['gua'] == g].iloc[0]
        rt = t[t['gua'] == g].iloc[0]
        line = (f'  {g} {GUA_NAME[g]} ({GUA_MEANING[g][:4]}) | '
                f'{_fmt(rs["n"], "int")} {_fmt(rs["mean"])} {_fmt(rs["median"])} '
                f'{_fmt(rs["win"], "pct")} {_fmt(rs["sum"], "sum")}  | '
                f'{_fmt(rt["n"], "int")} {_fmt(rt["mean"])} {_fmt(rt["median"])} '
                f'{_fmt(rt["win"], "pct")} {_fmt(rt["sum"], "sum")}')
        print(line + _star(rs['mean']))

    valid = s.dropna(subset=['mean'])
    if len(valid):
        best = valid.loc[valid['mean'].idxmax()]
        worst = valid.loc[valid['mean'].idxmin()]
        print(f'\n  全量 spread (最优-最差): {best["mean"] - worst["mean"]:.2f} pp '
              f'[最优 {best["gua"]} {GUA_NAME[best["gua"]]} {best["mean"]:+.2f}%, '
              f'最差 {worst["gua"]} {GUA_NAME[worst["gua"]]} {worst["mean"]:+.2f}%]')


def print_2d(title, df, row_factor, col_factor, ret_col, min_n=5):
    print('\n' + '-' * 100)
    print(f'  {title}  ({row_factor} × {col_factor},  最少 {min_n} 样本才显示)')
    print('-' * 100)
    cnt = df.pivot_table(index=row_factor, columns=col_factor, values=ret_col, aggfunc='count', fill_value=0)
    mean = df.pivot_table(index=row_factor, columns=col_factor, values=ret_col, aggfunc='mean')
    cnt = cnt.reindex(index=GUA_ORDER, columns=GUA_ORDER, fill_value=0)
    mean = mean.reindex(index=GUA_ORDER, columns=GUA_ORDER)

    print(f'  {row_factor[:5]}\\{col_factor[:5]} | ' + ' '.join(f'{g}{GUA_NAME[g]:<2}' for g in GUA_ORDER))
    for rg in GUA_ORDER:
        cells = []
        for cg in GUA_ORDER:
            n = int(cnt.loc[rg, cg])
            m = mean.loc[rg, cg]
            if n < min_n or pd.isna(m):
                cells.append('   -   ')
            else:
                cells.append(f'{m:+5.1f}')
        print(f'  {rg} {GUA_NAME[rg]:<3}     | ' + ' '.join(cells))

    print('  [count]')
    for rg in GUA_ORDER:
        print(f'  {rg} {GUA_NAME[rg]:<3}     | ' + ' '.join(f'{int(cnt.loc[rg, cg]):>5}' for cg in GUA_ORDER))


def split_filter(sig, trd, name, mask_sig, mask_trd):
    """对比 mask=True (保留) vs mask=False (剔除) 的双视角"""
    print('\n' + '=' * 118)
    print(f'  过滤效果 · {name}')
    print('=' * 118)
    hdr = (f'  {"组":<22} | {"全 n":>5} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>10}'
           f'  | {"买 n":>5} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>9}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for label, mS, mT in (('保留 (filter pass)', mask_sig, mask_trd),
                          ('剔除 (filter out) ', ~mask_sig, ~mask_trd)):
        sub_s = sig[mS]
        sub_t = trd[mT]
        ms = sub_s['actual_ret'].dropna()
        mt = sub_t['ret_pct'].dropna() if len(sub_t) else pd.Series(dtype=float)
        line = (f'  {label:<22} | '
                f'{_fmt(len(sub_s), "int")} {_fmt(ms.mean() if len(ms) else np.nan)} '
                f'{_fmt(ms.median() if len(ms) else np.nan)} '
                f'{_fmt((ms > 0).mean() * 100 if len(ms) else np.nan, "pct")} '
                f'{_fmt(ms.sum(), "sum")}  | '
                f'{_fmt(len(sub_t), "int")} {_fmt(mt.mean() if len(mt) else np.nan)} '
                f'{_fmt(mt.median() if len(mt) else np.nan)} '
                f'{_fmt((mt > 0).mean() * 100 if len(mt) else np.nan, "pct")} '
                f'{_fmt(mt.sum() if len(mt) else 0.0, "sum")}')
        print(line)


def main():
    print(f'\n  离卦 ({GUA} {GUA_NAME_ZH}) 三因子双视角分析 — 大盘月卦 / 地卦 / 人卦')
    li_s, li_t = load_data()
    print(f'  全量 signal: {len(li_s)} 条 (剔 is_skip)   |   实买 trade: {len(li_t)} 笔')
    print(f'  全量 actual_ret 均 {li_s["actual_ret"].mean():+.2f}%  胜 {(li_s["actual_ret"]>0).mean()*100:.1f}%')
    if len(li_t):
        print(f'  实买 ret_pct    均 {li_t["ret_pct"].mean():+.2f}%  胜 {(li_t["ret_pct"]>0).mean()*100:.1f}%')

    # 一维三因子
    print_factor('因子 1 · 大盘月卦 m_gua  (中周期, 周线压缩)', li_s, li_t, 'm_gua')
    print_factor('因子 2 · 个股地卦 di_gua (象卦, 价格三线静态)', li_s, li_t, 'di_gua')
    print_factor('因子 3 · 个股人卦 ren_gua (人气, 资金氛围)',     li_s, li_t, 'ren_gua')

    # 二维 (全量为主, 样本量足)
    print('\n\n  ' + '#' * 60)
    print('  二维交叉 · 全量 actual_ret')
    print('  ' + '#' * 60)
    print_2d('m_gua × di_gua',  li_s, 'm_gua',  'di_gua',  'actual_ret')
    print_2d('m_gua × ren_gua', li_s, 'm_gua',  'ren_gua', 'actual_ret')
    print_2d('di_gua × ren_gua', li_s, 'di_gua', 'ren_gua', 'actual_ret')

    # 现有过滤拆分
    print('\n\n  ' + '#' * 60)
    print('  现有过滤效果验证 (li_exclude_ren_gua=[001], li_allow_di_gua=[000])')
    print('  ' + '#' * 60)

    split_filter(li_s, li_t, 'ren_gua != 001 (现有 exclude)',
                 li_s['ren_gua'] != '001', li_t['ren_gua'] != '001')
    split_filter(li_s, li_t, 'di_gua == 000 (现有 allow)',
                 li_s['di_gua'] == '000', li_t['di_gua'] == '000')

    # 注: 现有 cfg 已经把 di_gua 限制到 000 了, 所以 li_s 里实际全是 di_gua=000.
    # 上面 split_filter 里 "剔除" 那行如果 n=0, 说明已经被前置过滤了.


if __name__ == '__main__':
    main()
