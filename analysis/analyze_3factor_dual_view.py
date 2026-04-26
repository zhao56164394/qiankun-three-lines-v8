# -*- coding: utf-8 -*-
"""三因子 (大盘周卦 m_gua / 个股地卦 di_gua / 个股人卦 ren_gua) 对收益的影响

数据源: data_layer/data/backtest_8gua_naked_result.json
  - signal_detail: 14130 条全量候选信号 (actual_ret = 信号到平仓的真实收益)
  - trade_log:       445 条实际成交 (ret_pct = 资金约束 + 5 仓限制下实际买入的笔)

双视角:
  - 全量 (signal): 假设无资金约束都买入, 看每个桶的"理论收益"
  - 买入 (trade):  5 仓 + 等分仓 实际成交, 看每个桶的"实战收益"

为什么双视角同时看:
  - 全量视角剔除资金约束, 暴露因子的"纯 alpha"
  - 买入视角受时间序列 + 仓位约束影响, 显示因子在实战中能否兑现
  - 两视角差异越大, 说明该因子对资金占用的敏感性越高
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

    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'m_gua': str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
    m_map = dict(zip(ms['date'], ms['m_gua'].fillna('')))

    sig['m_gua'] = sig['signal_date'].astype(str).map(m_map).fillna('').astype(str)
    trd['m_gua'] = trd['buy_date'].astype(str).map(m_map).fillna('').astype(str)
    return sig, trd


def agg_factor(df, factor_col, ret_col):
    rows = []
    for g in GUA_ORDER:
        sub = df[df[factor_col] == g]
        n = len(sub)
        if n == 0:
            rows.append({'gua': g, 'n': 0, 'mean': np.nan, 'median': np.nan,
                         'win': np.nan, 'sum': 0.0})
            continue
        rets = sub[ret_col].dropna()
        rows.append({
            'gua': g,
            'n': n,
            'mean': rets.mean() if len(rets) else np.nan,
            'median': rets.median() if len(rets) else np.nan,
            'win': (rets > 0).mean() * 100 if len(rets) else np.nan,
            'sum': rets.sum() if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


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


def print_factor(title, sig, trd, factor_col):
    sig_a = agg_factor(sig, factor_col, 'actual_ret')
    trd_a = agg_factor(trd, factor_col, 'ret_pct') if len(trd) else None

    sig_total = len(sig)
    trd_total = len(trd) if trd is not None else 0

    print('\n' + '=' * 110)
    print(f'  {title}  |  全量 N={sig_total:,}  买入 N={trd_total:,}')
    print('=' * 110)
    hdr = f'  {"卦":<14} | {"全量 n":>6} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>10}  | {"买入 n":>5} {"均收":>8} {"中位":>8} {"胜率":>6} {"累积":>9}'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for g in GUA_ORDER:
        rs = sig_a[sig_a['gua'] == g].iloc[0]
        if trd_a is not None and len(trd_a) > 0:
            rt = trd_a[trd_a['gua'] == g].iloc[0]
        else:
            rt = pd.Series({'n': 0, 'mean': np.nan, 'median': np.nan, 'win': np.nan, 'sum': 0.0})
        # marker for signal mean
        m = ''
        if not pd.isna(rs['mean']):
            if rs['mean'] >= 5:   m = ' ★★'
            elif rs['mean'] >= 1: m = ' ★'
            elif rs['mean'] <= -5: m = ' ✗✗'
            elif rs['mean'] <= -1: m = ' ✗'
        line = (f'  {g} {GUA_NAME[g]} ({GUA_MEANING[g][:4]}) | '
                f'{_fmt(rs["n"], "int")} {_fmt(rs["mean"])} {_fmt(rs["median"])} '
                f'{_fmt(rs["win"], "pct")} {_fmt(rs["sum"], "sum")}  | '
                f'{_fmt(rt["n"], "int")} {_fmt(rt["mean"])} {_fmt(rt["median"])} '
                f'{_fmt(rt["win"], "pct")} {_fmt(rt["sum"], "sum")}')
        print(line + m)

    sig_valid = sig_a.dropna(subset=['mean'])
    if len(sig_valid):
        best = sig_valid.loc[sig_valid['mean'].idxmax()]
        worst = sig_valid.loc[sig_valid['mean'].idxmin()]
        print(f'\n  全量 spread (最优 - 最差): {best["mean"] - worst["mean"]:.2f} pp  '
              f'(最优 {best["gua"]} {GUA_NAME[best["gua"]]} {best["mean"]:+.2f}%, '
              f'最差 {worst["gua"]} {GUA_NAME[worst["gua"]]} {worst["mean"]:+.2f}%)')


def main():
    print('\n  三因子 (大盘周卦 / 个股地卦 / 个股人卦) 对收益的影响 · 双视角\n')
    sig, trd = load_data()
    print(f'  全量 signal: {len(sig):,} 条 | 买入 trade: {len(trd):,} 条')
    print(f'  全量 signal 期望收益: {sig["actual_ret"].mean():+.2f}%  胜率 {(sig["actual_ret"]>0).mean()*100:.1f}%')
    print(f'  实际 trade 收益:    {trd["ret_pct"].mean():+.2f}%  胜率 {(trd["ret_pct"]>0).mean()*100:.1f}%')

    print_factor('因子 1 · 大盘周卦 m_gua (中周期市场状态)',     sig, trd, 'm_gua')
    print_factor('因子 2 · 个股地卦 di_gua (象卦, 个股自身)',     sig, trd, 'di_gua')
    print_factor('因子 3 · 个股人卦 ren_gua (人气, 资金氛围)',   sig, trd, 'ren_gua')


if __name__ == '__main__':
    main()
