# -*- coding: utf-8 -*-
"""日卦 vs 天卦 · 市场解释力/预测力对比

聚焦 "日卦分治 + 年卦过滤" 架构下, 日卦 和 天卦 谁更适合当分治变量.

评估口径 (重叠期 2014-06-24 ~ 最新):
  A. 基础统计     — 样本、段数、段长分布
  B. 同期解释力   — 位爻 vs 当日收益 Spearman
  C. 前瞻预测力   — 位爻/gua_code vs fwd_5/10/20 IC
  D. 段级方向命中 — ≥5/≥10 天段, 卦意(阳/阴) vs 段收益方向
  E. 牛熊周期契合 — 8 轮周期的阳卦%
  F. 分治信号质量 — 8 态按 fwd_20 排序 -> 强 4 态 vs 弱 4 态 收益差 + 胜率
"""
import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FND = os.path.join(ROOT, 'data_layer', 'data', 'foundation')
GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
            '100': '震', '101': '离', '110': '兑', '111': '乾'}

# 牛熊周期 (与 analyze_month_gua.py 一致)
CYCLES = [
    ('2014-06-24', '2015-06-30', '2014-06-2015-06 杠杆牛',   '牛'),
    ('2015-07-01', '2016-01-31', '2015-07-2016-01 股灾+熔断','熊'),
    ('2016-02-01', '2018-12-31', '2016-02-2018-12 慢牛转熊', '震'),
    ('2019-01-01', '2021-02-28', '2019-01-2021-02 结构牛',   '牛'),
    ('2021-03-01', '2024-02-28', '2021-03-2024-02 深熊',     '熊'),
    ('2024-03-01', '2026-04-24', '2024-03-      政策牛+震荡','牛'),
]


def load_index():
    idx = pd.read_csv(os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv'),
                      encoding='utf-8-sig', usecols=['date', 'close'])
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.sort_values('date').reset_index(drop=True)
    idx['ret_1'] = idx['close'].pct_change()
    for n in [5, 10, 20]:
        idx[f'fwd_{n}'] = idx['close'].shift(-n) / idx['close'] - 1
    return idx


def load_gua_daily():
    """日卦 (multi_scale_gua_daily.csv)"""
    fp = os.path.join(FND, 'multi_scale_gua_daily.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'd_gua', 'd_pos', 'd_spd', 'd_acc'],
                     dtype={'d_gua': str})
    df['date'] = pd.to_datetime(df['date'])
    df['gua'] = df['d_gua'].apply(lambda x: str(x).zfill(3) if pd.notna(x) else None)
    df['pos'] = df['d_pos']
    return df[['date', 'gua', 'pos']].dropna(subset=['gua'])


def load_gua_tian():
    """天卦 (market_bagua_daily.csv)"""
    fp = os.path.join(FND, 'market_bagua_daily.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'gua_code', 'yao_1'],
                     dtype={'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['gua'] = df['gua_code'].apply(lambda x: str(x).zfill(3) if pd.notna(x) else None)
    df['pos'] = df['yao_1']
    return df[['date', 'gua', 'pos']].dropna(subset=['gua'])


def spearman(a, b):
    """自写 spearman (rank 后 Pearson), 避开 scipy"""
    s = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(s) < 10:
        return np.nan
    return s['a'].rank().corr(s['b'].rank())


def block_A_basics(df, name):
    print(f'\n--- A. 基础统计 ({name}) ---')
    print(f'  观察数: {len(df):,}   起止: {df["date"].min().date()} ~ {df["date"].max().date()}')

    # 段长
    chg = df['gua'] != df['gua'].shift()
    seg_id = chg.cumsum()
    seg_lens = seg_id.value_counts().sort_index()
    print(f'  段数 {len(seg_lens)}   平均段长 {seg_lens.mean():.1f} 天   中位 {seg_lens.median():.0f}')
    le3 = (seg_lens <= 3).sum() / len(seg_lens) * 100
    ge10 = (seg_lens >= 10).sum() / len(seg_lens) * 100
    ge20 = (seg_lens >= 20).sum() / len(seg_lens) * 100
    print(f'  ≤3 天段 {le3:.1f}%   ≥10 天段 {ge10:.1f}%   ≥20 天段 {ge20:.1f}%')

    # 8 态分布
    dist = df['gua'].value_counts(normalize=True).reindex(GUA_ORDER, fill_value=0)
    ent = -sum(p * math.log2(p) for p in dist.values if p > 0) / 3.0
    print(f'  8 态分布 (%): ' + ' '.join(f'{g}{GUA_NAME[g]}{dist[g]*100:.1f}' for g in GUA_ORDER))
    print(f'  熵比 {ent:.3f}   最小态 {dist.min()*100:.2f}%   最大态 {dist.max()*100:.2f}%')


def block_B_contemporaneous(df, name):
    print(f'\n--- B. 同期解释力 ({name}) ---')
    # 位爻 vs 当日收益
    ic_pos_ret = spearman(df['pos'], df['ret_1'])
    # gua_code 顺序 vs 当日收益
    df = df.copy()
    df['gua_int'] = df['gua'].map({g: i for i, g in enumerate(GUA_ORDER)})
    ic_gua_ret = spearman(df['gua_int'], df['ret_1'])
    # 阳卦(位=1) 当日收益均值 vs 阴卦
    mean_up = df[df['pos'] == 1]['ret_1'].mean() * 100
    mean_dn = df[df['pos'] == 0]['ret_1'].mean() * 100
    print(f'  位爻 vs 当日 ret Spearman  : {ic_pos_ret:+.4f}')
    print(f'  gua_code vs 当日 ret Spear: {ic_gua_ret:+.4f}')
    print(f'  位=1 当日均涨 {mean_up:+.3f}%   位=0 当日均涨 {mean_dn:+.3f}%   差距 {mean_up-mean_dn:+.3f}%')


def block_C_predictive(df, name):
    print(f'\n--- C. 前瞻预测力 ({name}) ---')
    df = df.copy()
    df['gua_int'] = df['gua'].map({g: i for i, g in enumerate(GUA_ORDER)})
    print(f'  {"窗口":<6} {"位爻IC":>10} {"gua_codeIC":>12} {"位=1均%":>10} {"位=0均%":>10} {"差距%":>8}')
    for n in [5, 10, 20]:
        col = f'fwd_{n}'
        ic_pos = spearman(df['pos'], df[col])
        ic_gua = spearman(df['gua_int'], df[col])
        mu_up = df[df['pos'] == 1][col].mean() * 100
        mu_dn = df[df['pos'] == 0][col].mean() * 100
        print(f'  fwd_{n:<3} {ic_pos:>+10.4f} {ic_gua:>+12.4f} {mu_up:>+9.2f} {mu_dn:>+9.2f} {mu_up-mu_dn:>+7.2f}')


def block_D_segment_direction(df, name):
    print(f'\n--- D. 段级方向命中 ({name}) ---')
    # 提取段
    chg = df['gua'] != df['gua'].shift()
    seg_id = chg.cumsum()
    df = df.copy()
    df['seg_id'] = seg_id.values

    segs = []
    for sid, sub in df.groupby('seg_id', sort=True):
        if len(sub) < 2:
            continue
        d0, d1 = sub['date'].iloc[0], sub['date'].iloc[-1]
        c0, c1 = sub['close'].iloc[0], sub['close'].iloc[-1]
        gua = sub['gua'].iloc[0]
        segs.append({
            'gua': gua, 'pos': int(gua[0]),
            'days': len(sub),
            'ret': (c1 / c0 - 1),
        })
    sdf = pd.DataFrame(segs)
    print(f'  有效段数 {len(sdf)} (去掉单日段)')
    for thr in [3, 5, 10, 20]:
        big = sdf[sdf['days'] >= thr]
        if len(big) == 0:
            continue
        hit = ((big['pos'] == 1) & (big['ret'] > 0)) | ((big['pos'] == 0) & (big['ret'] <= 0))
        hit_rate = hit.sum() / len(big) * 100
        up_ret_mean = big[big['pos'] == 1]['ret'].mean() * 100
        dn_ret_mean = big[big['pos'] == 0]['ret'].mean() * 100
        print(f'  ≥{thr:>2}天段: n={len(big):>4}   卦意命中 {hit_rate:>5.1f}%   '
              f'阳卦段均收益 {up_ret_mean:+6.2f}%   阴卦段均收益 {dn_ret_mean:+6.2f}%')


def block_E_cycles(df, name):
    print(f'\n--- E. 牛熊周期契合 ({name}) ---')
    print(f'  {"周期":<36} {"天数":>5} {"阳卦%":>7} {"乾%":>5} {"坤%":>5} {"指数涨跌%":>10} {"真向":>4}')
    for s, e, label, truth in CYCLES:
        sub = df[(df['date'] >= s) & (df['date'] <= e)]
        if len(sub) == 0:
            continue
        up_pct = (sub['pos'] == 1).sum() / len(sub) * 100
        qian_pct = (sub['gua'] == '111').sum() / len(sub) * 100
        kun_pct = (sub['gua'] == '000').sum() / len(sub) * 100
        ret_tot = (sub['close'].iloc[-1] / sub['close'].iloc[0] - 1) * 100
        print(f'  {label:<36} {len(sub):>5} {up_pct:>6.1f}% {qian_pct:>4.1f}% {kun_pct:>4.1f}% {ret_tot:>+9.1f}% {truth:>4}')


def block_F_partition_quality(df, name):
    print(f'\n--- F. 分治信号质量 ({name}) ---')
    # 按 fwd_20 均值把 8 态排序, top4 / bot4 分组
    means = df.groupby('gua')['fwd_20'].mean().sort_values(ascending=False)
    means = means.reindex(GUA_ORDER).dropna()
    sorted_guas = means.sort_values(ascending=False)
    top4 = list(sorted_guas.head(4).index)
    bot4 = list(sorted_guas.tail(4).index)

    ord_str = ' > '.join(f'{g}{GUA_NAME[g]}({means[g]*100:+.2f})' for g in sorted_guas.index)
    print(f'  8 态按 fwd_20 排序: {ord_str}')
    print(f'  强 4 态 (top4): {" ".join(g+GUA_NAME[g] for g in top4)}')
    print(f'  弱 4 态 (bot4): {" ".join(g+GUA_NAME[g] for g in bot4)}')

    for n in [5, 10, 20]:
        col = f'fwd_{n}'
        top_ret = df[df['gua'].isin(top4)][col].mean() * 100
        bot_ret = df[df['gua'].isin(bot4)][col].mean() * 100
        top_win = (df[df['gua'].isin(top4)][col] > 0).mean() * 100
        bot_win = (df[df['gua'].isin(bot4)][col] > 0).mean() * 100
        print(f'  fwd_{n:<3} 强组均 {top_ret:+6.2f}%  弱组均 {bot_ret:+6.2f}%  差距 {top_ret-bot_ret:+6.2f}%   '
              f'强胜率 {top_win:.1f}%  弱胜率 {bot_win:.1f}%')


def analyze(gua_df, idx, name):
    # 对齐公共日期
    df = gua_df.merge(idx, on='date', how='inner')
    print(f'\n========== {name} ==========')
    block_A_basics(df, name)
    block_B_contemporaneous(df, name)
    block_C_predictive(df, name)
    block_D_segment_direction(df, name)
    block_E_cycles(df, name)
    block_F_partition_quality(df, name)
    return df


def main():
    idx = load_index()
    d_gua = load_gua_daily()
    t_gua = load_gua_tian()

    # 限定重叠期 (天卦从 2014-06-24 起)
    overlap_start = max(d_gua['date'].min(), t_gua['date'].min())
    overlap_end = min(d_gua['date'].max(), t_gua['date'].max())
    print(f'重叠期: {overlap_start.date()} ~ {overlap_end.date()}')

    d_gua = d_gua[(d_gua['date'] >= overlap_start) & (d_gua['date'] <= overlap_end)]
    t_gua = t_gua[(t_gua['date'] >= overlap_start) & (t_gua['date'] <= overlap_end)]

    analyze(d_gua, idx, '日卦 (d_gua)')
    analyze(t_gua, idx, '天卦 (tian_gua)')


if __name__ == '__main__':
    main()
