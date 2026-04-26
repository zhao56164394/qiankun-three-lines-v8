# -*- coding: utf-8 -*-
"""个股级 3 卦对比: 个股日卦 vs 地卦 vs 人卦

评估谁更适合做选股信号, 在 "日卦分治 + 年卦过滤" 架构下作为
个股维度的分治/过滤变量.

数据源:
  - 个股日卦: stock_daily_gua.csv (新造, v10 规则)
  - 地卦:     stock_bagua_daily.csv (trend_anchor / speed / heat_momo)
  - 人卦:     daily_bagua_sequence.csv (五维评分压缩)

前瞻收益: daily_forward_returns.csv (ret_fwd_20d, 百分比形式)

评估指标 (重叠期):
  A. 基础: 样本、股数、8 态分布、平均段长
  B. Spearman IC (pos vs fwd_20; gua_code vs fwd_20)
  C. 阳/阴卦 (位爻) 前瞻收益差
  D. 8 态按 fwd_20 均值排序 → 强 4 / 弱 4 分组差距
  E. 段级方向命中率 (≥N 天段)
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


def load_fwd():
    df = pd.read_csv(os.path.join(FND, 'daily_forward_returns.csv'),
                     encoding='utf-8-sig',
                     usecols=['date', 'code', 'ret_fwd_20d'],
                     dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['fwd_20'] = df['ret_fwd_20d'] / 100.0  # 源是百分比
    return df[['date', 'code', 'fwd_20']]


def _normalize_gua(s):
    if pd.isna(s):
        return None
    x = str(s).strip()
    if x in ('', 'nan'):
        return None
    return x.zfill(3)


def load_stock_daily():
    df = pd.read_csv(os.path.join(FND, 'stock_daily_gua.csv'),
                     encoding='utf-8-sig',
                     usecols=['date', 'code', 'pos', 'gua_code'],
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['gua'] = df['gua_code'].apply(_normalize_gua)
    df['pos'] = df['pos'].astype(float).astype('Int64')
    return df[['date', 'code', 'gua', 'pos']].dropna(subset=['gua'])


def load_di_gua():
    df = pd.read_csv(os.path.join(FND, 'stock_bagua_daily.csv'),
                     encoding='utf-8-sig',
                     usecols=['date', 'code', 'gua_code', 'yao_1'],
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['gua'] = df['gua_code'].apply(_normalize_gua)
    df['pos'] = df['yao_1'].astype('Int64')
    return df[['date', 'code', 'gua', 'pos']].dropna(subset=['gua'])


def load_ren_gua():
    df = pd.read_csv(os.path.join(FND, 'daily_bagua_sequence.csv'),
                     encoding='utf-8-sig',
                     usecols=['date', 'code', 'gua_code'],
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['gua'] = df['gua_code'].apply(_normalize_gua)
    # 人卦没有 yao_1 单列, 用 gua_code 首位推
    df['pos'] = df['gua'].str[0].astype('Int64')
    return df[['date', 'code', 'gua', 'pos']].dropna(subset=['gua'])


def spearman(a, b):
    s = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(s) < 100:
        return np.nan
    return s['a'].rank().corr(s['b'].rank())


def analyze(df_gua, fwd, name, start_date=None, end_date=None):
    print(f'\n========== {name} ==========')

    if start_date is not None:
        df_gua = df_gua[df_gua['date'] >= start_date]
    if end_date is not None:
        df_gua = df_gua[df_gua['date'] <= end_date]

    # A. 基础
    print(f'--- A. 基础 ---')
    print(f'  观察数: {len(df_gua):,}   日期: {df_gua["date"].min().date()} ~ {df_gua["date"].max().date()}')
    print(f'  股票数: {df_gua["code"].nunique()}')

    dist = df_gua['gua'].value_counts(normalize=True).reindex(GUA_ORDER, fill_value=0) * 100
    ent = -sum((p / 100) * math.log2(p / 100) for p in dist.values if p > 0) / 3.0
    print(f'  8 态 (%): ' + ' '.join(f'{g}{GUA_NAME[g]}{dist[g]:.1f}' for g in GUA_ORDER))
    print(f'  熵比 {ent:.3f}   最小态 {dist.min():.2f}%   最大态 {dist.max():.2f}%')

    # 平均段长 (per stock)
    srt = df_gua.sort_values(['code', 'date'])
    chg = (srt['gua'] != srt.groupby('code')['gua'].shift()) | srt.groupby('code')['gua'].shift().isna()
    seg_id_global = chg.cumsum()
    seg_lens = seg_id_global.value_counts()
    print(f'  段数 {len(seg_lens):,}   平均段长 {seg_lens.mean():.1f} 天   中位 {seg_lens.median():.0f}')

    # 合并 fwd
    m = df_gua.merge(fwd, on=['date', 'code'], how='inner')
    m = m.dropna(subset=['fwd_20'])
    if len(m) == 0:
        print('  合并 fwd 后为空, 跳过后续')
        return None

    m['gua_int'] = m['gua'].map({g: i for i, g in enumerate(GUA_ORDER)})

    # B. IC
    ic_pos = spearman(m['pos'], m['fwd_20'])
    ic_gua = spearman(m['gua_int'], m['fwd_20'])
    print(f'\n--- B. Spearman IC vs fwd_20 ---')
    print(f'  位爻 IC     : {ic_pos:+.4f}')
    print(f'  gua_code IC : {ic_gua:+.4f}')

    # C. 阳/阴卦差距
    mu_up = m[m['pos'] == 1]['fwd_20'].mean() * 100
    mu_dn = m[m['pos'] == 0]['fwd_20'].mean() * 100
    n_up = (m['pos'] == 1).sum()
    n_dn = (m['pos'] == 0).sum()
    print(f'\n--- C. 阳/阴卦 fwd_20 ---')
    print(f'  位=1 (阳卦) n={n_up:>9,}  均值 {mu_up:+6.3f}%')
    print(f'  位=0 (阴卦) n={n_dn:>9,}  均值 {mu_dn:+6.3f}%')
    print(f'  差距 {mu_up - mu_dn:+6.3f}%')

    # D. 8 态排序 + 强 4/弱 4
    print(f'\n--- D. 8 态 fwd_20 均值 (%) ---')
    means = m.groupby('gua')['fwd_20'].agg(['mean', 'count']).reindex(GUA_ORDER)
    means['mean_pct'] = means['mean'] * 100
    srt8 = means.sort_values('mean_pct', ascending=False)
    for g, row in srt8.iterrows():
        print(f'  {g}{GUA_NAME[g]}: {row["mean_pct"]:+6.3f}%   (n={int(row["count"]):,})')

    top4 = list(srt8.head(4).index)
    bot4 = list(srt8.tail(4).index)
    top_ret = m[m['gua'].isin(top4)]['fwd_20'].mean() * 100
    bot_ret = m[m['gua'].isin(bot4)]['fwd_20'].mean() * 100
    top_win = (m[m['gua'].isin(top4)]['fwd_20'] > 0).mean() * 100
    bot_win = (m[m['gua'].isin(bot4)]['fwd_20'] > 0).mean() * 100
    print(f'  强 4 态: {" ".join(g + GUA_NAME[g] for g in top4)}  均 {top_ret:+6.3f}%  胜率 {top_win:.1f}%')
    print(f'  弱 4 态: {" ".join(g + GUA_NAME[g] for g in bot4)}  均 {bot_ret:+6.3f}%  胜率 {bot_win:.1f}%')
    print(f'  强-弱差距: {top_ret - bot_ret:+6.3f}%   胜率差 {top_win - bot_win:+5.1f}%')

    return {
        'name': name,
        'n': len(m),
        'ent': ent,
        'min_pct': dist.min(),
        'mean_seg': seg_lens.mean(),
        'ic_pos': ic_pos,
        'ic_gua': ic_gua,
        'spread_pos': mu_up - mu_dn,
        'spread_strong_weak': top_ret - bot_ret,
        'spread_win': top_win - bot_win,
    }


def main():
    print('> 加载前瞻收益 (daily_forward_returns.csv)...')
    fwd = load_fwd()
    print(f'  {len(fwd):,} 行')

    print('> 加载 个股日卦 (stock_daily_gua.csv)...')
    d = load_stock_daily()
    print(f'  {len(d):,} 行, {d["date"].min().date()} ~ {d["date"].max().date()}')

    print('> 加载 地卦 (stock_bagua_daily.csv)...')
    di = load_di_gua()
    print(f'  {len(di):,} 行, {di["date"].min().date()} ~ {di["date"].max().date()}')

    print('> 加载 人卦 (daily_bagua_sequence.csv)...')
    ren = load_ren_gua()
    print(f'  {len(ren):,} 行, {ren["date"].min().date()} ~ {ren["date"].max().date()}')

    # 重叠期: 3 者最晚起始
    start = max(d['date'].min(), di['date'].min(), ren['date'].min())
    end = min(d['date'].max(), di['date'].max(), ren['date'].max())
    print(f'\n>>> 重叠期: {start.date()} ~ {end.date()}\n')

    results = []
    results.append(analyze(d, fwd, '个股日卦 (v10)', start, end))
    results.append(analyze(di, fwd, '地卦', start, end))
    results.append(analyze(ren, fwd, '人卦', start, end))

    # 汇总
    print('\n\n========== 3 卦综合对比 ==========')
    hdr = f'{"卦":<14} {"观察":>11} {"熵比":>5} {"最小态%":>7} {"段长":>5} {"位IC":>8} {"guaIC":>8} {"阳-阴%":>7} {"强-弱%":>7} {"胜率差%":>8}'
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        if r is None:
            continue
        print(f'{r["name"]:<14} {r["n"]:>11,} {r["ent"]:>5.3f} '
              f'{r["min_pct"]:>6.2f}% {r["mean_seg"]:>5.1f} '
              f'{r["ic_pos"]:>+8.4f} {r["ic_gua"]:>+8.4f} '
              f'{r["spread_pos"]:>+6.3f} {r["spread_strong_weak"]:>+6.3f} {r["spread_win"]:>+7.1f}')


if __name__ == '__main__':
    main()
