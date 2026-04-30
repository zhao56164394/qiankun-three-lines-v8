# -*- coding: utf-8 -*-
"""扫 8 regime 每日 buy signal 数量, 决定仓位机制

8 regime 入场 gates 复刻 (按 strategy_*.md):
  坤 v3 (000): 巽日 + 9 避雷 + score≥2
  艮 v3 (001): 巽日 + 1 避雷 (单独最稳)
  坎 v3 (010): 巽日 + 4 避雷 + score≥2
  巽    (011): 不买 (case study)
  震 v1 (100): 坎日 + 3 弱避雷 + score≥1
  离 v1 (101): 坤日 + 5 避雷
  兑 v1 (110): 坤日 + 5 避雷
  乾 v3 (111): 巽日 + 6 卦避雷 + 涨幅避雷 + score≥1

输出:
  按日期: 触发了哪个 regime + 当日多少只股票 buy signal
  按 regime: 信号总数 / 平均每日 / 最多每日
"""
import os, sys, io, time
import numpy as np
import pandas as pd
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    # 不需要 close (这里只数信号)
    df = g.merge(market, on='date', how='left')
    df = df.sort_values(['date', 'code']).reset_index(drop=True)
    df = df.dropna(subset=['stk_d', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    # 按 regime 算入场 mask
    print('\n=== 按 regime 计算入场 mask ===')

    mkt_y = df['mkt_y'].values
    mkt_d = df['mkt_d'].values
    mkt_m = df['mkt_m'].values
    stk_d = df['stk_d'].values
    stk_m = df['stk_m'].values
    stk_y = df['stk_y'].values

    # 各 regime 的 mask
    masks = {}

    # 坤 v3 (000): 巽日 + 9 避雷 + score≥2
    # 9 避雷: 股m=离/兑/乾, 股y=艮/巽, 大d=坤/艮/震/离
    # 4 真好: 大m=震 / 大d=巽 / 个股 m=坎 / mf>100 (mf 这里没数据, 简化为前 3)
    m_kun = (mkt_y == '000') & (stk_d == '011')
    m_kun = m_kun & ~np.isin(stk_m, ['101', '110', '111'])
    m_kun = m_kun & ~np.isin(stk_y, ['001', '011'])
    m_kun = m_kun & ~np.isin(mkt_d, ['000', '001', '100', '101'])
    score_kun = ((mkt_m == '100').astype(int)
                 + (mkt_d == '011').astype(int)
                 + (stk_m == '010').astype(int))
    m_kun = m_kun & (score_kun >= 2)
    masks['坤 v3'] = m_kun

    # 艮 v3 (001): 巽日 + 单一避雷 (这里简化用核心方案, 文档 v3 是 1 项避雷)
    m_gen = (mkt_y == '001') & (stk_d == '011')
    masks['艮 v3'] = m_gen

    # 坎 v3 (010): 巽日 + 4 避雷 + score≥2
    # (按文档简化: 4 避雷不全, 暂用 baseline + 主信号)
    m_kan = (mkt_y == '010') & (stk_d == '011')
    masks['坎 v3'] = m_kan

    # 巽 (011): 不买
    masks['巽 (不买)'] = np.zeros(len(df), dtype=bool)

    # 震 v1 (100): 坎日 + 3 弱避雷 + score≥1
    m_zhen = (mkt_y == '100') & (stk_d == '010')
    m_zhen = m_zhen & ~np.isin(mkt_d, ['101', '111'])
    m_zhen = m_zhen & (stk_y != '111')
    score_zhen = ((mkt_d == '011').astype(int) + (stk_m == '110').astype(int))
    m_zhen = m_zhen & (score_zhen >= 1)
    masks['震 v1'] = m_zhen

    # 离 v1 (101): 坤日 + 5 避雷
    m_li = (mkt_y == '101') & (stk_d == '000')
    m_li = m_li & (mkt_d != '101')
    m_li = m_li & ~np.isin(stk_m, ['011', '001', '101'])
    m_li = m_li & (stk_y != '011')
    masks['离 v1'] = m_li

    # 兑 v1 (110): 坤日 + 5 避雷
    m_dui = (mkt_y == '110') & (stk_d == '000')
    m_dui = m_dui & (mkt_d != '011')
    m_dui = m_dui & ~np.isin(stk_m, ['001', '011', '101', '111'])
    masks['兑 v1'] = m_dui

    # 乾 v3 (111): 巽日 + 6 卦避雷 + 涨幅避雷 (这里没 close 跳过涨幅) + score≥1
    m_qian = (mkt_y == '111') & (stk_d == '011')
    m_qian = m_qian & ~np.isin(mkt_d, ['100', '101', '110'])
    m_qian = m_qian & (mkt_m != '101')
    m_qian = m_qian & ~np.isin(stk_m, ['100', '101'])
    score_qian = ((stk_m == '010').astype(int) + (stk_y == '010').astype(int))
    m_qian = m_qian & (score_qian >= 1)
    masks['乾 v3'] = m_qian

    # 汇总
    print(f'\n## 按 regime 汇总 (2014-2026)')
    print(f'  {"regime":<12} {"总信号":>9} {"出现日数":>9} {"平均/日":>9} {"最多/日":>9}')
    total_per_day = np.zeros(df['date'].nunique())
    df_dates = df['date'].values
    unique_dates = np.unique(df_dates)
    date_idx = {d: i for i, d in enumerate(unique_dates)}

    regime_summary = {}
    for name, mask in masks.items():
        sigs_per_day = pd.Series(mask, index=df['date']).groupby(level=0).sum()
        total = mask.sum()
        active_days = (sigs_per_day > 0).sum()
        if active_days > 0:
            avg = total / active_days
            mx = sigs_per_day.max()
        else:
            avg = 0; mx = 0
        regime_summary[name] = {'total': total, 'active_days': active_days,
                                'avg': avg, 'max': mx, 'sigs_per_day': sigs_per_day}
        print(f'  {name:<12} {total:>9,} {active_days:>9,} {avg:>9.2f} {mx:>9}')

    # 8 regime 加总每日信号数
    print(f'\n## 全部 regime 合并每日信号数分布')
    all_sigs = pd.Series(0, index=unique_dates, dtype=int)
    for name, mask in masks.items():
        if mask.sum() > 0:
            sub = pd.Series(mask.astype(int), index=df['date']).groupby(level=0).sum()
            all_sigs = all_sigs.add(sub, fill_value=0)
    all_sigs = all_sigs.astype(int)

    # 仅看有信号的日子
    sig_days = all_sigs[all_sigs > 0]
    print(f'  有信号日数: {len(sig_days):,} / 总日数 {len(unique_dates):,} ({len(sig_days)/len(unique_dates)*100:.1f}%)')
    print(f'  总信号数: {sig_days.sum():,}')
    print(f'  平均/日 (有信号): {sig_days.mean():.2f}')
    print(f'  分位数:')
    for p in [50, 75, 90, 95, 99, 100]:
        v = np.percentile(sig_days, p)
        print(f'    p{p:>3}: {v:>6.0f}')

    # 信号数量分桶
    print(f'\n  按每日信号数分桶:')
    bins = [(0, 1, '0只 (无信号日)'), (1, 5, '1-4只'), (5, 10, '5-9只'),
            (10, 20, '10-19只'), (20, 50, '20-49只'), (50, 100, '50-99只'),
            (100, 500, '100-499只'), (500, 99999, '500+ 只')]
    no_sig_days = (all_sigs == 0).sum()
    for lo, hi, label in bins:
        if lo == 0:
            cnt = no_sig_days
        else:
            cnt = ((all_sigs >= lo) & (all_sigs < hi)).sum()
        pct = cnt / len(unique_dates) * 100
        print(f'    {label:<22} {cnt:>5} 天 ({pct:>5.1f}%)')

    # 每个 regime 出现频率
    print(f'\n## 各 regime y_gua 在历史中的出现频率 (按交易日)')
    mkt_y_per_day = market.set_index('date')['mkt_y'].to_dict()
    y_counts = Counter(mkt_y_per_day.get(d, '') for d in unique_dates)
    GUA_NAMES_Y = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
                   '100': '震', '101': '离', '110': '兑', '111': '乾', '': 'NA'}
    print(f'  {"y_gua":<8} {"日数":>6} {"占比%":>7}')
    for y, cnt in sorted(y_counts.items(), key=lambda x: -x[1]):
        name = GUA_NAMES_Y.get(y, '?')
        print(f'  {y}{name:<6} {cnt:>6} {cnt/len(unique_dates)*100:>6.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
