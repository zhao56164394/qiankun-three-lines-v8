# -*- coding: utf-8 -*-
"""诊断: 信号质量 vs 仓位数限制

3 个待解决问题:
  1. 同一天高质量的票筛选出来 — 需要更细的 score 或更强的排序
  2. 提高仓位数能否囊括高质量信号 — 看仓位 10/20/50 信号利用率
  3. 每日限买能否提高每日票的质量 — 看每日 cap 1/2/3 的高 score 命中率

不重新跑回测, 用 v2 入场扫描结果 (含所有 8 regime 信号)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
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
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date').reset_index(drop=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    # 算 future ret_30 (向前 30 个交易日)
    df['close_fwd_30'] = df.groupby('code', sort=False)['close'].shift(-30)
    df['ret_30'] = (df['close_fwd_30'] / df['close'] - 1) * 100

    # 给每行计算 regime 和 score (8 regime 全开, 跟 v2 一致)
    print('计算 regime 和 score (向量化)...')
    mkt_y = df['mkt_y'].values; mkt_d = df['mkt_d'].values; mkt_m = df['mkt_m'].values
    stk_d = df['stk_d'].values; stk_m = df['stk_m'].values; stk_y = df['stk_y'].values
    mf = df['main_force'].values; mf5 = df['mf_5d'].values; sh5 = df['sanhu_5d'].values

    df['regime'] = ''
    df['score'] = 0
    df['ok'] = False

    # 坤 v3: 巽日 + 9 避雷 + 4 项 score
    m = ((mkt_y == '000') & (stk_d == '011')
         & ~np.isin(stk_m, ['101', '110', '111'])
         & ~np.isin(stk_y, ['001', '011'])
         & ~np.isin(mkt_d, ['000', '001', '100', '101']))
    sc = ((mkt_m == '100').astype(int)
          + (mkt_d == '011').astype(int)
          + ((~np.isnan(mf)) & (mf > 100)).astype(int)
          + (stk_m == '010').astype(int))
    df.loc[m, 'regime'] = '坤 v3'
    df.loc[m, 'score'] = sc[m]
    df.loc[m, 'ok'] = True

    # 艮 v3: 巽日
    m = (mkt_y == '001') & (stk_d == '011')
    df.loc[m & ~df['ok'], 'regime'] = '艮 v3'
    df.loc[m & ~df['ok'], 'score'] = 1
    df.loc[m & ~df['ok'], 'ok'] = True

    # 坎 v3: 巽日 + 4 避雷 + 5 项 score
    m = ((mkt_y == '010') & (stk_d == '011')
         & ~np.isin(mkt_m, ['100', '110']) & (stk_y != '111') & (stk_m != '110'))
    sc = ((mkt_m == '011').astype(int)
          + (mkt_d == '001').astype(int)
          + ((~np.isnan(mf5)) & (mf5 < -50)).astype(int)
          + ((~np.isnan(mf)) & (mf > 100)).astype(int)
          + ((~np.isnan(sh5)) & (sh5 < -100)).astype(int))
    df.loc[m & ~df['ok'], 'regime'] = '坎 v3'
    df.loc[m & ~df['ok'], 'score'] = sc[m & ~df['ok'].values]
    df.loc[m & ~df['ok'], 'ok'] = True

    # 震 v1: 坎日
    m = ((mkt_y == '100') & (stk_d == '010')
         & ~np.isin(mkt_d, ['101', '111']) & (stk_y != '111'))
    sc = ((mkt_d == '011').astype(int) + (stk_m == '110').astype(int))
    df.loc[m & ~df['ok'], 'regime'] = '震 v1'
    df.loc[m & ~df['ok'], 'score'] = sc[m & ~df['ok'].values]
    df.loc[m & (df['regime'] == '震 v1') & (df['score'] >= 1), 'ok'] = True

    # 离 v1: 坤日
    m = ((mkt_y == '101') & (stk_d == '000')
         & (mkt_d != '101')
         & ~np.isin(stk_m, ['011', '001', '101'])
         & (stk_y != '011'))
    df.loc[m & ~df['ok'], 'regime'] = '离 v1'
    df.loc[m & ~df['ok'], 'score'] = 1
    df.loc[m & ~df['ok'], 'ok'] = True

    # 兑 v1: 坤日
    m = ((mkt_y == '110') & (stk_d == '000')
         & (mkt_d != '011')
         & ~np.isin(stk_m, ['001', '011', '101', '111']))
    df.loc[m & ~df['ok'], 'regime'] = '兑 v1'
    df.loc[m & ~df['ok'], 'score'] = 1
    df.loc[m & ~df['ok'], 'ok'] = True

    # 乾 v3: 巽日
    m = ((mkt_y == '111') & (stk_d == '011')
         & ~np.isin(mkt_d, ['100', '101', '110'])
         & (mkt_m != '101')
         & ~np.isin(stk_m, ['100', '101']))
    sc = ((stk_m == '010').astype(int) + (stk_y == '010').astype(int))
    df.loc[m & ~df['ok'], 'regime'] = '乾 v3'
    df.loc[m & ~df['ok'], 'score'] = sc[m & ~df['ok'].values]
    df.loc[m & (df['regime'] == '乾 v3') & (df['score'] >= 1), 'ok'] = True

    # 过滤掉非入场 (regime='' 或 score=0 当 score>=1 要求时)
    sigs = df[df['ok']].copy()
    sigs = sigs.dropna(subset=['ret_30'])  # 必须有未来 30 日
    print(f'  入场信号 (含 score=0 等待二次过滤): {len(sigs):,}')

    # 各 regime × score 实际 ret_30 期望 (这是文档的理论值)
    print('\n=== 1. 各 regime × score 真实 ret_30 期望 (vs 文档) ===')
    print(f'  {"regime":<10} {"score":>5} {"n":>8} {"胜率%":>7} {"ret_30%":>9} {"主升期":>9} {"假期":>9}')
    for r, sub_r in sigs.groupby('regime'):
        for sc, sub in sub_r.groupby('score'):
            if len(sub) < 100: continue
            zsl = sub[sub['ret_30'] >= 10]
            fake = sub[sub['ret_30'] < 10]
            print(f'  {r:<10} {sc:>5} {len(sub):>8,} {(sub["ret_30"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_30"].mean():>+8.2f} '
                  f'{zsl["ret_30"].mean() if len(zsl)>0 else 0:>+8.2f} '
                  f'{fake["ret_30"].mean() if len(fake)>0 else 0:>+8.2f}')

    # 每日信号数分布
    print('\n=== 2. 每日 (有信号日) 信号数分布 ===')
    daily = sigs.groupby('date').size()
    print(f'  有信号日: {len(daily):,}')
    print(f'  分位:')
    for p in [50, 75, 90, 95, 99]:
        print(f'    p{p}: {np.percentile(daily, p):.0f}')

    # 假设我们有不同的仓位上限 K = 5/10/20/50/200, 每天能买多少高 score?
    print('\n=== 3. 仓位上限 K 下的 max score 命中分布 ===')
    print(f'  (对每天的所有信号按 score 降序排序, 看 Top K 里多少是 score≥X)')
    for K in [5, 10, 20, 50, 200]:
        # 按日期分组取 Top K
        sub = sigs.sort_values('score', ascending=False).groupby('date').head(K)
        print(f'\n  K={K} (每日上限 {K} 只), 命中信号:')
        print(f'    总命中: {len(sub):,}')
        # 算这 K 个里 ret_30 期望 / 胜率
        ret_avg = sub['ret_30'].mean()
        win_rate = (sub['ret_30'] > 0).mean() * 100
        print(f'    ret_30 期望: {ret_avg:+.2f}% / 胜率 {win_rate:.1f}%')
        # 按 regime 分布
        print(f'    按 regime 分布:')
        rd = sub.groupby('regime').agg(n=('ret_30','count'), avg=('ret_30','mean'))
        for r, row in rd.iterrows():
            print(f'      {r:<10}: n={row["n"]:>7,.0f} ret_30={row["avg"]:>+5.2f}%')

    # 每天最高 score 的 regime
    print('\n=== 4. 各 regime 最高 score 期望 (这是真高质量) ===')
    print(f'  (排序后取每天 Top 1, 看是哪个 regime, ret_30 多少)')
    top1 = sigs.sort_values('score', ascending=False).groupby('date').head(1)
    print(f'  top1 命中数: {len(top1):,}')
    print(f'  top1 ret_30: {top1["ret_30"].mean():+.2f}% / 胜率 {(top1["ret_30"]>0).mean()*100:.1f}%')

    print(f'\n=== 5. 各 regime 在每日 max score 中的占比 ===')
    rd = top1.groupby('regime').size().sort_values(ascending=False)
    print(rd)

    # 关键: 如果只买 top 1 of each day, 总收益期望是?
    # 注: 这里看的是 ret_30, 真回测要 bull 卖
    print(f'\n=== 6. 不同 K 下, 资金理论上的"每日加权 30 日 ret" 期望 ===')
    print(f'  (假设无持仓占用, 每天买 K 只, 30 日后卖)')
    for K in [1, 3, 5, 10, 20]:
        sub = sigs.sort_values('score', ascending=False).groupby('date').head(K)
        n_per_day = sub.groupby('date').size().mean()
        per_day_avg_ret = sub.groupby('date')['ret_30'].mean().mean()
        print(f'  K={K}: 平均每天 {n_per_day:.1f} 只, 每日均 ret_30 {per_day_avg_ret:+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
