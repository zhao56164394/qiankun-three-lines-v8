# -*- coding: utf-8 -*-
"""坤 + 入池 v4 — 每日信号分布

入场条件 (5 道关卡):
  Gate 0: 入池 (d_trend 上穿11 → 下穿11)
  Gate 1: 大盘 y_gua = 000 坤
  Gate 2: 个股 d_gua = 011 巽 (出池)
  Gate 3: 强避雷 (股y=巽, 股m=乾 跳过)
  Gate 4: score (大m=震/大d=巽/大m=坎/股m=坎) >= 2

输出:
  1. 每日信号数分布 (有几天 1只 / 2只 / ... / 50+只)
  2. 同一天多只信号时, score 分布 (有没有"高 score 1只 + 低 score 一堆")
  3. 同一天多只信号的 ret 排序: 是不是高 score 真的赚得多
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
REGIME_Y = '000'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def main():
    t0 = time.time()
    print('=== 坤+入池 v4 每日信号分布 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
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

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    cross_arr = df['cross_below_11'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                # 强避雷
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue
                # score
                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                if score < 2:
                    in_pool = False
                    continue

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi],
                    'code': code_arr[gi],
                    'score': int(score),
                    'ret_30': ret_30,
                    'close': close_arr[gi],
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  总入场信号: {len(df_e):,}')
    print(f'  涉及天数: {df_e["date"].nunique():,}')

    # ============ 1. 每日信号数分布 ============
    print('\n=== 1. 每日信号数分布 ===')
    daily_count = df_e.groupby('date').size()
    print(f'  有信号日总数: {len(daily_count):,}')
    print(f'  分位:')
    for p in [25, 50, 75, 90, 95, 99]:
        print(f'    p{p}: {np.percentile(daily_count, p):.0f}')
    print(f'  最大: {daily_count.max()}')
    print(f'  最小: {daily_count.min()}')
    print(f'  平均: {daily_count.mean():.1f}')

    print(f'\n  信号数分桶 (有信号日里):')
    bins = [(1, 1, '1 只'), (2, 2, '2 只'), (3, 3, '3 只'), (4, 5, '4-5 只'),
            (6, 10, '6-10 只'), (11, 20, '11-20 只'), (21, 50, '21-50 只'),
            (51, 100, '51-100 只'), (101, 99999, '100+ 只')]
    for lo, hi, label in bins:
        cnt = ((daily_count >= lo) & (daily_count <= hi)).sum()
        if cnt > 0:
            pct = cnt / len(daily_count) * 100
            print(f'    {label:<14}: {cnt:>5} 天 ({pct:>5.1f}%)')

    # ============ 2. 多信号日, score 分布 ============
    print('\n=== 2. 多信号日 score 分布 ===')
    print('  (按当日信号数分组, 看每个 score 等级各占多少)')

    df_e['n_today'] = df_e['date'].map(daily_count)
    df_e['n_bucket'] = pd.cut(df_e['n_today'],
                                bins=[0, 1, 5, 10, 20, 50, 9999],
                                labels=['1只', '2-5只', '6-10只', '11-20只', '21-50只', '50+只'])

    print(f'  {"信号数":<10} {"事件":>8} {"score=2":>9} {"score=3":>9} {"score=4":>9} {"avg ret":>9} {"win%":>7}')
    for nb in ['1只', '2-5只', '6-10只', '11-20只', '21-50只', '50+只']:
        sub = df_e[df_e['n_bucket'] == nb]
        if len(sub) == 0: continue
        s2 = (sub['score']==2).sum()
        s3 = (sub['score']==3).sum()
        s4 = (sub['score']>=4).sum()
        avg = sub['ret_30'].mean()
        win = (sub['ret_30']>0).mean() * 100
        print(f'  {nb:<10} {len(sub):>8,} {s2:>9,} {s3:>9,} {s4:>9,} {avg:>+8.2f}% {win:>6.1f}%')

    # ============ 3. 同一天多信号, 按 score 排序的实测 ============
    print('\n=== 3. 同一天有多个信号时, 按 score+code 排序后 Top K 的表现 ===')
    print('  (每天按 score 降序, 同 score 按 code 字母序, 取每天 Top K)')

    # 按日期-score 排序
    df_sorted = df_e.sort_values(['date', 'score', 'code'],
                                   ascending=[True, False, True]).copy()
    df_sorted['rank_today'] = df_sorted.groupby('date').cumcount() + 1

    print(f'\n  {"取每天前K名":<16} {"事件":>8} {"avg ret":>9} {"win%":>7} {"avg score":>10}')
    for K in [1, 3, 5, 10, 20, 999]:
        sub = df_sorted[df_sorted['rank_today'] <= K]
        if K == 999:
            label = '全部信号 (基线)'
        else:
            label = f'前 {K} 只'
        avg = sub['ret_30'].mean()
        win = (sub['ret_30']>0).mean() * 100
        avg_sc = sub['score'].mean()
        print(f'  {label:<16} {len(sub):>8,} {avg:>+8.2f}% {win:>6.1f}% {avg_sc:>9.2f}')

    # ============ 4. 单日"高 score 笔" vs "低 score 笔" 实测 ============
    print('\n=== 4. 同一天 "前 N 名 (按 score)" vs "其余" 表现差 ===')
    print('  目的: 看 score 排序在每日内的真实区分度')
    for K in [1, 3, 5, 10]:
        top = df_sorted[df_sorted['rank_today'] <= K]
        rest = df_sorted[df_sorted['rank_today'] > K]
        if len(top) == 0 or len(rest) == 0: continue
        diff = top['ret_30'].mean() - rest['ret_30'].mean()
        print(f'  Top {K}: avg {top["ret_30"].mean():+.2f}% (n={len(top):,}) | '
              f'其余: avg {rest["ret_30"].mean():+.2f}% (n={len(rest):,}) | '
              f'差 {diff:+.2f}%')

    # ============ 5. 满仓时高 score 信号被错过的频率 ============
    print('\n=== 5. K=15 满仓的影响 (假设资金限制 K=15) ===')
    # 假设每日只能买 K=15 只 (资金限制), 模拟剩余信号期望
    K = 15
    # 假设最简化情况: 每天信号 > K 时, 后面的被错过
    over_K = df_sorted[df_sorted['rank_today'] > K]
    in_K = df_sorted[df_sorted['rank_today'] <= K]
    days_over_K = (daily_count > K).sum()
    print(f'  >K={K} 信号的天数: {days_over_K} ({days_over_K/len(daily_count)*100:.1f}%)')
    print(f'  错过的信号 (rank > {K}): n={len(over_K):,}')
    if len(over_K):
        print(f'  错过的信号 score 分布: {dict(over_K["score"].value_counts())}')
        print(f'  错过的信号 ret_30 期望: {over_K["ret_30"].mean():+.2f}% (vs Top {K}: {in_K["ret_30"].mean():+.2f}%)')

    # ============ 6. 极端日 (信号最多日) 详情 ============
    print('\n=== 6. 信号最多的 5 天详情 ===')
    top_days = daily_count.nlargest(5)
    for d, n in top_days.items():
        sub = df_e[df_e['date'] == d]
        print(f'\n  {d}: {n} 只信号')
        print(f'    score 分布: {dict(sub["score"].value_counts().sort_index())}')
        print(f'    平均 ret_30: {sub["ret_30"].mean():+.2f}%')
        print(f'    胜率: {(sub["ret_30"]>0).mean()*100:.1f}%')
        # 按 score 看 top 5
        sub_sorted = sub.sort_values(['score', 'code'], ascending=[False, True])
        print(f'    Top 5 (按 score+code):')
        for _, r in sub_sorted.head(5).iterrows():
            print(f'      {r["code"]} score={r["score"]} ret_30={r["ret_30"]:+.2f}% close={r["close"]:.2f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
