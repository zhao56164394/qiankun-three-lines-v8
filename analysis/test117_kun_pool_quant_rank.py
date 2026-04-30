# -*- coding: utf-8 -*-
"""坤+入池 v4 — 个股量化排名 (替代 score)

score 是群体分类 (0/1/2/3/4), 同分时无法区分.
现在用纯连续指标做个股排名:

候选量化指标 (基于业务直觉):
  A. d_trend (趋势线绝对值) — 越低越底部
  B. d_trend_5d_min — 最近 5 日 trend 最低值
  C. d_trend_30d_min — 最近 30 日 trend 最低值 (绝对底部)
  D. mf (主力线当日) — 越高越主力建仓
  E. mf_5d (主力 5 日均) — 平稳建仓
  F. sanhu (散户当日) — 越低散户走得越干净
  G. sanhu_5d — 平稳低位
  H. close (绝对低价) — 低价股黑马多
  I. days_in_pool — 伏蛰天数

测试方法:
  对每个量化指标 X, 在每日内按 X 排序
  分位 Top 10% / 20% vs Bottom 10% / 20%
  看 ret_30 差异是否单调 (高分位真好/低分位真差)

如果某个指标 Top 10% 比 Bottom 10% 差 5+%, 那就是掐尖利器.
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


def main():
    t0 = time.time()
    print('=== 坤+入池 v4 个股量化排名分析 ===\n')

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
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)

    # 算多种衍生指标
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['mf_30d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(30, min_periods=15).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_30d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(30, min_periods=15).mean())
    df['trend_5d_min'] = df.groupby('code', sort=False)['d_trend'].transform(
        lambda s: s.rolling(5, min_periods=3).min())
    df['trend_30d_min'] = df.groupby('code', sort=False)['d_trend'].transform(
        lambda s: s.rolling(30, min_periods=10).min())
    df['close_5d_min'] = df.groupby('code', sort=False)['close'].transform(
        lambda s: s.rolling(5, min_periods=3).min())
    # 当前价相对 30d 最低价的位置 (0 = 在最低, 1 = 高很多)
    df['close_30d_min'] = df.groupby('code', sort=False)['close'].transform(
        lambda s: s.rolling(30, min_periods=10).min())
    df['close_30d_max'] = df.groupby('code', sort=False)['close'].transform(
        lambda s: s.rolling(30, min_periods=10).max())
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
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    mf30_arr = df['mf_30d'].to_numpy().astype(np.float64)
    sh_arr = df['retail'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)
    sh30_arr = df['sanhu_30d'].to_numpy().astype(np.float64)
    t5min_arr = df['trend_5d_min'].to_numpy().astype(np.float64)
    t30min_arr = df['trend_30d_min'].to_numpy().astype(np.float64)
    c30min_arr = df['close_30d_min'].to_numpy().astype(np.float64)
    c30max_arr = df['close_30d_max'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件 (在池+坤+巽+避雷, 不限 score)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False
        pool_enter_idx = -1

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True
                pool_enter_idx = i

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100

                # 价格在 30 日波动中的相对位置 (0 = 在最低, 1 = 在最高)
                cmin = c30min_arr[gi]; cmax = c30max_arr[gi]
                close_pos_30d = (close_arr[gi] - cmin) / (cmax - cmin + 1e-9) if cmax > cmin else 0.5

                events.append({
                    'date': date_arr[gi],
                    'code': code_arr[gi],
                    'ret_30': ret_30,
                    'days_in_pool': i - pool_enter_idx,
                    'trend': trend_arr[gi],
                    'trend_5d_min': t5min_arr[gi],
                    'trend_30d_min': t30min_arr[gi],
                    'mf': mf_arr[gi],
                    'mf_5d': mf5_arr[gi],
                    'mf_30d': mf30_arr[gi],
                    'sanhu': sh_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                    'sanhu_30d': sh30_arr[gi],
                    'close': close_arr[gi],
                    'close_pos_30d': close_pos_30d,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}, 涉及 {df_e["date"].nunique()} 天')

    daily_count = df_e.groupby('date').size()
    df_e['n_today'] = df_e['date'].map(daily_count)

    multi = df_e[df_e['n_today'] >= 5].copy()
    print(f'  多信号日 (>=5 只) 的事件: {len(multi):,}\n')

    # ============ 各量化指标按 5 分位的 ret 单调性 ============
    print('=== 1. 单一量化指标 5 分位 ret 单调性 ===\n')
    print('  (在每日内对该指标分位, 看 q1 vs q5 ret 差异)\n')

    indicators = [
        'days_in_pool', 'trend', 'trend_5d_min', 'trend_30d_min',
        'mf', 'mf_5d', 'mf_30d',
        'sanhu', 'sanhu_5d', 'sanhu_30d',
        'close', 'close_pos_30d',
    ]

    print(f'  {"指标":<18} {"q1 (低)":>9} {"q2":>8} {"q3":>8} {"q4":>8} {"q5 (高)":>9} {"q5-q1":>7}')
    rank_results = []
    for col in indicators:
        # 用每日的分位 (5 分位)
        multi_sorted = multi.sort_values(['date', col, 'code']).copy()
        multi_sorted['rank_today'] = multi_sorted.groupby('date').cumcount()
        multi_sorted['n_per_day'] = multi_sorted.groupby('date')['date'].transform('count')
        multi_sorted['quintile'] = (multi_sorted['rank_today'] / multi_sorted['n_per_day'] * 5).astype(int)
        multi_sorted['quintile'] = multi_sorted['quintile'].clip(0, 4)

        q_means = multi_sorted.groupby('quintile')['ret_30'].mean()
        if len(q_means) == 5:
            diff = q_means.iloc[4] - q_means.iloc[0]
            print(f'  {col:<18} {q_means.iloc[0]:>+8.2f}% {q_means.iloc[1]:>+7.2f}% '
                  f'{q_means.iloc[2]:>+7.2f}% {q_means.iloc[3]:>+7.2f}% '
                  f'{q_means.iloc[4]:>+8.2f}% {diff:>+6.2f}')
            rank_results.append((col, q_means.tolist(), diff))

    print('\n=== 2. 单调性最强的指标 (|q5-q1| 最大) ===')
    rank_results.sort(key=lambda x: abs(x[2]), reverse=True)
    for col, qs, diff in rank_results[:5]:
        sign = '高分位赚多' if diff > 0 else '低分位赚多'
        print(f'  [{diff:+.2f}%] {col}: {sign}')
        print(f'    q1={qs[0]:+.2f}, q2={qs[1]:+.2f}, q3={qs[2]:+.2f}, q4={qs[3]:+.2f}, q5={qs[4]:+.2f}')

    # ============ 组合一个加权 quality 分 ============
    # 基于上面找到的最有区分度指标, 标准化后加权
    print('\n=== 3. 综合 quality 分 (z-score 加权) ===')
    print('  规则:')
    print('    - 找出每日内 ret 最高的 quintile, 用对应方向作权重')

    # 用每日内 z-score 标准化每个指标
    factors = {}
    for col in indicators:
        # 用每日组内的 z-score
        multi_grp = multi.groupby('date')[col]
        z = (multi[col] - multi_grp.transform('mean')) / (multi_grp.transform('std') + 1e-9)
        factors[col] = z.values

    # 用 q5-q1 的方向作权重 (正 = 高分位赚多用 +1, 负用 -1)
    # 但考虑只用强单调指标
    strong_indicators = [(col, diff) for col, qs, diff in rank_results if abs(diff) > 1.0]
    print(f'\n  强单调指标 (|q5-q1| > 1%):')
    for col, diff in strong_indicators:
        print(f'    {col}: {"+1" if diff > 0 else "-1"} ({diff:+.2f}%)')

    if len(strong_indicators) >= 2:
        # 综合 quality = 各强单调指标 z-score 加权和
        z_total = np.zeros(len(multi))
        for col, diff in strong_indicators:
            sign = 1 if diff > 0 else -1
            z_total += factors[col] * sign

        multi['quality'] = z_total
        # 按 quality 分位
        multi_sorted = multi.sort_values(['date', 'quality', 'code'],
                                            ascending=[True, False, True]).copy()
        multi_sorted['rank_today'] = multi_sorted.groupby('date').cumcount()
        multi_sorted['n_per_day'] = multi_sorted.groupby('date')['date'].transform('count')
        multi_sorted['quintile'] = (multi_sorted['rank_today'] / multi_sorted['n_per_day'] * 5).astype(int)
        multi_sorted['quintile'] = multi_sorted['quintile'].clip(0, 4)

        print(f'\n  综合 quality 5 分位 ret:')
        q_means = multi_sorted.groupby('quintile')['ret_30'].mean()
        q_wins = multi_sorted.groupby('quintile')['ret_30'].apply(lambda x: (x > 0).mean() * 100)
        for q in range(5):
            print(f'    q{q+1}: ret {q_means.iloc[q]:+.2f}% win {q_wins.iloc[q]:.1f}% n={(multi_sorted["quintile"]==q).sum()}')

        # Top 1 / Top 3 / Top 5 表现 (含单信号日)
        print('\n  全样本 Top K 表现 (用综合 quality 排序):')
        # 给单信号日也算 quality (z=0)
        df_e['quality'] = 0.0
        for col, diff in strong_indicators:
            grp = df_e.groupby('date')[col]
            z = (df_e[col] - grp.transform('mean')) / (grp.transform('std') + 1e-9)
            sign = 1 if diff > 0 else -1
            df_e['quality'] = df_e['quality'] + z.fillna(0) * sign

        df_sorted = df_e.sort_values(['date', 'quality', 'code'],
                                       ascending=[True, False, True]).copy()
        df_sorted['rank_today'] = df_sorted.groupby('date').cumcount() + 1

        print(f'  {"取每天前K名":<18} {"事件":>8} {"avg ret":>9} {"win%":>7}')
        for K in [1, 3, 5, 10, 999]:
            sub = df_sorted[df_sorted['rank_today'] <= K]
            if K == 999: label = '全部 (基线)'
            else: label = f'前 {K} 只'
            avg = sub['ret_30'].mean()
            win = (sub['ret_30']>0).mean() * 100
            print(f'  {label:<18} {len(sub):>8,} {avg:>+8.2f}% {win:>6.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
