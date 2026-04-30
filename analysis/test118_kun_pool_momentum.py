# -*- coding: utf-8 -*-
"""坤+入池 v4 — 逆向找掐尖指标

核心假设错误: test117 用静态因子 (mf/sanhu/close) 的低位排序, 但其实那只是
"还没动" 的股, 不是 "正在动" 的股.

新思路 — 从结果倒推:
  1. 把买入后 ret_30 >= 15% 的股 (真主升浪股) 拎出来作为 "成功池"
  2. 把 ret_30 < 0 的股 (真亏损股) 拎出来作为 "失败池"
  3. 看 day0 (买入当日) 哪些指标在两池有显著差异

候选额外指标 (动能型):
  - trend_5d_change (5 日 trend 变化, 反弹力度)
  - mf_5d_change (5 日主力变化)
  - close_5d_change (5 日 close 变化, 实际涨幅)
  - close_3d_change
  - 价格相对 5d / 30d 高低位
  - 入池后 close 走势 (从入池到买点)
  - mf_to_sanhu_ratio (主力相对散户)
  - trend 加速度 (二阶导)

目标: 找一个"动能 z-score"能在每日内明确区分大涨 vs 亏损
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
    print('=== 坤+入池 v4 逆向找掐尖指标 ===\n')

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

    # 衍生指标
    grp = df.groupby('code', sort=False)
    df['mf_5d'] = grp['main_force'].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = grp['retail'].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df['trend_5d_ago'] = grp['d_trend'].shift(5)
    df['trend_3d_ago'] = grp['d_trend'].shift(3)
    df['mf_5d_ago'] = grp['main_force'].shift(5)
    df['close_5d_ago'] = grp['close'].shift(5)
    df['close_3d_ago'] = grp['close'].shift(3)
    df['close_30d_min'] = grp['close'].transform(lambda s: s.rolling(30, min_periods=10).min())
    df['t_prev'] = grp['d_trend'].shift(1)
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
    t5ago_arr = df['trend_5d_ago'].to_numpy().astype(np.float64)
    t3ago_arr = df['trend_3d_ago'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    mf5ago_arr = df['mf_5d_ago'].to_numpy().astype(np.float64)
    sh_arr = df['retail'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)
    c5ago_arr = df['close_5d_ago'].to_numpy().astype(np.float64)
    c3ago_arr = df['close_3d_ago'].to_numpy().astype(np.float64)
    c30min_arr = df['close_30d_min'].to_numpy().astype(np.float64)

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
        pool_enter_idx = -1
        pool_enter_close = -1

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True
                pool_enter_idx = i
                pool_enter_close = close_arr[gi]

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100

                # 动能指标
                trend_5d_chg = trend_arr[gi] - t5ago_arr[gi] if not np.isnan(t5ago_arr[gi]) else np.nan
                trend_3d_chg = trend_arr[gi] - t3ago_arr[gi] if not np.isnan(t3ago_arr[gi]) else np.nan
                mf_5d_chg = mf_arr[gi] - mf5ago_arr[gi] if not np.isnan(mf5ago_arr[gi]) else np.nan
                close_5d_chg = (close_arr[gi]/c5ago_arr[gi] - 1)*100 if not np.isnan(c5ago_arr[gi]) and c5ago_arr[gi] > 0 else np.nan
                close_3d_chg = (close_arr[gi]/c3ago_arr[gi] - 1)*100 if not np.isnan(c3ago_arr[gi]) and c3ago_arr[gi] > 0 else np.nan
                pool_close_chg = (close_arr[gi]/pool_enter_close - 1)*100 if pool_enter_close > 0 else np.nan
                close_to_30d_min = (close_arr[gi]/c30min_arr[gi] - 1)*100 if not np.isnan(c30min_arr[gi]) and c30min_arr[gi] > 0 else np.nan

                events.append({
                    'date': date_arr[gi],
                    'code': code_arr[gi],
                    'ret_30': ret_30,
                    'days_in_pool': i - pool_enter_idx,
                    # 动能型
                    'trend_5d_chg': trend_5d_chg,
                    'trend_3d_chg': trend_3d_chg,
                    'mf_5d_chg': mf_5d_chg,
                    'close_5d_chg': close_5d_chg,
                    'close_3d_chg': close_3d_chg,
                    'pool_close_chg': pool_close_chg,
                    'close_to_30d_min': close_to_30d_min,
                    # 静态
                    'trend': trend_arr[gi],
                    'mf': mf_arr[gi],
                    'mf_5d': mf5_arr[gi],
                    'sanhu': sh_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                    'close': close_arr[gi],
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}, 涉及 {df_e["date"].nunique()} 天')

    # ============ 1. 成功池 vs 失败池 直接对比 ============
    print('\n=== 1. 成功池 (ret>=15%) vs 失败池 (ret<0) 在 day0 指标的差异 ===\n')
    success = df_e[df_e['ret_30'] >= 15]
    fail = df_e[df_e['ret_30'] < 0]
    print(f'  成功池 n={len(success):,} ({len(success)/len(df_e)*100:.1f}%, 平均 ret {success["ret_30"].mean():+.1f}%)')
    print(f'  失败池 n={len(fail):,} ({len(fail)/len(df_e)*100:.1f}%, 平均 ret {fail["ret_30"].mean():+.1f}%)')

    metrics = ['days_in_pool', 'trend_5d_chg', 'trend_3d_chg', 'mf_5d_chg',
               'close_5d_chg', 'close_3d_chg', 'pool_close_chg', 'close_to_30d_min',
               'trend', 'mf', 'mf_5d', 'sanhu', 'sanhu_5d', 'close']

    print(f'\n  {"指标":<22} {"成功(中位)":>10} {"失败(中位)":>10} {"差":>8} {"成功(均)":>10} {"失败(均)":>10}')
    rows = []
    for m in metrics:
        s_med = success[m].median()
        f_med = fail[m].median()
        s_mean = success[m].mean()
        f_mean = fail[m].mean()
        diff_med = s_med - f_med
        rows.append((m, s_med, f_med, diff_med, s_mean, f_mean))
        print(f'  {m:<22} {s_med:>+9.2f} {f_med:>+9.2f} {diff_med:>+7.2f} {s_mean:>+9.2f} {f_mean:>+9.2f}')

    # 排 |差| 大的指标
    print('\n=== 2. 区分度最强的指标 (按 |中位数差|) ===')
    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    for m, s_med, f_med, d, s_mean, f_mean in rows[:7]:
        print(f'  [{d:+.2f}] {m}: 成功 {s_med:+.2f}, 失败 {f_med:+.2f}')

    # ============ 3. 用 day0 close_3d_chg 等动能因子分位 ============
    print('\n=== 3. close_3d_chg / close_5d_chg / mf_5d_chg 5 分位 ret 单调性 ===')
    daily_count = df_e.groupby('date').size()
    df_e['n_today'] = df_e['date'].map(daily_count)
    multi = df_e[df_e['n_today'] >= 5].copy()

    momentum_indicators = ['close_3d_chg', 'close_5d_chg', 'trend_3d_chg',
                            'trend_5d_chg', 'mf_5d_chg', 'pool_close_chg', 'close_to_30d_min']

    print(f'\n  {"指标":<22} {"q1 (低)":>9} {"q2":>8} {"q3":>8} {"q4":>8} {"q5 (高)":>9} {"q5-q1":>7}')
    rank_results = []
    for col in momentum_indicators:
        sub = multi.dropna(subset=[col]).copy()
        if len(sub) == 0: continue
        sub_sorted = sub.sort_values(['date', col, 'code']).copy()
        sub_sorted['rank_today'] = sub_sorted.groupby('date').cumcount()
        sub_sorted['n_per_day'] = sub_sorted.groupby('date')['date'].transform('count')
        sub_sorted['quintile'] = (sub_sorted['rank_today'] / sub_sorted['n_per_day'] * 5).astype(int).clip(0, 4)

        q_means = sub_sorted.groupby('quintile')['ret_30'].mean()
        if len(q_means) == 5:
            diff = q_means.iloc[4] - q_means.iloc[0]
            print(f'  {col:<22} {q_means.iloc[0]:>+8.2f}% {q_means.iloc[1]:>+7.2f}% '
                  f'{q_means.iloc[2]:>+7.2f}% {q_means.iloc[3]:>+7.2f}% '
                  f'{q_means.iloc[4]:>+8.2f}% {diff:>+6.2f}')
            rank_results.append((col, q_means.tolist(), diff))

    print('\n=== 4. 动能指标排名 (|q5-q1| 最大) ===')
    rank_results.sort(key=lambda x: abs(x[2]), reverse=True)
    for col, qs, diff in rank_results:
        sign = '高分位赚多 ✅' if diff > 0 else '低分位赚多'
        print(f'  [{diff:+.2f}%] {col}: {sign}')

    # ============ 5. 多因子加权重新算 quality, 测每日 Top1 ============
    print('\n=== 5. 综合动能 quality (强单调因子加权) ===')
    strong_mom = [(c, qs, d) for c, qs, d in rank_results if abs(d) > 1.0]
    if len(strong_mom) >= 1:
        # z-score 加权
        for col, qs, d in strong_mom:
            sign = 1 if d > 0 else -1
            print(f'  {col}: weight = {sign} (|q5-q1| = {abs(d):.2f}%)')

        # 给 df_e 算综合 quality (跨整个 df_e, 用日度组内 z-score)
        z_total = np.zeros(len(df_e))
        z_valid = np.zeros(len(df_e), dtype=bool)
        for col, qs, d in strong_mom:
            sign = 1 if d > 0 else -1
            grp = df_e.groupby('date')[col]
            mu = grp.transform('mean'); sigma = grp.transform('std')
            z = (df_e[col] - mu) / (sigma + 1e-9)
            z_filled = z.fillna(0).values
            z_total += z_filled * sign

        df_e['mom_quality'] = z_total
        df_sorted = df_e.sort_values(['date', 'mom_quality', 'code'],
                                       ascending=[True, False, True]).copy()
        df_sorted['rank_today'] = df_sorted.groupby('date').cumcount() + 1

        print(f'\n  {"取每天前K名":<18} {"事件":>8} {"avg ret":>9} {"win%":>7}')
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
