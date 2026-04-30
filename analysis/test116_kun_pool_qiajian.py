# -*- coding: utf-8 -*-
"""坤+入池 v4 — 同一天多信号"掐尖"诊断

每天只买 1 只 → 必须找一个能在 score 同分时区分高低的"二级指标"

测 7 个候选二级指标, 看哪个在每日内排序后, Top 1 表现明显高于其余:
  1. 入池后伏蛰天数 (days_in_pool)
  2. trend_at_buy (买入日 trend 高低)
  3. mf_at_buy (买入日主力线)
  4. mf_5d (5 日主力均)
  5. sanhu_at_buy (买入日散户线)
  6. sanhu_5d
  7. close (买入价低高)
  8. 当日信号总数 (n_today, 拥挤度反向用)

判定方法:
  对每个二级指标, 按其值升/降序在每日内排序
  取 Top 1 vs 其余, 看 ret_30 差异
  差异大 = 该指标可以分高下
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
    print('=== 坤+入池 v4 同日多信号 掐尖诊断 ===\n')

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
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
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
    sh_arr = df['retail'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件 (在池+坤+巽+避雷+score>=2)...')
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
                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                if score < 2:
                    in_pool = False
                    continue

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                # 算入池后跌幅 (close 从入池日到买点的变化)
                pool_close_change = (close_arr[gi] / close_arr[s+pool_enter_idx] - 1) * 100 if pool_enter_idx >= 0 else 0
                events.append({
                    'date': date_arr[gi],
                    'code': code_arr[gi],
                    'score': int(score),
                    'ret_30': ret_30,
                    'days_in_pool': i - pool_enter_idx,
                    'trend_at_buy': trend_arr[gi],
                    'mf_at_buy': mf_arr[gi],
                    'mf_5d': mf5_arr[gi],
                    'sanhu_at_buy': sh_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                    'close': close_arr[gi],
                    'pool_close_change': pool_close_change,
                })
                in_pool = False
                pool_enter_idx = -1

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}, 涉及 {df_e["date"].nunique()} 天')

    daily_count = df_e.groupby('date').size()
    df_e['n_today'] = df_e['date'].map(daily_count)

    # 只看多信号日 (n_today >= 3, 真有"掐尖"问题)
    multi = df_e[df_e['n_today'] >= 3].copy()
    print(f'  多信号日 (>=3 只) 的事件: {len(multi):,} ({len(multi)/len(df_e)*100:.1f}%)')

    # ============ 分析每个二级指标的区分度 ============
    print('\n=== 同日多信号: 各二级指标 Top1 vs 其余 ===\n')
    print('  (仅看 n_today >= 3 的日子)\n')

    indicators = [
        ('score', '降', 'score 高'),
        ('days_in_pool', '降', '伏蛰久'),
        ('days_in_pool', '升', '伏蛰短 (反向)'),
        ('trend_at_buy', '升', 'trend 低'),
        ('trend_at_buy', '降', 'trend 高 (反向)'),
        ('mf_at_buy', '降', '主力当日强'),
        ('mf_at_buy', '升', '主力当日弱 (反向)'),
        ('mf_5d', '升', 'mf 5 日均低'),
        ('mf_5d', '降', 'mf 5 日均高 (反向)'),
        ('sanhu_at_buy', '升', '散户当日低'),
        ('sanhu_5d', '升', '散户 5 日均低'),
        ('sanhu_5d', '降', '散户 5 日均高 (反向)'),
        ('close', '升', '低价股'),
        ('close', '降', '高价股 (反向)'),
        ('pool_close_change', '升', '入池后跌多 (近底)'),
        ('pool_close_change', '降', '入池后跌少/反弹 (远底)'),
    ]

    print(f'  {"指标":<25} {"方向":<6} {"Top1 ret":>9} {"其余 ret":>9} {"差":>7} {"评价":<8}')
    results = []
    for col, direction, label in indicators:
        ascending = (direction == '升')
        # 同分破并: 用 code 作 tiebreaker (固定)
        sorted_df = multi.sort_values(['date', col, 'code'],
                                        ascending=[True, ascending, True]).copy()
        sorted_df['rank_today'] = sorted_df.groupby('date').cumcount() + 1

        top1 = sorted_df[sorted_df['rank_today'] == 1]
        rest = sorted_df[sorted_df['rank_today'] > 1]
        if len(top1) == 0 or len(rest) == 0:
            continue
        diff = top1['ret_30'].mean() - rest['ret_30'].mean()
        evaluation = '✅ 强' if diff > 1.5 else ('○' if diff > 0 else '❌')
        results.append((label, col, direction, top1['ret_30'].mean(), rest['ret_30'].mean(), diff))
        print(f'  {label:<23} {direction:<5} {top1["ret_30"].mean():>+8.2f}% {rest["ret_30"].mean():>+8.2f}% '
              f'{diff:>+6.2f} {evaluation}')

    # 排序输出
    results.sort(key=lambda x: x[5], reverse=True)
    print('\n=== Top 5 最佳掐尖指标 ===')
    for r in results[:5]:
        label, col, direction, t1, rs, diff = r
        print(f'  [{diff:+.2f}%] {label} (col={col}, {direction}序)')

    # ============ 双指标组合 ============
    print('\n=== 双指标组合 (主排序 + 二级 tiebreaker) ===\n')
    # 先按 score 降序, 同 score 内按某个二级指标排
    secondary_candidates = ['days_in_pool', 'trend_at_buy', 'mf_at_buy', 'mf_5d',
                              'sanhu_5d', 'close', 'pool_close_change']

    print(f'  {"主":<8} {"次":<22} {"方向":<6} {"Top1 ret":>9} {"其余 ret":>9} {"差":>7}')
    for sec in secondary_candidates:
        for asc in [True, False]:
            direction = '升' if asc else '降'
            sorted_df = multi.sort_values(['date', 'score', sec, 'code'],
                                            ascending=[True, False, asc, True]).copy()
            sorted_df['rank_today'] = sorted_df.groupby('date').cumcount() + 1
            top1 = sorted_df[sorted_df['rank_today'] == 1]
            rest = sorted_df[sorted_df['rank_today'] > 1]
            if len(top1) and len(rest):
                diff = top1['ret_30'].mean() - rest['ret_30'].mean()
                print(f'  score↓   {sec:<22} {direction:<5} {top1["ret_30"].mean():>+8.2f}% '
                      f'{rest["ret_30"].mean():>+8.2f}% {diff:>+6.2f}')

    # ============ Top1 / Top3 / Top5 全样本 (含单信号日) ============
    print('\n=== 全样本 (含单信号日) 几种排序的 Top1 表现 ===')
    print('  (单信号日 Top1 = 唯一信号; 多信号日 Top1 = 排序第 1)\n')

    # 准备几个候选排序
    cand_orders = [
        ('score↓ + days_in_pool↑', ['score', 'days_in_pool', 'code'], [False, True, True]),
        ('score↓ + trend↑',         ['score', 'trend_at_buy', 'code'], [False, True, True]),
        ('score↓ + mf↓',            ['score', 'mf_at_buy', 'code'], [False, False, True]),
        ('score↓ + mf_5d↑',         ['score', 'mf_5d', 'code'], [False, True, True]),
        ('score↓ + close↑',         ['score', 'close', 'code'], [False, True, True]),
        ('score↓ + pool_chg↑',      ['score', 'pool_close_change', 'code'], [False, True, True]),
    ]

    print(f'  {"排序":<28} {"Top1 n":>6} {"Top1 ret":>9} {"win%":>7}')
    for label, cols, ascs in cand_orders:
        sorted_df = df_e.sort_values(['date'] + cols,
                                       ascending=[True] + ascs).copy()
        sorted_df['rank_today'] = sorted_df.groupby('date').cumcount() + 1
        top1 = sorted_df[sorted_df['rank_today'] == 1]
        avg = top1['ret_30'].mean()
        win = (top1['ret_30'] > 0).mean() * 100
        print(f'  {label:<28} {len(top1):>6,} {avg:>+8.2f}% {win:>6.1f}%')

    # 基线: 全部信号 (不掐尖)
    print(f'\n  基线 (全部信号): n={len(df_e):,} avg ret {df_e["ret_30"].mean():+.2f}% '
          f'win {(df_e["ret_30"]>0).mean()*100:.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
