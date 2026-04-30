# -*- coding: utf-8 -*-
"""巽日触发时 retail 当日值是好指标还是差指标 (单独看)

baseline 当前: retail<-250 入池, retail>=0 出池
所以买入时 retail 必然 ∈ [-250, 0)

若放宽出池阈值, 比如 retail>=50 / >=100 / 不主动出池,
就允许买入时 retail >= 0 的事件. 这部分质量如何?

测试:
  扫所有"曾入池过 + 巽日 + 大盘y=000" 的事件, 不主动出池
  按当日 retail 分箱, 看 ret_30 / 实际卖出 ret / win

特别关注:
  retail < -100  (深, 散户还在卖)
  retail [-100, 0)  (回升中)
  retail [0, 100)  (转正)
  retail >= 100  (散户回流)

如果 retail>=0 区间 ret 显著 < 全集 → retail>=0 出池是对的
如果 retail>=0 区间 ret 接近或更高 → 出池阈值要放宽
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
POOL_THR = -250


def simulate_sell(buy_idx, td_arr, close_arr, max_end):
    buy_price = close_arr[buy_idx]
    cross_count = 0
    running_max = td_arr[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days_held = k - buy_idx
        if not np.isnan(td_arr[k]):
            running_max = max(running_max, td_arr[k])
        if running_max >= 89 and td_arr[k] < 89 and td_arr[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close_arr[k]/buy_price-1)*100, days_held
        if days_held >= 20:
            seg = td_arr[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return k, 'ts20', (close_arr[k]/buy_price-1)*100, days_held
        if days_held >= MAX_HOLD:
            return k, 'timeout', (close_arr[k]/buy_price-1)*100, days_held
    k = max_end
    return k, 'force_close', (close_arr[k]/buy_price-1)*100, k - buy_idx


def main():
    t0 = time.time()
    print('=== 巽日触发时 retail 是好指标还是差指标 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫事件 (曾入池 retail<-250, 不主动出池, 看触发当日 retail) ...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        ever_pooled = False  # 该股是否曾入池
        last_pool_enter_i = -1
        last_pool_min_retail = np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            # 入池
            if not ever_pooled and retail_arr[gi] < POOL_THR:
                ever_pooled = True
                last_pool_enter_i = i
                last_pool_min_retail = retail_arr[gi]
            elif ever_pooled and retail_arr[gi] < POOL_THR:
                # 二次入池, 重置 (因为前一次池可能已经"过期")
                # 但保留 ever_pooled 状态
                last_pool_enter_i = i
                last_pool_min_retail = retail_arr[gi]
            elif ever_pooled and retail_arr[gi] < last_pool_min_retail:
                last_pool_min_retail = retail_arr[gi]

            # 触发: 不挑出池, 看所有曾入池的巽日
            if ever_pooled and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                pool_days = i - last_pool_enter_i
                cur_retail = retail_arr[gi]
                # 模拟卖出
                max_end = min(s + n - 1, gi + MAX_HOLD)
                _, reason, ret_pct, days_held = simulate_sell(gi, trend_arr, close_arr, max_end)
                # 30 日固定 ret 也算
                ret_30 = (close_arr[gi+30] / close_arr[gi] - 1) * 100 if gi + 30 < e else np.nan
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'cur_retail': cur_retail,
                    'pool_days': pool_days,
                    'pool_min_retail': last_pool_min_retail,
                    'ret_pct': ret_pct, 'ret_30': ret_30,
                    'reason': reason, 'days_held': days_held,
                })

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')
    base_ret = df_e['ret_pct'].mean()
    base_win = (df_e['ret_pct']>0).mean()*100
    print(f'  全集 ret={base_ret:+.2f}% / win={base_win:.1f}% (实际卖出)')

    # ============ 1. cur_retail (触发当日) 分箱 ============
    print('\n=== 1. 触发当日 retail 分箱 (实际卖出 ret) ===\n')
    bins = [-np.inf, -250, -150, -50, 0, 50, 150, np.inf]
    labels = ['<-250 (深)', '[-250,-150)', '[-150,-50)', '[-50,0)', '[0,50)', '[50,150)', '>=150 (回流)']
    df_e['cb'] = pd.cut(df_e['cur_retail'], bins=bins, labels=labels, right=False)
    print(f'  {"档":<14} {"n":>6} {"占比":>5} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>6} {"lift":>7}')
    for lab in labels:
        sub = df_e[df_e['cb'] == lab]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        print(f'  {lab:<14} {len(sub):>6} {len(sub)/len(df_e)*100:>4.1f}% '
              f'{avg:>+8.2f}% {(sub["ret_pct"]>0).mean()*100:>6.1f}% '
              f'{sub["ret_pct"].median():>+7.2f}% {sub["days_held"].mean():>5.1f}d '
              f'{avg-base_ret:>+6.2f}')

    # ============ 2. cur_retail 细分箱 (看趋势) ============
    print('\n=== 2. 触发当日 retail 细分箱 ===\n')
    fine_bins = [-np.inf, -300, -200, -100, -50, 0, 50, 100, 200, np.inf]
    fine_labels = ['<-300', '[-300,-200)', '[-200,-100)', '[-100,-50)', '[-50,0)',
                    '[0,50)', '[50,100)', '[100,200)', '>=200']
    df_e['cb_fine'] = pd.cut(df_e['cur_retail'], bins=fine_bins, labels=fine_labels, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"lift":>7}')
    for lab in fine_labels:
        sub = df_e[df_e['cb_fine'] == lab]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_pct"]>0).mean()*100:>6.1f}% {avg-base_ret:>+6.2f}')

    # ============ 3. cur_retail 是单调有序还是 U / 阶梯? ============
    print('\n=== 3. 跨段稳定性 (主要分箱) ===\n')
    WINDOWS = [
        ('w1_2018',    '2018-01-01', '2019-01-01'),
        ('w2_2019',    '2019-01-01', '2020-01-01'),
        ('w4_2021',    '2021-01-01', '2022-01-01'),
        ('w5_2022',    '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    test_buckets = [
        ('< -250 (深)',  df_e[df_e['cur_retail'] < -250]),
        ('[-250,-100)',  df_e[(df_e['cur_retail'] >= -250) & (df_e['cur_retail'] < -100)]),
        ('[-100, 0)',    df_e[(df_e['cur_retail'] >= -100) & (df_e['cur_retail'] < 0)]),
        ('[0, 100)',     df_e[(df_e['cur_retail'] >= 0) & (df_e['cur_retail'] < 100)]),
        ('>= 100',       df_e[df_e['cur_retail'] >= 100]),
    ]
    for label, sub in test_buckets:
        print(f'\n  --- {label}: 全集 n={len(sub)} ret={sub["ret_pct"].mean():+.2f}% ---')
        for w in WINDOWS:
            ssub = sub[(sub['date'] >= w[1]) & (sub['date'] < w[2])]
            if len(ssub) >= 10:
                print(f'    {w[0]:<12} n={len(ssub):>4} ret={ssub["ret_pct"].mean():>+6.2f}% '
                      f'win {(ssub["ret_pct"]>0).mean()*100:>5.1f}%')

    # ============ 4. 池深 × cur_retail 交叉 ============
    print('\n=== 4. 池深 × 当日 retail 交叉 (avg ret%) ===\n')
    df_e['db'] = pd.cut(df_e['pool_min_retail'],
                            bins=[-np.inf, -500, -350, -250],
                            labels=['<-500', '[-500,-350)', '[-350,-250)'])
    df_e['cb2'] = pd.cut(df_e['cur_retail'],
                              bins=[-np.inf, -100, 0, 100, np.inf],
                              labels=['<-100', '[-100,0)', '[0,100)', '>=100'])
    pivot = df_e.pivot_table(values='ret_pct', index='db', columns='cb2',
                                  aggfunc='mean', observed=True)
    print(pivot.round(2).to_string())
    print('\n  n:')
    pn = df_e.pivot_table(values='ret_pct', index='db', columns='cb2',
                                aggfunc='count', observed=True)
    print(pn.to_string())

    # ============ 5. 不同出池阈值的资金回测预估 ============
    print('\n=== 5. 不同 cur_retail 上限的事件统计 ===\n')
    print('  (假设排名按 pool_min_retail↑, 这部分需要用资金回测验证)')
    for thr in [-100, -50, 0, 50, 100, 200, 9999]:
        sub = df_e[df_e['cur_retail'] < thr]
        if len(sub) == 0: continue
        sig_days = sub['date'].nunique()
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        if thr == 9999:
            label = '不限'
        else:
            label = f'<{thr}'
        print(f'  cur_retail {label:<8} n={len(sub):>5,}, 信号天 {sig_days}, '
              f'avg {avg:>+5.2f}%, win {win:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
