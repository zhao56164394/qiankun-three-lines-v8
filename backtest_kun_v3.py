# -*- coding: utf-8 -*-
"""坤 v3 资金回测 — F8 落地 (pool_min_retail<-300 AND pool_days<6)

策略组件 (Step 1-2 完成):
  入池: retail < -250
  出池: retail >= 0
  买点: 大盘 y_gua=000 坤 + 个股 d_gua=011 巽
  过滤: pool_min_retail < -300 AND pool_days < 6 (F8)
  排名: 同日多只按 pool_min_retail↑ 取 1
  卖点: bull_2nd / TS20 / 60d
  仓位: K=5 / 每日 1 只 / 200K

待加 (Step 3-4):
  - 大盘月/日卦, 个股月/年卦
  - 避雷与好规律 score
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
POOL_THR = -250
POOL_EXIT_RETAIL = 0
DEPTH_THR = -300
DAYS_THR = 6
LOOKBACK = 30
K_DEFAULT = 5

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]


def should_sell(td_buy_to_now, days_held):
    if days_held >= MAX_HOLD: return True, 'timeout'
    if len(td_buy_to_now) < 2: return False, None
    if days_held >= 20:
        valid = td_buy_to_now[~np.isnan(td_buy_to_now)]
        if len(valid) > 0 and valid.max() < 89:
            return True, 'ts20'
    cross_count = 0
    running_max = td_buy_to_now[0]
    for k in range(1, len(td_buy_to_now)):
        if not np.isnan(td_buy_to_now[k]):
            running_max = max(running_max, td_buy_to_now[k])
        if running_max >= 89 and td_buy_to_now[k] < 89 and td_buy_to_now[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2: return True, 'bull_2nd'
    return False, None


def run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates):
    SLOT_VALUE = INIT_CAPITAL / K
    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []
    picks_by_date = df_picks.set_index('date').to_dict('index')

    for today in all_dates:
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]:
                continue
            today_idx = code_date_idx[code][today]
            buy_idx = pos['buy_idx_global']
            days_held = today_idx - buy_idx
            td_seg = trend_arr[buy_idx:today_idx+1]
            sell, reason = should_sell(td_seg, days_held)
            if sell:
                sell_price = close_arr[today_idx]
                proceeds = pos['shares'] * sell_price
                cost = pos['shares'] * pos['buy_price']
                ret_pct = (sell_price/pos['buy_price']-1)*100
                cash += proceeds
                trades.append({'code':code, 'buy_date':pos['buy_date'], 'sell_date':today,
                                'profit':proceeds-cost, 'ret_pct':ret_pct, 'days':days_held,
                                'reason':reason})
                del holdings[code]

        if today in picks_by_date and len(holdings) < K:
            cand = picks_by_date[today]
            code = cand['code']
            if code not in holdings and code in code_date_idx and today in code_date_idx[code]:
                ridx = code_date_idx[code][today]
                buy_price = close_arr[ridx]
                if not np.isnan(buy_price) and buy_price > 0:
                    shares = int(SLOT_VALUE // buy_price // 100) * 100
                    if shares > 0:
                        cost = shares * buy_price
                        if cost <= cash:
                            cash -= cost
                            holdings[code] = {'buy_date':today, 'buy_idx_global':ridx,
                                                'buy_price':buy_price, 'shares':shares}

        mv = 0.0
        for code, pos in holdings.items():
            if code in code_date_idx and today in code_date_idx[code]:
                ti = code_date_idx[code][today]
                mv += pos['shares'] * close_arr[ti]
            else:
                mv += pos['shares'] * pos['buy_price']
        nav_history.append({'date':today, 'cash':cash, 'mv':mv, 'nav':cash+mv})

    last = all_dates[-1]
    for code, pos in list(holdings.items()):
        if code in code_date_idx and last in code_date_idx[code]:
            ti = code_date_idx[code][last]
            sp = close_arr[ti]
            cash += pos['shares']*sp
            trades.append({'code':code, 'buy_date':pos['buy_date'], 'sell_date':last,
                            'profit':pos['shares']*(sp-pos['buy_price']),
                            'ret_pct':(sp/pos['buy_price']-1)*100,
                            'days':ti - pos['buy_idx_global'], 'reason':'force_close'})

    df_t = pd.DataFrame(trades)
    df_n = pd.DataFrame(nav_history)
    final = df_n['nav'].iloc[-1]
    days = (pd.to_datetime(df_n['date'].iloc[-1]) - pd.to_datetime(df_n['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_n['peak'] = df_n['nav'].cummax()
    mdd = ((df_n['nav']-df_n['peak'])/df_n['peak']*100).min()
    win = (df_t['ret_pct']>0).mean()*100 if len(df_t) else 0
    avg = df_t['ret_pct'].mean() if len(df_t) else 0
    return {'final':final, 'total':(final/INIT_CAPITAL-1)*100, 'annual':annual,
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'df_t':df_t,
            'df_n':df_n}


def main():
    t0 = time.time()
    print('=== 坤 v3 资金回测 (F8: pool_min_retail<-300 + pool_days<6) ===\n')

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
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True; pool_enter_i = i; pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] < pool_min_retail:
                pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False; continue
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                events.append({'date':date_arr[gi], 'code':code_arr[gi],
                                'pool_days':i-pool_enter_i,
                                'pool_min_retail':pool_min_retail})
                in_pool = False

    df_e_full = pd.DataFrame(events)
    print(f'  全集事件: {len(df_e_full):,} (无过滤)')

    # 过滤方案
    schemes = {}
    schemes['baseline (无过滤)']             = df_e_full
    schemes['F3 (pool_days<6)']             = df_e_full[df_e_full['pool_days'] < DAYS_THR]
    schemes['F8 (retail<-300 + days<6)']    = df_e_full[(df_e_full['pool_min_retail'] < DEPTH_THR) &
                                                            (df_e_full['pool_days'] < DAYS_THR)]

    all_dates = sorted(df['date'].unique())

    K = K_DEFAULT
    print(f'\n=== K={K} 总览 ===\n')
    print(f'  {"方案":<32} {"事件":>6} {"信号天":>6} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = {}
    for label, df_x in schemes.items():
        df_picks = df_x.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')
        n_signal_days = len(df_picks)
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results[label] = (r, df_picks)
        print(f'  {label:<32} {len(df_x):>6,} {n_signal_days:>6} ¥{r["final"]/1000:>8.0f}K '
              f'{r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% {r["mdd"]:>+7.1f}% '
              f'{r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 跨段对比 F3 vs F8
    print(f'\n=== 跨段稳定性 (按买入年份分段) ===\n')
    print(f'  {"段":<14} | F3 笔  ret    win  | F8 笔  ret    win   | F8 vs F3 (笔, ret_diff)')
    f3_t = results['F3 (pool_days<6)'][0]['df_t']
    f8_t = results['F8 (retail<-300 + days<6)'][0]['df_t']
    f3_t['year'] = pd.to_datetime(f3_t['buy_date']).dt.year
    f8_t['year'] = pd.to_datetime(f8_t['buy_date']).dt.year
    for y in sorted(f3_t['year'].unique()):
        f3_sub = f3_t[f3_t['year'] == y]
        f8_sub = f8_t[f8_t['year'] == y]
        f3_n, f8_n = len(f3_sub), len(f8_sub)
        f3_r = f3_sub['ret_pct'].mean() if f3_n else 0
        f8_r = f8_sub['ret_pct'].mean() if f8_n else 0
        f3_w = (f3_sub['ret_pct']>0).mean()*100 if f3_n else 0
        f8_w = (f8_sub['ret_pct']>0).mean()*100 if f8_n else 0
        print(f'  {y:<14} | {f3_n:>3} {f3_r:>+6.2f}% {f3_w:>5.1f}% | {f8_n:>3} {f8_r:>+6.2f}% {f8_w:>5.1f}% '
              f'| ({f8_n-f3_n:+}, {f8_r-f3_r:+5.2f}%)')

    # 跨段窗口 (5 段)
    print(f'\n=== 5 段窗口对比 (买入日期) ===\n')
    print(f'  {"段":<12} | {"F3 笔":>5} {"F3 ret":>9} {"F3 win":>7} {"F3 PNL":>10} | '
          f'{"F8 笔":>5} {"F8 ret":>9} {"F8 win":>7} {"F8 PNL":>10}')
    for w in WINDOWS:
        f3_sub = f3_t[(f3_t['buy_date'] >= w[1]) & (f3_t['buy_date'] < w[2])]
        f8_sub = f8_t[(f8_t['buy_date'] >= w[1]) & (f8_t['buy_date'] < w[2])]
        f3_n, f8_n = len(f3_sub), len(f8_sub)
        f3_r = f3_sub['ret_pct'].mean() if f3_n else 0
        f8_r = f8_sub['ret_pct'].mean() if f8_n else 0
        f3_w = (f3_sub['ret_pct']>0).mean()*100 if f3_n else 0
        f8_w = (f8_sub['ret_pct']>0).mean()*100 if f8_n else 0
        f3_p = f3_sub['profit'].sum()
        f8_p = f8_sub['profit'].sum()
        print(f'  {w[0]:<12} | {f3_n:>5} {f3_r:>+8.2f}% {f3_w:>6.1f}% ¥{f3_p:>+9,.0f} | '
              f'{f8_n:>5} {f8_r:>+8.2f}% {f8_w:>6.1f}% ¥{f8_p:>+9,.0f}')

    # F3 / F8 NAV 曲线对比 (年度终值)
    print(f'\n=== NAV 年度终值 ===')
    f3_n = results['F3 (pool_days<6)'][0]['df_n']
    f8_n = results['F8 (retail<-300 + days<6)'][0]['df_n']
    f3_n['year'] = pd.to_datetime(f3_n['date']).dt.year
    f8_n['year'] = pd.to_datetime(f8_n['date']).dt.year
    print(f'  {"年":<6} | {"F3 NAV":>10} {"F3 增量":>9} | {"F8 NAV":>10} {"F8 增量":>9}')
    f3_prev = f8_prev = INIT_CAPITAL
    for y in sorted(f3_n['year'].unique()):
        f3_y = f3_n[f3_n['year'] == y]['nav'].iloc[-1]
        f8_y = f8_n[f8_n['year'] == y]['nav'].iloc[-1]
        print(f'  {y:<6} | ¥{f3_y/1000:>8.0f}K {(f3_y/f3_prev-1)*100:>+7.2f}% | '
              f'¥{f8_y/1000:>8.0f}K {(f8_y/f8_prev-1)*100:>+7.2f}%')
        f3_prev, f8_prev = f3_y, f8_y

    # 不同 K 看 F8
    print(f'\n=== F8 不同 K ===\n')
    f8_picks = results['F8 (retail<-300 + days<6)'][1]
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, f8_picks, code_date_idx, close_arr, trend_arr, all_dates)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 写出 F8 K=5 的 trades / nav
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    f8_r = results['F8 (retail<-300 + days<6)'][0]
    f8_r['df_t'].to_csv(os.path.join(out_dir, 'kun_v3_f8_trades.csv'), index=False, encoding='utf-8-sig')
    f8_r['df_n'].to_csv(os.path.join(out_dir, 'kun_v3_f8_nav.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 kun_v3_f8_{{trades,nav}}.csv')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
