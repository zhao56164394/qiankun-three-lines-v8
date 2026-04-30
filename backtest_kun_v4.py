# -*- coding: utf-8 -*-
"""坤 v4 资金回测 — D6+U1 波段 + trend<11 终结

入场: retail<-250 池中 + mf 上穿 50 + retail 上升
持仓: D6 三线齐降 (mf+retail+trend) 卖
再买: U1 mf 上升 买
终结: trend<11 (无 60d timeout)

仓位: K=5 / 200K / 排名 pool_min_retail↑

vs v3 (test128 R2 baseline +126.0%) 对比
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan
        pool_min_retail = np.inf

        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            if not in_pool and cur_below and not prev_below:
                in_pool = True
                pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]

            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)

            if in_pool and mf_cross_up and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                    'pool_min_retail': pool_min_retail,
                })
                in_pool = False

            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def run_backtest(K, df_picks, code_date_idx, code_arr, date_arr, close_arr,
                  trend_arr, mf_arr, retail_arr, all_dates):
    """仓位守恒的资金回测
       holdings[code] = {状态, 资金额, 起始 buy_price, ...}
       状态: 'holding' (持仓) / 'pending' (空仓等再买)
    """
    SLOT_VALUE = INIT_CAPITAL / K

    cash = INIT_CAPITAL
    holdings = {}  # code -> dict
    trades = []
    nav_history = []
    picks_by_date = df_picks.set_index('date').to_dict('index')

    for today in all_dates:
        # 1. 处理已有持仓 (D6 卖 / U1 买回 / trend<11 终结)
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]:
                continue
            today_idx = code_date_idx[code][today]

            # trend<11: 整段终结
            if not np.isnan(trend_arr[today_idx]) and trend_arr[today_idx] < 11:
                if pos['state'] == 'holding':
                    sell_price = close_arr[today_idx]
                    proceeds = pos['shares'] * sell_price
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                # 整段终结
                trades.append({
                    'code': code, 'buy_date': pos['initial_buy_date'],
                    'sell_date': today, 'state': 'segment_end',
                    'cum_pnl': pos['cum_pnl'],
                    'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
                    'days': today_idx - pos['initial_buy_idx'],
                    'legs': pos['legs'],
                    'reason': 'td<11',
                })
                del holdings[code]
                continue

            if today_idx < 1: continue

            mf_c = mf_arr[today_idx] - mf_arr[today_idx-1] if not np.isnan(mf_arr[today_idx-1]) else 0
            ret_c = retail_arr[today_idx] - retail_arr[today_idx-1] if not np.isnan(retail_arr[today_idx-1]) else 0
            td_c = trend_arr[today_idx] - trend_arr[today_idx-1] if not np.isnan(trend_arr[today_idx-1]) else 0

            if pos['state'] == 'holding':
                # D6: mf+retail+trend 三线齐降 → 卖
                if mf_c < 0 and ret_c < 0 and td_c < 0:
                    sell_price = close_arr[today_idx]
                    proceeds = pos['shares'] * sell_price
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:  # pending
                # U1: mf 上升 → 买回 (用之前的钱)
                if mf_c > 0:
                    buy_price = close_arr[today_idx]
                    if not np.isnan(buy_price) and buy_price > 0:
                        avail = pos['cash_at_pending']
                        shares = int(avail // buy_price // 100) * 100
                        if shares > 0:
                            cost = shares * buy_price
                            cash -= cost
                            pos['shares'] = shares
                            pos['cur_buy_price'] = buy_price
                            pos['state'] = 'holding'
                            pos['legs'] += 1
                            pos['cash_at_pending'] = avail - cost  # 残额

        # 2. 检查新买信号 (今日 picks)
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
                            holdings[code] = {
                                'state': 'holding',
                                'initial_buy_date': today,
                                'initial_buy_idx': ridx,
                                'initial_cost': cost,
                                'cur_buy_price': buy_price,
                                'shares': shares,
                                'cash_at_pending': 0,
                                'cum_pnl': 0,
                                'legs': 1,
                            }

        # 3. NAV
        mv = 0.0
        for code, pos in holdings.items():
            if pos['state'] == 'holding':
                if code in code_date_idx and today in code_date_idx[code]:
                    ti = code_date_idx[code][today]
                    mv += pos['shares'] * close_arr[ti]
                else:
                    mv += pos['shares'] * pos['cur_buy_price']
            else:  # pending
                mv += pos['cash_at_pending']
        nav_history.append({'date':today, 'cash':cash, 'mv':mv, 'nav':cash+mv})

    # 末尾强平
    last = all_dates[-1]
    for code, pos in list(holdings.items()):
        if code in code_date_idx and last in code_date_idx[code]:
            ti = code_date_idx[code][last]
            sp = close_arr[ti]
            if pos['state'] == 'holding':
                proceeds = pos['shares'] * sp
                profit = proceeds - pos['shares'] * pos['cur_buy_price']
                cash += proceeds
                pos['cum_pnl'] += profit
            else:
                cash += pos['cash_at_pending']
            trades.append({
                'code': code, 'buy_date': pos['initial_buy_date'],
                'sell_date': last, 'state': 'force_close',
                'cum_pnl': pos['cum_pnl'],
                'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
                'days': ti - pos['initial_buy_idx'],
                'legs': pos['legs'],
                'reason': 'force_close',
            })

    df_t = pd.DataFrame(trades)
    df_n = pd.DataFrame(nav_history)
    final = df_n['nav'].iloc[-1]
    days = (pd.to_datetime(df_n['date'].iloc[-1]) - pd.to_datetime(df_n['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_n['peak'] = df_n['nav'].cummax()
    mdd = ((df_n['nav']-df_n['peak'])/df_n['peak']*100).min()
    win = (df_t['cum_ret_pct']>0).mean()*100 if len(df_t) else 0
    avg = df_t['cum_ret_pct'].mean() if len(df_t) else 0
    df_n['pos_pct'] = df_n['mv'] / df_n['nav']
    avg_pos = df_n['pos_pct'].mean() * 100

    return {
        'final':final, 'total':(final/INIT_CAPITAL-1)*100, 'annual':annual,
        'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'avg_pos':avg_pos,
        'df_t':df_t, 'df_n':df_n,
    }


def main():
    t0 = time.time()
    print('=== 坤 v4: D6+U1 波段 + trend<11 终结 资金回测 ===\n')

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
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    arrays = {
        'code': code_arr,
        'date': date_arr,
        'retail': retail_arr,
        'mf': mf_arr,
        'starts': code_starts, 'ends': code_ends,
    }
    df_e = find_signals(arrays)
    print(f'  入场信号: {len(df_e):,}')

    df_picks = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                      ascending=[True, True, True]).drop_duplicates('date', keep='first')
    print(f'  每日 1 只: {len(df_picks)} 信号天')

    all_dates = sorted(df['date'].unique())

    print(f'\n=== 各 K 总览 ===\n')
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"段数":>4} {"胜率":>6} {"平均/段":>8} {"avg_pos":>8}')

    results = {}
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, df_picks, code_date_idx, code_arr, date_arr,
                          close_arr, trend_arr, mf_arr, retail_arr, all_dates)
        results[K] = r
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+6.2f}% '
              f'{r["avg_pos"]:>6.1f}%')

    # K=5 详细
    K = 5
    r = results[K]
    df_t = r['df_t']
    print(f'\n=== K={K} 详细 ===\n')

    print(f'\n  按年:')
    df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year
    for y in sorted(df_t['year'].unique()):
        sub = df_t[df_t['year']==y]
        print(f'    {y}: n={len(sub):>3} avg {sub["cum_ret_pct"].mean():>+6.2f}% '
              f'win {(sub["cum_ret_pct"]>0).mean()*100:>5.1f}% pnl ¥{sub["cum_pnl"].sum():>+10,.0f}')

    print(f'\n  按 reason:')
    for r_lab in df_t['reason'].unique():
        sub = df_t[df_t['reason']==r_lab]
        print(f'    {r_lab:<14} n={len(sub):>3} avg {sub["cum_ret_pct"].mean():>+6.2f}% '
              f'avg_legs={sub["legs"].mean():.1f} hold={sub["days"].mean():.1f}')

    print(f'\n  按 legs (波段次数):')
    for legs in sorted(df_t['legs'].unique())[:10]:
        sub = df_t[df_t['legs']==legs]
        print(f'    legs={legs}: n={len(sub):>3} avg {sub["cum_ret_pct"].mean():>+6.2f}% '
              f'win {(sub["cum_ret_pct"]>0).mean()*100:>5.1f}% hold={sub["days"].mean():.1f}')

    # vs v3 baseline
    print(f'\n=== vs v3 baseline ===')
    print(f'  v3 R2 (单次买卖, 60d timeout): K=5 +126.0% / 7.96% 年化 / -26.0% MDD / 47.7% win')
    print(f'  v4 D6+U1 (波段, trend<11):      K=5 {r["total"]:+.1f}% / {r["annual"]:+.2f}% 年化 / {r["mdd"]:+.1f}% MDD / {r["win"]:.1f}% win')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_t.to_csv(os.path.join(out_dir, 'kun_v4_d6u1_trades.csv'), index=False, encoding='utf-8-sig')
    r['df_n'].to_csv(os.path.join(out_dir, 'kun_v4_d6u1_nav.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 kun_v4_d6u1_{{trades,nav}}.csv')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
