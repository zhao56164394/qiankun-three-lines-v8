# -*- coding: utf-8 -*-
"""坤 v4/v5 排雷+排名 资金回测对比

4 配置 × 2 版本 = 8 组合:
  A: 默认 (pool_min_retail ↑ 排名, 无排雷)
  B: A + 排雷 stk_y ∈ {100, 101}
  C: B + 排名换成 cur_mf ↑
  D: C + 排雷加 mkt_d = 111

每个配置跑 K=5, 看 总收益 / 年化 / MDD / 胜率
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals(arrays, mode, mkt_d_arr, mkt_m_arr, stk_y_arr):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = -np.inf if mode == 'v4' else np.nan
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

            if mode == 'v4':
                e2_ok = (last_mf <= 50) and (mf[gi] > 50)
            else:  # v5
                mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
                trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
                e2_ok = mf_rising and trend_ok

            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            if in_pool and e2_ok and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                    'pool_min_retail': pool_min_retail,
                    'cur_mf': mf[gi], 'cur_retail': retail[gi],
                    'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                    'stk_y': stk_y_arr[gi],
                })
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def run_backtest(K, df_picks, code_date_idx, code_arr, date_arr, close_arr,
                 trend_arr, mf_arr, retail_arr, all_dates):
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
            cur_close = close_arr[today_idx]

            if not np.isnan(trend_arr[today_idx]) and trend_arr[today_idx] < 11:
                if pos['state'] == 'holding':
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                trades.append(_make_trade(code, pos, today, today_idx - pos['initial_buy_idx'], 'td<11'))
                del holdings[code]
                continue

            if today_idx < 1: continue
            mf_c = mf_arr[today_idx] - mf_arr[today_idx-1] if not np.isnan(mf_arr[today_idx-1]) else 0
            ret_c = retail_arr[today_idx] - retail_arr[today_idx-1] if not np.isnan(retail_arr[today_idx-1]) else 0
            td_c = trend_arr[today_idx] - trend_arr[today_idx-1] if not np.isnan(trend_arr[today_idx-1]) else 0

            if pos['state'] == 'holding':
                if mf_c < 0 and ret_c < 0 and td_c < 0:
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:
                if mf_c > 0:
                    buy_price = cur_close
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
                            pos['cash_at_pending'] = avail - cost

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
                                'initial_buy_price': buy_price,
                                'initial_cost': cost,
                                'cur_buy_price': buy_price,
                                'shares': shares,
                                'cash_at_pending': 0,
                                'cum_pnl': 0,
                                'legs': 1,
                            }

        mv = 0.0
        for code, pos in holdings.items():
            if pos['state'] == 'holding':
                if code in code_date_idx and today in code_date_idx[code]:
                    ti = code_date_idx[code][today]
                    mv += pos['shares'] * close_arr[ti]
                else:
                    mv += pos['shares'] * pos['cur_buy_price']
            else:
                mv += pos['cash_at_pending']
        nav_history.append({'date':today, 'cash':cash, 'mv':mv, 'nav':cash+mv})

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
            trades.append(_make_trade(code, pos, last, ti - pos['initial_buy_idx'], 'force_close'))

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
    return {'final':final, 'total':(final/INIT_CAPITAL-1)*100, 'annual':annual,
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'avg_pos':avg_pos}


def _make_trade(code, pos, sell_date, days, reason):
    return {
        'code': code,
        'buy_date': pos['initial_buy_date'],
        'sell_date': sell_date,
        'cum_pnl': pos['cum_pnl'],
        'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
        'days': days,
        'legs': pos['legs'],
        'reason': reason,
    }


def apply_config(df_e, cfg):
    """对事件应用排雷+排名"""
    df = df_e.copy()
    if cfg.get('avoid_stk_y_100_101'):
        df = df[~df['stk_y'].isin(['100', '101'])]
    if cfg.get('avoid_mkt_d_111'):
        df = df[df['mkt_d'] != '111']

    rank_col = cfg.get('rank_col', 'pool_min_retail')
    rank_asc = cfg.get('rank_asc', True)
    df_picks = df.sort_values(['date', rank_col, 'code'],
                              ascending=[True, rank_asc, True]).drop_duplicates('date', keep='first')
    return df, df_picks


def main():
    t0 = time.time()
    print('=== test159: v4/v5 排雷+排名 资金回测 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3).replace({'nan':''})
    g['y_gua'] = g['y_gua'].astype(str).str.zfill(3).replace({'nan':''})
    g.rename(columns={'d_gua':'stk_d', 'y_gua':'stk_y'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        mkt[c] = mkt[c].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt.drop_duplicates('date').rename(columns={'d_gua':'mkt_d','m_gua':'mkt_m','y_gua':'mkt_y'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','stk_d','d_trend','mkt_y']).reset_index(drop=True)
    # 不锁 regime, 跟 v4/v5 对齐 (全市场)
    print(f'  全市场数据: {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    arrays = {
        'code': code_arr, 'date': date_arr,
        'retail': retail_arr, 'mf': mf_arr, 'td': trend_arr,
        'starts': code_starts, 'ends': code_ends,
    }
    all_dates = sorted(set(date_arr.tolist()))

    # ===== 配置 =====
    CONFIGS = [
        ('A 默认',         {'rank_col':'pool_min_retail', 'rank_asc':True}),
        ('B +排雷stk_y',   {'rank_col':'pool_min_retail', 'rank_asc':True,
                            'avoid_stk_y_100_101':True}),
        ('C +排cur_mf',    {'rank_col':'cur_mf',          'rank_asc':True,
                            'avoid_stk_y_100_101':True}),
        ('D +排雷mkt_d',   {'rank_col':'cur_mf',          'rank_asc':True,
                            'avoid_stk_y_100_101':True, 'avoid_mkt_d_111':True}),
    ]

    # ===== 跑 v4 / v5 ×ed 4 配置 =====
    K = 5
    print(f'\n{"="*92}')
    print(f'  K={K} 全样本资金回测')
    print(f'{"="*92}')
    print(f'\n{"":<12}{"配置":<14}{"事件":>6}{"信号天":>7}{"期末":>10}{"总":>9}{"年化":>9}'
          f'{"MDD":>9}{"段":>5}{"胜":>6}{"段平均":>8}{"avg_pos":>8}')

    for ver in ['v4', 'v5']:
        print()
        df_e_full = find_signals(arrays, ver, mkt_d_arr, mkt_m_arr, stk_y_arr)
        for cfg_name, cfg in CONFIGS:
            df_e_filt, df_picks = apply_config(df_e_full, cfg)
            r = run_backtest(K, df_picks, code_date_idx, code_arr, date_arr,
                              close_arr, trend_arr, mf_arr, retail_arr, all_dates)
            print(f'  [{ver}] {cfg_name:<14}'
                  f'{len(df_e_filt):>6}{len(df_picks):>7}'
                  f' ¥{r["final"]/1000:>7.0f}K'
                  f' {r["total"]:>+7.1f}%{r["annual"]:>+7.2f}%'
                  f' {r["mdd"]:>+7.1f}% {r["n_trades"]:>4}'
                  f' {r["win"]:>5.1f}% {r["avg"]:>+6.2f}%'
                  f' {r["avg_pos"]:>6.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
