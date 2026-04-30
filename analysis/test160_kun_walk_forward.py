# -*- coding: utf-8 -*-
"""坤 regime D 配置 按年 walk-forward + IS/OOS 验证

D 配置 (v4):
  入场: E1 retail<-250 池 + E2v4 mf 上穿 50 + E3 retail 上升
  排雷: stk_y ∈ {100, 101} + mkt_d = 111
  排名: cur_mf ↑ (asc)
  持仓: D6 卖, U1 买
  终结: T0 (trend<11)
  Regime: 大盘 y_gua = 000 (坤)

按入场年聚合 PnL, 看 D 在每个段是否都改善.
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v4(arrays, mkt_d_arr, mkt_m_arr, stk_y_arr):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = -np.inf; last_retail = np.nan
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
            if code not in code_date_idx or today not in code_date_idx[code]: continue
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

    return pd.DataFrame(trades), pd.DataFrame(nav_history)


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


def metric(df_t, df_n):
    final = df_n['nav'].iloc[-1]
    days = (pd.to_datetime(df_n['date'].iloc[-1]) - pd.to_datetime(df_n['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_n['peak'] = df_n['nav'].cummax()
    mdd = ((df_n['nav']-df_n['peak'])/df_n['peak']*100).min()
    win = (df_t['cum_ret_pct']>0).mean()*100 if len(df_t) else 0
    return {'final': final, 'total': (final/INIT_CAPITAL-1)*100,
            'annual': annual, 'mdd': mdd, 'win': win, 'n': len(df_t)}


def apply_config(df_e, cfg):
    df = df_e.copy()
    if cfg.get('avoid_stk_y_100_101'):
        df = df[~df['stk_y'].isin(['100', '101'])]
    if cfg.get('avoid_mkt_d_111'):
        df = df[df['mkt_d'] != '111']
    rank_col = cfg.get('rank_col', 'pool_min_retail')
    rank_asc = cfg.get('rank_asc', True)
    return df.sort_values(['date', rank_col, 'code'],
                          ascending=[True, rank_asc, True]).drop_duplicates('date', keep='first')


def main():
    t0 = time.time()
    print('=== test160: 坤 regime D 配置 walk-forward + IS/OOS ===\n')

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
    df = df[df['mkt_y'] == '000'].reset_index(drop=True)
    print(f'  坤 regime 数据: {len(df):,} 行')

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

    # 找全部 v4 信号
    df_e = find_signals_v4(arrays, mkt_d_arr, mkt_m_arr, stk_y_arr)
    print(f'  v4 入场事件 (全): {len(df_e):,}')
    df_e['year'] = df_e['date'].str[:4]
    print(f'  按入场年:')
    for y, sub in df_e.groupby('year'):
        print(f'    {y}: {len(sub):>4}')

    K = 5
    CONFIGS = [
        ('A 默认',  {'rank_col':'pool_min_retail', 'rank_asc':True}),
        ('D 全包',  {'rank_col':'cur_mf',  'rank_asc':True,
                     'avoid_stk_y_100_101':True, 'avoid_mkt_d_111':True}),
    ]

    # ===== 全样本资金回测对比 =====
    print(f'\n{"="*82}')
    print(f'  K={K} 全样本')
    print(f'{"="*82}')

    results = {}
    for cfg_name, cfg in CONFIGS:
        df_picks = apply_config(df_e, cfg)
        df_t, df_n = run_backtest(K, df_picks, code_date_idx, code_arr, date_arr,
                                   close_arr, trend_arr, mf_arr, retail_arr, all_dates)
        m = metric(df_t, df_n)
        df_t['buy_year'] = pd.to_datetime(df_t['buy_date']).dt.year
        results[cfg_name] = {'df_t': df_t, 'df_n': df_n, 'm': m}
        print(f'  {cfg_name:<8} 总{m["total"]:>+7.1f}% 年{m["annual"]:>+5.2f}% '
              f'MDD{m["mdd"]:>+6.1f}% 胜{m["win"]:>5.1f}% 段{m["n"]:>4}')

    # ===== 按入场年 PnL 对比 =====
    print(f'\n--- 按入场年 PnL (segment-level) ---\n')
    print(f'  {"年":<6}', end='')
    for cfg_name, _ in CONFIGS:
        print(f'  {cfg_name:<26}', end='')
    print()

    years_all = sorted(df_e['year'].unique())
    for y in years_all:
        print(f'  {y:<6}', end='')
        for cfg_name, _ in CONFIGS:
            df_t = results[cfg_name]['df_t']
            sub = df_t[df_t['buy_year'] == int(y)]
            if len(sub) == 0:
                print(f'  {"--":<26}', end='')
                continue
            avg = sub['cum_ret_pct'].mean()
            win = (sub['cum_ret_pct'] > 0).mean() * 100
            tot = sub['cum_pnl'].sum()
            print(f'  n={len(sub):>3} avg{avg:>+6.1f}% w{win:>4.0f}% pnl{tot/1000:>+5.0f}K  ', end='')
        print()

    # ===== NAV 进度对比 (年末) =====
    print(f'\n--- NAV 进度 (年末) ---\n')
    print(f'  {"日期":<11}', end='')
    for cfg_name, _ in CONFIGS:
        print(f'  {cfg_name:<14}', end='')
    print()

    df_n_a = results['A 默认']['df_n'].copy()
    df_n_a['date'] = df_n_a['date'].astype(str)
    df_n_d = results['D 全包']['df_n'].copy()
    df_n_d['date'] = df_n_d['date'].astype(str)

    # 按年末打点
    df_n_a['year'] = df_n_a['date'].str[:4]
    df_n_d['year'] = df_n_d['date'].str[:4]
    eoy_a = df_n_a.groupby('year').last().reset_index()
    eoy_d = df_n_d.groupby('year').last().reset_index()
    for i in range(len(eoy_a)):
        ya = eoy_a.iloc[i]
        yd = eoy_d.iloc[i] if i < len(eoy_d) else None
        print(f'  {ya["date"]:<11}', end='')
        print(f'  ¥{ya["nav"]/1000:>6.0f}K {(ya["nav"]/INIT_CAPITAL-1)*100:>+5.1f}%  ', end='')
        if yd is not None:
            print(f'  ¥{yd["nav"]/1000:>6.0f}K {(yd["nav"]/INIT_CAPITAL-1)*100:>+5.1f}%', end='')
        print()

    # ===== IS / OOS 拆分 =====
    print(f'\n--- IS (2015-2017) / OOS (2018+) 拆分 ---\n')
    for cfg_name, _ in CONFIGS:
        df_t = results[cfg_name]['df_t']
        is_ = df_t[df_t['buy_year'] <= 2017]
        oos = df_t[df_t['buy_year'] >= 2018]
        print(f'  {cfg_name:<8} IS  n={len(is_):>3} avg {is_["cum_ret_pct"].mean():>+6.2f}% '
              f'win {(is_["cum_ret_pct"]>0).mean()*100:>5.1f}% pnl ¥{is_["cum_pnl"].sum()/1000:>+6.0f}K')
        print(f'  {cfg_name:<8} OOS n={len(oos):>3} avg {oos["cum_ret_pct"].mean():>+6.2f}% '
              f'win {(oos["cum_ret_pct"]>0).mean()*100:>5.1f}% pnl ¥{oos["cum_pnl"].sum()/1000:>+6.0f}K')
        print()

    # ===== D 在每年是否都改善 A =====
    print(f'\n--- D vs A: 每年 lift ---\n')
    print(f'  {"年":<6} {"A avg":>8} {"D avg":>8} {"D-A":>8} {"D 改善?":>8}')
    df_t_a = results['A 默认']['df_t']
    df_t_d = results['D 全包']['df_t']
    for y in years_all:
        sub_a = df_t_a[df_t_a['buy_year'] == int(y)]
        sub_d = df_t_d[df_t_d['buy_year'] == int(y)]
        if len(sub_a) == 0 or len(sub_d) == 0: continue
        a_avg = sub_a['cum_ret_pct'].mean()
        d_avg = sub_d['cum_ret_pct'].mean()
        good = '✓' if d_avg > a_avg else '✗'
        print(f'  {y:<6} {a_avg:>+7.2f}% {d_avg:>+7.2f}% {d_avg-a_avg:>+7.2f}% {good:>8}')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    results['D 全包']['df_t'].to_csv(os.path.join(out_dir, 'kun_d_trades.csv'),
                                       index=False, encoding='utf-8-sig')
    print(f'\n  写出 kun_d_trades.csv')
    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
