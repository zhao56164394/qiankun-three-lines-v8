# -*- coding: utf-8 -*-
"""v5 + 坤 regime — 池深 (pool_min_retail) 单因子评估

3 层评估:
  L1: 单笔 ret% — 拆 4/8 桶, 看每桶 avg ret + 跨年 lift
  L2: 阈值排雷 — retail < {-300, -400, -500} 不进, 看资金回测
  L3: 同日排名 — pool_min_retail 升/降序 top-1/3/5

baseline: v5 默认 (pool_min_retail asc 排名)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v5(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = np.nan; last_retail = np.nan
        pool_min_retail = np.inf
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
                pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                    'pool_min_retail': pool_min_retail,
                })
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    bp = close[buy_idx]
    cum_mult = 1.0; holding = True
    cur_buy_price = bp
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
            return (cum_mult-1)*100
        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0
        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]
                holding = True
    if holding:
        cum_mult *= close[max_end] / cur_buy_price
    return (cum_mult-1)*100


def run_backtest(K, df_picks, code_date_idx, code_arr, date_arr, close_arr,
                 trend_arr, mf_arr, retail_arr, all_dates):
    SLOT_VALUE = INIT_CAPITAL / K
    cash = INIT_CAPITAL
    holdings = {}; trades = []; nav_history = []
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
                trades.append({'code':code,'buy_date':pos['initial_buy_date'],
                                'cum_pnl':pos['cum_pnl'],
                                'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100})
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
        nav_history.append({'date':today, 'nav':cash+mv})

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
            trades.append({'code':code,'buy_date':pos['initial_buy_date'],
                            'cum_pnl':pos['cum_pnl'],
                            'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100})

    df_t = pd.DataFrame(trades)
    df_n = pd.DataFrame(nav_history)
    final = df_n['nav'].iloc[-1]
    days = (pd.to_datetime(df_n['date'].iloc[-1]) - pd.to_datetime(df_n['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_n['peak'] = df_n['nav'].cummax()
    mdd = ((df_n['nav']-df_n['peak'])/df_n['peak']*100).min()
    win = (df_t['cum_ret_pct']>0).mean()*100 if len(df_t) else 0
    # 按年 PnL
    if len(df_t) > 0:
        df_t['buy_year'] = pd.to_datetime(df_t['buy_date']).dt.year
        yr_pnl = df_t.groupby('buy_year').agg(
            n=('cum_pnl', 'count'),
            avg=('cum_ret_pct', 'mean'),
            pnl=('cum_pnl', 'sum'),
            win=('cum_ret_pct', lambda x: (x>0).mean()*100),
        ).to_dict('index')
    else:
        yr_pnl = {}
    return {'total':(final/INIT_CAPITAL-1)*100,'annual':annual,'mdd':mdd,
            'n':len(df_t),'win':win,'final':final, 'yr_pnl': yr_pnl}


def main():
    t0 = time.time()
    print('=== test161: v5 + 坤 regime — 池深单因子 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3).replace({'nan':''})
    g.rename(columns={'d_gua':'stk_d'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    mkt['mkt_y'] = mkt['y_gua'].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt[['date', 'mkt_y']].drop_duplicates('date')

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

    df_e = find_signals_v5(arrays)
    print(f'  v5 入场事件: {len(df_e):,}')

    # 算单笔 ret%
    print(f'  计算单笔 ret%...')
    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets
    df_e['year'] = df_e['date'].str[:4]

    baseline_avg = df_e['ret_pct'].mean()
    baseline_win = (df_e['ret_pct']>0).mean()*100
    print(f'  baseline: avg={baseline_avg:+.2f}%, win={baseline_win:.1f}%, '
          f'p10={df_e["ret_pct"].quantile(0.1):+.1f}, p90={df_e["ret_pct"].quantile(0.9):+.1f}')

    # ===== L1: 单笔 ret% 拆桶 =====
    print(f'\n{"="*70}')
    print(f'  L1: 池深拆 8 桶 (单笔 ret%)')
    print(f'{"="*70}')
    print(f'\n  {"bucket":<8} {"retail 范围":<22} {"n":>5} {"avg":>8} {"lift":>8} {"win":>6} {"年 lift (16/17/18)":>22}')

    df_e['pool_q8'] = pd.qcut(df_e['pool_min_retail'], 8, labels=[f'q{i+1}' for i in range(8)])
    years = sorted(df_e['year'].unique())
    for q, sub in df_e.groupby('pool_q8', observed=True):
        avg = sub['ret_pct'].mean()
        lift = avg - baseline_avg
        win = (sub['ret_pct']>0).mean()*100
        rng_lo = sub['pool_min_retail'].min()
        rng_hi = sub['pool_min_retail'].max()
        # 跨年 lift
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_e[df_e['year']==y]
            if len(ys) >= 10 and len(yb) >= 30:
                yl.append(f'{ys["ret_pct"].mean() - yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  {q:<8} [{rng_lo:>+5.0f}, {rng_hi:>+5.0f}]    {len(sub):>5} '
              f'{avg:>+7.2f}% {lift:>+7.2f}% {win:>5.1f}%  {" ".join(yl)}')

    # ===== L2: 阈值排雷 =====
    print(f'\n{"="*70}')
    print(f'  L2: 池深阈值排雷 (单笔 ret%)')
    print(f'{"="*70}')
    print(f'\n  {"阈值":<18} {"保留 n":>7} {"avg":>8} {"lift":>8} {"win":>6} {"年 lift":>22}')
    for thr in [-200, -250, -300, -400, -500, -700, -1000]:
        sub = df_e[df_e['pool_min_retail'] <= thr]
        if len(sub) < 50: continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_e[df_e['year']==y]
            if len(ys) >= 10 and len(yb) >= 30:
                yl.append(f'{ys["ret_pct"].mean() - yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  retail <= {thr:>+5}    {len(sub):>7} {avg:>+7.2f}% {avg-baseline_avg:>+7.2f}% '
              f'{win:>5.1f}%  {" ".join(yl)}')

    # ===== L3: 同日排名资金回测 =====
    print(f'\n{"="*70}')
    print(f'  L3+L4: 同日排名 资金回测 (K=5) + 按年 PnL')
    print(f'{"="*70}')

    K = 5
    configs = [
        ('baseline 池深↑(asc) top-1', None, True),
        ('池深↓(desc) top-1',         None, False),
        ('retail<=-300 + 池深↑',       -300, True),
        ('retail<=-400 + 池深↑',       -400, True),
        ('retail<=-500 + 池深↑',       -500, True),
    ]
    print(f'\n  {"配置":<28} {"信号天":>5} {"总":>9} {"年化":>8} {"MDD":>9} {"段":>4} {"胜":>6}')
    yr_table = {}
    for name, thr, asc in configs:
        df_filt = df_e if thr is None else df_e[df_e['pool_min_retail'] <= thr]
        if len(df_filt) < 100: continue
        df_picks = df_filt.sort_values(['date', 'pool_min_retail', 'code'],
                                         ascending=[True, asc, True]).drop_duplicates('date', keep='first')
        r = run_backtest(K, df_picks, code_date_idx, code_arr, date_arr,
                          close_arr, trend_arr, mf_arr, retail_arr, all_dates)
        yr_table[name] = r['yr_pnl']
        print(f'  {name:<28} {len(df_picks):>5} {r["total"]:>+7.1f}% {r["annual"]:>+6.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n"]:>4} {r["win"]:>5.1f}%')

    # ===== L4: 按年 PnL 矩阵 =====
    print(f'\n{"="*70}')
    print(f'  L4: 按入场年 PnL (单段平均 ret%)')
    print(f'{"="*70}')
    all_years = sorted(set(y for r in yr_table.values() for y in r.keys()))
    print(f'\n  {"配置":<28}', end='')
    for y in all_years:
        print(f' {y:>16}', end='')
    print()
    for name, yr_pnl in yr_table.items():
        print(f'  {name:<28}', end='')
        for y in all_years:
            if y in yr_pnl:
                d = yr_pnl[y]
                print(f' n={d["n"]:>3}{d["avg"]:>+5.1f}%{d["pnl"]/1000:>+5.0f}K', end='')
            else:
                print(f' {"--":>16}', end='')
        print()

    # 按入场年累计 NAV (取最强配置)
    print(f'\n{"="*70}')
    print(f'  最佳: retail<=-400 + 池深↑ — IS/OOS 拆分')
    print(f'{"="*70}')
    best_name = 'retail<=-400 + 池深↑'
    if best_name in yr_table:
        yp = yr_table[best_name]
        is_pnl = sum(yp[y]['pnl'] for y in yp if y <= 2017)
        oos_pnl = sum(yp[y]['pnl'] for y in yp if y >= 2018)
        is_n = sum(yp[y]['n'] for y in yp if y <= 2017)
        oos_n = sum(yp[y]['n'] for y in yp if y >= 2018)
        print(f'\n  IS (2015-2017): n={is_n}, pnl=¥{is_pnl/1000:+.0f}K')
        print(f'  OOS (2018+):    n={oos_n}, pnl=¥{oos_pnl/1000:+.0f}K')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
