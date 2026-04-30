# -*- coding: utf-8 -*-
"""3 池 (A=101+011, B=011+001, C=001+001) 并行 + F 机制资金回测

入场:
  3 个池子任一满足就入场, 同日多个候选按 cur_retail 升序选 (越深越优先)
  签到日 = trend 下穿 11 那天
  建仓日: 散户线上穿 0 + trend>11 (60d 内)

机制 (F):
  卖出 (D6'): 散户线连续 2 天 <=0
  再买入 (U1): mf 上升
  T0: trend < 11

资金回测:
  K = 1, 2, 3, 5
  按建仓日排序选股
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
WAIT_MAX = 60


def yyy(d, m, y, thr=50):
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def find_signals(arrays, mkt_arrs):
    """所有波段起点 + 阴阳"""
    cs = arrays['starts']; ce = arrays['ends']
    td = arrays['td']; close = arrays['close']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d_t = arrays['stk_d_t']; stk_m_t = arrays['stk_m_t']; stk_y_t = arrays['stk_y_t']
    mkt_d_t = mkt_arrs['mkt_d_t']; mkt_m_t = mkt_arrs['mkt_m_t']; mkt_y_t = mkt_arrs['mkt_y_t']
    date = arrays['date']; code = arrays['code']
    sigs = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur) or np.isnan(prev): continue
            if prev > 11 and cur <= 11:
                mkt_y = yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i])
                stk_y = yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i])
                # Plan A/B/C 任一满足
                pa = (mkt_y == '101' and stk_y == '011')
                pb = (mkt_y == '011' and stk_y == '001')
                pc = (mkt_y == '001' and stk_y == '001')
                if pa or pb or pc:
                    plan = 'A' if pa else ('B' if pb else 'C')
                    sigs.append({'signal_idx': i,'signal_date': date[i],'code': code[i],
                                 'cur_mf': mf[i], 'cur_retail': retail[i],
                                 'mkt_yy': mkt_y, 'stk_yy': stk_y, 'plan': plan,
                                 'code_end': e})
    return pd.DataFrame(sigs)


def find_entry_retail_cross(signal_idx, code_end, td, retail):
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if td[k] <= 11: continue
        if retail[k-1] <= 0 and retail[k] > 0:
            return k
    return -1


def simulate_F(buy_idx, code_end, td, close, mf, retail):
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    sell_idx = code_end - 1; reason = 'fc'
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return k, 'T0', (cum-1)*100, legs
        if k<1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        mfc = mf[k]-mf[k-1]
        if holding:
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc>0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return code_end-1, 'fc', (cum-1)*100, legs


def run_capital_backtest(K, picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates):
    """F 机制资金回测"""
    SLOT_VALUE = INIT_CAPITAL / K
    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []
    picks_by_date = {}
    for _, p in picks.iterrows():
        d = p['entry_date']
        picks_by_date.setdefault(d, []).append(p.to_dict())

    for today in all_dates:
        # 1. 处理已持仓 (F 机制)
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]: continue
            today_idx = code_date_idx[code][today]
            cur_close = close_arr[today_idx]

            # T0
            if not np.isnan(trend_arr[today_idx]) and trend_arr[today_idx] < 11:
                if pos['state'] == 'holding':
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds; pos['cum_pnl'] += profit
                trades.append(_make_trade(code, pos, today, today_idx - pos['initial_buy_idx'], 'T0'))
                del holdings[code]
                continue

            if today_idx < 1: continue
            if np.isnan(retail_arr[today_idx]) or np.isnan(retail_arr[today_idx-1]): continue
            if np.isnan(mf_arr[today_idx]) or np.isnan(mf_arr[today_idx-1]): continue

            mfc = mf_arr[today_idx] - mf_arr[today_idx-1]

            if pos['state'] == 'holding':
                # F 卖: 散户线连续 2 天 <=0
                if retail_arr[today_idx-1] <= 0 and retail_arr[today_idx] <= 0:
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:
                # U1 买回: mf 上升
                if mfc > 0:
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

        # 2. 新建仓
        if today in picks_by_date and len(holdings) < K:
            cands = picks_by_date[today]
            for cand in cands:
                if len(holdings) >= K: break
                code = cand['code']
                if code in holdings: continue
                if code not in code_date_idx or today not in code_date_idx[code]: continue
                ridx = code_date_idx[code][today]
                buy_price = close_arr[ridx]
                if np.isnan(buy_price) or buy_price <= 0: continue
                shares = int(SLOT_VALUE // buy_price // 100) * 100
                if shares <= 0: continue
                cost = shares * buy_price
                if cost > cash: continue
                cash -= cost
                holdings[code] = {
                    'state': 'holding', 'plan': cand['plan'],
                    'initial_buy_date': today, 'initial_buy_idx': ridx,
                    'initial_buy_price': buy_price, 'initial_cost': cost,
                    'cur_buy_price': buy_price, 'shares': shares,
                    'cash_at_pending': 0, 'cum_pnl': 0, 'legs': 1,
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
                cash += proceeds; pos['cum_pnl'] += profit
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
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'avg_pos':avg_pos,
            'df_t':df_t, 'df_n':df_n}


def _make_trade(code, pos, sell_date, days, reason):
    return {
        'code': code, 'plan': pos.get('plan', '?'),
        'buy_date': pos['initial_buy_date'], 'sell_date': sell_date,
        'cum_pnl': pos['cum_pnl'],
        'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
        'days': days, 'legs': pos['legs'], 'reason': reason,
    }


def main():
    t0 = time.time()
    print('=== test188: 3 池并行 (101+011 / 011+001 / 001+001) + F 机制资金回测 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'm_trend', 'y_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_trend', 'm_trend', 'y_trend'])
    mkt['date'] = mkt['date'].astype(str)
    mkt = mkt.drop_duplicates('date').rename(columns={
        'd_trend':'mkt_d_t', 'm_trend':'mkt_m_t', 'y_trend':'mkt_y_t'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_d_t = df['d_trend'].to_numpy().astype(np.float64)
    stk_m_t = df['m_trend'].to_numpy().astype(np.float64)
    stk_y_t = df['y_trend'].to_numpy().astype(np.float64)
    mkt_d_t = df['mkt_d_t'].to_numpy().astype(np.float64)
    mkt_m_t = df['mkt_m_t'].to_numpy().astype(np.float64)
    mkt_y_t = df['mkt_y_t'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    arrays = {'code':code_arr,'date':date_arr,'close':close_arr,'td':trend_arr,
              'retail':retail_arr,'mf':mf_arr,
              'stk_d_t':stk_d_t,'stk_m_t':stk_m_t,'stk_y_t':stk_y_t,
              'starts':code_starts,'ends':code_ends}
    mkt_arrs = {'mkt_d_t':mkt_d_t, 'mkt_m_t':mkt_m_t, 'mkt_y_t':mkt_y_t}

    print('  扫信号...')
    df_sig = find_signals(arrays, mkt_arrs)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'    A+B+C 信号: {len(df_sig):,}')
    for plan, g_ in df_sig.groupby('plan'):
        print(f'      Plan {plan}: {len(g_)}')

    # 找建仓日
    print('  扫建仓日...')
    entries = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end'])
        ei = find_entry_retail_cross(si, ce, trend_arr, retail_arr)
        if ei < 0:
            entries.append({'entry_idx': -1, 'entry_date': None})
        else:
            entries.append({'entry_idx': ei, 'entry_date': date_arr[ei]})
    df_sig['entry_idx'] = [e['entry_idx'] for e in entries]
    df_sig['entry_date'] = [e['entry_date'] for e in entries]

    n_entry = (df_sig['entry_idx'] >= 0).sum()
    print(f'    成功建仓: {n_entry:,} / {len(df_sig)} ({n_entry/len(df_sig)*100:.1f}%)')

    df_picks = df_sig[df_sig['entry_idx'] >= 0].copy()
    df_picks = df_picks.sort_values(['entry_date', 'cur_retail', 'code']).reset_index(drop=True)
    df_picks['code_end'] = df_picks['code_end'].astype(int)

    # 单笔模拟
    print('\n  单笔模拟评估...')
    rets = []; legs_l = []; reasons = []
    for _, p in df_picks.iterrows():
        ei = int(p['entry_idx']); ce = int(p['code_end'])
        sk, r, ret, legs = simulate_F(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
        rets.append(ret); legs_l.append(legs); reasons.append(r)
    df_picks['ret_pct'] = rets
    df_picks['legs'] = legs_l
    df_picks['reason'] = reasons
    df_picks['year'] = df_picks['entry_date'].str[:4]

    print(f'\n  --- 单笔总览 ---')
    print(f'    n={len(df_picks)}')
    print(f'    avg ret: {df_picks["ret_pct"].mean():+.2f}%')
    print(f'    median: {df_picks["ret_pct"].median():+.2f}%')
    print(f'    win: {(df_picks["ret_pct"]>0).mean()*100:.1f}%')
    print(f'    ≥+50%: {(df_picks["ret_pct"]>=50).sum()}')
    print(f'    ≥+100%: {(df_picks["ret_pct"]>=100).sum()} ({(df_picks["ret_pct"]>=100).mean()*100:.2f}%)')
    print(f'    ≥+200%: {(df_picks["ret_pct"]>=200).sum()}')
    print(f'    avg legs: {df_picks["legs"].mean():.1f}')

    print(f'\n  --- 按 Plan ---')
    for plan, g_ in df_picks.groupby('plan'):
        n = len(g_)
        print(f'    Plan {plan}: n={n}, avg={g_["ret_pct"].mean():+5.2f}%, '
              f'win={(g_["ret_pct"]>0).mean()*100:5.1f}%, ≥+100%={(g_["ret_pct"]>=100).sum()}, '
              f'≥+200%={(g_["ret_pct"]>=200).sum()}')

    print(f'\n  --- 按年 ---')
    for y, g_ in df_picks.groupby('year'):
        n = len(g_); h100 = (g_['ret_pct']>=100).sum()
        print(f'    {y}: n={n:>4}, avg={g_["ret_pct"].mean():+5.2f}%, '
              f'win={(g_["ret_pct"]>0).mean()*100:5.1f}%, ≥+100%={h100}')

    # 资金回测
    print(f'\n  资金回测...\n')
    all_dates = sorted(set(date_arr.tolist()))
    print(f'  {"K":<3} {"final":>10} {"总":>9} {"年化":>7} {"MDD":>7} {"段":>4} {"胜":>6} {"avg":>7} {"pos":>6}')
    for K in [1, 2, 3, 5]:
        r = run_capital_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% '
              f'{r["annual"]:>+5.2f}% {r["mdd"]:>+5.1f}% {r["n_trades"]:>4} '
              f'{r["win"]:>5.1f}% {r["avg"]:>+5.2f}% {r["avg_pos"]:>5.1f}%')

    # 详细 K=2 按年
    print(f'\n  --- K=2 详细 ---')
    r = run_capital_backtest(2, df_picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates)
    df_t = r['df_t']
    if len(df_t) > 0:
        df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year
        print(f'    {"年":<6} {"段":>4} {"avg":>7} {"win":>6} {"≥+100":>5} {"pnl":>10}')
        for y, g_ in df_t.groupby('year'):
            avg = g_['cum_ret_pct'].mean()
            win = (g_['cum_ret_pct']>0).mean()*100
            h100 = (g_['cum_ret_pct']>=100).sum()
            print(f'    {y:<6} {len(g_):>4} {avg:>+5.2f}% {win:>5.1f}% {h100:>5} ¥{g_["cum_pnl"].sum()/1000:>+5.0f}K')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_picks.to_csv(os.path.join(out_dir, 'plan_abc_picks.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 plan_abc_picks.csv')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
