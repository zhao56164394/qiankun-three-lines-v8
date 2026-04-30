# -*- coding: utf-8 -*-
"""复利资金回测对比: ABC / score>=2 / score>=3 / hard / ABC ∩ score>=2

复利逻辑:
  每次新建仓时 SLOT_VALUE = NAV_now / K (动态)
  NAV_now = cash + 持仓市值
  这样盈利会滚雪球, 亏损也放大
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


def find_signals(starts, ends, td, date_arr, code_arr,
                 mf, retail, stk_d_t, stk_m_t, stk_y_t,
                 mkt_d_t, mkt_m_t, mkt_y_t):
    sigs = []
    for ci in range(len(starts)):
        s = starts[ci]; e = ends[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            if np.isnan(td[i]) or np.isnan(td[i-1]): continue
            if td[i-1] > 11 and td[i] <= 11:
                stk_yy = yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i])
                mkt_yy = yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i])
                pa = (mkt_yy == '101' and stk_yy == '011')
                pb = (mkt_yy == '011' and stk_yy == '001')
                pc = (mkt_yy == '001' and stk_yy == '001')
                in_abc = pa or pb or pc
                plan = 'A' if pa else ('B' if pb else ('C' if pc else '-'))
                sigs.append({
                    'signal_idx': i, 'signal_date': date_arr[i], 'code': code_arr[i],
                    'cur_mf_sig': mf[i], 'cur_retail_sig': retail[i],
                    'mkt_yy_sig': mkt_yy, 'stk_yy_sig': stk_yy,
                    'in_abc': in_abc, 'plan': plan,
                    'code_end': e, 'code_start': s,
                })
    return pd.DataFrame(sigs)


def find_entry(signal_idx, code_end, td, retail):
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if np.isnan(td[k]) or np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if td[k] <= 11: continue
        if retail[k-1] <= 0 and retail[k] > 0: return k
    return -1


def feats_at_entry(ei, code_start, td, mf, retail, stk_d_t, stk_m_t, stk_y_t):
    stk_yy = yyy(stk_d_t[ei], stk_m_t[ei], stk_y_t[ei])
    s5 = max(code_start, ei - 4)
    s30 = max(code_start, ei - 29)
    return {
        'stk_yy_e': stk_yy,
        'cur_mf': mf[ei],
        'mf5': np.nanmean(mf[s5:ei+1]),
        'rt5': np.nanmean(retail[s5:ei+1]),
        'mf30_min': np.nanmin(mf[s30:ei+1]),
        'mf30_max': np.nanmax(mf[s30:ei+1]),
        'rt30_min': np.nanmin(retail[s30:ei+1]),
    }


def calc_score(f):
    s = 0
    if f['stk_yy_e'] == '011': s += 1
    if not np.isnan(f['cur_mf']) and f['cur_mf'] >= 100: s += 1
    if not np.isnan(f['mf30_max']) and f['mf30_max'] >= 200: s += 1
    if not np.isnan(f['mf30_min']) and f['mf30_min'] <= -200: s += 1
    if not np.isnan(f['rt30_min']) and f['rt30_min'] <= -200: s += 1
    if not np.isnan(f['mf5']) and f['mf5'] >= 100: s += 1
    if not np.isnan(f['rt5']) and f['rt5'] <= -100: s += 1
    return s


def hard_filter(f):
    if np.isnan(f['cur_mf']) or f['cur_mf'] < 100: return False
    if np.isnan(f['mf30_max']) or f['mf30_max'] < 200: return False
    if np.isnan(f['rt30_min']) or f['rt30_min'] > -200: return False
    return True


def simulate_F(buy_idx, code_end, td, close, mf, retail):
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return (cum-1)*100, legs
        if k < 1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        mfc = mf[k]-mf[k-1]
        if holding:
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc > 0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return (cum-1)*100, legs


def _mk_trade(code, pos, sd, days, reason):
    return {'code': code, 'tag': pos.get('tag','?'),
            'buy_date': pos['initial_buy_date'], 'sell_date': sd,
            'cum_pnl': pos['cum_pnl'],
            'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
            'days': days, 'legs': pos['legs'], 'reason': reason}


def run_compound(K, picks, code_date_idx, close_arr, td_arr, mf_arr, retail_arr, all_dates):
    """复利回测: 每次新建仓 SLOT = NAV/K"""
    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []
    picks_by_date = {}
    for _, p in picks.iterrows():
        d = p['entry_date']
        picks_by_date.setdefault(d, []).append(p.to_dict())

    def calc_mv(today):
        mv = 0.0
        for c, pos in holdings.items():
            if pos['state'] == 'holding':
                if c in code_date_idx and today in code_date_idx[c]:
                    ti = code_date_idx[c][today]
                    mv += pos['shares'] * close_arr[ti]
                else:
                    mv += pos['shares'] * pos['cur_buy_price']
            else:
                mv += pos['cash_at_pending']
        return mv

    for today in all_dates:
        # 1. 持仓 F
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]: continue
            ti = code_date_idx[code][today]
            cc = close_arr[ti]
            if not np.isnan(td_arr[ti]) and td_arr[ti] < 11:
                if pos['state'] == 'holding':
                    proceeds = pos['shares'] * cc
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds; pos['cum_pnl'] += profit
                trades.append(_mk_trade(code, pos, today, ti - pos['initial_buy_idx'], 'T0'))
                del holdings[code]; continue
            if ti < 1: continue
            if np.isnan(retail_arr[ti]) or np.isnan(retail_arr[ti-1]): continue
            if np.isnan(mf_arr[ti]) or np.isnan(mf_arr[ti-1]): continue
            mfc = mf_arr[ti] - mf_arr[ti-1]
            if pos['state'] == 'holding':
                if retail_arr[ti-1] <= 0 and retail_arr[ti] <= 0:
                    proceeds = pos['shares'] * cc
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds; pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:
                if mfc > 0:
                    bp = cc
                    if not np.isnan(bp) and bp > 0:
                        avail = pos['cash_at_pending']
                        sh = int(avail // bp // 100) * 100
                        if sh > 0:
                            cost = sh * bp
                            cash -= cost
                            pos['shares'] = sh
                            pos['cur_buy_price'] = bp
                            pos['state'] = 'holding'
                            pos['legs'] += 1
                            pos['cash_at_pending'] = avail - cost

        # 2. 新建仓 — 复利
        if today in picks_by_date and len(holdings) < K:
            cands = picks_by_date[today]
            for cand in cands:
                if len(holdings) >= K: break
                code = cand['code']
                if code in holdings: continue
                if code not in code_date_idx or today not in code_date_idx[code]: continue
                ridx = code_date_idx[code][today]
                bp = close_arr[ridx]
                if np.isnan(bp) or bp <= 0: continue
                # 当前 NAV
                mv_now = calc_mv(today)
                nav_now = cash + mv_now
                slot_value = nav_now / K
                sh = int(slot_value // bp // 100) * 100
                cost = sh * bp
                if cost > cash:
                    sh = int(cash // bp // 100) * 100
                    cost = sh * bp
                if sh <= 0: continue
                cash -= cost
                holdings[code] = {
                    'state': 'holding', 'tag': cand.get('tag','?'),
                    'initial_buy_date': today, 'initial_buy_idx': ridx,
                    'initial_buy_price': bp, 'initial_cost': cost,
                    'cur_buy_price': bp, 'shares': sh,
                    'cash_at_pending': 0, 'cum_pnl': 0, 'legs': 1,
                }

        # 3. NAV
        mv = calc_mv(today)
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
            trades.append(_mk_trade(code, pos, last, ti - pos['initial_buy_idx'], 'fc'))

    df_t = pd.DataFrame(trades); df_n = pd.DataFrame(nav_history)
    if len(df_n) == 0:
        return {'final':INIT_CAPITAL,'total':0,'annual':0,'mdd':0,'n_trades':0,
                'win':0,'avg':0,'avg_pos':0,'df_t':df_t,'df_n':df_n}
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


def main():
    t0 = time.time()
    print('=== test192: 复利资金回测对比 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code','board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board']=='主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date','code','d_trend','m_trend','y_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date','d_trend','m_trend','y_trend'])
    mkt['date'] = mkt['date'].astype(str)
    mkt = mkt.drop_duplicates('date').rename(columns={
        'd_trend':'mkt_d_t','m_trend':'mkt_m_t','y_trend':'mkt_y_t'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date','code','close','retail','main_force'])
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
    td_arr = df['d_trend'].to_numpy().astype(np.float64)
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

    print('  扫信号 + entry...')
    df_sig = find_signals(code_starts, code_ends, td_arr, date_arr, code_arr,
                           mf_arr, retail_arr, stk_d_t, stk_m_t, stk_y_t,
                           mkt_d_t, mkt_m_t, mkt_y_t)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'    全市场起点: {len(df_sig):,} (含 ABC: {df_sig["in_abc"].sum():,})')

    rows = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end']); cs = int(s['code_start'])
        ei = find_entry(si, ce, td_arr, retail_arr)
        if ei < 0: continue
        f = feats_at_entry(ei, cs, td_arr, mf_arr, retail_arr, stk_d_t, stk_m_t, stk_y_t)
        f['signal_idx'] = si; f['entry_idx'] = ei
        f['entry_date'] = date_arr[ei]
        f['code'] = s['code']; f['code_end'] = ce
        f['cur_retail_e'] = retail_arr[ei]
        f['in_abc'] = s['in_abc']; f['plan'] = s['plan']
        f['score'] = calc_score(f)
        f['hard'] = hard_filter(f)
        rows.append(f)
    df_e = pd.DataFrame(rows)
    print(f'    F entry: {len(df_e):,}')

    print('  单笔模拟...')
    rets = []; legs_l = []
    for _, p_ in df_e.iterrows():
        ret, legs = simulate_F(int(p_['entry_idx']), int(p_['code_end']),
                                td_arr, close_arr, mf_arr, retail_arr)
        rets.append(ret); legs_l.append(legs)
    df_e['ret_pct'] = rets
    df_e['legs'] = legs_l

    versions = {
        'ABC':         df_e[df_e['in_abc']].copy(),
        'score>=2':    df_e[df_e['score'] >= 2].copy(),
        'score>=3':    df_e[df_e['score'] >= 3].copy(),
        'hard_AND':    df_e[df_e['hard']].copy(),
        'ABC ∩ s>=2':  df_e[df_e['in_abc'] & (df_e['score']>=2)].copy(),
        'ABC ∩ s>=3':  df_e[df_e['in_abc'] & (df_e['score']>=3)].copy(),
        'ABC ∪ hard':  df_e[df_e['in_abc'] | df_e['hard']].copy(),
    }

    print(f'\n  --- 单笔总览 ---')
    print(f'  {"版本":<14} {"n":>7} {"avg":>7} {"win%":>6} {"≥+50":>5} {"≥+100":>6} {"≥+200":>6} {"暴涨率":>7}')
    for name, dfv in versions.items():
        n = len(dfv)
        if n == 0: continue
        avg = dfv['ret_pct'].mean()
        win = (dfv['ret_pct']>0).mean()*100
        h50 = (dfv['ret_pct']>=50).sum()
        h100 = (dfv['ret_pct']>=100).sum()
        h200 = (dfv['ret_pct']>=200).sum()
        rate = h100/n*100 if n else 0
        print(f'  {name:<14} {n:>7,} {avg:>+5.2f}% {win:>5.1f}% '
              f'{h50:>5} {h100:>6} {h200:>6} {rate:>+6.2f}%')

    # 资金回测 (复利)
    all_dates = sorted(set(date_arr.tolist()))
    print(f'\n  ============ 复利资金回测 ============')

    summary = []
    for name, dfv in versions.items():
        if len(dfv) == 0: continue
        dfv = dfv.copy()
        dfv['neg_score'] = -dfv['score']
        df_picks = dfv.sort_values(['entry_date','neg_score','cur_retail_e','code']).reset_index(drop=True)
        df_picks['code_end'] = df_picks['code_end'].astype(int)
        df_picks['tag'] = name
        print(f'\n  --- 版本 [{name}] (n_picks={len(df_picks):,}) ---')
        print(f'  {"K":<3} {"final":>10} {"总":>9} {"年化":>7} {"MDD":>7} {"段":>5} {"胜":>6} {"avg":>7} {"pos":>6}')
        for K in [1, 2, 3, 5]:
            r = run_compound(K, df_picks, code_date_idx, close_arr, td_arr, mf_arr, retail_arr, all_dates)
            print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% '
                  f'{r["annual"]:>+5.2f}% {r["mdd"]:>+5.1f}% {r["n_trades"]:>5} '
                  f'{r["win"]:>5.1f}% {r["avg"]:>+5.2f}% {r["avg_pos"]:>5.1f}%')
            summary.append({'ver':name, 'K':K, 'final':r['final'], 'total':r['total'],
                            'annual':r['annual'], 'mdd':r['mdd'], 'segs':r['n_trades']})

    print(f'\n\n  ============ 全版本最佳 K 总结 ============')
    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
