# -*- coding: utf-8 -*-
"""新策略 + 老建仓/卖出机制 资金回测

入场预筛 (波段起点, signal_date):
  - mkt 阴阳 = 000 (mkt_d/m/y_t 全 ≤50)
  - stk 阴阳 = 011 (stk_d_t ≤50, stk_m_t > 50, stk_y_t > 50)
  - cur_mf ≤ -100 AND cur_retail ≤ -100
  - 触发日 = trend 下穿 11 当天

建仓 (entry_date, 在波段内):
  - 从 signal_date+1 开始, 等到第一次满足:
    mf 上升 (chg>0) AND retail 上升 (chg>0) AND trend > 11
  - 或波段终结 (下次 trend 下穿 11) 还没建仓 → 放弃

之后:
  - D6 卖: mf_chg<0 AND retail_chg<0 AND trend_chg<0
  - U1 买: mf_chg>0
  - T0 清仓: trend < 11 (当前波段终结)

资金回测:
  - INIT_CAPITAL = 200,000
  - K = 2 (单股 50% 仓位 / K)
  - 同日多个 signal 按 cur_mf 升序选最低 (越深越优先)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
WAIT_MAX = 60  # 等待 trend 重新上穿 11 最多 60 天


def find_signals(arrays, mkt_arrs):
    """找波段起点 (trend 下穿 11) 满足新策略 4 条件的 signal_date"""
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
                # 波段起点
                # 检查 4 条件
                cond_mkt = (not np.isnan(mkt_d_t[i]) and mkt_d_t[i] <= 50 and
                              not np.isnan(mkt_m_t[i]) and mkt_m_t[i] <= 50 and
                              not np.isnan(mkt_y_t[i]) and mkt_y_t[i] <= 50)
                cond_stk = (not np.isnan(stk_d_t[i]) and stk_d_t[i] <= 50 and
                              not np.isnan(stk_m_t[i]) and stk_m_t[i] > 50 and
                              not np.isnan(stk_y_t[i]) and stk_y_t[i] > 50)
                cond_mf = (not np.isnan(mf[i]) and mf[i] <= -100)
                cond_ret = (not np.isnan(retail[i]) and retail[i] <= -100)
                if cond_mkt and cond_stk and cond_mf and cond_ret:
                    sigs.append({
                        'signal_idx': i,
                        'signal_date': date[i],
                        'code': code[i],
                        'cur_mf': mf[i],
                        'cur_retail': retail[i],
                        'code_end': e,
                    })
    return pd.DataFrame(sigs)


def find_entry(signal_idx, code_end, td, mf, retail, max_wait=WAIT_MAX):
    """从 signal_idx+1 开始, 找第一个满足建仓条件的日期
       返回 (entry_idx, reason). 找不到返回 (-1, 'no_entry')
       建仓: trend 上穿 11 后 mf 上升 + retail 上升 + trend > 11
    """
    end_search = min(code_end - 1, signal_idx + max_wait)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(td[k-1]): continue
        # 等到 trend 重新 > 11
        if td[k] <= 11: continue
        # 检查建仓
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        mf_c = mf[k] - mf[k-1]
        ret_c = retail[k] - retail[k-1]
        if mf_c > 0 and ret_c > 0:
            return k, 'entry_ok'
        # trend 上穿 11 后, 还要等 mf+retail 上升 — 继续找
    return -1, 'no_entry'


def simulate_t0(buy_idx, code_end, td, close, mf, retail):
    """从建仓日 buy_idx 开始 D6 卖 / U1 买, T0 清仓 (trend<11)
       返回 sell_idx, reason, ret_pct, legs
    """
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 1

    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue

        # T0 清仓
        if td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
            return k, 'td<11', (cum_mult-1)*100, legs

        if k < 1: continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(td[k-1]): continue

        mf_c = mf[k] - mf[k-1]
        ret_c = retail[k] - retail[k-1]
        td_c = td[k] - td[k-1]

        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                # D6 卖
                cum_mult *= close[k] / cur_buy_price
                holding = False
        else:
            if mf_c > 0:
                # U1 买
                cur_buy_price = close[k]
                holding = True
                legs += 1

    # 段末未清, 强平
    if holding:
        cum_mult *= close[code_end-1] / cur_buy_price
    return code_end-1, 'fc', (cum_mult-1)*100, legs


def run_capital_backtest(K, picks, code_date_idx, code_arr, date_arr, close_arr,
                          trend_arr, mf_arr, retail_arr, all_dates, code_ends_map):
    """完整资金回测
       picks: DataFrame with columns: entry_date, entry_idx, code, code_end, signal_date
              已按 entry_date 排序, 同日多股按 cur_mf 升序
    """
    SLOT_VALUE = INIT_CAPITAL / K
    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []

    # picks_by_date: dict[date -> list of dicts]
    picks_by_date = {}
    for _, p in picks.iterrows():
        d = p['entry_date']
        picks_by_date.setdefault(d, []).append(p.to_dict())

    for today in all_dates:
        # 1. 处理已持仓
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]:
                continue
            today_idx = code_date_idx[code][today]
            cur_close = close_arr[today_idx]

            # T0 清仓
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
            if np.isnan(mf_arr[today_idx]) or np.isnan(mf_arr[today_idx-1]):
                continue
            if np.isnan(retail_arr[today_idx]) or np.isnan(retail_arr[today_idx-1]):
                continue
            if np.isnan(trend_arr[today_idx]) or np.isnan(trend_arr[today_idx-1]):
                continue

            mf_c = mf_arr[today_idx] - mf_arr[today_idx-1]
            ret_c = retail_arr[today_idx] - retail_arr[today_idx-1]
            td_c = trend_arr[today_idx] - trend_arr[today_idx-1]

            if pos['state'] == 'holding':
                if mf_c < 0 and ret_c < 0 and td_c < 0:
                    # D6 卖
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:  # pending
                if mf_c > 0:
                    # U1 买回
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

        # 2. 处理新建仓
        if today in picks_by_date and len(holdings) < K:
            cands = picks_by_date[today]
            for cand in cands:
                if len(holdings) >= K: break
                code = cand['code']
                if code in holdings: continue
                if code not in code_date_idx or today not in code_date_idx[code]:
                    continue
                ridx = code_date_idx[code][today]
                buy_price = close_arr[ridx]
                if np.isnan(buy_price) or buy_price <= 0: continue
                shares = int(SLOT_VALUE // buy_price // 100) * 100
                if shares <= 0: continue
                cost = shares * buy_price
                if cost > cash: continue
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

    # 段末清算
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
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'avg_pos':avg_pos,
            'df_t':df_t, 'df_n':df_n}


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


def main():
    t0 = time.time()
    print('=== test181: 新策略 + D6/U1/T0 资金回测 ===\n')

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
              'stk_d_t':stk_d_t, 'stk_m_t':stk_m_t, 'stk_y_t':stk_y_t,
              'starts':code_starts,'ends':code_ends}
    mkt_arrs = {'mkt_d_t':mkt_d_t, 'mkt_m_t':mkt_m_t, 'mkt_y_t':mkt_y_t}

    print('  扫描信号...')
    df_sig = find_signals(arrays, mkt_arrs)
    print(f'    波段起点信号: {len(df_sig):,}')

    # 限 2016+
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'    2016+ 信号: {len(df_sig):,}')

    # 找建仓日
    print('  扫描建仓日...')
    entries = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end'])
        ei, reason = find_entry(si, ce, trend_arr, mf_arr, retail_arr)
        if ei < 0:
            entries.append({'entry_idx': -1, 'entry_date': None, 'reason': reason})
        else:
            entries.append({'entry_idx': ei, 'entry_date': date_arr[ei], 'reason': reason})
    df_sig = df_sig.reset_index(drop=True)
    df_sig['entry_idx'] = [e['entry_idx'] for e in entries]
    df_sig['entry_date'] = [e['entry_date'] for e in entries]
    df_sig['entry_reason'] = [e['reason'] for e in entries]

    n_entry = (df_sig['entry_idx'] >= 0).sum()
    print(f'    成功建仓: {n_entry:,} / {len(df_sig)} ({n_entry/len(df_sig)*100:.1f}%)')

    df_picks = df_sig[df_sig['entry_idx'] >= 0].copy()
    df_picks = df_picks.sort_values(['entry_date', 'cur_mf', 'code']).reset_index(drop=True)

    # 段末 (code_end) 用于 simulate
    df_picks['code_end'] = df_picks['code_end'].astype(int)

    # 单笔模拟先 (评估)
    print('\n  单笔模拟评估 (无仓位约束)...')
    rets = []; legs_list = []; reasons = []; sell_dates = []; days_list = []
    for _, p in df_picks.iterrows():
        ei = int(p['entry_idx']); ce = int(p['code_end'])
        sk, r, ret, legs = simulate_t0(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
        rets.append(ret); legs_list.append(legs); reasons.append(r)
        sell_dates.append(date_arr[sk])
        days_list.append(sk - ei)
    df_picks['ret_pct'] = rets
    df_picks['legs'] = legs_list
    df_picks['sell_reason'] = reasons
    df_picks['sell_date'] = sell_dates
    df_picks['days'] = days_list

    print(f'\n  --- 单笔统计 ---')
    print(f'    n={len(df_picks)}')
    print(f'    avg ret: {df_picks["ret_pct"].mean():+.2f}%')
    print(f'    median ret: {df_picks["ret_pct"].median():+.2f}%')
    print(f'    win: {(df_picks["ret_pct"]>0).mean()*100:.1f}%')
    print(f'    ≥+50%: {(df_picks["ret_pct"]>=50).sum()} ({(df_picks["ret_pct"]>=50).mean()*100:.2f}%)')
    print(f'    ≥+100%: {(df_picks["ret_pct"]>=100).sum()} ({(df_picks["ret_pct"]>=100).mean()*100:.2f}%)')
    print(f'    ≥+200%: {(df_picks["ret_pct"]>=200).sum()}')
    print(f'    avg legs: {df_picks["legs"].mean():.2f}')
    print(f'    avg days: {df_picks["days"].mean():.1f}')

    print(f'\n  按年:')
    df_picks['year'] = df_picks['entry_date'].str[:4]
    for y, g in df_picks.groupby('year'):
        n = len(g); avg = g['ret_pct'].mean()
        win = (g['ret_pct']>0).mean()*100
        h100 = (g['ret_pct']>=100).sum()
        print(f'    {y}: n={n:>3}, avg={avg:>+5.1f}%, win={win:>4.1f}%, ≥+100%={h100}')

    # 资金回测
    print(f'\n  资金回测 (K=2)...')
    all_dates = sorted(set(date_arr.tolist()))
    code_ends_map = {}  # not used

    for K in [1, 2, 3, 5]:
        r = run_capital_backtest(K, df_picks, code_date_idx, code_arr, date_arr,
                                   close_arr, trend_arr, mf_arr, retail_arr, all_dates, code_ends_map)
        print(f'\n  --- K={K} ---')
        print(f'    final: ¥{r["final"]:>10,.0f}  total: {r["total"]:>+7.2f}%  annual: {r["annual"]:>+5.2f}%')
        print(f'    MDD: {r["mdd"]:>+6.1f}%  trades: {r["n_trades"]:>3}  win: {r["win"]:>5.1f}%  avg: {r["avg"]:>+6.2f}%')
        print(f'    avg_pos: {r["avg_pos"]:>5.1f}%')

        df_t = r['df_t']
        if len(df_t) > 0:
            df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year
            print(f'    按年:')
            for y, g in df_t.groupby('year'):
                avg = g['cum_ret_pct'].mean()
                win = (g['cum_ret_pct']>0).mean()*100
                print(f'      {y}: n={len(g):>3}, avg={avg:>+6.1f}%, win={win:>4.1f}%, '
                      f'pnl=¥{g["cum_pnl"].sum()/1000:>+5.0f}K')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_picks.to_csv(os.path.join(out_dir, 'strategy_v1_picks.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 strategy_v1_picks.csv')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
