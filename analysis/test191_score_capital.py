# -*- coding: utf-8 -*-
"""新入池 (score 排名 + 7 组独立规律) 资金回测, 对比 test188 ABC 池

入池规则 (基础 F 机制 + score 过滤):
  Gate 1: 全市场 trend 下穿 11 (无 ABC 过滤)
  Gate 2: F 机制 entry (散户线上穿 0, 60d 内, td>11)
  Gate 3: 7 组独立规律 score 命中数

7 组规律 (test190 IS+OOS 双通过, 按业务维度去重):
  1. stk_yy == 011        (个股 011 巽, OOS 1.61x)
  2. cur_mf >= 100        (主力线热, OOS 1.16x)
  3. mf30_max >= 200      (30日有爆发, OOS 1.32x)
  4. mf30_min <= -200     (30日深坑, OOS 1.43x)
  5. rt30_min <= -200     (30日散户深坑, OOS 1.54x)
  6. mf5 >= 100           (入场前主力高, OOS 1.30x)
  7. rt5 <= -100          (入场前散户低, OOS 1.10x)

3 种入池版本:
  v1: score >= 2  (软门槛)
  v2: score >= 3  (中门槛, test190 跨段最稳)
  v3: 硬 AND      (cur_mf>=100 AND mf30_max>=200 AND rt30_min<=-200)

机制 (跟 test188 一致):
  卖 D6': retail[k-1]<=0 AND retail[k]<=0
  买回 U1: mf 上升
  T0: trend < 11
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


def find_signals_full(starts, ends, td, date_arr, code_arr,
                      mf, retail, stk_d_t, stk_m_t, stk_y_t):
    """全市场波段起点 + 提取 signal_idx 时的卦象/数值"""
    sigs = []
    for ci in range(len(starts)):
        s = starts[ci]; e = ends[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            if np.isnan(td[i]) or np.isnan(td[i-1]): continue
            if td[i-1] > 11 and td[i] <= 11:
                sigs.append({
                    'signal_idx': i, 'signal_date': date_arr[i],
                    'code': code_arr[i],
                    'cur_mf': mf[i], 'cur_retail': retail[i],
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


def compute_features_at_entry(ei, code_start, td, mf, retail,
                              stk_d_t, stk_m_t, stk_y_t):
    """entry 时刻特征"""
    stk_yy = yyy(stk_d_t[ei], stk_m_t[ei], stk_y_t[ei])
    s5 = max(code_start, ei - 4)
    s30 = max(code_start, ei - 29)
    cur_mf = mf[ei]
    mf5 = np.nanmean(mf[s5:ei+1])
    rt5 = np.nanmean(retail[s5:ei+1])
    mf30_min = np.nanmin(mf[s30:ei+1])
    mf30_max = np.nanmax(mf[s30:ei+1])
    rt30_min = np.nanmin(retail[s30:ei+1])
    return {
        'stk_yy': stk_yy,
        'cur_mf': cur_mf, 'mf5': mf5, 'rt5': rt5,
        'mf30_min': mf30_min, 'mf30_max': mf30_max,
        'rt30_min': rt30_min,
    }


def calc_score(feats):
    """7 组独立规律命中数"""
    s = 0
    if feats['stk_yy'] == '011': s += 1
    if not np.isnan(feats['cur_mf']) and feats['cur_mf'] >= 100: s += 1
    if not np.isnan(feats['mf30_max']) and feats['mf30_max'] >= 200: s += 1
    if not np.isnan(feats['mf30_min']) and feats['mf30_min'] <= -200: s += 1
    if not np.isnan(feats['rt30_min']) and feats['rt30_min'] <= -200: s += 1
    if not np.isnan(feats['mf5']) and feats['mf5'] >= 100: s += 1
    if not np.isnan(feats['rt5']) and feats['rt5'] <= -100: s += 1
    return s


def hard_filter(feats):
    """v3 硬 AND: cur_mf>=100 AND mf30_max>=200 AND rt30_min<=-200"""
    if np.isnan(feats['cur_mf']) or feats['cur_mf'] < 100: return False
    if np.isnan(feats['mf30_max']) or feats['mf30_max'] < 200: return False
    if np.isnan(feats['rt30_min']) or feats['rt30_min'] > -200: return False
    return True


def simulate_F(buy_idx, code_end, td, close, mf, retail):
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return k, 'T0', (cum-1)*100, legs
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
    return code_end-1, 'fc', (cum-1)*100, legs


def _make_trade(code, pos, sell_date, days, reason):
    return {
        'code': code, 'tag': pos.get('tag', '?'),
        'buy_date': pos['initial_buy_date'], 'sell_date': sell_date,
        'cum_pnl': pos['cum_pnl'],
        'cum_ret_pct': pos['cum_pnl'] / pos['initial_cost'] * 100,
        'days': days, 'legs': pos['legs'], 'reason': reason,
    }


def run_capital_backtest(K, picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates):
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
        # 1. 持仓处理 F
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]: continue
            today_idx = code_date_idx[code][today]
            cur_close = close_arr[today_idx]

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
                if retail_arr[today_idx-1] <= 0 and retail_arr[today_idx] <= 0:
                    proceeds = pos['shares'] * cur_close
                    profit = proceeds - pos['shares'] * pos['cur_buy_price']
                    cash += proceeds
                    pos['cum_pnl'] += profit
                    pos['cash_at_pending'] = proceeds
                    pos['state'] = 'pending'
                    pos['legs'] += 1
            else:
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
                    'state': 'holding', 'tag': cand.get('tag', '?'),
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
    if len(df_n) == 0:
        return {'final':INIT_CAPITAL,'total':0,'annual':0,'mdd':0,
                'n_trades':0,'win':0,'avg':0,'avg_pos':0,
                'df_t':df_t,'df_n':df_n}
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
    print('=== test191: 新入池 (score) 资金回测 vs ABC 池 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'm_trend', 'y_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner')
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
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    # 1. 全市场扫信号
    print('  扫信号 + entry...')
    df_sig = find_signals_full(code_starts, code_ends, trend_arr, date_arr, code_arr,
                                mf_arr, retail_arr, stk_d_t, stk_m_t, stk_y_t)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'    全市场起点: {len(df_sig):,}')

    # 2. F entry
    rows = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end']); cs = int(s['code_start'])
        ei = find_entry(si, ce, trend_arr, retail_arr)
        if ei < 0: continue
        feats = compute_features_at_entry(ei, cs, trend_arr, mf_arr, retail_arr,
                                          stk_d_t, stk_m_t, stk_y_t)
        feats['signal_idx'] = si
        feats['entry_idx'] = ei
        feats['entry_date'] = date_arr[ei]
        feats['code'] = s['code']
        feats['code_end'] = ce
        feats['cur_retail'] = retail_arr[ei]
        feats['score'] = calc_score(feats)
        feats['hard'] = hard_filter(feats)
        rows.append(feats)
    df_e = pd.DataFrame(rows)
    print(f'    F entry: {len(df_e):,}')

    # 单笔模拟
    print('  单笔模拟评估...')
    rets = []; legs_l = []; reasons = []
    for _, p_ in df_e.iterrows():
        ei = int(p_['entry_idx']); ce = int(p_['code_end'])
        sk, r, ret, legs = simulate_F(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
        rets.append(ret); legs_l.append(legs); reasons.append(r)
    df_e['ret_pct'] = rets
    df_e['legs'] = legs_l
    df_e['reason'] = reasons
    df_e['year'] = df_e['entry_date'].str[:4]

    # 三个版本入池
    versions = {
        'v1_score>=2': df_e[df_e['score'] >= 2].copy(),
        'v2_score>=3': df_e[df_e['score'] >= 3].copy(),
        'v3_hard_AND': df_e[df_e['hard']].copy(),
    }

    print(f'\n  --- 单笔总览对比 ---')
    print(f'  {"版本":<14} {"n":>7} {"avg":>7} {"win%":>6} {"≥+50":>5} {"≥+100":>6} {"≥+200":>6} {"暴涨率":>7}')
    print(f'  {"all (无过滤)":<14} {len(df_e):>7,} {df_e["ret_pct"].mean():>+5.2f}% '
          f'{(df_e["ret_pct"]>0).mean()*100:>5.1f}% '
          f'{(df_e["ret_pct"]>=50).sum():>5} {(df_e["ret_pct"]>=100).sum():>6} '
          f'{(df_e["ret_pct"]>=200).sum():>6} {(df_e["ret_pct"]>=100).mean()*100:>+6.2f}%')
    for name, dfv in versions.items():
        n = len(dfv); avg = dfv['ret_pct'].mean()
        win = (dfv['ret_pct']>0).mean()*100
        h50 = (dfv['ret_pct']>=50).sum()
        h100 = (dfv['ret_pct']>=100).sum()
        h200 = (dfv['ret_pct']>=200).sum()
        rate = h100/n*100 if n else 0
        print(f'  {name:<14} {n:>7,} {avg:>+5.2f}% {win:>5.1f}% '
              f'{h50:>5} {h100:>6} {h200:>6} {rate:>+6.2f}%')

    # 资金回测
    all_dates = sorted(set(date_arr.tolist()))
    for name, dfv in versions.items():
        print(f'\n  ============ 版本 {name} 资金回测 ============')
        # 同日多候选按 score 倒序 + cur_retail 升序选 (越高分越优先, 同分越深越优先)
        if 'score' in dfv.columns:
            dfv['neg_score'] = -dfv['score']
            df_picks = dfv.sort_values(['entry_date', 'neg_score', 'cur_retail', 'code']).reset_index(drop=True)
        else:
            df_picks = dfv.sort_values(['entry_date', 'cur_retail', 'code']).reset_index(drop=True)
        df_picks['code_end'] = df_picks['code_end'].astype(int)
        df_picks['tag'] = name

        print(f'  {"K":<3} {"final":>10} {"总":>9} {"年化":>7} {"MDD":>7} {"段":>5} {"胜":>6} {"avg":>7} {"pos":>6}')
        for K in [1, 2, 3, 5]:
            r = run_capital_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates)
            print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% '
                  f'{r["annual"]:>+5.2f}% {r["mdd"]:>+5.1f}% {r["n_trades"]:>5} '
                  f'{r["win"]:>5.1f}% {r["avg"]:>+5.2f}% {r["avg_pos"]:>5.1f}%')

    # 重点: 各版本 K=3 按年
    print(f'\n\n  ============ 各版本 K=3 按年明细 ============')
    for name, dfv in versions.items():
        if 'score' in dfv.columns:
            dfv['neg_score'] = -dfv['score']
            df_picks = dfv.sort_values(['entry_date', 'neg_score', 'cur_retail', 'code']).reset_index(drop=True)
        else:
            df_picks = dfv.sort_values(['entry_date', 'cur_retail', 'code']).reset_index(drop=True)
        df_picks['code_end'] = df_picks['code_end'].astype(int)
        df_picks['tag'] = name
        r = run_capital_backtest(3, df_picks, code_date_idx, close_arr, trend_arr, mf_arr, retail_arr, all_dates)
        df_t = r['df_t']
        if len(df_t) == 0: continue
        df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year
        print(f'\n  --- {name} K=3 ---')
        print(f'    {"年":<6} {"段":>4} {"avg":>7} {"win":>6} {"≥+100":>5} {"pnl":>10}')
        for y, g_ in df_t.groupby('year'):
            avg = g_['cum_ret_pct'].mean()
            win = (g_['cum_ret_pct']>0).mean()*100
            h100 = (g_['cum_ret_pct']>=100).sum()
            print(f'    {y:<6} {len(g_):>4} {avg:>+5.2f}% {win:>5.1f}% {h100:>5} ¥{g_["cum_pnl"].sum()/1000:>+5.0f}K')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
