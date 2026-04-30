# -*- coding: utf-8 -*-
"""验证 BugM2 的真实身份 — 入池触发条件 vs 池机制

BugM2 的真实逻辑 (从 test138 重新解释):
  retail<-250 状态下, mf>50 那天才入池, mf<=50 出池
  → 等价于: 当日 retail<-250 AND mf>50 = 入池信号

新假设: 这是一个 "入池触发条件" (单日双指标), 不是池机制
  T0: 当日 retail<-250 AND mf>50 (单日触发, 当日就买)

分支:
  T1: 当日 retail<-250 AND mf>50 (= 触发条件, 同步触发)
  T2: 触发后 N 日内允许买 (买窗口扩展)
  T3: BugM2 原状 (触发后保留池, mf>50 持续期都可买)

对比 A3 (真池机制):
  A3: retail<-250 入池, retail>=0 AND mf<=50 出池

哪个是真信号?
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
LOOKBACK = 30


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


def scan(arrays, mode):
    """模式:
       T1: 当日 retail<-250 AND mf>50 即入池, 当日有效 (单日)
       T1b: 当日 retail<-250 AND mf>50, 触发后 5 日有效
       T1c: 当日 retail<-250 AND mf>50, 触发后 10 日有效
       T1d: 当日 retail<-250 AND mf>50, 触发后 20 日有效
       T2: BugM2 原版 (触发后 mf 一直>50 都有效)
       A3: retail<-250 上沿穿透入池, retail>=0 AND mf<=50 出池
    """
    code_starts = arrays['starts']; code_ends = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d = arrays['stk_d']; mkt_y = arrays['mkt_y']
    date = arrays['date']; code = arrays['code']

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf
        prev_retail_below = False
        last_trigger_i = -999  # T1b/T1c/T1d 用

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            # ============ T1 系列: 双指标入池 ============
            if mode in ['T1', 'T1b', 'T1c', 'T1d']:
                if cur_below and mf[gi] > 50:
                    last_trigger_i = i
                    if not in_pool:
                        in_pool = True
                        pool_enter_i = i
                        pool_min_retail = retail[gi]
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]
                # T1: 单日; T1b: 5 日; T1c: 10 日; T1d: 20 日
                window = {'T1': 0, 'T1b': 5, 'T1c': 10, 'T1d': 20}[mode]
                if i - last_trigger_i > window:
                    in_pool = False

            # ============ T2: BugM2 原版 ============
            elif mode == 'T2':
                if cur_below and mf[gi] > 50 and not in_pool:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]
                # mf<=50 出池 (BugM2 原状)
                if in_pool and mf[gi] <= 50:
                    in_pool = False

            # ============ A3: 真池机制 ============
            elif mode == 'A3':
                if not in_pool and cur_below and not prev_retail_below:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]
                if in_pool and retail[gi] >= 0 and mf[gi] <= 50:
                    in_pool = False

            # 触发巽日
            if in_pool and mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'pool_days': i - pool_enter_i,
                    'pool_min_retail': pool_min_retail,
                    'cur_retail': retail[gi],
                    'cur_mf': mf[gi],
                })

            prev_retail_below = cur_below

    return pd.DataFrame(events)


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
        nav_history.append({'date':today, 'mv':mv, 'nav':cash+mv})

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
    df_n['pos_pct'] = df_n['mv'] / df_n['nav']
    avg_pos = df_n['pos_pct'].mean() * 100
    return {'final':final, 'total':(final/INIT_CAPITAL-1)*100, 'annual':annual,
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'avg_pos':avg_pos,
            'df_t':df_t}


def main():
    t0 = time.time()
    print('=== 验证 BugM2 真实身份: 入池触发条件 vs 池机制 ===\n')

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
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {df['date'].iat[s+j]: s+j for j in range(e-s)}

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'mf': df['main_force'].to_numpy().astype(np.float64),
        'stk_d': df['stk_d'].to_numpy(),
        'mkt_y': df['mkt_y'].to_numpy(),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    all_dates = sorted(df['date'].unique())

    schemes = {
        'T1 当日 retail<-250 AND mf>50 (单日)':       'T1',
        'T1b 同上 + 5 日窗口':                         'T1b',
        'T1c 同上 + 10 日窗口':                        'T1c',
        'T1d 同上 + 20 日窗口':                        'T1d',
        'T2 BugM2 原版 (mf>50 持续期)':                'T2',
        'A3 真池机制 retail<-250→retail>=0 AND mf<=50': 'A3',
    }

    K = 5
    print(f'\n=== K={K} 资金回测 ===\n')
    print(f'  {"方案":<48} {"事件":>6} {"信号天":>6} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = {}
    for label, mode in schemes.items():
        df_e = scan(arrays, mode)
        df_picks = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')
        n_sig_days = len(df_picks)
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results[label] = (r, df_e, df_picks)
        print(f'  {label:<48} {len(df_e):>6,} {n_sig_days:>6} ¥{r["final"]/1000:>8.0f}K '
              f'{r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% {r["mdd"]:>+7.1f}% '
              f'{r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 顺丰 002352 各模式触发日
    print('\n=== 顺丰 002352 在各模式下的触发日 ===\n')
    for label in schemes:
        df_e = results[label][1]
        sf = df_e[df_e['code'] == '002352']
        sf = sf[sf['date'].between('2016-01-01', '2016-04-01')]
        print(f'  {label}: {len(sf)} 个触发')
        if len(sf):
            for _, r in sf.iterrows():
                print(f'    {r["date"]} pool_days={r["pool_days"]:>3} '
                      f'cur_retail={r["cur_retail"]:>+6.0f} cur_mf={r["cur_mf"]:>+6.0f}')

    # 跨段对比 A3 vs T2 (原 BugM2)
    print('\n=== 跨段对比 A3 vs T2 (BugM2 原版) ===\n')
    a3_t = results['A3 真池机制 retail<-250→retail>=0 AND mf<=50'][0]['df_t']
    t2_t = results['T2 BugM2 原版 (mf>50 持续期)'][0]['df_t']
    a3_t['year'] = pd.to_datetime(a3_t['buy_date']).dt.year
    t2_t['year'] = pd.to_datetime(t2_t['buy_date']).dt.year
    print(f'  {"年":<6} | {"A3":<22} | {"T2":<22} | 差')
    for y in sorted(set(a3_t['year'].unique()) | set(t2_t['year'].unique())):
        a = a3_t[a3_t['year'] == y]
        t = t2_t[t2_t['year'] == y]
        if len(a) and len(t):
            print(f'  {y:<6} | {len(a):>3} 笔 {a["ret_pct"].mean():>+6.2f}% '
                  f'{(a["ret_pct"]>0).mean()*100:>5.1f}% '
                  f'| {len(t):>3} 笔 {t["ret_pct"].mean():>+6.2f}% '
                  f'{(t["ret_pct"]>0).mean()*100:>5.1f}% '
                  f'| {t["ret_pct"].mean()-a["ret_pct"].mean():>+5.2f}%')

    # 不同 K 看最优
    best_label = max(results, key=lambda k: results[k][0]['total'])
    print(f'\n=== 最优 ({best_label}) 不同 K ===\n')
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6}')
    best_picks = results[best_label][2]
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, best_picks, code_date_idx, close_arr, trend_arr, all_dates)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
