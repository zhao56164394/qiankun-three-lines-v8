# -*- coding: utf-8 -*-
"""修正入池逻辑 (A 方向): 真正"首次入池"

之前 bug: 在 retail<-250 持续期内, mf 上下波动会反复触发"出池-入池"
  实际记录的"入池日"是: 池被关掉后又重新进的日子, 不是真正第一次破 -250

修正: retail<-250 是连续状态, 用 enter 状态机
  enter 触发: retail 从 >=-250 跌到 <-250 (上沿穿透)
  exit 触发: 各种条件
  retail 持续 <-250 不重新触发 enter

测试 4 种 (用真正首次入池):
  A0: retail>=0 出池
  A1: mf<=50 出池
  A2: retail>=0 OR mf<=50 出池
  A3: retail>=0 AND mf<=50 出池

对比之前的 M0/M2 (有 bug 版本)
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
POOL_THR = -250
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


def scan(arrays, exit_mode):
    """exit_mode:
       A0 retail>=0
       A1 mf<=50
       A2 retail>=0 OR mf<=50
       A3 retail>=0 AND mf<=50
       BugM0 (旧 M0)
       BugM2 (旧 M2)
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
        prev_retail_below = False  # retail 上一日是否 <-250

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < POOL_THR

            # ============ 入池: 真正首次穿透 ============
            if exit_mode in ['A0', 'A1', 'A2', 'A3']:
                # 状态机: 只在"上沿穿透"才入池, 持续 <-250 不重新入池
                if not in_pool and cur_below and not prev_retail_below:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                # 池中持续刷新最深值
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]
            else:
                # 旧 buggy 版本: not in_pool && cur_below 即入池
                if not in_pool and cur_below:
                    in_pool = True
                    pool_enter_i = i
                    pool_min_retail = retail[gi]
                if in_pool and retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]

            # ============ 出池 ============
            if in_pool:
                exit_now = False
                if exit_mode in ['A0', 'BugM0']:
                    if retail[gi] >= 0: exit_now = True
                elif exit_mode in ['A1', 'BugM2']:
                    if mf[gi] <= 50: exit_now = True
                elif exit_mode == 'A2':
                    if retail[gi] >= 0 or mf[gi] <= 50: exit_now = True
                elif exit_mode == 'A3':
                    if retail[gi] >= 0 and mf[gi] <= 50: exit_now = True
                if exit_now:
                    in_pool = False
                    prev_retail_below = cur_below
                    continue

            # ============ 巽日触发 ============
            if in_pool and mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'pool_enter_date': date[s+pool_enter_i],
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
    print('=== 修正入池: 真正"首次入池" + 4 种出池机制 ===\n')

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

    print('\n=== 修正后的 K=5 资金回测 ===\n')
    schemes = {
        'BugM0 (旧 retail>=0)':         'BugM0',
        'BugM2 (旧 mf<=50)':           'BugM2',
        'A0 真首次+retail>=0':           'A0',
        'A1 真首次+mf<=50':              'A1',
        'A2 真首次+任一(retail>=0 OR mf<=50)':  'A2',
        'A3 真首次+全部(retail>=0 AND mf<=50)': 'A3',
    }

    K = 5
    print(f'  {"方案":<40} {"事件":>6} {"信号天":>6} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = {}
    for label, mode in schemes.items():
        df_e = scan(arrays, mode)
        df_picks = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')
        n_sig_days = len(df_picks)
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results[label] = (r, df_e, df_picks)
        print(f'  {label:<40} {len(df_e):>6,} {n_sig_days:>6} ¥{r["final"]/1000:>8.0f}K '
              f'{r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% {r["mdd"]:>+7.1f}% '
              f'{r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 顺丰例子检查 (验证修正)
    print('\n=== 顺丰 002352 入池日检查 ===\n')
    for label in ['BugM0 (旧 retail>=0)', 'A0 真首次+retail>=0', 'BugM2 (旧 mf<=50)', 'A1 真首次+mf<=50']:
        df_e = results[label][1]
        sf = df_e[df_e['code'] == '002352']
        sf = sf[sf['date'].between('2016-01-01', '2016-04-01')]
        print(f'  {label}:')
        if len(sf):
            print(sf[['pool_enter_date', 'date', 'pool_days', 'pool_min_retail', 'cur_retail', 'cur_mf']].to_string(index=False))
        else:
            print('    无事件')
        print()

    # 跨段对比 BugM2 vs A1 (相同思路, 看修正前后差距)
    if 'BugM2 (旧 mf<=50)' in results and 'A1 真首次+mf<=50' in results:
        print('\n=== 跨段对比 BugM2 vs A1 (修正后) ===\n')
        bug_t = results['BugM2 (旧 mf<=50)'][0]['df_t']
        a1_t = results['A1 真首次+mf<=50'][0]['df_t']
        bug_t['year'] = pd.to_datetime(bug_t['buy_date']).dt.year
        a1_t['year'] = pd.to_datetime(a1_t['buy_date']).dt.year
        for y in sorted(set(bug_t['year'].unique()) | set(a1_t['year'].unique())):
            b = bug_t[bug_t['year'] == y]
            a = a1_t[a1_t['year'] == y]
            if len(b) and len(a):
                print(f'  {y}: BugM2 {len(b):>3} 笔 {b["ret_pct"].mean():>+5.2f}% / '
                      f'A1 {len(a):>3} 笔 {a["ret_pct"].mean():>+5.2f}%')

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
