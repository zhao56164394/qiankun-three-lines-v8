# -*- coding: utf-8 -*-
"""池深池天作为硬过滤 — F1-F4 测试

baseline (test128 R2 = +126.0%):
  入池: retail<-250, 出池: retail>=0
  买点: 大盘 y=000 + 个股 d=011
  排名: pool_min_retail↑ (深池优先)
  K=5

硬过滤方案:
  F0: 不过滤 (= R2 baseline)
  F1: pool_min_retail < -400 (砍约 50% 浅池)
  F2: pool_min_retail < -500 (砍约 85%)
  F3: pool_days < 6 (砍约 47% 旧池)
  F4: F1 + F3 (新且深)
  F5: pool_min_retail < -350 (温和)
  F6: pool_days < 9 (温和)

排名都用 pool_min_retail↑ 不变
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
POOL_EXIT_RETAIL = 0
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
        nav_history.append({'date':today, 'cash':cash, 'mv':mv, 'nav':cash+mv})

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

    # 计算每年的 pnl 看资金是否大量闲置
    df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year if len(df_t) else None

    return {'final':final, 'total':(final/INIT_CAPITAL-1)*100, 'annual':annual,
            'mdd':mdd, 'n_trades':len(df_t), 'win':win, 'avg':avg, 'df_t':df_t,
            'df_n':df_n}


def main():
    t0 = time.time()
    print('=== 池深池天硬过滤 F1-F6 测试 (K=5) ===\n')

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
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True; pool_enter_i = i; pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] < pool_min_retail:
                pool_min_retail = retail_arr[gi]
            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False; continue
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                events.append({'date':date_arr[gi], 'code':code_arr[gi],
                                'pool_days':i-pool_enter_i,
                                'pool_min_retail':pool_min_retail})
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  全集: {len(df_e):,} 事件, {df_e["date"].nunique()} 个有信号天')
    all_dates = sorted(df['date'].unique())
    total_days = len(all_dates)

    # 各方案的过滤
    schemes = []
    schemes.append(('F0 不过滤 (R2 baseline)',     df_e))
    schemes.append(('F1 retail<-400',               df_e[df_e['pool_min_retail'] < -400]))
    schemes.append(('F2 retail<-500',               df_e[df_e['pool_min_retail'] < -500]))
    schemes.append(('F3 pool_days<6',               df_e[df_e['pool_days'] < 6]))
    schemes.append(('F4 retail<-400 + days<6',     df_e[(df_e['pool_min_retail'] < -400) & (df_e['pool_days'] < 6)]))
    schemes.append(('F5 retail<-350',               df_e[df_e['pool_min_retail'] < -350]))
    schemes.append(('F6 pool_days<9',               df_e[df_e['pool_days'] < 9]))
    schemes.append(('F7 retail<-300',               df_e[df_e['pool_min_retail'] < -300]))
    schemes.append(('F8 retail<-300 + days<6',     df_e[(df_e['pool_min_retail'] < -300) & (df_e['pool_days'] < 6)]))

    K = 5
    print(f'\n=== K=5 资金回测对比 (排名都按 pool_min_retail↑) ===\n')
    print(f'  {"方案":<32} {"事件":>6} {"信号天":>5} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = []
    for label, df_x in schemes:
        if len(df_x) == 0:
            print(f'  {label:<32} 0 events')
            continue
        df_picks = df_x.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')
        n_signal_days = len(df_picks)
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results.append((label, len(df_x), n_signal_days, r))
        print(f'  {label:<32} {len(df_x):>6,} {n_signal_days:>5} ¥{r["final"]/1000:>8.0f}K '
              f'{r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% {r["mdd"]:>+7.1f}% '
              f'{r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    print(f'\n  总交易日: {total_days}')
    print(f'  说明: 信号天数 = 至少有一只股可买的天数, 越少 = 仓位越容易闲置')

    # 看最佳方案的资金占用情况
    print(f'\n=== 各方案的资金占用 (最高仓位日 / 满仓天数比例) ===')
    for label, ne, nsd, r in results:
        df_n = r['df_n']
        df_n['pos_pct'] = df_n['mv'] / df_n['nav']
        avg_pos = df_n['pos_pct'].mean() * 100
        full_pos_days = (df_n['pos_pct'] > 0.95).sum()
        print(f'  {label:<32} 平均仓位 {avg_pos:>5.1f}%, 满仓天数 {full_pos_days}/{len(df_n)}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
