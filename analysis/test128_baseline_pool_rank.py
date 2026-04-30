# -*- coding: utf-8 -*-
"""池深池天作为排名因子 — 测试 baseline + 多种排名方案

baseline (无任何过滤):
  入池: retail < -250
  出池: retail >= 0
  买点: 大盘 y_gua=000 坤 + 个股 d_gua=011 巽
  卖点: bull_2nd / TS20 / 60d
  K=5 每日买 1 只 (40K/股)

排名方案 (同日多只时按排名取 1):
  R0: 按 code 升 (无排名, baseline)
  R1: pool_days 升 (新池优先)
  R2: pool_min_retail 升 (深池优先)
  R3: pool_days 升 + pool_min_retail 升 (新且深)
  R4: pool_min_retail 升 + pool_days 升 (深且新, 顺序换)
  R5: 综合 score (pool_days<6 +1, pool_min_retail<-500 +1, ...)

输出: K=5 资金回测的总收益 / 年化 / MDD / 笔数 / 胜率 / 单笔均
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
    if days_held >= MAX_HOLD:
        return True, 'timeout'
    if len(td_buy_to_now) < 2:
        return False, None
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
            if cross_count >= 2:
                return True, 'bull_2nd'
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
                trades.append({
                    'code': code, 'buy_date': pos['buy_date'], 'sell_date': today,
                    'buy_price': pos['buy_price'], 'sell_price': sell_price,
                    'profit': proceeds-cost, 'ret_pct': ret_pct,
                    'days': days_held, 'reason': reason,
                })
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
                            holdings[code] = {
                                'buy_date': today, 'buy_idx_global': ridx,
                                'buy_price': buy_price, 'shares': shares,
                            }

        mv = 0.0
        for code, pos in holdings.items():
            if code in code_date_idx and today in code_date_idx[code]:
                ti = code_date_idx[code][today]
                mv += pos['shares'] * close_arr[ti]
            else:
                mv += pos['shares'] * pos['buy_price']
        nav_history.append({'date': today, 'cash': cash, 'mv': mv,
                              'nav': cash+mv, 'pos': len(holdings)})

    last = all_dates[-1]
    for code, pos in list(holdings.items()):
        if code in code_date_idx and last in code_date_idx[code]:
            ti = code_date_idx[code][last]
            sp = close_arr[ti]
            cash += pos['shares']*sp
            trades.append({
                'code': code, 'buy_date': pos['buy_date'], 'sell_date': last,
                'buy_price': pos['buy_price'], 'sell_price': sp,
                'profit': pos['shares']*(sp-pos['buy_price']),
                'ret_pct': (sp/pos['buy_price']-1)*100,
                'days': ti - pos['buy_idx_global'],
                'reason': 'force_close',
            })

    df_t = pd.DataFrame(trades)
    df_n = pd.DataFrame(nav_history)
    final = df_n['nav'].iloc[-1]
    days = (pd.to_datetime(df_n['date'].iloc[-1]) - pd.to_datetime(df_n['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_n['peak'] = df_n['nav'].cummax()
    mdd = ((df_n['nav']-df_n['peak'])/df_n['peak']*100).min()
    win = (df_t['ret_pct']>0).mean()*100 if len(df_t) else 0
    avg = df_t['ret_pct'].mean() if len(df_t) else 0
    return {
        'final': final, 'total': (final/INIT_CAPITAL-1)*100,
        'annual': annual, 'mdd': mdd, 'n_trades': len(df_t),
        'win': win, 'avg': avg, 'df_t': df_t,
    }


def main():
    t0 = time.time()
    print('=== baseline + 池深池天排名 K=5 资金回测 ===\n')

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
    print(f'  {len(df):,} 行')

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

    print('扫信号 (无避雷无 score)...')
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
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail_arr[gi]

            if in_pool and retail_arr[gi] < pool_min_retail:
                pool_min_retail = retail_arr[gi]

            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False
                continue

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'pool_days': i - pool_enter_i,
                    'pool_min_retail': pool_min_retail,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')
    print(f'  按天分布: 总 {df_e["date"].nunique()} 天有信号, '
          f'avg {len(df_e)/df_e["date"].nunique():.1f} 只/天, '
          f'max {df_e.groupby("date").size().max()} 只')

    all_dates = sorted(df['date'].unique())

    # 6 种排名方案
    schemes = []

    # R0: 按 code 升 (baseline)
    df_r0 = df_e.sort_values(['date', 'code']).drop_duplicates('date', keep='first')
    schemes.append(('R0 baseline (code↑)', df_r0))

    # R1: pool_days 升
    df_r1 = df_e.sort_values(['date', 'pool_days', 'code'],
                                  ascending=[True, True, True]).drop_duplicates('date', keep='first')
    schemes.append(('R1 pool_days↑', df_r1))

    # R2: pool_min_retail 升 (越小越深)
    df_r2 = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                  ascending=[True, True, True]).drop_duplicates('date', keep='first')
    schemes.append(('R2 pool_min_retail↑ (深池)', df_r2))

    # R3: pool_days↑ + pool_min_retail↑
    df_r3 = df_e.sort_values(['date', 'pool_days', 'pool_min_retail', 'code'],
                                  ascending=[True, True, True, True]).drop_duplicates('date', keep='first')
    schemes.append(('R3 pool_days↑ + retail↑', df_r3))

    # R4: pool_min_retail↑ + pool_days↑
    df_r4 = df_e.sort_values(['date', 'pool_min_retail', 'pool_days', 'code'],
                                  ascending=[True, True, True, True]).drop_duplicates('date', keep='first')
    schemes.append(('R4 retail↑ + pool_days↑', df_r4))

    # R5: 复合 score (新池 +1, 深池 +1, 同 score 内 ret↑)
    df_e['rscore'] = 0
    df_e.loc[df_e['pool_days'] < 6, 'rscore'] += 1
    df_e.loc[df_e['pool_min_retail'] < -500, 'rscore'] += 1
    df_r5 = df_e.sort_values(['date', 'rscore', 'pool_min_retail', 'code'],
                                  ascending=[True, False, True, True]).drop_duplicates('date', keep='first')
    schemes.append(('R5 score (pd<6 + retail<-500)', df_r5))

    K = 5
    print(f'\n=== K=5 资金回测对比 ===\n')
    print(f'  {"方案":<32} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    for label, df_picks in schemes:
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        print(f'  {label:<32} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 也跑一下 K=3 / K=10 看仓位变化
    print(f'\n=== R3 (新且深) 不同 K 的对比 ===\n')
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, df_r3, code_date_idx, close_arr, trend_arr, all_dates)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
