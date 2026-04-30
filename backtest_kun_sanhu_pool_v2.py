# -*- coding: utf-8 -*-
"""坤 + 散户线入池 v2: 修正出池机制

vs v1 (backtest_kun_sanhu_pool.py):
  入池: retail < -250 (不变)
  新增出池: retail >= 0 (散户回流 → 池信号失效)
  原版只在触发买入/避雷/score<2 时退池, 导致很多事件挂池数百日, 是伪信号

test126 实证:
  V1 (无主动出池): n=8633, [120,365) 池天有 2312 笔 +20.86% (全是 bug 制造)
  V4 (retail>=0):  n=5173, max 池天=63, [0,6) +11% 是真金区

排序: F+B 双轨 (priority score=3, 然后 score=2 中 sanhu_5d 最低)
卖点: bull_2nd + TS20 + 60d 兜底
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
MAX_HOLD_DAYS = 60
REGIME_Y = '000'
TRIGGER_GUA = '011'
POOL_THR = -250
POOL_EXIT_RETAIL = 0  # 新增: retail>=0 出池


def should_sell(td_buy_to_now, days_held):
    if days_held >= MAX_HOLD_DAYS:
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


def run_backtest(K, df_picks, code_index, code_date_idx, code_arr, date_arr,
                 close_arr, trend_arr, all_dates):
    SLOT_VALUE = INIT_CAPITAL / K
    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []
    picks_by_date = df_picks.set_index('date').to_dict('index')

    for di, today in enumerate(all_dates):
        # 1. 卖
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
                    'shares': pos['shares'], 'profit': proceeds-cost,
                    'ret_pct': ret_pct, 'days': days_held,
                    'score': pos['score'], 'reason': reason,
                })
                del holdings[code]

        # 2. 买
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
                                'score': cand['score'],
                            }

        # 3. NAV
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
                'shares': pos['shares'], 'profit': pos['shares']*(sp-pos['buy_price']),
                'ret_pct': (sp/pos['buy_price']-1)*100,
                'days': ti - pos['buy_idx_global'],
                'score': pos['score'], 'reason': 'force_close',
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
        'K': K, 'final': final, 'total': (final/INIT_CAPITAL-1)*100,
        'annual': annual, 'mdd': mdd, 'n_trades': len(df_t),
        'win': win, 'avg': avg, 'df_t': df_t, 'df_n': df_n,
    }


def main():
    t0 = time.time()
    print('=== 坤 + sanhu<-250 入池 v2 (retail>=0 出池) 资金回测 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_index = {code_arr[code_starts[i]]: (code_starts[i], code_ends[i])
                    for i in range(len(code_starts))}
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        c = code_arr[s]
        code_date_idx[c] = {date_arr[s+j]: s+j for j in range(e-s)}

    LOOKBACK = 30
    print('扫所有合格信号 (带主动出池)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD_DAYS + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1

        for i in range(LOOKBACK, n - MAX_HOLD_DAYS - 1):
            gi = s + i

            # 入池
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i

            # 主动出池: retail 回流
            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False
                continue

            # 触发买入
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue

                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                if score < 2:
                    in_pool = False
                    continue

                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'score': score, 'sanhu_5d': sh5_arr[gi],
                    'pool_days': i - pool_enter_i,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  合格信号: {len(df_e):,}')
    print(f'  池天 avg={df_e["pool_days"].mean():.1f} 中位={df_e["pool_days"].median():.0f} max={df_e["pool_days"].max()}')

    # F+B 双轨
    df_e_sorted = df_e.sort_values(['date', 'score', 'sanhu_5d', 'code'],
                                       ascending=[True, False, True, True])
    df_picks = df_e_sorted.drop_duplicates('date', keep='first').copy()
    print(f'  每日 1 只 (F+B 双轨): {len(df_picks)} 天')
    print(f'  picks score 分布:')
    print(df_picks['score'].value_counts().sort_index())

    all_dates = sorted(df['date'].unique())
    print(f'\n=== 资金回测 (200K, F+B 每日 1 只, 不同 K) ===\n')
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = []
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, df_picks, code_index, code_date_idx, code_arr, date_arr,
                          close_arr, trend_arr, all_dates)
        results.append(r)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    best = max(results, key=lambda x: x['total'])
    print(f'\n=== K={best["K"]} 详细 (最优) ===')
    df_t = best['df_t']

    if len(df_t):
        print(f'\n  按 score:')
        for sc in sorted(df_t['score'].unique()):
            sub = df_t[df_t['score']==sc]
            print(f'    score={sc}: n={len(sub)} avg {sub["ret_pct"].mean():>+6.2f}% '
                  f'win {(sub["ret_pct"]>0).mean()*100:>5.1f}% pnl ¥{sub["profit"].sum():>+10,.0f}')

        print(f'\n  按年:')
        df_t['year'] = pd.to_datetime(df_t['buy_date']).dt.year
        for y in sorted(df_t['year'].unique()):
            sub = df_t[df_t['year']==y]
            print(f'    {y}: n={len(sub):>3} avg {sub["ret_pct"].mean():>+6.2f}% '
                  f'win {(sub["ret_pct"]>0).mean()*100:>5.1f}% pnl ¥{sub["profit"].sum():>+10,.0f}')

        print(f'\n  按 reason:')
        for r in df_t['reason'].unique():
            sub = df_t[df_t['reason']==r]
            print(f'    {r:<14} n={len(sub):>3} avg {sub["ret_pct"].mean():>+6.2f}% hold {sub["days"].mean():>5.1f}')

    # vs v1 对比
    print('\n=== vs v1 (无主动出池) 对比 ===')
    print(f'  v1 K=3 (历史): +195.8% / +10.72% 年化 / -28.7% MDD / 66 笔 / 56% win')
    r3 = next((x for x in results if x['K'] == 3), None)
    if r3:
        print(f'  v2 K=3 (当前): {r3["total"]:+.1f}% / {r3["annual"]:+.2f}% 年化 / {r3["mdd"]:+.1f}% MDD '
              f'/ {r3["n_trades"]} 笔 / {r3["win"]:.1f}% win')

    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    best['df_t'].to_csv(os.path.join(out_dir, 'kun_sanhu_pool_v2_trades.csv'), index=False, encoding='utf-8-sig')
    best['df_n'].to_csv(os.path.join(out_dir, 'kun_sanhu_pool_v2_nav.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 kun_sanhu_pool_v2_{{trades,nav}}.csv')
    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
