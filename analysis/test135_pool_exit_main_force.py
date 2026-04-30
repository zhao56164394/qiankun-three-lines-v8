# -*- coding: utf-8 -*-
"""验证 retail 和 main_force 关系 + 用主力线做出池机制

验证:
  1. retail 和 main_force 的相关性 (是否镜像)
  2. 它们的分布范围

测试出池机制 (用主力线):
  M0 retail>=0 (当前 baseline)
  M1 main_force <= 0 (主力转弱)
  M2 main_force <= 50 (主力明显弱)
  M3 main_force <= -100 (主力彻底走)
  M4 main_force_5d 转负
  M5 main_force_5d <= -50
  M6 main_force 60d 没新高 (主力不再加码)
  M7 main_force 30d 没新高
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
    code_starts = arrays['starts']; code_ends = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; mf5 = arrays['mf5']
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
        pool_max_mf = -np.inf  # 池中 main_force 最大值
        pool_max_mf_i = -1     # main_force 最高的位置 (用于 M6/M7)

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i

            if not in_pool and retail[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail[gi]
                pool_max_mf = mf[gi]
                pool_max_mf_i = i
            if in_pool:
                if retail[gi] < pool_min_retail:
                    pool_min_retail = retail[gi]
                if mf[gi] > pool_max_mf:
                    pool_max_mf = mf[gi]
                    pool_max_mf_i = i

            # 出池条件
            if in_pool:
                exit_now = False
                if exit_mode == 'M0' and retail[gi] >= 0:
                    exit_now = True
                elif exit_mode == 'M1' and mf[gi] <= 0:
                    exit_now = True
                elif exit_mode == 'M2' and mf[gi] <= 50:
                    exit_now = True
                elif exit_mode == 'M3' and mf[gi] <= -100:
                    exit_now = True
                elif exit_mode == 'M4' and not np.isnan(mf5[gi]) and mf5[gi] <= 0:
                    exit_now = True
                elif exit_mode == 'M5' and not np.isnan(mf5[gi]) and mf5[gi] <= -50:
                    exit_now = True
                elif exit_mode == 'M6' and (i - pool_max_mf_i) > 60:
                    exit_now = True
                elif exit_mode == 'M7' and (i - pool_max_mf_i) > 30:
                    exit_now = True
                # E7 baseline: 60日无 retail 新低
                elif exit_mode == 'E7':
                    # 复用之前 E7 数据
                    pass
                if exit_now:
                    in_pool = False
                    continue

            if in_pool and mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'pool_days': i - pool_enter_i,
                    'pool_min_retail': pool_min_retail,
                    'cur_retail': retail[gi],
                    'cur_mf': mf[gi],
                })

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
    print('=== retail vs main_force 关系 + 主力线出池机制 ===\n')

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
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  {len(df):,} 行')

    # 1. retail vs main_force 关系
    print('\n=== 1. retail vs main_force 分布 ===')
    print(f'  retail:     min={df["retail"].min():.0f}, max={df["retail"].max():.0f}, '
          f'avg={df["retail"].mean():.0f}, std={df["retail"].std():.0f}')
    print(f'  main_force: min={df["main_force"].min():.0f}, max={df["main_force"].max():.0f}, '
          f'avg={df["main_force"].mean():.0f}, std={df["main_force"].std():.0f}')

    # 用一小批数据看相关性
    sample = df.sample(min(100000, len(df)), random_state=42)
    corr = sample['retail'].corr(sample['main_force'])
    print(f'  retail vs main_force 相关系数: {corr:.4f}')

    # 散户极低时主力是什么状态
    print('\n=== 2. retail<-250 时 main_force 分布 ===')
    sub = df[df['retail'] < -250]
    print(f'  retail<-250 时 n={len(sub):,}')
    print(f'  对应 main_force: min={sub["main_force"].min():.0f}, max={sub["main_force"].max():.0f}, '
          f'avg={sub["main_force"].mean():.0f}, 中位 {sub["main_force"].median():.0f}')
    # 分箱
    print(f'  main_force 分布:')
    bins = [-1000, -100, 0, 100, 200, 500, 1000, np.inf]
    for i in range(len(bins)-1):
        n = ((sub['main_force'] >= bins[i]) & (sub['main_force'] < bins[i+1])).sum()
        print(f'    [{bins[i]:.0f}, {bins[i+1]:.0f}): {n:,} ({n/len(sub)*100:.1f}%)')

    # 设置数据
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
        'mf5': df['mf_5d'].to_numpy().astype(np.float64),
        'stk_d': df['stk_d'].to_numpy(),
        'mkt_y': df['mkt_y'].to_numpy(),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    all_dates = sorted(df['date'].unique())

    print('\n=== 3. K=5 资金回测对比 ===\n')
    schemes = {
        'M0 retail>=0 (当前)':         'M0',
        'M1 mf<=0':                  'M1',
        'M2 mf<=50':                 'M2',
        'M3 mf<=-100':               'M3',
        'M4 mf_5d<=0':               'M4',
        'M5 mf_5d<=-50':             'M5',
        'M6 60日 mf 无新高':            'M6',
        'M7 30日 mf 无新高':            'M7',
    }

    K = 5
    print(f'  {"方案":<28} {"事件":>6} {"信号天":>6} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7}')

    results = {}
    for label, mode in schemes.items():
        df_e = scan(arrays, mode)
        df_picks = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                          ascending=[True, True, True]).drop_duplicates('date', keep='first')
        n_sig_days = len(df_picks)
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results[label] = (r, df_picks)
        print(f'  {label:<28} {len(df_e):>6,} {n_sig_days:>6} ¥{r["final"]/1000:>8.0f}K '
              f'{r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% {r["mdd"]:>+7.1f}% '
              f'{r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}%')

    # 跨段对比 M0 vs 最优 M
    best_label = max(results, key=lambda k: results[k][0]['total'])
    print(f'\n=== 跨段对比 M0 vs {best_label} ===\n')
    print(f'  {"年":<6} | {"M0 笔/ret/win":<18} | {best_label[:14]:<18} | 差')
    m0_t = results['M0 retail>=0 (当前)'][0]['df_t']
    bb_t = results[best_label][0]['df_t']
    m0_t['year'] = pd.to_datetime(m0_t['buy_date']).dt.year
    bb_t['year'] = pd.to_datetime(bb_t['buy_date']).dt.year
    for y in sorted(set(m0_t['year'].unique()) | set(bb_t['year'].unique())):
        m0 = m0_t[m0_t['year'] == y]
        bb = bb_t[bb_t['year'] == y]
        if len(m0) and len(bb):
            print(f'  {y:<6} | {len(m0):>3} {m0["ret_pct"].mean():>+6.2f}% {(m0["ret_pct"]>0).mean()*100:>5.1f}% '
                  f'| {len(bb):>3} {bb["ret_pct"].mean():>+6.2f}% {(bb["ret_pct"]>0).mean()*100:>5.1f}% '
                  f'| {bb["ret_pct"].mean()-m0["ret_pct"].mean():>+5.2f}%')

    # 不同 K 看最优
    print(f'\n=== {best_label} 不同 K ===\n')
    print(f'  {"K":<3} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6}')
    best_picks = results[best_label][1]
    for K in [1, 3, 5, 10, 15]:
        r = run_backtest(K, best_picks, code_date_idx, close_arr, trend_arr, all_dates)
        print(f'  {K:<3} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
