# -*- coding: utf-8 -*-
"""八卦分治资金回测 v6 — 仓位上限 K × 每日买入 N 二维网格

固定: v4/v5 的 quality 排序逻辑 + MIN_QUALITY=7 (v5 q=7 最优)
变量:
  K (持仓上限): 5 / 10 / 15 / 20 / 30
  N (每日新增买入上限): 1 / 3 / 5 / 10 / K (≤K)

每个组合跑一次, 输出对比表.
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
MAX_HOLD_DAYS = 60
MIN_QUALITY = 7

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

QUALITY_TABLE = {
    ('坎 v3', 3): 20, ('坎 v3', 4): 25, ('坎 v3', 5): 30,
    ('坎 v3', 2): 13,
    ('坤 v3', 3): 12, ('坤 v3', 4): 15,
    ('坤 v3', 2): 10,
    ('震 v1', 2): 9, ('震 v1', 1): 8,
    ('坤 v3', 1): 7,
    ('坎 v3', 1): 6,
    ('乾 v3', 1): 5, ('乾 v3', 2): 5,
    ('离 v1', 1): 4,
    ('坤 v3', 0): 3,
    ('兑 v1', 1): 2,
    ('坎 v3', 0): 1,
}


def regime_buy_decide(mkt_y, mkt_d, mkt_m, stk_d, stk_m, stk_y,
                      ret_10d=None, mf=None, mf_5d=None, sanhu_5d=None):
    if mkt_y == '011': return None
    if mkt_y == '000':
        if stk_d != '011': return None
        if stk_m in {'101', '110', '111'}: return None
        if stk_y in {'001', '011'}: return None
        if mkt_d in {'000', '001', '100', '101'}: return None
        score = 0
        if mkt_m == '100': score += 1
        if mkt_d == '011': score += 1
        if mf is not None and not np.isnan(mf) and mf > 100: score += 1
        if stk_m == '010': score += 1
        return ('坤 v3', score)
    if mkt_y == '001': return None  # 艮关闭
    if mkt_y == '010':
        if stk_d != '011': return None
        if mkt_m in {'100', '110'}: return None
        if stk_y == '111': return None
        if stk_m == '110': return None
        score = 0
        if mkt_m == '011': score += 1
        if mkt_d == '001': score += 1
        if mf_5d is not None and not np.isnan(mf_5d) and mf_5d < -50: score += 1
        if mf is not None and not np.isnan(mf) and mf > 100: score += 1
        if sanhu_5d is not None and not np.isnan(sanhu_5d) and sanhu_5d < -100: score += 1
        return ('坎 v3', score)
    if mkt_y == '100':
        if stk_d != '010': return None
        if mkt_d in {'101', '111'}: return None
        if stk_y == '111': return None
        score = 0
        if mkt_d == '011': score += 1
        if stk_m == '110': score += 1
        if score < 1: return None
        return ('震 v1', score)
    if mkt_y == '101':
        if stk_d != '000': return None
        if mkt_d == '101': return None
        if stk_m in {'011', '001', '101'}: return None
        if stk_y == '011': return None
        return ('离 v1', 1)
    if mkt_y == '110':
        if stk_d != '000': return None
        if mkt_d == '011': return None
        if stk_m in {'001', '011', '101', '111'}: return None
        return ('兑 v1', 1)
    if mkt_y == '111':
        if stk_d != '011': return None
        if mkt_d in {'100', '101', '110'}: return None
        if mkt_m == '101': return None
        if stk_m in {'100', '101'}: return None
        if ret_10d is not None and ret_10d > 15: return None
        score = 0
        if stk_m == '010': score += 1
        if stk_y == '010': score += 1
        if score < 1: return None
        return ('乾 v3', score)
    return None


def get_quality(regime, score):
    return QUALITY_TABLE.get((regime, score), 0)


def should_sell(td_buy_to_now, days_held, regime):
    if days_held >= MAX_HOLD_DAYS:
        return True, 'timeout'
    if len(td_buy_to_now) < 2:
        return False, None
    if regime == '坤 v3' and days_held >= 20:
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


def run_backtest(K, N, df, code_arr, date_arr, close_arr, trend_arr,
                 stk_d_arr, stk_m_arr, stk_y_arr, mf_arr, mf5_arr, sh5_arr,
                 code_index, code_date_idx, df_by_date, all_dates, mkt_lookup):
    """K = 持仓上限, N = 每日买入上限 (N ≤ K)"""
    N = min(N, K)
    SLOT_VALUE = INIT_CAPITAL / K  # 单股资金 = 总额 / 持仓数

    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []

    for di, today in enumerate(all_dates):
        if today not in mkt_lookup: continue
        mkt = mkt_lookup[today]
        mkt_y = mkt['mkt_y']; mkt_d = mkt['mkt_d']; mkt_m = mkt['mkt_m']

        # 卖
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]: continue
            today_idx = code_date_idx[code][today]
            buy_idx = pos['buy_idx_global']
            days_held = today_idx - buy_idx
            td_seg = trend_arr[buy_idx:today_idx+1]
            sell, reason = should_sell(td_seg, days_held, pos['regime'])
            if sell:
                sell_price = close_arr[today_idx]
                proceeds = pos['shares'] * sell_price
                cost = pos['shares'] * pos['buy_price']
                ret_pct = (sell_price / pos['buy_price'] - 1) * 100
                cash += proceeds
                trades.append({
                    'buy_date': pos['buy_date'], 'sell_date': today,
                    'profit': proceeds - cost, 'ret_pct': ret_pct,
                    'days': days_held, 'regime': pos['regime'],
                    'score': pos['score'], 'quality': pos['quality'],
                    'reason': reason,
                })
                del holdings[code]

        # 候选
        candidates = []
        if today in df_by_date.groups:
            today_idx_in_df = df_by_date.groups[today]
            for ridx in today_idx_in_df:
                code = code_arr[ridx]
                if code in holdings: continue
                ret_10d = None
                if mkt_y == '111':
                    if code in code_date_idx and today in code_date_idx[code]:
                        ti = code_date_idx[code][today]
                        cs, _ = code_index[code]
                        if ti - cs >= 10:
                            ret_10d = (close_arr[ti] / close_arr[ti-10] - 1) * 100
                decide = regime_buy_decide(
                    mkt_y, mkt_d, mkt_m,
                    stk_d_arr[ridx], stk_m_arr[ridx], stk_y_arr[ridx],
                    ret_10d=ret_10d,
                    mf=mf_arr[ridx], mf_5d=mf5_arr[ridx], sanhu_5d=sh5_arr[ridx],
                )
                if decide is None: continue
                regime, score = decide
                quality = get_quality(regime, score)
                if quality < MIN_QUALITY: continue
                candidates.append({
                    'code': code, 'ridx': ridx, 'regime': regime,
                    'score': score, 'quality': quality,
                })

        candidates.sort(key=lambda x: (-x['quality'], x['code']))

        # 买: 每日上限 N, 持仓上限 K
        slots_left = K - len(holdings)
        max_buy = min(slots_left, N)
        if max_buy > 0 and candidates:
            for cand in candidates[:max_buy]:
                ridx = cand['ridx']
                buy_price = close_arr[ridx]
                if np.isnan(buy_price) or buy_price <= 0: continue
                shares_avail = int(SLOT_VALUE // buy_price // 100) * 100
                if shares_avail <= 0: continue
                cost = shares_avail * buy_price
                if cost > cash: continue
                cash -= cost
                holdings[cand['code']] = {
                    'buy_date': today, 'buy_idx_global': ridx,
                    'buy_price': buy_price, 'shares': shares_avail,
                    'regime': cand['regime'], 'score': cand['score'],
                    'quality': cand['quality'],
                }

        # NAV
        market_value = 0.0
        for code, pos in holdings.items():
            if code in code_date_idx and today in code_date_idx[code]:
                ti = code_date_idx[code][today]
                market_value += pos['shares'] * close_arr[ti]
            else:
                market_value += pos['shares'] * pos['buy_price']
        nav = cash + market_value
        nav_history.append({'date': today, 'nav': nav, 'pos_count': len(holdings)})

    # 收尾
    last_date = all_dates[-1]
    for code, pos in list(holdings.items()):
        if code in code_date_idx and last_date in code_date_idx[code]:
            ti = code_date_idx[code][last_date]
            sell_price = close_arr[ti]
            proceeds = pos['shares'] * sell_price
            cost = pos['shares'] * pos['buy_price']
            ret_pct = (sell_price / pos['buy_price'] - 1) * 100
            cash += proceeds
            trades.append({
                'buy_date': pos['buy_date'], 'sell_date': last_date,
                'profit': proceeds - cost, 'ret_pct': ret_pct,
                'days': ti - pos['buy_idx_global'],
                'regime': pos['regime'], 'score': pos['score'],
                'quality': pos['quality'], 'reason': 'force_close',
            })

    df_tr = pd.DataFrame(trades)
    df_nav = pd.DataFrame(nav_history)
    final = df_nav['nav'].iloc[-1]
    days = (pd.to_datetime(df_nav['date'].iloc[-1]) - pd.to_datetime(df_nav['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_nav['peak'] = df_nav['nav'].cummax()
    mdd = ((df_nav['nav']-df_nav['peak'])/df_nav['peak']*100).min()
    win_rate = (df_tr['ret_pct']>0).mean()*100 if len(df_tr) else 0
    avg_ret = df_tr['ret_pct'].mean() if len(df_tr) else 0
    full = (df_nav['pos_count']==K).sum()/len(df_nav)*100
    empty = (df_nav['pos_count']==0).sum()/len(df_nav)*100
    return {
        'K': K, 'N': N,
        'final': final, 'total_ret': (final/INIT_CAPITAL-1)*100,
        'annual': annual, 'mdd': mdd, 'n_trades': len(df_tr),
        'win': win_rate, 'avg_ret': avg_ret, 'full%': full, 'empty%': empty,
    }


def main():
    t0 = time.time()
    print(f'=== 八卦分治资金回测 v6 — 仓位 × 每日上限 网格 ===')
    print(f'  固定: quality 排序 + MIN_QUALITY={MIN_QUALITY}')
    print(f'  K (持仓上限): 5 / 10 / 15 / 20')
    print(f'  N (每日新买上限): 1 / 3 / 5 / 10 / K')

    print(f'\n=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date').reset_index(drop=True)
    mkt_lookup = market.set_index('date').to_dict('index')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_index = {code_arr[code_starts[i]]: (code_starts[i], code_ends[i]) for i in range(len(code_starts))}

    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        code = code_arr[s]
        code_date_idx[code] = {date_arr[s+j]: s+j for j in range(e-s)}

    df_by_date = df.groupby('date', sort=True)
    all_dates = sorted(df['date'].unique())
    print(f'  日期范围: {all_dates[0]} → {all_dates[-1]}, {time.time()-t0:.1f}s')

    print(f'\n=== 网格扫描 ===')
    results = []
    grid = []
    for K in [5, 10, 15, 20]:
        for N in [1, 3, 5, 10, K]:
            if N > K: continue
            grid.append((K, N))
    grid = sorted(set(grid))

    print(f'  共 {len(grid)} 个组合\n')
    for K, N in grid:
        t1 = time.time()
        r = run_backtest(K, N, df, code_arr, date_arr, close_arr, trend_arr,
                         stk_d_arr, stk_m_arr, stk_y_arr, mf_arr, mf5_arr, sh5_arr,
                         code_index, code_date_idx, df_by_date, all_dates, mkt_lookup)
        results.append(r)
        print(f'  K={K} N={N}: 期末¥{r["final"]/1000:>5.0f}K 收益{r["total_ret"]:>+6.1f}% '
              f'年{r["annual"]:>+5.2f}% MDD{r["mdd"]:>+5.1f}% 笔{r["n_trades"]:>3} 胜{r["win"]:>4.1f}% '
              f'均{r["avg_ret"]:>+4.1f}% 满{r["full%"]:>4.0f}% 空{r["empty%"]:>3.0f}% '
              f'({time.time()-t1:.1f}s)')

    df_res = pd.DataFrame(results)
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    df_res.to_csv(os.path.join(out_dir, 'capital_v6_grid.csv'), index=False)
    print(f'\n  写出 capital_v6_grid.csv')
    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
