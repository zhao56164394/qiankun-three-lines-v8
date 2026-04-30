# -*- coding: utf-8 -*-
"""池深池天作为排名因子 v2 — 纯排名, 不过滤

baseline = R2 (单维 retail↑, +126.0%/-26.0% MDD)

新方案 (分层 / 复合 score):
  R6: Tier 排名 (Tier A=深+新, B=深 OR 新, C=其他), 不砍信号
  R7: 复合 score (retail<-500 +1, retail<-400 +1, retail<-300 +1, days<6 +1, days<9 +1), 高分优先
  R8: 加权 score (retail 越深分越高 + days 越短分越高)

不砍信号: 信号天数永远 = 410
排名键最高分先, 同分内 retail↑
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
    print('=== 池深池天纯排名 R6/R7/R8 vs R2 baseline ===\n')

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
    print(f'  全集事件: {len(df_e):,}, {df_e["date"].nunique()} 信号天\n')

    # 各排名方案 (都不砍信号, 信号天=410)
    schemes = {}

    # R0: 无排名 (按 code)
    df_r0 = df_e.sort_values(['date', 'code']).drop_duplicates('date', keep='first')
    schemes['R0 baseline (code↑)'] = df_r0

    # R2: 单维 retail↑ (test128 实测最优)
    df_r2 = df_e.sort_values(['date', 'pool_min_retail', 'code'],
                                  ascending=[True, True, True]).drop_duplicates('date', keep='first')
    schemes['R2 retail↑'] = df_r2

    # R6: 三层 Tier 排名
    # Tier 0 (最优): retail<-500 AND days<6
    # Tier 1: retail<-500 (深) OR days<6 (新)
    # Tier 2: retail<-400
    # Tier 3: 其他
    def get_tier_v1(row):
        if row['pool_min_retail'] < -500 and row['pool_days'] < 6: return 0
        if row['pool_min_retail'] < -500 or row['pool_days'] < 6: return 1
        if row['pool_min_retail'] < -400: return 2
        return 3

    df_e['tier_v1'] = df_e.apply(get_tier_v1, axis=1)
    df_r6 = df_e.sort_values(['date', 'tier_v1', 'pool_min_retail', 'code'],
                                  ascending=[True, True, True, True]).drop_duplicates('date', keep='first')
    schemes['R6 三层 Tier (深+新先)'] = df_r6

    # R7: 简单 score (越多条件命中越优)
    df_e['rscore'] = 0
    df_e.loc[df_e['pool_min_retail'] < -500, 'rscore'] += 2  # 深池 +2
    df_e.loc[df_e['pool_min_retail'] < -400, 'rscore'] += 1  # 中池 +1
    df_e.loc[df_e['pool_days'] < 6, 'rscore'] += 1           # 新池 +1
    df_r7 = df_e.sort_values(['date', 'rscore', 'pool_min_retail', 'code'],
                                  ascending=[True, False, True, True]).drop_duplicates('date', keep='first')
    schemes['R7 score (深+2/中+1/新+1)'] = df_r7

    # R8: 把池深、池天合成 z-score 排名
    # 用经验: pool_min_retail 区间 [-1000, -250], pool_days 区间 [0, 30+]
    # 标准化: depth_z = (-pool_min_retail - 250) / 750  (深池 -> 大)
    #         days_z = max(0, 30 - pool_days) / 30   (新池 -> 大)
    df_e['depth_z'] = (-df_e['pool_min_retail'] - 250) / 750
    df_e['days_z'] = np.clip((30 - df_e['pool_days']) / 30, 0, 1)
    df_e['composite'] = df_e['depth_z'] * 0.7 + df_e['days_z'] * 0.3
    df_r8 = df_e.sort_values(['date', 'composite', 'code'],
                                  ascending=[True, False, True]).drop_duplicates('date', keep='first')
    schemes['R8 加权 (depth*0.7+days*0.3)'] = df_r8

    # R9: 日 + 深, 不要 retail 二级
    df_e['days_first'] = df_e['pool_days'] < 6
    df_r9 = df_e.sort_values(['date', 'days_first', 'pool_min_retail', 'code'],
                                  ascending=[True, False, True, True]).drop_duplicates('date', keep='first')
    schemes['R9 days<6 优先 + retail↑'] = df_r9

    all_dates = sorted(df['date'].unique())
    K = 5
    print(f'=== K={K} 各排名方案对比 (信号天都=410, 不砍) ===\n')
    print(f'  {"方案":<32} {"期末":>10} {"总收益":>9} {"年化":>9} {"MDD":>9} {"笔":>4} {"胜率":>6} {"均/笔":>7} {"平均仓":>7}')

    results = {}
    for label, df_picks in schemes.items():
        r = run_backtest(K, df_picks, code_date_idx, close_arr, trend_arr, all_dates)
        results[label] = r
        print(f'  {label:<32} ¥{r["final"]/1000:>8.0f}K {r["total"]:>+7.1f}% {r["annual"]:>+7.2f}% '
              f'{r["mdd"]:>+7.1f}% {r["n_trades"]:>4} {r["win"]:>5.1f}% {r["avg"]:>+5.2f}% {r["avg_pos"]:>5.1f}%')

    # 跨段对比 R2 vs 最优新方案
    best_label = max(results, key=lambda k: results[k]['total'])
    print(f'\n=== 跨段稳定性: R2 vs {best_label} (买入年份) ===\n')
    print(f'  {"年":<6} | R2 笔/ret/win                | best 笔/ret/win              | 收益差')
    r2_t = results['R2 retail↑']['df_t']
    bb_t = results[best_label]['df_t']
    r2_t['year'] = pd.to_datetime(r2_t['buy_date']).dt.year
    bb_t['year'] = pd.to_datetime(bb_t['buy_date']).dt.year
    for y in sorted(set(r2_t['year'].unique()) | set(bb_t['year'].unique())):
        r2_sub = r2_t[r2_t['year'] == y]
        bb_sub = bb_t[bb_t['year'] == y]
        r2_n, bb_n = len(r2_sub), len(bb_sub)
        r2_r = r2_sub['ret_pct'].mean() if r2_n else 0
        bb_r = bb_sub['ret_pct'].mean() if bb_n else 0
        r2_w = (r2_sub['ret_pct']>0).mean()*100 if r2_n else 0
        bb_w = (bb_sub['ret_pct']>0).mean()*100 if bb_n else 0
        print(f'  {y:<6} | {r2_n:>3} {r2_r:>+6.2f}% {r2_w:>5.1f}%             | '
              f'{bb_n:>3} {bb_r:>+6.2f}% {bb_w:>5.1f}%             | {bb_r-r2_r:+5.2f}%')

    # 排名一致性: R2 和最优新方案的 picks 重合度
    print(f'\n=== R2 vs {best_label} picks 重合度 ===')
    r2_picks_set = set(zip(schemes['R2 retail↑']['date'], schemes['R2 retail↑']['code']))
    bb_picks_set = set(zip(schemes[best_label]['date'], schemes[best_label]['code']))
    overlap = r2_picks_set & bb_picks_set
    print(f'  R2 picks: {len(r2_picks_set)}')
    print(f'  best picks: {len(bb_picks_set)}')
    print(f'  重合: {len(overlap)} ({len(overlap)/len(r2_picks_set)*100:.1f}%)')
    print(f'  best 独有: {len(bb_picks_set - r2_picks_set)}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
