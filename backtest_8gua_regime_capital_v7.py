# -*- coding: utf-8 -*-
"""八卦分治资金回测 v7 — 加主板过滤

v6 K×N 网格最优 K=15 N=15: 总收益 +197.3%/-35.5%
但 v6 没过滤主板, 含 36 笔创业板 (+65K) + 4 笔科创板 (-7K).

v7 加 main_board_universe 过滤, 仅保留主板 (含 ST), 重跑 K=15 N=15.
固定: quality 排序 + MIN_QUALITY=7
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
K = 15
N = 15
MAX_HOLD_DAYS = 60
MIN_QUALITY = 7
SLOT_VALUE = INIT_CAPITAL / K

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
    if mkt_y == '001': return None
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


def main():
    t0 = time.time()
    print(f'=== 八卦分治资金回测 v7 (主板过滤) ===')
    print(f'  K={K} 持仓 / N={N} 每日 / SLOT=¥{SLOT_VALUE:,.0f}')
    print(f'  仅主板 (含 ST), 排除创业板/科创板/北交所')

    print(f'\n=== 加载 ===')
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())
    print(f'  主板: {len(main_codes):,} 只')

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
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date').reset_index(drop=True)
    mkt_lookup = market.set_index('date').to_dict('index')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
    print(f'  {len(df):,} 行 (主板过滤后), {time.time()-t0:.1f}s')

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
    print(f'  日期范围: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} 个交易日)')

    print(f'\n=== 开始回测 ===')

    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []

    for di, today in enumerate(all_dates):
        if today not in mkt_lookup: continue
        mkt = mkt_lookup[today]
        mkt_y = mkt['mkt_y']; mkt_d = mkt['mkt_d']; mkt_m = mkt['mkt_m']

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
                    'code': code, 'buy_date': pos['buy_date'], 'sell_date': today,
                    'buy_price': pos['buy_price'], 'sell_price': sell_price,
                    'shares': pos['shares'], 'cost': cost, 'proceeds': proceeds,
                    'profit': proceeds - cost, 'ret_pct': ret_pct, 'days': days_held,
                    'regime': pos['regime'], 'score': pos['score'],
                    'quality': pos['quality'], 'reason': reason,
                })
                del holdings[code]

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

        market_value = 0.0
        for code, pos in holdings.items():
            if code in code_date_idx and today in code_date_idx[code]:
                ti = code_date_idx[code][today]
                market_value += pos['shares'] * close_arr[ti]
            else:
                market_value += pos['shares'] * pos['buy_price']
        nav = cash + market_value
        nav_history.append({'date': today, 'cash': cash, 'mv': market_value,
                            'nav': nav, 'pos_count': len(holdings),
                            'mkt_y': mkt_y})

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
                'code': code, 'buy_date': pos['buy_date'], 'sell_date': last_date,
                'buy_price': pos['buy_price'], 'sell_price': sell_price,
                'shares': pos['shares'], 'cost': cost, 'proceeds': proceeds,
                'profit': proceeds - cost, 'ret_pct': ret_pct,
                'days': ti - pos['buy_idx_global'],
                'regime': pos['regime'], 'score': pos['score'],
                'quality': pos['quality'], 'reason': 'force_close',
            })

    print(f'\n=== 回测结果 (主板) ===')
    df_tr = pd.DataFrame(trades)
    df_nav = pd.DataFrame(nav_history)

    final = df_nav['nav'].iloc[-1]
    days = (pd.to_datetime(df_nav['date'].iloc[-1]) - pd.to_datetime(df_nav['date'].iloc[0])).days
    annual = ((final/INIT_CAPITAL)**(365/days)-1)*100 if days > 0 else 0
    df_nav['peak'] = df_nav['nav'].cummax()
    mdd = ((df_nav['nav']-df_nav['peak'])/df_nav['peak']*100).min()
    win = (df_tr['ret_pct']>0).mean()*100 if len(df_tr) else 0
    avg = df_tr['ret_pct'].mean() if len(df_tr) else 0

    print(f'  期末: ¥{final:,.0f}')
    print(f'  总收益: {(final/INIT_CAPITAL-1)*100:+.2f}%')
    print(f'  年化: {annual:+.2f}%')
    print(f'  MDD: {mdd:.2f}%')
    print(f'  笔数: {len(df_tr)}')
    print(f'  胜率: {win:.1f}%')
    print(f'  均收益: {avg:+.2f}%')
    print(f'  满仓率: {(df_nav["pos_count"]==K).sum()/len(df_nav)*100:.1f}%')
    print(f'  空仓率: {(df_nav["pos_count"]==0).sum()/len(df_nav)*100:.1f}%')

    if len(df_tr):
        print(f'\n  按 regime × score:')
        for r in sorted(df_tr['regime'].unique()):
            for sc in sorted(df_tr[df_tr['regime'] == r]['score'].unique()):
                sub = df_tr[(df_tr['regime'] == r) & (df_tr['score'] == sc)]
                if len(sub) == 0: continue
                q = sub['quality'].iloc[0]
                print(f'  {r:<10} sc={sc} q={q:>2}: n={len(sub):>4} '
                      f'胜 {(sub["ret_pct"]>0).mean()*100:>5.1f}% '
                      f'均 {sub["ret_pct"].mean():>+6.2f}% '
                      f'盈亏 ¥{sub["profit"].sum():>+10,.0f}')

        print(f'\n  按年:')
        df_tr['year'] = pd.to_datetime(df_tr['buy_date']).dt.year
        for y in sorted(df_tr['year'].unique()):
            sub = df_tr[df_tr['year'] == y]
            print(f'  {y}: n={len(sub):>3} 胜 {(sub["ret_pct"]>0).mean()*100:>5.1f}% '
                  f'均 {sub["ret_pct"].mean():>+6.2f}% 盈亏 ¥{sub["profit"].sum():>+10,.0f}')

        print(f'\n  按入场价:')
        for lo, hi, label in [(0,3,'<3'),(3,5,'3-5'),(5,10,'5-10'),(10,20,'10-20'),(20,50,'20-50'),(50,200,'50+')]:
            sub = df_tr[(df_tr['buy_price']>=lo) & (df_tr['buy_price']<hi)]
            if len(sub):
                print(f'  {label:>6}元: n={len(sub):>3} 胜 {(sub["ret_pct"]>0).mean()*100:>5.1f}% '
                      f'均 {sub["ret_pct"].mean():>+6.2f}% 盈亏 ¥{sub["profit"].sum():>+10,.0f}')

    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_nav.to_csv(os.path.join(out_dir, 'capital_nav_v7.csv'), index=False, encoding='utf-8-sig')
    df_tr.to_csv(os.path.join(out_dir, 'capital_trades_v7.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出 capital_{{nav,trades}}_v7.csv')
    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
