# -*- coding: utf-8 -*-
"""八卦分治资金回测 v3 — 基于 v2 实测拆解, 只买高质量 score 信号

v2 实测发现 (按 regime × score 拆解):
  坤 score=3 (+9.82%, 27 笔, +54K) ⭐
  坎 score=3 (+12.85%, 16 笔, +39K) ⭐
  震 score=1 (+13.31%, 27 笔, +69K) ⭐
  其他: 坤 score=2 / 坎 score=2 / 乾 / 艮 / 离 / 兑 全亏

v3 决策:
  - 坤 v3 提门槛: score ≥ 3 (不是 ≥2)
  - 坎 v3 提门槛: score ≥ 3 (不是 ≥2)
  - 震 v1 保留: score ≥ 1
  - 乾 v3 关闭 (35-30% 胜率, 整体 -¥10K)
  - 艮 v3 关闭 (39% 胜率, -¥32K)
  - 离 v1 关闭 (47% 胜率, -¥10K)
  - 兑 v1 关闭 (50% 胜率, -¥4K)
  - 巽 不买

底仓 20 万, 每日上限 10 只, 等资金
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

INIT_CAPITAL = 200_000
MAX_PER_DAY = 10
MAX_HOLD_DAYS = 60
SLOT_VALUE = INIT_CAPITAL / MAX_PER_DAY

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


# ============================================================
# 入场判断 (v2: 完整 score)
# ============================================================
def regime_buy_decide(mkt_y, mkt_d, mkt_m, stk_d, stk_m, stk_y,
                     ret_10d=None, mf=None, mf_5d=None, sanhu_5d=None):
    """返回 (regime_name, score) 或 None"""
    # 巽 不买
    if mkt_y == '011':
        return None

    # 坤 v3: 巽日 + 9 避雷 + score≥3 (4 项: 大m=震, 大d=巽, mf>100, 股m=坎) — v3 提门槛
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
        if score < 3: return None  # v3: 提到 ≥3
        return ('坤 v3', score)

    # 艮 v3: 关闭 (实测 -6%, 39% 胜率)
    if mkt_y == '001':
        return None

    # 坎 v3 提门槛 score≥3 (5 项: 大m=巽, 大d=艮, mf_5d<-50, mf>100, sanhu_5d<-100)
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
        if score < 3: return None  # v3: 提到 ≥3
        return ('坎 v3', score)

    # 震 v1: 保留 score≥1
    if mkt_y == '100':
        if stk_d != '010': return None
        if mkt_d in {'101', '111'}: return None
        if stk_y == '111': return None
        score = 0
        if mkt_d == '011': score += 1
        if stk_m == '110': score += 1
        if score < 1: return None
        return ('震 v1', score)

    # 离 v1: 关闭 (实测 -1.54%)
    if mkt_y == '101':
        return None

    # 兑 v1: 关闭 (实测 -0.90%)
    if mkt_y == '110':
        return None

    # 乾 v3: 关闭 (score=2 -2.6, score=1 -2.6, 整体 -¥10K)
    if mkt_y == '111':
        return None

    return None


# ============================================================
# 卖点
# ============================================================
def should_sell(td_buy_to_now, days_held, regime):
    """
    bull (跨 regime 主路): 第 2 次下穿 89
    + 60 日兜底
    + 坤 v3 加 TS20: 持仓 ≥ 20 日 且 trend 从未到 89 → 强卖
    """
    if days_held >= MAX_HOLD_DAYS:
        return True, 'timeout'
    if len(td_buy_to_now) < 2:
        return False, None

    # 坤 v3 TS20
    if regime == '坤 v3' and days_held >= 20:
        valid = td_buy_to_now[~np.isnan(td_buy_to_now)]
        if len(valid) > 0 and valid.max() < 89:
            return True, 'ts20'

    # bull 主路
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
    print(f'=== 八卦分治资金回测 v3 ===')
    print(f'  v2 → v3 修正:')
    print(f'    1. 坤 v3 提门槛 score≥3 (实测 score=2 +0.6 / score=3 +9.8)')
    print(f'    2. 坎 v3 提门槛 score≥3 (实测 score=2 -5.3 / score=3 +12.9)')
    print(f'    3. 关闭 乾/艮/离/兑 (实测都亏)')
    print(f'    4. 保留 震 v1 (实测 score=1 +13.31% ⭐)')
    print(f'  策略池: 仅 坤 v3 (score≥3) / 坎 v3 (score≥3) / 震 v1 (score≥1)')
    print()
    print(f'  底仓: ¥{INIT_CAPITAL:,}, 每日上限 {MAX_PER_DAY} 只 (¥{SLOT_VALUE:,.0f}/只)')

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

    # 计算 mf_5d_mean, sanhu_5d_mean (按 code 分组)
    print(f'  计算 mf_5d / sanhu_5d 滑动均值...')
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  滑动完成 {time.time()-t0:.1f}s')

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

    print(f'  构建索引...')
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        code = code_arr[s]
        code_date_idx[code] = {date_arr[s+j]: s+j for j in range(e-s)}

    print(f'  按日期分组...')
    df_by_date = df.groupby('date', sort=True)
    all_dates = sorted(df['date'].unique())
    print(f'  日期范围: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} 个交易日), {time.time()-t0:.1f}s')

    print(f'\n=== 开始回测 ===')

    cash = INIT_CAPITAL
    holdings = {}
    trades = []
    nav_history = []

    log_every = 250

    for di, today in enumerate(all_dates):
        if today not in mkt_lookup:
            continue
        mkt = mkt_lookup[today]
        mkt_y = mkt['mkt_y']; mkt_d = mkt['mkt_d']; mkt_m = mkt['mkt_m']

        # 1. 卖出
        sells_today = []
        for code, pos in list(holdings.items()):
            if code not in code_date_idx or today not in code_date_idx[code]:
                continue
            today_idx = code_date_idx[code][today]
            buy_idx = pos['buy_idx_global']
            days_held = today_idx - buy_idx

            td_seg = trend_arr[buy_idx:today_idx+1]
            sell, reason = should_sell(td_seg, days_held, pos['regime'])

            if sell:
                sell_price = close_arr[today_idx]
                proceeds = pos['shares'] * sell_price
                cost = pos['shares'] * pos['buy_price']
                profit = proceeds - cost
                ret_pct = (sell_price / pos['buy_price'] - 1) * 100
                cash += proceeds
                trades.append({
                    'code': code, 'buy_date': pos['buy_date'], 'sell_date': today,
                    'buy_price': pos['buy_price'], 'sell_price': sell_price,
                    'shares': pos['shares'], 'cost': cost, 'proceeds': proceeds,
                    'profit': profit, 'ret_pct': ret_pct, 'days': days_held,
                    'regime': pos['regime'], 'score': pos['score'], 'reason': reason,
                })
                sells_today.append(code)
                del holdings[code]

        # 2. 扫今日候选
        candidates = []
        if today in df_by_date.groups:
            today_idx_in_df = df_by_date.groups[today]
            for ridx in today_idx_in_df:
                code = code_arr[ridx]
                if code in holdings: continue

                # ret_10d (乾 regime 用)
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
                candidates.append({
                    'code': code, 'ridx': ridx, 'regime': regime, 'score': score,
                    'close': close_arr[ridx],
                })

        candidates.sort(key=lambda x: (-x['score'], x['code']))

        # 3. 买入 (T+0 当日 close)
        slots_left = MAX_PER_DAY - len(holdings)
        if slots_left > 0 and candidates:
            picks = candidates[:slots_left]
            for cand in picks:
                code = cand['code']
                ridx = cand['ridx']
                buy_price = close_arr[ridx]  # T+0 当日 close
                if np.isnan(buy_price) or buy_price <= 0: continue
                shares_avail = int(SLOT_VALUE // buy_price // 100) * 100
                if shares_avail <= 0: continue
                cost = shares_avail * buy_price
                if cost > cash: continue
                cash -= cost
                holdings[code] = {
                    'buy_date': today,
                    'buy_idx_global': ridx,
                    'buy_price': buy_price,
                    'shares': shares_avail,
                    'regime': cand['regime'],
                    'score': cand['score'],
                }

        # 4. NAV
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

        if (di + 1) % log_every == 0:
            print(f'  day {di+1}/{len(all_dates)} {today}: '
                  f'NAV ¥{nav:>10,.0f} cash {cash:>9,.0f} '
                  f'pos {len(holdings):>2} {GUA_NAMES.get(mkt_y, "?")} '
                  f'sells {len(sells_today)}')

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
                'code': code, 'buy_date': pos['buy_date'], 'sell_date': last_date,
                'buy_price': pos['buy_price'], 'sell_price': sell_price,
                'shares': pos['shares'], 'cost': cost, 'proceeds': proceeds,
                'profit': proceeds - cost, 'ret_pct': ret_pct,
                'days': ti - pos['buy_idx_global'],
                'regime': pos['regime'], 'score': pos['score'],
                'reason': 'force_close',
            })
            del holdings[code]

    # ============================================================
    print(f'\n=== 回测结果 ===')
    df_trades = pd.DataFrame(trades)
    df_nav = pd.DataFrame(nav_history)

    final_nav = df_nav['nav'].iloc[-1]
    total_return = (final_nav / INIT_CAPITAL - 1) * 100
    days = (pd.to_datetime(df_nav['date'].iloc[-1]) - pd.to_datetime(df_nav['date'].iloc[0])).days
    annual_return = ((final_nav / INIT_CAPITAL) ** (365/days) - 1) * 100 if days > 0 else 0

    df_nav['nav_peak'] = df_nav['nav'].cummax()
    df_nav['drawdown'] = (df_nav['nav'] - df_nav['nav_peak']) / df_nav['nav_peak'] * 100
    mdd = df_nav['drawdown'].min()

    win_rate = (df_trades['ret_pct'] > 0).mean() * 100 if len(df_trades) > 0 else 0
    avg_ret = df_trades['ret_pct'].mean() if len(df_trades) > 0 else 0
    avg_days = df_trades['days'].mean() if len(df_trades) > 0 else 0

    print(f'\n  期初: ¥{INIT_CAPITAL:,}')
    print(f'  期末: ¥{final_nav:>10,.0f}')
    print(f'  总收益: {total_return:>+.2f}%')
    print(f'  年化收益: {annual_return:>+.2f}%')
    print(f'  最大回撤: {mdd:>.2f}%')
    print(f'  交易笔数: {len(df_trades):,}')
    print(f'  胜率: {win_rate:.1f}%')
    print(f'  平均收益: {avg_ret:>+.2f}%')
    print(f'  平均持仓: {avg_days:.1f} 日')

    if len(df_trades) > 0:
        print(f'\n  按 regime 分解:')
        print(f'  {"regime":<10} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"总盈亏":>11} {"均持仓":>7}')
        for r in df_trades['regime'].unique():
            sub = df_trades[df_trades['regime'] == r]
            print(f'  {r:<10} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} ¥{sub["profit"].sum():>10,.0f} '
                  f'{sub["days"].mean():>6.1f}')

        print(f'\n  按 reason:')
        print(f'  {"reason":<14} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"均持仓":>7}')
        for r in df_trades['reason'].unique():
            sub = df_trades[df_trades['reason'] == r]
            print(f'  {r:<14} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} {sub["days"].mean():>6.1f}')

        print(f'\n  按年:')
        df_trades['year'] = pd.to_datetime(df_trades['buy_date']).dt.year
        print(f'  {"year":<6} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"总盈亏":>11}')
        for y in sorted(df_trades['year'].unique()):
            sub = df_trades[df_trades['year'] == y]
            print(f'  {y:<6} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} ¥{sub["profit"].sum():>10,.0f}')

    print(f'\n  持仓数分布:')
    pc = df_nav['pos_count'].value_counts().sort_index()
    print(f'  {"持仓":<6} {"日数":>6} {"占比%":>7}')
    for k, v in pc.items():
        print(f'  {k:<6} {v:>6} {v/len(df_nav)*100:>6.1f}')

    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_nav.to_csv(os.path.join(out_dir, 'capital_nav_v3.csv'), index=False, encoding='utf-8-sig')
    df_trades.to_csv(os.path.join(out_dir, 'capital_trades_v3.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出: data_layer/data/results/capital_{{nav,trades}}_v3.csv')
    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
