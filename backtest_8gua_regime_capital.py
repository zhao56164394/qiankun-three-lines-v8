# -*- coding: utf-8 -*-
"""八卦分治资金回测 v1

底仓 20 万, 每日上限 10 只 (2 万/只), 按 score 排序选股, 等资金.

策略 (按 strategy_*.md):
  坤 v3 (000): 巽日 + 9 避雷 + score(大m=震/大d=巽/股m=坎)≥2
  艮 v3 (001): 巽日, 卖点 td80→m_dui (这里简化用 bull)
  坎 v3 (010): 巽日 + 4 避雷 + score(大m=震/大d=巽/股m=坎/mf 简化)≥2
  巽    (011): 不买
  震 v1 (100): 坎日 + 3 弱避雷 + score(大d=巽/股m=兑)≥1
  离 v1 (101): 坤日 + 5 避雷
  兑 v1 (110): 坤日 + 5 避雷
  乾 v3 (111): 巽日 + 6 避雷 + 涨幅<15% + score(股m=坎/股y=坎)≥1

卖点: 全部 bull (第 2 次下穿 89), 60 日兜底
  - 例外: 艮 v3 用 td80→m_dui (单独维护)

按日模拟:
  Day t (T+0 信号):
    1. 检查持仓: 算每只股的卖出信号 → 按收盘价卖
    2. 扫 t 日的所有触发信号 (按 regime mask 过滤)
    3. 按 score 排序, 取 Top K (K = max_per_day - 当前持仓数)
    4. 第 t+1 日开盘买入 (T+1)
"""
import os, sys, io, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 资金配置
# ============================================================
INIT_CAPITAL = 200_000  # 底仓 20 万
MAX_PER_DAY = 10        # 每日上限 10 只
MAX_HOLD_DAYS = 60      # 兜底 60 日
SLOT_VALUE = INIT_CAPITAL / MAX_PER_DAY  # 单股 2 万

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


# ============================================================
# 入场判断 (按 8 regime)
# ============================================================
def regime_buy_decide(mkt_y, mkt_d, mkt_m, stk_d, stk_m, stk_y, ret_10d=None):
    """返回 (regime_name, score) 或 None
    mkt/stk 是单值, regime 是字符串 (如 '坤 v3')
    """
    # 巽 regime 不买
    if mkt_y == '011':
        return None

    # 坤 v3
    if mkt_y == '000':
        if stk_d != '011': return None
        # 9 避雷
        if stk_m in {'101', '110', '111'}: return None
        if stk_y in {'001', '011'}: return None
        if mkt_d in {'000', '001', '100', '101'}: return None
        # score (3 项, 简化, 不含 mf)
        score = 0
        if mkt_m == '100': score += 1
        if mkt_d == '011': score += 1
        if stk_m == '010': score += 1
        if score < 2: return None
        return ('坤 v3', score)

    # 艮 v3
    if mkt_y == '001':
        if stk_d != '011': return None
        return ('艮 v3', 1)

    # 坎 v3
    if mkt_y == '010':
        if stk_d != '011': return None
        # score 简化版 (用 4 项中 3 项, mf 没有)
        score = 0
        if mkt_m == '100': score += 1
        if mkt_d == '011': score += 1
        if stk_m == '010': score += 1
        if score < 2: return None
        return ('坎 v3', score)

    # 震 v1
    if mkt_y == '100':
        if stk_d != '010': return None  # 坎触发
        if mkt_d in {'101', '111'}: return None
        if stk_y == '111': return None
        # score
        score = 0
        if mkt_d == '011': score += 1
        if stk_m == '110': score += 1
        if score < 1: return None
        return ('震 v1', score)

    # 离 v1
    if mkt_y == '101':
        if stk_d != '000': return None  # 坤触发
        if mkt_d == '101': return None
        if stk_m in {'011', '001', '101'}: return None
        if stk_y == '011': return None
        return ('离 v1', 1)

    # 兑 v1
    if mkt_y == '110':
        if stk_d != '000': return None  # 坤触发
        if mkt_d == '011': return None
        if stk_m in {'001', '011', '101', '111'}: return None
        return ('兑 v1', 1)

    # 乾 v3
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


# ============================================================
# 卖点 (bull 主路, 跨 regime 一致)
# ============================================================
def should_sell_bull(td_buy_to_now, days_held):
    """检查持仓是否触发 bull (第 2 次下穿 89) 或 60 日兜底
    td_buy_to_now: 从买入日到当前的 trend 数组
    """
    if days_held >= MAX_HOLD_DAYS:
        return True, 'timeout'
    if len(td_buy_to_now) < 2:
        return False, None

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


# ============================================================
# 主回测
# ============================================================
def main():
    t0 = time.time()
    print(f'=== 八卦分治资金回测 v1 ===')
    print(f'  底仓: ¥{INIT_CAPITAL:,}')
    print(f'  每日上限: {MAX_PER_DAY} 只 (¥{SLOT_VALUE:,.0f}/只)')
    print(f'  兜底: {MAX_HOLD_DAYS} 日')

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
                        columns=['date', 'code', 'open', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    # 按 code 切片以便快速查找历史价
    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    open_arr = df['open'].to_numpy().astype(np.float64)
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    code_index = {code_arr[code_starts[i]]: (code_starts[i], code_ends[i]) for i in range(len(code_starts))}

    # 每个 code 的 date → row index 的映射
    print(f'  构建索引...')
    code_date_idx = {}
    for i in range(len(code_starts)):
        s, e = code_starts[i], code_ends[i]
        code = code_arr[s]
        code_date_idx[code] = {date_arr[s+j]: s+j for j in range(e-s)}

    # 按日期排序: 每天一个 buy candidate list
    print(f'  按日期分组...')
    df_by_date = df.groupby('date', sort=True)
    all_dates = sorted(df['date'].unique())
    print(f'  日期范围: {all_dates[0]} → {all_dates[-1]} ({len(all_dates)} 个交易日)')

    # ============================================================
    # 主循环
    # ============================================================
    print(f'\n=== 开始回测 ===')

    cash = INIT_CAPITAL
    holdings = {}  # code → {'buy_date', 'buy_idx_global', 'buy_price', 'shares', 'regime', 'score'}
    trades = []    # 完成交易记录
    nav_history = []  # 每日资产净值

    log_every = 250

    for di, today in enumerate(all_dates):
        if today not in mkt_lookup:
            continue
        mkt = mkt_lookup[today]
        mkt_y = mkt['mkt_y']; mkt_d = mkt['mkt_d']; mkt_m = mkt['mkt_m']

        # ============================================================
        # 1. 持仓: 在今日收盘前判断卖出
        # ============================================================
        sells_today = []
        for code, pos in list(holdings.items()):
            if code not in code_date_idx:
                continue
            if today not in code_date_idx[code]:
                continue  # 个股今日停牌或没数据
            today_idx = code_date_idx[code][today]
            buy_idx = pos['buy_idx_global']
            days_held = today_idx - buy_idx

            # 取 buy_idx 到 today_idx 的 trend 数组
            td_seg = trend_arr[buy_idx:today_idx+1]
            sell, reason = should_sell_bull(td_seg, days_held)

            if sell:
                # 今日收盘卖
                sell_price = close_arr[today_idx]
                proceeds = pos['shares'] * sell_price
                cost = pos['shares'] * pos['buy_price']
                profit = proceeds - cost
                ret_pct = (sell_price / pos['buy_price'] - 1) * 100
                cash += proceeds
                trades.append({
                    'code': code,
                    'buy_date': pos['buy_date'],
                    'sell_date': today,
                    'buy_price': pos['buy_price'],
                    'sell_price': sell_price,
                    'shares': pos['shares'],
                    'cost': cost,
                    'proceeds': proceeds,
                    'profit': profit,
                    'ret_pct': ret_pct,
                    'days': days_held,
                    'regime': pos['regime'],
                    'score': pos['score'],
                    'reason': reason,
                })
                sells_today.append(code)
                del holdings[code]

        # ============================================================
        # 2. 扫今日所有买入候选
        # ============================================================
        candidates = []
        if today in df_by_date.groups:
            today_idx_in_df = df_by_date.groups[today]
            for ridx in today_idx_in_df:
                code = code_arr[ridx]
                if code in holdings: continue  # 已持有

                # ret_10d 仅乾 regime 用 (查前 10 行)
                ret_10d = None
                if mkt_y == '111':
                    if code in code_date_idx:
                        cd = code_date_idx[code]
                        if today in cd:
                            ti = cd[today]
                            # 找前 10 个交易日的 close
                            cs, _ = code_index[code]
                            if ti - cs >= 10:
                                ret_10d = (close_arr[ti] / close_arr[ti-10] - 1) * 100

                decide = regime_buy_decide(
                    mkt_y, mkt_d, mkt_m,
                    stk_d_arr[ridx], stk_m_arr[ridx], stk_y_arr[ridx],
                    ret_10d=ret_10d,
                )
                if decide is None: continue
                regime, score = decide

                # 候选: (score, code, ridx, regime)
                candidates.append({
                    'code': code, 'ridx': ridx, 'regime': regime, 'score': score,
                    'close': close_arr[ridx],
                })

        # 按 score 降序, 同分按 code 字母序 (deterministic)
        candidates.sort(key=lambda x: (-x['score'], x['code']))

        # ============================================================
        # 3. 决定今日要买几只 (上限 = max - 持仓)
        # ============================================================
        slots_left = MAX_PER_DAY - len(holdings)
        if slots_left > 0 and len(candidates) > 0:
            # 挑前 slots_left 只
            picks = candidates[:slots_left]
            for cand in picks:
                # 查 t+1 日开盘价
                code = cand['code']
                if code not in code_date_idx: continue
                # 找 today 之后的下一个交易日
                cs, ce = code_index[code]
                today_idx_local = code_date_idx[code].get(today)
                if today_idx_local is None: continue
                if today_idx_local + 1 >= ce: continue  # 没有下一日数据
                buy_idx_global = today_idx_local + 1
                buy_price = open_arr[buy_idx_global]
                if np.isnan(buy_price) or buy_price <= 0: continue

                # 计算可买股数 (1 手 = 100 股, 向下取整)
                shares_avail = int(SLOT_VALUE // buy_price // 100) * 100
                if shares_avail <= 0: continue
                cost = shares_avail * buy_price
                if cost > cash: continue  # 钱不够
                cash -= cost
                holdings[code] = {
                    'buy_date': date_arr[buy_idx_global],
                    'buy_idx_global': buy_idx_global,
                    'buy_price': buy_price,
                    'shares': shares_avail,
                    'regime': cand['regime'],
                    'score': cand['score'],
                }

        # ============================================================
        # 4. 算今日 NAV
        # ============================================================
        market_value = 0.0
        for code, pos in holdings.items():
            if code in code_date_idx and today in code_date_idx[code]:
                ti = code_date_idx[code][today]
                market_value += pos['shares'] * close_arr[ti]
            else:
                # 用买入价兜底
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

    # ============================================================
    # 收尾: 强制平所有仓 (按最后日收盘)
    # ============================================================
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
    # 输出报告
    # ============================================================
    print(f'\n=== 回测结果 ===')
    df_trades = pd.DataFrame(trades)
    df_nav = pd.DataFrame(nav_history)

    final_nav = df_nav['nav'].iloc[-1]
    total_return = (final_nav / INIT_CAPITAL - 1) * 100
    days = (pd.to_datetime(df_nav['date'].iloc[-1]) - pd.to_datetime(df_nav['date'].iloc[0])).days
    annual_return = ((final_nav / INIT_CAPITAL) ** (365/days) - 1) * 100 if days > 0 else 0

    # MDD
    df_nav['nav_peak'] = df_nav['nav'].cummax()
    df_nav['drawdown'] = (df_nav['nav'] - df_nav['nav_peak']) / df_nav['nav_peak'] * 100
    mdd = df_nav['drawdown'].min()

    # 胜率
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

    # 按 regime 分解
    if len(df_trades) > 0:
        print(f'\n  按 regime 分解:')
        print(f'  {"regime":<10} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"总盈亏":>11} {"均持仓":>7}')
        for r in df_trades['regime'].unique():
            sub = df_trades[df_trades['regime'] == r]
            print(f'  {r:<10} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} ¥{sub["profit"].sum():>10,.0f} '
                  f'{sub["days"].mean():>6.1f}')

        # 按年分解
        print(f'\n  按年分解:')
        df_trades['year'] = pd.to_datetime(df_trades['buy_date']).dt.year
        print(f'  {"year":<6} {"笔数":>6} {"胜率%":>7} {"均收益%":>9} {"总盈亏":>11}')
        for y in sorted(df_trades['year'].unique()):
            sub = df_trades[df_trades['year'] == y]
            print(f'  {y:<6} {len(sub):>6} {(sub["ret_pct"]>0).mean()*100:>6.1f} '
                  f'{sub["ret_pct"].mean():>+8.2f} ¥{sub["profit"].sum():>10,.0f}')

    # 持仓数分布
    print(f'\n  持仓数分布 (NAV 历史):')
    pc = df_nav['pos_count'].value_counts().sort_index()
    print(f'  {"持仓":<6} {"日数":>6} {"占比%":>7}')
    for k, v in pc.items():
        print(f'  {k:<6} {v:>6} {v/len(df_nav)*100:>6.1f}')

    # 输出 NAV 时间序列
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_nav.to_csv(os.path.join(out_dir, 'capital_nav_v1.csv'), index=False, encoding='utf-8-sig')
    df_trades.to_csv(os.path.join(out_dir, 'capital_trades_v1.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  写出: data_layer/data/results/capital_{{nav,trades}}_v1.csv')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
