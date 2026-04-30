# -*- coding: utf-8 -*-
"""验证用户观察的业务规律:
  1. retail<-250 后, mf 上穿 50 那天是不是真的就是巽日?
  2. 巽日买入后, 用 mf 拐点波段操作 + trend<11 终结, 是否更好?

测试三种交易模式:
  W0: 单次买卖 (现有 baseline) - 巽日买, bull/TS20/60d 卖
  W1: 主力线波段 - mf 上升持, mf 下降卖, 拐点反复
  W2: W1 + trend<11 强制终结整段
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
MAX_SEGMENT = 90  # 整段最长持仓 (从首次买入算)
TRIGGER_GUA = '011'
REGIME_Y = '000'
LOOKBACK = 30


def find_signal_events(arrays):
    """扫所有"retail<-250 上穿期内 + mf 上穿 50 + 巽日"的事件
    并附带后续 90 日的 mf/close/trend 序列, 用于波段卖出"""
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
        prev_below = False
        last_mf = -np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            # 入池: retail 上沿穿透 -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True

            # 检测 mf 上穿 50
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)

            # 触发买点: 池中 + mf 刚上穿 50 + 巽日 + 大盘 y=000
            if (in_pool and mf_cross_up and
                mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA):
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_local': i,
                    'buy_idx_global': gi,
                    'cur_retail': retail[gi],
                    'cur_mf': mf[gi],
                })
                in_pool = False  # 触发后出池, 不重复触发

            # 出池条件: 暂时只在触发后 (后续可加更多)
            last_mf = mf[gi]
            prev_below = cur_below

    return pd.DataFrame(events)


def find_buy_at_cross(arrays):
    """更宽松: 池中 + (mf 上穿 50 OR 巽日) — 看哪个先发生
    实际看用户描述, mf 上穿 50 那天就是巽日的情况频率多少"""
    code_starts = arrays['starts']; code_ends = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d = arrays['stk_d']; mkt_y = arrays['mkt_y']
    date = arrays['date']; code = arrays['code']

    same_day = 0  # mf 上穿当日就是巽日
    diff_day = 0  # mf 上穿后等了几天才到巽日
    miss = 0      # mf 上穿后没到巽日就出池
    gap_days = []

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        waiting_for_sun = False
        wait_start_i = -1

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            if not in_pool and cur_below and not prev_below:
                in_pool = True

            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)

            if in_pool and mf_cross_up:
                # 检测当天是不是巽日 + 大盘 y=000
                if mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                    same_day += 1
                    in_pool = False
                else:
                    waiting_for_sun = True
                    wait_start_i = i

            if waiting_for_sun and i > wait_start_i:
                if mkt_y[gi] == REGIME_Y and stk_d[gi] == TRIGGER_GUA:
                    diff_day += 1
                    gap_days.append(i - wait_start_i)
                    waiting_for_sun = False
                    in_pool = False
                # 退出条件: 等了 30 日还没到巽日
                if i - wait_start_i > 30:
                    miss += 1
                    waiting_for_sun = False
                    in_pool = False

            last_mf = mf[gi]
            prev_below = cur_below

    return same_day, diff_day, miss, gap_days


def simulate_segment_swing(buy_idx, mf_arr, close_arr, trend_arr, max_end):
    """从 buy_idx 开始, 走 mf 拐点波段:
       初始持仓 (1 单位)
       mf 当日 < 前一日 → 卖 (这是"下降")
       卖出后等待: mf 当日 > 前一日 + > 50 → 再买
       trend<11 → 整段终结
       max_end → 强制终结

       返回: 累积收益 (复利) / 段内交易笔数 / 持仓天数 / 终结原因
    """
    buy_price = close_arr[buy_idx]
    cum_ret_mult = 1.0
    n_legs = 0
    in_position = True  # 起始就持仓
    cur_buy_price = buy_price

    for k in range(buy_idx + 1, max_end + 1):
        # 检查 trend<11 整段终结
        if not np.isnan(trend_arr[k]) and trend_arr[k] < 11:
            if in_position:
                cum_ret_mult *= (close_arr[k] / cur_buy_price)
                n_legs += 1
            return cum_ret_mult - 1, n_legs, k - buy_idx, 'trend_below_11'

        # mf 拐点判断
        mf_today = mf_arr[k]
        mf_prev = mf_arr[k-1]
        if np.isnan(mf_today) or np.isnan(mf_prev):
            continue

        if in_position:
            # 持仓状态: mf 下降 → 卖
            if mf_today < mf_prev:
                cum_ret_mult *= (close_arr[k] / cur_buy_price)
                n_legs += 1
                in_position = False
        else:
            # 空仓: mf 上升 + > 50 → 买回
            if mf_today > mf_prev and mf_today > 50:
                cur_buy_price = close_arr[k]
                in_position = True

    # max_end 还在持仓 -> 强平
    if in_position:
        cum_ret_mult *= (close_arr[max_end] / cur_buy_price)
        n_legs += 1
    return cum_ret_mult - 1, n_legs, max_end - buy_idx, 'max_end'


def simulate_w0_baseline(buy_idx, trend_arr, close_arr, max_end):
    """W0: 现有 bull_2nd / TS20 / 60d 卖点"""
    buy_price = close_arr[buy_idx]
    cross_count = 0
    running_max = trend_arr[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days_h = k - buy_idx
        if not np.isnan(trend_arr[k]):
            running_max = max(running_max, trend_arr[k])
        if running_max >= 89 and trend_arr[k] < 89 and trend_arr[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return (close_arr[k]/buy_price-1), 1, days_h, 'bull_2nd'
        if days_h >= 20:
            seg = trend_arr[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return (close_arr[k]/buy_price-1), 1, days_h, 'ts20'
        if days_h >= MAX_HOLD:
            return (close_arr[k]/buy_price-1), 1, days_h, 'timeout'
    return (close_arr[max_end]/buy_price-1), 1, max_end-buy_idx, 'force_close'


def main():
    t0 = time.time()
    print('=== 用户思路验证: mf 上穿 50 + 巽日 + 主力线波段 ===\n')

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

    # ============ 验证 1: mf 上穿 50 当日 = 巽日 频率 ============
    print('\n=== 验证 1: mf 上穿 50 当日是不是巽日 ===\n')
    same_day, diff_day, miss, gap_days = find_buy_at_cross(arrays)
    total = same_day + diff_day + miss
    print(f'  total: {total}')
    print(f'  同日触发 (mf 上穿 50 当日 + 巽日): {same_day} ({same_day/total*100:.1f}%)')
    print(f'  延后触发 (上穿后 30 日内的巽日): {diff_day} ({diff_day/total*100:.1f}%)')
    print(f'  错过 (30 日内无巽日): {miss} ({miss/total*100:.1f}%)')
    if gap_days:
        print(f'  延后天数: avg={np.mean(gap_days):.1f} 中位={np.median(gap_days):.0f} max={max(gap_days)}')

    # ============ 验证 2 + 3: 三种卖出模式对比 ============
    print('\n=== 扫信号 (mf 上穿 50 + 巽日 + 池中) ===\n')
    df_e = find_signal_events(arrays)
    print(f'  事件: {len(df_e):,}')

    # 每个事件用 3 种模式模拟
    rows_w0, rows_w1, rows_w2 = [], [], []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        # 找该股结尾
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        s = code_starts[ci]; e = code_ends[ci]
        max_end_w0 = min(e - 1, gi + MAX_HOLD)
        max_end_w1 = min(e - 1, gi + MAX_SEGMENT)

        ret_w0, legs_w0, days_w0, r_w0 = simulate_w0_baseline(gi, trend_arr, close_arr, max_end_w0)
        ret_w1, legs_w1, days_w1, r_w1 = simulate_segment_swing(
            gi, arrays['mf'], close_arr, np.full_like(trend_arr, np.inf), max_end_w1)  # W1 不含 trend<11
        ret_w2, legs_w2, days_w2, r_w2 = simulate_segment_swing(
            gi, arrays['mf'], close_arr, trend_arr, max_end_w1)  # W2 含 trend<11

        rows_w0.append({'date': ev['date'], 'code': ev['code'],
                          'ret_pct': ret_w0*100, 'legs': legs_w0, 'days': days_w0, 'reason': r_w0})
        rows_w1.append({'date': ev['date'], 'code': ev['code'],
                          'ret_pct': ret_w1*100, 'legs': legs_w1, 'days': days_w1, 'reason': r_w1})
        rows_w2.append({'date': ev['date'], 'code': ev['code'],
                          'ret_pct': ret_w2*100, 'legs': legs_w2, 'days': days_w2, 'reason': r_w2})

    df_w0 = pd.DataFrame(rows_w0)
    df_w1 = pd.DataFrame(rows_w1)
    df_w2 = pd.DataFrame(rows_w2)

    print('\n=== 单事件级对比 (无资金回测, 平均 ret%) ===\n')
    print(f'  {"模式":<24} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>5} {"avg_legs":>8}')
    for label, df_x in [('W0 单次买卖 (bull/TS20/60d)', df_w0),
                          ('W1 mf 拐点波段', df_w1),
                          ('W2 W1 + trend<11 终结', df_w2)]:
        avg = df_x['ret_pct'].mean()
        win = (df_x['ret_pct']>0).mean()*100
        med = df_x['ret_pct'].median()
        days = df_x['days'].mean()
        legs = df_x['legs'].mean()
        print(f'  {label:<24} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}% '
              f'{days:>4.1f}d {legs:>7.1f}')

    # 跨段对比
    print('\n=== 跨段对比 ===\n')
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print(f'  {"段":<12} {"W0 n/ret/win":<22} {"W1 n/ret/win":<22} {"W2 n/ret/win":<22}')
    for w in WINDOWS:
        w0 = df_w0[(df_w0['date'] >= w[1]) & (df_w0['date'] < w[2])]
        w1 = df_w1[(df_w1['date'] >= w[1]) & (df_w1['date'] < w[2])]
        w2 = df_w2[(df_w2['date'] >= w[1]) & (df_w2['date'] < w[2])]
        if len(w0):
            print(f'  {w[0]:<12} '
                  f'{len(w0):>3} {w0["ret_pct"].mean():>+5.2f}% {(w0["ret_pct"]>0).mean()*100:>5.1f}%   '
                  f'{len(w1):>3} {w1["ret_pct"].mean():>+5.2f}% {(w1["ret_pct"]>0).mean()*100:>5.1f}%   '
                  f'{len(w2):>3} {w2["ret_pct"].mean():>+5.2f}% {(w2["ret_pct"]>0).mean()*100:>5.1f}%')

    # 看波段次数分布 (W2)
    print('\n=== W2 波段次数分布 ===')
    legs_dist = df_w2['legs'].value_counts().sort_index()
    for legs, cnt in legs_dist.items():
        sub = df_w2[df_w2['legs'] == legs]
        print(f'  {legs} 次买卖: {cnt} 笔, ret={sub["ret_pct"].mean():+.2f}%, '
              f'win={(sub["ret_pct"]>0).mean()*100:.1f}%')

    print('\n=== W2 终结原因分布 ===')
    print(df_w2['reason'].value_counts())

    # 顺丰例子
    print('\n=== 顺丰 002352 各模式表现 ===')
    sf = df_e[df_e['code'] == '002352']
    sf = sf[sf['date'].between('2016-01-01', '2016-04-01')]
    if len(sf):
        for _, ev in sf.iterrows():
            d = ev['date']
            w0r = df_w0[(df_w0['code']=='002352') & (df_w0['date']==d)]
            w1r = df_w1[(df_w1['code']=='002352') & (df_w1['date']==d)]
            w2r = df_w2[(df_w2['code']=='002352') & (df_w2['date']==d)]
            print(f'  {d} buy_retail={ev["cur_retail"]:.0f} buy_mf={ev["cur_mf"]:.0f}')
            print(f'    W0: ret={w0r["ret_pct"].iloc[0]:+.2f}%, {w0r["days"].iloc[0]}d, {w0r["reason"].iloc[0]}')
            print(f'    W1: ret={w1r["ret_pct"].iloc[0]:+.2f}%, {w1r["days"].iloc[0]}d, {w1r["legs"].iloc[0]}腿, {w1r["reason"].iloc[0]}')
            print(f'    W2: ret={w2r["ret_pct"].iloc[0]:+.2f}%, {w2r["days"].iloc[0]}d, {w2r["legs"].iloc[0]}腿, {w2r["reason"].iloc[0]}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
