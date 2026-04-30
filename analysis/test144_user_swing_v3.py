# -*- coding: utf-8 -*-
"""用户波段策略 v3 — 双降卖、拐点+趋势/主力验证再买、仓位守恒

入场 (建仓):
  retail<-250 池中 + mf 上穿 50 + retail 上升 → 100% 满仓建仓

卖出 (条件: 主散双降都 >0 且降幅显著):
  mf > 0 AND retail > 0  (双线都在阳线上方)
  AND mf 显著下降 AND retail 显著下降 (定义"显著": 至少 N 点)
  → 全部卖出, 记录卖出价

再买入 (拐点 + 趋势/主力确认):
  上次已卖出
  AND mf 出现拐点上升 (mf_today > mf_prev AND mf_prev <= mf_prev_prev)
  AND mf > 50 (主力线在强位)
  AND trend > 11 (趋势线没破)
  → 用与上次卖出相同仓位重新买入

清仓 (整段终结):
  trend < 11 (趋势破)
  OR 连续 2 个涨停 (短打顶)
  OR 60 日 timeout

测多个 mf 显著下降阈值, 看哪个最优:
  V1: mf 下降 >= 50, retail 下降 >= 50
  V2: mf 下降 >= 100, retail 下降 >= 50
  V3: 5d mf 累计下降 >= 100, 5d retail 累计下降 >= 50
  V4: 5d mf 累计下降 >= 200, 5d retail 累计下降 >= 100
  V5: mf 上一日为正且当日转负 (穿 0)
  V6: 比较 baseline (单次买卖 bull_2nd)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
LOOKBACK = 30


def find_signals(arrays):
    """E1+E2+E3 入场"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']

    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            if not in_pool and cur_below and not prev_below:
                in_pool = True

            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)

            if in_pool and mf_cross_up and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                    'cur_retail': retail[gi],
                    'cur_mf': mf[gi],
                })
                in_pool = False

            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below

    return pd.DataFrame(events)


def simulate_swing(buy_idx, td, close, gua, mf, retail, max_end,
                    mf_drop_thr, retail_drop_thr,
                    use_5d=False):
    """带波段的模拟
       cum_mult: 累积收益乘数 (持仓期跟价格走)
       记录每次买卖

       简化: 仓位状态 holding (1.0 满仓) / cash (0.0)
       buy_price: 当前买入价
    """
    bp_first = close[buy_idx]
    holding = True
    cur_buy_price = bp_first
    cum_mult = 1.0  # 整体累积收益倍数
    legs = 0  # 交易腿数
    end_reason = None
    end_idx = None

    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx

        # 整段终结优先级最高
        # 1. trend < 11
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= (close[k] / cur_buy_price)
                legs += 1
            end_reason = 'td_below_11'
            end_idx = k
            break

        # 2. 连 2 涨停
        if k >= buy_idx + 2:
            if (not np.isnan(close[k-1]) and close[k-1] > 0 and
                not np.isnan(close[k-2]) and close[k-2] > 0):
                chg1 = (close[k]/close[k-1] - 1) * 100
                chg2 = (close[k-1]/close[k-2] - 1) * 100
                if chg1 >= 9.7 and chg2 >= 9.7:
                    if holding:
                        cum_mult *= (close[k] / cur_buy_price)
                        legs += 1
                    end_reason = '2_limit_up'
                    end_idx = k
                    break

        # 3. 60d timeout
        if days >= MAX_HOLD:
            if holding:
                cum_mult *= (close[k] / cur_buy_price)
                legs += 1
            end_reason = 'timeout'
            end_idx = k
            break

        # 双降检测 (持仓时)
        if holding and k >= 1:
            if use_5d and k >= buy_idx + 5:
                # 5 日累计降幅
                mf_drop = mf[k-4] - mf[k]  # 5 日内的下降
                retail_drop = retail[k-4] - retail[k]
            else:
                mf_drop = mf[k-1] - mf[k] if not np.isnan(mf[k-1]) else 0
                retail_drop = retail[k-1] - retail[k] if not np.isnan(retail[k-1]) else 0

            # 双线都 > 0 且降幅显著
            if (not np.isnan(mf[k]) and not np.isnan(retail[k]) and
                mf[k] > 0 and retail[k] > 0 and
                mf_drop >= mf_drop_thr and retail_drop >= retail_drop_thr):
                # 卖出
                cum_mult *= (close[k] / cur_buy_price)
                legs += 1
                holding = False
                continue

        # 再买入检测 (空仓时)
        if not holding and k >= 2:
            # mf 拐点上升: mf[k] > mf[k-1] AND mf[k-1] <= mf[k-2]
            if (not np.isnan(mf[k]) and not np.isnan(mf[k-1]) and not np.isnan(mf[k-2])
                and mf[k] > mf[k-1] and mf[k-1] <= mf[k-2]
                and mf[k] > 50  # 主力线强位
                and not np.isnan(td[k]) and td[k] > 11):  # 趋势线没破
                # 再买入
                cur_buy_price = close[k]
                holding = True
                continue

    if end_idx is None:
        # 走完未触发终结
        end_idx = max_end
        if holding:
            cum_mult *= (close[max_end] / cur_buy_price)
            legs += 1
        end_reason = 'fc'

    final_ret = (cum_mult - 1) * 100
    return end_idx, end_reason, final_ret, legs


def sell_baseline_bull2(buy_idx, td, close, max_end):
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close[k]/bp-1)*100, 1
        if days >= 20:
            seg = td[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return k, 'ts20', (close[k]/bp-1)*100, 1
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100, 1
    return max_end, 'fc', (close[max_end]/bp-1)*100, 1


def main():
    t0 = time.time()
    print('=== 用户波段 v3: 双降卖+拐点验证再买+仓位守恒 ===\n')

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

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
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
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    gua_arr = df['stk_d'].to_numpy()
    mf_arr = arrays['mf']
    retail_arr = arrays['retail']

    df_e = find_signals(arrays)
    print(f'  入场信号: {len(df_e):,}')

    # 多种参数版本
    schemes = [
        ('V1 双降日比较 mf>=50 ret>=50',     50, 50, False),
        ('V2 双降日比较 mf>=100 ret>=50',   100, 50, False),
        ('V3 双降日比较 mf>=150 ret>=80',   150, 80, False),
        ('V4 5d 累计 mf>=100 ret>=50',     100, 50, True),
        ('V5 5d 累计 mf>=200 ret>=100',    200, 100, True),
        ('V6 5d 累计 mf>=300 ret>=150',    300, 150, True),
    ]

    print('\n=== 各参数下波段表现 ===\n')
    print(f'  {"方案":<32} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>5} {"avg_legs":>8} {"max":>7} {"min":>7}')

    rows_all = {}
    for label, mf_thr, ret_thr, use_5d in schemes:
        rows = []
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e = code_ends[ci]
            max_end = min(e - 1, gi + MAX_HOLD)
            si, r, ret, legs = simulate_swing(gi, trend_arr, close_arr, gua_arr,
                                                  mf_arr, retail_arr, max_end,
                                                  mf_thr, ret_thr, use_5d)
            rows.append({'date': ev['date'], 'code': ev['code'],
                          'days': si - gi, 'reason': r, 'ret_pct': ret, 'legs': legs})
        df_x = pd.DataFrame(rows)
        rows_all[label] = df_x
        avg = df_x['ret_pct'].mean()
        win = (df_x['ret_pct']>0).mean()*100
        med = df_x['ret_pct'].median()
        days = df_x['days'].mean()
        legs = df_x['legs'].mean()
        mx = df_x['ret_pct'].max()
        mn = df_x['ret_pct'].min()
        print(f'  {label:<32} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}% {days:>4.1f}d '
              f'{legs:>7.1f} {mx:>+6.1f}% {mn:>+6.1f}%')

    # baseline 对比
    rows_b = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)
        si, r, ret, legs = sell_baseline_bull2(gi, trend_arr, close_arr, max_end)
        rows_b.append({'date': ev['date'], 'code': ev['code'],
                        'days': si - gi, 'reason': r, 'ret_pct': ret, 'legs': legs})
    df_b = pd.DataFrame(rows_b)
    avg = df_b['ret_pct'].mean()
    win = (df_b['ret_pct']>0).mean()*100
    print(f'  {"baseline bull_2nd (单次)":<32} {avg:>+8.2f}% {win:>6.1f}% '
          f'{df_b["ret_pct"].median():>+7.2f}% {df_b["days"].mean():>4.1f}d {1.0:>7.1f} '
          f'{df_b["ret_pct"].max():>+6.1f}% {df_b["ret_pct"].min():>+6.1f}%')

    # 神火 vs 顺丰
    print('\n=== 神火 vs 顺丰 各方案 ===\n')
    for code, dt in [('000933', '2016-02-17'), ('002352', '2016-01-19')]:
        print(f'  {code} {dt}:')
        for label, df_x in rows_all.items():
            sub = df_x[(df_x['code'] == code) & (df_x['date'] == dt)]
            if len(sub):
                r = sub.iloc[0]
                print(f'    {label:<30}: {r["ret_pct"]:>+7.2f}% / {r["days"]:>3}d / '
                      f'{r["legs"]} 腿 / {r["reason"]}')
        sub = df_b[(df_b['code'] == code) & (df_b['date'] == dt)]
        if len(sub):
            r = sub.iloc[0]
            print(f'    baseline bull_2nd            : {r["ret_pct"]:>+7.2f}% / {r["days"]:>3}d / 1 腿 / {r["reason"]}')

    # 跨段
    WINDOWS = [
        ('w1_2018', '2018-01-01', '2019-01-01'),
        ('w2_2019', '2019-01-01', '2020-01-01'),
        ('w4_2021', '2021-01-01', '2022-01-01'),
        ('w5_2022', '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print('\n=== 跨段 avg_ret ===\n')
    print(f'  {"段":<12}', end='')
    for label in [s[0] for s in schemes]:
        print(f' {label[:10]:>10}', end='')
    print(f' {"baseline":>10}')
    for w in WINDOWS:
        print(f'  {w[0]:<12}', end='')
        for label in [s[0] for s in schemes]:
            df_x = rows_all[label]
            sub = df_x[(df_x['date'] >= w[1]) & (df_x['date'] < w[2])]
            if len(sub):
                print(f' {sub["ret_pct"].mean():>+9.2f}%', end='')
            else:
                print(f' {"-":>10}', end='')
        sub_b = df_b[(df_b['date'] >= w[1]) & (df_b['date'] < w[2])]
        if len(sub_b):
            print(f' {sub_b["ret_pct"].mean():>+9.2f}%')
        else:
            print(f' {"-":>10}')

    # 最优方案 reason 分布
    best_label = max(rows_all, key=lambda k: rows_all[k]['ret_pct'].mean())
    print(f'\n=== 最优 ({best_label}) reason 分布 ===')
    df_best = rows_all[best_label]
    for r, cnt in df_best['reason'].value_counts().items():
        sub = df_best[df_best['reason'] == r]
        print(f'  {r:<18} n={cnt:>4} ret={sub["ret_pct"].mean():>+5.2f}% '
              f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}% legs={sub["legs"].mean():.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
