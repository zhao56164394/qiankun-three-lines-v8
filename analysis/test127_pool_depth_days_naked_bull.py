# -*- coding: utf-8 -*-
"""池深池天裸跑 — 入池 + 买卖点, 无其他过滤

裸跑组件:
  入池: retail < -250 (新)
  出池: retail >= 0 (新, 修正 v1 bug)
  买点: 个股 d_gua = 011 巽日触发
  卖点: bull_2nd + TS20 + 60d 兜底 (用实际卖出 ret, 不是固定 30 日)

排除: 大盘年/月/日卦, 个股月/年卦, score, 避雷 (这些都进第 3-4 步)

输出:
  1. 全集 baseline (跨所有 mkt_y)
  2. 池深 / 池天 分箱: 实际 ret / 胜率 / 平均持仓
  3. 池深 × 池天 交叉
  4. 跨 5 段稳定性 (池天分档)
  5. 跨 5 段稳定性 (池深分档)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
TRIGGER_GUA = '011'
POOL_THR = -250
POOL_EXIT_RETAIL = 0


def simulate_sell(buy_idx, td_arr, close_arr, max_end):
    """从 buy_idx 开始, 走 bull_2nd / TS20 / 60d 兜底, 返回 (sell_idx, reason, ret_pct)"""
    buy_price = close_arr[buy_idx]
    cross_count = 0
    running_max = td_arr[buy_idx]

    for k in range(buy_idx + 1, max_end + 1):
        days_held = k - buy_idx
        if not np.isnan(td_arr[k]):
            running_max = max(running_max, td_arr[k])

        # bull_2nd
        if running_max >= 89 and td_arr[k] < 89 and td_arr[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                ret = (close_arr[k] / buy_price - 1) * 100
                return k, 'bull_2nd', ret, days_held

        # ts20
        if days_held >= 20:
            seg = td_arr[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                ret = (close_arr[k] / buy_price - 1) * 100
                return k, 'ts20', ret, days_held

        # 60d 兜底
        if days_held >= MAX_HOLD:
            ret = (close_arr[k] / buy_price - 1) * 100
            return k, 'timeout', ret, days_held

    # 走到末尾
    k = max_end
    ret = (close_arr[k] / buy_price - 1) * 100
    return k, 'force_close', ret, k - buy_idx


def main():
    t0 = time.time()
    print('=== 池深池天裸跑: 入池 + 买点 + 卖点 (无大盘/月年卦/score 过滤) ===\n')

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
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫事件 + 模拟卖出...')
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

            # 触发: 个股 d_gua=011, 不限其他
            if in_pool and stk_d_arr[gi] == TRIGGER_GUA:
                pool_days = i - pool_enter_i
                # 模拟卖出
                max_end = min(s + n - 1, gi + MAX_HOLD)
                sell_idx, reason, ret_pct, days_held = simulate_sell(
                    gi, trend_arr, close_arr, max_end)
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'pool_days': pool_days,
                    'pool_min_retail': pool_min_retail,
                    'ret_pct': ret_pct, 'reason': reason,
                    'days_held': days_held,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')
    base = df_e['ret_pct'].mean()
    base_win = (df_e['ret_pct']>0).mean() * 100
    print(f'  全集: ret={base:+.2f}% / win={base_win:.1f}% / 持仓 avg {df_e["days_held"].mean():.1f}d')
    print(f'  按 reason: {df_e["reason"].value_counts().to_dict()}')

    # ============ 1. 池天分箱 ============
    print('\n=== 池天分箱 ===')
    fine_bins = [0, 3, 6, 9, 15, 30, 60, 120, 365, 9999]
    fine_labels = ['[0,3)', '[3,6)', '[6,9)', '[9,15)', '[15,30)', '[30,60)',
                    '[60,120)', '[120,365)', '[365+)']
    df_e['db'] = pd.cut(df_e['pool_days'], bins=fine_bins, labels=fine_labels, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"持仓":>6} {"lift":>7}')
    for lab in fine_labels:
        sub = df_e[df_e['db'] == lab]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_pct"]>0).mean()*100:>6.1f}% {sub["days_held"].mean():>5.1f}d '
              f'{avg-base:>+6.2f}')

    # ============ 2. 池深分箱 ============
    print('\n=== 池深分箱 (pool_min_retail) ===')
    bins_retail = [-np.inf, -1000, -700, -500, -400, -300, POOL_THR + 0.01]
    labels_retail = ['<-1000', '[-1000,-700)', '[-700,-500)', '[-500,-400)', '[-400,-300)', '[-300,-250)']
    df_e['ddep'] = pd.cut(df_e['pool_min_retail'], bins=bins_retail, labels=labels_retail, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"持仓":>6} {"lift":>7}')
    for lab in labels_retail:
        sub = df_e[df_e['ddep'] == lab]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        print(f'  {lab:<14} {len(sub):>6} {avg:>+8.2f}% '
              f'{(sub["ret_pct"]>0).mean()*100:>6.1f}% {sub["days_held"].mean():>5.1f}d '
              f'{avg-base:>+6.2f}')

    # ============ 3. 池深 × 池天 交叉 ============
    print('\n=== 池深 × 池天 交叉 (avg_ret%) ===')
    pivot = df_e.pivot_table(values='ret_pct', index='ddep', columns='db',
                                aggfunc='mean', observed=True)
    print(pivot.round(2).to_string())

    print('\n=== 池深 × 池天 交叉 (n) ===')
    pivot_n = df_e.pivot_table(values='ret_pct', index='ddep', columns='db',
                                  aggfunc='count', observed=True)
    print(pivot_n.to_string())

    # ============ 4. 跨段稳定性: 池天 ============
    WINDOWS = [
        ('w1_2018',    '2018-01-01', '2019-01-01'),
        ('w2_2019',    '2019-01-01', '2020-01-01'),
        ('w4_2021',    '2021-01-01', '2022-01-01'),
        ('w5_2022',    '2022-01-01', '2023-01-01'),
        ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ]
    print('\n=== 池天 [0,6) 跨段 ===')
    s06 = df_e[df_e['pool_days'] < 6]
    for w in WINDOWS:
        sub = s06[(s06['date'] >= w[1]) & (s06['date'] < w[2])]
        if len(sub):
            print(f'  {w[0]:<12} n={len(sub):>5,} ret={sub["ret_pct"].mean():>+6.2f}% '
                  f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}%')

    print('\n=== 池天 [6,30) 死亡区跨段 ===')
    s_dead = df_e[(df_e['pool_days'] >= 6) & (df_e['pool_days'] < 30)]
    for w in WINDOWS:
        sub = s_dead[(s_dead['date'] >= w[1]) & (s_dead['date'] < w[2])]
        if len(sub):
            print(f'  {w[0]:<12} n={len(sub):>5,} ret={sub["ret_pct"].mean():>+6.2f}% '
                  f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}%')

    print('\n=== 池天 [30,60) 跨段 ===')
    s_mid = df_e[(df_e['pool_days'] >= 30) & (df_e['pool_days'] < 60)]
    for w in WINDOWS:
        sub = s_mid[(s_mid['date'] >= w[1]) & (s_mid['date'] < w[2])]
        if len(sub):
            print(f'  {w[0]:<12} n={len(sub):>5,} ret={sub["ret_pct"].mean():>+6.2f}% '
                  f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}%')

    # ============ 5. 跨段稳定性: 池深 ============
    print('\n=== 池深 < -500 跨段 ===')
    s_deep = df_e[df_e['pool_min_retail'] < -500]
    for w in WINDOWS:
        sub = s_deep[(s_deep['date'] >= w[1]) & (s_deep['date'] < w[2])]
        if len(sub):
            print(f'  {w[0]:<12} n={len(sub):>5,} ret={sub["ret_pct"].mean():>+6.2f}% '
                  f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}%')

    print('\n=== 池深 [-500,-300) 中档跨段 ===')
    s_mid_d = df_e[(df_e['pool_min_retail'] >= -500) & (df_e['pool_min_retail'] < -300)]
    for w in WINDOWS:
        sub = s_mid_d[(s_mid_d['date'] >= w[1]) & (s_mid_d['date'] < w[2])]
        if len(sub):
            print(f'  {w[0]:<12} n={len(sub):>5,} ret={sub["ret_pct"].mean():>+6.2f}% '
                  f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
