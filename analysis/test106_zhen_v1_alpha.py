# -*- coding: utf-8 -*-
"""震 v1 vs 全市场 alpha 检查

A 版: 震 regime + 坎触发 + 3 项弱避雷 (86K, 无 score)
B 版: 上 + score≥1 (大d=巽 OR 股m=兑) (11K)
卖点: bull (跨 regime 一致)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
MAX_HOLD = 60
QIAN_RUN = 10

WINDOWS = [
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
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
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    all_market = []
    a_events = []  # 仅避雷
    b_events = []  # +score≥1

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = e - s
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != '100': continue  # 震

            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            all_market.append({'date': date_arr[gi], 'ret_30': ret_30})

            # 入场: 坎触发 + 弱避雷
            if stk_d_arr[gi] != '010': continue  # 坎
            if mkt_d_arr[gi] in {'101', '111'}: continue
            if stk_y_arr[gi] == '111': continue

            # bull 卖点
            if i + MAX_HOLD + 1 > n: continue
            td_seg = trend_arr[s+i:s+i+MAX_HOLD+1]
            cl_seg = cl[i:i+MAX_HOLD+1]

            sell_k = MAX_HOLD; cnt = 0; running_max = td_seg[0]
            for k in range(1, MAX_HOLD + 1):
                if not np.isnan(td_seg[k]):
                    running_max = max(running_max, td_seg[k])
                if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
                    cnt += 1
                    if cnt == 2:
                        sell_k = k; break

            ret_bull = (cl_seg[sell_k] / cl_seg[0] - 1) * 100
            entry = {'date': date_arr[gi], 'ret_30': ret_30, 'ret_bull': ret_bull}
            a_events.append(entry)

            # B 版 score≥1
            score = 0
            if mkt_d_arr[gi] == '011': score += 1
            if stk_m_arr[gi] == '110': score += 1
            if score >= 1:
                b_events.append(entry)

    df_all = pd.DataFrame(all_market)
    df_a = pd.DataFrame(a_events)
    df_b = pd.DataFrame(b_events)

    print(f'  全市场 (震 regime 内): {len(df_all):,}')
    print(f'  A 版 (仅避雷): {len(df_a):,}')
    print(f'  B 版 (score≥1): {len(df_b):,}')

    for name, df_v in [('A 版 (仅避雷)', df_a), ('B 版 (score≥1)', df_b)]:
        print(f'\n## {name} vs 全市场 alpha (30 日)')
        print(f'  {"窗口":<14} {"全市场n":>8} {"全市场":>7} {"v n":>7} {"v 30日":>8} {"alpha 30":>9}  {"v bull":>8} {"alpha bull":>10}')
        for w_name, lo, hi in WINDOWS:
            df_all_seg = df_all[(df_all['date'] >= lo) & (df_all['date'] < hi)]
            df_v_seg = df_v[(df_v['date'] >= lo) & (df_v['date'] < hi)] if len(df_v) > 0 else pd.DataFrame()
            if len(df_all_seg) < 100: continue
            all_ret = df_all_seg['ret_30'].mean()
            v_30 = df_v_seg['ret_30'].mean() if len(df_v_seg) > 50 else float('nan')
            v_bull = df_v_seg['ret_bull'].mean() if len(df_v_seg) > 50 else float('nan')
            alpha_30 = v_30 - all_ret if not np.isnan(v_30) else float('nan')
            alpha_bull = v_bull - all_ret if not np.isnan(v_bull) else float('nan')
            print(f'  {w_name:<14} {len(df_all_seg):>8,} {all_ret:>+6.2f} {len(df_v_seg):>7,} '
                  f'{v_30:>+7.2f} {alpha_30:>+8.2f}  {v_bull:>+7.2f} {alpha_bull:>+9.2f}')

        print(f'\n  全样本:')
        print(f'  全市场: {df_all["ret_30"].mean():+.2f}%')
        print(f'  v (30 日): {df_v["ret_30"].mean():+.2f}%')
        print(f'  v (bull 卖): {df_v["ret_bull"].mean():+.2f}%')
        print(f'  alpha 30 日: {df_v["ret_30"].mean() - df_all["ret_30"].mean():+.2f}%')
        print(f'  alpha bull (vs 全市场 30 日): {df_v["ret_bull"].mean() - df_all["ret_30"].mean():+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
