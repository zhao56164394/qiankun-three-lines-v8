# -*- coding: utf-8 -*-
"""兑 v1 vs 全市场 alpha 检查

入场: 兑 regime + 坤触发 + 5 卦避雷
卖点: bull (第 2 次下穿 89, 跨 regime 一致)

对比: 同窗口 + 同 regime 内, 全市场任意 30 日均涨幅 vs v1 期望
看 alpha 是否真在 w2 / w5 都正
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
    ('w2_2019', '2019-01-01', '2020-01-01'),
    ('w5_2022', '2022-01-01', '2023-01-01'),
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

    # 1. 全市场 30 日 (在兑 regime 内, 任何卦)
    all_market = []
    # 2. 兑 v1 (5 关卡入场, bull 卖点)
    v1_events = []

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = e - s
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != '110': continue  # 兑

            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            all_market.append({'date': date_arr[gi], 'ret_30': ret_30})

            # v1 入场
            if stk_d_arr[gi] != '000': continue
            if mkt_d_arr[gi] == '011': continue
            if stk_m_arr[gi] in {'001', '011', '101', '111'}: continue

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
            v1_events.append({'date': date_arr[gi], 'ret_30': ret_30, 'ret_bull': ret_bull})

    df_all = pd.DataFrame(all_market)
    df_v1 = pd.DataFrame(v1_events)

    print(f'  全市场 (兑 regime 内): {len(df_all):,}')
    print(f'  v1 事件: {len(df_v1):,}')

    print(f'\n## v1 vs 全市场 alpha (30 日)')
    print(f'  {"窗口":<12} {"全市场n":>8} {"全市场":>7} {"v1 n":>6} {"v1 30日":>8} {"alpha 30":>8}  {"v1 bull":>8}')
    for w_name, lo, hi in WINDOWS:
        df_all_seg = df_all[(df_all['date'] >= lo) & (df_all['date'] < hi)]
        df_v1_seg = df_v1[(df_v1['date'] >= lo) & (df_v1['date'] < hi)]
        if len(df_all_seg) < 100: continue
        all_ret = df_all_seg['ret_30'].mean()
        v1_30 = df_v1_seg['ret_30'].mean() if len(df_v1_seg) > 0 else float('nan')
        v1_bull = df_v1_seg['ret_bull'].mean() if len(df_v1_seg) > 0 else float('nan')
        alpha_30 = v1_30 - all_ret
        print(f'  {w_name:<12} {len(df_all_seg):>8,} {all_ret:>+6.2f} {len(df_v1_seg):>6,} '
              f'{v1_30:>+7.2f} {alpha_30:>+8.2f}  {v1_bull:>+7.2f}')

    print(f'\n  全样本:')
    print(f'  全市场: {df_all["ret_30"].mean():+.2f}%')
    print(f'  v1 (30 日): {df_v1["ret_30"].mean():+.2f}%')
    print(f'  v1 (bull 卖): {df_v1["ret_bull"].mean():+.2f}%')
    print(f'  alpha 30 日: {df_v1["ret_30"].mean() - df_all["ret_30"].mean():+.2f}%')
    print(f'  alpha bull (vs 全市场 30 日): {df_v1["ret_bull"].mean() - df_all["ret_30"].mean():+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
