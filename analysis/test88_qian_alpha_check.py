# -*- coding: utf-8 -*-
"""验证乾 regime 真相: 时间长 + 大盘涨, 策略却只 +4.89%, 哪里出问题?

对比维度:
  1. 大盘指数 (000001.SH) 在每段 30 日均涨幅 (基准1)
  2. 全市场所有股票 30 日均涨幅 (基准2 — 等权)
  3. 巽日 baseline 30 日均涨幅 (我们用的)
  4. v3 策略 (避雷+score) 30 日均涨幅

如果"大盘涨 +5%, 全市场等权 +3%, 我们 +4%", 那说明我们勉强 = 全市场
如果"大盘涨 +8%, 全市场等权 +5%, 我们 +5%", 那其实 = 全市场, alpha 很弱
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '111'

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
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
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
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # ===== 1. 全市场每天等权 30 日涨幅 (基准 — 在乾 regime 内) =====
    print(f'\n=== 全市场等权 30 日涨幅 (在乾 regime 巽日入场对比) ===')
    all_market = []
    qian_xun_baseline = []
    qian_xun_v3 = []

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue  # 必须乾 regime

            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100

            # 1. 全市场 (任何卦)
            all_market.append({'date': date_arr[gi], 'ret_30': ret_30})

            # 2. 巽日 baseline
            if stk_d_arr[gi] == '011':
                qian_xun_baseline.append({'date': date_arr[gi], 'ret_30': ret_30})

                # 3. v3 (避雷+score)
                if mkt_d_arr[gi] in {'100', '101', '110'}: continue
                if mkt_m_arr[gi] == '101': continue
                if stk_m_arr[gi] in {'100', '101'}: continue
                if i - 10 >= 0:
                    ret_10d = (cl[i] / cl[i-10] - 1) * 100
                    if ret_10d > 15: continue
                score = 0
                if stk_m_arr[gi] == '010': score += 1
                if stk_y_arr[gi] == '010': score += 1
                if score < 1: continue

                qian_xun_v3.append({'date': date_arr[gi], 'ret_30': ret_30})

    df_all = pd.DataFrame(all_market)
    df_xun = pd.DataFrame(qian_xun_baseline)
    df_v3 = pd.DataFrame(qian_xun_v3)

    for w_name, lo, hi in WINDOWS:
        df_all_seg = df_all[(df_all['date'] >= lo) & (df_all['date'] < hi)]
        df_xun_seg = df_xun[(df_xun['date'] >= lo) & (df_xun['date'] < hi)]
        df_v3_seg = df_v3[(df_v3['date'] >= lo) & (df_v3['date'] < hi)]
        if len(df_all_seg) < 100: continue
        all_ret = df_all_seg['ret_30'].mean()
        xun_ret = df_xun_seg['ret_30'].mean() if len(df_xun_seg) > 0 else float('nan')
        v3_ret = df_v3_seg['ret_30'].mean() if len(df_v3_seg) > 0 else float('nan')
        # alpha
        alpha_xun = xun_ret - all_ret
        alpha_v3 = v3_ret - all_ret
        print(f'  {w_name:<12} 全市场 n={len(df_all_seg):>7,} 30日 {all_ret:>+5.2f}%  '
              f'巽日 {xun_ret:>+5.2f}% (alpha {alpha_xun:>+5.2f})  '
              f'v3 n={len(df_v3_seg):>5,} {v3_ret:>+5.2f}% (alpha {alpha_v3:>+5.2f})')

    print(f'\n  全样本 全市场: {df_all["ret_30"].mean():+.2f}%')
    print(f'  全样本 巽日:   {df_xun["ret_30"].mean():+.2f}%')
    print(f'  全样本 v3:     {df_v3["ret_30"].mean():+.2f}%')
    print(f'  v3 vs 全市场 alpha: {df_v3["ret_30"].mean() - df_all["ret_30"].mean():+.2f}%')


if __name__ == '__main__':
    main()
