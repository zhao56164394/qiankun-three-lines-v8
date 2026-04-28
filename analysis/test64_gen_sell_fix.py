# -*- coding: utf-8 -*-
"""卖点修正: 由于艮主升浪 87% timeout, bull 不合适
重测 cross1 (M3) / fix30 / fix45 的拆段表现
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
QIAN_RUN = 10


def main():
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    rows = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != '001': continue
            if stk_d_arr[gi] != '011': continue
            if stk_m_arr[gi] == '111': continue
            score = 0
            if stk_y_arr[gi] == '101': score += 1
            if gi - 5 >= s:
                sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1]))
                if sanhu_5d < -50: score += 1
                elif sanhu_5d < -30: score += 1
            if score < 2: continue

            buy = i; end = i + MAX_HOLD + 1
            if s + end > e: continue
            td_seg = trend_arr[s+buy:s+end]
            gua_seg = stk_d_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]

            # 各卖点
            sells = {}
            sells['fix15'] = 15
            sells['fix20'] = 20
            sells['fix30'] = 30
            sells['fix45'] = 45
            sells['fix60'] = 60

            # cross1
            s4 = MAX_HOLD; running_max = td_seg[0]
            for k in range(1, len(td_seg)):
                if not np.isnan(td_seg[k]):
                    running_max = max(running_max, td_seg[k])
                if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
                    s4 = k; break
            sells['cross1'] = s4

            # cross1 + fix30 兜底 (取早的)
            sells['cross1_fix30'] = min(s4, 30)
            # cross1 + fix45 兜底
            sells['cross1_fix45'] = min(s4, 45)
            # cross1 + fix20 兜底
            sells['cross1_fix20'] = min(s4, 20)

            n_qian = int((gua_seg[:31] == '111').sum())  # 30 日内乾天数

            row = {
                'date': date_arr[gi],
                'is_zsl': n_qian >= QIAN_RUN,
            }
            for name, k in sells.items():
                row[f'r_{name}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                row[f'h_{name}'] = k
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'

    print(f'\n入场: {len(df_h)}')

    print(f'\n## 候选卖点对比 (含 cross1+fix 组合)')
    print(f'  {"机制":<16} {"全期望":>7} {"胜率":>6} {"持仓":>5} {"主升期望":>9} {"假期望":>8} {"w2期望":>8} {"w4期望":>8}')
    for name in ['fix15', 'fix20', 'fix30', 'fix45', 'fix60', 'cross1',
                  'cross1_fix20', 'cross1_fix30', 'cross1_fix45']:
        rcol = f'r_{name}'; hcol = f'h_{name}'
        sub = df_h.dropna(subset=[rcol])
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        print(f'  {name:<16} {ret:>+6.2f} {win:>5.1f} {hold:>5.1f} {zsl:>+8.2f} {fake:>+7.2f} {ret_w2:>+7.2f} {ret_w4:>+7.2f}')

    # 推荐: 资金周转效率 (期望 / 持仓)
    print(f'\n## 资金周转效率 (期望%/持仓日)')
    print(f'  {"机制":<16} {"期望%":>7} {"持仓":>5} {"日均收益bps":>12}')
    for name in ['fix15', 'fix20', 'fix30', 'fix45', 'fix60', 'cross1',
                  'cross1_fix20', 'cross1_fix30', 'cross1_fix45']:
        rcol = f'r_{name}'; hcol = f'h_{name}'
        sub = df_h.dropna(subset=[rcol])
        ret = sub[rcol].mean()
        hold = sub[hcol].mean()
        bps_per_day = ret / hold * 100  # bps
        print(f'  {name:<16} {ret:>+6.2f} {hold:>5.1f} {bps_per_day:>+11.1f}')


if __name__ == '__main__':
    main()
