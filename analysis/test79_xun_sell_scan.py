# -*- coding: utf-8 -*-
"""阶段 6: 巽 regime 卖点扫描

入场:
  Gate 1: 大盘 y_gua = 011 巽
  Gate 2: 个股 d_gua = 011 巽
  Gate 3: 强避雷 - 大盘 d_gua = 111 乾 跳过
  Gate 4: 必须命中 大盘 d_gua = 000 坤 (唯一通过的好规律)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
QIAN_RUN = 10


def calc_sells(td_seg, gua_seg, stk_m_seg, max_n):
    n = max_n
    results = {}
    for win in [15, 20, 30, 45, 60]:
        results[f'fix{win}'] = min(win, n - 1)
    s4 = n - 1
    running_max = td_seg[0]
    for k in range(1, n):
        if not np.isnan(td_seg[k]):
            running_max = max(running_max, td_seg[k])
        if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
            s4 = k; break
    results['cross1'] = s4

    s5 = n - 1; cnt = 0; running_max = td_seg[0]
    for k in range(1, n):
        if not np.isnan(td_seg[k]):
            running_max = max(running_max, td_seg[k])
        if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
            cnt += 1
            if cnt == 2:
                s5 = k; break
    results['bull'] = s5

    for ts, name in [(15, 'bull_ts15'), (20, 'bull_ts20')]:
        sx = n - 1; cnt = 0; running_max = td_seg[0]
        for k in range(1, n):
            if not np.isnan(td_seg[k]):
                running_max = max(running_max, td_seg[k])
            if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
                cnt += 1
                if cnt == 2: sx = k; break
            if k >= ts:
                seg_max = np.nanmax(td_seg[:k+1])
                if seg_max < 89: sx = k; break
        results[name] = sx

    s_md = n - 1
    for k in range(n):
        if stk_m_seg[k] == '110': s_md = k; break
    results['m_dui'] = s_md

    # td80→m_dui
    s_sm = n - 1; in_state = False
    for k in range(1, n):
        if not in_state:
            if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                in_state = True
            continue
        if stk_m_seg[k] == '110':
            s_sm = k; break
    results['td80_mdui'] = s_sm

    s_m1 = n - 1; in_qian = False
    for k in range(0, n):
        if gua_seg[k] == '111': in_qian = True
        elif in_qian and gua_seg[k] != '111':
            s_m1 = k; break
    results['m1'] = s_m1

    return results


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
                              columns=['date', 'd_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 入场扫描 (巽 + 巽日 + 大d=坤 必中, 大d≠乾 避雷) ===')
    entries = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != '011': continue  # 巽 regime
            if stk_d_arr[gi] != '011': continue  # 巽日
            if mkt_d_arr[gi] == '111': continue  # 避雷大d=乾
            if mkt_d_arr[gi] != '000': continue  # 必须大d=坤 (唯一好规律)

            buy = i; end = i + MAX_HOLD + 1
            if s + end > e: continue
            td_seg = trend_arr[s+buy:s+end]
            gua_seg = stk_d_arr[s+buy:s+end]
            stk_m_seg = stk_m_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]

            sells = calc_sells(td_seg, gua_seg, stk_m_seg, MAX_HOLD + 1)

            seg30 = stk_d_arr[s+buy:s+min(buy+30, e-s)]
            n_qian = int((seg30 == '111').sum())

            entry = {'date': date_arr[gi], 'code': code_arr[gi],
                     'is_zsl': n_qian >= QIAN_RUN}
            for name, k in sells.items():
                if k < len(cl_seg):
                    entry[f'r_{name}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                    entry[f'h_{name}'] = k
                else:
                    entry[f'r_{name}'] = float('nan')
                    entry[f'h_{name}'] = float('nan')
            entries.append(entry)

    df_en = pd.DataFrame(entries)
    print(f'  入场: {len(df_en):,}')

    df_en['seg'] = ''
    df_en.loc[(df_en['date'] >= '2019-01-01') & (df_en['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_en.loc[(df_en['date'] >= '2022-01-01') & (df_en['date'] < '2023-01-01'), 'seg'] = 'w5_2022'
    df_en.loc[(df_en['date'] >= '2023-01-01') & (df_en['date'] < '2025-01-01'), 'seg'] = 'w6_2023_24'

    print(f'\n## 卖点对比 (n={len(df_en)})')
    print(f'  {"机制":<14} {"全期望%":>8} {"胜率%":>7} {"持仓":>5} {"主升期望%":>10} {"假期望%":>9} {"w2":>7} {"w5":>7} {"w6":>7}')
    sells_to_show = ['fix15', 'fix20', 'fix30', 'fix45', 'fix60',
                      'cross1', 'bull', 'bull_ts15', 'bull_ts20',
                      'm_dui', 'td80_mdui', 'm1']
    for sn in sells_to_show:
        rcol = f'r_{sn}'; hcol = f'h_{sn}'
        sub = df_en.dropna(subset=[rcol])
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        rets = []
        for sg in ['w2_2019', 'w5_2022', 'w6_2023_24']:
            sg_sub = sub[sub['seg'] == sg]
            r = sg_sub[rcol].mean() if len(sg_sub) > 30 else float('nan')
            rets.append(r)
        print(f'  {sn:<14} {ret:>+7.2f} {win:>6.1f} {hold:>5.1f} {zsl:>+9.2f} {fake:>+8.2f} {rets[0]:>+6.2f} {rets[1]:>+6.2f} {rets[2]:>+6.2f}')


if __name__ == '__main__':
    main()
