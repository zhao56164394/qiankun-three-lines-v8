# -*- coding: utf-8 -*-
"""乾 v3 完整版: 加新避雷"前 10 日涨幅 > 15%" 重测

入场:
  Gate 1: 大盘 y_gua = 011 乾
  Gate 2: 个股 d_gua = 011 巽
  Gate 3: 6 项卦象避雷 (test81 发现)
  Gate 4: 新避雷"前 10 日涨幅 > 15%" (test86 发现)
  Gate 5: 软排名 score ≥ 1 (个股 m_gua/y_gua = 010 坎)

卖点: bull (跨 regime 主流)
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

    # 4 个版本: A 基础(只 regime+巽), B 加 6 卦避雷, C+10d涨幅<15%, D 完整 v3
    def run_version(version_name, check_avoid_gua, check_avoid_runup, check_score):
        entries = []
        for ci in range(len(code_starts)):
            s = code_starts[ci]; e = code_ends[ci]
            if e - s < LOOKBACK + MAX_HOLD + 5: continue
            for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
                gi = s + i
                if mkt_y_arr[gi] != '111': continue
                if stk_d_arr[gi] != '011': continue

                if check_avoid_gua:
                    if mkt_d_arr[gi] in {'100', '101', '110'}: continue
                    if mkt_m_arr[gi] == '101': continue
                    if stk_m_arr[gi] in {'100', '101'}: continue

                if check_avoid_runup:
                    # 前 10 日涨幅 > 15%
                    if i - 10 >= 0:
                        ret_10d = (close_arr[s+i] / close_arr[s+i-10] - 1) * 100
                        if ret_10d > 15: continue

                if check_score:
                    score = 0
                    if stk_m_arr[gi] == '010': score += 1
                    if stk_y_arr[gi] == '010': score += 1
                    if score < 1: continue

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
                entries.append(entry)

        df_en = pd.DataFrame(entries)
        df_en['seg'] = ''
        for w_name, (a, b) in [('w2', ('2019-01-01', '2020-01-01')),
                               ('w3', ('2020-01-01', '2021-01-01')),
                               ('w4', ('2021-01-01', '2022-01-01')),
                               ('w5', ('2022-01-01', '2023-01-01')),
                               ('w6', ('2023-01-01', '2025-01-01')),
                               ('w7', ('2025-01-01', '2026-04-21'))]:
            df_en.loc[(df_en['date'] >= a) & (df_en['date'] < b), 'seg'] = w_name

        print(f'\n## {version_name} (n={len(df_en):,})')
        for sn in ['fix30', 'bull', 'bull_ts20']:
            rcol = f'r_{sn}'; hcol = f'h_{sn}'
            sub = df_en.dropna(subset=[rcol])
            ret = sub[rcol].mean()
            win = (sub[rcol] > 0).mean() * 100
            hold = sub[hcol].mean()
            zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
            fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
            rets = []
            for sg in ['w2', 'w3', 'w4', 'w5', 'w6', 'w7']:
                sg_sub = sub[sub['seg'] == sg]
                r = sg_sub[rcol].mean() if len(sg_sub) > 50 else float('nan')
                rets.append(r)
            n_pos = sum(1 for r in rets if not np.isnan(r) and r > 0)
            n_seg = sum(1 for r in rets if not np.isnan(r))
            seg_str = ' '.join(f'{r:>+5.1f}' if not np.isnan(r) else '   --' for r in rets)
            print(f'  {sn:<14} {ret:>+6.2f} 胜 {win:>5.1f} 持仓 {hold:>4.1f} 主升 {zsl:>+6.2f} 假 {fake:>+6.2f} 段{n_pos}/{n_seg}  {seg_str}')

    print('=== 乾 regime v3 多版本对比 ===')
    print('版本 A: 仅 乾 regime + 巽日 (baseline)')
    run_version('A 基础', False, False, False)

    print('\n版本 B: A + 6 卦象避雷')
    run_version('B 6卦避雷', True, False, False)

    print('\n版本 C: B + 前10日涨幅<=15% 避雷')
    run_version('C 加涨幅避雷', True, True, False)

    print('\n版本 D: C + score ≥ 1 (个股 m/y=坎)')
    run_version('D 完整v3', True, True, True)


if __name__ == '__main__':
    main()
