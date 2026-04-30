# -*- coding: utf-8 -*-
"""阶段 6: 坤+入池 卖点扫描

入场: 在池 + 坤 regime + 巽日 + 强避雷 + score≥2
  强避雷: 股y=巽, 股m=乾
  score (4 项): 大m=震/大d=巽/大m=坎/股m=坎

测试卖点: bull / bull_TS20 / cross1 / fix15-60 / m_dui / td80_mdui

目标: 选稳定 + 假突破不大亏
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
QIAN_RUN = 10
REGIME_Y = '000'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]


def calc_sells(td_seg, stk_m_seg, max_n):
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

    for ts, name in [(15, 'bull_ts15'), (20, 'bull_ts20'), (30, 'bull_ts30')]:
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

    s_sm = n - 1; in_state = False
    for k in range(1, n):
        if not in_state:
            if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                in_state = True
            continue
        if stk_m_seg[k] == '110':
            s_sm = k; break
    results['td80_mdui'] = s_sm

    return results


def main():
    t0 = time.time()
    print('=== 阶段 6: 坤+入池 卖点扫描 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
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
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)
    print(f'  {len(df):,} 行 (主板)')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    cross_arr = df['cross_below_11'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫入场事件 (在池+坤+巽+避雷+score>=2) ===')
    entries = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                # 强避雷
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue
                # score (4 项)
                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                if score < 2:
                    in_pool = False
                    continue

                buy = i; end = i + MAX_HOLD + 1
                if s + end > e:
                    in_pool = False
                    continue
                td_seg = trend_arr[s+buy:s+end]
                stk_m_seg = stk_m_arr[s+buy:s+end]
                cl_seg = close_arr[s+buy:s+end]

                sells = calc_sells(td_seg, stk_m_seg, MAX_HOLD + 1)

                seg30 = stk_d_arr[s+buy:s+min(buy+30, e-s)]
                n_qian = int((seg30 == '111').sum())

                entry = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN, 'score': score}
                for nm, k in sells.items():
                    if k < len(cl_seg):
                        entry[f'r_{nm}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                        entry[f'h_{nm}'] = k
                entries.append(entry)
                in_pool = False

    df_en = pd.DataFrame(entries)
    df_en['seg'] = ''
    for w_name, lo, hi in WINDOWS:
        df_en.loc[(df_en['date'] >= lo) & (df_en['date'] < hi), 'seg'] = w_name

    print(f'  事件: {len(df_en):,}')

    sells_to_eval = ['fix15', 'fix20', 'fix30', 'fix45', 'fix60',
                     'cross1', 'bull', 'bull_ts15', 'bull_ts20', 'bull_ts30',
                     'm_dui', 'td80_mdui']

    print(f'\n## 卖点对比')
    print(f'  {"卖点":<14} {"全":>7} {"胜":>5} {"持仓":>5} {"主升期":>7} {"假期":>7}  walk-forward(w1..w6)        段稳')
    for sn in sells_to_eval:
        rcol = f'r_{sn}'; hcol = f'h_{sn}'
        sub = df_en.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')

        rets = []
        for w_name, _, _ in WINDOWS:
            sg_short = w_name[:2]
            sg_sub = sub[sub['seg'] == w_name]
            r = sg_sub[rcol].mean() if len(sg_sub) > 50 else float('nan')
            rets.append(r)
        n_pos = sum(1 for r in rets if not np.isnan(r) and r > 0)
        n_seg = sum(1 for r in rets if not np.isnan(r))
        seg_str = ' '.join(f'{r:>+5.1f}' if not np.isnan(r) else '   --' for r in rets)
        print(f'  {sn:<14} {ret:>+6.2f} {win:>5.1f} {hold:>5.1f} {zsl:>+7.2f} {fake:>+6.2f}  {seg_str}  {n_pos}/{n_seg}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
