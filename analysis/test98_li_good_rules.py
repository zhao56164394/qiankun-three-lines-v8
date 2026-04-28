# -*- coding: utf-8 -*-
"""阶段 5: 离 regime 好规律 IS/OOS 软排名

入场池: 离 regime + 坤触发 + 强+弱避雷 (5 项)
  强: 大d=101离, 股m=011巽
  弱: 股y=011巽, 股m=001艮, 股m=101离

IS = w2 + w3 + w4 + w5 (4 段, 跨牛熊)
OOS = w6 + w7 (2 段)

判定:
  IS lift ≥ +0.5% AND OOS lift ≥ +0.5% → ★ 真好
  IS lift ≥ +1% AND OOS lift < +0.5%   → ✗ 切片
  IS lift < +0.5% AND OOS lift ≥ +0.5% → ○ OOS-only
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '101'
TRIGGER_GUA = '000'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WIN_IS = [('w2', '2019-01-01', '2020-01-01'),
          ('w3', '2020-01-01', '2021-01-01'),
          ('w4', '2021-01-01', '2022-01-01'),
          ('w5', '2022-01-01', '2023-01-01')]
WIN_OOS = [('w6', '2023-01-01', '2025-01-01'),
           ('w7', '2025-01-01', '2026-04-21')]

MIN_N = 200


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

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 入场池: regime + trigger + avoid
    print(f'\n=== 入场池 (离 regime + 坤触发 + 5 项避雷) ===')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            # 避雷
            if mkt_d_arr[gi] == '101': continue  # 强: 大d=离
            if stk_m_arr[gi] in {'011', '001', '101'}: continue  # 强股m=巽 + 弱股m=艮/离
            if stk_y_arr[gi] == '011': continue  # 弱股y=巽

            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
            })
    df_e = pd.DataFrame(events)
    df_e['seg'] = ''
    for w_name, lo, hi in WIN_IS + WIN_OOS:
        df_e.loc[(df_e['date'] >= lo) & (df_e['date'] < hi), 'seg'] = w_name
    df_e = df_e[df_e['seg'] != ''].copy()

    df_e['period'] = df_e['seg'].apply(lambda s: 'IS' if s in {'w2', 'w3', 'w4', 'w5'} else 'OOS')

    print(f'  事件: {len(df_e):,}')
    is_data = df_e[df_e['period'] == 'IS']; oos_data = df_e[df_e['period'] == 'OOS']
    print(f'  IS (w2-w5): n={len(is_data):,}, 期望 {is_data["ret_30"].mean():+.2f}%')
    print(f'  OOS (w6-w7): n={len(oos_data):,}, 期望 {oos_data["ret_30"].mean():+.2f}%')

    is_base = is_data['ret_30'].mean()
    oos_base = oos_data['ret_30'].mean()

    # 扫候选好规律
    print(f'\n## IS/OOS 好规律扫描')
    print(f'  {"规律":<14} {"n":>7} {"全期望":>7}  {"IS n":>6} {"IS ret":>7} {"IS lift":>8}  {"OOS n":>6} {"OOS ret":>8} {"OOS lift":>9}  {"判定":>6}')
    candidates = []
    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 1000: continue
            label = f'{label_short}={state}{GUA_NAMES[state]}'
            sub_is = sub[sub['period'] == 'IS']
            sub_oos = sub[sub['period'] == 'OOS']
            if len(sub_is) < MIN_N or len(sub_oos) < MIN_N: continue
            is_ret = sub_is['ret_30'].mean()
            oos_ret = sub_oos['ret_30'].mean()
            is_lift = is_ret - is_base
            oos_lift = oos_ret - oos_base

            if is_lift >= 0.5 and oos_lift >= 0.5:
                verdict = '★ 真好'
            elif is_lift >= 1.0 and oos_lift < 0.5:
                verdict = '✗ 切片'
            elif is_lift < 0.5 and oos_lift >= 1.0:
                verdict = '○ OOS'
            elif is_lift <= -0.5 and oos_lift <= -0.5:
                verdict = '× 双负'
            else:
                verdict = '— 弱'

            candidates.append((label, len(sub), sub['ret_30'].mean(), len(sub_is), is_ret, is_lift,
                              len(sub_oos), oos_ret, oos_lift, verdict, col, state))

    candidates.sort(key=lambda x: (x[5] + x[8]), reverse=True)
    for c in candidates:
        label, n, full_ret, is_n, is_ret, is_lift, oos_n, oos_ret, oos_lift, verdict, _, _ = c
        print(f'  {label:<14} {n:>7,} {full_ret:>+6.2f}  {is_n:>6,} {is_ret:>+6.2f} {is_lift:>+7.2f}  '
              f'{oos_n:>6,} {oos_ret:>+7.2f} {oos_lift:>+8.2f}  {verdict:>6}')

    # 真好规律 union 效果
    print(f'\n## 真好规律 score=1 / score=2 效果')
    real_good = [(c[10], c[11]) for c in candidates if c[9] == '★ 真好']
    print(f'  真好规律数: {len(real_good)}')
    for col, state in real_good:
        print(f'    {col}={state} ({GUA_NAMES[state]})')

    if real_good:
        score_arr = np.zeros(len(df_e))
        for col, state in real_good:
            score_arr += (df_e[col].values == state).astype(int)
        df_e['score'] = score_arr

        for sc_min in [0, 1, 2]:
            sub = df_e[df_e['score'] >= sc_min]
            sub_is = sub[sub['period'] == 'IS']; sub_oos = sub[sub['period'] == 'OOS']
            print(f'\n  score >= {sc_min}: n={len(sub):,}')
            print(f'    IS: {sub_is["ret_30"].mean():+.2f}% (n={len(sub_is):,})')
            print(f'    OOS: {sub_oos["ret_30"].mean():+.2f}% (n={len(sub_oos):,})')
            print(f'    全: {sub["ret_30"].mean():+.2f}%')
            print(f'    主升率: {(sub["n_qian"]>=QIAN_RUN).mean()*100:.1f}%')


if __name__ == '__main__':
    main()
