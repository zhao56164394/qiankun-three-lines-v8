# -*- coding: utf-8 -*-
"""阶段 5: 坤 regime + 入池版 IS/OOS 软排名

入场池: 在池 + 坤 regime + 巽日 + 强避雷 (24,390 事件)
  强避雷: 股y=巽, 股m=乾

IS = w1_2018 + w2_2019 + w4_2021 + w5_2022 (4 段)
OOS = w6_2023_24 (主战场)

判定:
  IS lift ≥ +1% AND OOS lift ≥ +0.5% → ★ 真好
  IS lift ≥ +1% AND OOS lift < +0.5%  → ✗ 切片
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30

REGIME_Y = '000'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WIN_IS = [('w1', '2018-01-01', '2019-01-01'),
          ('w2', '2019-01-01', '2020-01-01'),
          ('w4', '2021-01-01', '2022-01-01'),
          ('w5', '2022-01-01', '2023-01-01')]
WIN_OOS = [('w6', '2023-01-01', '2025-01-01')]

MIN_N = 200


def main():
    t0 = time.time()
    print('=== 阶段 5: 坤+入池 IS/OOS 软排名 ===\n')

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
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)

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
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场池 (在池 + 坤 + 巽 + 强避雷)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                # 强避雷
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue
                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi], 'ret_30': ret_30,
                    'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                    'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
                    'mf': mf_arr[gi], 'mf_5d': mf5_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    df_e['seg'] = ''
    for w_name, lo, hi in WIN_IS + WIN_OOS:
        df_e.loc[(df_e['date'] >= lo) & (df_e['date'] < hi), 'seg'] = w_name
    df_e = df_e[df_e['seg'] != ''].copy()
    df_e['period'] = df_e['seg'].apply(lambda s: 'IS' if s in {'w1', 'w2', 'w4', 'w5'} else 'OOS')

    print(f'  事件: {len(df_e):,}')
    is_data = df_e[df_e['period'] == 'IS']; oos_data = df_e[df_e['period'] == 'OOS']
    print(f'  IS (w1+w2+w4+w5): n={len(is_data):,}, ret {is_data["ret_30"].mean():+.2f}%')
    print(f'  OOS (w6): n={len(oos_data):,}, ret {oos_data["ret_30"].mean():+.2f}%')

    is_base = is_data['ret_30'].mean()
    oos_base = oos_data['ret_30'].mean()

    # 候选
    candidates = []
    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            candidates.append((col, '==', state, f'{label_short}={state}{GUA_NAMES[state]}'))
    for thr in [-200, -100, -50, 50, 100, 200]:
        if thr < 0:
            candidates.append(('mf', '<', thr, f'mf<{thr}'))
            candidates.append(('mf_5d', '<', thr, f'mf_5d<{thr}'))
            candidates.append(('sanhu_5d', '<', thr, f'sanhu_5d<{thr}'))
        else:
            candidates.append(('mf', '>', thr, f'mf>{thr}'))
            candidates.append(('mf_5d', '>', thr, f'mf_5d>{thr}'))

    print(f'\n## IS/OOS 扫描:')
    print(f'  {"规律":<14} {"全 n":>6} {"全 ret":>7}  {"IS n":>6} {"IS lift":>8}  {"OOS n":>6} {"OOS lift":>9}  判定')
    rules = []
    for col, op, thr, label in candidates:
        if op == '==':
            sub = df_e[df_e[col] == thr]
        elif op == '<':
            sub = df_e[df_e[col] < thr]
        elif op == '>':
            sub = df_e[df_e[col] > thr]
        if len(sub) < 1000: continue
        sub_is = sub[sub['period'] == 'IS']
        sub_oos = sub[sub['period'] == 'OOS']
        if len(sub_is) < MIN_N or len(sub_oos) < MIN_N: continue
        is_ret = sub_is['ret_30'].mean()
        oos_ret = sub_oos['ret_30'].mean()
        is_lift = is_ret - is_base
        oos_lift = oos_ret - oos_base

        if is_lift >= 1.0 and oos_lift >= 0.5:
            verdict = '★ 真好'
        elif is_lift >= 1.0 and oos_lift < 0.5:
            verdict = '✗ 切片'
        elif is_lift < 0.5 and oos_lift >= 1.0:
            verdict = '○ OOS'
        elif is_lift <= -0.5 and oos_lift <= -0.5:
            verdict = '× 双负'
        else:
            verdict = '— 弱'

        rules.append((label, len(sub), sub['ret_30'].mean(), len(sub_is), is_ret, is_lift,
                      len(sub_oos), oos_ret, oos_lift, verdict, col, op, thr))

    rules.sort(key=lambda x: (x[5] + x[8]), reverse=True)
    for r in rules:
        label, n, full_ret, is_n, is_ret, is_lift, oos_n, oos_ret, oos_lift, verdict, _, _, _ = r
        print(f'  {label:<14} {n:>6,} {full_ret:>+6.2f}  {is_n:>6,} {is_lift:>+7.2f}  '
              f'{oos_n:>6,} {oos_lift:>+8.2f}  {verdict}')

    print(f'\n## ★ 真好规律列表:')
    real_good = [r for r in rules if r[9] == '★ 真好']
    print(f'  数: {len(real_good)}')
    for r in real_good:
        print(f'  {r[0]:<14} (col={r[10]}, op={r[11]}, thr={r[12]})')

    # score 分级
    if real_good:
        print(f'\n## 软排名 score 分级')
        score_arr = np.zeros(len(df_e))
        for r in real_good:
            col, op, thr = r[10], r[11], r[12]
            if op == '==':
                score_arr += (df_e[col].values == thr).astype(int)
            elif op == '<':
                score_arr += (df_e[col].values < thr).astype(int)
            elif op == '>':
                score_arr += (df_e[col].values > thr).astype(int)
        df_e['score'] = score_arr

        for sc_min in [0, 1, 2, 3]:
            sub = df_e[df_e['score'] >= sc_min]
            sub_is = sub[sub['period'] == 'IS']; sub_oos = sub[sub['period'] == 'OOS']
            if len(sub) == 0: continue
            print(f'  score >= {sc_min}: n={len(sub):,} ({len(sub)/len(df_e)*100:.0f}%)')
            print(f'    IS: {sub_is["ret_30"].mean():+.2f}% (n={len(sub_is):,})')
            print(f'    OOS: {sub_oos["ret_30"].mean():+.2f}% (n={len(sub_oos):,})')
            print(f'    全: {sub["ret_30"].mean():+.2f}% 胜 {(sub["ret_30"]>0).mean()*100:.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
