# -*- coding: utf-8 -*-
"""阶段 5: 乾 regime IS/OOS 软排名

IS  = w2_2019 + w3_2020 + w4_2021 + w5_2022 (4 段, 跨牛熊抱团崩)
OOS = w6_2023_24 + w7_2025_26 (2 段, 抱团延续)
  → 选 OOS=w6+w7 因为是更近的市场, 投产时更接近未来
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
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

# 强避雷 6 条
AVOID = [('mkt_d', '100'), ('mkt_d', '101'), ('mkt_d', '110'),
         ('mkt_m', '101'),
         ('stk_m', '100'), ('stk_m', '101')]

IS_SEGS = ['w2_2019', 'w3_2020', 'w4_2021', 'w5_2022']
OOS_SEGS = ['w6_2023_24', 'w7_2025_26']

WINDOWS = {
    'w2_2019':    ('2019-01-01', '2020-01-01'),
    'w3_2020':    ('2020-01-01', '2021-01-01'),
    'w4_2021':    ('2021-01-01', '2022-01-01'),
    'w5_2022':    ('2022-01-01', '2023-01-01'),
    'w6_2023_24': ('2023-01-01', '2025-01-01'),
    'w7_2025_26': ('2025-01-01', '2026-04-21'),
}


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
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫乾 regime 巽日事件 (含数值特征) ===')
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
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            mf_5d = float(np.nanmean(mf_arr[gi-5:gi+1])) if gi-5 >= s else float('nan')
            sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1])) if gi-5 >= s else float('nan')
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
                'trend': float(trend_arr[gi]),
                'mf': float(mf_arr[gi]),
                'mf_5d': mf_5d,
                'sanhu': float(sanhu_arr[gi]),
                'sanhu_5d': sanhu_5d,
            })
    df_e = pd.DataFrame(events)

    df_e['seg'] = ''
    for w_name, (a, b) in WINDOWS.items():
        df_e.loc[(df_e['date'] >= a) & (df_e['date'] < b), 'seg'] = w_name
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  事件: {len(df_e):,}')

    avoid_mask = pd.Series(False, index=df_e.index)
    for col, state in AVOID:
        avoid_mask = avoid_mask | (df_e[col] == state)
    pool = df_e[~avoid_mask].copy()
    print(f'  避雷后: {len(pool):,}')

    bl_is = pool[pool['seg'].isin(IS_SEGS)]['ret_30'].mean()
    bl_oos = pool[pool['seg'].isin(OOS_SEGS)]['ret_30'].mean()
    print(f'  IS baseline: {bl_is:+.2f}%  OOS baseline: {bl_oos:+.2f}%')

    candidates = []

    # 卦象单维
    for col in ['mkt_d', 'mkt_m', 'stk_m', 'stk_y']:
        for state in GUAS:
            sub = pool[pool[col] == state]
            if len(sub) < 500: continue
            label = f'{col}={state}{GUA_NAMES[state]}'
            sub_is = sub[sub['seg'].isin(IS_SEGS)]
            sub_oos = sub[sub['seg'].isin(OOS_SEGS)]
            if len(sub_is) < 200 or len(sub_oos) < 200: continue
            r_is = sub_is['ret_30'].mean()
            r_oos = sub_oos['ret_30'].mean()
            l_is = r_is - bl_is
            l_oos = r_oos - bl_oos
            candidates.append((label, len(sub), len(sub_is), len(sub_oos), r_is, r_oos, l_is, l_oos))

    # 数值阈值
    for col, thresh_list in [('trend', [40, 50, 60]),
                              ('mf', [-50, 0, 50, 100]),
                              ('mf_5d', [-50, 0, 50]),
                              ('sanhu', [-50, 0, 50]),
                              ('sanhu_5d', [-100, -50, 0, 50])]:
        for thresh in thresh_list:
            for op_label in ['>', '<']:
                if op_label == '>':
                    sub = pool[pool[col] > thresh]
                else:
                    sub = pool[pool[col] < thresh]
                if len(sub) < 1000: continue
                label = f'{col}{op_label}{thresh}'
                sub_is = sub[sub['seg'].isin(IS_SEGS)]
                sub_oos = sub[sub['seg'].isin(OOS_SEGS)]
                if len(sub_is) < 200 or len(sub_oos) < 200: continue
                r_is = sub_is['ret_30'].mean()
                r_oos = sub_oos['ret_30'].mean()
                l_is = r_is - bl_is
                l_oos = r_oos - bl_oos
                candidates.append((label, len(sub), len(sub_is), len(sub_oos), r_is, r_oos, l_is, l_oos))

    print(f'\n## ★ 真好规律 (IS lift ≥ +1, OOS lift ≥ +0.5)')
    real_good = [c for c in candidates if c[6] >= 1.0 and c[7] >= 0.5]
    real_good.sort(key=lambda x: x[6] + x[7], reverse=True)
    print(f'  {"label":<22} {"n":>6} {"IS n":>6} {"OOS n":>6} {"IS lift":>8} {"OOS lift":>9} {"综合":>7}')
    if not real_good: print('  无')
    for c in real_good:
        print(f'  {c[0]:<22} {c[1]:>6,} {c[2]:>6,} {c[3]:>6,} {c[6]:>+7.2f} {c[7]:>+8.2f} {c[6]+c[7]:>+6.2f}')

    print(f'\n## ✗ IS 强 OOS 弱 (切片福利)')
    slice_warning = [c for c in candidates if c[6] >= 1.5 and c[7] < 0.5]
    slice_warning.sort(key=lambda x: x[6] - x[7], reverse=True)
    print(f'  {"label":<22} {"n":>6} {"IS lift":>8} {"OOS lift":>9}')
    for c in slice_warning[:10]:
        print(f'  {c[0]:<22} {c[1]:>6,} {c[6]:>+7.2f} {c[7]:>+8.2f}')

    if real_good:
        seen_cols = set()
        top_rules = []
        for c in real_good:
            label = c[0]
            col_key = label.split('=')[0] if '=' in label else (
                label.split('<')[0] if '<' in label else label.split('>')[0])
            if col_key in seen_cols: continue
            seen_cols.add(col_key)
            top_rules.append(c)
            if len(top_rules) >= 5: break

        print(f'\n## 软排名 score 分级 (用 top {len(top_rules)} 不嵌套规律)')
        for c in top_rules:
            print(f'  + {c[0]}')

        pool_score = pool.copy()
        pool_score['score'] = 0
        for c in top_rules:
            label = c[0]
            if '=' in label and not (label.startswith('mf') or label.startswith('sanhu') or label.startswith('trend')):
                col, state_full = label.split('=')
                state = state_full[:3]
                pool_score.loc[pool_score[col] == state, 'score'] += 1
            else:
                for op in ['>', '<']:
                    if op in label:
                        col, thresh = label.split(op)
                        thresh = float(thresh)
                        if op == '>':
                            pool_score.loc[pool_score[col] > thresh, 'score'] += 1
                        else:
                            pool_score.loc[pool_score[col] < thresh, 'score'] += 1
                        break

        print(f'\n  {"score":>6} {"n":>6} {"IS%":>8} {"OOS%":>8} {"全%":>8} {"主升率%":>8}')
        for s in sorted(pool_score['score'].unique()):
            sub = pool_score[pool_score['score'] == s]
            ret_is = sub[sub['seg'].isin(IS_SEGS)]['ret_30'].mean()
            ret_oos = sub[sub['seg'].isin(OOS_SEGS)]['ret_30'].mean()
            ret_all = sub['ret_30'].mean()
            zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
            print(f'  {s:>6} {len(sub):>6,} {ret_is:>+7.2f} {ret_oos:>+7.2f} {ret_all:>+7.2f} {zsl:>7.1f}')


if __name__ == '__main__':
    main()
