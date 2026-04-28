# -*- coding: utf-8 -*-
"""阶段 5: 艮 regime 软排名 (w2/w4 互换 IS/OOS)

输入: 已避雷池 (强避雷: 个股 m_gua=111 乾)
设计:
  方案 A: w2 = IS, w4 = OOS  (用大熊段建模, 抱团段验证)
  方案 B: w4 = IS, w2 = OOS  (用抱团段建模, 大熊段验证)

  必须两个方案都过 = 真好规律
  仅一个过 = case study 候选

候选维度:
  - 6 个卦象单维条件
  - 主力线 / 散户线 / 趋势线 阈值
  - 前 30 日卦象频率
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '001'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

# 强避雷条件
AVOID = [('stk_m', '111')]


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

    print(f'\n=== 扫艮 regime 巽日事件 (含数值特征) ===')
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
            # 数值特征
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
    print(f'  事件: {len(df_e):,}')

    # 标段
    df_e['seg'] = ''
    df_e.loc[(df_e['date'] >= '2019-01-01') & (df_e['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_e.loc[(df_e['date'] >= '2021-01-01') & (df_e['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    df_e = df_e[df_e['seg'] != ''].copy()

    # 应用强避雷
    avoid_mask = pd.Series(False, index=df_e.index)
    for col, state in AVOID:
        avoid_mask = avoid_mask | (df_e[col] == state)
    pool = df_e[~avoid_mask].copy()
    print(f'  避雷后: {len(pool):,}')

    bl_w2 = pool[pool['seg'] == 'w2_2019']['ret_30'].mean()
    bl_w4 = pool[pool['seg'] == 'w4_2021']['ret_30'].mean()
    print(f'  w2 baseline: {bl_w2:+.2f}% / w4 baseline: {bl_w4:+.2f}%')

    # 扫候选
    candidates = []

    # 卦象单维
    for col in ['mkt_d', 'mkt_m', 'stk_m', 'stk_y']:
        for state in GUAS:
            sub = pool[pool[col] == state]
            if len(sub) < 50: continue
            label = f'{col}={state}{GUA_NAMES[state]}'
            n_w2 = (sub['seg'] == 'w2_2019').sum()
            n_w4 = (sub['seg'] == 'w4_2021').sum()
            if n_w2 < 30 and n_w4 < 30: continue
            r_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean() if n_w2 > 0 else float('nan')
            r_w4 = sub[sub['seg'] == 'w4_2021']['ret_30'].mean() if n_w4 > 0 else float('nan')
            l_w2 = r_w2 - bl_w2 if n_w2 > 0 else float('nan')
            l_w4 = r_w4 - bl_w4 if n_w4 > 0 else float('nan')
            candidates.append((label, len(sub), n_w2, n_w4, r_w2, r_w4, l_w2, l_w4))

    # 数值阈值
    for col, thresh_list in [('trend', [40, 50, 60]),
                              ('mf', [0, 30, 50, 100]),
                              ('mf_5d', [-20, 0, 30]),
                              ('sanhu', [-30, 0, 30]),
                              ('sanhu_5d', [-50, -30, 0, 30])]:
        for thresh in thresh_list:
            for op_label, op_func in [('>', lambda x: pool[col] > thresh),
                                        ('<', lambda x: pool[col] < thresh)]:
                sub = pool[op_func(None)]
                if len(sub) < 50: continue
                label = f'{col}{op_label}{thresh}'
                n_w2 = (sub['seg'] == 'w2_2019').sum()
                n_w4 = (sub['seg'] == 'w4_2021').sum()
                if n_w2 < 30 and n_w4 < 30: continue
                r_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean() if n_w2 > 0 else float('nan')
                r_w4 = sub[sub['seg'] == 'w4_2021']['ret_30'].mean() if n_w4 > 0 else float('nan')
                l_w2 = r_w2 - bl_w2 if n_w2 > 0 else float('nan')
                l_w4 = r_w4 - bl_w4 if n_w4 > 0 else float('nan')
                candidates.append((label, len(sub), n_w2, n_w4, r_w2, r_w4, l_w2, l_w4))

    # 分类
    print(f'\n## ★★ 强好规律 (两段 lift ≥ +1, 两段都有 ≥30 样本)')
    strong_good = [c for c in candidates if c[2] >= 30 and c[3] >= 30 and c[6] >= 1.0 and c[7] >= 1.0]
    if not strong_good: print('  无')
    for c in strong_good:
        print(f'  {c[0]:<22} n {c[1]:>4} | w2 n{c[2]:>4} lift {c[6]:>+5.2f} | w4 n{c[3]:>4} lift {c[7]:>+5.2f}')

    print(f'\n## ★ 弱好规律 (一段 lift ≥ +2, 另一段无足够样本)')
    weak_good = [c for c in candidates if (c[2] >= 30 and c[6] >= 2.0 and c[3] < 30) or
                                            (c[3] >= 30 and c[7] >= 2.0 and c[2] < 30)]
    if not weak_good: print('  无')
    for c in weak_good:
        l_w2 = f'{c[6]:+5.2f}' if c[2] >= 30 else 'NA'
        l_w4 = f'{c[7]:+5.2f}' if c[3] >= 30 else 'NA'
        print(f'  {c[0]:<22} n {c[1]:>4} | w2 n{c[2]:>4} lift {l_w2} | w4 n{c[3]:>4} lift {l_w4}')

    print(f'\n## ✗ IS强OOS弱 (一段 lift ≥ +2, 另一段 lift < 0, 切片福利警告)')
    slice_warning = [c for c in candidates if c[2] >= 30 and c[3] >= 30 and
                     ((c[6] >= 2 and c[7] < 0) or (c[7] >= 2 and c[6] < 0))]
    if not slice_warning: print('  无')
    for c in slice_warning:
        print(f'  {c[0]:<22} n {c[1]:>4} | w2 n{c[2]:>4} lift {c[6]:>+5.2f} | w4 n{c[3]:>4} lift {c[7]:>+5.2f}')

    # 软排名验证
    if strong_good:
        print(f'\n## 软排名 score 分级 (用强好规律)')
        rules = strong_good
        # 给每个事件算 score
        pool_score = pool.copy()
        pool_score['score'] = 0
        for c in rules:
            label = c[0]
            if '=' in label:
                col, state = label.split('=')
                state = state[:3]
                pool_score.loc[pool_score[col] == state, 'score'] += 1
            else:
                # 数值阈值
                for op in ['>', '<']:
                    if op in label:
                        col, thresh = label.split(op)
                        thresh = float(thresh)
                        if op == '>':
                            pool_score.loc[pool_score[col] > thresh, 'score'] += 1
                        else:
                            pool_score.loc[pool_score[col] < thresh, 'score'] += 1
                        break
        print(f'  {"score":>6} {"n":>6} {"w2 n":>6} {"w2%":>7} {"w4 n":>6} {"w4%":>7} {"全期望%":>8} {"主升率%":>8}')
        for s in sorted(pool_score['score'].unique()):
            sub = pool_score[pool_score['score'] == s]
            n_w2 = (sub['seg'] == 'w2_2019').sum()
            r_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean() if n_w2 > 0 else float('nan')
            n_w4 = (sub['seg'] == 'w4_2021').sum()
            r_w4 = sub[sub['seg'] == 'w4_2021']['ret_30'].mean() if n_w4 > 0 else float('nan')
            ret_full = sub['ret_30'].mean()
            zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
            r_w2_str = f'{r_w2:+.2f}' if n_w2 > 0 else '--'
            r_w4_str = f'{r_w4:+.2f}' if n_w4 > 0 else '--'
            print(f'  {s:>6} {len(sub):>6} {n_w2:>6} {r_w2_str:>7} {n_w4:>6} {r_w4_str:>7} {ret_full:>+7.2f} {zsl:>7.1f}')


if __name__ == '__main__':
    main()
