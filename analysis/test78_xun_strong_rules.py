# -*- coding: utf-8 -*-
"""阶段 5: 巽 regime — 找"两段都强"的规律 (软排名)

策略: 既然 baseline 全期 -4.02%, 单条规律加分难以翻正
       → 必须找单维条件中 w2 和 w5 都 lift > +5% 的强规律
       → 命中多条 score 累加, 看分级是否单调 + 是否能翻正

样本: 巽 regime 巽日 (35,815 control + treatment 之间)
段:
  IS = w2_2019 (6,574, baseline -11.66%)
  IS = w5_2022 (7,552, baseline +2.42%)
  OOS = w6_2023_24 (108, 太少不参与候选筛选)

筛选标准 (严格防切片):
  - w2 lift ≥ +5% AND w5 lift ≥ +3% → 真好规律
  - 必须两段样本都 ≥ 100
  - 数值阈值/卦象单维, 不做 AND 组合
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
QIAN_RUN = 10
REGIME_Y = '011'
TRIGGER_GUA = '011'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

# 应用之前发现的强避雷 (大盘 d_gua=111)
AVOID = [('mkt_d', '111')]


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

    print(f'\n=== 扫巽 regime 巽日事件 ===')
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
            trend_5d_slope = float(trend_arr[gi] - trend_arr[gi-5]) if gi-5 >= s else float('nan')
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
                'trend': float(trend_arr[gi]),
                'mf': float(mf_arr[gi]),
                'mf_5d': mf_5d,
                'sanhu': float(sanhu_arr[gi]),
                'sanhu_5d': sanhu_5d,
                'trend_5d_slope': trend_5d_slope,
            })
    df_e = pd.DataFrame(events)

    df_e['seg'] = ''
    df_e.loc[(df_e['date'] >= '2019-01-01') & (df_e['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_e.loc[(df_e['date'] >= '2022-01-01') & (df_e['date'] < '2023-01-01'), 'seg'] = 'w5_2022'
    df_e.loc[(df_e['date'] >= '2023-01-01') & (df_e['date'] < '2025-01-01'), 'seg'] = 'w6_2023_24'
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  事件: {len(df_e):,}')

    # 应用强避雷
    avoid_mask = pd.Series(False, index=df_e.index)
    for col, state in AVOID:
        avoid_mask = avoid_mask | (df_e[col] == state)
    pool = df_e[~avoid_mask].copy()
    print(f'  避雷后 (大d≠乾): {len(pool):,}')

    bl = {}
    for sg in ['w2_2019', 'w5_2022', 'w6_2023_24']:
        sub = pool[pool['seg'] == sg]
        bl[sg] = sub['ret_30'].mean() if len(sub) > 0 else 0
        print(f'  baseline {sg:<14} n={len(sub):>5} 期望 {bl[sg]:>+5.2f}%')

    # 扫候选 — 严格双段验证
    candidates = []

    # 卦象单维
    for col in ['mkt_d', 'mkt_m', 'stk_m', 'stk_y']:
        for state in GUAS:
            sub = pool[pool[col] == state]
            if len(sub) < 200: continue
            label = f'{col}={state}{GUA_NAMES[state]}'
            n_w2 = (sub['seg'] == 'w2_2019').sum()
            n_w5 = (sub['seg'] == 'w5_2022').sum()
            if n_w2 < 100 or n_w5 < 100: continue
            r_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean()
            r_w5 = sub[sub['seg'] == 'w5_2022']['ret_30'].mean()
            l_w2 = r_w2 - bl['w2_2019']
            l_w5 = r_w5 - bl['w5_2022']
            candidates.append((label, len(sub), n_w2, n_w5, r_w2, r_w5, l_w2, l_w5))

    # 数值阈值
    for col, thresh_list in [('trend', [40, 50, 60]),
                              ('trend_5d_slope', [-5, 0, 5, 10]),
                              ('mf', [-50, 0, 50, 100]),
                              ('mf_5d', [-50, 0, 50]),
                              ('sanhu', [-30, 0, 30, 50]),
                              ('sanhu_5d', [-50, -20, 0, 30])]:
        for thresh in thresh_list:
            for op_label in ['>', '<']:
                if op_label == '>':
                    sub = pool[pool[col] > thresh]
                else:
                    sub = pool[pool[col] < thresh]
                if len(sub) < 500: continue
                label = f'{col}{op_label}{thresh}'
                n_w2 = (sub['seg'] == 'w2_2019').sum()
                n_w5 = (sub['seg'] == 'w5_2022').sum()
                if n_w2 < 100 or n_w5 < 100: continue
                r_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean()
                r_w5 = sub[sub['seg'] == 'w5_2022']['ret_30'].mean()
                l_w2 = r_w2 - bl['w2_2019']
                l_w5 = r_w5 - bl['w5_2022']
                candidates.append((label, len(sub), n_w2, n_w5, r_w2, r_w5, l_w2, l_w5))

    # 严格双段都过
    print(f'\n## ★★ 真好规律 (w2 lift ≥ +5, w5 lift ≥ +3, 防切片)')
    real_good = [c for c in candidates if c[6] >= 5.0 and c[7] >= 3.0]
    real_good.sort(key=lambda x: x[6] + x[7], reverse=True)
    print(f'  {"label":<22} {"n":>5} {"w2 n":>5} {"w5 n":>5} {"w2 lift":>8} {"w5 lift":>8}')
    if not real_good: print('  无')
    for c in real_good:
        print(f'  {c[0]:<22} {c[1]:>5} {c[2]:>5} {c[3]:>5} {c[6]:>+7.2f} {c[7]:>+7.2f}')

    print(f'\n## ★ 次好规律 (w2 lift ≥ +3, w5 lift ≥ +1)')
    semi_good = [c for c in candidates
                  if c[6] >= 3.0 and c[7] >= 1.0
                  and not (c[6] >= 5.0 and c[7] >= 3.0)]
    semi_good.sort(key=lambda x: x[6] + x[7], reverse=True)
    print(f'  {"label":<22} {"n":>5} {"w2 n":>5} {"w5 n":>5} {"w2 lift":>8} {"w5 lift":>8}')
    for c in semi_good[:10]:
        print(f'  {c[0]:<22} {c[1]:>5} {c[2]:>5} {c[3]:>5} {c[6]:>+7.2f} {c[7]:>+7.2f}')

    print(f'\n## ✗ 切片福利 (w2 强 w5 弱, 或反向)')
    slice_warning = [c for c in candidates
                      if (c[6] >= 5 and c[7] < 0) or (c[7] >= 5 and c[6] < 0)]
    slice_warning.sort(key=lambda x: abs(x[6] - x[7]), reverse=True)
    for c in slice_warning[:5]:
        print(f'  {c[0]:<22} w2 lift {c[6]:>+5.2f} | w5 lift {c[7]:>+5.2f}')

    print(f'\n## ☑ 双段都正 (w2 lift > 0 AND w5 lift > 0), 至少一段 ≥ +3')
    both_pos = [c for c in candidates
                 if c[6] > 0 and c[7] > 0
                 and (c[6] >= 3 or c[7] >= 3)]
    both_pos.sort(key=lambda x: x[6] + x[7], reverse=True)
    print(f'  {"label":<22} {"n":>5} {"w2 n":>5} {"w5 n":>5} {"w2 lift":>8} {"w5 lift":>8}')
    for c in both_pos[:15]:
        print(f'  {c[0]:<22} {c[1]:>5} {c[2]:>5} {c[3]:>5} {c[6]:>+7.2f} {c[7]:>+7.2f}')

    # 调用所有 both_pos 作 score
    real_good = both_pos[:8]

    # 软排名验证
    if real_good:
        # 注意去除嵌套 (sanhu_5d<0 包含 sanhu_5d<-20)
        # 这里只取 top 5 不嵌套的
        seen_cols = set()
        top_rules = []
        for c in real_good:
            label = c[0]
            col_key = label.split('=')[0] if '=' in label else label.split('<')[0].split('>')[0]
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

        print(f'\n  {"score":>6} {"n":>6} {"w2 期望%":>9} {"w5 期望%":>9} {"w6 期望%":>9} {"全期望%":>9} {"主升率%":>8}')
        for s in sorted(pool_score['score'].unique()):
            sub = pool_score[pool_score['score'] == s]
            ret_w2 = sub[sub['seg'] == 'w2_2019']['ret_30'].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
            ret_w5 = sub[sub['seg'] == 'w5_2022']['ret_30'].mean() if (sub['seg'] == 'w5_2022').sum() > 0 else float('nan')
            ret_w6 = sub[sub['seg'] == 'w6_2023_24']['ret_30'].mean() if (sub['seg'] == 'w6_2023_24').sum() > 0 else float('nan')
            ret_all = sub['ret_30'].mean()
            zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
            r_w2_str = f'{ret_w2:+.2f}' if not np.isnan(ret_w2) else '--'
            r_w5_str = f'{ret_w5:+.2f}' if not np.isnan(ret_w5) else '--'
            r_w6_str = f'{ret_w6:+.2f}' if not np.isnan(ret_w6) else '--'
            print(f'  {s:>6} {len(sub):>6,} {r_w2_str:>9} {r_w5_str:>9} {r_w6_str:>9} {ret_all:>+8.2f} {zsl:>7.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
