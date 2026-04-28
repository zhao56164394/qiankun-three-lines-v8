# -*- coding: utf-8 -*-
"""Step 27 — 坤 regime 好规律 IS/OOS 双重验证

做法:
  - IS: w1_2018 + w2_2019 + w4_2021 + w5_2022 (4 段, 跨牛熊震荡)
  - OOS: w6_2023_24 + w7_2025_26 (但 w7 在坤 regime 内无样本, 实际用 w6)

  对避雷后的 102K 事件:
    1. IS 中找 跨段一致 + lift > +1% 的好规律
    2. OOS (w6_2023_24) 验证, lift 仍 > 0 才算真好
    3. 业务可解释加分

输出: 真好规律列表 + 每条规律的"分数" (lift / IS段数 / OOS段数)
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
REGIME_Y = '000'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

# IS / OOS 拆分
IS_SEGS = ['w1_2018', 'w2_2019', 'w4_2021', 'w5_2022']
OOS_SEGS = ['w6_2023_24']  # w3 / w7 坤 regime 无样本

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

# Step 26 找到的避雷条件 (硬过滤)
AVOID = [
    ('mkt_d', '000'), ('mkt_d', '001'), ('mkt_d', '100'), ('mkt_d', '101'),
    ('stk_y', '001'), ('stk_y', '011'),
    ('stk_m', '101'), ('stk_m', '110'), ('stk_m', '111'),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
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
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫坤 regime 巽日 + 避雷过滤 + 提特征 ===')
    t1 = time.time()
    avoid_set = set(AVOID)
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = stk_d_arr[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            # 避雷
            avoid = False
            for col, val in AVOID:
                arr_map = {'mkt_d': mkt_d_arr, 'mkt_m': mkt_m_arr, 'mkt_y': mkt_y_arr,
                           'stk_d': stk_d_arr, 'stk_m': stk_m_arr, 'stk_y': stk_y_arr}
                if arr_map[col][gi] == val:
                    avoid = True
                    break
            if avoid: continue

            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            wlo = i - LOOKBACK + 1
            events.append({
                'date': date_arr[gi], 'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi],
                'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
                'trend': td[i], 'mf': mf[i], 'sanhu': sanhu[i],
                'trend_5d': td[i] - td[max(i-4, wlo)],
                'trend_30d': td[i] - td[wlo],
                'mf_30d_mean': mf[wlo:i+1].mean(),
                'mf_30d_min': mf[wlo:i+1].min(),
                'mf_30d_max': mf[wlo:i+1].max(),
                'sanhu_30d_min': sanhu[wlo:i+1].min(),
                'sanhu_5d_mean': sanhu[max(i-4, wlo):i+1].mean(),
            })
    df_e = pd.DataFrame(events)
    print(f'  避雷后事件: {len(df_e):,}, {time.time()-t1:.1f}s')

    df_e['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label
    df_e = df_e[df_e['seg'] != ''].copy()

    # IS / OOS 子集
    df_is = df_e[df_e['seg'].isin(IS_SEGS)].copy()
    df_oos = df_e[df_e['seg'].isin(OOS_SEGS)].copy()
    print(f'\n## 拆分')
    print(f'  IS (w1+w2+w4+w5): {len(df_is):,}, baseline {df_is["ret_30"].mean():+.2f}%')
    print(f'  OOS (w6_2023_24): {len(df_oos):,}, baseline {df_oos["ret_30"].mean():+.2f}%')

    # 段 baseline
    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret_30'].mean() if len(seg) > 0 else 0

    # === 候选好规律: 卦象单维 + 数值阈值 ===
    print(f'\n## 候选好规律 IS/OOS 验证')
    print(f'  {"规律":<35} {"IS n":>6} {"IS ret":>7} {"IS lift":>7}', end='')
    print(f' {"OOS n":>6} {"OOS ret":>7} {"OOS lift":>8}', end='')
    print(f' {"判定":>10}')
    print('  ' + '-' * 100)

    candidates = [
        # 卦象单维
        ('mkt_d=011 巽', df_e['mkt_d'] == '011'),
        ('mkt_d=010 坎', df_e['mkt_d'] == '010'),
        ('mkt_d=011 巽 OR 010 坎', df_e['mkt_d'].isin(['011', '010'])),
        ('mkt_m=010 坎', df_e['mkt_m'] == '010'),
        ('mkt_m=100 震', df_e['mkt_m'] == '100'),
        ('stk_y=000 坤', df_e['stk_y'] == '000'),
        ('stk_y=010 坎', df_e['stk_y'] == '010'),
        ('stk_y=000坤 OR 010坎', df_e['stk_y'].isin(['000', '010'])),
        ('stk_m=010 坎', df_e['stk_m'] == '010'),
        ('stk_m=000 坤', df_e['stk_m'] == '000'),
        ('stk_m=011 巽', df_e['stk_m'] == '011'),
        # 数值阈值
        ('trend_5d > 5', df_e['trend_5d'] > 5),
        ('trend_5d > 10', df_e['trend_5d'] > 10),
        ('trend_5d > 15', df_e['trend_5d'] > 15),
        ('trend_30d > 0 (前 30 日已涨)', df_e['trend_30d'] > 0),
        ('trend_30d < -10 (前 30 日深跌)', df_e['trend_30d'] < -10),
        ('mf > 50 (主力当日已发力)', df_e['mf'] > 50),
        ('mf > 100', df_e['mf'] > 100),
        ('mf_30d_min < -100', df_e['mf_30d_min'] < -100),
        ('mf_30d_min < -200', df_e['mf_30d_min'] < -200),
        ('mf_30d_min < -300', df_e['mf_30d_min'] < -300),
        ('mf_30d_max > 100', df_e['mf_30d_max'] > 100),
        ('sanhu_30d_min < -100', df_e['sanhu_30d_min'] < -100),
        ('sanhu_30d_min < -150', df_e['sanhu_30d_min'] < -150),
        ('sanhu_5d_mean < -30', df_e['sanhu_5d_mean'] < -30),
        ('sanhu_5d_mean < -50', df_e['sanhu_5d_mean'] < -50),
    ]

    base_is = df_is['ret_30'].mean()
    base_oos = df_oos['ret_30'].mean()

    real_good = []
    for label, mask in candidates:
        is_mask = mask.loc[df_is.index]
        oos_mask = mask.loc[df_oos.index]
        sub_is = df_is[is_mask]
        sub_oos = df_oos[oos_mask]

        if len(sub_is) < 500 or len(sub_oos) < 100:
            continue
        is_ret = sub_is['ret_30'].mean()
        oos_ret = sub_oos['ret_30'].mean()
        is_lift = is_ret - base_is
        oos_lift = oos_ret - base_oos

        if is_lift >= 1 and oos_lift >= 0.5:
            verdict = '★真好'
            real_good.append((label, len(sub_is), is_ret, is_lift, len(sub_oos), oos_ret, oos_lift))
        elif is_lift >= 1 and oos_lift < 0.5:
            verdict = '✗ IS 强OOS弱'
        elif is_lift < 1 and oos_lift >= 0.5:
            verdict = '○ OOS 偶尔'
        else:
            verdict = '— 弱'

        print(f'  {label:<35} {len(sub_is):>6,} {is_ret:>+5.2f}% {is_lift:>+5.2f}', end='')
        print(f' {len(sub_oos):>6,} {oos_ret:>+5.2f}% {oos_lift:>+6.2f}', end='')
        print(f'  {verdict}')

    # 排序输出真好
    print(f'\n## ★ 真好规律排序 (按 IS+OOS 综合 lift)')
    real_good.sort(key=lambda x: -(x[3] + x[6]))
    print(f'  {"规律":<35} {"IS":>10} {"OOS":>10} {"综合 lift":>10}')
    for label, n_is, ir, il, n_oos, or_, ol in real_good:
        print(f'  {label:<35} {ir:>+5.2f}% [{n_is:>5}] {or_:>+5.2f}% [{n_oos:>5}] {il+ol:>+8.2f}')

    # ============== 多规律组合: 至少 N 项命中 ==============
    if real_good:
        # 取 IS+OOS 综合 lift > 1.5 的真好规律
        top_rules = [(r[0], None) for r in real_good if r[3] + r[6] >= 2]
        # 构造 mask
        # 重建 mask
        cand_dict = dict(candidates)
        top_masks = [(label, cand_dict[label]) for label, _ in top_rules]

        print(f'\n## Top 规律 ({len(top_masks)} 条) 命中数量分级')
        # 对每个事件算"命中几条"
        hit_count = pd.Series(0, index=df_e.index)
        for label, mask in top_masks:
            hit_count = hit_count + mask.astype(int)
        df_e['hit_count'] = hit_count

        for k in range(0, min(len(top_masks) + 1, 8)):
            mask_k = (df_e['hit_count'] == k)
            if mask_k.sum() < 100: continue
            sub = df_e[mask_k]
            ret = sub['ret_30'].mean()
            zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
            print(f'  命中 {k} 条: n={mask_k.sum():>6,} ({mask_k.sum()/len(df_e)*100:>4.1f}%)  '
                  f'期望 {ret:>+5.2f}%  主升率 {zsl:>5.1f}%')

        # walk-forward 命中 ≥3 条
        for thresh in [2, 3, 4]:
            mask_t = (df_e['hit_count'] >= thresh)
            if mask_t.sum() < 100: continue
            print(f'\n  ## 命中 ≥{thresh} 条 walk-forward:')
            for w_label in IS_SEGS + OOS_SEGS:
                seg_b = df_e[df_e['seg'] == w_label]
                seg_k = df_e[mask_t & (df_e['seg'] == w_label)]
                if len(seg_b) < 50: continue
                b = seg_b['ret_30'].mean()
                k = seg_k['ret_30'].mean() if len(seg_k) > 0 else float('nan')
                diff = k - b
                mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
                tag = '[OOS]' if w_label in OOS_SEGS else '[IS]'
                print(f'    {w_label:<14} {tag:<6} 全 {b:>+5.2f}%  ≥{thresh}条 {k:>+5.2f}% ({len(seg_k):>5})  '
                      f'lift {diff:>+5.2f} {mark}')


if __name__ == '__main__':
    main()
