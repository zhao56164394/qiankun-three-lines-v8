# -*- coding: utf-8 -*-
"""Step 19 — 避雷 v2: 在 (巽日 + 避雷B) 子集里找剩余假巽特征

输入: Step 18 的避雷 B 通过 406K 事件
内部: 真巽 / 假巽 / 中性 重新分布
寻找 v1 没识别的 假巽特征
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
ZSL_THRESH = 10
FAKE_THRESH = 5

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)

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
    df = df.dropna(subset=['d_trend', 'close', 'd_gua', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d = df['d_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy(); stk_y = df['y_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy(); mkt_m = df['mkt_m'].to_numpy(); mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # === 扫描所有巽日, 同时记录避雷 B 是否通过 ===
    print(f'\n=== 扫描巽日 + 应用避雷 B ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = stk_d[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(td)

        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            win_lo = i - LOOKBACK + 1
            mf30 = mf[win_lo:i+1].mean()
            trend5 = td[i] - td[max(i-4, win_lo)]
            global_idx = s + i

            # 避雷 B
            avoid_B = (
                mkt_y[global_idx] == '011' or
                mkt_d[global_idx] == '110' or
                stk_m[global_idx] == '101' or
                trend5 < 0 or
                mf30 > 0
            )
            if avoid_B: continue

            ev = {
                'date': date_arr[global_idx],
                'n_qian': int(n_qian),
                'ret_30': ret_30,
                'stk_m': stk_m[global_idx], 'stk_y': stk_y[global_idx],
                'mkt_d': mkt_d[global_idx], 'mkt_m': mkt_m[global_idx], 'mkt_y': mkt_y[global_idx],
                'trend': td[i], 'mf': mf[i], 'sanhu': sanhu[i],
                'trend_min_30d': td[win_lo:i+1].min(),
                'trend_max_30d': td[win_lo:i+1].max(),
                'trend_slope_30d': td[i] - td[win_lo],
                'trend_slope_5d': trend5,
                'mf_30d_mean': mf30,
                'mf_30d_min': mf[win_lo:i+1].min(),
                'mf_30d_max': mf[win_lo:i+1].max(),
                'mf_5d_mean': mf[max(i-4, win_lo):i+1].mean(),
                'sanhu_30d_mean': sanhu[win_lo:i+1].mean(),
                'sanhu_30d_min': sanhu[win_lo:i+1].min(),
                'sanhu_5d_mean': sanhu[max(i-4, win_lo):i+1].mean(),
            }
            # 前 30 日 d_gua 频率
            for g_v in GUAS:
                ev[f'pct_d_{g_v}'] = (gua[win_lo:i+1] == g_v).sum() / LOOKBACK
            events.append(ev)

    print(f'  通过 B 事件: {len(events):,}, {time.time()-t1:.1f}s')

    df_e = pd.DataFrame(events)
    df_e['kind'] = 'mid'
    df_e.loc[df_e['n_qian'] >= ZSL_THRESH, 'kind'] = 'real'
    df_e.loc[df_e['n_qian'] <= FAKE_THRESH, 'kind'] = 'fake'

    n_real = (df_e['kind'] == 'real').sum()
    n_fake = (df_e['kind'] == 'fake').sum()
    n_mid = (df_e['kind'] == 'mid').sum()
    print(f'\n## 避雷 B 后 巽日分类:')
    print(f'  真巽 (乾≥{ZSL_THRESH}): {n_real:,} ({n_real/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="real"]["ret_30"].mean():+.2f}%')
    print(f'  假巽 (乾≤{FAKE_THRESH}): {n_fake:,} ({n_fake/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="fake"]["ret_30"].mean():+.2f}%')
    print(f'  中性 (5<乾<10):  {n_mid:,} ({n_mid/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="mid"]["ret_30"].mean():+.2f}%')

    df_real = df_e[df_e['kind'] == 'real']
    df_fake = df_e[df_e['kind'] == 'fake']

    # === 1. 数值特征 真假对比 ===
    print(f'\n## 1. 通过避雷B 后 数值特征 真巽 vs 假巽 (找 v2 避雷线索)')
    print(f'  {"特征":<22} {"真中位":>9} {"假中位":>9} {"差(假-真)":>10}')
    print('  ' + '-' * 60)
    num_cols = ['trend', 'mf', 'sanhu',
                'trend_min_30d', 'trend_max_30d', 'trend_slope_30d', 'trend_slope_5d',
                'mf_30d_mean', 'mf_30d_min', 'mf_30d_max', 'mf_5d_mean',
                'sanhu_30d_mean', 'sanhu_30d_min', 'sanhu_5d_mean']
    diffs = []
    for c in num_cols:
        rm = df_real[c].median(); fm = df_fake[c].median()
        diffs.append((c, rm, fm, fm - rm))
    diffs.sort(key=lambda x: -abs(x[3]))
    for c, rm, fm, diff in diffs:
        print(f'  {c:<22} {rm:>+8.2f} {fm:>+8.2f} {diff:>+8.2f}')

    # === 2. 卦象 真假对比 (剩余信号) ===
    print(f'\n## 2. 通过 B 后 当下卦象 真假对比')
    for col, label in [('stk_m', '个股 m_gua'), ('stk_y', '个股 y_gua'),
                        ('mkt_d', '大盘 d_gua'), ('mkt_m', '大盘 m_gua')]:
        diffs_g = []
        for g_v in GUAS:
            r = (df_real[col] == g_v).mean() * 100
            f = (df_fake[col] == g_v).mean() * 100
            diffs_g.append((g_v, r, f, f - r))
        diffs_g.sort(key=lambda x: -abs(x[3]))
        print(f'\n  {label} (按 |差| 排序):')
        for g_v, r, f, d in diffs_g[:5]:
            mark = '⚠' if d >= 3 else ('★' if d <= -3 else '')
            print(f'    {g_v}{GUA_NAMES[g_v]}  真 {r:>5.1f}%  假 {f:>5.1f}%  差 {d:>+5.1f}  {mark}')

    # === 3. 前 30 日 d_gua 频率 真假对比 ===
    print(f'\n## 3. 通过 B 后 前 30 日 d_gua 频率')
    diffs_pct = []
    for g_v in GUAS:
        c = f'pct_d_{g_v}'
        r = df_real[c].mean() * 100; f = df_fake[c].mean() * 100
        diffs_pct.append((g_v, r, f, f - r))
    diffs_pct.sort(key=lambda x: -abs(x[3]))
    for g_v, r, f, d in diffs_pct[:5]:
        mark = '⚠' if d >= 2 else ('★' if d <= -2 else '')
        print(f'  {g_v}{GUA_NAMES[g_v]}  真均 {r:>5.1f}%  假均 {f:>5.1f}%  差 {d:>+5.1f}  {mark}')

    # === 4. 单项 v2 避雷条件 ablation (排除满足者, 看 lift) ===
    print(f'\n## 4. v2 避雷单项 (在通过 B 的子集上, 进一步排除)')
    base_ret = df_e['ret_30'].mean()
    base_zsl = (df_e['n_qian'] >= ZSL_THRESH).mean() * 100
    print(f'  baseline 通过 B 子集: 期望 {base_ret:+.2f}%, 主升率 {base_zsl:.1f}% ({len(df_e):,} 事件)')
    print(f'  {"排除条件":<35} {"剩 n":>7} {"剩%":>5} {"期望":>7} {"vs B":>6} {"主升率":>7} {"lift":>6}')
    print('  ' + '-' * 75)

    avoid_v2 = [
        ('mkt_d == 101 离', df_e['mkt_d'] == '101'),
        ('mkt_d == 100 震', df_e['mkt_d'] == '100'),
        ('stk_y == 011 巽', df_e['stk_y'] == '011'),
        ('mkt_y == 110 兑', df_e['mkt_y'] == '110'),
        ('mf > 200 (主力当日已暴增)', df_e['mf'] > 200),
        ('mf_30d_min > -100 (洗盘不深)', df_e['mf_30d_min'] > -100),
        ('mf_30d_max > 200 (期内主力曾大增)', df_e['mf_30d_max'] > 200),
        ('sanhu_30d_min > -50 (散户从未深恐慌)', df_e['sanhu_30d_min'] > -50),
        ('sanhu_5d_mean > -10 (散户已不恐慌)', df_e['sanhu_5d_mean'] > -10),
        ('trend > 60 (位置已偏高)', df_e['trend'] > 60),
        ('trend_min_30d > 30 (前 30 日没探底)', df_e['trend_min_30d'] > 30),
        ('trend_slope_30d > 10 (前 30 日已涨)', df_e['trend_slope_30d'] > 10),
        ('mf_5d_mean < 0 (主力近 5 日还在卖)', df_e['mf_5d_mean'] < 0),
    ]
    for label, mask in avoid_v2:
        keep = df_e[~mask]
        if len(keep) < 1000: continue
        ret = keep['ret_30'].mean()
        zsl = (keep['n_qian'] >= ZSL_THRESH).mean() * 100
        lift = ret - base_ret
        zsl_lift = zsl - base_zsl
        mark = '✅' if lift >= 0.3 else ('❌' if lift <= -0.3 else '○')
        print(f'  {label:<35} {len(keep):>7,} {len(keep)/len(df_e)*100:>4.0f}% '
              f'{ret:>+6.2f}% {lift:>+5.2f} {zsl:>5.1f}% {zsl_lift:>+5.1f}  {mark}')

    # === 5. 避雷 B + Top 3 v2 项 组合 ===
    print(f'\n## 5. 组合 D = B + Top 3 v2 单项 (按上面 ✅)')
    # 选择 lift > 0.3 的 top 3
    avoid_D = (
        (df_e['mkt_d'] == '101') |
        (df_e['stk_y'] == '011') |
        (df_e['trend_min_30d'] > 30) |
        (df_e['mf_30d_max'] > 200)
    )
    keep_D = df_e[~avoid_D]
    print(f'  组合 D = B + (大盘d=101 OR 个股y=011 OR trend_min_30d>30 OR mf_30d_max>200)')
    print(f'    剩 {len(keep_D):,} ({len(keep_D)/len(df_e)*100:.0f}% of B, {len(keep_D)/748942*100:.0f}% of 巽日)')
    print(f'    期望: {keep_D["ret_30"].mean():+.2f}% (vs B {base_ret:+.2f}%, lift {keep_D["ret_30"].mean()-base_ret:+.2f}; vs 巽日 +{keep_D["ret_30"].mean()-2.16:.2f})')
    print(f'    主升率: {(keep_D["n_qian"]>=ZSL_THRESH).mean()*100:.1f}% (vs B {base_zsl:.1f})')

    # walk-forward
    print(f'\n## 6. 组合 D walk-forward')
    df_e['seg'] = ''
    keep_D = df_e[~avoid_D]
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label
    keep_D['seg'] = df_e.loc[keep_D.index, 'seg']

    print(f'  {"段":<14} {"D 事件":>7} {"D 期望":>7} {"B 期望":>7} {"D-B":>7}')
    print('  ' + '-' * 50)
    n_pass = 0
    for w in WINDOWS:
        seg_b = df_e[df_e['seg'] == w[0]]
        seg_d = keep_D[keep_D['seg'] == w[0]]
        if len(seg_b) < 100 or len(seg_d) < 100: continue
        b_ret = seg_b['ret_30'].mean(); d_ret = seg_d['ret_30'].mean()
        diff = d_ret - b_ret
        mark = '✅' if diff > 0.3 else ('❌' if diff < -0.3 else '○')
        if diff > 0.3: n_pass += 1
        print(f'  {w[0]:<14} {len(seg_d):>7,} {d_ret:>+6.2f}% {b_ret:>+6.2f}% {diff:>+6.2f} {mark}')
    print(f'\n  → {n_pass}/7 段 D > B + 0.3')


if __name__ == '__main__':
    main()
