# -*- coding: utf-8 -*-
"""Step 18 — 反向避雷: 找假巽日的特征, 过滤掉

定义:
  巽日: 当日 d_gua=011
  真巽日 (好买点): 买入后 30 日 乾天数 ≥ 10 (主升浪) → 期望 +12%
  假巽日 (坏买点): 买入后 30 日 乾天数 ≤ 5 (无主升浪)  → 期望 -10%

研究:
  对比真假巽日的特征分布:
    - 卦象 (个股 m/y, 大盘 d/m/y)
    - 当下 trend / mf / sanhu
    - 前 30 日 trend min/max/slope
    - 前 30 日 mf 累计/方向
    - 前 30 日 sanhu 累计/方向
    - 前 30 日 d_gua 频率

输出:
  - 真假巽日特征差异 Top 10
  - 各 单项避雷条件 lift (vs 巽日 baseline)
  - 多项组合避雷 lift
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
ZSL_THRESH = 10  # 主升浪定义
FAKE_THRESH = 5  # 假巽日定义 (乾≤5)

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

    # === 扫所有巽日 + 评估 ===
    print(f'\n=== 扫描巽日 + 30 日评估 + 提取特征 ===')
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

            ev = {
                'date': date_arr[s + i],
                'n_qian': int(n_qian),
                'ret_30': ret_30,
                # 当下卦象
                'stk_m': stk_m[s + i], 'stk_y': stk_y[s + i],
                'mkt_d': mkt_d[s + i], 'mkt_m': mkt_m[s + i], 'mkt_y': mkt_y[s + i],
                # 当下数值
                'trend': td[i], 'mf': mf[i], 'sanhu': sanhu[i],
                # 前 30 日统计
                'trend_min': td[win_lo:i+1].min(),
                'trend_max': td[win_lo:i+1].max(),
                'trend_mean': td[win_lo:i+1].mean(),
                'trend_slope_30d': td[i] - td[win_lo],
                'trend_slope_5d': td[i] - td[max(i-4, win_lo)],
                'mf_30d_mean': mf[win_lo:i+1].mean(),
                'mf_30d_sum': mf[win_lo:i+1].sum(),
                'mf_5d_mean': mf[max(i-4,win_lo):i+1].mean(),
                'sanhu_30d_mean': sanhu[win_lo:i+1].mean(),
                'sanhu_30d_min': sanhu[win_lo:i+1].min(),
                'sanhu_5d_mean': sanhu[max(i-4,win_lo):i+1].mean(),
                # 前 30 日卦象频率
                **{f'pct_d_{g}': (gua[win_lo:i+1] == g).sum() / LOOKBACK for g in GUAS},
            }
            events.append(ev)

    print(f'  事件: {len(events):,}, {time.time()-t1:.1f}s')

    df_e = pd.DataFrame(events)
    # 分类
    df_e['kind'] = 'mid'
    df_e.loc[df_e['n_qian'] >= ZSL_THRESH, 'kind'] = 'real'
    df_e.loc[df_e['n_qian'] <= FAKE_THRESH, 'kind'] = 'fake'

    n_real = (df_e['kind'] == 'real').sum()
    n_fake = (df_e['kind'] == 'fake').sum()
    n_mid = (df_e['kind'] == 'mid').sum()
    print(f'\n## 巽日分类:')
    print(f'  真巽 (乾≥{ZSL_THRESH}): {n_real:,} ({n_real/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="real"]["ret_30"].mean():+.2f}%')
    print(f'  假巽 (乾≤{FAKE_THRESH}): {n_fake:,} ({n_fake/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="fake"]["ret_30"].mean():+.2f}%')
    print(f'  中性 (5<乾<10):  {n_mid:,} ({n_mid/len(df_e)*100:.1f}%) 期望 {df_e[df_e["kind"]=="mid"]["ret_30"].mean():+.2f}%')

    df_real = df_e[df_e['kind'] == 'real']
    df_fake = df_e[df_e['kind'] == 'fake']

    # === 1. 数值特征对比 真 vs 假 ===
    print(f'\n## 1. 数值特征 真巽 vs 假巽')
    print(f'  {"特征":<22} {"真中位":>9} {"假中位":>9} {"差(假-真)":>10}')
    print('  ' + '-' * 60)
    num_cols = ['trend', 'mf', 'sanhu',
                'trend_min', 'trend_max', 'trend_mean', 'trend_slope_30d', 'trend_slope_5d',
                'mf_30d_mean', 'mf_30d_sum', 'mf_5d_mean',
                'sanhu_30d_mean', 'sanhu_30d_min', 'sanhu_5d_mean']
    diffs = []
    for c in num_cols:
        rm = df_real[c].median(); fm = df_fake[c].median()
        diff = fm - rm
        diffs.append((c, rm, fm, diff))
    diffs.sort(key=lambda x: -abs(x[3]))
    for c, rm, fm, diff in diffs:
        print(f'  {c:<22} {rm:>+8.2f} {fm:>+8.2f} {diff:>+8.2f}')

    # === 2. 卦象对比 真 vs 假 ===
    print(f'\n## 2. 当下卦象 真巽 vs 假巽 (假巽 lift = 假% - 真%, 正值表示假巽偏好)')
    for col, label in [('stk_m', '个股 m_gua'), ('stk_y', '个股 y_gua'),
                        ('mkt_d', '大盘 d_gua'), ('mkt_m', '大盘 m_gua'), ('mkt_y', '大盘 y_gua')]:
        print(f'\n  {label}:')
        print(f'    {"卦":<6} {"真%":>7} {"假%":>7} {"假-真":>7}')
        for g_v in GUAS:
            r_pct = (df_real[col] == g_v).mean() * 100
            f_pct = (df_fake[col] == g_v).mean() * 100
            diff = f_pct - r_pct
            mark = '⚠' if diff >= 4 else ('★' if diff <= -4 else '')
            print(f'    {g_v}{GUA_NAMES[g_v]:<3} {r_pct:>6.1f}% {f_pct:>6.1f}% {diff:>+5.1f}  {mark}')

    # === 3. 前 30 日 d_gua 频率 真 vs 假 ===
    print(f'\n## 3. 前 30 日 个股 d_gua 频率 真巽 vs 假巽')
    print(f'  {"卦":<6} {"真均%":>7} {"假均%":>7} {"假-真":>7}')
    for g_v in GUAS:
        c = f'pct_d_{g_v}'
        r_m = df_real[c].mean() * 100
        f_m = df_fake[c].mean() * 100
        diff = f_m - r_m
        mark = '⚠' if diff >= 3 else ('★' if diff <= -3 else '')
        print(f'  {g_v}{GUA_NAMES[g_v]:<3} {r_m:>6.1f}% {f_m:>6.1f}% {diff:>+5.1f}  {mark}')

    # === 4. 单项避雷 ablation ===
    print(f'\n## 4. 单项避雷条件 (在巽日基础上 排除满足条件的, 看 lift)')
    base_ret = df_e['ret_30'].mean()
    base_zsl = (df_e['n_qian'] >= ZSL_THRESH).mean() * 100
    print(f'  baseline 巽日: 期望 {base_ret:+.2f}%, 主升浪率 {base_zsl:.1f}% ({len(df_e):,} 事件)')
    print()
    print(f'  {"排除条件":<35} {"剩 n":>7} {"剩%":>5} {"期望":>7} {"vs巽":>6} {"主升率":>7} {"主升lift":>8}')
    print('  ' + '-' * 80)

    avoid_conds = [
        ('mkt_y == 011 巽 (大盘年熊)', df_e['mkt_y'] == '011'),
        ('mkt_d == 110 兑', df_e['mkt_d'] == '110'),
        ('stk_m == 101 离', df_e['stk_m'] == '101'),
        ('mkt_m == 101 离', df_e['mkt_m'] == '101'),
        ('mf < 0 (主力线<0)', df_e['mf'] < 0),
        ('sanhu_5d > 0 (散户已追)', df_e['sanhu_5d_mean'] > 0),
        ('trend > 70 (位置高)', df_e['trend'] > 70),
        ('trend_slope_5d < 0 (5日下降)', df_e['trend_slope_5d'] < 0),
        ('mf_30d_mean > 0 (主力 30 日已正)', df_e['mf_30d_mean'] > 0),
        ('mf_30d_sum < -100 (主力 30d 累计负)', df_e['mf_30d_sum'] < -100),
        ('sanhu_30d_min > -50 (散户从未深-)', df_e['sanhu_30d_min'] > -50),
    ]

    for label, mask in avoid_conds:
        keep = df_e[~mask]
        if len(keep) < 1000: continue
        ret = keep['ret_30'].mean()
        zsl = (keep['n_qian'] >= ZSL_THRESH).mean() * 100
        lift = ret - base_ret
        zsl_lift = zsl - base_zsl
        mark = '✅' if lift >= 0.3 else ('❌' if lift <= -0.3 else '○')
        print(f'  {label:<35} {len(keep):>7,} {len(keep)/len(df_e)*100:>4.0f}% '
              f'{ret:>+6.2f}% {lift:>+5.2f} {zsl:>5.1f}% {zsl_lift:>+6.1f}  {mark}')

    # === 5. 多项避雷组合 ===
    print(f'\n## 5. 多项避雷组合')

    # 组合 A: 较温和 (74%)
    avoid_A = (
        (df_e['mkt_y'] == '011') |
        (df_e['mkt_d'] == '110') |
        (df_e['stk_m'] == '101') |
        (df_e['trend_slope_5d'] < 0)
    )
    keep_A = df_e[~avoid_A]
    print(f'\n  组合A (温和): 大盘y=011 OR 大盘d=110 OR 个股m=101 OR trend_5d_slope<0')
    print(f'    剩 {len(keep_A):,} ({len(keep_A)/len(df_e)*100:.0f}%)')
    print(f'    期望: {keep_A["ret_30"].mean():+.2f}% (lift {keep_A["ret_30"].mean()-base_ret:+.2f})')
    print(f'    主升率: {(keep_A["n_qian"]>=ZSL_THRESH).mean()*100:.1f}% (lift {(keep_A["n_qian"]>=ZSL_THRESH).mean()*100-base_zsl:+.1f})')

    # 组合 B: 加上 mf_30d_mean>0 (强核心避雷)
    avoid_B = avoid_A | (df_e['mf_30d_mean'] > 0)
    keep_B = df_e[~avoid_B]
    print(f'\n  组合B (强): A + mf_30d_mean>0 (主力月均已正, 洗盘未足)')
    print(f'    剩 {len(keep_B):,} ({len(keep_B)/len(df_e)*100:.0f}%)')
    print(f'    期望: {keep_B["ret_30"].mean():+.2f}% (lift {keep_B["ret_30"].mean()-base_ret:+.2f})')
    print(f'    主升率: {(keep_B["n_qian"]>=ZSL_THRESH).mean()*100:.1f}% (lift {(keep_B["n_qian"]>=ZSL_THRESH).mean()*100-base_zsl:+.1f})')

    # 组合 C: 加上大盘 m_gua=101 离 + 大盘 y_gua=011 巽
    avoid_C = avoid_B | (df_e['mkt_m'] == '101') | (df_e['stk_y'] == '011')
    keep_C = df_e[~avoid_C]
    print(f'\n  组合C (最严): B + 大盘m=101 OR 个股y=011')
    print(f'    剩 {len(keep_C):,} ({len(keep_C)/len(df_e)*100:.0f}%)')
    print(f'    期望: {keep_C["ret_30"].mean():+.2f}% (lift {keep_C["ret_30"].mean()-base_ret:+.2f})')
    print(f'    主升率: {(keep_C["n_qian"]>=ZSL_THRESH).mean()*100:.1f}% (lift {(keep_C["n_qian"]>=ZSL_THRESH).mean()*100-base_zsl:+.1f})')

    # walk-forward 各组合
    print(f'\n## 6. 7 段 walk-forward 各组合 (期望 30d %)')
    df_e['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_e.loc[(df_e['date'] >= ws) & (df_e['date'] < we), 'seg'] = w_label

    for label, avoid_mask in [('A', avoid_A), ('B', avoid_B), ('C', avoid_C)]:
        keep = df_e[~avoid_mask]
        print(f'\n  组合 {label}:  ', end='')
        n_pass = 0
        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 100 or len(seg_k) < 100: continue
            b_ret = seg_b['ret_30'].mean(); k_ret = seg_k['ret_30'].mean()
            lift = k_ret - b_ret
            mark = '✅' if lift > 0.5 else ('❌' if lift < -0.5 else '○')
            if lift > 0.5: n_pass += 1
            print(f'{w[0][:6]} {lift:>+5.2f}{mark}  ', end='')
        print(f'  ({n_pass}/7)')


if __name__ == '__main__':
    main()
