# -*- coding: utf-8 -*-
"""Step 17b — 同口径对比: 任意巽卦买入 vs 任意指纹买入

baseline: 任何一天 d_gua=011 → 买, 30 日评估
condition: 任何一天 d_gua=011 + mf>0 + sanhu_5d<-20 + slope>5 → 买, 30 日评估

不再有"入池"概念, 同样的"巽日"集合, 加 vs 不加指纹

输出:
  - 全市场 巽日 vs 指纹日 的对比
  - 各段 lift
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
                        columns=['date', 'code', 'd_trend', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'd_gua']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    gua_arr = df['d_gua'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 扫描 巽日 + 指纹日 ===')
    t1 = time.time()
    base_events = []  # 任意巽日
    fp_events = []    # 加指纹
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = gua_arr[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(td)

        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue

            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = cl[i+EVAL_WIN] / cl[i] - 1
            max_ret_30 = cl[i:i+EVAL_WIN+1].max() / cl[i] - 1

            ev = {
                'buy_date': date_arr[s + i],
                'n_qian': int(n_qian),
                'ret_30': ret_30 * 100,
                'max_ret_30': max_ret_30 * 100,
                'is_zsl': n_qian >= 10,
            }
            base_events.append(ev)

            # 加指纹 filter
            if mf[i] <= 0: continue
            sanhu_5d = sanhu[max(i-4, 0):i+1].mean()
            if sanhu_5d >= -20: continue
            trend_5d = td[i] - td[max(i-4, 0)]
            if trend_5d <= 5: continue

            fp_events.append(ev)

    print(f'  巽日: {len(base_events):,}, 指纹日: {len(fp_events):,}, {time.time()-t1:.1f}s')
    print(f'  指纹/巽日: {len(fp_events)/len(base_events)*100:.1f}%')

    df_b = pd.DataFrame(base_events)
    df_f = pd.DataFrame(fp_events)
    df_b['seg'] = ''; df_f['seg'] = ''
    for w in WINDOWS:
        df_b.loc[(df_b['buy_date'] >= w[1]) & (df_b['buy_date'] < w[2]), 'seg'] = w[0]
        df_f.loc[(df_f['buy_date'] >= w[1]) & (df_f['buy_date'] < w[2]), 'seg'] = w[0]
    df_b = df_b[df_b['seg'] != '']; df_f = df_f[df_f['seg'] != '']

    # === 整体对比 ===
    print(f'\n## 1. 整体对比')
    print(f'  {"指标":<20} {"巽日 baseline":>15} {"指纹买入":>12} {"lift":>8}')
    print('  ' + '-' * 60)
    for col, label, fmt in [
        ('ret_30', '期望 30d %', '+.2f'),
        ('max_ret_30', '期望最高 %', '+.2f'),
    ]:
        b = df_b[col].mean(); f = df_f[col].mean()
        print(f'  {label:<20} {b:>14.2f}% {f:>11.2f}% {f-b:>+7.2f}')
    win_b = (df_b['ret_30'] > 0).mean() * 100
    win_f = (df_f['ret_30'] > 0).mean() * 100
    print(f'  {"胜率 %":<20} {win_b:>14.1f}% {win_f:>11.1f}% {win_f-win_b:>+7.1f}')
    zsl_b = df_b['is_zsl'].mean() * 100
    zsl_f = df_f['is_zsl'].mean() * 100
    print(f'  {"主升浪率 %":<20} {zsl_b:>14.1f}% {zsl_f:>11.1f}% {zsl_f-zsl_b:>+7.1f}')
    qm_b = df_b['n_qian'].mean(); qm_f = df_f['n_qian'].mean()
    print(f'  {"乾均日":<20} {qm_b:>14.2f}  {qm_f:>11.2f}  {qm_f-qm_b:>+7.2f}')

    # === 7 段 walk-forward ===
    print(f'\n## 2. walk-forward 7 段对比 (期望 30d %)')
    print(f'  {"段":<14} {"巽日":>8} {"指纹":>8} {"lift":>8}', end='')
    print(f' {"巽胜率":>7} {"指纹胜率":>9} {"巽主升":>7} {"指纹主升":>8}')
    print('  ' + '-' * 100)
    n_pass = 0
    for w in WINDOWS:
        b = df_b[df_b['seg'] == w[0]]
        f = df_f[df_f['seg'] == w[0]]
        if len(b) < 30 or len(f) < 30: continue
        b_ret = b['ret_30'].mean(); f_ret = f['ret_30'].mean()
        b_win = (b['ret_30'] > 0).mean() * 100; f_win = (f['ret_30'] > 0).mean() * 100
        b_zsl = b['is_zsl'].mean() * 100; f_zsl = f['is_zsl'].mean() * 100
        lift = f_ret - b_ret
        mark = '✅' if lift >= 1 else ('❌' if lift <= -1 else '○')
        if lift >= 1: n_pass += 1
        print(f'  {w[0]:<14} {b_ret:>+7.2f}% {f_ret:>+7.2f}% {lift:>+7.2f} '
              f'{b_win:>5.1f}%/{f_win:>5.1f}% {b_zsl:>5.1f}%/{f_zsl:>5.1f}% {mark}')
    print(f'\n  → {n_pass}/7 段 lift ≥ +1%')

    # === 3. 单项 ablation ===
    print(f'\n## 3. 单项指纹 ablation (各项独立 lift, vs 巽日 baseline)')
    print(f'  {"条件":<28} {"事件 n":>8} {"期望":>7} {"vs巽":>7} {"主升率":>7}')
    print('  ' + '-' * 65)
    base_ret = df_b['ret_30'].mean()
    base_zsl = df_b['is_zsl'].mean() * 100

    print(f'  {"baseline 巽日":<28} {len(df_b):>8,} {base_ret:>+6.2f}% {"--":>7} {base_zsl:>5.1f}%')

    # 重扫单项 (单 mf>0)
    def rescan(filter_fn, label):
        evs = []
        for ci in range(len(code_starts)):
            s = code_starts[ci]; e = code_ends[ci]
            if e - s < LOOKBACK + EVAL_WIN + 5: continue
            td = trend_arr[s:e]; cl = close_arr[s:e]; gua = gua_arr[s:e]
            mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
            n = len(td)
            for i in range(LOOKBACK, n - EVAL_WIN):
                if gua[i] != '011': continue
                if not filter_fn(td, mf, sanhu, i): continue
                seg_gua = gua[i:i+EVAL_WIN]
                n_qian = (seg_gua == '111').sum()
                ret_30 = cl[i+EVAL_WIN] / cl[i] - 1
                evs.append({'ret': ret_30 * 100, 'n_qian': int(n_qian)})
        d = pd.DataFrame(evs)
        if len(d) == 0:
            print(f'  {label:<28} 无事件')
            return
        ret_m = d['ret'].mean(); zsl_m = (d['n_qian'] >= 10).mean() * 100
        print(f'  {label:<28} {len(d):>8,} {ret_m:>+6.2f}% {ret_m - base_ret:>+6.2f} {zsl_m:>5.1f}%')

    rescan(lambda td, mf, sh, i: mf[i] > 0, '+ mf > 0')
    rescan(lambda td, mf, sh, i: sh[max(i-4,0):i+1].mean() < -20, '+ sanhu_5d < -20')
    rescan(lambda td, mf, sh, i: td[i] - td[max(i-4,0)] > 5, '+ trend_5d_slope > 5')


if __name__ == '__main__':
    main()
