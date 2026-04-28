# -*- coding: utf-8 -*-
"""阶段 3+4: 巽 regime case study — 反向 regime 里什么股能产主升浪?

样本设计:
  treatment = 巽 regime 内, d_gua=111 连续 ≥10 日的起点 (主升浪)
  control   = 同一 regime 内, 任意巽日 (基础事件) 但未达主升浪

特征提取 day0-1 (前一日):
  卦象 6 个 (个股+大盘 d/m/y) + 数值 (trend / mf / sanhu 当下/5d) + 前 30 日卦象频率

避雷扫描:
  在巽 regime 巽日上扫单维条件, 段稳定性: ≥w2 + w5 都 fail 则记
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

WINDOWS = [
    ('w2_2019', '2019-01-01', '2020-01-01'),
    ('w5_2022', '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]


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

    # ===== 1. 提取主升浪 (treatment) + 同期巽日对照 (control) =====
    print(f'\n=== 提取 treatment (主升浪起点) + control (巽日) ===')
    treatment = []
    control = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue

        # 找主升浪起点
        gua = stk_d_arr[s:e]
        n = len(gua)
        i = 0
        zsl_starts = set()
        while i < n:
            if gua[i] != '111':
                i += 1; continue
            j = i
            while j < n and gua[j] == '111':
                j += 1
            length = j - i
            gi = s + i
            if length >= QIAN_RUN and mkt_y_arr[gi] == REGIME_Y and i - 1 >= LOOKBACK - 1:
                pi = s + i - 1
                feat = extract_features(pi, s, trend_arr, mf_arr, sanhu_arr,
                                        stk_d_arr, stk_m_arr, stk_y_arr,
                                        mkt_d_arr, mkt_m_arr, mkt_y_arr)
                feat['date'] = date_arr[pi]
                feat['code'] = code_arr[pi]
                feat['run_length'] = length
                # 30 日收益
                if i + 30 < n:
                    feat['ret_30'] = (close_arr[s+i+30] / close_arr[s+i] - 1) * 100
                else:
                    feat['ret_30'] = float('nan')
                treatment.append(feat)
                zsl_starts.add(i)
            i = j

        # 同期 control = 巽日 (但不是主升浪起点 day0-1)
        for k in range(LOOKBACK, n - EVAL_WIN):
            gi = s + k
            if mkt_y_arr[gi] != REGIME_Y: continue
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            # 是否次日是主升浪起点? 排除
            if (k + 1) in zsl_starts:
                continue
            cl = close_arr[s:e]
            seg_gua = gua[k:k+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[k+EVAL_WIN] / cl[k] - 1) * 100
            feat = extract_features(gi, s, trend_arr, mf_arr, sanhu_arr,
                                    stk_d_arr, stk_m_arr, stk_y_arr,
                                    mkt_d_arr, mkt_m_arr, mkt_y_arr)
            feat['date'] = date_arr[gi]
            feat['code'] = code_arr[gi]
            feat['n_qian'] = int(n_qian)
            feat['ret_30'] = ret_30
            feat['is_zsl'] = n_qian >= QIAN_RUN
            control.append(feat)

    df_t = pd.DataFrame(treatment)
    df_c = pd.DataFrame(control)
    print(f'  treatment (主升浪起点): {len(df_t):,}')
    print(f'  control (基础巽日): {len(df_c):,}')
    if len(df_t) > 0:
        print(f'  主升浪 30 日均收益: {df_t["ret_30"].mean():+.2f}%')
    if len(df_c) > 0:
        print(f'  对照 30 日均收益: {df_c["ret_30"].mean():+.2f}%')

    # ===== 2. 卦象指纹对比 =====
    if len(df_t) > 0 and len(df_c) > 0:
        print(f'\n## 卦象指纹 (主升浪占比 - 对照占比, 按绝对值排序)')
        rows = []
        for col in ['stk_m', 'stk_y', 'mkt_d', 'mkt_m']:
            for state in GUAS:
                if col not in df_t.columns or col not in df_c.columns: continue
                t_pct = (df_t[col] == state).mean() * 100
                c_pct = (df_c[col] == state).mean() * 100
                lift = t_pct - c_pct
                rows.append((col, state, t_pct, c_pct, lift))
        rows.sort(key=lambda x: abs(x[4]), reverse=True)
        short = {'stk_m': '股m', 'stk_y': '股y', 'mkt_d': '大d', 'mkt_m': '大m'}
        print(f'  {"卦":<14} {"主升%":>7} {"对照%":>7} {"lift":>6}')
        for col, state, t, c, lift in rows[:15]:
            if abs(lift) < 1: continue
            label = f'{short[col]}={state}{GUA_NAMES[state]}'
            mark = '★' if lift > 0 else '✗'
            print(f'  {mark} {label:<12} {t:>6.1f} {c:>6.1f} {lift:>+5.1f}')

        print(f'\n## 数值指纹 (中位数对比)')
        print(f'  {"特征":<22} {"主升中位":>10} {"对照中位":>10} {"差":>8}')
        num_cols = ['trend', 'trend_5d_slope', 'mf', 'mf_5d_mean',
                    'sanhu', 'sanhu_5d_mean', 'mf_30d_min', 'sanhu_30d_min']
        for col in num_cols:
            if col not in df_t.columns: continue
            t_med = df_t[col].median()
            c_med = df_c[col].median()
            diff = t_med - c_med
            print(f'  {col:<22} {t_med:>+9.2f} {c_med:>+9.2f} {diff:>+7.2f}')

    # ===== 3. 巽 regime 巽日反向避雷扫描 =====
    print(f'\n=== 阶段 4: 巽 regime 巽日反向避雷 (跨 w2 + w5 + w6) ===')
    df_e = df_c.copy()  # control 就是巽日全集
    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['date'] >= w[1]) & (df_e['date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  事件: {len(df_e):,}')

    seg_baselines = {}
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        seg_baselines[w[0]] = seg['ret_30'].mean() if len(seg) > 0 else 0
        print(f'  baseline {w[0]:<12} n={len(seg):>5} 期望 {seg_baselines[w[0]]:>+5.2f}%')

    # 单维候选
    avoid_strong = []
    avoid_weak = []
    good_strong = []
    good_weak = []

    for col, label_short in [('mkt_d', '大d'), ('mkt_m', '大m'),
                              ('stk_y', '股y'), ('stk_m', '股m')]:
        for state in GUAS:
            sub = df_e[df_e[col] == state]
            if len(sub) < 200: continue
            n_full = len(sub); ret_full = sub['ret_30'].mean()
            label = f'{label_short}={state}{GUA_NAMES[state]}'

            seg_results = {}
            for w in WINDOWS:
                seg = sub[sub['seg'] == w[0]]
                if len(seg) < 50:
                    seg_results[w[0]] = None
                    continue
                ret = seg['ret_30'].mean()
                lift = ret - seg_baselines[w[0]]
                seg_results[w[0]] = (len(seg), ret, lift)

            n_pass = sum(1 for r in seg_results.values() if r and r[2] >= 1.0)
            n_fail = sum(1 for r in seg_results.values() if r and r[2] <= -1.0)
            n_valid = sum(1 for r in seg_results.values() if r is not None)

            if n_valid >= 2 and n_fail >= 2 and n_pass == 0:
                avoid_strong.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid == 3 and n_fail >= 1 and n_pass <= 1 and n_fail >= n_pass:
                avoid_weak.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid >= 2 and n_pass >= 2 and n_fail == 0:
                good_strong.append((col, state, label, n_full, ret_full, seg_results))
            elif n_valid >= 2 and n_pass >= 1 and n_fail == 0:
                good_weak.append((col, state, label, n_full, ret_full, seg_results))

    def fmt_seg(seg_results):
        s = []
        for w in WINDOWS:
            r = seg_results.get(w[0])
            if r is None: s.append(f'{w[0]}=NA')
            else: s.append(f'{w[0]}=n{r[0]} lift{r[2]:+.2f}')
        return ' | '.join(s)

    print(f'\n## ★★ 强避雷 (≥2 段 fail, 0 段 pass)')
    if not avoid_strong: print('  无')
    for col, state, label, n, ret, seg in avoid_strong:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    print(f'\n## ★★ 强好规律 (≥2 段 pass, 0 段 fail)')
    if not good_strong: print('  无')
    for col, state, label, n, ret, seg in good_strong:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    print(f'\n## ★ 弱好规律 (1+ 段 pass, 0 段 fail)')
    if not good_weak: print('  无')
    for col, state, label, n, ret, seg in good_weak[:8]:
        print(f'  {label:<14} 全n {n:,} 全期望 {ret:+.2f}%')
        print(f'    {fmt_seg(seg)}')

    # 验证强避雷效果
    if avoid_strong:
        print(f'\n## 强避雷 union 效果')
        avoid_mask = pd.Series(False, index=df_e.index)
        for col, state, _, _, _, _ in avoid_strong:
            avoid_mask = avoid_mask | (df_e[col] == state)
        keep = df_e[~avoid_mask]
        base = df_e['ret_30'].mean()
        kept_ret = keep['ret_30'].mean()
        kept_zsl = (keep['n_qian'] >= QIAN_RUN).mean() * 100 if 'n_qian' in keep.columns else 0
        print(f'  剩 {len(keep):,} ({len(keep)/len(df_e)*100:.0f}%)')
        print(f'  期望: {kept_ret:+.2f}% (vs {base:+.2f}%, lift {kept_ret-base:+.2f})')
        print(f'  主升率: {kept_zsl:.1f}%')

        for w in WINDOWS:
            seg_b = df_e[df_e['seg'] == w[0]]
            seg_k = keep[keep['seg'] == w[0]]
            if len(seg_b) < 50: continue
            b = seg_b['ret_30'].mean(); k = seg_k['ret_30'].mean() if len(seg_k) > 0 else float('nan')
            diff = k - b
            mark = '✅' if diff > 0.5 else ('❌' if diff < -0.5 else '○')
            print(f'    {w[0]:<14} 全 {b:>+5.2f}%, 避雷后 {k:>+5.2f}% ({len(seg_k):>4}), lift {diff:>+5.2f} {mark}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


def extract_features(idx, s, trend_arr, mf_arr, sanhu_arr,
                     stk_d_arr, stk_m_arr, stk_y_arr,
                     mkt_d_arr, mkt_m_arr, mkt_y_arr):
    feat = {
        'stk_d': stk_d_arr[idx], 'stk_m': stk_m_arr[idx], 'stk_y': stk_y_arr[idx],
        'mkt_d': mkt_d_arr[idx], 'mkt_m': mkt_m_arr[idx], 'mkt_y': mkt_y_arr[idx],
        'trend': float(trend_arr[idx]),
        'mf': float(mf_arr[idx]),
        'sanhu': float(sanhu_arr[idx]),
    }
    if idx - 5 >= s:
        feat['trend_5d_slope'] = float(trend_arr[idx] - trend_arr[idx-5])
        feat['mf_5d_mean'] = float(np.nanmean(mf_arr[idx-5:idx+1]))
        feat['sanhu_5d_mean'] = float(np.nanmean(sanhu_arr[idx-5:idx+1]))
    else:
        feat['trend_5d_slope'] = float('nan')
        feat['mf_5d_mean'] = float('nan')
        feat['sanhu_5d_mean'] = float('nan')
    if idx - 30 >= s:
        feat['mf_30d_min'] = float(np.nanmin(mf_arr[idx-30:idx+1]))
        feat['sanhu_30d_min'] = float(np.nanmin(sanhu_arr[idx-30:idx+1]))
    else:
        feat['mf_30d_min'] = float('nan')
        feat['sanhu_30d_min'] = float('nan')
    return feat


if __name__ == '__main__':
    main()
