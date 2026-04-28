# -*- coding: utf-8 -*-
"""阶段 3b: 艮 regime (2021-05-26 ~ 2021-06-16) case study 指纹

样本:
  主升浪起点 (treatment): 138 个 (艮 regime 期间 d_gua=111 连续>=10 起点)
  对照 (control): 同期艮 regime 内, d_gua=011 巽 (反转入口) 但未达主升浪

day0-1 提取特征:
  - 卦象: 个股 d/m/y_gua, 大盘 d/m_gua
  - 数值: d_trend (当下/5d斜率), main_force (当下/5d均), retail (当下/5d均)
  - 前 30 日卦象频率

输出: 主升中位 vs 对照中位 排序, 给 case study 文档使用
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QIAN_RUN = 10
LOOKBACK = 30
GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']


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
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    DATE_LO, DATE_HI = '2021-05-26', '2021-06-16'

    print(f'\n=== 扫艮 regime {DATE_LO}~{DATE_HI} ===')
    treatment = []  # 主升浪起点
    control = []    # 同期艮 regime 内, 巽日 (反转入口) 但未达主升浪
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + 5: continue
        gua = stk_d_arr[s:e]
        n = len(gua)

        # 找主升浪起点 (treatment)
        i = 0
        while i < n:
            if gua[i] != '111':
                i += 1; continue
            j = i
            while j < n and gua[j] == '111':
                j += 1
            length = j - i
            gi = s + i
            if length >= QIAN_RUN and mkt_y_arr[gi] == '001':
                d = date_arr[gi]
                if DATE_LO <= d <= DATE_HI and i - 1 >= LOOKBACK - 1:
                    # day0-1 (巽日入口) 取特征
                    pi = s + i - 1
                    feat = extract_features(pi, s, trend_arr, mf_arr, sanhu_arr,
                                            stk_d_arr, stk_m_arr, stk_y_arr,
                                            mkt_d_arr, mkt_m_arr, mkt_y_arr,
                                            close_arr, j)
                    feat['date'] = date_arr[pi]
                    feat['code'] = code_arr[pi]
                    feat['run_length'] = length
                    treatment.append(feat)
            i = j

        # 找对照 (同期艮 regime 巽日)
        for k in range(LOOKBACK, n):
            gi = s + k
            d = date_arr[gi]
            if not (DATE_LO <= d <= DATE_HI): continue
            if mkt_y_arr[gi] != '001': continue
            if stk_d_arr[gi] != '011': continue
            # 排除 = 主升浪起点 day0-1 (巽日 + 第二天进入连续乾) 的, 这种是 treatment
            # 但简单起见 control 包含所有同期巽日, 后面看效果时排除 treatment
            feat = extract_features(gi, s, trend_arr, mf_arr, sanhu_arr,
                                    stk_d_arr, stk_m_arr, stk_y_arr,
                                    mkt_d_arr, mkt_m_arr, mkt_y_arr,
                                    close_arr, gi+1)
            feat['date'] = date_arr[gi]
            feat['code'] = code_arr[gi]
            control.append(feat)

    df_t = pd.DataFrame(treatment)
    df_c = pd.DataFrame(control)
    print(f'  主升浪起点 (treatment): {len(df_t):,}')
    print(f'  对照 (同期巽日): {len(df_c):,}')

    # 排除 control 中明显是 treatment 的 (date+code 一致)
    if len(df_t) > 0 and len(df_c) > 0:
        treat_keys = set(zip(df_t['date'], df_t['code']))
        # treatment 是 day0-1 (巽日) , control 也是巽日, 但 treatment 第二天进乾连续
        # 简单: 排除完全同 date+code 的
        df_c = df_c[~df_c.apply(lambda r: (r['date'], r['code']) in treat_keys, axis=1)].copy()
        print(f'  对照排除 treatment 后: {len(df_c):,}')

    # ===== 卦象指纹对比 =====
    print(f'\n## 卦象 lift (主升浪占比 - 对照占比, 按绝对值排序)')
    print(f'  {"维度+卦":<14} {"主升%":>7} {"对照%":>7} {"lift":>7}')
    rows = []
    for col in ['stk_m', 'stk_y', 'mkt_d', 'mkt_m']:
        for state in GUAS:
            if len(df_t) == 0: continue
            t_pct = (df_t[col] == state).mean() * 100 if col in df_t.columns else 0
            c_pct = (df_c[col] == state).mean() * 100 if col in df_c.columns and len(df_c) > 0 else 0
            lift = t_pct - c_pct
            rows.append((col, state, t_pct, c_pct, lift))
    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    short = {'stk_m': '股m', 'stk_y': '股y', 'mkt_d': '大d', 'mkt_m': '大m'}
    for col, state, t, c, lift in rows[:15]:
        if abs(lift) < 1: continue
        label = f'{short[col]}={state}{GUA_NAMES[state]}'
        mark = '★' if lift > 0 else '✗'
        print(f'  {mark} {label:<14} {t:>6.1f} {c:>6.1f} {lift:>+6.1f}')

    # ===== 数值指纹 =====
    print(f'\n## 数值指纹 (中位数对比)')
    print(f'  {"特征":<22} {"主升中位":>10} {"对照中位":>10} {"差":>8}')
    num_cols = ['trend', 'trend_5d_slope', 'mf', 'mf_5d_mean',
                'sanhu', 'sanhu_5d_mean', 'mf_30d_min', 'sanhu_30d_min']
    for col in num_cols:
        if col not in df_t.columns: continue
        if len(df_t) == 0 or len(df_c) == 0: continue
        t_med = df_t[col].median()
        c_med = df_c[col].median()
        diff = t_med - c_med
        print(f'  {col:<22} {t_med:>+9.2f} {c_med:>+9.2f} {diff:>+7.2f}')

    # ===== 前 30 日卦象频率对比 =====
    print(f'\n## day0-1 前 30 日个股 d_gua 频率 (主升 vs 对照)')
    print(f'  {"卦":<10} {"主升频率%":>10} {"对照频率%":>10} {"lift":>7}')
    for state in GUAS:
        col = f'prev30_d_{state}'
        if col not in df_t.columns: continue
        t_freq = df_t[col].mean() * 100
        c_freq = df_c[col].mean() * 100
        lift = t_freq - c_freq
        label = f'{state}{GUA_NAMES[state]}'
        mark = '★' if lift > 5 else ('✗' if lift < -5 else ' ')
        print(f'  {mark} {label:<8} {t_freq:>9.1f} {c_freq:>9.1f} {lift:>+6.1f}')

    # ===== 主升浪起点上一日卦 (验证巽日入口规律是否在艮 regime 也成立) =====
    if len(df_t) > 0:
        print(f'\n## treatment day0-1 当日卦 (即主升浪开始前一日)')
        prev_cnt = Counter(df_t['stk_d'])
        for prev, cnt in prev_cnt.most_common():
            ratio = cnt / len(df_t) * 100
            label = f'{prev}{GUA_NAMES.get(prev, "?")}'
            print(f'    {label:<10} {cnt:>5} ({ratio:.1f}%)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


def extract_features(idx, s, trend_arr, mf_arr, sanhu_arr,
                     stk_d_arr, stk_m_arr, stk_y_arr,
                     mkt_d_arr, mkt_m_arr, mkt_y_arr,
                     close_arr, end_idx):
    """提取 idx 位置的特征 (idx 必须在 s+30 之后)"""
    feat = {
        'stk_d': stk_d_arr[idx], 'stk_m': stk_m_arr[idx], 'stk_y': stk_y_arr[idx],
        'mkt_d': mkt_d_arr[idx], 'mkt_m': mkt_m_arr[idx], 'mkt_y': mkt_y_arr[idx],
        'trend': trend_arr[idx],
        'mf': mf_arr[idx],
        'sanhu': sanhu_arr[idx],
    }
    # 5d slope/mean
    if idx - 5 >= s:
        feat['trend_5d_slope'] = trend_arr[idx] - trend_arr[idx-5]
        feat['mf_5d_mean'] = np.nanmean(mf_arr[idx-5:idx+1])
        feat['sanhu_5d_mean'] = np.nanmean(sanhu_arr[idx-5:idx+1])
    else:
        feat['trend_5d_slope'] = float('nan')
        feat['mf_5d_mean'] = float('nan')
        feat['sanhu_5d_mean'] = float('nan')

    # 30d min
    if idx - 30 >= s:
        feat['mf_30d_min'] = np.nanmin(mf_arr[idx-30:idx+1])
        feat['sanhu_30d_min'] = np.nanmin(sanhu_arr[idx-30:idx+1])
        # 前 30 日卦象频率
        prev_d = stk_d_arr[idx-30:idx]
        for state in GUAS:
            feat[f'prev30_d_{state}'] = (prev_d == state).mean()
    else:
        feat['mf_30d_min'] = float('nan')
        feat['sanhu_30d_min'] = float('nan')
        for state in GUAS:
            feat[f'prev30_d_{state}'] = float('nan')

    return feat


if __name__ == '__main__':
    main()
