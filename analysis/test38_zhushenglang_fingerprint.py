# -*- coding: utf-8 -*-
"""Step 15 — 主升浪前夜特征指纹

定义:
  主升浪事件 = 个股 d_gua='111' 连续 ≥10 日的段, 取段起点 day0
  前夜窗口 = [day0-30, day0-1] 共 30 个交易日

提取 day0-1 那天的特征 (避免主升浪本身泄露):
  - 卦象: stk d/m/y_gua, mkt d/m/y_gua
  - 趋势线 d_trend: 当前值 + 前30日均值/最低/起点终点斜率
  - 主力线 d_mf: 当前值 + 前30日均值/最近5日均值
  - 散户线: EMA((close - MA11)/MA11 * 480, 7) * 5, 现场算

对照组: 从所有 (code, date) 中随机抽 N 个非主升浪起点的 day0, 同方法提特征

输出:
  - 各特征在主升浪 vs 对照 的均值/中位/分位数差异
  - 统计上能区分的 Top 特征
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QIAN_RUN = 10  # 乾连续多少日算主升浪
LOOKBACK = 30  # 前夜窗口

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']


def compute_sanhu(close):
    """散户线: EMA((close - MA11)/MA11 * 480, 7) * 5"""
    s = pd.Series(close)
    ma11 = s.rolling(11, min_periods=11).mean()
    raw = (s - ma11) / ma11 * 480
    sanhu = raw.ewm(span=7, adjust=False).mean() * 5
    return sanhu.values


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_mf', 'd_gua', 'm_gua', 'y_gua'])
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
    mf_arr = df['main_force'].to_numpy().astype(np.float32)  # 主力线 (现成字段)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)   # 散户线 (现成字段)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d = df['d_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy(); stk_y = df['y_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy(); mkt_m = df['mkt_m'].to_numpy(); mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # === 找主升浪事件起点 ===
    print(f'\n=== 找主升浪事件 (d_gua=111 连续 ≥{QIAN_RUN} 日) ===')
    t1 = time.time()
    qian_starts = []  # 全局 idx
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + QIAN_RUN: continue
        gua = stk_d[s:e]
        is_qian = (gua == '111')
        # 找连续 ≥QIAN_RUN 段
        i = 0
        n = len(gua)
        while i < n:
            if is_qian[i]:
                j = i
                while j < n and is_qian[j]:
                    j += 1
                run_len = j - i
                if run_len >= QIAN_RUN and i >= LOOKBACK:
                    qian_starts.append(s + i)
                i = j
            else:
                i += 1
    qian_starts = np.array(qian_starts)
    print(f'  主升浪事件: {len(qian_starts):,}, {time.time()-t1:.1f}s')

    # === 提取主升浪前夜特征 (day0-1) ===
    def extract_features(idx, code_starts_arr):
        """对一个 day0 idx, 提取 day0-1 的特征"""
        prev = idx - 1
        if prev < 0: return None
        code_seg = np.searchsorted(code_starts_arr, idx, side='right') - 1
        s_code = code_starts_arr[code_seg]
        if prev < s_code + LOOKBACK:
            return None

        feat = {}
        feat['stk_d'] = stk_d[prev]; feat['stk_m'] = stk_m[prev]; feat['stk_y'] = stk_y[prev]
        feat['mkt_d'] = mkt_d[prev]; feat['mkt_m'] = mkt_m[prev]; feat['mkt_y'] = mkt_y[prev]
        feat['trend'] = trend_arr[prev]
        feat['mf'] = mf_arr[prev]
        feat['sanhu'] = sanhu_arr[prev]
        win_lo = prev - LOOKBACK + 1
        feat['trend_30d_mean'] = trend_arr[win_lo:prev+1].mean()
        feat['trend_30d_min'] = trend_arr[win_lo:prev+1].min()
        feat['trend_30d_max'] = trend_arr[win_lo:prev+1].max()
        feat['trend_slope'] = trend_arr[prev] - trend_arr[win_lo]
        feat['trend_5d_slope'] = trend_arr[prev] - trend_arr[max(prev-4, win_lo)]
        feat['mf_30d_mean'] = mf_arr[win_lo:prev+1].mean()
        feat['mf_5d_mean'] = mf_arr[prev-4:prev+1].mean()
        feat['sanhu_30d_mean'] = sanhu_arr[win_lo:prev+1].mean()
        feat['sanhu_30d_min'] = sanhu_arr[win_lo:prev+1].min()
        feat['sanhu_5d_mean'] = sanhu_arr[prev-4:prev+1].mean()
        # 前 30 日 d_gua 频率
        prev_guas = stk_d[win_lo:prev+1]
        for g_v in GUAS:
            feat[f'pct_d_{g_v}'] = (prev_guas == g_v).sum() / LOOKBACK
        # 前 30 日 mkt_d 频率
        prev_mkt_d = mkt_d[win_lo:prev+1]
        for g_v in GUAS:
            feat[f'pct_mkt_d_{g_v}'] = (prev_mkt_d == g_v).sum() / LOOKBACK
        return feat

    # === 主升浪事件 提特征 ===
    print(f'\n=== 提取主升浪事件特征 ===')
    t3 = time.time()
    qian_features = []
    for idx in qian_starts:
        feat = extract_features(idx, code_starts)
        if feat is not None:
            qian_features.append(feat)
    df_q = pd.DataFrame(qian_features)
    print(f'  主升浪样本: {len(df_q):,}, {time.time()-t3:.1f}s')

    # === 对照组: 随机抽样 ===
    print(f'\n=== 对照组随机抽样 ===')
    t4 = time.time()
    rng = np.random.RandomState(42)
    qian_set = set(qian_starts.tolist())
    n_total = len(df)
    n_sample = len(df_q) * 5
    ctrl_features = []
    attempts = 0
    while len(ctrl_features) < n_sample and attempts < n_sample * 30:
        attempts += 1
        idx = rng.randint(0, n_total)
        if idx in qian_set: continue
        feat = extract_features(idx, code_starts)
        if feat is not None:
            ctrl_features.append(feat)
    df_ctrl = pd.DataFrame(ctrl_features)
    print(f'  对照样本: {len(df_ctrl):,}, {time.time()-t4:.1f}s')

    # === 数值特征对比 ===
    print(f'\n## 1. 数值特征对比 (主升浪 day0-1 vs 对照)')
    print(f'  {"特征":<22} {"主升浪 mean":>12} {"对照 mean":>10} {"差":>8} {"主中位":>8} {"对中位":>8}')
    print('  ' + '-' * 75)
    num_cols = ['trend', 'mf', 'sanhu',
                'trend_30d_mean', 'trend_30d_min', 'trend_30d_max', 'trend_slope', 'trend_5d_slope',
                'mf_30d_mean', 'mf_5d_mean', 'sanhu_30d_mean', 'sanhu_30d_min', 'sanhu_5d_mean']
    for c in num_cols:
        if c not in df_q.columns: continue
        q_mean = df_q[c].mean(); ctrl_mean = df_ctrl[c].mean()
        q_med = df_q[c].median(); ctrl_med = df_ctrl[c].median()
        diff = q_mean - ctrl_mean
        print(f'  {c:<22} {q_mean:>11.2f} {ctrl_mean:>9.2f} {diff:>+7.2f} {q_med:>7.2f} {ctrl_med:>7.2f}')

    # === 分位数对比 ===
    print(f'\n## 2. trend / sanhu / mf 分位数对比')
    for c in ['trend', 'sanhu', 'mf']:
        if c not in df_q.columns: continue
        print(f'\n  {c}:')
        for p in [10, 25, 50, 75, 90]:
            qv = df_q[c].quantile(p/100); cv = df_ctrl[c].quantile(p/100)
            print(f'    p{p}: 主升 {qv:>8.2f}  对照 {cv:>8.2f}  差 {qv-cv:>+7.2f}')

    # === 卦象频率对比 ===
    print(f'\n## 3. 当下 (day0-1) 卦象 占比 主升浪 vs 对照')
    for col, label in [('stk_d', '个股 d_gua'), ('stk_m', '个股 m_gua'), ('stk_y', '个股 y_gua'),
                        ('mkt_d', '大盘 d_gua'), ('mkt_m', '大盘 m_gua'), ('mkt_y', '大盘 y_gua')]:
        print(f'\n  {label}:')
        print(f'    {"卦":<6} {"主升 %":>8} {"对照 %":>8} {"lift":>6}')
        for g_v in GUAS:
            q_pct = (df_q[col] == g_v).mean() * 100
            c_pct = (df_ctrl[col] == g_v).mean() * 100
            mark = '★' if q_pct - c_pct >= 5 else ('✗' if q_pct - c_pct <= -5 else '')
            print(f'    {g_v}{GUA_NAMES[g_v]:<3} {q_pct:>7.1f}% {c_pct:>7.1f}% {q_pct-c_pct:>+5.1f}  {mark}')

    # === 前 30 日卦象频率对比 ===
    print(f'\n## 4. 前 30 日 个股 d_gua 出现频率 主升浪 vs 对照')
    print(f'  {"卦":<6} {"主升均%":>8} {"对照均%":>8} {"lift":>6}')
    for g_v in GUAS:
        c = f'pct_d_{g_v}'
        q_m = df_q[c].mean() * 100
        c_m = df_ctrl[c].mean() * 100
        mark = '★' if q_m - c_m >= 3 else ('✗' if q_m - c_m <= -3 else '')
        print(f'  {g_v}{GUA_NAMES[g_v]:<3} {q_m:>7.1f}% {c_m:>7.1f}% {q_m-c_m:>+5.1f}  {mark}')


if __name__ == '__main__':
    main()
