# -*- coding: utf-8 -*-
"""Step 16 — 主升浪前夜指纹 walk-forward 验证

研究问题:
  之前发现的主升浪前夜指纹 (巽卦65% / 主力线>0 / 散户线<-20 / trend加速) 是否跨期稳定?
  还是某段切片福利?

方法:
  1. 主升浪事件按 day0 日期分到 7 段窗口
  2. 每段计算各指纹的 主升浪 vs 对照 lift
  3. 看哪些指纹 ≥5 段 lift 同向

输出:
  - 每段主升浪事件数 (是否集中)
  - 每段下各指纹 lift (主升 % - 对照 %)
  - 跨段稳定性判定
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QIAN_RUN = 10
LOOKBACK = 30

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
    stk_d = df['d_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy(); mkt_m = df['mkt_m'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 找主升浪事件
    print(f'\n=== 找主升浪事件 ===')
    qian_starts = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + QIAN_RUN: continue
        gua = stk_d[s:e]
        is_qian = (gua == '111')
        i = 0; n = len(gua)
        while i < n:
            if is_qian[i]:
                j = i
                while j < n and is_qian[j]:
                    j += 1
                if j - i >= QIAN_RUN and i >= LOOKBACK:
                    qian_starts.append(s + i)
                i = j
            else:
                i += 1
    qian_starts = np.array(qian_starts)
    print(f'  事件: {len(qian_starts):,}')

    def extract_features(idx):
        prev = idx - 1
        if prev < 0: return None
        code_seg = np.searchsorted(code_starts, idx, side='right') - 1
        s_code = code_starts[code_seg]
        if prev < s_code + LOOKBACK: return None
        win_lo = prev - LOOKBACK + 1
        feat = {
            'date': date_arr[idx],
            'stk_d': stk_d[prev], 'stk_m': stk_m[prev],
            'mkt_d': mkt_d[prev], 'mkt_m': mkt_m[prev],
            'trend': trend_arr[prev],
            'mf': mf_arr[prev],
            'sanhu': sanhu_arr[prev],
            'trend_5d_slope': trend_arr[prev] - trend_arr[max(prev-4, win_lo)],
            'sanhu_5d_mean': sanhu_arr[prev-4:prev+1].mean(),
            'mf_5d_mean': mf_arr[prev-4:prev+1].mean(),
        }
        return feat

    # 主升浪特征
    print(f'\n=== 提取主升浪特征 ===')
    qian_features = [extract_features(idx) for idx in qian_starts]
    qian_features = [f for f in qian_features if f is not None]
    df_q = pd.DataFrame(qian_features)

    # 对照组
    print(f'\n=== 对照组 ===')
    rng = np.random.RandomState(42)
    qian_set = set(qian_starts.tolist())
    n_total = len(df)
    n_sample = len(df_q) * 5
    ctrl_features = []
    while len(ctrl_features) < n_sample:
        idx = rng.randint(0, n_total)
        if idx in qian_set: continue
        feat = extract_features(idx)
        if feat is not None:
            ctrl_features.append(feat)
    df_ctrl = pd.DataFrame(ctrl_features)

    # 打段
    df_q['seg'] = ''; df_ctrl['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_q.loc[(df_q['date'] >= ws) & (df_q['date'] < we), 'seg'] = w_label
        df_ctrl.loc[(df_ctrl['date'] >= ws) & (df_ctrl['date'] < we), 'seg'] = w_label
    df_q = df_q[df_q['seg'] != '']; df_ctrl = df_ctrl[df_ctrl['seg'] != '']

    # === 1. 主升浪事件分布 ===
    print(f'\n## 1. 主升浪事件 7 段分布')
    print(f'  {"段":<14} {"主升浪":>8} {"对照":>8} {"主升占比":>8}')
    print('  ' + '-' * 50)
    for w in WINDOWS:
        n_q = (df_q['seg'] == w[0]).sum()
        n_c = (df_ctrl['seg'] == w[0]).sum()
        if n_c == 0: continue
        ratio = n_q / (n_q + n_c) * 100
        print(f'  {w[0]:<14} {n_q:>8,} {n_c:>8,} {ratio:>7.1f}%')

    # === 2. 各段 巽卦比例 walk-forward ===
    print(f'\n## 2. day0-1 个股 d_gua = 巽 (011) 比例 walk-forward')
    print(f'  {"段":<14} {"主升 巽%":>9} {"对照 巽%":>9} {"lift":>6} {"判定":>4}')
    print('  ' + '-' * 50)
    n_pass = 0
    for w in WINDOWS:
        sub_q = df_q[df_q['seg'] == w[0]]
        sub_c = df_ctrl[df_ctrl['seg'] == w[0]]
        if len(sub_q) < 30 or len(sub_c) < 30:
            print(f'  {w[0]:<14}  样本不足')
            continue
        q_xun = (sub_q['stk_d'] == '011').mean() * 100
        c_xun = (sub_c['stk_d'] == '011').mean() * 100
        lift = q_xun - c_xun
        mark = '✅' if lift >= 30 else ('❌' if lift <= 0 else '○')
        if lift >= 30: n_pass += 1
        print(f'  {w[0]:<14} {q_xun:>8.1f}% {c_xun:>8.1f}% {lift:>+5.1f}  {mark}')
    print(f'  → {n_pass}/7 段 lift ≥+30%')

    # === 3. 主力线 / 散户线 / trend 跨段 ===
    print(f'\n## 3. 数值指纹 7 段验证')
    for col, label, q_thresh, ctrl_thresh, comparison in [
        ('mf', '主力线 mf 中位', None, None, 'higher'),
        ('sanhu', '散户线 sanhu 中位', None, None, 'lower'),
        ('mf_5d_mean', '主力线 5d 均值', None, None, 'higher'),
        ('sanhu_5d_mean', '散户线 5d 均值', None, None, 'lower'),
        ('trend_5d_slope', 'trend 5d 斜率', None, None, 'higher'),
        ('trend', 'trend 当下', None, None, 'higher'),
    ]:
        print(f'\n  {label}:')
        print(f'  {"段":<14} {"主升中位":>9} {"对照中位":>9} {"差":>7} {"判定":>4}')
        print('  ' + '-' * 50)
        n_pass = 0
        for w in WINDOWS:
            sub_q = df_q[df_q['seg'] == w[0]]
            sub_c = df_ctrl[df_ctrl['seg'] == w[0]]
            if len(sub_q) < 30 or len(sub_c) < 30: continue
            q_med = sub_q[col].median(); c_med = sub_c[col].median()
            diff = q_med - c_med
            if comparison == 'higher':
                mark = '✅' if diff > 0 else '❌'
                if diff > 0: n_pass += 1
            else:
                mark = '✅' if diff < 0 else '❌'
                if diff < 0: n_pass += 1
            print(f'  {w[0]:<14} {q_med:>+8.2f} {c_med:>+8.2f} {diff:>+6.2f}  {mark}')
        print(f'  → {n_pass}/7 段方向正确')

    # === 4. 综合指纹组合 命中率 ===
    print(f'\n## 4. 组合指纹 命中率 (在主升浪 vs 对照 各段)')
    # 定义: 综合指纹 = 巽 + mf>0 + sanhu<-20 + trend_5d_slope>5
    def is_fingerprint(d):
        return (d['stk_d'] == '011') & (d['mf'] > 0) & (d['sanhu'] < -20) & (d['trend_5d_slope'] > 5)
    df_q['fp'] = is_fingerprint(df_q)
    df_ctrl['fp'] = is_fingerprint(df_ctrl)
    print(f'  {"段":<14} {"主升 命中":>9} {"对照 命中":>9} {"主升/对照":>9} {"判定":>4}')
    print('  ' + '-' * 55)
    n_pass = 0
    for w in WINDOWS:
        sub_q = df_q[df_q['seg'] == w[0]]
        sub_c = df_ctrl[df_ctrl['seg'] == w[0]]
        if len(sub_q) < 30 or len(sub_c) < 30: continue
        q_fp = sub_q['fp'].mean() * 100
        c_fp = sub_c['fp'].mean() * 100
        ratio = q_fp / max(c_fp, 0.01)
        mark = '✅' if ratio >= 5 else ('❌' if ratio < 1.5 else '○')
        if ratio >= 5: n_pass += 1
        print(f'  {w[0]:<14} {q_fp:>8.1f}% {c_fp:>8.1f}% {ratio:>7.1f}x {mark}')
    print(f'  → {n_pass}/7 段 主升:对照 ≥ 5x')

    # 总览
    overall_q = df_q['fp'].mean() * 100
    overall_c = df_ctrl['fp'].mean() * 100
    print(f'\n  总体: 主升 {overall_q:.1f}% 命中, 对照 {overall_c:.1f}% 命中, 比 {overall_q/max(overall_c, 0.01):.1f}x')

    # === 5. 所有 main_force>0 的对照样本里, 多少能成主升浪? (precision) ===
    # 我们用全市场反向: 如果按指纹买, 命中主升浪的比例
    # 这需要全样本扫描
    print(f'\n## 5. 反向: 按指纹找买点, 命中主升浪比例')
    # 直接用 df_ctrl + df_q 合并
    df_all = pd.concat([df_q.assign(is_qian=1), df_ctrl.assign(is_qian=0)], ignore_index=True)
    fp_subset = df_all[df_all['fp']]
    if len(fp_subset) > 0:
        precision = fp_subset['is_qian'].mean() * 100
        # 但 ctrl 是 5x 抽样, 实际比例需要还原
        # df_ctrl 是 5x df_q, 但 df_ctrl 在总人口中只是采样, 不能直接算 precision
        # 改成: 在主升浪样本中命中率 (recall) vs 对照
        recall = (df_q['fp']).mean() * 100
        print(f'  Recall (主升浪中有多少满足指纹): {recall:.1f}%')
        print(f'  对照中有多少满足指纹 (false positive rate): {df_ctrl["fp"].mean() * 100:.1f}%')
        print(f'  指纹 lift: {df_q["fp"].mean() / max(df_ctrl["fp"].mean(), 0.001):.1f}x')


if __name__ == '__main__':
    main()
