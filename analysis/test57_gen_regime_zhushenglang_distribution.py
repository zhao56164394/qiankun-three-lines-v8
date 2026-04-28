# -*- coding: utf-8 -*-
"""阶段 1.5: 艮 regime (mkt_y=001) 主升浪事件分布

输出:
  - 艮 regime 期间 d_gua=111 乾连续段数量、长度分布
  - 主升浪 (≥10 日) 事件的: 数量, 起点期间分布 (按段)
  - 起点前一日的卦象分布 (用什么基础事件能抓到主升浪)

用于决策: 艮 regime 里到底有没有主升浪? 长什么样? 用什么基础事件能在前夜介入?
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
                        columns=['date', 'code', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # ===== 全市场 d_gua=111 连续段 (任何 regime) =====
    print(f'\n=== 扫所有 d_gua=111 连续段 ===')
    runs = []  # (code, start_date, end_date, length, mkt_y_at_start, prev_gua, ret_in_run, ret_30)
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 5: continue
        gua = stk_d_arr[s:e]
        cl = close_arr[s:e]
        n = len(gua)
        i = 0
        while i < n:
            if gua[i] != '111':
                i += 1; continue
            j = i
            while j < n and gua[j] == '111':
                j += 1
            length = j - i
            if length >= 1:
                # 起点 i (天数 j-i)
                gi = s + i
                prev = gua[i-1] if i > 0 else 'NULL'
                # run 内涨幅
                if j - 1 < n:
                    ret_run = (cl[j-1] / cl[i] - 1) * 100
                else:
                    ret_run = float('nan')
                # 起点 30 日内涨幅 (跟 baseline 同口径)
                if i + 30 < n:
                    ret_30 = (cl[i+30] / cl[i] - 1) * 100
                else:
                    ret_30 = float('nan')
                runs.append({
                    'code': code_arr[gi], 'start_date': date_arr[gi],
                    'end_date': date_arr[s + j - 1],
                    'length': length,
                    'mkt_y': mkt_y_arr[gi],
                    'prev_gua': prev,
                    'ret_run': ret_run,
                    'ret_30': ret_30,
                })
            i = j
    df_r = pd.DataFrame(runs)
    print(f'  乾连续段总数: {len(df_r):,}')

    # 标段
    df_r['seg'] = ''
    for w in WINDOWS:
        df_r.loc[(df_r['start_date'] >= w[1]) & (df_r['start_date'] < w[2]), 'seg'] = w[0]

    # ===== 8 regime 主升浪 (length>=10) 分布 =====
    print(f'\n## 8 regime × 乾连续段长度分布')
    print(f'  {"regime":<10} {"全段":>8} {"≥3":>8} {"≥5":>8} {"≥10":>8} {"≥20":>7} {"主升期望%":>10}')
    for r in GUAS:
        sub = df_r[df_r['mkt_y'] == r]
        if len(sub) == 0: continue
        n_all = len(sub)
        n_3 = (sub['length'] >= 3).sum()
        n_5 = (sub['length'] >= 5).sum()
        n_10 = (sub['length'] >= 10).sum()
        n_20 = (sub['length'] >= 20).sum()
        zsl = sub[sub['length'] >= QIAN_RUN]
        zsl_ret = zsl['ret_30'].mean() if len(zsl) > 0 else float('nan')
        label = f'{r}{GUA_NAMES[r]}'
        print(f'  {label:<10} {n_all:>8,} {n_3:>8,} {n_5:>8,} {n_10:>8,} {n_20:>7,} {zsl_ret:>+9.2f}')

    # ===== 艮 regime 主升浪详细 =====
    print(f'\n## 艮 regime 主升浪 (length≥{QIAN_RUN}) 详细')
    gen_zsl = df_r[(df_r['mkt_y'] == '001') & (df_r['length'] >= QIAN_RUN)]
    print(f'  事件数: {len(gen_zsl):,}')
    if len(gen_zsl) > 0:
        print(f'  平均长度: {gen_zsl["length"].mean():.1f} 日')
        print(f'  起点 30 日均收益: {gen_zsl["ret_30"].mean():+.2f}%')
        print(f'  起点 run 内均收益: {gen_zsl["ret_run"].mean():+.2f}%')

        # 段分布
        print(f'\n  段分布:')
        print(f'  {"seg":<14} {"事件":>6} {"平均长度":>9} {"30日%":>7}')
        for w in WINDOWS:
            seg = gen_zsl[gen_zsl['seg'] == w[0]]
            if len(seg) == 0:
                print(f'  {w[0]:<14} {0:>6} {"--":>9} {"--":>7}')
                continue
            print(f'  {w[0]:<14} {len(seg):>6} {seg["length"].mean():>8.1f} {seg["ret_30"].mean():>+6.2f}')

        # 起点前一卦
        print(f'\n  起点前一日卦象分布:')
        prev_cnt = Counter(gen_zsl['prev_gua'])
        for prev, cnt in prev_cnt.most_common():
            ratio = cnt / len(gen_zsl) * 100
            label = f'{prev}{GUA_NAMES.get(prev, "?")}' if prev in GUA_NAMES else prev
            print(f'    {label:<10} {cnt:>5} ({ratio:.1f}%)')

    # ===== 8 regime 主升浪样本对比 =====
    print(f'\n## 8 regime 主升浪事件 (≥10日) 数量横向比较')
    print(f'  {"regime":<10} {"≥10日事件":>10} {"段分布":>20}')
    for r in GUAS:
        sub = df_r[(df_r['mkt_y'] == r) & (df_r['length'] >= QIAN_RUN)]
        if len(sub) == 0: continue
        seg_dist = sub['seg'].value_counts().sort_index()
        seg_str = ' '.join(f'{s.split("_")[0]}={c}' for s, c in seg_dist.items() if s)
        label = f'{r}{GUA_NAMES[r]}'
        print(f'  {label:<10} {len(sub):>10,} {seg_str:>20}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
