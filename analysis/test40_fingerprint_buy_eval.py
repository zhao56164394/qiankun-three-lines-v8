# -*- coding: utf-8 -*-
"""Step 17 — 指纹买入 期望验证

策略:
  买点 = 指纹满足 (个股d=011巽 & mf>0 & sanhu_5d_mean<-20 & trend_5d_slope>5)
  评估窗口 = 买入后固定 30 个交易日 (与 Step 14b 同口径, 卖点未优化故用固定窗)
  记录: 30 日内乾天数, 30 日收益, 30 日内最高收益, 30 日是否触及主升浪 (乾≥10日)

对比:
  - baseline (Step 14b): 单纯巽卦买入, 30d 期望 +2.86%, 主升浪率 50%
  - 加指纹后能提到多少?

walk-forward:
  按买入日期分 7 段
  每段算: 期望/胜率/主升浪率
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
EVAL_WIN = 30  # 买后 30 日窗口

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

    # === 全市场扫描指纹买点 ===
    print(f'\n=== 全市场扫描 4 项指纹买点 ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = gua_arr[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(td)

        # 指纹: 巽卦 + mf>0 + sanhu_5d_mean<-20 + trend_5d_slope>5
        # 滚动算 sanhu_5d_mean, trend_5d_slope (用 prev=i-1, 但买入 i 等同当日收盘买)
        # 这里以"day0"= 当前 i 看是不是指纹日, 评估 [i, i+EVAL_WIN]
        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue
            if mf[i] <= 0: continue
            sanhu_5d = sanhu[max(i-4, 0):i+1].mean()
            if sanhu_5d >= -20: continue
            trend_5d = td[i] - td[max(i-4, 0)]
            if trend_5d <= 5: continue

            # 指纹满足, 买入 i, 评估 [i, i+EVAL_WIN]
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = cl[i+EVAL_WIN] / cl[i] - 1
            max_ret_30 = cl[i:i+EVAL_WIN+1].max() / cl[i] - 1

            events.append({
                'buy_date': date_arr[s + i],
                'n_qian': int(n_qian),
                'ret_30': ret_30 * 100,
                'max_ret_30': max_ret_30 * 100,
                'is_zhushenglang': n_qian >= 10,
            })

    print(f'  事件: {len(events):,}, {time.time()-t1:.1f}s')

    df_e = pd.DataFrame(events)
    df_e['seg'] = ''
    for w in WINDOWS:
        df_e.loc[(df_e['buy_date'] >= w[1]) & (df_e['buy_date'] < w[2]), 'seg'] = w[0]
    df_e = df_e[df_e['seg'] != ''].copy()
    print(f'  打段后: {len(df_e):,}')

    # === 整体表现 ===
    print(f'\n## 1. 指纹买入整体表现')
    print(f'  事件数: {len(df_e):,}')
    print(f'  期望 30d 收益: {df_e["ret_30"].mean():+.2f}%')
    print(f'  中位 30d 收益: {df_e["ret_30"].median():+.2f}%')
    print(f'  胜率 (>0): {(df_e["ret_30"] > 0).mean()*100:.1f}%')
    print(f'  期望 30d 最高: {df_e["max_ret_30"].mean():+.2f}%')
    print(f'  主升浪 (乾≥10) 占比: {df_e["is_zhushenglang"].mean()*100:.1f}%')
    print(f'  乾天数中位: {df_e["n_qian"].median():.0f} 日')
    print(f'  乾天数均值: {df_e["n_qian"].mean():.1f} 日')

    # === 对比 baseline (单纯巽卦买入, Step 14b) ===
    print(f'\n## 2. vs baseline (单纯巽卦买入)')
    print(f'  baseline (来自 Step 14b 13.4 万事件):')
    print(f'    期望 30d: +2.86%, 胜率 51.2%, 主升浪率 50%')
    print(f'    乾天数中位: 9 日, 均值 ~9.5')
    print(f'  指纹买入 ({len(df_e):,} 事件):')
    lift = df_e["ret_30"].mean() - 2.86
    win_lift = (df_e["ret_30"] > 0).mean()*100 - 51.2
    zsl_lift = df_e["is_zhushenglang"].mean()*100 - 50.1
    print(f'    期望 30d: {df_e["ret_30"].mean():+.2f}% (lift {lift:+.2f})')
    print(f'    胜率: {(df_e["ret_30"] > 0).mean()*100:.1f}% (lift {win_lift:+.1f})')
    print(f'    主升浪率: {df_e["is_zhushenglang"].mean()*100:.1f}% (lift {zsl_lift:+.1f})')

    # === 7 段 walk-forward ===
    print(f'\n## 3. 指纹买入 walk-forward 7 段')
    print(f'  {"段":<14} {"n":>6} {"期望":>7} {"胜率":>6} {"主升率":>7} {"乾均":>5}')
    print('  ' + '-' * 55)
    n_pos = 0
    for w in WINDOWS:
        seg = df_e[df_e['seg'] == w[0]]
        if len(seg) < 20:
            print(f'  {w[0]:<14} {len(seg):>6}  样本不足')
            continue
        ret = seg['ret_30'].mean()
        win = (seg['ret_30'] > 0).mean() * 100
        zsl = seg['is_zhushenglang'].mean() * 100
        qm = seg['n_qian'].mean()
        mark = '✅' if ret > 2.86 else ('❌' if ret < 0 else '○')
        if ret > 2.86: n_pos += 1
        print(f'  {w[0]:<14} {len(seg):>6,} {ret:>+6.2f}% {win:>5.1f}% {zsl:>5.1f}% {qm:>4.1f} {mark}')
    print(f'\n  → {n_pos}/7 段 期望 > baseline 2.86%')

    # === 4. 乾天数分布 (看是不是更集中在高乾天数) ===
    print(f'\n## 4. 30 日乾天数分布对比 (vs Step 14b baseline)')
    print(f'  {"乾天数":<10} {"指纹买":>10} {"baseline":>10} {"lift":>6}')
    print('  ' + '-' * 45)
    bl_dist = {  # 来自 Step 14b
        (0, 0): 3.3, (1, 2): 3.9, (3, 5): 14.5, (6, 9): 28.2,
        (10, 14): 34.0, (15, 19): 14.4, (20, 30): 1.8
    }
    for (lo, hi), bl_pct in bl_dist.items():
        sub = df_e[(df_e['n_qian'] >= lo) & (df_e['n_qian'] <= hi)]
        pct = len(sub) / len(df_e) * 100 if len(df_e) > 0 else 0
        label = f'{lo}' if lo == hi else f'{lo}-{hi}'
        lift = pct - bl_pct
        print(f'  {label:<10} {pct:>9.1f}% {bl_pct:>9.1f}% {lift:>+5.1f}')

    # === 5. 按指纹满足项数 ramp (留几项 vs 全留) ===
    # 留作 Step 18 拓展


if __name__ == '__main__':
    main()
