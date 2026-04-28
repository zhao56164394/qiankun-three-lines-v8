# -*- coding: utf-8 -*-
"""Step 14b — 用 "买入后固定 30 天" 窗口看主升浪质量

旧版用 "卖到乾切" 期间, 但卖点未优化, 退出过早导致乾天数被截断.
改用买入后 [t+0, t+30] 固定 30 日窗口, 看其中乾天数 + 收益.

研究:
  1. 30 日内 乾天数分布
  2. 乾天数 vs 30 日收益
  3. 主升浪 (乾天数≥10 日) 占比
  4. 30 日内最长连续乾
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENTRY_LOW = 11
POOL_TIMEOUT = 60
EVAL_WINDOW = 30  # 买入后多少天看主升浪


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'd_gua']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float32)
    gua_arr = df['d_gua'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    completed = []
    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = gua_arr[s:e]; n = len(td)

        entry_mask = (td[:-1] > ENTRY_LOW) & (td[1:] <= ENTRY_LOW)
        entry_idx = np.where(entry_mask)[0] + 1
        if len(entry_idx) == 0: continue

        xun_mask = (gua == '011')
        xun_idx = np.where(xun_mask)[0]

        last_done = -1
        for ent in entry_idx:
            if ent <= last_done: continue
            window_end = min(n - 1, ent + POOL_TIMEOUT)
            xun_in_win = xun_idx[(xun_idx >= ent) & (xun_idx <= window_end)]
            if len(xun_in_win) == 0:
                last_done = window_end
                continue
            buy_local = xun_in_win[0]

            # 买入后 30 日固定窗口
            eval_end = buy_local + EVAL_WINDOW
            if eval_end >= n:
                # 数据不足 30 日, 跳过
                last_done = buy_local + 1
                continue

            seg_gua = gua[buy_local:eval_end]
            n_qian = (seg_gua == '111').sum()
            ret_30 = cl[eval_end] / cl[buy_local] - 1

            # 最长连续乾 + 段数
            max_run = 0; cur = 0; n_runs = 0; in_run = False
            for g_v in seg_gua:
                if g_v == '111':
                    cur += 1
                    if cur > max_run: max_run = cur
                    if not in_run: n_runs += 1; in_run = True
                else:
                    cur = 0; in_run = False

            # 30 日内最高价收益
            max_close_in_win = cl[buy_local:eval_end + 1].max()
            max_ret = max_close_in_win / cl[buy_local] - 1

            completed.append({
                'buy_date': date_arr[s + buy_local],
                'pool_wait': buy_local - ent,
                'n_qian': int(n_qian),
                'max_qian_run': max_run,
                'n_qian_runs': n_runs,
                'ret_30': ret_30 * 100,
                'max_ret_30': max_ret * 100,
            })
            last_done = buy_local + 1  # 允许下一次入池/买入

    print(f'  扫描完成 {time.time()-t1:.1f}s, 事件 {len(completed):,}')

    df_c = pd.DataFrame(completed)
    print(f'\n## 30 日窗口 baseline')
    print(f'  事件: {len(df_c):,}')
    print(f'  期望 30d 收益: {df_c["ret_30"].mean():+.2f}%')
    print(f'  胜率 (>0): {(df_c["ret_30"] > 0).mean()*100:.1f}%')
    print(f'  期望 30d 内最高: {df_c["max_ret_30"].mean():+.2f}%')

    # === 1. 30 日内乾天数分布 ===
    print(f'\n## 1. 30 日内 乾天数 分布')
    print(f'  {"乾天数":<10} {"n":>8} {"占比":>5} {"均30d":>8} {"胜率":>6} {"均最高":>8}')
    print('  ' + '-' * 55)
    for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 9), (10, 14), (15, 19), (20, 30)]:
        sub = df_c[(df_c['n_qian'] >= lo) & (df_c['n_qian'] <= hi)]
        if len(sub) == 0: continue
        label = f'{lo}' if lo == hi else f'{lo}-{hi}'
        win = (sub['ret_30'] > 0).mean() * 100
        print(f'  {label:<10} {len(sub):>8,} {len(sub)/len(df_c)*100:>4.1f}% '
              f'{sub["ret_30"].mean():>+7.2f}% {win:>5.1f}% {sub["max_ret_30"].mean():>+7.2f}%')

    # === 2. Pearson ===
    print(f'\n## 2. 乾天数 vs 30d 收益')
    print(f'  Pearson 相关系数: {df_c["n_qian"].corr(df_c["ret_30"]):.3f}')
    print(f'  Pearson 相关系数 (vs 30d 最高): {df_c["n_qian"].corr(df_c["max_ret_30"]):.3f}')

    # === 3. 最长连续乾 ===
    print(f'\n## 3. 30 日内 最长连续乾段 vs 收益')
    print(f'  {"最长乾":<10} {"n":>8} {"占比":>5} {"均30d":>8} {"胜率":>6}')
    print('  ' + '-' * 50)
    for lo, hi in [(0, 0), (1, 2), (3, 4), (5, 9), (10, 14), (15, 30)]:
        sub = df_c[(df_c['max_qian_run'] >= lo) & (df_c['max_qian_run'] <= hi)]
        if len(sub) == 0: continue
        label = f'{lo}' if lo == hi else f'{lo}-{hi}'
        win = (sub['ret_30'] > 0).mean() * 100
        print(f'  {label:<10} {len(sub):>8,} {len(sub)/len(df_c)*100:>4.1f}% '
              f'{sub["ret_30"].mean():>+7.2f}% {win:>5.1f}%')

    # === 4. 主升浪 vs 假突破 ===
    print(f'\n## 4. 主升浪 vs 假突破 (用 30 日内乾天数)')
    for thresh in [5, 8, 10, 12, 15]:
        real = df_c[df_c['n_qian'] >= thresh]
        fake = df_c[df_c['n_qian'] < thresh]
        print(f'\n  阈值 乾≥{thresh} 日:')
        print(f'    主升浪 {len(real):>7,} ({len(real)/len(df_c)*100:>4.1f}%)  '
              f'30d {real["ret_30"].mean():>+6.2f}%  胜率 {(real["ret_30"]>0).mean()*100:>5.1f}%  '
              f'最高 {real["max_ret_30"].mean():>+6.2f}%')
        print(f'    假突破 {len(fake):>7,} ({len(fake)/len(df_c)*100:>4.1f}%)  '
              f'30d {fake["ret_30"].mean():>+6.2f}%  胜率 {(fake["ret_30"]>0).mean()*100:>5.1f}%')


if __name__ == '__main__':
    main()
