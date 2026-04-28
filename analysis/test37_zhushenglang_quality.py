# -*- coding: utf-8 -*-
"""Step 14 — 主升浪质量分析

定义:
  买点 = 阶段 2 巽卦买入
  卖点 = 阶段 3 乾→其他 (一日乘乘)
  乾天数 = 持仓期间 d_gua=111 出现的天数 (整段累加)
  乾段数 = 持仓期间 出现 [...111(连续 N 日)非111...] 的乾连续段数

研究:
  1. 乾天数分布 (0/1/2/3.../N+ 日)
  2. 乾天数 vs 收益散点 (合并桶看趋势)
  3. 乾段数分布 (1/2/3+ 段, 看是否有反复乾→其他→乾)
  4. 0 乾事件: 买入后没乾就卖 (乾卦从来没出现过怎么会触发卖点? — 实际是触发卖前就有乾, 但卖点是"乾→其他"那天, 之前必有乾)
     特殊情况: timeout 卖出可能 0 乾
  5. 区分 "主升浪" (乾天数 ≥ N) 和 "假突破" (乾天数 < N)
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
HOLD_TIMEOUT = 250

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


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

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
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

        sell_mask = np.r_[False, (gua[:-1] == '111') & (gua[1:] != '111')]
        sell_idx_arr = np.where(sell_mask)[0]

        last_done = -1
        for ent in entry_idx:
            if ent <= last_done: continue
            window_end = min(n - 1, ent + POOL_TIMEOUT)
            xun_in_win = xun_idx[(xun_idx >= ent) & (xun_idx <= window_end)]
            if len(xun_in_win) == 0:
                last_done = window_end
                continue
            buy_local = xun_in_win[0]

            future_sells = sell_idx_arr[sell_idx_arr > buy_local]
            if len(future_sells) == 0:
                if n - 1 - buy_local > HOLD_TIMEOUT:
                    sell_local = buy_local + HOLD_TIMEOUT
                    exit_type = 'timeout'
                else:
                    last_done = n - 1
                    continue
            else:
                sell_local = future_sells[0]
                if sell_local - buy_local > HOLD_TIMEOUT:
                    sell_local = buy_local + HOLD_TIMEOUT
                    exit_type = 'timeout'
                else:
                    exit_type = 'qian_change'

            ret = cl[sell_local] / cl[buy_local] - 1
            # 持仓期间 (buy_local 到 sell_local-1, 因为 sell_local 当日卖出, 不算入持仓的 gua 序列?)
            # 一日乘乘: 卖点是 prev=111 & curr!=111 那天, 所以 sell_local-1 必为乾, sell_local 不为乾
            # 我们算 [buy_local, sell_local] 区间的乾天数 (含买入日, 不含卖出日)
            seg_gua = gua[buy_local:sell_local]
            n_qian = (seg_gua == '111').sum()
            n_days = sell_local - buy_local
            qian_ratio = n_qian / n_days if n_days > 0 else 0

            # 乾段数 (连续乾算一段)
            n_qian_runs = 0
            in_run = False
            for g_v in seg_gua:
                if g_v == '111':
                    if not in_run:
                        n_qian_runs += 1
                        in_run = True
                else:
                    in_run = False

            # 最长连续乾
            max_qian_run = 0
            cur = 0
            for g_v in seg_gua:
                if g_v == '111':
                    cur += 1
                    if cur > max_qian_run: max_qian_run = cur
                else:
                    cur = 0

            completed.append({
                'hold_days': n_days,
                'n_qian': int(n_qian),
                'n_qian_runs': n_qian_runs,
                'max_qian_run': max_qian_run,
                'qian_ratio': qian_ratio,
                'ret_pct': ret * 100,
                'exit_type': exit_type,
                'buy_date': date_arr[s + buy_local],
                'pool_wait': buy_local - ent,
            })
            last_done = sell_local

    print(f'  扫描完成 {time.time()-t1:.1f}s, 事件 {len(completed):,}')

    df_c = pd.DataFrame(completed)

    # === 1. 乾天数分布 ===
    print(f'\n## 1. 持仓期间 乾天数 分布')
    print(f'  {"乾天数":<8} {"n":>8} {"占比":>5} {"均收益%":>8} {"胜率":>6} {"均持仓":>6}')
    print('  ' + '-' * 55)
    bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 14), (15, 24), (25, 49), (50, 9999)]
    for lo, hi in bins:
        sub = df_c[(df_c['n_qian'] >= lo) & (df_c['n_qian'] <= hi)]
        if len(sub) == 0: continue
        label = f'{lo}' if lo == hi else f'{lo}-{hi if hi < 9999 else "+"}'
        win = (sub['ret_pct'] > 0).mean() * 100
        print(f'  {label:<8} {len(sub):>8,} {len(sub)/len(df_c)*100:>4.1f}% '
              f'{sub["ret_pct"].mean():>+7.2f}% {win:>5.1f}% {sub["hold_days"].mean():>5.0f}日')

    # 关键: 乾 0 天的事件
    n0 = (df_c['n_qian'] == 0).sum()
    if n0 > 0:
        print(f'\n  注: 乾 0 天事件 = {n0:,} ({n0/len(df_c)*100:.1f}%) — 这些是没真正主升浪的"假突破"')

    # === 2. 乾天数 vs 收益的相关性 ===
    print(f'\n## 2. 乾天数 vs 收益')
    corr = df_c['n_qian'].corr(df_c['ret_pct'])
    print(f'  相关系数 (Pearson): {corr:.3f}')

    # 各乾天数桶的累计收益贡献
    print(f'\n  累计收益贡献 (按乾天数 cumulative)')
    df_sorted = df_c.sort_values('n_qian')
    cum_ret = df_sorted['ret_pct'].cumsum().values
    cum_n = np.arange(1, len(df_sorted) + 1)
    # 在不同乾天数门槛下的累计贡献
    for thresh in [0, 1, 2, 5, 10, 20, 50]:
        mask = df_sorted['n_qian'].values >= thresh
        if mask.sum() == 0: continue
        sub_ret = df_sorted.loc[mask, 'ret_pct']
        print(f'  乾≥{thresh:>2}日: {mask.sum():>7,} 单 ({mask.sum()/len(df_c)*100:>4.1f}%)  '
              f'均收益 {sub_ret.mean():>+6.2f}%  贡献总收益 {sub_ret.sum()/df_c["ret_pct"].sum()*100:>5.1f}%')

    # === 3. 乾段数分布 ===
    print(f'\n## 3. 乾段数 分布 (持仓期间出现几段连续乾)')
    print(f'  {"段数":<8} {"n":>8} {"占比":>5} {"均收益%":>8} {"胜率":>6}')
    print('  ' + '-' * 50)
    for r in [0, 1, 2, 3, 4, 5]:
        sub = df_c[df_c['n_qian_runs'] == r] if r < 5 else df_c[df_c['n_qian_runs'] >= 5]
        if len(sub) == 0: continue
        label = f'{r}' if r < 5 else '5+'
        win = (sub['ret_pct'] > 0).mean() * 100
        print(f'  {label:<8} {len(sub):>8,} {len(sub)/len(df_c)*100:>4.1f}% '
              f'{sub["ret_pct"].mean():>+7.2f}% {win:>5.1f}%')

    # === 4. 最长连续乾 vs 收益 ===
    print(f'\n## 4. 最长连续乾段 vs 收益')
    print(f'  {"最长乾":<8} {"n":>8} {"占比":>5} {"均收益%":>8} {"胜率":>6}')
    print('  ' + '-' * 50)
    for lo, hi in [(0, 0), (1, 2), (3, 4), (5, 9), (10, 19), (20, 9999)]:
        sub = df_c[(df_c['max_qian_run'] >= lo) & (df_c['max_qian_run'] <= hi)]
        if len(sub) == 0: continue
        label = f'{lo}' if lo == hi else f'{lo}-{hi if hi < 9999 else "+"}'
        win = (sub['ret_pct'] > 0).mean() * 100
        print(f'  {label:<8} {len(sub):>8,} {len(sub)/len(df_c)*100:>4.1f}% '
              f'{sub["ret_pct"].mean():>+7.2f}% {win:>5.1f}%')

    # === 5. 区分主升浪 vs 假突破 ===
    print(f'\n## 5. 主升浪 vs 假突破')
    real_thresh = 5  # 乾 ≥ 5 日 = 主升浪
    real = df_c[df_c['n_qian'] >= real_thresh]
    fake = df_c[df_c['n_qian'] < real_thresh]
    print(f'  主升浪 (乾≥{real_thresh}日): {len(real):>7,} ({len(real)/len(df_c)*100:.1f}%)')
    print(f'    均收益 {real["ret_pct"].mean():>+6.2f}%, 胜率 {(real["ret_pct"]>0).mean()*100:.1f}%, 持仓 {real["hold_days"].mean():.0f}日')
    print(f'  假突破 (乾<{real_thresh}日): {len(fake):>7,} ({len(fake)/len(df_c)*100:.1f}%)')
    print(f'    均收益 {fake["ret_pct"].mean():>+6.2f}%, 胜率 {(fake["ret_pct"]>0).mean()*100:.1f}%, 持仓 {fake["hold_days"].mean():.0f}日')

    # 假设: 如果只买"会发展成主升浪"的票, 期望多少
    # 但实际"会不会成主升浪"是事后看, 这里看上限
    print(f'\n  上限 (假设能预知, 只买主升浪): 期望 {real["ret_pct"].mean():+.2f}% (vs 当前 baseline {df_c["ret_pct"].mean():+.2f}%)')
    print(f'  下限 (剔除主升浪): 期望 {fake["ret_pct"].mean():+.2f}%')


if __name__ == '__main__':
    main()
