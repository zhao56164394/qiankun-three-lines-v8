# -*- coding: utf-8 -*-
"""Step 13 — 三阶段事件流验证

策略 (个股日卦层):
  阶段 1 [入池]: d_trend 下穿 11 (prev>11, curr<=11)  — 标记低位
  阶段 2 [买入]: 入池后 60 个交易日内, 出现 d_gua=011巽 → 当日收盘买入
                 60 日超时则取消池位
  阶段 3 [卖出]: 持仓中, 上一日 d_gua=111乾, 当日 d_gua≠111乾 → 当日收盘卖
                 (一日乘乘) 持仓最多 250 日, 超时强卖

输出:
  - 全市场入池/买入/卖出/超时 计数
  - 完整事件 (有买有卖) 的期望收益、胜率、平均持仓
  - 入池→买入失败率 (入池后 60 日没出现巽)
  - 买入→卖出失败率 (买入后 250 日没乾→其他)
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
POOL_TIMEOUT = 60   # 入池后多少天必须出现巽
HOLD_TIMEOUT = 250  # 买入后多少天没出乾→其他, 强卖

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

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
    print('=== 加载数据 ===')
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
    print(f'  {len(code_starts):,} 只股')

    # 全市场扫
    n_pool_entry = 0
    n_pool_timeout = 0   # 入池后没出巽
    n_buy = 0            # 入池后出现巽
    n_sell_qian = 0      # 乾→其他 卖
    n_sell_timeout = 0   # 持仓超时
    n_sell_pending = 0   # 数据末未结算

    completed = []  # (entry_date, buy_date, sell_date, hold_days, ret, exit_type)

    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30: continue
        td = trend_arr[s:e]
        cl = close_arr[s:e]
        gua = gua_arr[s:e]
        n = len(td)

        # 阶段 1: 入池 (下穿 11)
        entry_mask = (td[:-1] > ENTRY_LOW) & (td[1:] <= ENTRY_LOW)
        entry_idx = np.where(entry_mask)[0] + 1
        if len(entry_idx) == 0:
            continue

        # 寻找巽: 局部 idx
        xun_mask = (gua == '011')
        xun_idx = np.where(xun_mask)[0]

        # 寻找乾→其他卖点: prev=='111' & curr!='111' (skip first)
        sell_mask = np.r_[False, (gua[:-1] == '111') & (gua[1:] != '111')]
        sell_idx_arr = np.where(sell_mask)[0]

        last_done = -1  # 上次结算的局部 idx, 防止重叠
        for ent in entry_idx:
            if ent <= last_done: continue
            n_pool_entry += 1
            # 阶段 2: 入池后 [ent, ent+POOL_TIMEOUT] 内寻找巽
            window_end = min(n - 1, ent + POOL_TIMEOUT)
            xun_in_win = xun_idx[(xun_idx >= ent) & (xun_idx <= window_end)]
            if len(xun_in_win) == 0:
                n_pool_timeout += 1
                last_done = window_end
                continue
            buy_local = xun_in_win[0]
            n_buy += 1

            # 阶段 3: 买入后, 找下一个 乾→其他 卖点
            future_sells = sell_idx_arr[sell_idx_arr > buy_local]
            if len(future_sells) == 0:
                # 数据末没卖, 看是否超时
                if n - 1 - buy_local > HOLD_TIMEOUT:
                    sell_local = buy_local + HOLD_TIMEOUT
                    exit_type = 'timeout'
                    n_sell_timeout += 1
                else:
                    n_sell_pending += 1
                    last_done = n - 1
                    continue
            else:
                sell_local = future_sells[0]
                if sell_local - buy_local > HOLD_TIMEOUT:
                    sell_local = buy_local + HOLD_TIMEOUT
                    exit_type = 'timeout'
                    n_sell_timeout += 1
                else:
                    exit_type = 'qian_change'
                    n_sell_qian += 1

            ret = cl[sell_local] / cl[buy_local] - 1
            completed.append({
                'code_idx': ci,
                'entry_date': date_arr[s + ent],
                'buy_date': date_arr[s + buy_local],
                'sell_date': date_arr[s + sell_local],
                'pool_wait': buy_local - ent,
                'hold_days': sell_local - buy_local,
                'ret_pct': ret * 100,
                'exit_type': exit_type,
            })
            last_done = sell_local

    print(f'  扫描完成, {time.time()-t1:.1f}s')
    df_c = pd.DataFrame(completed)

    # === 报告 ===
    print(f'\n## 三阶段事件流统计')
    print(f'  阶段 1 [下穿11入池]:                   {n_pool_entry:>8,}')
    print(f'    ├─ 60日内出现巽 → 进阶段 2:          {n_buy:>8,} ({n_buy/max(n_pool_entry,1)*100:.1f}%)')
    print(f'    └─ 60日超时无巽 → 取消:              {n_pool_timeout:>8,} ({n_pool_timeout/max(n_pool_entry,1)*100:.1f}%)')
    print()
    print(f'  阶段 2 [巽卦买入]:                     {n_buy:>8,}')
    print(f'    ├─ 乾→其他 卖出:                     {n_sell_qian:>8,} ({n_sell_qian/max(n_buy,1)*100:.1f}%)')
    print(f'    ├─ 250日超时强卖:                    {n_sell_timeout:>8,} ({n_sell_timeout/max(n_buy,1)*100:.1f}%)')
    print(f'    └─ 数据末未结算 (排除):              {n_sell_pending:>8,}')

    if len(df_c) == 0:
        print('\n  无完成事件!')
        return

    # 整体表现
    print(f'\n## 完成事件表现 ({len(df_c):,} 单)')
    print(f'  期望收益: {df_c["ret_pct"].mean():>+6.2f}%')
    print(f'  中位收益: {df_c["ret_pct"].median():>+6.2f}%')
    print(f'  胜率 (>0): {(df_c["ret_pct"] > 0).mean()*100:>5.1f}%')
    print(f'  平均池等待: {df_c["pool_wait"].mean():.1f} 日 (中位 {df_c["pool_wait"].median():.0f})')
    print(f'  平均持仓: {df_c["hold_days"].mean():.1f} 日 (中位 {df_c["hold_days"].median():.0f})')

    # 按 exit_type 拆
    print(f'\n## 按 exit 类型拆')
    for et in ['qian_change', 'timeout']:
        sub = df_c[df_c['exit_type'] == et]
        if len(sub) == 0: continue
        print(f'  {et:<14} n={len(sub):>6,}  期望 {sub["ret_pct"].mean():>+6.2f}%  '
              f'胜率 {(sub["ret_pct"] > 0).mean()*100:>5.1f}%  '
              f'持仓 {sub["hold_days"].mean():.0f}日')

    # === walk-forward 7 段 ===
    print(f'\n## walk-forward 7 段 (按买入日期分段)')
    df_c['seg'] = ''
    for w_label, ws, we in WINDOWS:
        df_c.loc[(df_c['buy_date'] >= ws) & (df_c['buy_date'] < we), 'seg'] = w_label
    df_c = df_c[df_c['seg'] != ''].copy()

    print(f'  {"段":<14} {"n":>6} {"期望":>7} {"胜率":>6} {"乾切n":>6} {"超时n":>6}')
    print('  ' + '-' * 60)
    for w_label, _, _ in WINDOWS:
        seg = df_c[df_c['seg'] == w_label]
        if len(seg) == 0:
            print(f'  {w_label:<14}  无事件')
            continue
        n_q = (seg['exit_type'] == 'qian_change').sum()
        n_t = (seg['exit_type'] == 'timeout').sum()
        print(f'  {w_label:<14} {len(seg):>6,} {seg["ret_pct"].mean():>+6.2f}% '
              f'{(seg["ret_pct"] > 0).mean()*100:>5.1f}% {n_q:>6,} {n_t:>6,}')

    # 段稳定性判定
    print(f'\n## 段稳定性')
    seg_means = df_c.groupby('seg')['ret_pct'].mean()
    n_pos = (seg_means > 3).sum()
    n_neg = (seg_means < -3).sum()
    n_valid = len(seg_means)
    print(f'  正期望 (>+3%) 段: {n_pos}/{n_valid}')
    print(f'  负期望 (<-3%) 段: {n_neg}/{n_valid}')
    if n_pos >= 5 and n_neg <= 1:
        print(f'  ★ 真稳定策略')
    elif n_pos >= 4:
        print(f'  ○ 准稳定')
    else:
        print(f'  ✗ 不稳')


if __name__ == '__main__':
    main()
