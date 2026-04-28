# -*- coding: utf-8 -*-
"""Step 9 — 个股 d_trend 上穿 11 → 下穿 89 成功率扫描

策略:
  买点 = d_trend 上穿 11 (prev<11, curr≥11)
  卖点 = d_trend 下穿 89 (prev>89, curr≤89)
  成功 = 买点之后, 持仓期内出现下穿 89 卖点
  超时 = 250 个交易日 (~1 年) 内没出现下穿 89

输出:
  全市场总事件数
  成功率 = 成功事件 / (成功 + 超时), 不算 pending
  平均成功事件持仓天数 + 收益
  失败事件 (超时) 平均收益
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOW = 11
HIGH = 89
TIMEOUT = 250  # 持仓最大天数


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    print(f'  {len(code_starts):,} 只股票')

    # 全市场扫
    n_buy_total = 0
    n_success = 0  # 下穿 89
    n_timeout = 0  # 超过 250 日无下穿
    n_pending = 0  # 数据结束前还没下穿 (排除统计)
    success_hold_days = []
    success_rets = []
    timeout_rets = []  # 超时强卖收益
    timeout_max_trend = []

    # 每事件记录: (buy_idx, code, date, outcome, hold_days, ret)
    events_log = []

    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30:
            continue
        td = trend_arr[s:e]
        cl = close_arr[s:e]
        n = len(td)

        # buy: prev<11, curr>=11
        buy_mask = (td[:-1] < LOW) & (td[1:] >= LOW)
        buy_idx = np.where(buy_mask)[0] + 1  # 局部 idx in [s..e)

        # sell: prev>89, curr<=89
        sell_mask = (td[:-1] > HIGH) & (td[1:] <= HIGH)
        sell_idx = np.where(sell_mask)[0] + 1

        # 处理 overlap: 持仓中忽略二次 buy
        last_exit = -1
        for b in buy_idx:
            if b <= last_exit:
                continue  # 持仓期间, 跳过
            n_buy_total += 1
            # 找下一个 sell > b
            future_sells = sell_idx[sell_idx > b]
            if len(future_sells) == 0:
                # 数据结束前无卖
                hold = n - 1 - b
                if hold > TIMEOUT:
                    # 超时, 强卖
                    s_local = b + TIMEOUT
                    n_timeout += 1
                    ret = cl[s_local] / cl[b] - 1
                    timeout_rets.append(ret)
                    timeout_max_trend.append(td[b:s_local+1].max())
                    last_exit = s_local
                else:
                    # 真 pending
                    n_pending += 1
                    last_exit = n - 1
            else:
                s_local = future_sells[0]
                hold = s_local - b
                if hold > TIMEOUT:
                    # 超时
                    s_local = b + TIMEOUT
                    n_timeout += 1
                    ret = cl[s_local] / cl[b] - 1
                    timeout_rets.append(ret)
                    timeout_max_trend.append(td[b:s_local+1].max())
                    last_exit = s_local
                else:
                    n_success += 1
                    success_hold_days.append(hold)
                    ret = cl[s_local] / cl[b] - 1
                    success_rets.append(ret)
                    last_exit = s_local

    print(f'\n  扫描完成, {time.time()-t1:.1f}s')

    n_resolved = n_success + n_timeout
    print(f'\n## 全市场结果')
    print(f'  上穿 11 总事件:        {n_buy_total:,}')
    print(f'  ├─ 成功 (下穿 89):    {n_success:,}  ({n_success/max(n_resolved,1)*100:.1f}% of resolved)')
    print(f'  ├─ 超时 ({TIMEOUT}日):     {n_timeout:,}  ({n_timeout/max(n_resolved,1)*100:.1f}% of resolved)')
    print(f'  └─ pending (未结算):  {n_pending:,}  ({n_pending/n_buy_total*100:.1f}%)')

    if success_rets:
        print(f'\n  成功事件:')
        print(f'    平均持仓天数: {np.mean(success_hold_days):.0f} 日')
        print(f'    中位持仓:    {int(np.median(success_hold_days))} 日')
        print(f'    平均收益:    {np.mean(success_rets)*100:+.2f}%')
        print(f'    中位收益:    {np.median(success_rets)*100:+.2f}%')

    if timeout_rets:
        print(f'\n  超时事件 (250 日强卖):')
        print(f'    平均收益:    {np.mean(timeout_rets)*100:+.2f}%')
        print(f'    中位收益:    {np.median(timeout_rets)*100:+.2f}%')
        print(f'    平均期内最高 d_trend: {np.mean(timeout_max_trend):.1f} (是否触及过 89?)')
        n_reached_89 = sum(1 for v in timeout_max_trend if v >= HIGH)
        print(f'    触及过 89 的: {n_reached_89:,}/{n_timeout:,} ({n_reached_89/max(n_timeout,1)*100:.1f}%)')

    # 综合期望: 用所有 resolved 事件 (混合成功 + 超时)
    if n_resolved > 0:
        all_rets = success_rets + timeout_rets
        print(f'\n  全部 resolved 事件混合 (期望收益):')
        print(f'    平均: {np.mean(all_rets)*100:+.2f}%')
        print(f'    中位: {np.median(all_rets)*100:+.2f}%')
        win_rate = sum(1 for r in all_rets if r > 0) / len(all_rets) * 100
        print(f'    >0 占比: {win_rate:.1f}%')


if __name__ == '__main__':
    main()
