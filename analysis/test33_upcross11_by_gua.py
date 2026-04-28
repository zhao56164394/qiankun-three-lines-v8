# -*- coding: utf-8 -*-
"""Step 10 — 上穿 11 事件 × 卦象分桶, 看哪些卦象成功率最高/最低

每个上穿 11 事件标记触发日 6 类卦:
  大盘 y_gua / m_gua / d_gua
  个股 y_gua / m_gua / d_gua

按各卦分桶看:
  - 成功率 (下穿 89 占 resolved)
  - 平均收益
  - 期望收益 (mixing 成功 + 超时)
  - 胜率 (>0)
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
TIMEOUT = 250

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
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
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float32)
    stk_d = df['d_gua'].to_numpy()
    stk_m = df['m_gua'].to_numpy()
    stk_y = df['y_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy()
    mkt_m = df['mkt_m'].to_numpy()
    mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫所有事件 + 记录卦象
    events = []  # (outcome, ret, mkt_y, mkt_m, mkt_d, stk_y, stk_m, stk_d)
    t1 = time.time()
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < 30:
            continue
        td = trend_arr[s:e]
        cl = close_arr[s:e]
        n = len(td)
        buy_mask = (td[:-1] < LOW) & (td[1:] >= LOW)
        buy_idx = np.where(buy_mask)[0] + 1
        sell_mask = (td[:-1] > HIGH) & (td[1:] <= HIGH)
        sell_idx = np.where(sell_mask)[0] + 1

        last_exit = -1
        for b in buy_idx:
            if b <= last_exit: continue
            global_b = s + b
            future_sells = sell_idx[sell_idx > b]
            if len(future_sells) == 0:
                hold = n - 1 - b
                if hold > TIMEOUT:
                    s_local = b + TIMEOUT
                    outcome = 'timeout'
                    ret = cl[s_local] / cl[b] - 1
                    last_exit = s_local
                else:
                    continue  # pending, 不进统计
            else:
                s_local = future_sells[0]
                hold = s_local - b
                if hold > TIMEOUT:
                    s_local = b + TIMEOUT
                    outcome = 'timeout'
                    ret = cl[s_local] / cl[b] - 1
                else:
                    outcome = 'success'
                    ret = cl[s_local] / cl[b] - 1
                last_exit = s_local

            events.append((outcome, ret,
                           mkt_y[global_b], mkt_m[global_b], mkt_d[global_b],
                           stk_y[global_b], stk_m[global_b], stk_d[global_b]))

    print(f'  扫描 + 记录卦象: {time.time()-t1:.1f}s, 事件 {len(events):,}')

    df_e = pd.DataFrame(events, columns=['outcome', 'ret', 'mkt_y', 'mkt_m', 'mkt_d',
                                          'stk_y', 'stk_m', 'stk_d'])
    df_e['success'] = (df_e['outcome'] == 'success')
    df_e['ret_pct'] = df_e['ret'] * 100

    # 全市场基准
    base_succ = df_e['success'].mean() * 100
    base_ret = df_e['ret_pct'].mean()
    base_winrate = (df_e['ret_pct'] > 0).mean() * 100
    print(f'\n## 全市场基准:')
    print(f'  事件: {len(df_e):,}')
    print(f'  成功率 (下穿89): {base_succ:.1f}%')
    print(f'  期望收益: {base_ret:+.2f}%')
    print(f'  胜率 (>0): {base_winrate:.1f}%')

    # 各卦象单维分桶
    for col, label in [('mkt_y', '大盘 y_gua'), ('mkt_m', '大盘 m_gua'), ('mkt_d', '大盘 d_gua'),
                        ('stk_y', '个股 y_gua'), ('stk_m', '个股 m_gua'), ('stk_d', '个股 d_gua')]:
        print(f'\n## 按 {label} 分桶 (n≥500)')
        grp = df_e.groupby(col).agg(
            n=('success', 'size'),
            succ=('success', 'mean'),
            ret=('ret_pct', 'mean'),
            win=('ret_pct', lambda x: (x > 0).mean())
        )
        grp['succ'] *= 100; grp['win'] *= 100
        grp = grp[grp['n'] >= 500].sort_values('ret', ascending=False)
        print(f'  {"卦":<6} {"n":>6} {"成功率":>7} {"vs base":>8} {"期望ret":>8} {"胜率":>6}')
        print('  ' + '-' * 50)
        for k, r in grp.iterrows():
            print(f'  {k}{GUA_NAMES.get(k, "?"):<3} {int(r["n"]):>6,} {r["succ"]:>6.1f}% '
                  f'{r["succ"]-base_succ:>+7.1f} {r["ret"]:>+7.2f}% {r["win"]:>5.1f}%')

    # === Top/Bot 双维: 大盘 y × 大盘 m, 用期望 ret 排序 ===
    print(f'\n## 二维 (大盘 y × 大盘 m) 期望收益 Top/Bot')
    grp2 = df_e.groupby(['mkt_y', 'mkt_m']).agg(
        n=('success', 'size'), ret=('ret_pct', 'mean'), succ=('success', 'mean')
    )
    grp2['succ'] *= 100
    grp2 = grp2[grp2['n'] >= 200].sort_values('ret', ascending=False)
    print(f'  Top 8:')
    for (y, m), r in grp2.head(8).iterrows():
        print(f'    {y}{GUA_NAMES[y]} | {m}{GUA_NAMES[m]}  n={int(r["n"]):>5}  成功率 {r["succ"]:>5.1f}%  期望 {r["ret"]:>+6.2f}%')
    print(f'  Bot 5:')
    for (y, m), r in grp2.tail(5).iterrows():
        print(f'    {y}{GUA_NAMES[y]} | {m}{GUA_NAMES[m]}  n={int(r["n"]):>5}  成功率 {r["succ"]:>5.1f}%  期望 {r["ret"]:>+6.2f}%')


if __name__ == '__main__':
    main()
