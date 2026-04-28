# -*- coding: utf-8 -*-
"""持仓分析: 为什么艮 regime bull 卖点要 57 日?

拆解:
  - 退出类型分布 (bull_2nd / timeout / 中途乾切等)
  - timeout (60 日兜底) 占多少比例
  - 真主升浪 vs 假突破 各自的持仓
  - cross_count (89 下穿次数) 分布: 多少股根本没穿过 89?
  - d_trend 最大值分布: 多少股 d_trend 从未到 89?
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
MAX_HOLD = 60
QIAN_RUN = 10
REGIME_Y = '001'
TRIGGER_GUA = '011'
AVOID_STK_M = '111'


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
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print(f'\n=== 入场 (艮 regime + 巽日 + 避雷 + score≥2) ===')
    holdings = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            if stk_m_arr[gi] == AVOID_STK_M: continue
            score = 0
            if stk_y_arr[gi] == '101': score += 1
            if gi - 5 >= s:
                sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1]))
                if sanhu_5d < -50: score += 1
                elif sanhu_5d < -30: score += 1
            if score < 2: continue

            # bull 卖点详细分析
            buy = i
            end = i + MAX_HOLD + 1
            if s + end > e: continue
            td_seg = trend_arr[s+buy:s+end]
            gua_seg = stk_d_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]

            # 走 bull 逻辑
            sell_idx = MAX_HOLD
            sell_type = 'timeout'
            cnt = 0
            running_max = td_seg[0]
            cross_dates = []  # 记录所有下穿 89 的日子
            for k in range(1, len(td_seg)):
                if not np.isnan(td_seg[k]):
                    running_max = max(running_max, td_seg[k])
                if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
                    cnt += 1
                    cross_dates.append(k)
                    if cnt == 2:
                        sell_idx = k
                        sell_type = 'bull_2nd'
                        break

            # 一些诊断指标
            td_max = float(np.nanmax(td_seg[:sell_idx+1]))
            ever_89 = td_max >= 89
            n_qian = int((gua_seg[:sell_idx+1] == '111').sum())

            holdings.append({
                'date': date_arr[gi],
                'code': code_arr[gi],
                'sell_type': sell_type,
                'hold': sell_idx,
                'cross_count': cnt,
                'first_cross_day': cross_dates[0] if cross_dates else -1,
                'second_cross_day': cross_dates[1] if len(cross_dates) >= 2 else -1,
                'td_max': td_max,
                'ever_89': ever_89,
                'n_qian': n_qian,
                'is_zsl': n_qian >= QIAN_RUN,
                'ret': (cl_seg[sell_idx] / cl_seg[0] - 1) * 100,
            })

    df_h = pd.DataFrame(holdings)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'  入场: {len(df_h)}')

    # ===== 退出类型分布 =====
    print(f'\n## 退出类型分布')
    print(f'  {"类型":<14} {"数量":>5} {"占比%":>6} {"持仓":>5} {"期望%":>7} {"主升率%":>8}')
    for st in df_h['sell_type'].unique():
        sub = df_h[df_h['sell_type'] == st]
        ratio = len(sub) / len(df_h) * 100
        hold = sub['hold'].mean()
        ret = sub['ret'].mean()
        zsl = sub['is_zsl'].mean() * 100
        print(f'  {st:<14} {len(sub):>5} {ratio:>5.1f} {hold:>5.1f} {ret:>+7.2f} {zsl:>7.1f}')

    # ===== timeout 占比的关键问题 =====
    print(f'\n## ★ 为什么持仓 57 日?')
    n_timeout = (df_h['sell_type'] == 'timeout').sum()
    n_bull_2nd = (df_h['sell_type'] == 'bull_2nd').sum()
    print(f'  timeout (60 日兜底): {n_timeout}/{len(df_h)} = {n_timeout/len(df_h)*100:.1f}%')
    print(f'  bull_2nd (真触发): {n_bull_2nd}/{len(df_h)} = {n_bull_2nd/len(df_h)*100:.1f}%')

    # ===== d_trend 是否到过 89 =====
    print(f'\n## d_trend 最大值分布 (持仓 60 日窗口内)')
    n_to_89 = df_h['ever_89'].sum()
    print(f'  到过 89: {n_to_89}/{len(df_h)} = {n_to_89/len(df_h)*100:.1f}%')
    print(f'  从未到 89: {len(df_h)-n_to_89}/{len(df_h)} = {(len(df_h)-n_to_89)/len(df_h)*100:.1f}%')
    print(f'\n  td_max 分布:')
    for thresh in [50, 70, 80, 89, 95, 99]:
        cnt = (df_h['td_max'] >= thresh).sum()
        print(f'    ≥{thresh}: {cnt} ({cnt/len(df_h)*100:.1f}%)')

    # ===== cross_count 分布 =====
    print(f'\n## 下穿 89 次数分布')
    cnt_dist = df_h['cross_count'].value_counts().sort_index()
    for cnt, n in cnt_dist.items():
        print(f'  {cnt} 次: {n} ({n/len(df_h)*100:.1f}%)')

    # ===== 主升 vs 假 持仓拆解 =====
    print(f'\n## 主升浪事件 vs 假突破 持仓拆解')
    for label, mask in [('主升浪 (n_qian≥10)', df_h['is_zsl']),
                          ('假突破 (n_qian<10)', ~df_h['is_zsl'])]:
        sub = df_h[mask]
        if len(sub) == 0: continue
        print(f'\n  {label} (n={len(sub)}):')
        for st in sub['sell_type'].unique():
            sub2 = sub[sub['sell_type'] == st]
            print(f'    {st:<14} {len(sub2):>5} ({len(sub2)/len(sub)*100:.1f}%) 平均持仓 {sub2["hold"].mean():.1f}, 期望 {sub2["ret"].mean():+.2f}%')

    # ===== 第 1 次穿 89 日子 (主升浪) =====
    print(f'\n## 主升浪事件: 第 1 次穿 89 vs 第 2 次穿 89 距离')
    zsl = df_h[df_h['is_zsl']]
    crossed = zsl[zsl['first_cross_day'] >= 0]
    if len(crossed) > 0:
        print(f'  第 1 次穿 89 平均日: {crossed["first_cross_day"].mean():.1f}')
    crossed2 = zsl[zsl['second_cross_day'] >= 0]
    if len(crossed2) > 0:
        print(f'  第 2 次穿 89 平均日: {crossed2["second_cross_day"].mean():.1f}')
        print(f'  两次间隔平均: {(crossed2["second_cross_day"] - crossed2["first_cross_day"]).mean():.1f} 日')
        print(f'  能等到第 2 次穿的: {len(crossed2)}/{len(zsl)} = {len(crossed2)/len(zsl)*100:.1f}%')

    # ===== 跟坤对比 =====
    print(f'\n## 拆段持仓 (w2 大熊 vs w4 抱团)')
    for seg_name in ['w2_2019', 'w4_2021']:
        sub = df_h[df_h['seg'] == seg_name]
        if len(sub) == 0: continue
        print(f'\n  {seg_name} (n={len(sub)}):')
        n_to = (sub['ever_89']).sum()
        print(f'    d_trend 到过 89: {n_to}/{len(sub)} = {n_to/len(sub)*100:.1f}%')
        for st in sub['sell_type'].unique():
            sub2 = sub[sub['sell_type'] == st]
            print(f'    {st:<14} {len(sub2):>4} ({len(sub2)/len(sub)*100:.1f}%) 持仓 {sub2["hold"].mean():.1f}')


if __name__ == '__main__':
    main()
