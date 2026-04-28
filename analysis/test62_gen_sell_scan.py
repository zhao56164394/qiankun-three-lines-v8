# -*- coding: utf-8 -*-
"""阶段 6: 艮 regime 卖点扫描

入场: 艮 regime 巽日 + 强避雷 + score ≥ 2
卖点候选 (重点短窗口 + bull):
  S1: 固定 15 日
  S2: 固定 20 日
  S3: 固定 30 日 (baseline)
  S4: 第 1 次下穿 89
  S5: 第 2 次下穿 89 (V8 bull)
  S6: bull + TS15 (15 日未到 89 强卖, 短期版)
  S7: bull + TS20 (跟坤 v3 一致)
  S8: 乾→其他 (M1)
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

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

# 强避雷: 个股 m_gua=111 乾
AVOID_STK_M = '111'
# 强好规律 (软排名 score)
GOOD_RULES = [
    ('stk_y', '101'),    # 个股年卦离
]


def calc_sells(buy_idx, end_idx, td_seg, gua_seg):
    """对单个买点, 算各卖点的退出日 (相对 buy_idx)"""
    n = end_idx - buy_idx
    results = {}

    # S1/S2/S3: 固定窗口
    for win, name in [(15, 'fix15'), (20, 'fix20'), (30, 'fix30')]:
        results[name] = min(win, n - 1)

    # S4: 第 1 次下穿 89
    s4 = n - 1
    running_max = td_seg[0]
    for k in range(1, n):
        if not np.isnan(td_seg[k]):
            running_max = max(running_max, td_seg[k])
        if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
            s4 = k
            break
    results['cross1'] = s4

    # S5: 第 2 次下穿 89 (bull)
    s5 = n - 1
    cnt = 0
    running_max = td_seg[0]
    for k in range(1, n):
        if not np.isnan(td_seg[k]):
            running_max = max(running_max, td_seg[k])
        if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
            cnt += 1
            if cnt == 2:
                s5 = k
                break
    results['bull'] = s5

    # S6/S7: bull + TS15/TS20
    for ts, name in [(15, 'bull_ts15'), (20, 'bull_ts20')]:
        sx = n - 1
        cnt = 0
        running_max = td_seg[0]
        for k in range(1, n):
            if not np.isnan(td_seg[k]):
                running_max = max(running_max, td_seg[k])
            if running_max >= 89 and td_seg[k] < 89 and td_seg[k-1] >= 89:
                cnt += 1
                if cnt == 2:
                    sx = k; break
            # TS
            if k >= ts:
                seg_max = np.nanmax(td_seg[:k+1])
                if seg_max < 89:
                    sx = k; break
        results[name] = sx

    # S8: 乾→其他 (M1)
    s8 = n - 1
    in_qian = False
    for k in range(0, n):
        if gua_seg[k] == '111':
            in_qian = True
        elif in_qian and gua_seg[k] != '111':
            s8 = k
            break
    results['m1_qian'] = s8

    return results


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
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
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
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 入场: 艮 regime + 巽日 + 强避雷 + score>=2
    print(f'\n=== 扫艮 regime 巽日入场 (避雷 + score>=2) ===')
    entries = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if stk_d_arr[gi] != TRIGGER_GUA: continue
            # 避雷
            if stk_m_arr[gi] == AVOID_STK_M: continue
            # score
            score = 0
            if stk_y_arr[gi] == '101': score += 1
            # sanhu_5d
            if gi - 5 >= s:
                sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1]))
                if sanhu_5d < -50: score += 1
                elif sanhu_5d < -30: score += 1  # 注意: 这两条互斥 (-50 是 -30 的子集), 实际是嵌套. 取 -30 阈值 (更宽)
            if score < 2: continue

            # 取段
            buy_idx = i
            end_idx = i + MAX_HOLD + 1
            if s + end_idx > e: continue
            td_seg = trend_arr[s+buy_idx:s+end_idx]
            gua_seg = stk_d_arr[s+buy_idx:s+end_idx]
            cl_seg = close_arr[s+buy_idx:s+end_idx]

            sells = calc_sells(0, MAX_HOLD + 1, td_seg, gua_seg)
            entry = {
                'date': date_arr[gi],
                'code': code_arr[gi],
                'buy_close': cl_seg[0],
            }
            for k, v in sells.items():
                if v < len(cl_seg):
                    entry[f'ret_{k}'] = (cl_seg[v] / cl_seg[0] - 1) * 100
                    entry[f'hold_{k}'] = v
                else:
                    entry[f'ret_{k}'] = float('nan')
                    entry[f'hold_{k}'] = float('nan')
            # 主升浪标记 (前 30 日内 d_gua=111 是否有 ≥10 连续)
            seg30 = stk_d_arr[s+buy_idx:s+min(buy_idx+30, e-s)]
            n_qian = (seg30 == '111').sum()
            entry['n_qian'] = int(n_qian)
            entry['is_zsl'] = n_qian >= QIAN_RUN
            entries.append(entry)

    df_en = pd.DataFrame(entries)
    print(f'  入场事件: {len(df_en):,}')

    # 标段
    df_en['seg'] = ''
    df_en.loc[(df_en['date'] >= '2019-01-01') & (df_en['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_en.loc[(df_en['date'] >= '2021-01-01') & (df_en['date'] < '2022-01-01'), 'seg'] = 'w4_2021'

    # 卖点对比
    print(f'\n## 卖点机制对比')
    print(f'  {"机制":<14} {"全期望%":>8} {"胜率%":>7} {"持仓":>5} {"主升期望%":>10} {"假期望%":>9} {"w2期望":>8} {"w4期望":>8}')
    sell_names = ['fix15', 'fix20', 'fix30', 'cross1', 'bull', 'bull_ts15', 'bull_ts20', 'm1_qian']
    for sn in sell_names:
        rcol = f'ret_{sn}'; hcol = f'hold_{sn}'
        sub = df_en.dropna(subset=[rcol])
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        print(f'  {sn:<14} {ret:>+7.2f} {win:>6.1f} {hold:>5.1f} {zsl:>+9.2f} {fake:>+8.2f} {ret_w2:>+7.2f} {ret_w4:>+7.2f}')

    # 主升浪 vs 假突破 拆开看
    print(f'\n## 主升浪事件 (n_qian>=10) 在各机制下')
    df_zsl = df_en[df_en['is_zsl']].copy()
    print(f'  主升浪入场数: {len(df_zsl)}')
    for sn in sell_names:
        rcol = f'ret_{sn}'; hcol = f'hold_{sn}'
        sub = df_zsl.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean()
        hold = sub[hcol].mean()
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        print(f'  {sn:<14} {ret:>+7.2f} 持仓 {hold:>4.1f} | w2 {ret_w2:>+7.2f} | w4 {ret_w4:>+7.2f}')

    # score=3 子集 (最优组合)
    print(f'\n## score=3 子集表现 (假设 GOOD_RULES 实际能命中所有 3 条)')
    # 重新算 score=3
    # 由于上面 sanhu_5d<-30 和 <-50 嵌套, 实际 score 算的是 stk_y=离 + sanhu<-30 OR <-50
    # 这里对原始事件再走一遍找 score=3
    # 实际本扫描里 score>=2 才进 entries, score=3 是同时命中多条
    # 由于实现里 sanhu<-50 蕴含 sanhu<-30, 算 +2 分而非 +1
    # 重新跑 score 严格区分


if __name__ == '__main__':
    main()
