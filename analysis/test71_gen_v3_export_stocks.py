# -*- coding: utf-8 -*-
"""导出艮 regime v3 入场票, 给用户去看 K 线图验证

输出 4 类样本:
  A. 主升浪 + m_dui 触发 (理想成功案例)
  B. 主升浪 + timeout (m_dui 没出, 但赚钱)
  C. 假突破 + 止损区域 (-10% 内出局)
  D. 同 w4_2021 同日多只信号 (看实盘的资金分配场景)

每类各 5 只
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 180
QIAN_RUN = 10


def main():
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

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    rows = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != '001': continue
            if stk_d_arr[gi] != '011': continue
            if stk_m_arr[gi] == '111': continue
            score = 0
            if stk_y_arr[gi] == '101': score += 1
            if gi - 5 >= s:
                sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1]))
                if sanhu_5d < -50: score += 1
                elif sanhu_5d < -30: score += 1
            if score < 2: continue

            buy = i; end = i + MAX_HOLD + 1
            if s + end > e: continue
            gua_seg = stk_d_arr[s+buy:s+end]
            stk_m_seg = stk_m_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]
            td_seg = trend_arr[s+buy:s+end]

            # v3 状态机
            in_state = False
            sell_idx = MAX_HOLD; sell_type = 'timeout'
            for k in range(1, len(td_seg)):
                if not in_state:
                    if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                        in_state = True
                    continue
                if stk_m_seg[k] == '110':
                    sell_idx = k; sell_type = 'm_dui'; break

            n_qian = int((gua_seg[:31] == '111').sum())
            ret = (cl_seg[sell_idx] / cl_seg[0] - 1) * 100
            ret_max = (max(cl_seg[:sell_idx+1]) / cl_seg[0] - 1) * 100
            ret_min = (min(cl_seg[:sell_idx+1]) / cl_seg[0] - 1) * 100
            td_max = float(np.nanmax(td_seg[:sell_idx+1]))

            rows.append({
                'date': date_arr[gi],
                'code': code_arr[gi],
                'sell_type': sell_type,
                'hold': sell_idx,
                'is_zsl': n_qian >= QIAN_RUN,
                'n_qian': n_qian,
                'ret': ret,
                'ret_max': ret_max,
                'ret_min': ret_min,
                'td_max': td_max,
                'score': score,
            })

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    # ===== 类 A: 主升浪 + m_dui 触发 (理想成功) =====
    print(f'\n## A. 主升浪 + m_dui 触发 (理想成功案例, 卖在月卦兑顶)')
    a_pool = df_h[(df_h['is_zsl']) & (df_h['sell_type'] == 'm_dui')].copy()
    a_pool = a_pool.sort_values('ret', ascending=False)
    a_top = a_pool.head(8)
    print(f'  {"日期":<12} {"代码":<8} {"段":<10} {"持仓":>4} {"乾天":>4} {"收益%":>7} {"最大涨%":>8} {"td_max":>7}')
    for _, r in a_top.iterrows():
        print(f'  {r["date"]:<12} {r["code"]:<8} {r["seg"]:<10} {int(r["hold"]):>4} {r["n_qian"]:>4} {r["ret"]:>+6.2f} {r["ret_max"]:>+7.2f} {r["td_max"]:>6.1f}')

    # ===== 类 B: 主升浪 + timeout (没卖在顶, 还在涨) =====
    print(f'\n## B. 主升浪 + timeout (m_dui 没出, 等了 180 日, 看是否真还在涨)')
    b_pool = df_h[(df_h['is_zsl']) & (df_h['sell_type'] == 'timeout')].copy()
    b_pool = b_pool.sort_values('ret', ascending=False)
    b_top = b_pool.head(8)
    print(f'  {"日期":<12} {"代码":<8} {"段":<10} {"持仓":>4} {"乾天":>4} {"收益%":>7} {"最大涨%":>8} {"td_max":>7}')
    for _, r in b_top.iterrows():
        print(f'  {r["date"]:<12} {r["code"]:<8} {r["seg"]:<10} {int(r["hold"]):>4} {r["n_qian"]:>4} {r["ret"]:>+6.2f} {r["ret_max"]:>+7.2f} {r["td_max"]:>6.1f}')

    # ===== 类 C: 假突破 (-10%~-20% 区间) =====
    print(f'\n## C. 假突破样本 (亏 -5%~-25%, 验证假突破特征)')
    c_pool = df_h[(~df_h['is_zsl']) & (df_h['ret'] < -3) & (df_h['ret'] > -25)].copy()
    c_pool = c_pool.sort_values('ret').head(8)
    print(f'  {"日期":<12} {"代码":<8} {"段":<10} {"持仓":>4} {"乾天":>4} {"收益%":>7} {"最大跌%":>8} {"td_max":>7}')
    for _, r in c_pool.iterrows():
        print(f'  {r["date"]:<12} {r["code"]:<8} {r["seg"]:<10} {int(r["hold"]):>4} {r["n_qian"]:>4} {r["ret"]:>+6.2f} {r["ret_min"]:>+7.2f} {r["td_max"]:>6.1f}')

    # ===== 类 D: 同一日多只信号 (实盘资金分配场景) =====
    print(f'\n## D. 同一日触发多只信号的日子 (实盘资金分配场景)')
    counts = df_h.groupby('date').size().sort_values(ascending=False)
    top_dates = counts.head(5).index
    for d in top_dates:
        sub = df_h[df_h['date'] == d].sort_values('ret', ascending=False)
        print(f'\n  {d} ({len(sub)} 只):')
        print(f'    {"代码":<8} {"持仓":>4} {"乾天":>4} {"收益%":>7} {"卖点":>10}')
        for _, r in sub.head(10).iterrows():
            print(f'    {r["code"]:<8} {int(r["hold"]):>4} {r["n_qian"]:>4} {r["ret"]:>+6.2f} {r["sell_type"]:>10}')

    # ===== 总结统计 =====
    print(f'\n## v3 状态机退出统计 (用于看图前知道大致预期)')
    grp = df_h.groupby(['is_zsl', 'sell_type']).agg(
        n=('ret', 'count'),
        avg_hold=('hold', 'mean'),
        avg_ret=('ret', 'mean'),
        avg_max=('ret_max', 'mean'),
        avg_min=('ret_min', 'mean'),
    ).round(2)
    print(grp)


if __name__ == '__main__':
    main()
