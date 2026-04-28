# -*- coding: utf-8 -*-
"""60 日 timeout 后的延续走势 — 是否值得拉长持仓?

测试:
  - 60 日 timeout 的 103 个事件, 在 60-90 / 60-120 / 60-180 日继续走势
  - 找到一个真正"主升浪结束"的信号

新候选:
  E1: 80 日固定
  E2: 100 日固定
  E3: 120 日固定
  E4: 主升浪 d_gua=111 跑完 (连续乾段结束) + 缓冲 N 日
  E5: trend < 50 (跌破中性)
  E6: m_gua 切到 110 兑 (中期顶)
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 180  # 拉长到 180 日看续涨
QIAN_RUN = 10


def main():
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
            cl_seg = close_arr[s+buy:s+end]
            stk_m_seg = stk_m_arr[s+buy:s+end]
            td_seg = trend_arr[s+buy:s+end]
            n = len(cl_seg)
            p0 = cl_seg[0]

            sells = {}

            # 固定窗口
            for win in [30, 45, 60, 80, 100, 120, 180]:
                k = min(win, n - 1)
                sells[f'fix{win}'] = k

            # 主升浪结束 = 持续 N 日没有乾出现
            qian_end = MAX_HOLD - 1
            last_qian = -1
            for k in range(n):
                if gua_seg[k] == '111':
                    last_qian = k
                if last_qian >= 0 and k - last_qian >= 10:
                    qian_end = k; break
            sells['qian_end_10'] = qian_end

            # trend < 50
            t50 = MAX_HOLD - 1
            for k in range(n):
                if not np.isnan(td_seg[k]) and td_seg[k] < 50:
                    # 但要先到过 70 (已在主升)
                    if k > 0 and np.nanmax(td_seg[:k+1]) >= 70:
                        t50 = k; break
            sells['trend_below_50'] = t50

            # m_gua = 110 兑 (中期顶)
            m_dui = MAX_HOLD - 1
            for k in range(n):
                if stk_m_seg[k] == '110':
                    m_dui = k; break
            sells['m_dui'] = m_dui

            # m_gua 切到 110/111 (强势卦, 表示顶)
            m_strong = MAX_HOLD - 1
            for k in range(n):
                if stk_m_seg[k] in {'110', '111'}:
                    m_strong = k; break
            sells['m_strong'] = m_strong

            n_qian = int((gua_seg[:31] == '111').sum())

            row = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN}
            for name, k in sells.items():
                if k < n:
                    row[f'r_{name}'] = (cl_seg[k] / p0 - 1) * 100
                    row[f'h_{name}'] = k
                else:
                    row[f'r_{name}'] = float('nan')
                    row[f'h_{name}'] = float('nan')
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    print(f'\n## 长持仓机制对比')
    print(f'  {"机制":<20} {"全期望":>7} {"胜率":>5} {"持仓":>6} {"主升期":>8} {"假期":>7} {"w2":>7} {"w4":>7} {"日bps":>7}')
    names = ['fix30', 'fix45', 'fix60', 'fix80', 'fix100', 'fix120', 'fix180',
             'qian_end_10', 'trend_below_50', 'm_dui', 'm_strong']
    for name in names:
        rcol = f'r_{name}'; hcol = f'h_{name}'
        sub = df_h.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        bps = ret / hold * 100 if hold > 0 else 0
        print(f'  {name:<20} {ret:>+6.2f} {win:>4.1f} {hold:>5.1f} {zsl:>+7.2f} {fake:>+6.2f} {ret_w2:>+6.2f} {ret_w4:>+6.2f} {bps:>+6.1f}')

    # 重点: 拉长后续涨幅看
    print(f'\n## 60 日 vs 80 / 100 / 120 日对比 (主升浪是否还在继续涨?)')
    zsl = df_h[df_h['is_zsl']].dropna(subset=['r_fix60', 'r_fix80', 'r_fix100', 'r_fix120', 'r_fix180'])
    print(f'  主升浪样本: {len(zsl)}')
    print(f'  60 日:  {zsl["r_fix60"].mean():+.2f}%')
    print(f'  80 日:  {zsl["r_fix80"].mean():+.2f}%')
    print(f'  100 日: {zsl["r_fix100"].mean():+.2f}%')
    print(f'  120 日: {zsl["r_fix120"].mean():+.2f}%')
    print(f'  180 日: {zsl["r_fix180"].mean():+.2f}%')

    fake = df_h[~df_h['is_zsl']].dropna(subset=['r_fix60', 'r_fix80', 'r_fix100', 'r_fix120', 'r_fix180'])
    print(f'\n  假突破样本: {len(fake)}')
    print(f'  60 日:  {fake["r_fix60"].mean():+.2f}%')
    print(f'  80 日:  {fake["r_fix80"].mean():+.2f}%')
    print(f'  100 日: {fake["r_fix100"].mean():+.2f}%')
    print(f'  120 日: {fake["r_fix120"].mean():+.2f}%')


if __name__ == '__main__':
    main()
