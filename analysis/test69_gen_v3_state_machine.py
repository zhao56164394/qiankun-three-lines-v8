# -*- coding: utf-8 -*-
"""td80_mdui 卖点深度诊断 + 加止损保险

验证:
  - 退出类型分布
  - 主升浪 vs 假突破 各退出类型分布
  - 加 -10% 止损后效果
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 180
QIAN_RUN = 10


def sell_v3_state(td_seg, gua_seg, stk_m_seg, cl_seg, stop_pct=-10):
    """v3 卖点: td80 进入态 + m_dui 退出 + -10% 止损保险"""
    n = len(td_seg)
    p0 = cl_seg[0]
    in_state = False

    for k in range(1, n):
        # 价格止损 (任何阶段)
        if (cl_seg[k] / p0 - 1) * 100 <= stop_pct:
            return k, 'stop_pct'

        # 进入态
        if not in_state:
            if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                in_state = True
            continue

        # 退出: 月卦兑
        if stk_m_seg[k] == '110':
            return k, 'm_dui'

    return n - 1, 'timeout'


def sell_v3_no_stop(td_seg, gua_seg, stk_m_seg, cl_seg):
    """同上但无止损"""
    n = len(td_seg)
    in_state = False

    for k in range(1, n):
        if not in_state:
            if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                in_state = True
            continue
        if stk_m_seg[k] == '110':
            return k, 'm_dui'

    return n - 1, 'timeout'


def sell_v3_strict(td_seg, gua_seg, stk_m_seg, cl_seg):
    """更严格: td80 + m_dui OR m_qian (任一强势卦切换)"""
    n = len(td_seg)
    p0 = cl_seg[0]
    in_state = False

    for k in range(1, n):
        if (cl_seg[k] / p0 - 1) * 100 <= -10:
            return k, 'stop_pct'
        if not in_state:
            if not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                in_state = True
            continue
        if stk_m_seg[k] in {'110', '111'}:
            return k, 'm_strong'

    return n - 1, 'timeout'


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
            stk_m_seg = stk_m_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]
            td_seg = trend_arr[s+buy:s+end]
            n_qian = int((gua_seg[:31] == '111').sum())

            row = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN}

            for name, fn in [('v3_main', sell_v3_state),
                              ('v3_nostop', sell_v3_no_stop),
                              ('v3_strict', sell_v3_strict)]:
                k, st = fn(td_seg, gua_seg, stk_m_seg, cl_seg)
                row[f'r_{name}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                row[f'h_{name}'] = k
                row[f't_{name}'] = st
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    print(f'\n## v3 状态机卖点对比')
    print(f'  {"机制":<14} {"全期望":>7} {"胜率":>5} {"持仓":>5} {"主升期":>8} {"假期":>7} {"w2":>7} {"w4":>7}')
    for name, label in [('v3_main', 'td80→m_dui+止损'),
                         ('v3_nostop', 'td80→m_dui 无止损'),
                         ('v3_strict', 'td80→m_strong+止损')]:
        rcol = f'r_{name}'; hcol = f'h_{name}'; tcol = f't_{name}'
        sub = df_h.dropna(subset=[rcol])
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean()
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean()
        print(f'  {label:<22} {ret:>+6.2f} {win:>4.1f} {hold:>5.1f} {zsl:>+7.2f} {fake:>+6.2f} {ret_w2:>+6.2f} {ret_w4:>+6.2f}')

    # v3_main 退出类型
    print(f'\n## v3_main (td80→m_dui+止损) 退出类型分布')
    sub = df_h
    type_dist = sub['t_v3_main'].value_counts()
    for t, n_ in type_dist.items():
        s2 = sub[sub['t_v3_main'] == t]
        print(f'  {t:<14} {n_:>4} ({n_/len(sub)*100:.1f}%) 持仓 {s2["h_v3_main"].mean():.1f} 期望 {s2["r_v3_main"].mean():+.2f}%')

    # 主升 vs 假
    print(f'\n## 主升浪事件下的 v3_main 退出')
    zsl = df_h[df_h['is_zsl']]
    type_dist = zsl['t_v3_main'].value_counts()
    for t, n_ in type_dist.items():
        s2 = zsl[zsl['t_v3_main'] == t]
        print(f'  {t:<14} {n_:>4} ({n_/len(zsl)*100:.1f}%) 持仓 {s2["h_v3_main"].mean():.1f} 期望 {s2["r_v3_main"].mean():+.2f}%')

    print(f'\n## 假突破事件下的 v3_main 退出')
    fake = df_h[~df_h['is_zsl']]
    type_dist = fake['t_v3_main'].value_counts()
    for t, n_ in type_dist.items():
        s2 = fake[fake['t_v3_main'] == t]
        print(f'  {t:<14} {n_:>4} ({n_/len(fake)*100:.1f}%) 持仓 {s2["h_v3_main"].mean():.1f} 期望 {s2["r_v3_main"].mean():+.2f}%')


if __name__ == '__main__':
    main()
