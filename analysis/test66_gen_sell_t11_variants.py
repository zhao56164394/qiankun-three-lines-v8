# -*- coding: utf-8 -*-
"""T11 (-10% 价格止损) 深度诊断 + 优化变种

测:
  T11a: 单独 -10%, 60 日兜底
  T11b: -10% + T2 (乾→质变) 触发, 取早的
  T11c: -8%
  T11d: -12%
  T11e: -10% + 持仓 ≥ 30 日强出 (避免继续等)
  T11f: -10% + 30 日内未涨过 +5% 强出
  T11g: 移动止损 (从最高价回撤 -10%)
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 60
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
            n = len(cl_seg)
            p0 = cl_seg[0]

            sells = {}
            sell_types = {}

            # T11a: -10% 单独, 60 日兜底
            sell_idx = MAX_HOLD; st = 'timeout'
            for k in range(1, n):
                if (cl_seg[k] / p0 - 1) * 100 <= -10:
                    sell_idx = k; st = 'stop10'; break
            sells['T11a'] = sell_idx
            sell_types['T11a'] = st

            # T11c: -8%
            sell_idx = MAX_HOLD; st = 'timeout'
            for k in range(1, n):
                if (cl_seg[k] / p0 - 1) * 100 <= -8:
                    sell_idx = k; st = 'stop8'; break
            sells['T11c'] = sell_idx
            sell_types['T11c'] = st

            # T11d: -12%
            sell_idx = MAX_HOLD; st = 'timeout'
            for k in range(1, n):
                if (cl_seg[k] / p0 - 1) * 100 <= -12:
                    sell_idx = k; st = 'stop12'; break
            sells['T11d'] = sell_idx
            sell_types['T11d'] = st

            # T11b: -10% OR 乾→质变 (取早)
            sell_idx_drop = MAX_HOLD
            for k in range(1, n):
                if (cl_seg[k] / p0 - 1) * 100 <= -10:
                    sell_idx_drop = k; break
            sell_idx_zb = MAX_HOLD
            in_qian = False
            for k in range(n):
                if gua_seg[k] == '111':
                    in_qian = True
                elif in_qian and gua_seg[k] in {'000', '010', '001'}:
                    sell_idx_zb = k; break
            sell_idx = min(sell_idx_drop, sell_idx_zb)
            if sell_idx == sell_idx_drop and sell_idx_drop < MAX_HOLD:
                st = 'stop10'
            elif sell_idx == sell_idx_zb and sell_idx_zb < MAX_HOLD:
                st = 'zhibian'
            else:
                st = 'timeout'
            sells['T11b'] = sell_idx
            sell_types['T11b'] = st

            # T11e: -10% + 持仓 ≥30 日强出
            sell_idx = MAX_HOLD; st = 'timeout'
            for k in range(1, n):
                if (cl_seg[k] / p0 - 1) * 100 <= -10:
                    sell_idx = k; st = 'stop10'; break
                if k >= 30:
                    sell_idx = k; st = 'force_30'; break
            sells['T11e'] = sell_idx
            sell_types['T11e'] = st

            # T11f: -10% + 30 日内未涨过 +5% 强出 (低于 +5% 就走)
            sell_idx = MAX_HOLD; st = 'timeout'
            running_max = p0
            for k in range(1, n):
                if cl_seg[k] > running_max:
                    running_max = cl_seg[k]
                if (cl_seg[k] / p0 - 1) * 100 <= -10:
                    sell_idx = k; st = 'stop10'; break
                if k >= 30 and (running_max / p0 - 1) * 100 < 5:
                    sell_idx = k; st = 'no_progress'; break
            sells['T11f'] = sell_idx
            sell_types['T11f'] = st

            # T11g: 移动止损 (从最高价回撤 -10%)
            sell_idx = MAX_HOLD; st = 'timeout'
            running_max = p0
            for k in range(1, n):
                if cl_seg[k] > running_max:
                    running_max = cl_seg[k]
                if (cl_seg[k] / running_max - 1) * 100 <= -10:
                    sell_idx = k; st = 'trail10'; break
            sells['T11g'] = sell_idx
            sell_types['T11g'] = st

            # T11h: 移动止损 -7%
            sell_idx = MAX_HOLD; st = 'timeout'
            running_max = p0
            for k in range(1, n):
                if cl_seg[k] > running_max:
                    running_max = cl_seg[k]
                if (cl_seg[k] / running_max - 1) * 100 <= -7:
                    sell_idx = k; st = 'trail7'; break
            sells['T11h'] = sell_idx
            sell_types['T11h'] = st

            # T11i: 移动止损 -8%
            sell_idx = MAX_HOLD; st = 'timeout'
            running_max = p0
            for k in range(1, n):
                if cl_seg[k] > running_max:
                    running_max = cl_seg[k]
                if (cl_seg[k] / running_max - 1) * 100 <= -8:
                    sell_idx = k; st = 'trail8'; break
            sells['T11i'] = sell_idx
            sell_types['T11i'] = st

            n_qian = int((gua_seg[:31] == '111').sum())

            row = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN}
            for name, k in sells.items():
                if k < n:
                    row[f'r_{name}'] = (cl_seg[k] / p0 - 1) * 100
                    row[f'h_{name}'] = k
                    row[f't_{name}'] = sell_types[name]
                else:
                    row[f'r_{name}'] = float('nan')
                    row[f'h_{name}'] = float('nan')
                    row[f't_{name}'] = 'timeout'
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    print(f'\n## T11 价格止损系列对比')
    print(f'  {"机制":<8} {"全期望":>7} {"胜率":>5} {"持仓":>5} {"主升期":>7} {"假期":>6} {"w2":>6} {"w4":>6} {"日bps":>6}')
    names = ['T11a', 'T11b', 'T11c', 'T11d', 'T11e', 'T11f', 'T11g', 'T11h', 'T11i']
    descs = {
        'T11a': '-10% 60d兜底',
        'T11b': '-10% OR 质变',
        'T11c': '-8% 60d兜底',
        'T11d': '-12% 60d兜底',
        'T11e': '-10% OR 30d强出',
        'T11f': '-10% OR 30d无进展',
        'T11g': '移动止损 -10%',
        'T11h': '移动止损 -7%',
        'T11i': '移动止损 -8%',
    }
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
        desc = descs.get(name, '')
        print(f'  {name:<6} {ret:>+6.2f} {win:>4.1f} {hold:>5.1f} {zsl:>+6.2f} {fake:>+5.2f} {ret_w2:>+5.2f} {ret_w4:>+5.2f} {bps:>+5.1f}   {desc}')

    # T11a 退出类型分布 (诊断)
    print(f'\n## T11a (-10% / 60d兜底) 退出类型分布')
    sub = df_h.dropna(subset=['r_T11a'])
    type_dist = sub['t_T11a'].value_counts()
    for t, n in type_dist.items():
        s2 = sub[sub['t_T11a'] == t]
        print(f'  {t:<10} {n:>4} ({n/len(sub)*100:.1f}%) 持仓 {s2["h_T11a"].mean():.1f} 期望 {s2["r_T11a"].mean():+.2f}%')

    # 移动止损 T11g 退出类型
    print(f'\n## T11g (移动止损 -10%) 退出类型分布')
    sub = df_h.dropna(subset=['r_T11g'])
    type_dist = sub['t_T11g'].value_counts()
    for t, n in type_dist.items():
        s2 = sub[sub['t_T11g'] == t]
        print(f'  {t:<10} {n:>4} ({n/len(sub)*100:.1f}%) 持仓 {s2["h_T11g"].mean():.1f} 期望 {s2["r_T11g"].mean():+.2f}%')


if __name__ == '__main__':
    main()
