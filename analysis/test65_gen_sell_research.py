# -*- coding: utf-8 -*-
"""艮 regime 卖点深度研究: 找真触发型信号

放弃 d_trend 穿 89 系列 (在艮上失效), 测艮特有信号:

机制清单:
  T1: 乾→其他 (M1)
  T2: 乾→质变 (乾→坤/坎/艮)
  T3: 乾→兑 (乾→中期顶)
  T4: 主力线 mf 转负 (>+50 → <0)
  T5: 主力线 mf 急跌 (5 日内 mf 跌 ≥80)
  T6: 散户线 sanhu 急升 (5 日均 > +30)
  T7: 趋势线 trend 见顶下行 (从最高点跌 ≥10)
  T8: 趋势线 trend 见顶下行 ≥20
  T9: 个股 m_gua 切到 110 兑 (中期顶)
  T10: 跌破入场价 -8%
  T11: 跌破入场价 -10%
  T12: 跌破前 5 日最低价
  T13: T2 (乾→质变) + T11 (-10% 止损) 兜底, 不挂 60 日
  T14: T1 (乾→其他) + T11 (-10% 止损) + 60 日兜底
  T15: T7 (trend 顶 -10) + T11 兜底
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


def find_qian_change(gua_seg, target_set=None):
    """乾 → target_set 中任一卦切换. target_set=None 表示乾→任何非乾"""
    n = len(gua_seg)
    in_qian = False
    for k in range(n):
        if gua_seg[k] == '111':
            in_qian = True
        elif in_qian:
            if target_set is None or gua_seg[k] in target_set:
                return k
    return n - 1


def find_mf_neg(mf_seg, was_pos_thresh=50):
    """主力线由 ≥+50 转 <0"""
    n = len(mf_seg)
    was_pos = False
    for k in range(n):
        if mf_seg[k] >= was_pos_thresh:
            was_pos = True
        elif was_pos and mf_seg[k] < 0:
            return k
    return n - 1


def find_mf_drop(mf_seg, drop=80, win=5):
    """主力线 5 日内跌 ≥80"""
    n = len(mf_seg)
    for k in range(win, n):
        if mf_seg[k-win] - mf_seg[k] >= drop:
            return k
    return n - 1


def find_sanhu_high(sanhu_seg, thresh=30, win=5):
    """散户线 5 日均 > +30"""
    n = len(sanhu_seg)
    for k in range(win-1, n):
        s5 = np.nanmean(sanhu_seg[k-win+1:k+1])
        if s5 > thresh:
            return k
    return n - 1


def find_trend_drop(trend_seg, drop=10):
    """趋势线从最高点跌 ≥drop"""
    n = len(trend_seg)
    if n == 0: return 0
    running_max = trend_seg[0]
    for k in range(1, n):
        if not np.isnan(trend_seg[k]):
            running_max = max(running_max, trend_seg[k])
        if not np.isnan(trend_seg[k]) and running_max - trend_seg[k] >= drop:
            return k
    return n - 1


def find_m_gua_change(stk_m_seg, targets={'110'}):
    """个股月卦切到目标"""
    n = len(stk_m_seg)
    for k in range(n):
        if stk_m_seg[k] in targets:
            return k
    return n - 1


def find_price_stop(close_seg, drop_pct=-8):
    """跌破入场价 X%"""
    n = len(close_seg)
    p0 = close_seg[0]
    for k in range(1, n):
        if (close_seg[k] / p0 - 1) * 100 <= drop_pct:
            return k
    return n - 1


def find_break_low_n(close_seg, win=5):
    """跌破前 win 日最低价"""
    n = len(close_seg)
    for k in range(win, n):
        if close_seg[k] < close_seg[k-win:k].min():
            return k
    return n - 1


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
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
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
            td_seg = trend_arr[s+buy:s+end]
            gua_seg = stk_d_arr[s+buy:s+end]
            stk_m_seg = stk_m_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]
            mf_seg = mf_arr[s+buy:s+end]
            sanhu_seg = sanhu_arr[s+buy:s+end]

            sells = {}
            sells['T1_qian_any'] = find_qian_change(gua_seg, None)
            sells['T2_qian_zhibian'] = find_qian_change(gua_seg, {'000', '010', '001'})
            sells['T3_qian_dui'] = find_qian_change(gua_seg, {'110'})
            sells['T4_mf_neg'] = find_mf_neg(mf_seg, 50)
            sells['T5_mf_drop80'] = find_mf_drop(mf_seg, 80, 5)
            sells['T6_sanhu_high30'] = find_sanhu_high(sanhu_seg, 30, 5)
            sells['T7_trend_drop10'] = find_trend_drop(td_seg, 10)
            sells['T8_trend_drop20'] = find_trend_drop(td_seg, 20)
            sells['T9_m_dui'] = find_m_gua_change(stk_m_seg, {'110'})
            sells['T10_p_drop8'] = find_price_stop(cl_seg, -8)
            sells['T11_p_drop10'] = find_price_stop(cl_seg, -10)
            sells['T12_break_low5'] = find_break_low_n(cl_seg, 5)

            # 组合: 触发信号 + 止损
            t2 = sells['T2_qian_zhibian']; t11 = sells['T11_p_drop10']
            sells['T13_zhibian_or_drop10'] = min(t2, t11)
            t1 = sells['T1_qian_any']
            sells['T14_qian_or_drop10'] = min(t1, t11)
            t7 = sells['T7_trend_drop10']
            sells['T15_trend10_or_drop10'] = min(t7, t11)

            n_qian = int((gua_seg[:31] == '111').sum())

            row = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN}
            for name, k in sells.items():
                if k < len(cl_seg):
                    row[f'r_{name}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                    row[f'h_{name}'] = k
                    # 是否 timeout (= MAX_HOLD)
                    row[f'to_{name}'] = (k >= MAX_HOLD - 1)
                else:
                    row[f'r_{name}'] = float('nan')
                    row[f'h_{name}'] = float('nan')
                    row[f'to_{name}'] = True
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    print(f'\n## 候选卖点全表')
    print(f'  {"机制":<25} {"全期望":>7} {"胜率":>6} {"持仓":>5} {"主升期":>8} {"假期":>7} {"w2":>7} {"w4":>7} {"timeout%":>9} {"日bps":>7}')
    names = ['T1_qian_any', 'T2_qian_zhibian', 'T3_qian_dui',
             'T4_mf_neg', 'T5_mf_drop80', 'T6_sanhu_high30',
             'T7_trend_drop10', 'T8_trend_drop20',
             'T9_m_dui',
             'T10_p_drop8', 'T11_p_drop10', 'T12_break_low5',
             'T13_zhibian_or_drop10', 'T14_qian_or_drop10', 'T15_trend10_or_drop10']
    for name in names:
        rcol = f'r_{name}'; hcol = f'h_{name}'; tocol = f'to_{name}'
        sub = df_h.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        to_pct = sub[tocol].mean() * 100
        bps = ret / hold * 100 if hold > 0 else 0
        print(f'  {name:<25} {ret:>+6.2f} {win:>5.1f} {hold:>5.1f} {zsl:>+7.2f} {fake:>+6.2f} {ret_w2:>+6.2f} {ret_w4:>+6.2f} {to_pct:>8.1f} {bps:>+6.1f}')


if __name__ == '__main__':
    main()
