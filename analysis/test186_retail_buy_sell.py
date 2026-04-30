# -*- coding: utf-8 -*-
"""新买卖规则 + 资金回测 + 单笔评估

入场预筛 (跟之前相同, 限 2016+):
  - mkt 阴阳 = 000
  - stk 阴阳 = 011
  - cur_mf ≤ -100, cur_retail ≤ -100
  - 触发日 = trend 下穿 11 当天

建仓 (从信号日+1 起 60 天内):
  - 散户线 上穿 0 (前 retail<=0, 当 retail>0) AND trend>11

之后 (D6 改, T0 不变):
  - 卖出 (新 D6'): 散户线连续 2 天 ≤ 0 (前一天 retail<=0, 当天 retail<=0)
  - 再买入 (U1 不变): mf 上升 (mf_chg > 0)
  - T0 清仓: trend < 11 (波段终结)

对比:
  - 老版 D6/U1 (test181): K=3 +86.21% / -57% MDD
  - 新版只 T0 (简化): 不切换, 拿全程
  - 新版散户线买卖

3 套都跑一遍.
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
WAIT_MAX = 60


def find_signals(arrays, mkt_arrs):
    cs = arrays['starts']; ce = arrays['ends']
    td = arrays['td']; close = arrays['close']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d_t = arrays['stk_d_t']; stk_m_t = arrays['stk_m_t']; stk_y_t = arrays['stk_y_t']
    mkt_d_t = mkt_arrs['mkt_d_t']; mkt_m_t = mkt_arrs['mkt_m_t']; mkt_y_t = mkt_arrs['mkt_y_t']
    date = arrays['date']; code = arrays['code']
    sigs = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur) or np.isnan(prev): continue
            if prev > 11 and cur <= 11:
                cond_mkt = (not np.isnan(mkt_d_t[i]) and mkt_d_t[i] <= 50 and
                              not np.isnan(mkt_m_t[i]) and mkt_m_t[i] <= 50 and
                              not np.isnan(mkt_y_t[i]) and mkt_y_t[i] <= 50)
                cond_stk = (not np.isnan(stk_d_t[i]) and stk_d_t[i] <= 50 and
                              not np.isnan(stk_m_t[i]) and stk_m_t[i] > 50 and
                              not np.isnan(stk_y_t[i]) and stk_y_t[i] > 50)
                cond_mf = (not np.isnan(mf[i]) and mf[i] <= -100)
                cond_ret = (not np.isnan(retail[i]) and retail[i] <= -100)
                if cond_mkt and cond_stk and cond_mf and cond_ret:
                    sigs.append({'signal_idx': i,'signal_date': date[i],'code': code[i],
                                 'cur_mf': mf[i],'cur_retail': retail[i],'code_end': e})
    return pd.DataFrame(sigs)


def find_entry_retail_cross(signal_idx, code_end, td, mf, retail):
    """新买点: 散户线上穿 0 (前<=0, 当>0) AND trend>11"""
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if td[k] <= 11: continue
        if retail[k-1] <= 0 and retail[k] > 0:
            return k
    return -1


def find_entry_old(signal_idx, code_end, td, mf, retail):
    """老买点 E2+E3"""
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(td[k-1]): continue
        if td[k] <= 11: continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        mf_c = mf[k] - mf[k-1]; ret_c = retail[k] - retail[k-1]
        if mf_c > 0 and ret_c > 0:
            return k
    return -1


def simulate_old_d6u1(buy_idx, code_end, td, close, mf, retail):
    """老版: D6 卖, U1 买, T0 清"""
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return k, 'T0', (cum-1)*100, legs
        if k<1: continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]) or np.isnan(retail[k]) or np.isnan(retail[k-1]) or np.isnan(td[k-1]): continue
        mfc = mf[k]-mf[k-1]; rc = retail[k]-retail[k-1]; tc = td[k]-td[k-1]
        if holding:
            if mfc<0 and rc<0 and tc<0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc>0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return code_end-1, 'fc', (cum-1)*100, legs


def simulate_only_t0(buy_idx, code_end, td, close):
    """简化: 入场后只用 T0 (trend<11) 一次性卖"""
    bp = close[buy_idx]
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            return k, 'T0', (close[k]/bp-1)*100, 1
    return code_end-1, 'fc', (close[code_end-1]/bp-1)*100, 1


def simulate_new_retail(buy_idx, code_end, td, close, mf, retail):
    """新版: 散户线连续 2 天 <=0 卖, mf 上升买回, T0 清"""
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return k, 'T0', (cum-1)*100, legs
        if k<1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        mfc = mf[k]-mf[k-1]
        if holding:
            # 散户线连续 2 天 <=0
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc>0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return code_end-1, 'fc', (cum-1)*100, legs


def simulate_new_retail_v2(buy_idx, code_end, td, close, mf, retail):
    """新版 v2: 散户线连续 2 天 <=0 卖, 散户线再上穿 0 买回, T0 清"""
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return k, 'T0', (cum-1)*100, legs
        if k<1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if holding:
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            # 散户线上穿 0
            if retail[k-1] <= 0 and retail[k] > 0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return code_end-1, 'fc', (cum-1)*100, legs


def main():
    t0 = time.time()
    print('=== test186: 新买卖规则 (散户线上穿 0 / 连续 2 天<=0) ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'm_trend', 'y_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_trend', 'm_trend', 'y_trend'])
    mkt['date'] = mkt['date'].astype(str)
    mkt = mkt.drop_duplicates('date').rename(columns={
        'd_trend':'mkt_d_t', 'm_trend':'mkt_m_t', 'y_trend':'mkt_y_t'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_d_t = df['d_trend'].to_numpy().astype(np.float64)
    stk_m_t = df['m_trend'].to_numpy().astype(np.float64)
    stk_y_t = df['y_trend'].to_numpy().astype(np.float64)
    mkt_d_t = df['mkt_d_t'].to_numpy().astype(np.float64)
    mkt_m_t = df['mkt_m_t'].to_numpy().astype(np.float64)
    mkt_y_t = df['mkt_y_t'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'close':close_arr,'td':trend_arr,
              'retail':retail_arr,'mf':mf_arr,
              'stk_d_t':stk_d_t,'stk_m_t':stk_m_t,'stk_y_t':stk_y_t,
              'starts':code_starts,'ends':code_ends}
    mkt_arrs = {'mkt_d_t':mkt_d_t, 'mkt_m_t':mkt_m_t, 'mkt_y_t':mkt_y_t}

    df_sig = find_signals(arrays, mkt_arrs)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'  入场预筛信号: {len(df_sig)}')

    # ===== 4 种买卖组合 =====
    configs = [
        # (买点, 卖点, label)
        ('old_e2e3',  'old_d6u1',     'A: 老买 E2+E3 / 老卖 D6+U1+T0 (test181)'),
        ('old_e2e3',  'only_t0',      'B: 老买 / 只 T0'),
        ('old_e2e3',  'new_retail',   'C: 老买 / 新卖 (散户线连续2天<=0 + U1)'),
        ('retail_up', 'old_d6u1',     'D: 新买 (散户线上穿0) / 老卖'),
        ('retail_up', 'only_t0',      'E: 新买 / 只 T0'),
        ('retail_up', 'new_retail',   'F: 新买 / 新卖 (散户线对称)'),
        ('retail_up', 'new_retail_v2','G: 新买 / 新卖 v2 (散户线对称, 买回也用上穿)'),
    ]

    print(f'\n  {"组合":<55} {"建仓":>5} {"avg":>8} {"med":>7} {"win":>5} '
          f'{"≥+50":>5} {"≥+100":>5} {"≥+200":>5} {"avg_legs":>7}')

    all_results = {}
    for entry_mode, sell_mode, label in configs:
        rets = []; legs_list = []
        n_entry = 0
        for _, s in df_sig.iterrows():
            si = int(s['signal_idx']); ce = int(s['code_end'])
            if entry_mode == 'old_e2e3':
                ei = find_entry_old(si, ce, trend_arr, mf_arr, retail_arr)
            else:
                ei = find_entry_retail_cross(si, ce, trend_arr, mf_arr, retail_arr)
            if ei < 0: continue
            n_entry += 1

            if sell_mode == 'old_d6u1':
                _, _, ret, legs = simulate_old_d6u1(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'only_t0':
                _, _, ret, legs = simulate_only_t0(ei, ce, trend_arr, close_arr)
            elif sell_mode == 'new_retail':
                _, _, ret, legs = simulate_new_retail(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'new_retail_v2':
                _, _, ret, legs = simulate_new_retail_v2(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            rets.append(ret); legs_list.append(legs)

        rets = np.array(rets); legs_arr = np.array(legs_list)
        n50 = (rets>=50).sum(); n100 = (rets>=100).sum(); n200 = (rets>=200).sum()
        all_results[label] = {'rets':rets, 'n_entry':n_entry, 'legs':legs_arr}
        if len(rets) == 0:
            print(f'  {label:<55} {n_entry:>5} {"--":>8}')
            continue
        print(f'  {label:<55} {n_entry:>5} {rets.mean():>+7.2f}% {np.median(rets):>+6.1f}% '
              f'{(rets>0).mean()*100:>4.1f}% {n50:>5} {n100:>5} {n200:>5} '
              f'{legs_arr.mean():>+6.1f}')

    # 按年看 G (期望最强的版本)
    print(f'\n  {"="*82}')
    print(f'  详细按年: 老 vs 新 (前 4 套)')
    print(f'  {"="*82}')

    for label, res in all_results.items():
        rets = res['rets']; n_entry = res['n_entry']
        if len(rets) == 0: continue
        print(f'\n  --- {label} ---')

    # 重新用更细的 跨年表
    print(f'\n  按年 r100 (≥+100% 暴涨股命中比例) — 7 个组合对比:')
    # 重跑一次 + 记年 + 记单笔 ret
    rows = []
    for entry_mode, sell_mode, label in configs:
        for _, s in df_sig.iterrows():
            si = int(s['signal_idx']); ce = int(s['code_end'])
            if entry_mode == 'old_e2e3':
                ei = find_entry_old(si, ce, trend_arr, mf_arr, retail_arr)
            else:
                ei = find_entry_retail_cross(si, ce, trend_arr, mf_arr, retail_arr)
            if ei < 0: continue

            if sell_mode == 'old_d6u1':
                _, _, ret, legs = simulate_old_d6u1(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'only_t0':
                _, _, ret, legs = simulate_only_t0(ei, ce, trend_arr, close_arr)
            elif sell_mode == 'new_retail':
                _, _, ret, legs = simulate_new_retail(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'new_retail_v2':
                _, _, ret, legs = simulate_new_retail_v2(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            rows.append({'cfg':label, 'year':date_arr[ei][:4], 'ret':ret})
    df_y = pd.DataFrame(rows)
    print(f'\n  {"组合":<55}', end='')
    years = sorted(df_y['year'].unique())
    for y in years: print(f' {y[-2:]:>5}', end='')
    print(f' {"总":>6}')
    for label in [c[2] for c in configs]:
        sub = df_y[df_y['cfg']==label]
        print(f'  {label:<55}', end='')
        for y in years:
            sy = sub[sub['year']==y]
            if len(sy) == 0: print(f' {"--":>5}', end=''); continue
            avg = sy['ret'].mean()
            print(f' {avg:>+4.1f}%', end='')
        avg_total = sub['ret'].mean() if len(sub) else 0
        print(f' {avg_total:>+5.2f}%')

    # 看 002068 案例
    print(f'\n  {"="*82}')
    print(f'  002068 (2022-04-20 信号) 各方案表现')
    print(f'  {"="*82}')
    sig_002068 = df_sig[(df_sig['code']=='002068') & (df_sig['signal_date']=='2022-04-20')]
    if len(sig_002068) > 0:
        s = sig_002068.iloc[0]
        si = int(s['signal_idx']); ce = int(s['code_end'])
        print(f'\n  signal {s["signal_date"]}, code_end_idx={ce}')

        # 各种入场点
        ei_old = find_entry_old(si, ce, trend_arr, mf_arr, retail_arr)
        ei_new = find_entry_retail_cross(si, ce, trend_arr, mf_arr, retail_arr)
        if ei_old >= 0:
            print(f'  老入场点 (E2+E3): {date_arr[ei_old]} @ {close_arr[ei_old]:.2f}')
        if ei_new >= 0:
            print(f'  新入场点 (散户线上穿0): {date_arr[ei_new]} @ {close_arr[ei_new]:.2f}')

        for entry_mode, sell_mode, label in configs:
            ei = ei_old if entry_mode == 'old_e2e3' else ei_new
            if ei < 0: continue
            if sell_mode == 'old_d6u1':
                sk, r, ret, legs = simulate_old_d6u1(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'only_t0':
                sk, r, ret, legs = simulate_only_t0(ei, ce, trend_arr, close_arr)
            elif sell_mode == 'new_retail':
                sk, r, ret, legs = simulate_new_retail(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            elif sell_mode == 'new_retail_v2':
                sk, r, ret, legs = simulate_new_retail_v2(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
            print(f'  {label:<55} ret={ret:>+6.1f}% legs={legs}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
