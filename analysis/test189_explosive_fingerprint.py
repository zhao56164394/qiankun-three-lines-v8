# -*- coding: utf-8 -*-
"""主升浪分析法 - 阶段 3 指纹观察
暴涨 (ret>=100%) vs 非暴涨 (ret<100%) 在 entry_idx 时的特征对比

事件抽取:
  全市场 trend 下穿 11 (波段起点) → F 机制 entry (散户线上穿 0)
  在 entry_idx 时提取所有特征
  暴涨样本: ret>=100% (主升浪)
  对照样本: ret<100%, 抽 5x 主升数

特征 (entry_idx 当天, 不是 signal_idx):
  1. 卦象 (二值化): 个股 d/m/y_yy, 大盘 d/m/y_yy
  2. 数值: trend, mf, retail, mf 5d/30d, retail 5d/30d, trend 5d slope
  3. 历史: 前 30 日 d_yy 各态频率, 信号至 entry 等待天数
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAIT_MAX = 60


def yyy(d, m, y, thr=50):
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def find_signals_all(starts, ends, td, date_arr, code_arr):
    """全市场 trend 下穿11 波段起点"""
    sigs = []
    for ci in range(len(starts)):
        s = starts[ci]; e = ends[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            if np.isnan(td[i]) or np.isnan(td[i-1]): continue
            if td[i-1] > 11 and td[i] <= 11:
                sigs.append((i, e))
    return sigs


def find_entry(signal_idx, code_end, td, retail):
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if np.isnan(td[k]) or np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if td[k] <= 11: continue
        if retail[k-1] <= 0 and retail[k] > 0: return k
    return -1


def simulate_F(buy_idx, code_end, td, close, mf, retail):
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return (cum-1)*100, legs
        if k < 1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        mfc = mf[k]-mf[k-1]
        if holding:
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc > 0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return (cum-1)*100, legs


def extract_features(idx, code_start, td, mf, retail,
                     stk_d_t, stk_m_t, stk_y_t,
                     mkt_d_t, mkt_m_t, mkt_y_t):
    """提取 entry_idx 时的特征. code_start 是该股第一条记录, 防越界"""
    # 阴阳
    stk_yy = yyy(stk_d_t[idx], stk_m_t[idx], stk_y_t[idx])
    mkt_yy = yyy(mkt_d_t[idx], mkt_m_t[idx], mkt_y_t[idx])

    # 数值 (当下)
    cur_td = td[idx]
    cur_mf = mf[idx]
    cur_rt = retail[idx]

    # 5d / 30d 窗口
    s5 = max(code_start, idx - 4)
    s30 = max(code_start, idx - 29)
    mf5 = np.nanmean(mf[s5:idx+1])
    mf30_mean = np.nanmean(mf[s30:idx+1])
    mf30_min = np.nanmin(mf[s30:idx+1])
    mf30_max = np.nanmax(mf[s30:idx+1])
    rt5 = np.nanmean(retail[s5:idx+1])
    rt30_mean = np.nanmean(retail[s30:idx+1])
    rt30_min = np.nanmin(retail[s30:idx+1])
    # trend slope
    td5_slope = td[idx] - td[s5] if not np.isnan(td[s5]) else np.nan
    td30_slope = td[idx] - td[s30] if not np.isnan(td[s30]) else np.nan

    # 前 30 日个股 d_yy 频率 (用 stk_d_t > 50 简化为 d 单态)
    # 这里直接看个股 d/m/y 三态在前 30 日各超过 50 的占比
    n30 = idx - s30 + 1
    if n30 > 0:
        d_pct = np.sum(stk_d_t[s30:idx+1] > 50) / n30
        m_pct = np.sum(stk_m_t[s30:idx+1] > 50) / n30
        y_pct = np.sum(stk_y_t[s30:idx+1] > 50) / n30
    else:
        d_pct = m_pct = y_pct = np.nan

    return {
        'stk_yy': stk_yy, 'mkt_yy': mkt_yy,
        'cur_td': cur_td, 'cur_mf': cur_mf, 'cur_rt': cur_rt,
        'mf5': mf5, 'mf30_mean': mf30_mean, 'mf30_min': mf30_min, 'mf30_max': mf30_max,
        'rt5': rt5, 'rt30_mean': rt30_mean, 'rt30_min': rt30_min,
        'td5_slope': td5_slope, 'td30_slope': td30_slope,
        'd_pct30': d_pct, 'm_pct30': m_pct, 'y_pct30': y_pct,
    }


def main():
    t0 = time.time()
    print('=== test189: 主升浪指纹观察 (暴涨 vs 非暴涨) ===\n')

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
    print(f'  数据: {len(df):,} 行, {df["code"].nunique()} 股 ({df["date"].min()}~{df["date"].max()})')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    td_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_d_t = df['d_trend'].to_numpy().astype(np.float64)
    stk_m_t = df['m_trend'].to_numpy().astype(np.float64)
    stk_y_t = df['y_trend'].to_numpy().astype(np.float64)
    mkt_d_t = df['mkt_d_t'].to_numpy().astype(np.float64)
    mkt_m_t = df['mkt_m_t'].to_numpy().astype(np.float64)
    mkt_y_t = df['mkt_y_t'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    starts = np.where(code_change)[0]
    ends = np.r_[starts[1:], len(code_arr)]
    code_start_for = np.zeros(len(df), dtype=np.int64)
    for ci in range(len(starts)):
        code_start_for[starts[ci]:ends[ci]] = starts[ci]

    # 1. 找信号
    print('  扫信号...')
    sigs = find_signals_all(starts, ends, td_arr, date_arr, code_arr)
    print(f'    波段起点: {len(sigs):,}')

    # 2. F 入场 + 模拟 + 提特征
    print('  F 入场 + 模拟 + 提特征...')
    rows = []
    for si, ce in sigs:
        ei = find_entry(si, ce, td_arr, retail_arr)
        if ei < 0: continue
        if date_arr[si] < '2016-01-01': continue
        ret, legs = simulate_F(ei, ce, td_arr, close_arr, mf_arr, retail_arr)
        feats = extract_features(ei, code_start_for[ei],
                                 td_arr, mf_arr, retail_arr,
                                 stk_d_t, stk_m_t, stk_y_t,
                                 mkt_d_t, mkt_m_t, mkt_y_t)
        feats['code'] = code_arr[ei]
        feats['signal_date'] = date_arr[si]
        feats['entry_date'] = date_arr[ei]
        feats['wait_days'] = ei - si
        feats['ret'] = ret
        feats['legs'] = legs
        rows.append(feats)
    df_e = pd.DataFrame(rows)
    df_e['year'] = df_e['signal_date'].str[:4]
    df_e = df_e.dropna(subset=['ret']).reset_index(drop=True)
    print(f'    入场建仓: {len(df_e):,}')

    # 暴涨 / 非暴涨
    df_e['is_explosive'] = df_e['ret'] >= 100
    df_e['is_super'] = df_e['ret'] >= 200
    n_total = len(df_e)
    n_exp = df_e['is_explosive'].sum()
    n_sup = df_e['is_super'].sum()
    print(f'\n  全样本: n={n_total:,}, 暴涨(>=100%)={n_exp:,} ({n_exp/n_total*100:.2f}%), '
          f'超级(>=200%)={n_sup:,} ({n_sup/n_total*100:.2f}%)')
    print(f'  非暴涨平均 ret: {df_e[~df_e["is_explosive"]]["ret"].mean():+.2f}%')
    print(f'  暴涨平均 ret: {df_e[df_e["is_explosive"]]["ret"].mean():+.2f}%')
    print(f'  超级平均 ret: {df_e[df_e["is_super"]]["ret"].mean():+.2f}%\n')

    # ============ 阶段 3 指纹观察 ============

    df_exp = df_e[df_e['is_explosive']].copy()
    df_non = df_e[~df_e['is_explosive']].copy()

    base_exp_rate = n_exp / n_total * 100
    base_sup_rate = n_sup / n_total * 100

    # ---- 1. 卦象指纹 ----
    print(f'{"="*82}')
    print(f'  1. 卦象指纹 (个股+大盘 阴阳)')
    print(f'{"="*82}\n')

    # stk_yy
    print(f'  --- 个股 yy ({"占比 暴涨":>12} vs {"占比 非暴涨":>12} {"差":>7} {"暴涨率 lift":>14}) ---')
    print(f'  {"yy":<5} {"暴%":>8} {"非暴%":>8} {"差(pp)":>8} '
          f'{"n_exp":>6} {"n_total":>8} {"暴涨率":>7} {"lift":>6}')
    for v, sub in df_e.groupby('stk_yy'):
        n_sub = len(sub); n_e = sub['is_explosive'].sum()
        if n_sub < 200: continue
        pct_exp = n_e/n_exp*100
        pct_non = (n_sub-n_e)/(n_total-n_exp)*100
        rate = n_e/n_sub*100
        lift = rate/base_exp_rate
        print(f'  {v:<5} {pct_exp:>+7.2f}% {pct_non:>+7.2f}% {pct_exp-pct_non:>+7.2f}pp '
              f'{n_e:>6} {n_sub:>8,} {rate:>+6.2f}% {lift:>+5.2f}x')

    # mkt_yy
    print(f'\n  --- 大盘 yy ---')
    print(f'  {"yy":<5} {"暴%":>8} {"非暴%":>8} {"差(pp)":>8} '
          f'{"n_exp":>6} {"n_total":>8} {"暴涨率":>7} {"lift":>6}')
    for v, sub in df_e.groupby('mkt_yy'):
        n_sub = len(sub); n_e = sub['is_explosive'].sum()
        if n_sub < 200: continue
        pct_exp = n_e/n_exp*100
        pct_non = (n_sub-n_e)/(n_total-n_exp)*100
        rate = n_e/n_sub*100
        lift = rate/base_exp_rate
        print(f'  {v:<5} {pct_exp:>+7.2f}% {pct_non:>+7.2f}% {pct_exp-pct_non:>+7.2f}pp '
              f'{n_e:>6} {n_sub:>8,} {rate:>+6.2f}% {lift:>+5.2f}x')

    # 64 组合
    print(f'\n  --- 64 组合 (n>=300, 按暴涨率排, lift >=1.3 或 lift <=0.5) ---')
    print(f'  {"mkt":<4} {"stk":<4} {"n":>6} {"n_exp":>6} {"暴涨率":>7} {"lift":>6} {"n_sup":>6} {"超级率":>7}')
    rows64 = []
    for (mv, sv), sub in df_e.groupby(['mkt_yy','stk_yy']):
        if len(sub) < 300: continue
        n = len(sub); h = sub['is_explosive'].sum(); s = sub['is_super'].sum()
        rows64.append({'mkt':mv,'stk':sv,'n':n,'h':h,'s':s,
                       'rate':h/n*100,'lift':(h/n*100)/base_exp_rate,
                       's_rate':s/n*100,'s_lift':(s/n*100)/base_sup_rate if base_sup_rate else 0})
    df64 = pd.DataFrame(rows64).sort_values('lift', ascending=False)
    for _, r in df64.iterrows():
        if r['lift'] >= 1.3 or r['lift'] <= 0.5:
            print(f'  {r["mkt"]:<4} {r["stk"]:<4} {r["n"]:>6,} {r["h"]:>6} '
                  f'{r["rate"]:>+6.2f}% {r["lift"]:>+5.2f}x {r["s"]:>6} {r["s_rate"]:>+6.2f}%')

    # ---- 2. 数值特征指纹 ----
    print(f'\n{"="*82}')
    print(f'  2. 数值特征 (median 暴涨 vs 非暴涨)')
    print(f'{"="*82}\n')

    num_cols = ['cur_td','cur_mf','cur_rt','mf5','mf30_mean','mf30_min','mf30_max',
                'rt5','rt30_mean','rt30_min','td5_slope','td30_slope',
                'd_pct30','m_pct30','y_pct30','wait_days','legs']
    print(f'  {"特征":<14} {"暴涨 median":>12} {"非暴 median":>12} {"diff":>10} {"暴涨 mean":>12} {"非暴 mean":>12}')
    for col in num_cols:
        em = df_exp[col].median(); nm = df_non[col].median()
        ea = df_exp[col].mean(); na = df_non[col].mean()
        print(f'  {col:<14} {em:>+11.2f} {nm:>+11.2f} {em-nm:>+9.2f} {ea:>+11.2f} {na:>+11.2f}')

    # ---- 3. 数值阈值扫: 看哪些区间 lift 高 ----
    print(f'\n{"="*82}')
    print(f'  3. 数值阈值 lift (n>=300, lift >=1.3 或 <=0.5)')
    print(f'{"="*82}\n')

    thresholds = {
        'cur_mf': [-300,-200,-100,-50,0,50,100,200],
        'cur_rt': [-400,-250,-150,-50,0,100,200,400],
        'mf30_min': [-500,-400,-300,-200,-100,0],
        'mf30_max': [-50,0,100,200,400,600],
        'rt30_min': [-500,-400,-300,-200,-100,0],
        'cur_td': [11,30,50,70,89],
        'mf5': [-200,-100,0,100,200],
        'rt5': [-200,-100,0,100,200],
        'wait_days': [3,7,15,30,45],
    }
    print(f'  {"条件":<24} {"n":>6} {"n_exp":>6} {"暴涨率":>7} {"lift":>6} {"超级 lift":>10}')
    for col, tl in thresholds.items():
        for op in ['<=','>=']:
            for thr in tl:
                if op == '<=':
                    sub = df_e[df_e[col] <= thr]
                else:
                    sub = df_e[df_e[col] >= thr]
                if len(sub) < 300 or len(sub) > n_total*0.95: continue
                h = sub['is_explosive'].sum(); s = sub['is_super'].sum()
                rate = h/len(sub)*100; lift = rate/base_exp_rate
                s_lift = (s/len(sub)*100)/base_sup_rate if base_sup_rate else 0
                if lift >= 1.3 or lift <= 0.5:
                    label = f'{col} {op} {thr}'
                    print(f'  {label:<24} {len(sub):>6,} {h:>6} {rate:>+6.2f}% {lift:>+5.2f}x {s_lift:>+9.2f}x')

    # ---- 4. 跨年稳定性 (Top 卦象组合) ----
    print(f'\n{"="*82}')
    print(f'  4. 跨年稳定性 (lift>=1.5 的 64 组合, 跨年暴涨率)')
    print(f'{"="*82}\n')

    years = sorted(df_e['year'].unique())
    print(f'  {"":<5}{"":<4}', end='')
    for y in years: print(f' {y[-2:]:>5}', end='')
    print(f' {"全 lift":>8}')

    # 全样本 baseline rate by year
    yr_base = {}
    for y in years:
        yr_base[y] = df_e[df_e['year']==y]['is_explosive'].mean()*100
    print(f'  {"baseline":<9}', end='')
    for y in years: print(f' {yr_base[y]:>+4.1f}%', end='')
    print(f' {base_exp_rate:>+7.2f}%')

    df64_top = df64[df64['lift'] >= 1.5]
    for _, r in df64_top.iterrows():
        sub = df_e[(df_e['mkt_yy']==r['mkt']) & (df_e['stk_yy']==r['stk'])]
        print(f'  {r["mkt"]:<4} {r["stk"]:<4}', end='')
        for y in years:
            ys = sub[sub['year']==y]
            if len(ys) < 20: print(f' {"--":>5}', end='')
            else:
                rr = ys['is_explosive'].mean()*100
                print(f' {rr:>+4.1f}%', end='')
        print(f' {r["lift"]:>+6.2f}x')

    # ---- 5. 落地: 找最强组合 (rate>=2.5x baseline) ----
    print(f'\n{"="*82}')
    print(f'  5. 候选入池 (lift >= 2x, n>=200)')
    print(f'{"="*82}\n')
    df_winners = df64[df64['lift'] >= 2.0]
    print(f'  共 {len(df_winners)} 组 lift>=2:')
    for _, r in df_winners.iterrows():
        print(f'  mkt={r["mkt"]} stk={r["stk"]}: n={r["n"]:,}, '
              f'暴涨率 {r["rate"]:.2f}% (lift {r["lift"]:.2f}x), 超级率 {r["s_rate"]:.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
