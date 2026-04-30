# -*- coding: utf-8 -*-
"""主升浪分析法 - 阶段 4+5 反向避雷 + 软排名

数据范围: 2016-01-01 ~ 2026-04-21
walk-forward 5 段:
  w1 = 2016-2017 (横盘+小盘余波)
  w2 = 2018-2019 (熊+反弹)
  w3 = 2020-2021 (大牛+抱团崩)
  w4 = 2022-2023 (熊+横盘)
  w5 = 2024-2026 (新牛起爆)

阶段 4 反向避雷:
  扫所有候选条件 (单维卦象 + 数值阈值)
  跨段判定: ≥3/5 段 lift<0.7 且 ≤1 段 lift>1.3 → 加入避雷名单

阶段 5 软排名:
  扫所有候选条件
  跨段判定: ≥3/5 段 lift>1.3 且 ≤1 段 lift<0.7 → 加入好规律
  IS = w1+w2+w3, OOS = w4+w5 拆开验证

事件: F 机制 entry 后是否暴涨 (ret>=100%)
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


def find_signals_all(starts, ends, td):
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
    stk_yy = yyy(stk_d_t[idx], stk_m_t[idx], stk_y_t[idx])
    mkt_yy = yyy(mkt_d_t[idx], mkt_m_t[idx], mkt_y_t[idx])
    cur_td = td[idx]; cur_mf = mf[idx]; cur_rt = retail[idx]
    s5 = max(code_start, idx - 4)
    s30 = max(code_start, idx - 29)
    mf5 = np.nanmean(mf[s5:idx+1])
    rt5 = np.nanmean(retail[s5:idx+1])
    mf30_min = np.nanmin(mf[s30:idx+1])
    mf30_max = np.nanmax(mf[s30:idx+1])
    rt30_min = np.nanmin(retail[s30:idx+1])
    rt30_mean = np.nanmean(retail[s30:idx+1])
    mf30_mean = np.nanmean(mf[s30:idx+1])
    return {
        'stk_yy': stk_yy, 'mkt_yy': mkt_yy,
        'stk_d1': '1' if (not np.isnan(stk_d_t[idx]) and stk_d_t[idx] > 50) else '0',
        'stk_m1': '1' if (not np.isnan(stk_m_t[idx]) and stk_m_t[idx] > 50) else '0',
        'stk_y1': '1' if (not np.isnan(stk_y_t[idx]) and stk_y_t[idx] > 50) else '0',
        'mkt_d1': '1' if (not np.isnan(mkt_d_t[idx]) and mkt_d_t[idx] > 50) else '0',
        'mkt_m1': '1' if (not np.isnan(mkt_m_t[idx]) and mkt_m_t[idx] > 50) else '0',
        'mkt_y1': '1' if (not np.isnan(mkt_y_t[idx]) and mkt_y_t[idx] > 50) else '0',
        'cur_td': cur_td, 'cur_mf': cur_mf, 'cur_rt': cur_rt,
        'mf5': mf5, 'rt5': rt5,
        'mf30_min': mf30_min, 'mf30_max': mf30_max, 'mf30_mean': mf30_mean,
        'rt30_min': rt30_min, 'rt30_mean': rt30_mean,
    }


def assign_seg(year):
    """5 段 walk-forward"""
    y = int(year)
    if y <= 2017: return 'w1'
    if y <= 2019: return 'w2'
    if y <= 2021: return 'w3'
    if y <= 2023: return 'w4'
    return 'w5'


def lift_per_seg(df_e, mask, segs, base_per_seg, min_n=50):
    """每段 lift; 段 n<min_n 返回 nan"""
    out = {}
    sub = df_e[mask]
    for s in segs:
        sub_s = sub[sub['seg'] == s]
        n = len(sub_s)
        if n < min_n:
            out[s] = (np.nan, n)
        else:
            rate = sub_s['is_explosive'].mean() * 100
            out[s] = (rate / base_per_seg[s] if base_per_seg[s] > 0 else np.nan, n)
    return out


def main():
    t0 = time.time()
    print('=== test190: 阶段 4+5 反向避雷 + 软排名 ===\n')

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

    print('  扫信号 + F 入场 + 提特征...')
    sigs = find_signals_all(starts, ends, td_arr)
    rows = []
    for si, ce in sigs:
        if date_arr[si] < '2016-01-01': continue
        ei = find_entry(si, ce, td_arr, retail_arr)
        if ei < 0: continue
        ret, legs = simulate_F(ei, ce, td_arr, close_arr, mf_arr, retail_arr)
        feats = extract_features(ei, code_start_for[ei],
                                 td_arr, mf_arr, retail_arr,
                                 stk_d_t, stk_m_t, stk_y_t,
                                 mkt_d_t, mkt_m_t, mkt_y_t)
        feats['signal_date'] = date_arr[si]
        feats['ret'] = ret
        rows.append(feats)
    df_e = pd.DataFrame(rows)
    df_e = df_e.dropna(subset=['ret']).reset_index(drop=True)
    df_e['year'] = df_e['signal_date'].str[:4]
    df_e['seg'] = df_e['year'].apply(assign_seg)
    df_e['is_explosive'] = df_e['ret'] >= 100

    n_total = len(df_e)
    base_rate = df_e['is_explosive'].mean() * 100
    print(f'  全样本: n={n_total:,}, 暴涨率 {base_rate:.2f}%')
    print(f'\n  各段:')
    segs = ['w1','w2','w3','w4','w5']
    seg_labels = {'w1':'2016-17','w2':'2018-19','w3':'2020-21','w4':'2022-23','w5':'2024-26'}
    base_per_seg = {}
    for s in segs:
        sub = df_e[df_e['seg']==s]
        n = len(sub); h = sub['is_explosive'].sum()
        rate = h/n*100 if n else 0
        base_per_seg[s] = rate
        print(f'    {s} ({seg_labels[s]}): n={n:,}, 暴涨={h}, 暴涨率={rate:.2f}%')

    # ============ 候选条件设计 ============
    candidates = []
    # 1. 单维卦象 (单分量 0/1) — 6 个 × 2 = 12 个
    for col in ['stk_d1','stk_m1','stk_y1','mkt_d1','mkt_m1','mkt_y1']:
        for v in ['0','1']:
            candidates.append({'type':'cat','col':col,'op':'==','val':v,
                               'label':f'{col}=={v}'})

    # 2. 阴阳 yy (8 态) — 这次进 candidates 但作为单条件
    for col in ['stk_yy','mkt_yy']:
        for v in ['000','001','010','011','100','101','110','111']:
            candidates.append({'type':'cat','col':col,'op':'==','val':v,
                               'label':f'{col}=={v}'})

    # 3. 数值阈值
    num_thrs = {
        'cur_mf': [(-200,'<='),(-100,'<='),(0,'<='),(0,'>='),(100,'>='),(200,'>='),(300,'>=')],
        'cur_rt': [(-300,'<='),(-200,'<='),(-100,'<='),(0,'<='),(0,'>='),(100,'>='),(200,'>=')],
        'mf30_min': [(-500,'<='),(-400,'<='),(-300,'<='),(-200,'<='),(-100,'<='),(0,'<=')],
        'mf30_max': [(0,'>='),(100,'>='),(200,'>='),(400,'>='),(600,'>=')],
        'rt30_min': [(-500,'<='),(-400,'<='),(-300,'<='),(-200,'<='),(-100,'<='),(0,'<=')],
        'mf5': [(-100,'<='),(0,'<='),(0,'>='),(100,'>='),(200,'>=')],
        'rt5': [(-200,'<='),(-100,'<='),(0,'<='),(0,'>='),(100,'>=')],
        'cur_td': [(20,'<='),(40,'<='),(60,'>='),(80,'>=')],
    }
    for col, thrs in num_thrs.items():
        for thr, op in thrs:
            candidates.append({'type':'num','col':col,'op':op,'val':thr,
                               'label':f'{col} {op} {thr}'})

    print(f'\n  候选条件总数: {len(candidates)}')

    # ============ 阶段 4: 反向避雷 ============
    print(f'\n{"="*82}')
    print(f'  阶段 4: 反向避雷 (≥3/5 段 lift<0.7 且 ≤1 段 lift>1.3)')
    print(f'{"="*82}\n')
    print(f'  {"条件":<26} {"全n":>7} {"全lift":>7} '
          f'{"w1":>7} {"w2":>7} {"w3":>7} {"w4":>7} {"w5":>7} {"差段数":>5}')

    avoid_list = []
    cand_results = []  # 给阶段 5 复用
    for c in candidates:
        if c['type'] == 'cat':
            mask = df_e[c['col']] == c['val']
        else:
            mask = (df_e[c['col']] <= c['val']) if c['op'] == '<=' else (df_e[c['col']] >= c['val'])
        sub = df_e[mask]
        n = len(sub)
        if n < 200: continue
        global_lift = (sub['is_explosive'].mean()*100) / base_rate if base_rate else 0

        seg_lifts = lift_per_seg(df_e, mask, segs, base_per_seg, min_n=50)
        n_bad = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]) and seg_lifts[s][0] < 0.7)
        n_good = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]) and seg_lifts[s][0] > 1.3)
        n_data = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]))

        cand_results.append({
            'cond':c, 'label':c['label'], 'n':n, 'glift':global_lift,
            'seg_lifts':seg_lifts, 'n_bad':n_bad, 'n_good':n_good, 'n_data':n_data
        })

        # 避雷判定
        if n_bad >= 3 and n_good <= 1 and n_data >= 4:
            avoid_list.append(c)
            line = f'  {c["label"]:<26} {n:>7,} {global_lift:>+5.2f}x'
            for s in segs:
                lf, ns = seg_lifts[s]
                if np.isnan(lf): line += f' {"--":>7}'
                else: line += f' {lf:>+5.2f}x'
            line += f' {n_bad:>5}'
            print(line)

    print(f'\n  --- 避雷名单 union 效果 ---')
    avoid_mask = np.zeros(len(df_e), dtype=bool)
    for c in avoid_list:
        if c['type'] == 'cat':
            avoid_mask |= (df_e[c['col']].to_numpy() == c['val'])
        else:
            arr = df_e[c['col']].to_numpy()
            avoid_mask |= (arr <= c['val']) if c['op'] == '<=' else (arr >= c['val'])
    df_safe = df_e[~avoid_mask].reset_index(drop=True)
    n_safe = len(df_safe); n_h = df_safe['is_explosive'].sum()
    safe_rate = n_h/n_safe*100 if n_safe else 0
    print(f'  避雷条件: {len(avoid_list)} 项')
    print(f'  原 n={n_total:,}, 暴涨率={base_rate:.2f}%')
    print(f'  避雷后 n={n_safe:,} ({n_safe/n_total*100:.1f}%), 暴涨率={safe_rate:.2f}% '
          f'(lift {safe_rate/base_rate:.2f}x)')

    # 避雷后各段
    safe_per_seg = {}
    for s in segs:
        sub = df_safe[df_safe['seg']==s]
        n = len(sub); h = sub['is_explosive'].sum()
        rate = h/n*100 if n else 0
        safe_per_seg[s] = rate
        lift_vs_base = rate/base_per_seg[s] if base_per_seg[s] else 0
        print(f'    {s} ({seg_labels[s]}): n={n:,}, 暴涨率={rate:.2f}%, vs base lift={lift_vs_base:.2f}x')

    # ============ 阶段 5: 软排名 (好规律) ============
    print(f'\n{"="*82}')
    print(f'  阶段 5: 软排名 (≥3/5 段 lift>1.3 且 ≤1 段 lift<0.7)')
    print(f'{"="*82}\n')
    print(f'  {"条件":<26} {"全n":>7} {"全lift":>7} '
          f'{"w1":>7} {"w2":>7} {"w3":>7} {"w4":>7} {"w5":>7} {"好段数":>5}')

    # 在避雷后池子里再扫一遍 (找避雷未吸收的好规律)
    good_list = []
    for c in candidates:
        if c['type'] == 'cat':
            mask = df_safe[c['col']] == c['val']
        else:
            mask = (df_safe[c['col']] <= c['val']) if c['op'] == '<=' else (df_safe[c['col']] >= c['val'])
        sub = df_safe[mask]
        n = len(sub)
        if n < 200: continue
        glift = (sub['is_explosive'].mean()*100) / safe_rate if safe_rate else 0

        seg_lifts = lift_per_seg(df_safe, mask, segs, safe_per_seg, min_n=50)
        n_bad = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]) and seg_lifts[s][0] < 0.7)
        n_good = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]) and seg_lifts[s][0] > 1.3)
        n_data = sum(1 for s in segs if not np.isnan(seg_lifts[s][0]))

        if n_good >= 3 and n_bad <= 1 and n_data >= 4:
            good_list.append({**c, 'glift':glift, 'seg_lifts':seg_lifts, 'n':n})
            line = f'  {c["label"]:<26} {n:>7,} {glift:>+5.2f}x'
            for s in segs:
                lf, ns = seg_lifts[s]
                if np.isnan(lf): line += f' {"--":>7}'
                else: line += f' {lf:>+5.2f}x'
            line += f' {n_good:>5}'
            print(line)

    # ============ IS/OOS 验证 ============
    print(f'\n{"="*82}')
    print(f'  IS (w1+w2+w3) / OOS (w4+w5) 拆开验证好规律')
    print(f'{"="*82}\n')

    df_is = df_safe[df_safe['seg'].isin(['w1','w2','w3'])].reset_index(drop=True)
    df_oos = df_safe[df_safe['seg'].isin(['w4','w5'])].reset_index(drop=True)
    is_rate = df_is['is_explosive'].mean()*100
    oos_rate = df_oos['is_explosive'].mean()*100
    print(f'  IS  (w1+w2+w3): n={len(df_is):,}, 暴涨率={is_rate:.2f}%')
    print(f'  OOS (w4+w5):    n={len(df_oos):,}, 暴涨率={oos_rate:.2f}%\n')

    print(f'  {"规律":<26} {"IS_n":>6} {"IS率":>6} {"IS lift":>7} '
          f'{"OOS_n":>6} {"OOS率":>6} {"OOS lift":>8} {"判定":>6}')
    real_good = []
    for r in good_list:
        c = r
        if c['type'] == 'cat':
            mis = df_is[c['col']] == c['val']
            mos = df_oos[c['col']] == c['val']
        else:
            mis = (df_is[c['col']] <= c['val']) if c['op'] == '<=' else (df_is[c['col']] >= c['val'])
            mos = (df_oos[c['col']] <= c['val']) if c['op'] == '<=' else (df_oos[c['col']] >= c['val'])
        sub_is = df_is[mis]; sub_oos = df_oos[mos]
        if len(sub_is) < 100 or len(sub_oos) < 50: continue
        is_r = sub_is['is_explosive'].mean()*100
        oos_r = sub_oos['is_explosive'].mean()*100
        is_l = is_r/is_rate if is_rate else 0
        oos_l = oos_r/oos_rate if oos_rate else 0
        if is_l >= 1.2 and oos_l >= 1.0:
            verd = '★真好'
            real_good.append(c)
        elif is_l >= 1.2 and oos_l >= 0.8:
            verd = '○一般'
        else:
            verd = '✗切片'
        print(f'  {c["label"]:<26} {len(sub_is):>6,} {is_r:>+5.2f}% {is_l:>+6.2f}x '
              f'{len(sub_oos):>6,} {oos_r:>+5.2f}% {oos_l:>+7.2f}x {verd:>6}')

    # ============ 软排名 score ============
    print(f'\n{"="*82}')
    print(f'  软排名 score 分布 (避雷后池子, ★真好规律命中数)')
    print(f'{"="*82}\n')

    if real_good:
        score = np.zeros(len(df_safe), dtype=int)
        for c in real_good:
            if c['type'] == 'cat':
                m = df_safe[c['col']] == c['val']
            else:
                m = (df_safe[c['col']] <= c['val']) if c['op'] == '<=' else (df_safe[c['col']] >= c['val'])
            score += m.to_numpy().astype(int)
        df_safe['score'] = score
        print(f'  共 {len(real_good)} 条 ★真好规律, score 范围 0~{len(real_good)}')
        print(f'  {"score":>5} {"n":>7} {"暴涨":>6} {"暴涨率":>7} {"lift_vs_safe":>14}')
        for sc in sorted(df_safe['score'].unique()):
            sub = df_safe[df_safe['score']==sc]
            n = len(sub); h = sub['is_explosive'].sum()
            rate = h/n*100 if n else 0
            lift = rate/safe_rate if safe_rate else 0
            print(f'  {sc:>5} {n:>7,} {h:>6} {rate:>+6.2f}% {lift:>+13.2f}x')

        # score 跨段
        print(f'\n  score >= 1 跨段验证:')
        for thr in [1, 2, 3]:
            if thr > len(real_good): break
            sub_s = df_safe[df_safe['score']>=thr]
            if len(sub_s) < 50: continue
            print(f'\n  score >= {thr}: n={len(sub_s):,}, '
                  f'总暴涨率={sub_s["is_explosive"].mean()*100:.2f}%')
            for s in segs:
                ssub = sub_s[sub_s['seg']==s]
                n = len(ssub); h = ssub['is_explosive'].sum()
                rate = h/n*100 if n else 0
                lift_vs_seg = rate/base_per_seg[s] if base_per_seg[s] else 0
                print(f'    {s} ({seg_labels[s]}): n={n:,}, 率={rate:.2f}%, lift_vs_全市场段={lift_vs_seg:.2f}x')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
