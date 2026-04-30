# -*- coding: utf-8 -*-
"""mkt=000 + stk=011 暴涨股各年代表 + 进一步过滤探索

输出:
1. mkt=000+stk=011 各年所有 ≥+100% 暴涨股完整清单
2. 加 cur_mf / cur_retail 过滤 看密度变化
3. 加 m_t / y_t 阈值看跨年稳定性是否还在
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_bands(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    td = arrays['td']; close = arrays['close']
    date = arrays['date']; code = arrays['code']
    bands = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < 30: continue
        cross_down_days = []
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur) or np.isnan(prev): continue
            if prev > 11 and cur <= 11:
                cross_down_days.append(i)
        for k in range(len(cross_down_days) - 1):
            band_start = cross_down_days[k]
            band_end = cross_down_days[k+1]
            if band_end <= band_start: continue
            seg_close = close[band_start:band_end+1]
            seg_close_v = seg_close[~np.isnan(seg_close)]
            if len(seg_close_v) < 2: continue
            idx_max_local = int(np.argmax(seg_close_v))
            max_close = seg_close_v[idx_max_local]
            min_before = np.min(seg_close_v[:idx_max_local+1])
            if min_before <= 0: continue
            gain = (max_close / min_before - 1) * 100
            seg_close_arr = close[band_start:band_start+idx_max_local+1]
            idx_min_local = int(np.nanargmin(seg_close_arr))
            idx_min_g = band_start + idx_min_local
            idx_max_g = band_start + idx_max_local
            bands.append({
                'code': code[band_start],
                'start_idx': band_start,
                'end_idx': band_end,
                'idx_min': idx_min_g,
                'idx_max': idx_max_g,
                'min_close': float(min_before),
                'max_close': float(max_close),
                'start_date': date[band_start],
                'min_date': date[idx_min_g],
                'max_date': date[idx_max_g],
                'days': band_end - band_start + 1,
                'days_to_max': idx_max_g - band_start,
                'gain_pct': gain,
            })
    return pd.DataFrame(bands)


def yyy(d, m, y, thr=50):
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def main():
    t0 = time.time()
    print('=== test179: mkt=000+stk=011 暴涨股 + 进一步过滤 ===\n')

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
              'starts':code_starts,'ends':code_ends}
    df_b_all = find_bands(arrays)
    df_b = df_b_all[df_b_all['start_date'] >= '2016-01-01'].copy()

    si = df_b['start_idx'].astype(int).values
    df_b['stk_m_t'] = [stk_m_t[i] for i in si]
    df_b['stk_y_t'] = [stk_y_t[i] for i in si]
    df_b['cur_retail'] = [retail_arr[i] for i in si]
    df_b['cur_mf'] = [mf_arr[i] for i in si]
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['year'] = df_b['start_date'].str[:4]

    # ===== 锁: mkt=000 + stk=011 =====
    sub = df_b[(df_b['mkt_yy']=='000') & (df_b['stk_yy']=='011')].copy()
    print(f'  mkt=000 + stk=011: n={len(sub):,}')

    # 全样本 baseline (每年)
    full_year_base = {}
    for y in sorted(df_b['year'].unique()):
        yb = df_b[df_b['year']==y]
        if len(yb) > 0:
            full_year_base[y] = (yb['gain_pct']>=100).mean()*100

    # ===== 1. 各年暴涨股清单 =====
    print(f'\n{"="*82}')
    print(f'  各年 ≥+100% 暴涨股完整清单')
    print(f'{"="*82}')

    big = sub[sub['gain_pct']>=100].copy()
    print(f'\n  共 {len(big)} 只 ≥+100%')
    for y, g_ in big.groupby('year'):
        if len(g_) == 0: continue
        g_ = g_.sort_values('gain_pct', ascending=False)
        print(f'\n  --- {y} ({len(g_)} 只) ---')
        print(f'  {"代码":<8} {"起点":<12} {"min点":<12} {"max点":<12} '
              f'{"持续":>4} {"min价":>6} {"max价":>6} {"涨幅":>9} '
              f'{"m_t":>5} {"y_t":>5} {"cur_ret":>8} {"cur_mf":>7}')
        for _, r in g_.iterrows():
            print(f'  {r["code"]:<8} {r["start_date"]:<12} {r["min_date"]:<12} {r["max_date"]:<12} '
                  f'{r["days"]:>3}d {r["min_close"]:>5.2f} {r["max_close"]:>5.2f} '
                  f'{r["gain_pct"]:>+8.1f}% {r["stk_m_t"]:>+4.0f} {r["stk_y_t"]:>+4.0f} '
                  f'{r["cur_retail"]:>+7.0f} {r["cur_mf"]:>+6.0f}')

    # ===== 2. 加 cur_mf / cur_retail 过滤 看跨年 =====
    print(f'\n{"="*82}')
    print(f'  加 cur_mf / cur_retail 过滤后 mkt=000+stk=011 跨年密度')
    print(f'{"="*82}')

    YEARS = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']

    print(f'\n  {"过滤":<22} {"n":>5} {"全":>5}', end='')
    for y in YEARS: print(f' {y[-2:]:>5}', end='')
    print()

    print(f'  {"baseline":<22} {len(df_b):>5,} {(df_b["gain_pct"]>=100).mean()*100:>+4.1f}%', end='')
    for y in YEARS: print(f' {full_year_base.get(y, 0):>+4.1f}%', end='')
    print()

    # 无过滤
    rows = sub
    n = len(rows); r = (rows['gain_pct']>=100).mean()*100
    print(f'  {"无过滤":<22} {n:>5} {r:>+4.1f}%', end='')
    for y in YEARS:
        ys = rows[rows['year']==y]
        if len(ys) < 20: print(f' {"--":>5}', end='')
        else: print(f' {(ys["gain_pct"]>=100).mean()*100:>+4.1f}%', end='')
    print()

    filters = [
        ('cur_mf<=-100', sub[sub['cur_mf']<=-100]),
        ('cur_mf<=0', sub[sub['cur_mf']<=0]),
        ('cur_retail<=-100', sub[sub['cur_retail']<=-100]),
        ('cur_retail<=0', sub[sub['cur_retail']<=0]),
        ('cur_mf<=-50 ∩ cur_ret<=-50', sub[(sub['cur_mf']<=-50)&(sub['cur_retail']<=-50)]),
        ('cur_mf<=-100 ∩ cur_ret<=-100', sub[(sub['cur_mf']<=-100)&(sub['cur_retail']<=-100)]),
        ('m_t>=60', sub[sub['stk_m_t']>=60]),
        ('m_t>=70', sub[sub['stk_m_t']>=70]),
        ('y_t>=70', sub[sub['stk_y_t']>=70]),
        ('m_t>=60 ∩ y_t>=70', sub[(sub['stk_m_t']>=60)&(sub['stk_y_t']>=70)]),
        ('m_t>=70 ∩ y_t>=70', sub[(sub['stk_m_t']>=70)&(sub['stk_y_t']>=70)]),
        ('m_t>=60 ∩ cur_mf<=-50', sub[(sub['stk_m_t']>=60)&(sub['cur_mf']<=-50)]),
        ('m_t>=70 ∩ cur_mf<=-100', sub[(sub['stk_m_t']>=70)&(sub['cur_mf']<=-100)]),
    ]

    for label, rows in filters:
        n = len(rows)
        if n < 50:
            print(f'  {label:<22} {n:>5} {"--":>5}')
            continue
        r = (rows['gain_pct']>=100).mean()*100
        print(f'  {label:<22} {n:>5} {r:>+4.1f}%', end='')
        for y in YEARS:
            ys = rows[rows['year']==y]
            if len(ys) < 10: print(f' {"--":>5}', end='')
            else: print(f' {(ys["gain_pct"]>=100).mean()*100:>+4.1f}%', end='')
        print()

    # ===== 3. 测试加大盘 trend 阈值代替阴阳 =====
    print(f'\n{"="*82}')
    print(f'  改用 mkt_y_t 数值阈值 (代替 mkt=000)')
    print(f'{"="*82}')

    s011 = df_b[df_b['stk_yy']=='011']
    print(f'\n  stk=011 子集: n={len(s011):,}\n')

    print(f'  {"mkt 条件":<28} {"n":>6} {"全 r100":>8}', end='')
    for y in YEARS: print(f' {y[-2:]:>5}', end='')
    print()

    mkt_filters = [
        ('mkt_yy=000', s011[s011['mkt_yy']=='000']),
        ('mkt_y_t<=30', s011[s011['mkt_y_t']<=30]),
        ('mkt_y_t<=40', s011[s011['mkt_y_t']<=40]),
        ('mkt_y_t<=50', s011[s011['mkt_y_t']<=50]),
        ('mkt_d_t<=30 ∩ mkt_m_t<=30', s011[(s011['mkt_d_t' if False else 'mkt_y_t']<=999)]),  # placeholder
    ]

    # 加上 mkt_d_t/m_t/y_t 各阈值
    for op_label, condition in [
        ('mkt_d+m+y<=50 全阴', (s011[(df_b.loc[s011.index, '__placeholder' if False else 'cur_mf']<999)])),  # placeholder
    ]:
        pass

    # 重新写, 直接条件
    df_s = s011.copy()
    # 用 buy_idx 取大盘 d_t / m_t / y_t
    df_s['mkt_d_t_v'] = [mkt_d_t[i] for i in df_s['start_idx'].astype(int).values]
    df_s['mkt_m_t_v'] = [mkt_m_t[i] for i in df_s['start_idx'].astype(int).values]
    df_s['mkt_y_t_v'] = [mkt_y_t[i] for i in df_s['start_idx'].astype(int).values]

    mkt_filters_real = [
        ('mkt_yy=000 (阴阳)', df_s[df_s['mkt_yy']=='000']),
        ('mkt_y_t<=30', df_s[df_s['mkt_y_t_v']<=30]),
        ('mkt_y_t<=40', df_s[df_s['mkt_y_t_v']<=40]),
        ('mkt_y_t<=50', df_s[df_s['mkt_y_t_v']<=50]),
        ('mkt_y_t<=60', df_s[df_s['mkt_y_t_v']<=60]),
        ('mkt_d_t<=50 ∩ y_t<=50', df_s[(df_s['mkt_d_t_v']<=50)&(df_s['mkt_y_t_v']<=50)]),
        ('mkt_d_t<=30 ∩ y_t<=30', df_s[(df_s['mkt_d_t_v']<=30)&(df_s['mkt_y_t_v']<=30)]),
        ('mkt_y_t<=50 ∩ m_t<=50', df_s[(df_s['mkt_y_t_v']<=50)&(df_s['mkt_m_t_v']<=50)]),
    ]

    for label, rows in mkt_filters_real:
        n = len(rows)
        if n < 50:
            print(f'  {label:<28} {n:>6} {"--":>8}')
            continue
        r = (rows['gain_pct']>=100).mean()*100
        print(f'  {label:<28} {n:>6,} {r:>+6.2f}%', end='')
        for y in YEARS:
            ys = rows[rows['year']==y]
            if len(ys) < 10: print(f' {"--":>5}', end='')
            else: print(f' {(ys["gain_pct"]>=100).mean()*100:>+4.1f}%', end='')
        print()

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
