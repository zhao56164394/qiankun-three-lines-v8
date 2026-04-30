# -*- coding: utf-8 -*-
"""验证 mkt=000+stk=011+cur_mf<=-100+cur_retail<=-100 策略

输出:
1. 666 笔入场全部跨年统计 (年/笔数/暴涨股/r100)
2. 暴涨股 39 只完整清单 (按年, 按涨幅倒序)
3. 信号天数密度 (每天最多触发几只? 同日多股能不能选)
4. 持续天数分布 (从入场到 max_close 多久)
5. 暴涨股股票代码集中度 (是否同一只股反复入选)
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
                'days_to_min': idx_min_g - band_start,
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
    print('=== test180: 验证 mkt=000+stk=011+cur双线<-100 ===\n')

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

    # 锁: mkt=000 + stk=011 + cur_mf<=-100 + cur_retail<=-100
    sub = df_b[(df_b['mkt_yy']=='000') &
                (df_b['stk_yy']=='011') &
                (df_b['cur_mf']<=-100) &
                (df_b['cur_retail']<=-100)].copy()
    print(f'  入场池: n={len(sub)}')

    # ===== 1. 跨年统计 =====
    print(f'\n{"="*82}')
    print(f'  1. 跨年统计')
    print(f'{"="*82}')
    print(f'\n  {"年":<6} {"信号":>5} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7} '
          f'{"baseline":>9}')
    for y, sub_y in sub.groupby('year'):
        n = len(sub_y); h100 = (sub_y['gain_pct']>=100).sum(); h200 = (sub_y['gain_pct']>=200).sum()
        yb = df_b[df_b['year']==y]
        bl = (yb['gain_pct']>=100).mean()*100
        ratio = (h100/n*100)/bl if bl else 0
        print(f'  {y:<6} {n:>5} {h100:>5} {h100/n*100:>+6.2f}% {h200:>5} {h200/n*100:>+6.2f}%   {bl:>+5.2f}%  {ratio:.2f}x')

    n_total = len(sub); h100_total = (sub['gain_pct']>=100).sum()
    bl_total = (df_b['gain_pct']>=100).mean()*100
    print(f'  {"总":<6} {n_total:>5} {h100_total:>5} '
          f'{h100_total/n_total*100:>+6.2f}% '
          f'{(sub["gain_pct"]>=200).sum():>5} '
          f'{(sub["gain_pct"]>=200).mean()*100:>+6.2f}%   '
          f'{bl_total:>+5.2f}%  '
          f'{(h100_total/n_total*100)/bl_total:.2f}x')

    # ===== 2. 暴涨股清单 =====
    print(f'\n{"="*82}')
    print(f'  2. 暴涨股清单 (按年, ≥+100%)')
    print(f'{"="*82}')
    big = sub[sub['gain_pct']>=100].copy()
    big = big.sort_values(['year', 'gain_pct'], ascending=[True, False])
    print(f'\n  共 {len(big)} 只\n')
    print(f'  {"代码":<8} {"起点":<12} {"min点":<12} {"max点":<12} {"持续":>4} '
          f'{"min价":>6} {"max价":>6} {"涨幅":>9} {"d_min":>5} {"d_max":>5}')
    for _, r in big.iterrows():
        print(f'  {r["code"]:<8} {r["start_date"]:<12} {r["min_date"]:<12} {r["max_date"]:<12} '
              f'{r["days"]:>3}d {r["min_close"]:>5.2f} {r["max_close"]:>5.2f} '
              f'{r["gain_pct"]:>+8.1f}% {r["days_to_min"]:>4}d {r["days_to_max"]:>4}d')

    # ===== 3. 信号天密度 =====
    print(f'\n{"="*82}')
    print(f'  3. 同日触发密度 (每天有几只信号)')
    print(f'{"="*82}')
    per_day = sub.groupby('start_date').size().sort_values(ascending=False)
    print(f'\n  信号天数: {len(per_day)}')
    print(f'  日均触发: {per_day.mean():.2f}')
    print(f'  最多 1 天触发: {per_day.max()}')
    print(f'\n  分布:')
    for n in [1, 2, 3, 5, 10, 20, 50]:
        cnt = (per_day == n).sum() if n < per_day.max() else (per_day >= n).sum()
        if n < per_day.max():
            print(f'    每日触发 {n} 只: {cnt} 天')
        else:
            print(f'    每日触发 ≥{n} 只: {cnt} 天')
    print(f'\n  --- 触发密度 top 10 天 ---')
    print(f'  {"日期":<12} {"触发数":>6}')
    for d, n in per_day.head(10).items():
        print(f'  {d:<12} {n:>6}')

    # ===== 4. 持续天数 =====
    print(f'\n{"="*82}')
    print(f'  4. 暴涨股持续天数分布')
    print(f'{"="*82}')
    print(f'\n  暴涨股 (39 只) 入场到 max_close 天数:')
    print(f'    min={big["days_to_max"].min()}, '
          f'p25={big["days_to_max"].quantile(0.25):.0f}, '
          f'med={big["days_to_max"].median():.0f}, '
          f'p75={big["days_to_max"].quantile(0.75):.0f}, '
          f'max={big["days_to_max"].max()}')

    print(f'\n  入场到 min_close 天数 (跌到底再涨):')
    print(f'    min={big["days_to_min"].min()}, '
          f'p25={big["days_to_min"].quantile(0.25):.0f}, '
          f'med={big["days_to_min"].median():.0f}, '
          f'p75={big["days_to_min"].quantile(0.75):.0f}, '
          f'max={big["days_to_min"].max()}')

    print(f'\n  整段持续天数:')
    print(f'    min={big["days"].min()}, p25={big["days"].quantile(0.25):.0f}, '
          f'med={big["days"].median():.0f}, p75={big["days"].quantile(0.75):.0f}, '
          f'max={big["days"].max()}')

    # ===== 5. 股票代码集中度 =====
    print(f'\n{"="*82}')
    print(f'  5. 暴涨股代码集中度')
    print(f'{"="*82}')
    print(f'\n  暴涨股不同股票数: {big["code"].nunique()} (总暴涨股 {len(big)})')
    rep_codes = big['code'].value_counts()
    print(f'\n  反复入选股票 (≥2 次):')
    for c, n in rep_codes[rep_codes>=2].items():
        print(f'    {c}: {n} 次')

    print(f'\n  --- 全样本 (666 笔) 不同股票数 ---')
    print(f'  样本中不同股票: {sub["code"].nunique()} (样本 {len(sub)} 笔)')

    # ===== 6. 暴涨股 + 非暴涨股 区分特征 =====
    print(f'\n{"="*82}')
    print(f'  6. 暴涨股 vs 非暴涨股 (在 666 笔内) 特征对比')
    print(f'{"="*82}')
    big_sub = sub[sub['gain_pct']>=100]
    small_sub = sub[sub['gain_pct']<100]
    print(f'\n  {"特征":<14} {"暴涨股 (n={})".format(len(big_sub)):<22} {"非暴涨 (n={})".format(len(small_sub)):<22}')
    for col in ['stk_m_t', 'stk_y_t', 'cur_retail', 'cur_mf']:
        b_med = big_sub[col].median()
        s_med = small_sub[col].median()
        b_p25 = big_sub[col].quantile(0.25)
        b_p75 = big_sub[col].quantile(0.75)
        print(f'  {col:<14} med={b_med:>+6.1f} [p25={b_p25:.0f}, p75={b_p75:.0f}]   med={s_med:>+6.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
