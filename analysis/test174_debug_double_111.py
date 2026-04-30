# -*- coding: utf-8 -*-
"""调试: 双 111 (mkt=111 + stk=111) 起点的实际 d_t / m_t / y_t 值"""
import os, sys, io
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
        in_band = False; band_start = -1
        # 关键: 起点定义是"上穿 11" - 进入波段时 prev<=11, cur>11
        # 但首日 (i=s) 没有 prev, 这里直接如果 td[s]>11 就当 in_band, 不算"上穿"
        if not np.isnan(td[s]) and td[s] > 11:
            in_band = True; band_start = s
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur): continue
            if not in_band:
                if not np.isnan(prev) and prev <= 11 and cur > 11:
                    in_band = True; band_start = i
            else:
                if cur <= 11:
                    band_end = i
                    if band_end > band_start:
                        seg_close = close[band_start:band_end+1]
                        seg_close_v = seg_close[~np.isnan(seg_close)]
                        if len(seg_close_v) >= 2:
                            idx_max_local = int(np.argmax(seg_close_v))
                            max_close = seg_close_v[idx_max_local]
                            min_before = np.min(seg_close_v[:idx_max_local+1])
                            if min_before > 0:
                                gain = (max_close / min_before - 1) * 100
                                bands.append({
                                    'code': code[band_start],
                                    'start_idx': band_start,
                                    'start_date': date[band_start],
                                    'days': band_end - band_start + 1,
                                    'gain_pct': gain,
                                    'is_first_day': band_start == s,  # 首日 fallback 触发
                                })
                    in_band = False; band_start = -1
        if in_band and band_start >= 0:
            band_end = e - 1
            if band_end > band_start:
                seg_close = close[band_start:band_end+1]
                seg_close_v = seg_close[~np.isnan(seg_close)]
                if len(seg_close_v) >= 2:
                    idx_max_local = int(np.argmax(seg_close_v))
                    max_close = seg_close_v[idx_max_local]
                    min_before = np.min(seg_close_v[:idx_max_local+1])
                    if min_before > 0:
                        gain = (max_close / min_before - 1) * 100
                        bands.append({
                            'code': code[band_start],
                            'start_idx': band_start,
                            'start_date': date[band_start],
                            'days': band_end - band_start + 1,
                            'gain_pct': gain,
                            'is_first_day': band_start == s,
                        })
    return pd.DataFrame(bands)


def yyy(d, m, y, thr=50):
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def main():
    print('=== 调试: 双 111 起点实际值 ===\n')

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
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
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
    df_b = find_bands(arrays)
    print(f'总波段: {len(df_b):,}')
    print(f'其中"首日 fallback"波段 (没真上穿, 数据起点就 trend>11): {df_b["is_first_day"].sum():,}')

    si = df_b['start_idx'].astype(int).values
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['stk_d_t'] = [stk_d_t[i] for i in si]
    df_b['stk_m_t'] = [stk_m_t[i] for i in si]
    df_b['stk_y_t'] = [stk_y_t[i] for i in si]
    df_b['mkt_d_t_v'] = [mkt_d_t[i] for i in si]
    df_b['mkt_m_t_v'] = [mkt_m_t[i] for i in si]
    df_b['mkt_y_t_v'] = [mkt_y_t[i] for i in si]

    # 双 111 样本
    d111 = df_b[(df_b['mkt_yy']=='111') & (df_b['stk_yy']=='111')].copy()
    print(f'\n双 111 样本: {len(d111)}')
    print(f'其中"首日 fallback" 占: {d111["is_first_day"].sum()} ({d111["is_first_day"].mean()*100:.1f}%)')

    print(f'\n双 111 起点 stk_d_t (个股日 trend) 分布:')
    print(f'  min={d111["stk_d_t"].min():.1f}, p25={d111["stk_d_t"].quantile(0.25):.1f}, '
          f'med={d111["stk_d_t"].median():.1f}, p75={d111["stk_d_t"].quantile(0.75):.1f}, '
          f'max={d111["stk_d_t"].max():.1f}')

    # stk_d_t > 11 但 ≤ 50 的应该 0 个 (因为 stk_yy=111 要求 d_t>50)
    # 检查实际分布
    print(f'\n  stk_d_t > 50: {(d111["stk_d_t"] > 50).sum()}')
    print(f'  stk_d_t 11-50: {((d111["stk_d_t"] > 11) & (d111["stk_d_t"] <= 50)).sum()}')

    print(f'\n双 111 前 10 例:')
    print(f'  {"代码":<8} {"起点日":<12} {"is_first":>9} {"持续":>5} '
          f'{"stk_d_t":>8} {"stk_m_t":>8} {"stk_y_t":>8} '
          f'{"mkt_d_t":>8} {"mkt_m_t":>8} {"mkt_y_t":>8} {"涨幅":>8}')
    for _, r in d111.head(10).iterrows():
        print(f'  {r["code"]:<8} {r["start_date"]:<12} {str(r["is_first_day"]):>9} '
              f'{r["days"]:>4}d {r["stk_d_t"]:>+7.1f} {r["stk_m_t"]:>+7.1f} {r["stk_y_t"]:>+7.1f} '
              f'{r["mkt_d_t_v"]:>+7.1f} {r["mkt_m_t_v"]:>+7.1f} {r["mkt_y_t_v"]:>+7.1f} '
              f'{r["gain_pct"]:>+7.1f}%')

    # ===== 排除 first_day 后的双 111 =====
    print(f'\n=== 排除"首日 fallback"后的双 111 ===')
    d111_clean = d111[~d111['is_first_day']]
    print(f'剩下双 111: {len(d111_clean)}')
    if len(d111_clean) > 0:
        print(f'r100 = {(d111_clean["gain_pct"]>=100).mean()*100:.2f}%')
        print(f'r200 = {(d111_clean["gain_pct"]>=200).mean()*100:.2f}%')

    # 排除 first_day 后, 全样本 baseline
    df_b_clean = df_b[~df_b['is_first_day']]
    base_r100 = (df_b_clean['gain_pct']>=100).mean()*100
    base_r200 = (df_b_clean['gain_pct']>=200).mean()*100
    print(f'\n排除 first_day 后, baseline:')
    print(f'  n={len(df_b_clean)}')
    print(f'  r100={base_r100:.2f}%, r200={base_r200:.2f}%')

    # 重新算 64 组合 排除 first_day
    print(f'\n=== 排除 first_day 后, 双 111 / mkt=111 / stk=111 单独跨年 ===')
    df_b_clean = df_b_clean.copy()
    df_b_clean['year'] = df_b_clean['start_date'].str[:4]

    print(f'\n{"年":<6} {"全 n":>6} {"r100":>7} | {"mkt=111 n":>10} {"r100":>7} | {"stk=111 n":>10} {"r100":>7} | {"双 n":>5} {"r100":>7}')
    for y in sorted(df_b_clean['year'].unique()):
        sub = df_b_clean[df_b_clean['year']==y]
        if len(sub) < 100: continue
        m111 = sub[sub['mkt_yy']=='111']
        s111 = sub[sub['stk_yy']=='111']
        d111y = sub[(sub['mkt_yy']=='111') & (sub['stk_yy']=='111')]
        r_a = (sub['gain_pct']>=100).mean()*100
        r_m = (m111['gain_pct']>=100).mean()*100 if len(m111) else 0
        r_s = (s111['gain_pct']>=100).mean()*100 if len(s111) else 0
        r_d = (d111y['gain_pct']>=100).mean()*100 if len(d111y) else 0
        print(f'{y:<6} {len(sub):>6} {r_a:>+6.2f}% | {len(m111):>10} {r_m:>+6.2f}% | '
              f'{len(s111):>10} {r_s:>+6.2f}% | {len(d111y):>5} {r_d:>+6.2f}%')


if __name__ == '__main__':
    main()
