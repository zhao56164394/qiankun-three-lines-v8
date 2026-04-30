# -*- coding: utf-8 -*-
"""波段是否会暴涨 — 3 个角度预测

角度 1: 波段起点 (上穿 11 那天) 的特征 → 预测整段是否会 ≥+100%
角度 2: 波段中段 (起点后 30 天 trend 仍 >11) 的特征 → 预测剩余段是否还会暴涨
角度 3: 起点当日大盘卦 (mkt_d / mkt_m / mkt_y) → 暴涨股密度

每个角度都看:
  - 数值因子 (cur_retail / cur_mf / cur_trend / mf_5d / ret_5d / mf_30d_min / ret_30d_min)
  - 卦象 (stk_d/m/y, mkt_d/m/y)
  - lift_100 / lift_200
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
        in_band = False
        band_start = -1
        if not np.isnan(td[s]) and td[s] > 11:
            in_band = True
            band_start = s
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur): continue
            if not in_band:
                if not np.isnan(prev) and prev <= 11 and cur > 11:
                    in_band = True
                    band_start = i
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
                                    'end_idx': band_end,
                                    'start_date': date[band_start],
                                    'days': band_end - band_start + 1,
                                    'gain_pct': gain,
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
                            'end_idx': band_end,
                            'start_date': date[band_start],
                            'days': band_end - band_start + 1,
                            'gain_pct': gain,
                        })
    return pd.DataFrame(bands)


def annotate_features(df_b, gi_arr, retail_arr, mf_arr, trend_arr, code_starts):
    """对每个波段, 取 gi (start_idx 或 mid_idx) 的特征"""
    rows = []
    for _, b in df_b.iterrows():
        gi = int(b['gi'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        ci_s = code_starts[ci]
        i5 = max(gi - 5, ci_s)
        i30 = max(gi - 30, ci_s)
        ret_5d = retail_arr[gi] - retail_arr[i5] if not np.isnan(retail_arr[i5]) else np.nan
        mf_5d = mf_arr[gi] - mf_arr[i5] if not np.isnan(mf_arr[i5]) else np.nan
        td_5d = trend_arr[gi] - trend_arr[i5] if not np.isnan(trend_arr[i5]) else np.nan
        seg_r = retail_arr[i30:gi+1]
        seg_m = mf_arr[i30:gi+1]
        ret_30d_min = np.nanmin(seg_r) if (gi+1) > i30 and len(seg_r) > 0 else np.nan
        mf_30d_min = np.nanmin(seg_m) if (gi+1) > i30 and len(seg_m) > 0 else np.nan
        rows.append({
            'cur_retail': retail_arr[gi],
            'cur_mf': mf_arr[gi],
            'cur_trend': trend_arr[gi],
            'ret_5d': ret_5d, 'mf_5d': mf_5d, 'td_5d': td_5d,
            'ret_30d_min': ret_30d_min, 'mf_30d_min': mf_30d_min,
        })
    return pd.DataFrame(rows)


def lift_numeric(df, col, base_r100, base_r200, n=5):
    sub = df.dropna(subset=[col]).copy()
    try:
        sub['__bin'] = pd.qcut(sub[col], n, labels=[f'q{i+1}' for i in range(n)], duplicates='drop')
    except: return pd.DataFrame()
    rows = []
    for q, g in sub.groupby('__bin', observed=True):
        nn = len(g); n100 = (g['gain_pct']>=100).sum(); n200 = (g['gain_pct']>=200).sum()
        rows.append({'col':col,'bin':q, 'mn':g[col].min(),'mx':g[col].max(),
                     'n':nn,'n100':n100,'n200':n200,
                     'r100':n100/nn*100,'r200':n200/nn*100,
                     'lift_100':(n100/nn*100)/base_r100 if base_r100 else 0,
                     'lift_200':(n200/nn*100)/base_r200 if base_r200 else 0})
    return pd.DataFrame(rows)


def lift_categorical(df, col, base_r100, base_r200, min_n=300):
    rows = []
    for v, g in df.groupby(col):
        if v == '' or len(g) < min_n: continue
        nn = len(g); n100 = (g['gain_pct']>=100).sum(); n200 = (g['gain_pct']>=200).sum()
        rows.append({'col':col,'value':v,'n':nn,
                     'n100':n100,'n200':n200,
                     'r100':n100/nn*100,'r200':n200/nn*100,
                     'lift_100':(n100/nn*100)/base_r100 if base_r100 else 0,
                     'lift_200':(n200/nn*100)/base_r200 if base_r200 else 0})
    return pd.DataFrame(rows).sort_values('lift_200', ascending=False)


def print_lift_block(name, df, base_r100, base_r200, num_cols, cat_cols):
    print(f'\n  Baseline: n={len(df):,}, r100={base_r100:.2f}%, r200={base_r200:.2f}%')

    print(f'\n  --- 数值 5 分位 ---')
    for col in num_cols:
        res = lift_numeric(df, col, base_r100, base_r200)
        if len(res) == 0: continue
        # 找 max lift_100 / min lift_100 桶
        mx = res.loc[res['lift_100'].idxmax()]
        mn_ = res.loc[res['lift_100'].idxmin()]
        print(f'    {col:<14} max-lift_100 桶 [{mx["mn"]:>+6.0f}, {mx["mx"]:>+6.0f}] '
              f'r100={mx["r100"]:>5.2f}% L100={mx["lift_100"]:>+5.2f}x | '
              f'min 桶 [{mn_["mn"]:>+6.0f}, {mn_["mx"]:>+6.0f}] L100={mn_["lift_100"]:>+5.2f}x')

    print(f'\n  --- 卦象 (top 3 + bottom 3 by lift_200) ---')
    for col in cat_cols:
        res = lift_categorical(df, col, base_r100, base_r200)
        if len(res) == 0: continue
        print(f'    {col}:')
        for _, r in res.head(3).iterrows():
            print(f'      ↑ {r["value"]:<5} n={r["n"]:>5} r100={r["r100"]:>5.2f}% L100={r["lift_100"]:>+5.2f}x  '
                  f'r200={r["r200"]:>5.2f}% L200={r["lift_200"]:>+5.2f}x')
        for _, r in res.tail(3).iterrows():
            print(f'      ↓ {r["value"]:<5} n={r["n"]:>5} r100={r["r100"]:>5.2f}% L100={r["lift_100"]:>+5.2f}x  '
                  f'r200={r["r200"]:>5.2f}% L200={r["lift_200"]:>+5.2f}x')


def main():
    t0 = time.time()
    print('=== test171: 暴涨波段预测 — 3 角度 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3).replace({'nan':''})
    g.rename(columns={'d_gua':'stk_d','m_gua':'stk_m','y_gua':'stk_y'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    for c in ['d_gua','m_gua','y_gua']:
        mkt[c] = mkt[c].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt.drop_duplicates('date').rename(columns={'d_gua':'mkt_d','m_gua':'mkt_m','y_gua':'mkt_y'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)
    print(f'  数据: {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'close':close_arr,'td':trend_arr,
              'starts':code_starts,'ends':code_ends}

    print('  提取波段...')
    df_b = find_bands(arrays)
    print(f'    总波段: {len(df_b):,}')

    n_total = len(df_b); n100 = (df_b['gain_pct']>=100).sum(); n200 = (df_b['gain_pct']>=200).sum()
    base_r100 = n100/n_total*100; base_r200 = n200/n_total*100

    NUM_COLS = ['cur_retail','cur_mf','cur_trend','ret_5d','mf_5d','td_5d','ret_30d_min','mf_30d_min']
    CAT_COLS = ['stk_d','stk_m','stk_y','mkt_d','mkt_m','mkt_y']

    # ===== 角度 1: 波段起点 =====
    print(f'\n{"="*82}')
    print(f'  角度 1: 波段起点 (上穿 11 那天) 特征')
    print(f'{"="*82}')
    df_b1 = df_b.copy()
    df_b1['gi'] = df_b1['start_idx']
    feat1 = annotate_features(df_b1, df_b1['gi'].values, retail_arr, mf_arr, trend_arr, code_starts)
    df_b1 = pd.concat([df_b1.reset_index(drop=True), feat1.reset_index(drop=True)], axis=1)
    # 加卦象
    for col in CAT_COLS:
        df_b1[col] = df[col].to_numpy()[df_b1['gi'].astype(int).values]

    print_lift_block('角度1', df_b1, base_r100, base_r200, NUM_COLS, CAT_COLS)

    # ===== 角度 2: 波段中段 (起点后 30 天) =====
    print(f'\n{"="*82}')
    print(f'  角度 2: 波段中段 (起点后 30 天, 仍在 trend>11) 特征')
    print(f'{"="*82}')
    df_b2_full = df_b[df_b['days'] > 30].copy()
    print(f'  波段长度 > 30 天的 (持续到 mid): {len(df_b2_full):,}')
    df_b2_full['gi'] = df_b2_full['start_idx'] + 30
    feat2 = annotate_features(df_b2_full, df_b2_full['gi'].values, retail_arr, mf_arr, trend_arr, code_starts)
    df_b2_full = pd.concat([df_b2_full.reset_index(drop=True), feat2.reset_index(drop=True)], axis=1)
    for col in CAT_COLS:
        df_b2_full[col] = df[col].to_numpy()[df_b2_full['gi'].astype(int).values]

    n_total2 = len(df_b2_full); n100_2 = (df_b2_full['gain_pct']>=100).sum(); n200_2 = (df_b2_full['gain_pct']>=200).sum()
    base_r100_2 = n100_2/n_total2*100; base_r200_2 = n200_2/n_total2*100
    print_lift_block('角度2', df_b2_full, base_r100_2, base_r200_2, NUM_COLS, CAT_COLS)

    # ===== 角度 3: 起点大盘卦 详细分布 =====
    print(f'\n{"="*82}')
    print(f'  角度 3: 起点大盘卦 详细分布 (mkt_d/m/y 全部桶, 不只 top/bot)')
    print(f'{"="*82}')

    for col in ['mkt_d','mkt_m','mkt_y','stk_y','stk_m']:
        res = lift_categorical(df_b1, col, base_r100, base_r200, min_n=500)
        if len(res) == 0: continue
        print(f'\n  {col}:')
        print(f'    {"value":<6} {"n":>6} {"≥100%":>6} {"r100":>7} {"L100":>7} '
              f'{"≥200%":>6} {"r200":>7} {"L200":>7}')
        for _, r in res.iterrows():
            print(f'    {r["value"]:<6} {r["n"]:>6,} {r["n100"]:>5} {r["r100"]:>+6.2f}% '
                  f'{r["lift_100"]:>+5.2f}x  {r["n200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    # ===== 联合: 持续 30 天的"中段确认" + 当时大盘卦 =====
    print(f'\n{"="*82}')
    print(f'  角度 2 强化: 中段 (>30d) 中, 当时 mkt_y 分布')
    print(f'{"="*82}')
    res = lift_categorical(df_b2_full, 'mkt_y', base_r100_2, base_r200_2, min_n=500)
    print(f'\n  mkt_y (mid 30d):')
    print(f'    {"value":<6} {"n":>6} {"≥100%":>6} {"r100":>7} {"L100":>7} '
          f'{"≥200%":>6} {"r200":>7} {"L200":>7}')
    for _, r in res.iterrows():
        print(f'    {r["value"]:<6} {r["n"]:>6,} {r["n100"]:>5} {r["r100"]:>+6.2f}% '
              f'{r["lift_100"]:>+5.2f}x  {r["n200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    print(f'\n  mkt_m (mid 30d):')
    res = lift_categorical(df_b2_full, 'mkt_m', base_r100_2, base_r200_2, min_n=500)
    print(f'    {"value":<6} {"n":>6} {"≥100%":>6} {"r100":>7} {"L100":>7} '
          f'{"≥200%":>6} {"r200":>7} {"L200":>7}')
    for _, r in res.iterrows():
        print(f'    {r["value"]:<6} {r["n"]:>6,} {r["n100"]:>5} {r["r100"]:>+6.2f}% '
              f'{r["lift_100"]:>+5.2f}x  {r["n200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
