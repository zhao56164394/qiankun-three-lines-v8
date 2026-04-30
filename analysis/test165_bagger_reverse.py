# -*- coding: utf-8 -*-
"""全市场暴涨股反向分析 — 不锁 regime

思路:
  1. 全 v5 入场池 (不锁 mkt_y), 算每个事件的 ret%
  2. 取所有 ret >= +100% / +200% / +500% 的暴涨股
  3. 看这些暴涨股入场时的特征分布:
     - 池深 / cur_mf / cur_retail / cur_trend / ret_5d / mf_5d
     - 卦象: stk_d / stk_m / stk_y / mkt_d / mkt_m / mkt_y
     - 日期分布 (年/月/年内位置)
  4. 用每个特征/卦做 lift 分析, 看哪些能把暴涨股富集
     lift = sub 中暴涨股密度 / baseline 暴涨股密度
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v5(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = np.nan; last_retail = np.nan
        pool_min_retail = np.inf
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True; pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                ci_s = arrays['starts'][ci]
                i5 = max(gi - 5, ci_s)
                ret_5d_v = retail[gi] - retail[i5] if not np.isnan(retail[i5]) else np.nan
                mf_5d_v = mf[gi] - mf[i5] if not np.isnan(mf[i5]) else np.nan
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'pool_min_retail':pool_min_retail,
                               'cur_mf':mf[gi],
                               'cur_retail':retail[gi],
                               'cur_trend':td[gi],
                               'ret_5d':ret_5d_v,
                               'mf_5d':mf_5d_v})
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    bp = close[buy_idx]; cum_mult = 1.0; holding = True
    cur_buy_price = bp
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding: cum_mult *= close[k] / cur_buy_price
            return (cum_mult-1)*100
        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0
        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]; holding = True
    if holding: cum_mult *= close[max_end] / cur_buy_price
    return (cum_mult-1)*100


def lift_analysis(df, col, baseline_density_100, baseline_density_200, n_bins=5, threshold=200):
    """对 col 做 5 桶 lift 分析"""
    if df[col].dtype == 'object' or df[col].dtype.name == 'string':
        # 分类
        counts = df.groupby(col).agg(
            n=('ret_pct', 'count'),
            n_h100=('ret_pct', lambda x: (x>=100).sum()),
            n_h200=('ret_pct', lambda x: (x>=200).sum()),
        ).reset_index()
        counts['r100'] = counts['n_h100'] / counts['n'] * 100
        counts['r200'] = counts['n_h200'] / counts['n'] * 100
        counts['lift_100'] = counts['r100'] / baseline_density_100
        counts['lift_200'] = counts['r200'] / baseline_density_200 if baseline_density_200 > 0 else 0
        counts = counts[counts['n'] >= 100].sort_values('lift_200', ascending=False)
        return counts
    else:
        # 数值 — 5 分位
        df_sub = df.dropna(subset=[col]).copy()
        try:
            df_sub['__bin'] = pd.qcut(df_sub[col], n_bins, labels=[f'q{i+1}' for i in range(n_bins)], duplicates='drop')
        except:
            return pd.DataFrame()
        counts = df_sub.groupby('__bin', observed=True).agg(
            n=('ret_pct', 'count'),
            mn=(col, 'min'),
            mx=(col, 'max'),
            n_h100=('ret_pct', lambda x: (x>=100).sum()),
            n_h200=('ret_pct', lambda x: (x>=200).sum()),
        ).reset_index()
        counts['r100'] = counts['n_h100'] / counts['n'] * 100
        counts['r200'] = counts['n_h200'] / counts['n'] * 100
        counts['lift_100'] = counts['r100'] / baseline_density_100
        counts['lift_200'] = counts['r200'] / baseline_density_200 if baseline_density_200 > 0 else 0
        return counts


def main():
    t0 = time.time()
    print('=== test165: 全市场暴涨股反向分析 ===\n')

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
    g.rename(columns={'d_gua':'stk_d', 'm_gua':'stk_m', 'y_gua':'stk_y'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        mkt[c] = mkt[c].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt.drop_duplicates('date').rename(columns={'d_gua':'mkt_d','m_gua':'mkt_m','y_gua':'mkt_y'})

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','stk_d','d_trend']).reset_index(drop=True)
    print(f'  全市场数据: {len(df):,} 行 (不锁 regime)')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'retail':retail_arr,'mf':mf_arr,'td':trend_arr,
              'starts':code_starts,'ends':code_ends}
    df_e = find_signals_v5(arrays)
    print(f'  v5 入场事件: {len(df_e):,}')

    # ret + 卦象 (按 buy_idx_global 取)
    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets

    # 加卦象 (从 df 取)
    df_e['stk_d'] = df['stk_d'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['stk_m'] = df['stk_m'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['stk_y'] = df['stk_y'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['mkt_d'] = df['mkt_d'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['mkt_m'] = df['mkt_m'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['mkt_y'] = df['mkt_y'].to_numpy()[df_e['buy_idx_global'].astype(int).values]
    df_e['year'] = df_e['date'].str[:4]
    df_e['month'] = df_e['date'].str[5:7]

    # ===== Baseline 全样本暴涨股密度 =====
    n_total = len(df_e)
    n_h100 = (df_e['ret_pct']>=100).sum()
    n_h200 = (df_e['ret_pct']>=200).sum()
    n_h500 = (df_e['ret_pct']>=500).sum()
    base_r100 = n_h100 / n_total * 100
    base_r200 = n_h200 / n_total * 100
    base_r500 = n_h500 / n_total * 100
    print(f'\n{"="*82}')
    print(f'  Baseline 全市场 v5 入场池 ({n_total:,}) 暴涨股密度')
    print(f'{"="*82}')
    print(f'  ≥+100%: {n_h100} ({base_r100:.2f}%)')
    print(f'  ≥+200%: {n_h200} ({base_r200:.2f}%)')
    print(f'  ≥+500%: {n_h500} ({base_r500:.2f}%)')

    # 按年份/月份分布
    print(f'\n  暴涨股按年份分布 (≥+100%):')
    by_year_h = df_e[df_e['ret_pct']>=100].groupby('year').size()
    by_year_n = df_e.groupby('year').size()
    for y in sorted(by_year_n.index):
        h = by_year_h.get(y, 0)
        n = by_year_n[y]
        rate = h/n*100 if n else 0
        bar = '█' * int(rate)
        print(f'    {y}: 总{n:>5}  暴涨股{h:>3} ({rate:>4.1f}%)  {bar}')

    # 按月份
    print(f'\n  暴涨股按月份分布 (≥+100%):')
    by_m_h = df_e[df_e['ret_pct']>=100].groupby('month').size()
    by_m_n = df_e.groupby('month').size()
    for m in sorted(by_m_n.index):
        h = by_m_h.get(m, 0)
        n = by_m_n[m]
        rate = h/n*100 if n else 0
        bar = '█' * int(rate)
        print(f'    {m}: 总{n:>5}  暴涨股{h:>3} ({rate:>4.1f}%)  {bar}')

    # ===== 数值因子 5 桶 lift =====
    print(f'\n{"="*82}')
    print(f'  数值因子 5 桶 lift (lift_200 = 桶内 ≥+200% 密度 / baseline)')
    print(f'{"="*82}')
    NUM_COLS = ['pool_min_retail', 'cur_mf', 'cur_retail', 'cur_trend', 'ret_5d', 'mf_5d']
    for col in NUM_COLS:
        print(f'\n  --- {col} ---')
        res = lift_analysis(df_e, col, base_r100, base_r200)
        print(f'  {"bin":<6} {"范围":<22} {"n":>5} {"≥100":>5} {"r100%":>7} '
              f'{"lift_100":>9} {"≥200":>5} {"r200%":>7} {"lift_200":>9}')
        for _, r in res.iterrows():
            print(f'  {str(r["__bin"]):<6} [{r["mn"]:>+8.0f}, {r["mx"]:>+5.0f}]   {r["n"]:>5} '
                  f'{r["n_h100"]:>5} {r["r100"]:>+6.2f}% {r["lift_100"]:>+8.2f}x '
                  f'{r["n_h200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+8.2f}x')

    # ===== 卦象 lift =====
    print(f'\n{"="*82}')
    print(f'  卦象 lift (按值)')
    print(f'{"="*82}')
    CAT_COLS = ['stk_d', 'stk_m', 'stk_y', 'mkt_d', 'mkt_m', 'mkt_y']
    for col in CAT_COLS:
        print(f'\n  --- {col} ---')
        res = lift_analysis(df_e, col, base_r100, base_r200)
        if len(res) == 0:
            print(f'    (无足够样本)')
            continue
        print(f'  {"value":<7} {"n":>6} {"≥100":>5} {"r100%":>7} {"lift_100":>9} '
              f'{"≥200":>5} {"r200%":>7} {"lift_200":>9}')
        for _, r in res.iterrows():
            print(f'  {r[col]:<7} {r["n"]:>6} {r["n_h100"]:>5} {r["r100"]:>+6.2f}% '
                  f'{r["lift_100"]:>+8.2f}x {r["n_h200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+8.2f}x')

    # ===== 暴涨股清单 (≥+200%) — 全部 =====
    print(f'\n{"="*82}')
    print(f'  ≥+200% 暴涨股全部清单')
    print(f'{"="*82}')
    bigwins = df_e[df_e['ret_pct']>=200].sort_values('ret_pct', ascending=False)
    print(f'\n  共 {len(bigwins)} 只\n')
    print(f'  {"日期":<12} {"代码":<8} {"ret":>8} {"池深":>6} {"cur_mf":>7} {"cur_ret":>8} '
          f'{"cur_td":>7} {"ret5d":>6} {"stk_y":>5} {"mkt_y":>5} {"mkt_m":>5}')
    for _, r in bigwins.iterrows():
        print(f'  {r["date"]:<12} {r["code"]:<8} {r["ret_pct"]:>+7.0f}% '
              f'{r["pool_min_retail"]:>+6.0f} {r["cur_mf"]:>+7.0f} {r["cur_retail"]:>+8.0f} '
              f'{r["cur_trend"]:>+6.1f} {r["ret_5d"]:>+5.0f} {r["stk_y"]:<5} {r["mkt_y"]:<5} {r["mkt_m"]:<5}')

    # ===== 暴涨股的 stk_y / mkt_y 共性 =====
    print(f'\n  ≥+200% 暴涨股的 stk_y 分布:')
    print(bigwins['stk_y'].value_counts().to_string())
    print(f'\n  ≥+200% 暴涨股的 mkt_y 分布:')
    print(bigwins['mkt_y'].value_counts().to_string())
    print(f'\n  ≥+200% 暴涨股的 mkt_m 分布:')
    print(bigwins['mkt_m'].value_counts().to_string())
    print(f'\n  ≥+200% 暴涨股的 stk_m 分布:')
    print(bigwins['stk_m'].value_counts().to_string())

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
