# -*- coding: utf-8 -*-
"""v5 + 坤 regime — cur_mf 横向 / 纵向 双维度评估

横向: 每日 cur_mf 最低 1 只, 按 cur_mf 绝对值拆桶
纵向: 同日多股 cur_mf 排名
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
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'pool_min_retail':pool_min_retail,
                               'cur_mf':mf[gi]})
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


def main():
    t0 = time.time()
    print('=== test163: cur_mf 横向 / 纵向 双维度 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3).replace({'nan':''})
    g.rename(columns={'d_gua':'stk_d'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    mkt['mkt_y'] = mkt['y_gua'].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt[['date','mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','stk_d','d_trend','mkt_y']).reset_index(drop=True)
    df = df[df['mkt_y'] == '000'].reset_index(drop=True)
    print(f'  坤 regime: {len(df):,} 行')

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

    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets
    df_e['year'] = df_e['date'].str[:4]

    baseline = df_e['ret_pct'].mean()
    years = sorted(df_e['year'].unique())
    print(f'  baseline avg: {baseline:+.2f}%, win: {(df_e["ret_pct"]>0).mean()*100:.1f}%')
    print(f'  cur_mf 分布: min={df_e["cur_mf"].min():.0f}, p25={df_e["cur_mf"].quantile(0.25):.0f}, '
          f'med={df_e["cur_mf"].median():.0f}, p75={df_e["cur_mf"].quantile(0.75):.0f}, '
          f'max={df_e["cur_mf"].max():.0f}\n')

    # ===== 横向: 每日 cur_mf 最低 1 只 =====
    print(f'{"="*80}')
    print(f'  横向: 每日 cur_mf 最低 1 只 (asc top-1)')
    print(f'{"="*80}\n')

    df_h = df_e.sort_values(['date', 'cur_mf']).drop_duplicates('date', keep='first').copy()
    print(f'  信号天数: {len(df_h)}')
    print(f'  每日 cur_mf 最低值分布: min={df_h["cur_mf"].min():.0f}, '
          f'p25={df_h["cur_mf"].quantile(0.25):.0f}, '
          f'med={df_h["cur_mf"].median():.0f}, '
          f'p75={df_h["cur_mf"].quantile(0.75):.0f}, '
          f'max={df_h["cur_mf"].max():.0f}')
    print(f'  每日最低 cur_mf avg ret: {df_h["ret_pct"].mean():+.2f}%, '
          f'win: {(df_h["ret_pct"]>0).mean()*100:.1f}%')

    # 5 桶
    print(f'\n  --- 按"每日最低 cur_mf 值"拆 5 桶 (横向看 mf 越低是否越好) ---')
    print(f'\n  {"桶":<6} {"cur_mf 范围":<22} {"n":>5} {"avg ret":>9} {"win":>6} {"年 lift (16/17/18)":>22}')
    df_h['p5'] = pd.qcut(df_h['cur_mf'], 5, labels=[f'q{i+1}' for i in range(5)])
    for q, sub in df_h.groupby('p5', observed=True):
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        rng_lo, rng_hi = sub['cur_mf'].min(), sub['cur_mf'].max()
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_h[df_h['year']==y]
            if len(ys) >= 5 and len(yb) >= 20:
                yl.append(f'{ys["ret_pct"].mean()-yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  {q:<6} [{rng_lo:>+5.0f}, {rng_hi:>+5.0f}]    {len(sub):>5} '
              f'{avg:>+8.2f}% {win:>5.1f}%  {" ".join(yl)}')

    # 阈值
    print(f'\n  --- 阈值: 每日最低 cur_mf 必须 ≤ X (跳过该日否则) ---')
    print(f'\n  {"阈值":<22} {"信号天":>7} {"avg ret":>9} {"win":>6} {"年 lift":>22}')
    for thr in [-200, -100, -50, 0, 50, 100, 200]:
        sub = df_h[df_h['cur_mf'] <= thr]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_h[df_h['year']==y]
            if len(ys) >= 5 and len(yb) >= 20:
                yl.append(f'{ys["ret_pct"].mean()-yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  当日最低 cur_mf <= {thr:>+4}    {len(sub):>7} '
              f'{avg:>+8.2f}% {win:>5.1f}%  {" ".join(yl)}')

    # ===== 纵向: 同日多股 cur_mf 排名 =====
    print(f'\n{"="*80}')
    print(f'  纵向: 同日多股 cur_mf 排名 (rank=1 最低)')
    print(f'{"="*80}\n')

    df_v = df_e.copy()
    df_v['day_rank'] = df_v.groupby('date')['cur_mf'].rank(method='first', ascending=True)
    df_v['day_n'] = df_v.groupby('date')['code'].transform('count')
    df_multi = df_v[df_v['day_n'] >= 2].copy()
    print(f'  多信号天 (≥2): {df_multi["date"].nunique()} 天, 总事件 {len(df_multi)}')

    print(f'\n  --- 同日 rank 1-5 的 avg ret (rank=1 最低 cur_mf) ---')
    print(f'\n  {"day_rank":<10} {"n":>5} {"avg ret":>9} {"win":>6} {"年 lift":>22}')
    base_multi = df_multi['ret_pct'].mean()
    for r in [1, 2, 3, 4, 5]:
        sub = df_multi[df_multi['day_rank'] == r]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_multi[df_multi['year']==y]
            if len(ys) >= 5 and len(yb) >= 20:
                yl.append(f'{ys["ret_pct"].mean()-yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  rank {r:<5}  {len(sub):>5} {avg:>+8.2f}% {win:>5.1f}%  {" ".join(yl)}')
    print(f'  base       {len(df_multi):>5} {base_multi:>+8.2f}% '
          f'{(df_multi["ret_pct"]>0).mean()*100:>5.1f}%  (多信号天 baseline)')

    # 分位
    print(f'\n  --- 同日 rank 分位 (top 25% / mid 50% / bot 25%) ---')
    df_multi['rank_pct'] = df_multi.groupby('date')['cur_mf'].rank(pct=True)
    print(f'\n  {"分位":<14} {"n":>5} {"avg ret":>9} {"win":>6} {"年 lift":>22}')
    for label, lo, hi in [('top 25%(低)', 0, 0.25), ('mid 50%', 0.25, 0.75), ('bot 25%(高)', 0.75, 1.001)]:
        sub = df_multi[(df_multi['rank_pct'] > lo) & (df_multi['rank_pct'] <= hi)]
        if len(sub) < 30: continue
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_multi[df_multi['year']==y]
            if len(ys) >= 5 and len(yb) >= 20:
                yl.append(f'{ys["ret_pct"].mean()-yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  {label:<14} {len(sub):>5} {avg:>+8.2f}% {win:>5.1f}%  {" ".join(yl)}')

    # ===== 联合: 每日最低 (rank 1) + cur_mf 绝对值 4 桶 =====
    print(f'\n{"="*80}')
    print(f'  联合: 每日最低 + cur_mf 绝对值 4 桶 (横+纵交叉)')
    print(f'{"="*80}\n')
    df_h_only = df_v[df_v['day_rank'] == 1].copy()
    df_h_only['p4'] = pd.qcut(df_h_only['cur_mf'], 4, labels=['q1', 'q2', 'q3', 'q4'])
    print(f'  {"桶":<6} {"cur_mf 范围":<22} {"n":>5} {"avg ret":>9} {"win":>6} {"年 lift":>22}')
    for q, sub in df_h_only.groupby('p4', observed=True):
        avg = sub['ret_pct'].mean()
        win = (sub['ret_pct']>0).mean()*100
        rng_lo, rng_hi = sub['cur_mf'].min(), sub['cur_mf'].max()
        yl = []
        for y in years:
            ys = sub[sub['year']==y]; yb = df_h_only[df_h_only['year']==y]
            if len(ys) >= 5 and len(yb) >= 20:
                yl.append(f'{ys["ret_pct"].mean()-yb["ret_pct"].mean():+5.1f}')
            else:
                yl.append('  -- ')
        print(f'  {q:<6} [{rng_lo:>+5.0f}, {rng_hi:>+5.0f}]    {len(sub):>5} '
              f'{avg:>+8.2f}% {win:>5.1f}%  {" ".join(yl)}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
