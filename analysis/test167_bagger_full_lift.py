# -*- coding: utf-8 -*-
"""1483 只 ≥+100% 暴涨股 全特征 lift 扫描 + 分簇检查

事件池: NoP (mf 上升+trend>11+retail 上升, 30d 不重复) ≈ 183K
暴涨股: 2105 只 ≥+100%

任务 1 - 共性指纹 (全市场扫):
  对每个特征, 计算: 该特征下暴涨股密度 / baseline 密度 (lift)
  数值列 5 桶, 卦象列 按值

任务 2 - 是否需要分簇:
  暴涨股按 "市场 regime / 入场时 retail / 入场时 trend" 分簇
  看每簇内的指纹是否相同

  - regime 分簇 (mkt_y=000/001/.../111): 暴涨股密度差异?
  - cur_retail 分簇 (深 / 中性 / 浅): 暴涨股的 mkt_m 是否一样?
  - 年份分簇: 2016/2020/2024 三大年的暴涨股指纹是否一样?
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_nopool(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        last_mf = np.nan; last_retail = np.nan
        last_trigger = -999
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if mf_rising and retail_rising and trend_ok and (i - last_trigger) >= 30:
                ci_s = arrays['starts'][ci]
                i5 = max(gi - 5, ci_s)
                i30 = max(gi - 30, ci_s)
                ret_5d_v = retail[gi] - retail[i5] if not np.isnan(retail[i5]) else np.nan
                mf_5d_v = mf[gi] - mf[i5] if not np.isnan(mf[i5]) else np.nan
                td_5d_v = td[gi] - td[i5] if not np.isnan(td[i5]) else np.nan
                # 30 日窗口最低值
                ret_30d_min = np.nanmin(retail[i30:gi+1]) if (gi+1) > i30 else np.nan
                mf_30d_min = np.nanmin(mf[i30:gi+1]) if (gi+1) > i30 else np.nan
                events.append({
                    'date':date[gi],'code':code[gi],
                    'buy_idx_global':gi,
                    'cur_retail':retail[gi],
                    'cur_mf':mf[gi],
                    'cur_trend':td[gi],
                    'ret_5d':ret_5d_v, 'mf_5d':mf_5d_v, 'td_5d':td_5d_v,
                    'ret_30d_min':ret_30d_min, 'mf_30d_min':mf_30d_min,
                })
                last_trigger = i
            last_mf = mf[gi]; last_retail = retail[gi]
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


def scan_numeric(df, col, base_r100, base_r200, n_bins=5):
    df_sub = df.dropna(subset=[col]).copy()
    try:
        df_sub['__bin'] = pd.qcut(df_sub[col], n_bins,
                                    labels=[f'q{i+1}' for i in range(n_bins)], duplicates='drop')
    except:
        return pd.DataFrame()
    rows = []
    for q, g in df_sub.groupby('__bin', observed=True):
        n = len(g); n100 = (g['ret_pct']>=100).sum(); n200 = (g['ret_pct']>=200).sum()
        rows.append({'col':col,'bin':q,
                     'mn':g[col].min(),'mx':g[col].max(),
                     'n':n,'n100':n100,'n200':n200,
                     'r100':n100/n*100,'r200':n200/n*100,
                     'lift_100':(n100/n*100)/base_r100 if base_r100 else 0,
                     'lift_200':(n200/n*100)/base_r200 if base_r200 else 0})
    return pd.DataFrame(rows)


def scan_categorical(df, col, base_r100, base_r200, min_n=500):
    rows = []
    for v, g in df.groupby(col):
        n = len(g)
        if n < min_n: continue
        n100 = (g['ret_pct']>=100).sum(); n200 = (g['ret_pct']>=200).sum()
        rows.append({'col':col,'value':v,'n':n,
                     'n100':n100,'n200':n200,
                     'r100':n100/n*100,'r200':n200/n*100,
                     'lift_100':(n100/n*100)/base_r100 if base_r100 else 0,
                     'lift_200':(n200/n*100)/base_r200 if base_r200 else 0})
    return pd.DataFrame(rows).sort_values('lift_200', ascending=False)


def main():
    t0 = time.time()
    print('=== test167: 暴涨股全特征 lift + 分簇检查 ===\n')

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
    print(f'  全市场: {len(df):,} 行')

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

    print('  生成 NoP 信号...')
    df_e = find_signals_nopool(arrays)
    print(f'    {len(df_e):,} 事件')

    print('  计算 ret + 加卦象...')
    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets

    gi_arr = df_e['buy_idx_global'].astype(int).values
    df_e['stk_d'] = df['stk_d'].to_numpy()[gi_arr]
    df_e['stk_m'] = df['stk_m'].to_numpy()[gi_arr]
    df_e['stk_y'] = df['stk_y'].to_numpy()[gi_arr]
    df_e['mkt_d'] = df['mkt_d'].to_numpy()[gi_arr]
    df_e['mkt_m'] = df['mkt_m'].to_numpy()[gi_arr]
    df_e['mkt_y'] = df['mkt_y'].to_numpy()[gi_arr]
    df_e['year'] = df_e['date'].str[:4]

    n_total = len(df_e)
    n_h100 = (df_e['ret_pct']>=100).sum()
    n_h200 = (df_e['ret_pct']>=200).sum()
    base_r100 = n_h100 / n_total * 100
    base_r200 = n_h200 / n_total * 100
    print(f'\n  Baseline: n={n_total:,}, ≥+100%={n_h100} ({base_r100:.2f}%), '
          f'≥+200%={n_h200} ({base_r200:.2f}%)\n')

    # ===== 1. 全市场全特征 lift =====
    print(f'{"="*82}')
    print(f'  任务 1: 全市场暴涨股共性指纹 lift')
    print(f'{"="*82}')

    NUM = ['cur_retail', 'cur_mf', 'cur_trend', 'ret_5d', 'mf_5d', 'td_5d',
            'ret_30d_min', 'mf_30d_min']
    print(f'\n  --- 数值因子 5 分位 ---')
    for col in NUM:
        res = scan_numeric(df_e, col, base_r100, base_r200)
        if len(res) == 0: continue
        print(f'\n  {col}:')
        print(f'    {"bin":<5} {"范围":<22} {"n":>7} {"≥100":>5} {"r100%":>7} '
              f'{"L100":>7} {"≥200":>5} {"r200%":>7} {"L200":>7}')
        for _, r in res.iterrows():
            print(f'    {str(r["bin"]):<5} [{r["mn"]:>+8.0f}, {r["mx"]:>+5.0f}]   {r["n"]:>7,} '
                  f'{r["n100"]:>5} {r["r100"]:>+6.2f}% {r["lift_100"]:>+5.2f}x '
                  f'{r["n200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    CAT = ['stk_d', 'stk_m', 'stk_y', 'mkt_d', 'mkt_m', 'mkt_y']
    print(f'\n  --- 卦象列 (按 lift_200 排) ---')
    for col in CAT:
        res = scan_categorical(df_e, col, base_r100, base_r200)
        if len(res) == 0: continue
        print(f'\n  {col}:')
        print(f'    {"value":<6} {"n":>7} {"≥100":>5} {"r100%":>7} {"L100":>6} '
              f'{"≥200":>5} {"r200%":>7} {"L200":>6}')
        for _, r in res.iterrows():
            print(f'    {r["value"]:<6} {r["n"]:>7,} {r["n100"]:>5} {r["r100"]:>+6.2f}% '
                  f'{r["lift_100"]:>+5.2f}x {r["n200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    # ===== 2. 分簇检查 =====
    print(f'\n{"="*82}')
    print(f'  任务 2: 是否需要分簇? — 子集内 lift 是否一样')
    print(f'{"="*82}')

    # 簇 1: 按年份大年/小年
    print(f'\n  --- 按年份分簇 (大年 16/20/24 vs 小年其他) ---')
    big_years = ['2016', '2020', '2024']
    df_e['year_cluster'] = df_e['year'].apply(lambda y: 'big' if y in big_years else 'small')
    for cluster, g in df_e.groupby('year_cluster'):
        n = len(g); n100 = (g['ret_pct']>=100).sum(); n200 = (g['ret_pct']>=200).sum()
        print(f'    {cluster}: n={n:,}, ≥+100%={n100} ({n100/n*100:.2f}%), '
              f'≥+200%={n200} ({n200/n*100:.2f}%)')

    # 大年的 mkt_m 分布 vs 小年的 mkt_m 分布
    print(f'\n  --- 大年/小年 中, mkt_m lift_200 对比 (看是否一致) ---')
    print(f'    {"mkt_m":<8}', end='')
    for cluster in ['big', 'small']:
        print(f' {cluster:>10}', end='')
    print()
    for v in sorted(df_e['mkt_m'].unique()):
        if v == '': continue
        print(f'    {v:<8}', end='')
        for cluster in ['big', 'small']:
            sub = df_e[(df_e['mkt_m']==v) & (df_e['year_cluster']==cluster)]
            if len(sub) < 200:
                print(f' {"--":>10}', end=''); continue
            r200 = (sub['ret_pct']>=200).sum()/len(sub)*100
            base_c = df_e[df_e['year_cluster']==cluster]
            base_r = (base_c['ret_pct']>=200).sum()/len(base_c)*100
            lift = r200 / base_r if base_r else 0
            print(f' {lift:>+8.2f}x', end='')
        print()

    # 簇 2: 按 cur_retail 分簇 (深抛 / 中性 / 上涨中)
    print(f'\n  --- 按 cur_retail 分簇 ---')
    df_e['ret_cluster'] = pd.cut(df_e['cur_retail'],
                                  bins=[-2000, -150, 0, 100, 2000],
                                  labels=['深(<-150)', '中下(-150~0)', '中上(0~100)', '高(>100)'])
    for cluster, g in df_e.groupby('ret_cluster', observed=True):
        n = len(g); n100 = (g['ret_pct']>=100).sum(); n200 = (g['ret_pct']>=200).sum()
        print(f'    {cluster}: n={n:,}, ≥+100%={n100} ({n100/n*100:.2f}%), '
              f'≥+200%={n200} ({n200/n*100:.2f}%)')

    # 不同 ret 簇内, 卦象 lift 是否一致
    print(f'\n  --- 各 ret 簇内, mkt_m lift_200 对比 ---')
    print(f'    {"mkt_m":<8}', end='')
    for cluster in ['深(<-150)', '中下(-150~0)', '中上(0~100)', '高(>100)']:
        print(f' {cluster:>14}', end='')
    print()
    for v in sorted(df_e['mkt_m'].unique()):
        if v == '': continue
        print(f'    {v:<8}', end='')
        for cluster in ['深(<-150)', '中下(-150~0)', '中上(0~100)', '高(>100)']:
            sub = df_e[(df_e['mkt_m']==v) & (df_e['ret_cluster']==cluster)]
            if len(sub) < 200:
                print(f' {"--":>14}', end=''); continue
            r200 = (sub['ret_pct']>=200).sum()/len(sub)*100
            base_c = df_e[df_e['ret_cluster']==cluster]
            base_r = (base_c['ret_pct']>=200).sum()/len(base_c)*100
            lift = r200 / base_r if base_r else 0
            print(f' {lift:>+12.2f}x', end='')
        print()

    # 簇 3: 按 mkt_y (8 regime) — 老式分治
    print(f'\n  --- 按 mkt_y 分簇 (8 regime) ---')
    print(f'    {"mkt_y":<6} {"n":>8} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7}')
    for v, g in df_e.groupby('mkt_y'):
        if v == '' or len(g) < 1000: continue
        n = len(g); n100 = (g['ret_pct']>=100).sum(); n200 = (g['ret_pct']>=200).sum()
        print(f'    {v:<6} {n:>8,} {n100:>5} {n100/n*100:>+6.2f}% {n200:>5} {n200/n*100:>+6.2f}%')

    # 不同 regime 内, cur_retail lift_200
    print(f'\n  --- 各 regime 内, cur_retail 5 桶 lift_200 是否单调一致 ---')
    print(f'    {"regime":<8} {"q1(深)":>10} {"q2":>8} {"q3":>8} {"q4":>8} {"q5(浅)":>10}')
    for regime, g in df_e.groupby('mkt_y'):
        if regime == '' or len(g) < 1500: continue
        try:
            g = g.copy()
            g['__bin'] = pd.qcut(g['cur_retail'], 5,
                                  labels=['q1','q2','q3','q4','q5'], duplicates='drop')
        except:
            continue
        base_r = (g['ret_pct']>=200).sum()/len(g)*100 if len(g) else 0
        print(f'    {regime:<8}', end='')
        for q in ['q1','q2','q3','q4','q5']:
            sub = g[g['__bin']==q]
            if len(sub) < 100:
                print(f' {"--":>9}', end=''); continue
            r = (sub['ret_pct']>=200).sum()/len(sub)*100
            lift = r/base_r if base_r else 0
            print(f' {lift:>+7.2f}x', end='')
        print()

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
