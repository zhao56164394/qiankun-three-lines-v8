# -*- coding: utf-8 -*-
"""固定 F 机制 (散户线买卖), 重扫入场池

机制:
  入场 (signal_date 后 60 天内): 散户线上穿 0 (retail[k-1]<=0, retail[k]>0) AND trend>11
  卖出 (D6'): 散户线连续 2 天 <=0 (retail[k-1]<=0, retail[k]<=0)
  再买入 (U1): mf 上升
  T0: trend<11

任务:
  1. 全市场扫所有 (mkt 阴阳, stk 阴阳) 64 组合 + 散户线机制 → 暴涨股密度 lift
  2. 跨年表现, 找跨年都稳的入场池
  3. 考虑 trend / mf / retail 数值阈值组合
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WAIT_MAX = 60


def find_signals_all(arrays, mkt_arrs):
    """所有波段起点 (trend 下穿 11), 不加阴阳/数值过滤"""
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
                sigs.append({
                    'signal_idx': i, 'signal_date': date[i], 'code': code[i],
                    'cur_mf': mf[i], 'cur_retail': retail[i],
                    'stk_d_t': stk_d_t[i], 'stk_m_t': stk_m_t[i], 'stk_y_t': stk_y_t[i],
                    'mkt_d_t': mkt_d_t[i], 'mkt_m_t': mkt_m_t[i], 'mkt_y_t': mkt_y_t[i],
                    'code_end': e,
                })
    return pd.DataFrame(sigs)


def find_entry_retail_cross(signal_idx, code_end, td, retail):
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if td[k] <= 11: continue
        if retail[k-1] <= 0 and retail[k] > 0:
            return k
    return -1


def simulate_F(buy_idx, code_end, td, close, mf, retail):
    """F: 散户线连续 2 天 <=0 卖, mf 上升买回, T0 清"""
    bp = close[buy_idx]; cum = 1.0; holding = True
    cur_buy = bp; legs = 1
    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue
        if td[k] < 11:
            if holding: cum *= close[k]/cur_buy
            return (cum-1)*100, legs
        if k<1: continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        mfc = mf[k]-mf[k-1]
        if holding:
            if retail[k-1] <= 0 and retail[k] <= 0:
                cum *= close[k]/cur_buy; holding = False
        else:
            if mfc>0:
                cur_buy = close[k]; holding = True; legs += 1
    if holding: cum *= close[code_end-1]/cur_buy
    return (cum-1)*100, legs


def yyy(d, m, y, thr=50):
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def main():
    t0 = time.time()
    print('=== test187: F 机制 + 入场池扫描 ===\n')

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

    # 全部波段起点
    print('  扫信号...')
    df_sig = find_signals_all(arrays, mkt_arrs)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'    全市场起点: {len(df_sig):,}')

    # F 入场建仓 + 模拟
    print('  F 入场建仓 + 模拟...')
    rets = []
    legs_l = []
    entry_dates = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end'])
        ei = find_entry_retail_cross(si, ce, trend_arr, retail_arr)
        if ei < 0:
            rets.append(np.nan); legs_l.append(0); entry_dates.append(None)
            continue
        ret, legs = simulate_F(ei, ce, trend_arr, close_arr, mf_arr, retail_arr)
        rets.append(ret); legs_l.append(legs); entry_dates.append(date_arr[ei])
    df_sig['ret'] = rets
    df_sig['legs'] = legs_l
    df_sig['entry_date'] = entry_dates
    df_sig['mkt_yy'] = [yyy(d,m,y) for d,m,y in zip(df_sig['mkt_d_t'], df_sig['mkt_m_t'], df_sig['mkt_y_t'])]
    df_sig['stk_yy'] = [yyy(d,m,y) for d,m,y in zip(df_sig['stk_d_t'], df_sig['stk_m_t'], df_sig['stk_y_t'])]
    df_sig['year'] = df_sig['signal_date'].str[:4]

    df_e = df_sig.dropna(subset=['ret']).reset_index(drop=True)
    print(f'    成功建仓: {len(df_e):,}\n')

    # baseline
    n_total = len(df_e); n100 = (df_e['ret']>=100).sum(); n200 = (df_e['ret']>=200).sum()
    base_r100 = n100/n_total*100; base_r200 = n200/n_total*100
    base_avg = df_e['ret'].mean()
    print(f'  全样本 baseline: n={n_total:,}, avg={base_avg:+.2f}%, '
          f'r100={base_r100:.2f}%, r200={base_r200:.2f}%\n')

    # ===== 1. mkt 单独阴阳 =====
    print(f'{"="*82}')
    print(f'  mkt 单独阴阳 (F 机制下)')
    print(f'{"="*82}')
    print(f'\n  {"mkt":<5} {"n":>6} {"avg":>7} {"≥100":>5} {"r100":>7} {"L100":>6} {"≥200":>5} {"r200":>7} {"L200":>6}')
    for v, sub in df_e.groupby('mkt_yy'):
        if len(sub) < 200: continue
        n = len(sub); h100 = (sub['ret']>=100).sum(); h200 = (sub['ret']>=200).sum()
        r100 = h100/n*100; r200 = h200/n*100
        print(f'  {v:<5} {n:>6,} {sub["ret"].mean():>+6.2f}% {h100:>5} '
              f'{r100:>+6.2f}% {r100/base_r100:>+5.2f}x {h200:>5} {r200:>+6.2f}% {r200/base_r200:>+5.2f}x')

    # ===== 2. stk 单独 =====
    print(f'\n{"="*82}')
    print(f'  stk 单独阴阳')
    print(f'{"="*82}')
    print(f'\n  {"stk":<5} {"n":>6} {"avg":>7} {"≥100":>5} {"r100":>7} {"L100":>6}')
    for v, sub in df_e.groupby('stk_yy'):
        if len(sub) < 200: continue
        n = len(sub); h100 = (sub['ret']>=100).sum()
        r100 = h100/n*100
        print(f'  {v:<5} {n:>6,} {sub["ret"].mean():>+6.2f}% {h100:>5} {r100:>+6.2f}% {r100/base_r100:>+5.2f}x')

    # ===== 3. 64 组合 =====
    print(f'\n{"="*82}')
    print(f'  mkt × stk 64 组合 (n>=100, 按 lift_200 排)')
    print(f'{"="*82}')

    rows = []
    for (mv, sv), sub in df_e.groupby(['mkt_yy', 'stk_yy']):
        if len(sub) < 100: continue
        n = len(sub); h100 = (sub['ret']>=100).sum(); h200 = (sub['ret']>=200).sum()
        rows.append({'mkt':mv,'stk':sv,'n':n,'h100':h100,'h200':h200,
                     'r100':h100/n*100,'r200':h200/n*100,
                     'lift_100':(h100/n*100)/base_r100,
                     'lift_200':(h200/n*100)/base_r200 if base_r200 else 0,
                     'avg':sub['ret'].mean()})
    df_c = pd.DataFrame(rows).sort_values('lift_200', ascending=False)
    print(f'\n  共 {len(df_c)} 组')
    print(f'  {"mkt":<4} {"stk":<4} {"n":>6} {"avg":>7} {"r100":>7} {"L100":>6} {"r200":>7} {"L200":>6}')
    for _, r in df_c.iterrows():
        print(f'  {r["mkt"]:<4} {r["stk"]:<4} {r["n"]:>6,} {r["avg"]:>+6.2f}% '
              f'{r["r100"]:>+6.2f}% {r["lift_100"]:>+5.2f}x {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    # ===== 4. cur_mf / cur_retail 阈值 (全样本) =====
    print(f'\n{"="*82}')
    print(f'  cur_mf / cur_retail 阈值 单独 (F 机制全样本)')
    print(f'{"="*82}')

    print(f'\n  --- cur_mf 阈值 ---')
    print(f'  {"条件":<22} {"n":>6} {"avg":>7} {"r100":>7} {"L100":>6}')
    for op, thr in [('<=', -200), ('<=', -100), ('<=', -50), ('<=', 0), ('>=', 0), ('>=', 100), ('>=', 200)]:
        sub = df_e[df_e['cur_mf']<=thr] if op=='<=' else df_e[df_e['cur_mf']>=thr]
        label = f'cur_mf {op} {thr}'
        if len(sub) < 100: continue
        n = len(sub); h100 = (sub['ret']>=100).sum()
        r100 = h100/n*100
        print(f'  {label:<22} {n:>6,} {sub["ret"].mean():>+6.2f}% {r100:>+6.2f}% {r100/base_r100:>+5.2f}x')

    print(f'\n  --- cur_retail 阈值 ---')
    for op, thr in [('<=', -250), ('<=', -150), ('<=', -50), ('<=', 0), ('>=', 0), ('>=', 100), ('>=', 200)]:
        sub = df_e[df_e['cur_retail']<=thr] if op=='<=' else df_e[df_e['cur_retail']>=thr]
        label = f'cur_retail {op} {thr}'
        if len(sub) < 100: continue
        n = len(sub); h100 = (sub['ret']>=100).sum()
        r100 = h100/n*100
        print(f'  {label:<22} {n:>6,} {sub["ret"].mean():>+6.2f}% {r100:>+6.2f}% {r100/base_r100:>+5.2f}x')

    # ===== 5. Top 10 组合跨年 稳定性 =====
    print(f'\n{"="*82}')
    print(f'  Top 10 组合 跨年稳定性 (按 r100 排)')
    print(f'{"="*82}')
    print(f'\n  {"":<5}{"":<4}', end='')
    years = sorted(df_e['year'].unique())
    for y in years: print(f' {y[-2:]:>5}', end='')
    print(f' {"全 r100":>8}')

    df_c2 = df_c.sort_values('r100', ascending=False).head(10)
    for _, r in df_c2.iterrows():
        sub = df_e[(df_e['mkt_yy']==r['mkt']) & (df_e['stk_yy']==r['stk'])]
        print(f'  {r["mkt"]:<4} {r["stk"]:<4}', end='')
        for y in years:
            ys = sub[sub['year']==y]
            if len(ys) < 20: print(f' {"--":>5}', end='')
            else:
                rr = (ys['ret']>=100).mean()*100
                print(f' {rr:>+4.1f}%', end='')
        print(f' {r["r100"]:>+6.2f}%')

    # ===== 6. 看跨年都稳的组合: r100 在 ≥6 个年份高于 baseline 1.5x =====
    print(f'\n{"="*82}')
    print(f'  跨年都稳 (高于年度 baseline 1.5x 的年数)')
    print(f'{"="*82}')

    # 每年 baseline
    year_base = {}
    for y in years:
        yb = df_e[df_e['year']==y]
        if len(yb) > 0:
            year_base[y] = (yb['ret']>=100).mean()*100

    rows3 = []
    for (mv, sv), sub in df_e.groupby(['mkt_yy', 'stk_yy']):
        if len(sub) < 200: continue
        won = 0; have = 0
        for y in years:
            ys = sub[sub['year']==y]
            if len(ys) < 30: continue
            rr = (ys['ret']>=100).mean()*100
            base_y = year_base.get(y, 0)
            if base_y > 0:
                have += 1
                if rr > base_y * 1.3:
                    won += 1
        if have >= 5:
            rows3.append({'mkt':mv,'stk':sv,'n':len(sub),
                          'r100':(sub['ret']>=100).mean()*100,
                          'won':won,'have':have,'win_rate':won/have*100})
    df_y = pd.DataFrame(rows3).sort_values(['win_rate', 'r100'], ascending=[False, False])
    print(f'\n  {"mkt":<4} {"stk":<4} {"n":>6} {"r100":>7} {"赢年":>5} {"有数据年":>9} {"赢率":>6}')
    for _, r in df_y.head(15).iterrows():
        print(f'  {r["mkt"]:<4} {r["stk"]:<4} {r["n"]:>6,} {r["r100"]:>+6.2f}% '
              f'{r["won"]:>5} {r["have"]:>9} {r["win_rate"]:>+5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
