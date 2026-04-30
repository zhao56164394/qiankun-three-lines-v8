# -*- coding: utf-8 -*-
"""stk=011 (d 阴/m 阳/y 阳) 三个深入

并行 3 个分析 (限 2016+):
A. m_t / y_t 数值阈值: 50/60/70/80, 看暴涨股密度变化
B. 加第三维 (cur_retail / cur_mf) 过滤后的密度
C. stk=011 ≥+100% 暴涨股清单 (跨年都有的, 各年取代表)
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
            # 找 max 处和 min 处的日期
            idx_max_g = band_start + idx_max_local
            seg_close_arr = close[band_start:band_start+idx_max_local+1]
            idx_min_local = int(np.nanargmin(seg_close_arr))
            idx_min_g = band_start + idx_min_local
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
    print('=== test177: stk=011 三深入分析 ===\n')

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
    df_b['stk_d_t'] = [stk_d_t[i] for i in si]
    df_b['stk_m_t'] = [stk_m_t[i] for i in si]
    df_b['stk_y_t'] = [stk_y_t[i] for i in si]
    df_b['mkt_d_t'] = [mkt_d_t[i] for i in si]
    df_b['mkt_m_t'] = [mkt_m_t[i] for i in si]
    df_b['mkt_y_t'] = [mkt_y_t[i] for i in si]
    df_b['cur_retail'] = [retail_arr[i] for i in si]
    df_b['cur_mf'] = [mf_arr[i] for i in si]
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['year'] = df_b['start_date'].str[:4]

    n_total = len(df_b); n100 = (df_b['gain_pct']>=100).sum(); n200 = (df_b['gain_pct']>=200).sum()
    base_r100 = n100/n_total*100
    print(f'  全样本 (2016+): {n_total:,}, baseline r100={base_r100:.2f}%')

    # ===== stk=011 子集 =====
    s011 = df_b[df_b['stk_yy']=='011'].copy()
    n_s011 = len(s011)
    s011_r100 = (s011['gain_pct']>=100).mean()*100
    print(f'  stk=011: n={n_s011:,}, r100={s011_r100:.2f}%')

    # ===== A: m_t / y_t 数值阈值精化 =====
    print(f'\n{"="*82}')
    print(f'  A: stk=011 内 m_t / y_t 阈值精化')
    print(f'{"="*82}')

    print(f'\n  --- m_t (个股月线 trend) 阈值 ---')
    print(f'  {"阈值":<14} {"n":>5} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7}')
    for thr in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        sub = s011[s011['stk_m_t'] >= thr]
        if len(sub) < 50: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        print(f'  m_t >= {thr:>3}    {n:>5} {h100:>5} {h100/n*100:>+6.2f}% {h200:>5} {h200/n*100:>+6.2f}%')

    print(f'\n  --- y_t (个股年线 trend) 阈值 ---')
    print(f'  {"阈值":<14} {"n":>5} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7}')
    for thr in [50, 55, 60, 65, 70, 75, 80, 85, 90]:
        sub = s011[s011['stk_y_t'] >= thr]
        if len(sub) < 50: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        print(f'  y_t >= {thr:>3}    {n:>5} {h100:>5} {h100/n*100:>+6.2f}% {h200:>5} {h200/n*100:>+6.2f}%')

    print(f'\n  --- m_t × y_t 联合 (4 桶) ---')
    for m_thr, y_thr in [(50, 50), (60, 60), (70, 70), (80, 80), (50, 70), (70, 50)]:
        sub = s011[(s011['stk_m_t'] >= m_thr) & (s011['stk_y_t'] >= y_thr)]
        if len(sub) < 50: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        # 跨年
        years_won = 0; years_with_data = 0
        for y, ys in sub.groupby('year'):
            if len(ys) < 30: continue
            yb = df_b[df_b['year']==y]
            base_y = (yb['gain_pct']>=100).mean()*100
            r_y = (ys['gain_pct']>=100).mean()*100
            years_with_data += 1
            if r_y > base_y * 1.3: years_won += 1
        print(f'  m_t>={m_thr},y_t>={y_thr}: n={n:>5} r100={h100/n*100:>+5.2f}% '
              f'r200={h200/n*100:>+5.2f}% 跨年赢{years_won}/{years_with_data}')

    # ===== B: 加 cur_retail / cur_mf 第三维 =====
    print(f'\n{"="*82}')
    print(f'  B: stk=011 + cur_retail / cur_mf 数值过滤')
    print(f'{"="*82}')

    print(f'\n  --- cur_retail 阈值 ---')
    print(f'  {"阈值":<22} {"n":>5} {"r100":>7} {"r200":>7}')
    for op, thr in [('<=', -250), ('<=', -150), ('<=', -50), ('<=', 0),
                     ('>=', 0), ('>=', 50), ('>=', 100), ('>=', 200)]:
        if op == '<=':
            sub = s011[s011['cur_retail'] <= thr]
            label = f'cur_retail <= {thr}'
        else:
            sub = s011[s011['cur_retail'] >= thr]
            label = f'cur_retail >= {thr}'
        if len(sub) < 50: continue
        n = len(sub); r100 = (sub['gain_pct']>=100).mean()*100; r200 = (sub['gain_pct']>=200).mean()*100
        print(f'  {label:<22} {n:>5} {r100:>+6.2f}% {r200:>+6.2f}%')

    print(f'\n  --- cur_mf 阈值 ---')
    print(f'  {"阈值":<22} {"n":>5} {"r100":>7} {"r200":>7}')
    for op, thr in [('<=', -200), ('<=', -100), ('<=', -50), ('<=', 0),
                     ('>=', 0), ('>=', 50), ('>=', 100), ('>=', 200)]:
        if op == '<=':
            sub = s011[s011['cur_mf'] <= thr]
            label = f'cur_mf <= {thr}'
        else:
            sub = s011[s011['cur_mf'] >= thr]
            label = f'cur_mf >= {thr}'
        if len(sub) < 50: continue
        n = len(sub); r100 = (sub['gain_pct']>=100).mean()*100; r200 = (sub['gain_pct']>=200).mean()*100
        print(f'  {label:<22} {n:>5} {r100:>+6.2f}% {r200:>+6.2f}%')

    print(f'\n  --- cur_mf × cur_retail 二维交叉 (4×4) ---')
    print(f'\n  {"":<8} {"ret<-100":>10} {"ret-100~0":>11} {"ret 0~100":>11} {"ret>100":>10}')
    for mf_label, mf_lo, mf_hi in [('mf<-100', -1e9, -100), ('mf-100~0', -100, 0),
                                      ('mf 0~100', 0, 100), ('mf>100', 100, 1e9)]:
        print(f'  {mf_label:<8}', end='')
        for r_lo, r_hi in [(-1e9, -100), (-100, 0), (0, 100), (100, 1e9)]:
            sub = s011[(s011['cur_mf'] > mf_lo) & (s011['cur_mf'] <= mf_hi) &
                        (s011['cur_retail'] > r_lo) & (s011['cur_retail'] <= r_hi)]
            if len(sub) < 30:
                print(f' {"--":>10}', end='')
            else:
                r100 = (sub['gain_pct']>=100).mean()*100
                print(f' {r100:>+6.2f}%(n{len(sub):>3})', end='')
        print()

    # ===== C: 暴涨股清单 =====
    print(f'\n{"="*82}')
    print(f'  C: stk=011 ≥+100% 暴涨股清单 (各年代表 8 只)')
    print(f'{"="*82}')

    big = s011[s011['gain_pct']>=100].copy()
    print(f'\n  stk=011 暴涨股共 {len(big)} 只')

    for y, sub in big.groupby('year'):
        if len(sub) == 0: continue
        sub = sub.sort_values('gain_pct', ascending=False)
        print(f'\n  --- {y} ({len(sub)} 只) ---')
        print(f'  {"代码":<8} {"起点":<12} {"min点":<12} {"max点":<12} {"持续":>4} '
              f'{"min价":>6} {"max价":>6} {"涨幅":>9} '
              f'{"d_t":>5} {"m_t":>5} {"y_t":>5} {"mkt":>5}')
        for _, r in sub.head(8).iterrows():
            print(f'  {r["code"]:<8} {r["start_date"]:<12} {r["min_date"]:<12} {r["max_date"]:<12} '
                  f'{r["days"]:>3}d {r["min_close"]:>5.2f} {r["max_close"]:>5.2f} '
                  f'{r["gain_pct"]:>+8.1f}% {r["stk_d_t"]:>+4.0f} {r["stk_m_t"]:>+4.0f} {r["stk_y_t"]:>+4.0f} '
                  f'{r["mkt_yy"]:>5}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
