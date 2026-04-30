# -*- coding: utf-8 -*-
"""stk=011 + 大盘阴阳 跨年稳定性

stk=011 全样本 baseline 4.02%
看 stk=011 + 各 mkt 阴阳 在每年的密度
找出每年都赢的组合
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
            bands.append({
                'code': code[band_start],
                'start_idx': band_start,
                'start_date': date[band_start],
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
    print('=== test178: stk=011 + mkt 阴阳 跨年稳定性 ===\n')

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
    df_b_all = find_bands(arrays)
    df_b = df_b_all[df_b_all['start_date'] >= '2016-01-01'].copy()

    si = df_b['start_idx'].astype(int).values
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['year'] = df_b['start_date'].str[:4]

    s011 = df_b[df_b['stk_yy']=='011'].copy()
    print(f'  stk=011 子集: n={len(s011):,}')

    # ===== mkt 阴阳跨年表 =====
    print(f'\n{"="*92}')
    print(f'  stk=011 + mkt 阴阳 各年密度 (r100%)')
    print(f'{"="*92}')

    states = ['000', '001', '010', '011', '100', '101', '110', '111']
    years = sorted(s011['year'].unique())

    print(f'\n  {"mkt":<7}', end='')
    for y in years: print(f' {y:>6}', end='')
    print(f' {"全":>5} {"赢年":>5}')

    # 全 stk=011 在每年的 baseline (作为对比)
    s011_year_base = {}
    for y in years:
        ys = s011[s011['year']==y]
        if len(ys) > 0:
            s011_year_base[y] = (ys['gain_pct']>=100).mean()*100
        else:
            s011_year_base[y] = 0

    # 全样本 baseline (作对比) -- 每年的全市场 baseline
    full_year_base = {}
    for y in years:
        yb = df_b[df_b['year']==y]
        if len(yb) > 0:
            full_year_base[y] = (yb['gain_pct']>=100).mean()*100
        else:
            full_year_base[y] = 0

    print(f'  {"baseline":<7}', end='')
    for y in years: print(f' {full_year_base[y]:>+5.1f}%', end='')
    print()
    print(f'  {"011 base":<7}', end='')
    for y in years: print(f' {s011_year_base[y]:>+5.1f}%', end='')
    print()
    print(f'  {"-"*92}')

    # 每个 mkt 状态
    rows_track = []
    for mv in states:
        sub_m = s011[s011['mkt_yy']==mv]
        if len(sub_m) < 200: continue
        n_total = len(sub_m)
        r100_total = (sub_m['gain_pct']>=100).mean()*100
        # 跨年
        won = 0; have = 0; year_rs = {}
        print(f'  mkt={mv}', end='')
        for y in years:
            ys = sub_m[sub_m['year']==y]
            if len(ys) < 30:
                print(f' {"--":>6}', end='')
                year_rs[y] = None
                continue
            r_y = (ys['gain_pct']>=100).mean()*100
            year_rs[y] = r_y
            base_y = full_year_base[y]
            if r_y > base_y * 1.3:
                won += 1
                print(f' {r_y:>+5.1f}%', end='')
            elif r_y < base_y * 0.7:
                print(f' {r_y:>+5.1f}%', end='')
            else:
                print(f' {r_y:>+5.1f}%', end='')
            have += 1
        print(f' {r100_total:>+4.1f}%  {won}/{have}')
        rows_track.append({
            'mkt': mv, 'n': n_total, 'r100': r100_total,
            'won': won, 'have': have,
            'win_rate': won/have*100 if have else 0,
        })

    # ===== 排序 =====
    print(f'\n{"="*82}')
    print(f'  排序: stk=011 + mkt 阴阳 跨年赢率 (r100 > full_baseline 1.3x 算赢)')
    print(f'{"="*82}')
    df_t = pd.DataFrame(rows_track).sort_values(['win_rate','r100'], ascending=[False, False])
    print(f'\n  {"mkt":<5} {"n":>6} {"r100":>7} {"赢年":>5} {"年":>4} {"赢率":>6}')
    for _, r in df_t.iterrows():
        print(f'  {r["mkt"]:<5} {r["n"]:>6,} {r["r100"]:>+6.2f}% {r["won"]:>4} {r["have"]:>4} '
              f'{r["win_rate"]:>+5.1f}%')

    # ===== 找"小年也赢" 的 (2017/2023) =====
    print(f'\n{"="*82}')
    print(f'  小年表现: 2017 / 2023 (横盘年, baseline ~1%)')
    print(f'{"="*82}')
    print(f'\n  全市场 2017 baseline: {full_year_base.get("2017", 0):.2f}%')
    print(f'  全市场 2023 baseline: {full_year_base.get("2023", 0):.2f}%')
    print(f'  全市场 2018 baseline: {full_year_base.get("2018", 0):.2f}%')
    print(f'  全市场 2022 baseline: {full_year_base.get("2022", 0):.2f}%')

    print(f'\n  --- 2017 / 2018 / 2022 / 2023 这 4 年同时 r100 > 全 baseline 的组合 ---')
    print(f'\n  {"mkt":<5} {"n":>5} {"全":>5} {"2017":>5} {"2018":>5} {"2022":>5} {"2023":>5}')
    for mv in states:
        sub_m = s011[s011['mkt_yy']==mv]
        if len(sub_m) < 200: continue
        rows = []
        for y in ['2017', '2018', '2022', '2023']:
            ys = sub_m[sub_m['year']==y]
            if len(ys) < 20:
                rows.append('--')
                continue
            r_y = (ys['gain_pct']>=100).mean()*100
            base_y = full_year_base[y]
            tag = '⭐' if r_y > base_y*1.3 else ('❌' if r_y < base_y*0.7 else '')
            rows.append(f'{r_y:>+4.1f}%{tag}')
        all_r = (sub_m['gain_pct']>=100).mean()*100
        print(f'  {mv:<5} {len(sub_m):>5} {all_r:>+4.1f}%  ', end='')
        for r in rows: print(f' {r:>7}', end='')
        print()

    # ===== 最稳的: 4 个小年都不输 =====
    print(f'\n{"="*82}')
    print(f'  最稳: stk=011 + 哪些 mkt 阴阳, 在 2017/2018/2022/2023 都不输 baseline')
    print(f'{"="*82}')
    safe_combos = []
    for mv in states:
        sub_m = s011[s011['mkt_yy']==mv]
        if len(sub_m) < 200: continue
        ok = True
        details = []
        for y in ['2017', '2018', '2022', '2023']:
            ys = sub_m[sub_m['year']==y]
            if len(ys) < 20:
                continue
            r_y = (ys['gain_pct']>=100).mean()*100
            base_y = full_year_base[y]
            details.append((y, r_y, base_y))
            if r_y < base_y * 0.9:  # 容忍 10%
                ok = False
                break
        if ok and len(details) >= 3:
            safe_combos.append((mv, len(sub_m), details))

    print(f'\n  共 {len(safe_combos)} 个稳定组合')
    for mv, n, details in safe_combos:
        print(f'\n  mkt={mv} (n={n}):')
        for y, r, base in details:
            print(f'    {y}: r100={r:+.2f}%, baseline={base:+.2f}%, ratio={r/base if base else 0:.2f}x')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
