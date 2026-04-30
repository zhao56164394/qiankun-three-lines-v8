# -*- coding: utf-8 -*-
"""波段 v2 + 排除 2014-2015 小盘股牛市

只用 2016-01-01 起的波段
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
    print('=== test176: 波段 v2 + 仅 2016+ ===\n')

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
    print(f'  全数据: {len(df):,} 行')

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

    print('  提取所有波段...')
    df_b_all = find_bands(arrays)
    print(f'    全部: {len(df_b_all):,}')

    # 起点日期 ≥ 2016-01-01
    df_b = df_b_all[df_b_all['start_date'] >= '2016-01-01'].copy()
    print(f'    2016+: {len(df_b):,}')

    si = df_b['start_idx'].astype(int).values
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['year'] = df_b['start_date'].str[:4]

    n_total = len(df_b); n100 = (df_b['gain_pct']>=100).sum(); n200 = (df_b['gain_pct']>=200).sum()
    base_r100 = n100/n_total*100; base_r200 = n200/n_total*100

    print(f'\n{"="*82}')
    print(f'  波段总览 (仅 2016+)')
    print(f'{"="*82}')
    print(f'  总: {n_total:,}')
    print(f'  ≥+100%: {n100:>6} ({base_r100:.2f}%)')
    print(f'  ≥+200%: {n200:>6} ({base_r200:.2f}%)')
    print(f'  涨幅 五数: min={df_b["gain_pct"].min():.1f}%, '
          f'p25={df_b["gain_pct"].quantile(0.25):.1f}, '
          f'med={df_b["gain_pct"].median():.1f}, '
          f'p75={df_b["gain_pct"].quantile(0.75):.1f}, '
          f'max={df_b["gain_pct"].max():.1f}%')

    # 大盘 单独
    print(f'\n{"="*82}')
    print(f'  大盘阴阳 单独')
    print(f'{"="*82}')
    print(f'\n  {"mkt":<6} {"n":>7} {"≥100":>5} {"r100":>7} {"L100":>7} '
          f'{"≥200":>5} {"r200":>7} {"L200":>7}')
    for v, sub in df_b.groupby('mkt_yy'):
        if len(sub) < 200: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        r100 = h100/n*100; r200 = h200/n*100
        l100 = r100/base_r100; l200 = r200/base_r200 if base_r200 else 0
        print(f'  {v:<6} {n:>7,} {h100:>5} {r100:>+6.2f}% {l100:>+5.2f}x  '
              f'{h200:>5} {r200:>+6.2f}% {l200:>+5.2f}x')

    # 个股 单独
    print(f'\n{"="*82}')
    print(f'  个股阴阳 单独')
    print(f'{"="*82}')
    print(f'\n  {"stk":<6} {"n":>7} {"≥100":>5} {"r100":>7} {"L100":>7} '
          f'{"≥200":>5} {"r200":>7} {"L200":>7}')
    for v, sub in df_b.groupby('stk_yy'):
        if len(sub) < 200: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        r100 = h100/n*100; r200 = h200/n*100
        l100 = r100/base_r100; l200 = r200/base_r200 if base_r200 else 0
        print(f'  {v:<6} {n:>7,} {h100:>5} {r100:>+6.2f}% {l100:>+5.2f}x  '
              f'{h200:>5} {r200:>+6.2f}% {l200:>+5.2f}x')

    # 64 组合
    print(f'\n{"="*82}')
    print(f'  大盘 × 个股 64 组合 (按 lift_200 排, n>=100)')
    print(f'{"="*82}')

    rows = []
    for (mv, sv), sub in df_b.groupby(['mkt_yy', 'stk_yy']):
        if len(sub) < 100: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        rows.append({
            'mkt': mv, 'stk': sv, 'n': n,
            'h100': h100, 'h200': h200,
            'r100': h100/n*100, 'r200': h200/n*100,
            'lift_100': (h100/n*100)/base_r100,
            'lift_200': (h200/n*100)/base_r200 if base_r200 else 0,
        })
    df_c = pd.DataFrame(rows).sort_values('lift_200', ascending=False)

    print(f'\n  共 {len(df_c)} 组')
    print(f'\n  {"mkt":<5} {"stk":<5} {"n":>6} {"≥100":>5} {"r100":>7} {"L100":>7} '
          f'{"≥200":>5} {"r200":>7} {"L200":>7}')
    for _, r in df_c.iterrows():
        print(f'  {r["mkt"]:<5} {r["stk"]:<5} {r["n"]:>6,} {r["h100"]:>5} '
              f'{r["r100"]:>+6.2f}% {r["lift_100"]:>+5.2f}x  '
              f'{r["h200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    # 8x8 矩阵
    print(f'\n{"="*82}')
    print(f'  8×8 矩阵: r100% (mkt × stk)')
    print(f'{"="*82}')
    print(f'\n        ', end='')
    states = ['000', '001', '010', '011', '100', '101', '110', '111']
    for s in states: print(f' {s:>7}', end='')
    print(f' {"行汇总":>16}')
    for mv in states:
        print(f'  mkt={mv}', end='')
        m_sub = df_b[df_b['mkt_yy']==mv]
        for sv in states:
            sub = m_sub[m_sub['stk_yy']==sv]
            if len(sub) < 50:
                print(f' {"--":>7}', end='')
            else:
                r100 = (sub['gain_pct']>=100).mean()*100
                print(f' {r100:>+5.2f}%', end='')
        if len(m_sub) > 0:
            r100_m = (m_sub['gain_pct']>=100).mean()*100
            print(f'   n={len(m_sub):>5,} {r100_m:>+5.2f}%')
        else:
            print()

    # Top 5 跨年
    print(f'\n{"="*82}')
    print(f'  Top 5 组合 跨年稳定性')
    print(f'{"="*82}')
    top5 = df_c.head(5).to_dict('records')
    for cfg in top5:
        mv, sv = cfg['mkt'], cfg['stk']
        sub_all = df_b[(df_b['mkt_yy']==mv) & (df_b['stk_yy']==sv)]
        print(f'\n  mkt={mv} stk={sv} (n={cfg["n"]}, r100={cfg["r100"]:.2f}%, L200={cfg["lift_200"]:.2f}x):')
        print(f'    {"年":<6} {"n":>5} {"r100":>7} {"r200":>7}')
        for y in sorted(sub_all['year'].unique()):
            ys = sub_all[sub_all['year']==y]
            if len(ys) < 30: continue
            r1 = (ys['gain_pct']>=100).mean()*100
            r2 = (ys['gain_pct']>=200).mean()*100
            tag = '⭐' if r1 > base_r100*1.3 else ('❌' if r1 < base_r100*0.7 else '')
            print(f'    {y:<6} {len(ys):>5} {r1:>+6.2f}% {r2:>+6.2f}%  {tag}')

    # 11 卦阴阳的稳定性 — 找跨年都比 baseline 强的
    print(f'\n{"="*82}')
    print(f'  跨年稳定排序 (每组合算 "胜年数 / 总年数"): 高于 baseline 1.3x 算赢)')
    print(f'{"="*82}')
    rows2 = []
    for (mv, sv), sub_all in df_b.groupby(['mkt_yy', 'stk_yy']):
        if len(sub_all) < 200: continue
        years_with_data = []
        years_won = 0
        years_lost = 0
        for y, ys in sub_all.groupby('year'):
            if len(ys) < 30: continue
            yb = df_b[df_b['year']==y]
            base_y = (yb['gain_pct']>=100).mean()*100
            r_y = (ys['gain_pct']>=100).mean()*100
            years_with_data.append((y, r_y, base_y))
            if r_y > base_y * 1.3:
                years_won += 1
            elif r_y < base_y * 0.7:
                years_lost += 1
        if len(years_with_data) >= 5:
            n = len(sub_all)
            r100 = (sub_all['gain_pct']>=100).mean()*100
            rows2.append({
                'mkt': mv, 'stk': sv, 'n': n,
                'r100': r100,
                'years': len(years_with_data),
                'won': years_won,
                'lost': years_lost,
                'win_rate': years_won/len(years_with_data)*100,
            })
    df_y = pd.DataFrame(rows2).sort_values(['win_rate', 'r100'], ascending=[False, False])
    print(f'\n  {"mkt":<5} {"stk":<5} {"n":>6} {"r100":>7} {"年":>4} {"赢":>4} {"输":>4} {"赢率":>6}')
    for _, r in df_y.head(15).iterrows():
        print(f'  {r["mkt"]:<5} {r["stk"]:<5} {r["n"]:>6,} {r["r100"]:>+6.2f}% '
              f'{r["years"]:>4} {r["won"]:>4} {r["lost"]:>4} {r["win_rate"]:>+5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
