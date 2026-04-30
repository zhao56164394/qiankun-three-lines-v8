# -*- coding: utf-8 -*-
"""波段起点的 大盘阴阳 × 个股阴阳 64 组合 暴涨股密度

阴阳定义:
  trend > 50 = 1 (阳), ≤50 = 0 (阴)

大盘 3 位: (mkt_d_trend, mkt_m_trend, mkt_y_trend) → 000-111 共 8 状态
个股 3 位: (stk_d_trend, stk_m_trend, stk_y_trend) → 000-111 共 8 状态

64 组合, 每组算:
  - 总波段数
  - ≥+100% 暴涨股数 / 比例
  - ≥+200% 暴涨股数 / 比例
  - 暴涨股密度 lift
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


def yyy(d, m, y, thr=50):
    """3 trend → 'XYZ' (阴阳串, 第一位 d, 第二位 m, 第三位 y)"""
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def main():
    t0 = time.time()
    print('=== test172: 起点 大盘阴阳×个股阴阳 64 组合 ===\n')

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
        'd_trend':'mkt_d_t', 'm_trend':'mkt_m_t', 'y_trend':'mkt_y_t'
    })

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)
    print(f'  数据: {len(df):,} 行')

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

    print('  提取波段...')
    df_b = find_bands(arrays)
    print(f'    总波段: {len(df_b):,}')

    # 起点的阴阳
    si = df_b['start_idx'].astype(int).values
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]

    n_total = len(df_b)
    n100 = (df_b['gain_pct']>=100).sum()
    n200 = (df_b['gain_pct']>=200).sum()
    base_r100 = n100/n_total*100
    base_r200 = n200/n_total*100
    print(f'  Baseline: r100={base_r100:.2f}%, r200={base_r200:.2f}%')

    # ===== 大盘单独阴阳 =====
    print(f'\n{"="*82}')
    print(f'  大盘阴阳 (mkt_d_t, mkt_m_t, mkt_y_t > 50) 单独')
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

    # ===== 个股单独阴阳 =====
    print(f'\n{"="*82}')
    print(f'  个股阴阳 (stk_d_t, stk_m_t, stk_y_t > 50) 单独')
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

    # ===== 64 组合 =====
    print(f'\n{"="*82}')
    print(f'  大盘 × 个股 64 组合 (按 lift_200 排序)')
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

    print(f'\n  共 {len(df_c)} 组 (n>=100)')
    print(f'\n  --- top 20 (按 L200) ---')
    print(f'  {"mkt":<5} {"stk":<5} {"n":>6} {"≥100":>5} {"r100":>7} {"L100":>7} '
          f'{"≥200":>5} {"r200":>7} {"L200":>7}')
    for _, r in df_c.head(20).iterrows():
        print(f'  {r["mkt"]:<5} {r["stk"]:<5} {r["n"]:>6,} {r["h100"]:>5} '
              f'{r["r100"]:>+6.2f}% {r["lift_100"]:>+5.2f}x  '
              f'{r["h200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    print(f'\n  --- bottom 10 ---')
    print(f'  {"mkt":<5} {"stk":<5} {"n":>6} {"≥100":>5} {"r100":>7} {"L100":>7} '
          f'{"≥200":>5} {"r200":>7} {"L200":>7}')
    for _, r in df_c.tail(10).iterrows():
        print(f'  {r["mkt"]:<5} {r["stk"]:<5} {r["n"]:>6,} {r["h100"]:>5} '
              f'{r["r100"]:>+6.2f}% {r["lift_100"]:>+5.2f}x  '
              f'{r["h200"]:>5} {r["r200"]:>+6.2f}% {r["lift_200"]:>+5.2f}x')

    # ===== 8x8 矩阵展示 =====
    print(f'\n{"="*82}')
    print(f'  8×8 矩阵: r100% (行=mkt, 列=stk)')
    print(f'{"="*82}')
    print(f'\n        ', end='')
    states = ['000', '001', '010', '011', '100', '101', '110', '111']
    for s in states: print(f' {s:>7}', end='')
    print(f' {"行汇总":>9}')
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
            print(f' {r100_m:>+6.2f}% n={len(m_sub):>5,}')
        else:
            print()

    # 列汇总 (个股)
    print(f'  --- 列汇总 (stk) ---')
    print(f'        ', end='')
    for sv in states:
        s_sub = df_b[df_b['stk_yy']==sv]
        if len(s_sub) > 0:
            r100_s = (s_sub['gain_pct']>=100).mean()*100
            print(f' {r100_s:>+5.2f}%', end='')
        else:
            print(f' {"--":>7}', end='')
    print()

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
