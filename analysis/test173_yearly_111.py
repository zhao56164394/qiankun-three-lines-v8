# -*- coding: utf-8 -*-
"""stk=111+mkt=111 跨年稳定性

每年统计:
  - 该年总波段数
  - mkt=111 波段数 / r100
  - stk=111 波段数 / r100
  - mkt=111+stk=111 波段数 / r100
  - mkt=111+stk=111 波段中, ≥+200% 笔数

要看 14.71x lift 是不是只来自 2014-2015 大牛市
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
        in_band = False; band_start = -1
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
    a = '1' if (not np.isnan(d) and d > thr) else '0'
    b = '1' if (not np.isnan(m) and m > thr) else '0'
    c = '1' if (not np.isnan(y) and y > thr) else '0'
    return a + b + c


def main():
    t0 = time.time()
    print('=== test173: stk=111+mkt=111 跨年稳定性 ===\n')

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

    print('  提取波段...')
    df_b = find_bands(arrays)
    print(f'    总波段: {len(df_b):,}')

    si = df_b['start_idx'].astype(int).values
    df_b['mkt_yy'] = [yyy(mkt_d_t[i], mkt_m_t[i], mkt_y_t[i]) for i in si]
    df_b['stk_yy'] = [yyy(stk_d_t[i], stk_m_t[i], stk_y_t[i]) for i in si]
    df_b['year'] = df_b['start_date'].str[:4]

    # ===== 跨年总览 =====
    print(f'\n{"="*100}')
    print(f'  按起点年: 全 / mkt=111 / stk=111 / mkt+stk=111 暴涨股密度')
    print(f'{"="*100}')

    print(f'\n  {"年":<6} | {"全部":>20} | {"mkt=111":>22} | {"stk=111":>22} | {"双 111":>22}')
    print(f'  {"":<6} | {"n":>5} {"r100":>7} {"r200":>6} | {"n":>4} {"r100":>7} {"r200":>6} | '
          f'{"n":>4} {"r100":>7} {"r200":>6} | {"n":>4} {"r100":>7} {"r200":>6}')
    print(f'  {"-"*6} | {"-"*20} | {"-"*22} | {"-"*22} | {"-"*22}')

    for y in sorted(df_b['year'].unique()):
        sub = df_b[df_b['year']==y]
        n = len(sub)
        if n < 100: continue
        r100 = (sub['gain_pct']>=100).mean()*100
        r200 = (sub['gain_pct']>=200).mean()*100

        m111 = sub[sub['mkt_yy']=='111']
        if len(m111) == 0:
            m_str = f' {"--":>4} {"--":>7} {"--":>6}'
        else:
            m_str = f' {len(m111):>4} {(m111["gain_pct"]>=100).mean()*100:>+6.2f}% {(m111["gain_pct"]>=200).mean()*100:>+5.2f}%'

        s111 = sub[sub['stk_yy']=='111']
        if len(s111) == 0:
            s_str = f' {"--":>4} {"--":>7} {"--":>6}'
        else:
            s_str = f' {len(s111):>4} {(s111["gain_pct"]>=100).mean()*100:>+6.2f}% {(s111["gain_pct"]>=200).mean()*100:>+5.2f}%'

        d111 = sub[(sub['mkt_yy']=='111') & (sub['stk_yy']=='111')]
        if len(d111) == 0:
            d_str = f' {"--":>4} {"--":>7} {"--":>6}'
        else:
            d_str = f' {len(d111):>4} {(d111["gain_pct"]>=100).mean()*100:>+6.2f}% {(d111["gain_pct"]>=200).mean()*100:>+5.2f}%'

        print(f'  {y:<6} | {n:>5} {r100:>+6.2f}% {r200:>+5.2f}% |{m_str} |{s_str} |{d_str}')

    # ===== 双 111 暴涨股清单 =====
    print(f'\n{"="*100}')
    print(f'  双 111 (mkt+stk=111) 暴涨股波段 (≥+100%) 完整清单')
    print(f'{"="*100}')
    d111 = df_b[(df_b['mkt_yy']=='111') & (df_b['stk_yy']=='111')]
    big = d111[d111['gain_pct']>=100].sort_values('gain_pct', ascending=False)
    print(f'\n  双 111 总波段: {len(d111)}')
    print(f'  双 111 ≥+100%: {len(big)} ({len(big)/len(d111)*100:.2f}%)')
    print(f'  双 111 ≥+200%: {(d111["gain_pct"]>=200).sum()} ({(d111["gain_pct"]>=200).mean()*100:.2f}%)')

    print(f'\n  按年份分布:')
    for y, sub in big.groupby('year'):
        print(f'    {y}: {len(sub)} 只')

    # ===== mkt=111+stk=000 (起点 trend 刚启动, 大盘强) =====
    print(f'\n{"="*100}')
    print(f'  对比: mkt=111+stk=000 (个股 trend 刚 >11, 月年线还在 <50) 跨年')
    print(f'{"="*100}')
    print(f'\n  这是更普遍的"大盘强 + 个股启动初"组合, 看跨年是否稳')
    m111s000 = df_b[(df_b['mkt_yy']=='111') & (df_b['stk_yy']=='000')]
    print(f'\n  样本总数: {len(m111s000)}')
    print(f'  按年:')
    print(f'  {"年":<6} {"n":>5} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7}')
    for y, sub in m111s000.groupby('year'):
        if len(sub) < 50: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        print(f'  {y:<6} {n:>5} {h100:>5} {h100/n*100:>+6.2f}% {h200:>5} {h200/n*100:>+6.2f}%')

    # ===== mkt=111 跨年 (单条件) =====
    print(f'\n{"="*100}')
    print(f'  mkt=111 (单大盘条件) 跨年稳定性')
    print(f'{"="*100}')
    m111 = df_b[df_b['mkt_yy']=='111']
    print(f'\n  {"年":<6} {"n":>5} {"≥100":>5} {"r100":>7} {"≥200":>5} {"r200":>7}')
    for y, sub in m111.groupby('year'):
        if len(sub) < 50: continue
        n = len(sub); h100 = (sub['gain_pct']>=100).sum(); h200 = (sub['gain_pct']>=200).sum()
        print(f'  {y:<6} {n:>5} {h100:>5} {h100/n*100:>+6.2f}% {h200:>5} {h200/n*100:>+6.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
