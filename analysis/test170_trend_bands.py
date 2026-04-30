# -*- coding: utf-8 -*-
"""trend 波段定义下的暴涨股统计

波段定义:
  起点 = trend 从 ≤11 上穿 11 的下一天 (trend > 11)
  终点 = trend 从 >11 下穿 11 的当天 (trend ≤ 11)
  一只股可以有多个波段

波段内涨幅:
  max_close = 波段内最高收盘价 (位置 idx_max)
  min_close_before_max = [start, idx_max] 区间内最低收盘价
  涨幅 = max_close / min_close_before_max - 1

暴涨股 = 涨幅 ≥ +100% 的波段

输出:
  - 波段总数 / 暴涨股波段数 / 占比
  - 每年波段数 / 暴涨股数 / 占比
  - 暴涨股代表 (按涨幅倒序前 30 只)
  - 波段长度分布 (持续天数)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_bands(arrays):
    """对每只股, 找所有 trend>11 波段"""
    cs = arrays['starts']; ce = arrays['ends']
    td = arrays['td']; close = arrays['close']
    date = arrays['date']; code = arrays['code']
    bands = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < 30: continue
        # 状态机: 找 trend ≤ 11 → trend > 11 切换点
        n = e - s
        in_band = False
        band_start = -1
        # trend 在 i=0 的状态
        if not np.isnan(td[s]) and td[s] > 11:
            in_band = True
            band_start = s
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur): continue
            if not in_band:
                # 找上穿
                if not np.isnan(prev) and prev <= 11 and cur > 11:
                    in_band = True
                    band_start = i
            else:
                # 找下穿
                if cur <= 11:
                    band_end = i  # 下穿当天作为终点
                    if band_end > band_start:
                        # 计算波段内涨幅
                        seg_close = close[band_start:band_end+1]
                        seg_close = seg_close[~np.isnan(seg_close)]
                        if len(seg_close) >= 2:
                            idx_max_local = int(np.argmax(seg_close))
                            max_close = seg_close[idx_max_local]
                            # idx_max 之前 (含 idx_max) 找 min
                            min_before = np.min(seg_close[:idx_max_local+1])
                            if min_before > 0:
                                gain = (max_close / min_before - 1) * 100
                            else:
                                gain = np.nan
                            bands.append({
                                'code': code[band_start],
                                'start_date': date[band_start],
                                'end_date': date[band_end],
                                'days': band_end - band_start + 1,
                                'min_close': float(min_before),
                                'max_close': float(max_close),
                                'gain_pct': gain,
                                'start_close': float(close[band_start]),
                                'end_close': float(close[band_end]),
                            })
                    in_band = False; band_start = -1
        # 处理段末仍在 band 的情况
        if in_band and band_start >= 0:
            band_end = e - 1
            if band_end > band_start:
                seg_close = close[band_start:band_end+1]
                seg_close = seg_close[~np.isnan(seg_close)]
                if len(seg_close) >= 2:
                    idx_max_local = int(np.argmax(seg_close))
                    max_close = seg_close[idx_max_local]
                    min_before = np.min(seg_close[:idx_max_local+1])
                    if min_before > 0:
                        gain = (max_close / min_before - 1) * 100
                    else:
                        gain = np.nan
                    bands.append({
                        'code': code[band_start],
                        'start_date': date[band_start],
                        'end_date': date[band_end],
                        'days': band_end - band_start + 1,
                        'min_close': float(min_before),
                        'max_close': float(max_close),
                        'gain_pct': gain,
                        'start_close': float(close[band_start]),
                        'end_close': float(close[band_end]),
                        'incomplete': True,
                    })
    return pd.DataFrame(bands)


def main():
    t0 = time.time()
    print('=== test170: trend 波段 暴涨股统计 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)
    print(f'  数据: {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'close':close_arr,'td':trend_arr,
              'starts':code_starts,'ends':code_ends}

    print('  提取波段...')
    df_b = find_bands(arrays)
    print(f'    总波段: {len(df_b):,}')

    # 总览
    n = len(df_b)
    n50 = (df_b['gain_pct']>=50).sum()
    n100 = (df_b['gain_pct']>=100).sum()
    n200 = (df_b['gain_pct']>=200).sum()
    n500 = (df_b['gain_pct']>=500).sum()
    print(f'\n{"="*82}')
    print(f'  全部 trend>11 波段统计')
    print(f'{"="*82}')
    print(f'  总波段: {n:,}')
    print(f'  ≥+50%:  {n50:>5} ({n50/n*100:.2f}%)')
    print(f'  ≥+100%: {n100:>5} ({n100/n*100:.2f}%)  ← 暴涨股波段')
    print(f'  ≥+200%: {n200:>5} ({n200/n*100:.2f}%)')
    print(f'  ≥+500%: {n500:>5} ({n500/n*100:.2f}%)')
    print(f'\n  涨幅五数: min={df_b["gain_pct"].min():.1f}%, '
          f'p25={df_b["gain_pct"].quantile(0.25):.1f}%, '
          f'med={df_b["gain_pct"].median():.1f}%, '
          f'p75={df_b["gain_pct"].quantile(0.75):.1f}%, '
          f'max={df_b["gain_pct"].max():.1f}%')

    # 波段持续天数
    print(f'\n  波段长度: min={df_b["days"].min()}, '
          f'p25={df_b["days"].quantile(0.25):.0f}, '
          f'med={df_b["days"].median():.0f}, '
          f'p75={df_b["days"].quantile(0.75):.0f}, '
          f'max={df_b["days"].max()}')

    # 按起点年
    df_b['start_year'] = df_b['start_date'].str[:4]
    print(f'\n{"="*82}')
    print(f'  按波段起点年 (≥+100% 暴涨股波段)')
    print(f'{"="*82}')
    print(f'\n  {"年":<6} {"波段":>6} {"≥50%":>5} {"≥100%":>6} {"r100":>7} '
          f'{"≥200%":>6} {"r200":>7} {"≥500%":>6}')
    for y, sub in df_b.groupby('start_year'):
        n_ = len(sub)
        h50 = (sub['gain_pct']>=50).sum()
        h100 = (sub['gain_pct']>=100).sum()
        h200 = (sub['gain_pct']>=200).sum()
        h500 = (sub['gain_pct']>=500).sum()
        print(f'  {y:<6} {n_:>6} {h50:>5} {h100:>6} {h100/n_*100:>+6.2f}% '
              f'{h200:>6} {h200/n_*100:>+6.2f}% {h500:>6}')

    # 暴涨股波段的长度
    big = df_b[df_b['gain_pct']>=100]
    print(f'\n{"="*82}')
    print(f'  暴涨股波段 (≥+100%) 长度分布')
    print(f'{"="*82}')
    print(f'  min={big["days"].min()}, '
          f'p25={big["days"].quantile(0.25):.0f}, '
          f'med={big["days"].median():.0f}, '
          f'p75={big["days"].quantile(0.75):.0f}, '
          f'p90={big["days"].quantile(0.90):.0f}, '
          f'max={big["days"].max()}')

    # 按持续天数分桶
    bins = [0, 30, 60, 100, 150, 250, 1000]
    labels = ['<30d', '30-60d', '60-100d', '100-150d', '150-250d', '>250d']
    big = big.copy()
    big['day_bin'] = pd.cut(big['days'], bins, labels=labels)
    df_b_bd = df_b.copy()
    df_b_bd['day_bin'] = pd.cut(df_b_bd['days'], bins, labels=labels)
    print(f'\n  {"持续":<10} {"全波段":>7} {"暴涨股":>6} {"密度":>6}')
    for label in labels:
        all_n = (df_b_bd['day_bin']==label).sum()
        big_n = (big['day_bin']==label).sum()
        rate = big_n/all_n*100 if all_n else 0
        print(f'  {label:<10} {all_n:>7} {big_n:>6} {rate:>5.2f}%')

    # 暴涨股 top 30
    print(f'\n{"="*82}')
    print(f'  ≥+100% 暴涨股波段 — 涨幅 top 30')
    print(f'{"="*82}')
    big_sorted = big.sort_values('gain_pct', ascending=False)
    print(f'\n  {"代码":<8} {"起点日":<12} {"终点日":<12} {"持续":>5} '
          f'{"min_C":>7} {"max_C":>7} {"涨幅":>9}')
    for _, r in big_sorted.head(30).iterrows():
        print(f'  {r["code"]:<8} {r["start_date"]:<12} {r["end_date"]:<12} '
              f'{r["days"]:>4}d {r["min_close"]:>6.2f} {r["max_close"]:>6.2f} '
              f'{r["gain_pct"]:>+8.1f}%')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_b.to_parquet(os.path.join(out_dir, 'trend_bands_all.parquet'))
    big.to_parquet(os.path.join(out_dir, 'trend_bands_baggers.parquet'))
    print(f'\n  写出 trend_bands_all.parquet ({len(df_b):,} 行)')
    print(f'  写出 trend_bands_baggers.parquet ({len(big):,} 行)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
