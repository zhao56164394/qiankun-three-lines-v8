# -*- coding: utf-8 -*-
"""坤 regime 同日选股 — 事件因子大表 (v4 旧 E2 / v5 新 E2 双版本)

输入: 主板 + 个股多尺度卦 + 大盘多尺度卦
入场:
  v4 = E1 (retail<-250 池) + E2v4 (mf 上穿 50) + E3 (retail 上升)
  v5 = E1 (retail<-250 池) + E2v5 (mf 上升+trend>11) + E3 (retail 上升)
持仓: D6 卖, U1 买
终结: T0 (trend<11), 与当前资金回测一致

每个事件标注因子:
  - 池态: pool_min_retail / pool_days / pool_min_mf
  - 入场态: cur_retail / cur_mf / cur_trend
  - 5d 斜率: mf_5d / retail_5d / trend_5d
  - 30d 统计: mf_30d_min / mean, retail_30d_min / mean
  - 卦象: mkt_d / mkt_m / mkt_y, stk_m / stk_y
  - 标签: ret% (D6+U1+T0 完整一段)

输出: data_layer/data/results/kun_event_factors_{v4,v5}.parquet
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v4(arrays):
    """v4: mf 上穿 50"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = -np.inf; last_retail = np.nan
        pool_min_retail = np.inf; pool_min_mf = np.inf
        pool_enter_i = -1
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
                pool_min_retail = retail[gi]
                pool_min_mf = mf[gi] if not np.isnan(mf[gi]) else np.inf
                pool_enter_i = i
            if in_pool:
                if retail[gi] < pool_min_retail: pool_min_retail = retail[gi]
                if not np.isnan(mf[gi]) and mf[gi] < pool_min_mf: pool_min_mf = mf[gi]
            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            if in_pool and mf_cross_up and retail_rising:
                events.append({'date': date[gi], 'code': code[gi],
                               'buy_idx_global': gi,
                               'pool_min_retail': pool_min_retail,
                               'pool_min_mf': pool_min_mf,
                               'pool_days': i - pool_enter_i})
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def find_signals_v5(arrays):
    """v5: mf 上升 AND trend>11"""
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
        pool_min_retail = np.inf; pool_min_mf = np.inf
        pool_enter_i = -1
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True
                pool_min_retail = retail[gi]
                pool_min_mf = mf[gi] if not np.isnan(mf[gi]) else np.inf
                pool_enter_i = i
            if in_pool:
                if retail[gi] < pool_min_retail: pool_min_retail = retail[gi]
                if not np.isnan(mf[gi]) and mf[gi] < pool_min_mf: pool_min_mf = mf[gi]
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                events.append({'date': date[gi], 'code': code[gi],
                               'buy_idx_global': gi,
                               'pool_min_retail': pool_min_retail,
                               'pool_min_mf': pool_min_mf,
                               'pool_days': i - pool_enter_i})
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    """D6+U1 + T0 (trend<11 终结) — 与 v4/v5 当前一致"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs = 0

    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'td<11', (cum_mult-1)*100, legs

        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0

        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]
                holding = True

    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
    return max_end, 'fc', (cum_mult-1)*100, legs


def annotate_factors(df_e, arrays, mkt_arrays, stk_gua_arrays, close_arr, trend_arr):
    """对每个事件加因子 + ret%"""
    code_starts = arrays['starts']; code_ends = arrays['ends']
    mf = arrays['mf']; retail = arrays['retail']
    mkt_d = mkt_arrays['mkt_d']; mkt_m = mkt_arrays['mkt_m']; mkt_y = mkt_arrays['mkt_y']
    mkt_date_idx = mkt_arrays['date_idx']
    stk_m = stk_gua_arrays['stk_m']; stk_y = stk_gua_arrays['stk_y']
    date_arr = arrays['date']

    rows = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)

        # 事件态
        cur_retail = retail[gi]
        cur_mf = mf[gi]
        cur_trend = trend_arr[gi]

        # 5d 斜率 (gi-5..gi)
        i5 = max(gi - 5, code_starts[ci])
        mf_5d = mf[gi] - mf[i5] if not np.isnan(mf[i5]) else np.nan
        ret_5d = retail[gi] - retail[i5] if not np.isnan(retail[i5]) else np.nan
        td_5d = trend_arr[gi] - trend_arr[i5] if not np.isnan(trend_arr[i5]) else np.nan

        # 30d 统计
        i30 = max(gi - 30, code_starts[ci])
        mf_seg = mf[i30:gi+1]
        ret_seg = retail[i30:gi+1]
        mf_30d_min = np.nanmin(mf_seg) if len(mf_seg) else np.nan
        mf_30d_mean = np.nanmean(mf_seg) if len(mf_seg) else np.nan
        ret_30d_min = np.nanmin(ret_seg) if len(ret_seg) else np.nan
        ret_30d_mean = np.nanmean(ret_seg) if len(ret_seg) else np.nan

        # 卦
        dt = date_arr[gi]
        m_idx = mkt_date_idx.get(dt)
        if m_idx is not None:
            mkt_d_v = mkt_d[m_idx]; mkt_m_v = mkt_m[m_idx]; mkt_y_v = mkt_y[m_idx]
        else:
            mkt_d_v = mkt_m_v = mkt_y_v = ''
        stk_m_v = stk_m[gi]; stk_y_v = stk_y[gi]

        # ret%
        _, reason, ret_pct, legs = simulate_t0(gi, trend_arr, close_arr, mf, retail, max_end)

        rows.append({
            'date': dt, 'code': ev['code'],
            'pool_min_retail': ev['pool_min_retail'],
            'pool_min_mf': ev['pool_min_mf'],
            'pool_days': ev['pool_days'],
            'cur_retail': cur_retail, 'cur_mf': cur_mf, 'cur_trend': cur_trend,
            'mf_5d': mf_5d, 'ret_5d': ret_5d, 'td_5d': td_5d,
            'mf_30d_min': mf_30d_min, 'mf_30d_mean': mf_30d_mean,
            'ret_30d_min': ret_30d_min, 'ret_30d_mean': ret_30d_mean,
            'mkt_d': mkt_d_v, 'mkt_m': mkt_m_v, 'mkt_y': mkt_y_v,
            'stk_m': stk_m_v, 'stk_y': stk_y_v,
            'reason': reason, 'ret_pct': ret_pct, 'legs': legs,
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    print('=== test157: v4 / v5 事件因子表打表 ===\n')

    # 主板池
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    # 个股卦
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for col in ['d_gua', 'm_gua', 'y_gua']:
        g[col] = g[col].astype(str).str.zfill(3).replace({'nan': '', 'NaN': ''})

    # 大盘卦
    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    for col in ['d_gua', 'm_gua', 'y_gua']:
        mkt[col] = mkt[col].astype(str).str.zfill(3).replace({'nan': '', 'NaN': ''})
    mkt = mkt.drop_duplicates('date').rename(columns={'d_gua': 'mkt_d', 'm_gua': 'mkt_m', 'y_gua': 'mkt_y'})

    # 个股 mfretail
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    # 合并
    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'd_trend']).reset_index(drop=True)

    # 大盘 y=000 锁 (坤 regime)
    df = df.merge(mkt, on='date', how='left')
    df = df[df['mkt_y'] == '000'].reset_index(drop=True)
    print(f'  坤 regime 数据: {len(df):,} 行')

    # numpy 化
    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    stk_m_arr = df['m_gua'].to_numpy()
    stk_y_arr = df['y_gua'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr, 'date': date_arr,
        'retail': retail_arr, 'mf': mf_arr, 'td': trend_arr,
        'starts': code_starts, 'ends': code_ends,
    }
    # mkt 因为已 merge 进 df, 直接用 row idx 取
    # 但 annotate_factors 期望 mkt_date_idx, 这里改成按 row idx
    # 简化: 直接用 df 的列, 在 annotate 里用 gi 索引

    stk_gua_arrays = {'stk_m': stk_m_arr, 'stk_y': stk_y_arr}
    mkt_arrays = {'mkt_d_arr': mkt_d_arr, 'mkt_m_arr': mkt_m_arr, 'mkt_y_arr': mkt_y_arr}

    # ===== v4 =====
    print(f'\n--- v4 (mf 上穿 50) ---')
    df_e_v4 = find_signals_v4(arrays)
    print(f'  入场事件: {len(df_e_v4):,}')
    rows = []
    for _, ev in df_e_v4.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)

        cur_retail = retail_arr[gi]; cur_mf = mf_arr[gi]; cur_trend = trend_arr[gi]
        i5 = max(gi - 5, code_starts[ci])
        mf_5d = mf_arr[gi] - mf_arr[i5] if not np.isnan(mf_arr[i5]) else np.nan
        ret_5d = retail_arr[gi] - retail_arr[i5] if not np.isnan(retail_arr[i5]) else np.nan
        td_5d = trend_arr[gi] - trend_arr[i5] if not np.isnan(trend_arr[i5]) else np.nan
        i30 = max(gi - 30, code_starts[ci])
        mf_seg = mf_arr[i30:gi+1]; ret_seg = retail_arr[i30:gi+1]
        mf_30d_min = np.nanmin(mf_seg) if len(mf_seg) else np.nan
        mf_30d_mean = np.nanmean(mf_seg) if len(mf_seg) else np.nan
        ret_30d_min = np.nanmin(ret_seg) if len(ret_seg) else np.nan
        ret_30d_mean = np.nanmean(ret_seg) if len(ret_seg) else np.nan

        _, reason, ret_pct, legs = simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)

        rows.append({
            'date': date_arr[gi], 'code': ev['code'],
            'pool_min_retail': ev['pool_min_retail'],
            'pool_min_mf': ev['pool_min_mf'],
            'pool_days': ev['pool_days'],
            'cur_retail': cur_retail, 'cur_mf': cur_mf, 'cur_trend': cur_trend,
            'mf_5d': mf_5d, 'ret_5d': ret_5d, 'td_5d': td_5d,
            'mf_30d_min': mf_30d_min, 'mf_30d_mean': mf_30d_mean,
            'ret_30d_min': ret_30d_min, 'ret_30d_mean': ret_30d_mean,
            'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi], 'mkt_y': mkt_y_arr[gi],
            'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
            'reason': reason, 'ret_pct': ret_pct, 'legs': legs,
        })
    tab_v4 = pd.DataFrame(rows)
    print(f'  v4 表: {len(tab_v4):,} 行')

    # ===== v5 =====
    print(f'\n--- v5 (mf 上升+trend>11) ---')
    df_e_v5 = find_signals_v5(arrays)
    print(f'  入场事件: {len(df_e_v5):,}')
    rows = []
    for _, ev in df_e_v5.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)

        cur_retail = retail_arr[gi]; cur_mf = mf_arr[gi]; cur_trend = trend_arr[gi]
        i5 = max(gi - 5, code_starts[ci])
        mf_5d = mf_arr[gi] - mf_arr[i5] if not np.isnan(mf_arr[i5]) else np.nan
        ret_5d = retail_arr[gi] - retail_arr[i5] if not np.isnan(retail_arr[i5]) else np.nan
        td_5d = trend_arr[gi] - trend_arr[i5] if not np.isnan(trend_arr[i5]) else np.nan
        i30 = max(gi - 30, code_starts[ci])
        mf_seg = mf_arr[i30:gi+1]; ret_seg = retail_arr[i30:gi+1]
        mf_30d_min = np.nanmin(mf_seg) if len(mf_seg) else np.nan
        mf_30d_mean = np.nanmean(mf_seg) if len(mf_seg) else np.nan
        ret_30d_min = np.nanmin(ret_seg) if len(ret_seg) else np.nan
        ret_30d_mean = np.nanmean(ret_seg) if len(ret_seg) else np.nan

        _, reason, ret_pct, legs = simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)

        rows.append({
            'date': date_arr[gi], 'code': ev['code'],
            'pool_min_retail': ev['pool_min_retail'],
            'pool_min_mf': ev['pool_min_mf'],
            'pool_days': ev['pool_days'],
            'cur_retail': cur_retail, 'cur_mf': cur_mf, 'cur_trend': cur_trend,
            'mf_5d': mf_5d, 'ret_5d': ret_5d, 'td_5d': td_5d,
            'mf_30d_min': mf_30d_min, 'mf_30d_mean': mf_30d_mean,
            'ret_30d_min': ret_30d_min, 'ret_30d_mean': ret_30d_mean,
            'mkt_d': mkt_d_arr[gi], 'mkt_m': mkt_m_arr[gi], 'mkt_y': mkt_y_arr[gi],
            'stk_m': stk_m_arr[gi], 'stk_y': stk_y_arr[gi],
            'reason': reason, 'ret_pct': ret_pct, 'legs': legs,
        })
    tab_v5 = pd.DataFrame(rows)
    print(f'  v5 表: {len(tab_v5):,} 行')

    # ===== 总览 =====
    print(f'\n=== 总览 ===\n')
    for tag, t in [('v4', tab_v4), ('v5', tab_v5)]:
        print(f'  --- {tag} ---')
        print(f'    n={len(t):,}, avg_ret={t["ret_pct"].mean():+.2f}%, '
              f'win={(t["ret_pct"]>0).mean()*100:.1f}%, '
              f'med={t["ret_pct"].median():+.2f}%, '
              f'p25={t["ret_pct"].quantile(0.25):+.2f}%, '
              f'p75={t["ret_pct"].quantile(0.75):+.2f}%')
        print(f'    年覆盖: {t["date"].str[:4].nunique()} 年')
        print(f'    年信号: ', end='')
        for y, sub in t.groupby(t['date'].str[:4]):
            print(f'{y}={len(sub)} ', end='')
        print()

    # ===== 写出 =====
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    tab_v4.to_parquet(os.path.join(out_dir, 'kun_event_factors_v4.parquet'))
    tab_v5.to_parquet(os.path.join(out_dir, 'kun_event_factors_v5.parquet'))
    print(f'\n  写出 kun_event_factors_v4.parquet ({len(tab_v4):,} 行)')
    print(f'  写出 kun_event_factors_v5.parquet ({len(tab_v5):,} 行)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
