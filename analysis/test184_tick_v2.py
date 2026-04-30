# -*- coding: utf-8 -*-
"""002068 tick 实时三线 v2 — 用 window=12 重算 trend (跟数据库一致)

简化:
  - trend / mf / retail 都用纯日 close 重算 (不用数据库的 d_trend, 因为数据库 d_trend 用 W/M 周期)
  - 但 strategy/indicator.calc_trend_line 的 period=55, 改成纯 close 函数 + window=12
  - mf / retail 用 strategy/indicator 公式 (跟数据库一致)
"""
import os, sys, io, time, zipfile
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_ROOT = r'E:/BaiduSyncdisk/A股数据_分笔数据/分笔成交_按月归档_沪深'

sys.path.insert(0, ROOT)
from strategy.indicator import _tdx_sma, _tdx_ema, calc_main_force_line, calc_retail_line


def calc_trend_line_w12(closes, highs, lows, window=12):
    """跟数据库一致, window=12"""
    C = np.array(closes, dtype=float)
    H = np.array(highs, dtype=float)
    L = np.array(lows, dtype=float)
    llv = pd.Series(L).rolling(window, min_periods=1).min().values
    hhv = pd.Series(H).rolling(window, min_periods=1).max().values
    denom = hhv - llv
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.where(denom > 0, (C - llv) / denom * 100, 50.0)
    sma1 = _tdx_sma(X, 5, 1)
    sma2 = _tdx_sma(sma1, 3, 1)
    V11 = 3 * sma1 - 2 * sma2
    trend = _tdx_ema(V11, 3)
    return trend


def load_tick(date_str, code):
    yyyymm = f'{date_str[:4]}-{date_str[4:6]}'
    zp_path = os.path.join(TICK_ROOT, yyyymm, f'{date_str}.zip')
    if not os.path.exists(zp_path): return None
    try:
        zp = zipfile.ZipFile(zp_path)
        if f'{code}.csv' not in zp.namelist(): return None
        with zp.open(f'{code}.csv') as f:
            raw = f.read()
        if raw[:3] == b'\xef\xbb\xbf': raw = raw[3:]
        text = raw.decode('utf-8', errors='replace')
        df = pd.read_csv(io.StringIO(text))
        df.columns = ['time', 'price', 'vol', 'dir']
        return df
    except: return None


def fmt_num(v, w=5, prec=1):
    if isinstance(v, str): return f'{v:>{w}}'
    if v is None or (isinstance(v,float) and np.isnan(v)): return f'{"--":>{w}}'
    return f'{v:>+{w}.{prec}f}'


def main():
    t0 = time.time()
    print('=== test184: 002068 tick 实时三线 v2 (window=12) ===\n')

    code = '002068'
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'high', 'low', 'close'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p['date'] = p['date'].astype(str)
    df = p[(p['code']==code) & (p['date']>='2021-01-01')].sort_values('date').reset_index(drop=True)
    print(f'  002068 日线: {len(df)} 行 ({df["date"].min()} ~ {df["date"].max()})')

    closes = df['close'].to_numpy().astype(np.float64)
    highs = df['high'].to_numpy().astype(np.float64)
    lows = df['low'].to_numpy().astype(np.float64)
    trend_arr = calc_trend_line_w12(closes, highs, lows)
    mf_arr = calc_main_force_line(closes)
    retail_arr = calc_retail_line(closes)

    # 跟数据库 d_trend 对比 sanity
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = g['date'].astype(str)
    g_002068 = g[g['code']==code].sort_values('date')
    df = df.merge(g_002068[['date','d_trend']], on='date', how='left')
    db_trend = df['d_trend'].to_numpy()
    diffs = []
    for i in range(len(df)):
        if not np.isnan(db_trend[i]) and not np.isnan(trend_arr[i]):
            diffs.append(abs(db_trend[i] - trend_arr[i]))
    if diffs:
        print(f'  trend w=12 重算 vs 数据库 diff: max={max(diffs):.2f}, avg={np.mean(diffs):.4f}\n')

    # 找 entry 和 测试段
    df_idx_2022_04_20 = df[df['date']=='2022-04-20'].index[0]
    df_idx_2022_09_05 = df[df['date']=='2022-09-05'].index[0]
    test_dates = df['date'].iloc[df_idx_2022_04_20:df_idx_2022_09_05+1].tolist()
    print(f'  测试段: {test_dates[0]} ~ {test_dates[-1]} 共 {len(test_dates)} 个交易日\n')

    # ===== 日 K 版 D6/U1 =====
    print(f'--- 日 K 版 D6/U1 流水 ---\n')
    holding = True
    cur_buy_price = closes[df_idx_2022_04_20]
    cum_mult = 1.0
    print(f'  {"日期":<12} {"动作":<10} {"价":>6} {"mf":>6} {"ret":>6} {"td":>5} {"本腿":>7}')
    print(f'  {test_dates[0]:<12} {"BUY":<10} {cur_buy_price:>5.2f} '
          f'{fmt_num(mf_arr[df_idx_2022_04_20])} {fmt_num(retail_arr[df_idx_2022_04_20])} '
          f'{fmt_num(trend_arr[df_idx_2022_04_20])}')

    actions_daily = []
    for k in range(df_idx_2022_04_20+1, df_idx_2022_09_05+1):
        td_v = trend_arr[k]; td_p = trend_arr[k-1]
        mf_v = mf_arr[k]; mf_p = mf_arr[k-1]
        rt_v = retail_arr[k]; rt_p = retail_arr[k-1]
        if np.isnan(td_v) or np.isnan(td_p): continue
        d_str = df['date'].iloc[k]

        if td_v < 11:
            if holding:
                cum_mult *= closes[k] / cur_buy_price
                leg = (closes[k]/cur_buy_price-1)*100
                print(f'  {d_str:<12} {"SELL T0":<10} {closes[k]:>5.2f} '
                      f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)} {leg:>+6.1f}%')
                actions_daily.append('SELL T0')
            break
        if np.isnan(mf_v) or np.isnan(mf_p) or np.isnan(rt_v) or np.isnan(rt_p): continue
        mfc = mf_v - mf_p; rc = rt_v - rt_p; tc = td_v - td_p
        if holding and mfc<0 and rc<0 and tc<0:
            cum_mult *= closes[k] / cur_buy_price
            leg = (closes[k]/cur_buy_price-1)*100
            print(f'  {d_str:<12} {"SELL D6":<10} {closes[k]:>5.2f} '
                  f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)} {leg:>+6.1f}%')
            holding = False
            actions_daily.append('SELL D6')
        elif (not holding) and mfc>0:
            cur_buy_price = closes[k]
            print(f'  {d_str:<12} {"BUY U1":<10} {closes[k]:>5.2f} '
                  f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)}')
            holding = True
            actions_daily.append('BUY U1')

    if holding:
        cum_mult *= closes[df_idx_2022_09_05] / cur_buy_price
        leg = (closes[df_idx_2022_09_05]/cur_buy_price-1)*100
        print(f'  (FC end) {test_dates[-1]:<12} {closes[df_idx_2022_09_05]:>5.2f} {leg:>+6.1f}%')

    print(f'\n  日 K 版: {len(actions_daily)+1} 动作, 累计 {cum_mult:.3f} ({(cum_mult-1)*100:+.1f}%)')

    # ===== Tick 版 =====
    print(f'\n\n--- Tick 版 D6/U1 流水 (window=12 trend) ---\n')
    holding = True
    cur_buy_price = closes[df_idx_2022_04_20]
    cum_mult = 1.0
    prev_mf = mf_arr[df_idx_2022_04_20]
    prev_ret = retail_arr[df_idx_2022_04_20]
    prev_td = trend_arr[df_idx_2022_04_20]

    print(f'  {"时间":<22} {"动作":<10} {"价":>6} {"mf":>6} {"ret":>6} {"td":>5} {"本腿":>7}')
    print(f'  {test_dates[0]+" close":<22} {"BUY":<10} {cur_buy_price:>5.2f} '
          f'{fmt_num(prev_mf)} {fmt_num(prev_ret)} {fmt_num(prev_td)}')

    actions_tick = []
    skip = 0
    terminated = False

    for ti in range(df_idx_2022_04_20+1, df_idx_2022_09_05+1):
        if terminated: break
        d_str = df['date'].iloc[ti]
        d_yyyymmdd = d_str.replace('-','')

        ticks = load_tick(d_yyyymmdd, code)
        if ticks is None or len(ticks) < 5:
            skip += 1
            # fallback 日 K
            mf_v = mf_arr[ti]; rt_v = retail_arr[ti]; td_v = trend_arr[ti]
            if not (np.isnan(mf_v) or np.isnan(rt_v) or np.isnan(td_v)):
                if td_v < 11:
                    if holding:
                        cum_mult *= closes[ti] / cur_buy_price
                        leg = (closes[ti]/cur_buy_price-1)*100
                        print(f'  {d_str+" close (no tick)":<22} {"SELL T0":<10} {closes[ti]:>5.2f} '
                              f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)} {leg:>+6.1f}%')
                        terminated = True
                else:
                    mfc = mf_v - prev_mf; rc = rt_v - prev_ret; tc = td_v - prev_td
                    if holding and mfc<0 and rc<0 and tc<0:
                        cum_mult *= closes[ti] / cur_buy_price
                        leg = (closes[ti]/cur_buy_price-1)*100
                        print(f'  {d_str+" close (no tick)":<22} {"SELL D6":<10} {closes[ti]:>5.2f} '
                              f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)} {leg:>+6.1f}%')
                        holding = False
                    elif (not holding) and mfc>0:
                        cur_buy_price = closes[ti]
                        print(f'  {d_str+" close (no tick)":<22} {"BUY U1":<10} {closes[ti]:>5.2f} '
                              f'{fmt_num(mf_v)} {fmt_num(rt_v)} {fmt_num(td_v)}')
                        holding = True
                prev_mf = mf_v; prev_ret = rt_v; prev_td = td_v
            continue

        # tick 数据 — 每分钟取最后一笔
        ticks['time'] = ticks['time'].astype(str)
        ticks['min'] = ticks['time'].str[:16]
        ticks_min = ticks.groupby('min').last().reset_index()

        hist_C = closes[:ti]
        hist_H = highs[:ti]
        hist_L = lows[:ti]

        day_high = -np.inf
        day_low = np.inf
        action_today = None  # 当天最多一次操作

        for _, t in ticks_min.iterrows():
            t_price = t['price']; t_time = t['time']
            day_high = max(day_high, t_price)
            day_low = min(day_low, t_price)

            full_C = np.r_[hist_C, t_price]
            full_H = np.r_[hist_H, day_high]
            full_L = np.r_[hist_L, day_low]
            td_now = calc_trend_line_w12(full_C, full_H, full_L)[-1]
            mf_now = calc_main_force_line(full_C)[-1]
            ret_now = calc_retail_line(full_C)[-1]
            mfc = mf_now - prev_mf; rc = ret_now - prev_ret; tc = td_now - prev_td

            if action_today is not None: continue

            if td_now < 11:
                if holding:
                    cum_mult *= t_price / cur_buy_price
                    leg = (t_price/cur_buy_price-1)*100
                    print(f'  {t_time:<22} {"SELL T0":<10} {t_price:>5.2f} '
                          f'{fmt_num(mf_now)} {fmt_num(ret_now)} {fmt_num(td_now)} {leg:>+6.1f}%')
                    holding = False
                    terminated = True
                    action_today = 'sell'
                    break

            if holding:
                if mfc<0 and rc<0 and tc<0:
                    cum_mult *= t_price / cur_buy_price
                    leg = (t_price/cur_buy_price-1)*100
                    print(f'  {t_time:<22} {"SELL D6":<10} {t_price:>5.2f} '
                          f'{fmt_num(mf_now)} {fmt_num(ret_now)} {fmt_num(td_now)} {leg:>+6.1f}%')
                    holding = False
                    action_today = 'sell'
            else:
                if mfc>0:
                    cur_buy_price = t_price
                    print(f'  {t_time:<22} {"BUY U1":<10} {t_price:>5.2f} '
                          f'{fmt_num(mf_now)} {fmt_num(ret_now)} {fmt_now(td_now) if False else fmt_num(td_now)}')
                    holding = True
                    action_today = 'buy'

        if action_today: actions_tick.append(action_today)

        # 当天结束, prev = 当日 close 的三线 (跟下一日比)
        prev_mf = mf_arr[ti]
        prev_ret = retail_arr[ti]
        prev_td = trend_arr[ti]

    if holding:
        cum_mult *= closes[df_idx_2022_09_05] / cur_buy_price
        leg = (closes[df_idx_2022_09_05]/cur_buy_price-1)*100
        print(f'  (FC end) {test_dates[-1]:<22} {closes[df_idx_2022_09_05]:>5.2f} {leg:>+6.1f}%')

    print(f'\n  Tick 版: 累计 {cum_mult:.3f} ({(cum_mult-1)*100:+.1f}%) — 中间 {len(actions_tick)} 动作')
    if skip>0: print(f'  ({skip} 天 tick 缺失)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
