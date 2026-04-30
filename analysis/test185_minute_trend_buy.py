# -*- coding: utf-8 -*-
"""002068 新买点规则: 分时 trend 上穿 11 触发买入

买点:
  分时 trend (window=55, tick 序列, 不足 55 个用上一交易日 tick 补) 上穿 11
  上穿那一刻:
    - 如果首次建仓 (or 信号日入场): E2+E3 → 日线 mf↑ AND retail↑ AND trend>11 (tick 实时算)
    - 如果是再买入 (pending → holding): U1 → 日线 mf↑ (tick 实时算)

卖点 (沿用 tick 版):
  D6 实时: 日线 mf↓ AND retail↓ AND trend↓ (tick 实时)
  T0 实时: 日线 trend < 11 (tick 实时)

注意:
  - "分时 trend" 用 tick 价当 close, 但 high/low 用同一个 tick 序列的 cummax/cummin
    简化: 不用 high/low, 直接用 tick price 的 rolling(55).max/min 作为 hhv/llv
  - 分时 trend 上穿事件: 上 tick td<=11 → 当前 tick td>11
"""
import os, sys, io, time, zipfile
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_ROOT = r'E:/BaiduSyncdisk/A股数据_分笔数据/分笔成交_按月归档_沪深'
sys.path.insert(0, ROOT)
from strategy.indicator import _tdx_sma, _tdx_ema, calc_main_force_line, calc_retail_line


def calc_trend_simple(prices, hhv_arr, llv_arr):
    """trend 公式 (window=55 等参数已经预先在 hhv/llv 里), 仅算 SMA1/SMA2/EMA"""
    denom = hhv_arr - llv_arr
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.where(denom > 0, (prices - llv_arr) / denom * 100, 50.0)
    sma1 = _tdx_sma(X, 5, 1)
    sma2 = _tdx_sma(sma1, 3, 1)
    V11 = 3 * sma1 - 2 * sma2
    return _tdx_ema(V11, 3)


def calc_d_trend_w12(closes, highs, lows, window=12):
    C = np.array(closes, dtype=float)
    H = np.array(highs, dtype=float)
    L = np.array(lows, dtype=float)
    llv = pd.Series(L).rolling(window, min_periods=1).min().values
    hhv = pd.Series(H).rolling(window, min_periods=1).max().values
    return calc_trend_simple(C, hhv, llv)


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


def fmt_n(v, w=5, prec=1):
    if isinstance(v, str): return f'{v:>{w}}'
    if v is None or (isinstance(v,float) and np.isnan(v)): return f'{"--":>{w}}'
    return f'{v:>+{w}.{prec}f}'


def main():
    t0 = time.time()
    print('=== test185: 002068 分时 trend 上穿 11 → 日线三线买点 ===\n')

    code = '002068'
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'high', 'low', 'close'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p['date'] = p['date'].astype(str)
    df = p[(p['code']==code) & (p['date']>='2021-01-01')].sort_values('date').reset_index(drop=True)

    closes = df['close'].to_numpy().astype(np.float64)
    highs = df['high'].to_numpy().astype(np.float64)
    lows = df['low'].to_numpy().astype(np.float64)
    d_trend = calc_d_trend_w12(closes, highs, lows)
    d_mf = calc_main_force_line(closes)
    d_retail = calc_retail_line(closes)

    # 找 entry 与 测试段
    df_idx_2022_04_20 = df[df['date']=='2022-04-20'].index[0]
    df_idx_2022_09_05 = df[df['date']=='2022-09-05'].index[0]
    test_dates = df['date'].iloc[df_idx_2022_04_20:df_idx_2022_09_05+1].tolist()
    print(f'  测试段: {test_dates[0]} ~ {test_dates[-1]} 共 {len(test_dates)} 个交易日\n')

    # 加载所有测试日 + 前一日的 tick (用于 55 窗口补足)
    print('  加载 tick 数据...')
    tick_cache = {}  # date_str -> DataFrame[time, price, min]
    pre_date = df['date'].iloc[df_idx_2022_04_20-1] if df_idx_2022_04_20>0 else None
    all_dates_to_load = [pre_date] + test_dates if pre_date else test_dates
    for d in all_dates_to_load:
        if d is None: continue
        tk = load_tick(d.replace('-',''), code)
        if tk is not None and len(tk) >= 5:
            tk['time'] = tk['time'].astype(str)
            tk['min'] = tk['time'].str[:16]
            tk = tk.groupby('min').last().reset_index()
            tick_cache[d] = tk
    print(f'    {len(tick_cache)} 个交易日有 tick\n')

    # ===== 模拟 =====
    holding = True  # 信号日入场, 已持仓
    cur_buy_price = closes[df_idx_2022_04_20]
    cum_mult = 1.0
    actions = [(test_dates[0]+' close', 'BUY (entry)', cur_buy_price,
                d_mf[df_idx_2022_04_20], d_retail[df_idx_2022_04_20], d_trend[df_idx_2022_04_20], None)]

    # 入场日 tick 不参与 (从下一交易日开始监控)
    terminated = False
    skip = 0
    cnt_buy = 0; cnt_sell = 0

    for ti in range(df_idx_2022_04_20+1, df_idx_2022_09_05+1):
        if terminated: break
        d_str = df['date'].iloc[ti]
        prev_d_str = df['date'].iloc[ti-1]

        if d_str not in tick_cache:
            skip += 1
            # fallback 日 K 判断
            mf_v = d_mf[ti]; rt_v = d_retail[ti]; td_v = d_trend[ti]
            mf_p = d_mf[ti-1]; rt_p = d_retail[ti-1]; td_p = d_trend[ti-1]
            if any(np.isnan(x) for x in [mf_v,rt_v,td_v,mf_p,rt_p,td_p]): continue

            if td_v < 11:
                if holding:
                    cum_mult *= closes[ti]/cur_buy_price
                    leg = (closes[ti]/cur_buy_price-1)*100
                    actions.append((d_str+' close (no tick)', 'SELL T0', closes[ti], mf_v,rt_v,td_v, leg))
                    holding = False; cnt_sell+=1; terminated = True
                continue

            mfc = mf_v - mf_p; rc = rt_v - rt_p; tc = td_v - td_p
            if holding and mfc<0 and rc<0 and tc<0:
                cum_mult *= closes[ti]/cur_buy_price
                leg = (closes[ti]/cur_buy_price-1)*100
                actions.append((d_str+' close (no tick)', 'SELL D6', closes[ti], mf_v,rt_v,td_v, leg))
                holding = False; cnt_sell+=1
            continue

        # 有 tick
        # 准备 minute 序列 — 当日 + 前一交易日 (用于补足 55 窗口)
        cur_ticks = tick_cache[d_str].copy()
        if prev_d_str in tick_cache:
            prev_ticks = tick_cache[prev_d_str].copy()
            full_ticks = pd.concat([prev_ticks, cur_ticks], ignore_index=True)
        else:
            full_ticks = cur_ticks

        # 计算分时 trend (window=55, 用 tick price 作为 hhv/llv 输入)
        prices_seq = full_ticks['price'].to_numpy().astype(np.float64)
        hhv_55 = pd.Series(prices_seq).rolling(55, min_periods=1).max().values
        llv_55 = pd.Series(prices_seq).rolling(55, min_periods=1).min().values
        td_seq = calc_trend_simple(prices_seq, hhv_55, llv_55)

        # 只关心当日的 tick (索引: prev_ticks 长度起开始)
        prev_len = len(tick_cache.get(prev_d_str, pd.DataFrame()))
        cur_start = prev_len

        # 上一日 close 时的日线三线 (作为 prev_mf/prev_ret/prev_td 跟今日比)
        prev_mf = d_mf[ti-1]
        prev_ret = d_retail[ti-1]
        prev_td = d_trend[ti-1]
        hist_C = closes[:ti]
        hist_H = highs[:ti]
        hist_L = lows[:ti]

        day_high = -np.inf; day_low = np.inf
        action_today = None  # 一天最多一个动作 (买或卖)

        # 遍历当日 tick
        for k in range(cur_start, len(full_ticks)):
            t_price = prices_seq[k]
            t_time = full_ticks['time'].iloc[k]
            day_high = max(day_high, t_price)
            day_low = min(day_low, t_price)

            # 分时 trend 当前值 + 前 1 tick 值
            mttd_now = td_seq[k]
            mttd_prev = td_seq[k-1] if k>0 else np.nan

            # 实时日线三线
            full_C = np.r_[hist_C, t_price]
            full_H = np.r_[hist_H, day_high]
            full_L = np.r_[hist_L, day_low]
            d_td_now = calc_d_trend_w12(full_C, full_H, full_L)[-1]
            d_mf_now = calc_main_force_line(full_C)[-1]
            d_ret_now = calc_retail_line(full_C)[-1]

            d_mfc = d_mf_now - prev_mf
            d_rc = d_ret_now - prev_ret
            d_tc = d_td_now - prev_td

            if action_today is not None: continue  # 一天一动作

            # T0 终结优先
            if d_td_now < 11:
                if holding:
                    cum_mult *= t_price / cur_buy_price
                    leg = (t_price/cur_buy_price-1)*100
                    actions.append((t_time, 'SELL T0', t_price, d_mf_now, d_ret_now, d_td_now, leg))
                    holding = False; cnt_sell+=1
                    action_today = 'sell'
                    terminated = True
                    break

            # 卖点: 实时 D6
            if holding:
                if d_mfc<0 and d_rc<0 and d_tc<0:
                    cum_mult *= t_price / cur_buy_price
                    leg = (t_price/cur_buy_price-1)*100
                    actions.append((t_time, 'SELL D6', t_price, d_mf_now, d_ret_now, d_td_now, leg))
                    holding = False; cnt_sell+=1
                    action_today = 'sell'
            else:
                # 买点: 分时 trend 上穿 11 (前 tick td<=11, 当前 td>11)
                if (not np.isnan(mttd_prev)) and mttd_prev <= 11 and mttd_now > 11:
                    # U1 再买入: mf 上升 (跟前日 close 比)
                    if d_mfc > 0:
                        cur_buy_price = t_price
                        actions.append((t_time + f' (mt_td {mttd_prev:.1f}→{mttd_now:.1f})',
                                          'BUY U1', t_price, d_mf_now, d_ret_now, d_td_now, None))
                        holding = True; cnt_buy+=1
                        action_today = 'buy'

        # 当日结束

    if holding:
        cum_mult *= closes[df_idx_2022_09_05] / cur_buy_price
        leg = (closes[df_idx_2022_09_05]/cur_buy_price-1)*100
        actions.append((test_dates[-1]+' close (FC)', 'SELL FC', closes[df_idx_2022_09_05],
                          None, None, None, leg))
        cnt_sell+=1

    # 输出
    print(f'  {"时间":<35} {"动作":<12} {"价":>6} {"mf":>7} {"ret":>7} {"td":>6} {"本腿":>7}')
    for a in actions:
        t, act, pr, mf, rt, td, leg = a
        print(f'  {t:<35} {act:<12} {pr:>5.2f} {fmt_n(mf,6,1)} {fmt_n(rt,6,1)} {fmt_n(td,5,1)} '
              f'{(f"{leg:+6.1f}%" if leg is not None else ""):>7}')

    print(f'\n  累计 {cum_mult:.3f} ({(cum_mult-1)*100:+.1f}%)')
    print(f'  买 {cnt_buy} 次, 卖 {cnt_sell} 次, 总 {len(actions)-1} 动作')
    if skip>0: print(f'  ({skip} 天 tick 缺失, 用日 close)')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
