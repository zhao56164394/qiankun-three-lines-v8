# -*- coding: utf-8 -*-
"""002068 用 tick 实时三线 重跑 D6/U1 走势

逻辑:
  1. 拿历史日线 close (002068 从 2022-04-20 之前的 365 天)
  2. 加载每个交易日的 tick 数据 (2022-04 ~ 2022-09)
  3. 对每天每个 tick:
     - 把 tick 价当作"当下 close" 接到日线 series 末尾
     - 重算 trend / mf / retail 在 tick 时刻的值
     - 检查 D6/U1 条件 (跟昨日收盘三线比变化)
     - 触发就记录: 时间 + 价 + 操作
  4. 对比之前日 K 版本的操作流水

注意: 当日内 tick 触发后, 当天后续 tick 仍可能再触发反向操作 — 加规则: 一天最多操作一次 (买或卖)
"""
import os, sys, io, time
import numpy as np
import pandas as pd
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_ROOT = r'E:/BaiduSyncdisk/A股数据_分笔数据/分笔成交_按月归档_沪深'

sys.path.insert(0, ROOT)
from strategy.indicator import calc_trend_line, calc_main_force_line, calc_retail_line


def load_tick(date_str, code):
    """date_str: yyyymmdd, return DataFrame [time, price, vol, dir]"""
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
    except Exception as e:
        print(f'  ! tick load err {date_str} {code}: {e}')
        return None


def main():
    t0 = time.time()
    print('=== test183: 002068 tick 实时三线 ===\n')

    code = '002068'
    # 取从 2020-01-01 到 2022-12 的日线 (历史 + tick 测试期)
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'open', 'high', 'low', 'close',
                                 'retail', 'main_force'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p['date'] = p['date'].astype(str)
    p_002068 = p[p['code']==code].sort_values('date').reset_index(drop=True)
    p_002068 = p_002068[p_002068['date'] >= '2021-01-01'].reset_index(drop=True)

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = g['date'].astype(str)
    g_002068 = g[g['code']==code].sort_values('date').reset_index(drop=True)
    df = p_002068.merge(g_002068[['date','d_trend']], on='date', how='left')
    print(f'  002068 日线: {len(df)} 行 ({df["date"].min()} ~ {df["date"].max()})')

    # 找 2022-04-20 (信号日) 之后到 2022-09-05 (T0 触发日, 见 test182)
    df['idx'] = range(len(df))
    entry_idx_in_df = df[df['date']=='2022-04-20'].index[0]
    end_idx_in_df = df[df['date']=='2022-09-05'].index[0] if (df['date']=='2022-09-05').any() else len(df)-1

    # 抽出测试段日期
    test_dates = df['date'].iloc[entry_idx_in_df:end_idx_in_df+1].tolist()
    print(f'  测试段: {test_dates[0]} ~ {test_dates[-1]} 共 {len(test_dates)} 个交易日\n')

    # ===== 日线版三线 (作为对照) =====
    closes_full = df['close'].to_numpy().astype(np.float64)
    highs_full = df['high'].to_numpy().astype(np.float64)
    lows_full = df['low'].to_numpy().astype(np.float64)
    trend_full = calc_trend_line(closes_full, highs_full, lows_full)
    mf_full = calc_main_force_line(closes_full)
    retail_full = calc_retail_line(closes_full)

    # 对比下我们重算的 trend 跟 d_trend 一致吗 (sanity)
    d_trend_data = df['d_trend'].to_numpy()
    diffs = []
    for i in range(len(df)):
        if not np.isnan(d_trend_data[i]) and not np.isnan(trend_full[i]):
            diffs.append(abs(d_trend_data[i] - trend_full[i]))
    if diffs:
        print(f'  trend 重算 vs 数据库 diff: max={max(diffs):.2f}, avg={np.mean(diffs):.4f}')

    # ===== 日 K 版 D6/U1 流水 (重现 test182 结果) =====
    print(f'\n--- 日 K 版 D6/U1 流水 (基于收盘价) ---')
    holding = True
    cur_buy_price = closes_full[entry_idx_in_df]
    cum_mult = 1.0
    daily_log = [{'day':test_dates[0], 'action':'BUY', 'price':cur_buy_price,
                  'mf':mf_full[entry_idx_in_df], 'ret':retail_full[entry_idx_in_df],
                  'td':trend_full[entry_idx_in_df]}]
    for k in range(entry_idx_in_df+1, end_idx_in_df+1):
        td_v = trend_full[k]; td_p = trend_full[k-1]
        mf_v = mf_full[k]; mf_p = mf_full[k-1]
        ret_v = retail_full[k]; ret_p = retail_full[k-1]
        if np.isnan(td_v) or np.isnan(td_p): continue
        # T0
        if td_v < 11:
            if holding:
                cum_mult *= closes_full[k] / cur_buy_price
                daily_log.append({'day':df['date'].iloc[k], 'action':'SELL T0',
                                  'price':closes_full[k], 'mf':mf_v, 'ret':ret_v, 'td':td_v,
                                  'leg':(closes_full[k]/cur_buy_price-1)*100})
            break
        if np.isnan(mf_v) or np.isnan(mf_p) or np.isnan(ret_v) or np.isnan(ret_p): continue
        mfc = mf_v - mf_p; rc = ret_v - ret_p; tc = td_v - td_p
        if holding:
            if mfc < 0 and rc < 0 and tc < 0:
                cum_mult *= closes_full[k] / cur_buy_price
                holding = False
                daily_log.append({'day':df['date'].iloc[k], 'action':'SELL D6',
                                  'price':closes_full[k], 'mf':mf_v, 'ret':ret_v, 'td':td_v,
                                  'leg':(closes_full[k]/cur_buy_price-1)*100})
        else:
            if mfc > 0:
                cur_buy_price = closes_full[k]
                holding = True
                daily_log.append({'day':df['date'].iloc[k], 'action':'BUY U1',
                                  'price':closes_full[k], 'mf':mf_v, 'ret':ret_v, 'td':td_v})
    if holding:
        cum_mult *= closes_full[end_idx_in_df] / cur_buy_price
        daily_log.append({'day':df['date'].iloc[end_idx_in_df], 'action':'SELL FC',
                          'price':closes_full[end_idx_in_df],
                          'leg':(closes_full[end_idx_in_df]/cur_buy_price-1)*100})

    print(f'  日 K 版: {len(daily_log)} 个动作, 累计倍率 {cum_mult:.3f} ({(cum_mult-1)*100:+.1f}%)')

    # ===== Tick 版三线 =====
    print(f'\n--- Tick 版 D6/U1 流水 ---')
    print(f'  (历史 0..entry_idx-1 用日 close, 最后一日用 tick price)\n')

    # 历史日线序列 (前 entry_idx 个)
    # 对于"测试日 d", 历史 = closes[0..d-1] (日线), 当日 = tick 流
    # 实时三线: 把 tick 价加在末尾 → 重算

    # 为加速: 当天用 tick, 前一天的 close/high/low/三线已知
    # 每个 tick 算 trend/mf/retail 时, 历史用 closes[0..d-1] + [tick_price]

    # 规则: 每天最多操作一次 (避免一天反复进出)
    holding = True
    cur_buy_price = closes_full[entry_idx_in_df]  # 还是按日 close 入场 (entry 是日级建仓)
    cum_mult = 1.0
    tick_log = [{'time':test_dates[0]+' close', 'action':'BUY', 'price':cur_buy_price,
                 'note':'entry day close'}]

    # 前一天 (entry 日) 三线 — 作为基准比较
    prev_mf = mf_full[entry_idx_in_df]
    prev_ret = retail_full[entry_idx_in_df]
    prev_td = trend_full[entry_idx_in_df]

    skip_days = 0
    for ti in range(entry_idx_in_df+1, end_idx_in_df+1):
        d_str = df['date'].iloc[ti]
        d_yyyymmdd = d_str.replace('-','')

        ticks = load_tick(d_yyyymmdd, code)
        if ticks is None or len(ticks) < 5:
            skip_days += 1
            # 用日 K 处理这一天
            mf_v = mf_full[ti]; ret_v = retail_full[ti]; td_v = trend_full[ti]
            if not (np.isnan(mf_v) or np.isnan(ret_v) or np.isnan(td_v)):
                if td_v < 11:
                    if holding:
                        cum_mult *= closes_full[ti] / cur_buy_price
                        tick_log.append({'time':d_str+' close (no tick)', 'action':'SELL T0',
                                          'price':closes_full[ti], 'mf':mf_v,'ret':ret_v,'td':td_v,
                                          'leg':(closes_full[ti]/cur_buy_price-1)*100})
                        break
                mfc = mf_v - prev_mf; rc = ret_v - prev_ret; tc = td_v - prev_td
                if holding and mfc < 0 and rc < 0 and tc < 0:
                    cum_mult *= closes_full[ti] / cur_buy_price
                    holding = False
                    tick_log.append({'time':d_str+' close (no tick)', 'action':'SELL D6',
                                      'price':closes_full[ti], 'mf':mf_v,'ret':ret_v,'td':td_v,
                                      'leg':(closes_full[ti]/cur_buy_price-1)*100})
                elif (not holding) and mfc > 0:
                    cur_buy_price = closes_full[ti]
                    holding = True
                    tick_log.append({'time':d_str+' close (no tick)', 'action':'BUY U1',
                                      'price':closes_full[ti], 'mf':mf_v,'ret':ret_v,'td':td_v})
                prev_mf = mf_v; prev_ret = ret_v; prev_td = td_v
            continue

        # 有 tick — 选若干代表 tick (避免太多, 用每分钟一个 tick)
        ticks['time'] = ticks['time'].astype(str)
        ticks['min'] = ticks['time'].str[:16]  # yyyy-mm-dd HH:MM
        # 每分钟取最后一笔
        ticks_min = ticks.groupby('min').last().reset_index()

        # 历史 close/high/low (前 ti 个)
        hist_C = closes_full[:ti]
        hist_H = highs_full[:ti]
        hist_L = lows_full[:ti]

        # 当日 tick 累计 high/low (随 tick 增长)
        day_high = -np.inf; day_low = np.inf
        action_today = None  # 一天只操作一次

        for _, t in ticks_min.iterrows():
            t_price = t['price']
            t_time = t['time']
            day_high = max(day_high, t_price)
            day_low = min(day_low, t_price)

            # 历史 + 当日 tick price
            full_C = np.r_[hist_C, t_price]
            full_H = np.r_[hist_H, day_high]
            full_L = np.r_[hist_L, day_low]

            td_now = calc_trend_line(full_C, full_H, full_L)[-1]
            mf_now = calc_main_force_line(full_C)[-1]
            ret_now = calc_retail_line(full_C)[-1]

            mfc = mf_now - prev_mf
            rc = ret_now - prev_ret
            tc = td_now - prev_td

            if action_today is not None: continue  # 当天已操作

            if td_now < 11:
                if holding:
                    cum_mult *= t_price / cur_buy_price
                    holding = False
                    tick_log.append({'time':t_time, 'action':'SELL T0',
                                      'price':t_price, 'mf':mf_now,'ret':ret_now,'td':td_now,
                                      'leg':(t_price/cur_buy_price-1)*100})
                    action_today = 'sell'
                    # T0 是终结条件, 不再继续
                    print(f'\n  T0 终结 @ {t_time} price {t_price:.2f}')
                    # 把累计标记一下
                    # 跳出整个交易日和后续日
                    break

            if holding:
                if mfc < 0 and rc < 0 and tc < 0:
                    cum_mult *= t_price / cur_buy_price
                    holding = False
                    tick_log.append({'time':t_time, 'action':'SELL D6',
                                      'price':t_price, 'mf':mf_now,'ret':ret_now,'td':td_now,
                                      'leg':(t_price/cur_buy_price-1)*100})
                    action_today = 'sell'
            else:
                if mfc > 0:
                    cur_buy_price = t_price
                    holding = True
                    tick_log.append({'time':t_time, 'action':'BUY U1',
                                      'price':t_price, 'mf':mf_now,'ret':ret_now,'td':td_now})
                    action_today = 'buy'

        # T0 终结判断 (跳出所有日)
        if action_today == 'sell' and not holding:
            # 检查最后 log 是不是 T0
            if tick_log[-1]['action'] == 'SELL T0':
                break

        # 当天结束, 用日 close 三线作为 prev (跟下一天比)
        prev_mf = mf_full[ti]
        prev_ret = retail_full[ti]
        prev_td = trend_full[ti]

    if holding:
        last_idx = end_idx_in_df
        cum_mult *= closes_full[last_idx] / cur_buy_price
        tick_log.append({'time':df['date'].iloc[last_idx]+' close', 'action':'SELL FC',
                          'price':closes_full[last_idx],
                          'leg':(closes_full[last_idx]/cur_buy_price-1)*100})

    print(f'\n  Tick 版: {len(tick_log)} 个动作, 累计倍率 {cum_mult:.3f} ({(cum_mult-1)*100:+.1f}%)')
    if skip_days > 0:
        print(f'  ({skip_days} 天 tick 数据缺失, 用日 close fallback)')

    # ===== 对比 =====
    print(f'\n{"="*82}')
    print(f'  日 K 版 vs Tick 版 流水对比')
    print(f'{"="*82}')

    print(f'\n--- 日 K 版 ---')
    print(f'  {"日期":<12} {"动作":<10} {"价":>6} {"mf":>6} {"ret":>6} {"td":>6} {"本腿":>7}')
    for log in daily_log:
        leg = log.get('leg', '')
        ls = f'{leg:+6.1f}%' if isinstance(leg, float) else ''
        print(f'  {log["day"]:<12} {log["action"]:<10} {log["price"]:>5.2f} '
              f'{log.get("mf",0):>+5.0f} {log.get("ret",0):>+5.0f} {log.get("td",0):>+5.1f} {ls:>7}')

    print(f'\n--- Tick 版 ---')
    print(f'  {"时间":<22} {"动作":<10} {"价":>6} {"mf":>6} {"ret":>6} {"td":>6} {"本腿":>7}')
    for log in tick_log:
        leg = log.get('leg', '')
        ls = f'{leg:+6.1f}%' if isinstance(leg, float) else ''
        print(f'  {log["time"]:<22} {log["action"]:<10} {log["price"]:>5.2f} '
              f'{log.get("mf","--"):>+5} {log.get("ret","--"):>+5} {log.get("td","--"):>+5} {ls:>7}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
