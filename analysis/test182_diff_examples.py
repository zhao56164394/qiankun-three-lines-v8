# -*- coding: utf-8 -*-
"""找"波段最大涨幅 ≥100% 但 D6/U1/T0 实际收益 <30%"的股票
对比每只股的:
  - signal_date (波段起点)
  - entry_date (建仓日)
  - 波段算法的 min_close / max_close / 理论涨幅
  - D6/U1 实际 sell_date / 实际涨幅 / 腿数
  - 中间所有 D6 卖出和 U1 买入的日期+价格
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WAIT_MAX = 60


def find_signals(arrays, mkt_arrs):
    cs = arrays['starts']; ce = arrays['ends']
    td = arrays['td']; close = arrays['close']
    retail = arrays['retail']; mf = arrays['mf']
    stk_d_t = arrays['stk_d_t']; stk_m_t = arrays['stk_m_t']; stk_y_t = arrays['stk_y_t']
    mkt_d_t = mkt_arrs['mkt_d_t']; mkt_m_t = mkt_arrs['mkt_m_t']; mkt_y_t = mkt_arrs['mkt_y_t']
    date = arrays['date']; code = arrays['code']
    sigs = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < 30: continue
        for i in range(s + 1, e):
            cur = td[i]; prev = td[i-1]
            if np.isnan(cur) or np.isnan(prev): continue
            if prev > 11 and cur <= 11:
                cond_mkt = (not np.isnan(mkt_d_t[i]) and mkt_d_t[i] <= 50 and
                              not np.isnan(mkt_m_t[i]) and mkt_m_t[i] <= 50 and
                              not np.isnan(mkt_y_t[i]) and mkt_y_t[i] <= 50)
                cond_stk = (not np.isnan(stk_d_t[i]) and stk_d_t[i] <= 50 and
                              not np.isnan(stk_m_t[i]) and stk_m_t[i] > 50 and
                              not np.isnan(stk_y_t[i]) and stk_y_t[i] > 50)
                cond_mf = (not np.isnan(mf[i]) and mf[i] <= -100)
                cond_ret = (not np.isnan(retail[i]) and retail[i] <= -100)
                if cond_mkt and cond_stk and cond_mf and cond_ret:
                    sigs.append({'signal_idx': i,'signal_date': date[i],'code': code[i],
                                 'cur_mf': mf[i],'cur_retail': retail[i],'code_end': e})
    return pd.DataFrame(sigs)


def find_entry(signal_idx, code_end, td, mf, retail):
    end_search = min(code_end - 1, signal_idx + WAIT_MAX)
    for k in range(signal_idx + 1, end_search + 1):
        if k - 1 < 0: continue
        if np.isnan(td[k]) or np.isnan(td[k-1]): continue
        if td[k] <= 11: continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        mf_c = mf[k] - mf[k-1]; ret_c = retail[k] - retail[k-1]
        if mf_c > 0 and ret_c > 0:
            return k
    return -1


def simulate_with_log(buy_idx, code_end, td, close, mf, retail, date_arr):
    """模拟并记录所有 D6/U1 事件"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    legs_log = [{'action': 'BUY (entry)', 'date': date_arr[buy_idx], 'price': bp_first,
                 'mf': mf[buy_idx], 'retail': retail[buy_idx], 'trend': td[buy_idx],
                 'cum_mult': 1.0}]

    for k in range(buy_idx + 1, code_end):
        if np.isnan(td[k]): continue

        if td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs_log.append({'action':'SELL (T0 trend<11)','date':date_arr[k],'price':close[k],
                                 'mf':mf[k],'retail':retail[k],'trend':td[k],
                                 'cum_mult':cum_mult, 'leg_ret':(close[k]/cur_buy_price-1)*100})
            return cum_mult, legs_log

        if k < 1: continue
        if np.isnan(mf[k]) or np.isnan(mf[k-1]): continue
        if np.isnan(retail[k]) or np.isnan(retail[k-1]): continue
        if np.isnan(td[k-1]): continue

        mf_c = mf[k] - mf[k-1]
        ret_c = retail[k] - retail[k-1]
        td_c = td[k] - td[k-1]

        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                holding = False
                legs_log.append({'action':'SELL (D6)','date':date_arr[k],'price':close[k],
                                 'mf':mf[k],'retail':retail[k],'trend':td[k],
                                 'cum_mult':cum_mult, 'leg_ret':(close[k]/cur_buy_price-1)*100})
        else:
            if mf_c > 0:
                cur_buy_price = close[k]
                holding = True
                legs_log.append({'action':'BUY (U1)','date':date_arr[k],'price':close[k],
                                 'mf':mf[k],'retail':retail[k],'trend':td[k],
                                 'cum_mult':cum_mult})

    if holding:
        cum_mult *= close[code_end-1] / cur_buy_price
        legs_log.append({'action':'SELL (force_close)','date':date_arr[code_end-1],
                          'price':close[code_end-1],
                          'mf':mf[code_end-1],'retail':retail[code_end-1],'trend':td[code_end-1],
                          'cum_mult':cum_mult, 'leg_ret':(close[code_end-1]/cur_buy_price-1)*100})
    return cum_mult, legs_log


def calc_band_max(buy_idx, code_end, td, close):
    """波段算法: 从 buy_idx 起, 直到下次 trend<=11, 找 max/min"""
    end_idx = code_end - 1
    for k in range(buy_idx + 1, code_end):
        if not np.isnan(td[k]) and td[k] <= 11:
            end_idx = k; break
    seg = close[buy_idx:end_idx+1]
    seg_v = seg[~np.isnan(seg)]
    if len(seg_v) < 2: return None, None, None, 0
    idx_max_local = int(np.argmax(seg_v))
    max_close = seg_v[idx_max_local]
    min_before = np.min(seg_v[:idx_max_local+1])
    if min_before <= 0: return None, None, None, 0
    return min_before, max_close, (max_close/min_before-1)*100, end_idx - buy_idx


def main():
    t0 = time.time()
    print('=== test182: 反差最大的股票看通达信 ===\n')

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
    mkt = mkt.drop_duplicates('date').rename(columns={'d_trend':'mkt_d_t', 'm_trend':'mkt_m_t', 'y_trend':'mkt_y_t'})

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
              'retail':retail_arr,'mf':mf_arr,
              'stk_d_t':stk_d_t,'stk_m_t':stk_m_t,'stk_y_t':stk_y_t,
              'starts':code_starts,'ends':code_ends}
    mkt_arrs = {'mkt_d_t':mkt_d_t, 'mkt_m_t':mkt_m_t, 'mkt_y_t':mkt_y_t}

    df_sig = find_signals(arrays, mkt_arrs)
    df_sig = df_sig[df_sig['signal_date'] >= '2016-01-01'].reset_index(drop=True)
    print(f'  信号: {len(df_sig)}')

    rows = []
    for _, s in df_sig.iterrows():
        si = int(s['signal_idx']); ce = int(s['code_end'])
        ei = find_entry(si, ce, trend_arr, mf_arr, retail_arr)
        if ei < 0: continue
        # 波段算法
        min_c, max_c, band_gain, band_days = calc_band_max(ei, ce, trend_arr, close_arr)
        if band_gain is None: continue
        # D6/U1 实际
        cum_mult, legs_log = simulate_with_log(ei, ce, trend_arr, close_arr, mf_arr, retail_arr, date_arr)
        actual_gain = (cum_mult - 1) * 100
        rows.append({
            'code': s['code'], 'signal_date': s['signal_date'],
            'entry_date': date_arr[ei], 'entry_idx': ei, 'code_end': ce,
            'band_min': min_c, 'band_max': max_c, 'band_gain': band_gain,
            'band_days': band_days,
            'actual_gain': actual_gain,
            'gap': band_gain - actual_gain,
            'n_legs': len(legs_log),
            'legs_log': legs_log,
        })

    df_r = pd.DataFrame(rows)
    print(f'  分析 {len(df_r)} 只\n')

    # 找反差最大: band_gain >= 100 且 actual < 30
    print(f'{"="*82}')
    print(f'  反差最大: 波段 ≥+100% 但实际 <+30%')
    print(f'{"="*82}\n')

    bad = df_r[(df_r['band_gain']>=100) & (df_r['actual_gain']<30)].sort_values('gap', ascending=False)
    print(f'  共 {len(bad)} 只\n')

    print(f'  {"代码":<8} {"信号日":<12} {"建仓日":<12} {"波段涨":>9} {"实际":>8} {"差":>8} {"腿":>3} {"波段天":>5}')
    for _, r in bad.head(20).iterrows():
        print(f'  {r["code"]:<8} {r["signal_date"]:<12} {r["entry_date"]:<12} '
              f'{r["band_gain"]:>+8.1f}% {r["actual_gain"]:>+7.1f}% {r["gap"]:>+7.1f}% '
              f'{r["n_legs"]:>3} {r["band_days"]:>5}')

    # 详细看前 5 只
    print(f'\n{"="*82}')
    print(f'  前 5 只反差最大的, 完整 D6/U1 流水')
    print(f'{"="*82}')
    for _, r in bad.head(5).iterrows():
        print(f'\n  ─── {r["code"]} 信号日={r["signal_date"]} 建仓日={r["entry_date"]} ───')
        print(f'  波段:  min={r["band_min"]:.2f} ({r["entry_date"]}+) → max={r["band_max"]:.2f}, 涨{r["band_gain"]:+.1f}% ({r["band_days"]}d)')
        print(f'  实际:  D6/U1 累计 {r["actual_gain"]:+.1f}%, 腿数 {r["n_legs"]}')
        print(f'  详细操作流水:')
        print(f'    {"动作":<22} {"日期":<12} {"价":>6} {"mf":>6} {"retail":>7} {"trend":>6} {"累计倍率":>8} {"本腿%":>7}')
        for log in r['legs_log']:
            leg_ret = log.get('leg_ret', '')
            leg_str = f'{leg_ret:>+6.1f}%' if isinstance(leg_ret, float) else ''
            print(f'    {log["action"]:<22} {log["date"]:<12} {log["price"]:>5.2f} '
                  f'{log["mf"]:>+5.0f} {log["retail"]:>+6.0f} {log["trend"]:>+5.1f} '
                  f'{log["cum_mult"]:>7.3f} {leg_str:>7}')

    # ===== 反过来看: 实际拿到 ≥+100% 的股 =====
    print(f'\n{"="*82}')
    print(f'  实际 D6/U1 拿到 ≥+100% 的股 (好案例)')
    print(f'{"="*82}\n')
    good = df_r[df_r['actual_gain']>=100].sort_values('actual_gain', ascending=False)
    print(f'  共 {len(good)} 只\n')
    print(f'  {"代码":<8} {"信号日":<12} {"建仓日":<12} {"波段涨":>9} {"实际":>8} {"腿":>3}')
    for _, r in good.iterrows():
        print(f'  {r["code"]:<8} {r["signal_date"]:<12} {r["entry_date"]:<12} '
              f'{r["band_gain"]:>+8.1f}% {r["actual_gain"]:>+7.1f}% {r["n_legs"]:>3}')

    # 写出
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_r_out = df_r.drop(columns=['legs_log'])
    df_r_out.to_csv(os.path.join(out_dir, 'strategy_v1_diff.csv'), index=False, encoding='utf-8-sig')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
