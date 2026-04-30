# -*- coding: utf-8 -*-
"""三模式各取 5 只暴涨股代表 - 通达信验证用"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_nopool(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        last_mf = np.nan; last_retail = np.nan
        last_trigger = -999
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if mf_rising and retail_rising and trend_ok and (i - last_trigger) >= 30:
                events.append({
                    'date':date[gi],'code':code[gi],
                    'buy_idx_global':gi,
                    'cur_retail':retail[gi],
                    'cur_mf':mf[gi],
                    'cur_trend':td[gi],
                })
                last_trigger = i
            last_mf = mf[gi]; last_retail = retail[gi]
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    bp = close[buy_idx]; cum_mult = 1.0; holding = True
    cur_buy_price = bp
    sell_idx = max_end; reason = 'fc'
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding: cum_mult *= close[k] / cur_buy_price
            return k, 'td<11', (cum_mult-1)*100
        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0
        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]; holding = True
    if holding: cum_mult *= close[max_end] / cur_buy_price
    return max_end, 'fc', (cum_mult-1)*100


def main():
    t0 = time.time()
    print('=== test169: 三模式各取代表股 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'retail':retail_arr,'mf':mf_arr,'td':trend_arr,
              'starts':code_starts,'ends':code_ends}

    df_e = find_signals_nopool(arrays)
    print(f'  NoP 触发: {len(df_e):,}')

    # 算 ret + 卖出日
    sell_dates = []; sell_reasons = []; rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        sk, sr, rt = simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)
        sell_dates.append(date_arr[sk])
        sell_reasons.append(sr)
        rets.append(rt)
    df_e['ret_pct'] = rets
    df_e['sell_date'] = sell_dates
    df_e['sell_reason'] = sell_reasons

    # 加买入价 / 卖出价
    df_e['buy_price'] = close_arr[df_e['buy_idx_global'].astype(int).values]
    sell_idx = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        sk, _, _ = simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end)
        sell_idx.append(sk)
    df_e['sell_price'] = close_arr[np.array(sell_idx)]
    df_e['hold_days'] = np.array(sell_idx) - df_e['buy_idx_global'].astype(int).values

    # 分模式
    def mode(r):
        if r < -150: return 'M1'
        elif r < 50: return 'M2'
        else: return 'M3'
    df_e['mode'] = df_e['cur_retail'].apply(mode)

    baggers = df_e[df_e['ret_pct']>=100].copy()
    print(f'  ≥+100% 暴涨股: {len(baggers)}')
    print(f'    M1: {(baggers["mode"]=="M1").sum()}')
    print(f'    M2: {(baggers["mode"]=="M2").sum()}')
    print(f'    M3: {(baggers["mode"]=="M3").sum()}')

    print(f'\n{"="*90}')
    print(f'  各模式 5 只代表暴涨股')
    print(f'{"="*90}')

    for m, m_desc in [('M1', '深抛反弹 (cur_retail < -150)'),
                       ('M2', '中性启动 (cur_retail -150 ~ +50)'),
                       ('M3', '高位续涨 (cur_retail > +50)')]:
        sub = baggers[baggers['mode']==m].sort_values('ret_pct', ascending=False)
        print(f'\n  --- {m}: {m_desc} ---')
        print(f'  {"代码":<8} {"买入日":<12} {"买价":>7} {"卖出日":<12} {"卖价":>7} '
              f'{"持仓":>5} {"ret%":>9} {"cur_ret":>8} {"cur_mf":>7} {"cur_td":>7} {"卖出原因":<10}')
        for _, r in sub.head(8).iterrows():
            print(f'  {r["code"]:<8} {r["date"]:<12} {r["buy_price"]:>6.2f} '
                  f'{r["sell_date"]:<12} {r["sell_price"]:>6.2f} '
                  f'{r["hold_days"]:>4}d {r["ret_pct"]:>+8.1f}% '
                  f'{r["cur_retail"]:>+7.0f} {r["cur_mf"]:>+6.0f} {r["cur_trend"]:>+6.1f} '
                  f'{r["sell_reason"]:<10}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
