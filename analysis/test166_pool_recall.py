# -*- coding: utf-8 -*-
"""验证 retail<-250 入池条件对暴涨股召回率的影响

3 套入场条件对比:
  V5  : retail<-250 池 + mf 上升 + trend>11 + retail 上升  (现状)
  NoP : 不要池, 只 mf 上升 + trend>11 + retail 上升
  Min : 只 mf 上升 + trend>11

每套都 simulate D6+U1+T0, 算 ret%, 数 ≥+100%, ≥+200% 的笔数.
看不要 -250 池, 暴涨股是不是被遗漏 (反过来, 遗漏的暴涨股入场时 retail 在哪).
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v5(arrays):
    """V5: retail<-250 池 + mf 上升 + trend>11 + retail 上升"""
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
        pool_min_retail = np.inf
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True; pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'pool_min_retail':pool_min_retail,
                               'cur_retail':retail[gi],
                               'cur_mf':mf[gi],
                               'cur_trend':td[gi]})
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
    return pd.DataFrame(events)


def find_signals_nopool(arrays):
    """NoP: 无池, mf 上升 + trend>11 + retail 上升, 但加 30 日内不重复触发"""
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
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'cur_retail':retail[gi],
                               'cur_mf':mf[gi],
                               'cur_trend':td[gi]})
                last_trigger = i
            last_mf = mf[gi]; last_retail = retail[gi]
    return pd.DataFrame(events)


def find_signals_min(arrays):
    """Min: 只 mf 上升 + trend>11, 30 日不重复"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        last_mf = np.nan
        last_trigger = -999
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if mf_rising and trend_ok and (i - last_trigger) >= 30:
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'cur_retail':retail[gi],
                               'cur_mf':mf[gi],
                               'cur_trend':td[gi]})
                last_trigger = i
            last_mf = mf[gi]
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    bp = close[buy_idx]; cum_mult = 1.0; holding = True
    cur_buy_price = bp
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding: cum_mult *= close[k] / cur_buy_price
            return (cum_mult-1)*100
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
    return (cum_mult-1)*100


def annotate_rets(df_e, code_starts, code_ends, trend_arr, close_arr, mf_arr, retail_arr):
    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e = df_e.copy()
    df_e['ret_pct'] = rets
    return df_e


def stat(df, name):
    n = len(df)
    n100 = (df['ret_pct']>=100).sum()
    n200 = (df['ret_pct']>=200).sum()
    n500 = (df['ret_pct']>=500).sum()
    return {
        'name': name, 'n': n,
        'h100': n100, 'h200': n200, 'h500': n500,
        'r100': n100/n*100 if n else 0,
        'r200': n200/n*100 if n else 0,
        'r500': n500/n*100 if n else 0,
        'avg': df['ret_pct'].mean() if n else 0,
        'win': (df['ret_pct']>0).mean()*100 if n else 0,
    }


def main():
    t0 = time.time()
    print('=== test166: -250 入池对暴涨股召回率影响 ===\n')

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
    print(f'  数据: {len(df):,} 行\n')

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

    # ===== 跑 3 套 =====
    print('=== 信号生成 ===')
    print('  V5 (retail<-250 池)...')
    df_v5 = find_signals_v5(arrays)
    print(f'    {len(df_v5):,} 事件')

    print('  NoP (无池, mf 上升+trend>11+retail 上升, 30d 不重复)...')
    df_nop = find_signals_nopool(arrays)
    print(f'    {len(df_nop):,} 事件')

    print('  Min (只 mf 上升+trend>11, 30d 不重复)...')
    df_min = find_signals_min(arrays)
    print(f'    {len(df_min):,} 事件')

    # ===== 算 ret =====
    print('\n=== 计算 ret ===')
    print('  V5...'); df_v5 = annotate_rets(df_v5, code_starts, code_ends, trend_arr, close_arr, mf_arr, retail_arr)
    print('  NoP...'); df_nop = annotate_rets(df_nop, code_starts, code_ends, trend_arr, close_arr, mf_arr, retail_arr)
    print('  Min...'); df_min = annotate_rets(df_min, code_starts, code_ends, trend_arr, close_arr, mf_arr, retail_arr)

    # ===== 总览 =====
    print(f'\n{"="*82}')
    print(f'  暴涨股召回率对比 (≥+100% 为暴涨股)')
    print(f'{"="*82}')
    print(f'\n  {"方案":<6} {"事件":>10} {"avg":>8} {"win":>6} {"≥100":>5} {"r100":>7} '
          f'{"≥200":>5} {"r200":>7} {"≥500":>4}')
    for df_, name in [(df_v5, 'V5'), (df_nop, 'NoP'), (df_min, 'Min')]:
        s = stat(df_, name)
        print(f'  {s["name"]:<6} {s["n"]:>10,} {s["avg"]:>+7.2f}% {s["win"]:>5.1f}% '
              f'{s["h100"]:>5} {s["r100"]:>+6.2f}% {s["h200"]:>5} {s["r200"]:>+6.2f}% {s["h500"]:>4}')

    # ===== V5 漏掉的暴涨股 (NoP 里有但 V5 没有) =====
    print(f'\n{"="*82}')
    print(f'  NoP 里 ≥+100% 但 V5 没有的暴涨股')
    print(f'{"="*82}')

    # 用 (date, code) 做 key 比对
    v5_keys = set(zip(df_v5['date'], df_v5['code']))
    nop_baggers = df_nop[df_nop['ret_pct']>=100].copy()
    nop_extra = nop_baggers[~nop_baggers.apply(lambda r: (r['date'], r['code']) in v5_keys, axis=1)]
    print(f'\n  NoP 总暴涨股: {len(nop_baggers)}')
    print(f'  V5 总暴涨股: {(df_v5["ret_pct"]>=100).sum()}')
    print(f'  NoP 独有: {len(nop_extra)}\n')

    # 但要小心 — V5 与 NoP 的"事件"含义不同 (V5 一段池子触发一次, NoP 30 日不重复)
    # 看看 NoP 独有的暴涨股 入场时 cur_retail 在哪 — 关键!
    print(f'  --- NoP 独有暴涨股的 cur_retail 分布 ---')
    if len(nop_extra) > 0:
        print(f'  cur_retail: min={nop_extra["cur_retail"].min():.0f}, '
              f'p25={nop_extra["cur_retail"].quantile(0.25):.0f}, '
              f'med={nop_extra["cur_retail"].median():.0f}, '
              f'p75={nop_extra["cur_retail"].quantile(0.75):.0f}, '
              f'max={nop_extra["cur_retail"].max():.0f}')
        # 拆桶
        print(f'\n  --- 按 cur_retail 拆桶 (NoP 独有暴涨股) ---')
        bins = [-1500, -500, -250, -100, 0, 100, 1000]
        labels = ['<-500', '-500~-250', '-250~-100', '-100~0', '0~100', '>100']
        nop_extra['cret_bin'] = pd.cut(nop_extra['cur_retail'], bins, labels=labels)
        print(f'  {"区间":<14} {"n":>6}')
        for label in labels:
            cnt = (nop_extra['cret_bin']==label).sum()
            bar = '█' * min(cnt, 50)
            print(f'  {label:<14} {cnt:>6}  {bar}')

    # ===== 关键: 把 ≥+100% 暴涨股全集合并起来, 看 cur_retail 分布 =====
    # 要把同股同日附近 (±60 日) 视为同一暴涨段, 防重
    print(f'\n{"="*82}')
    print(f'  全部 ≥+100% 暴涨股 (V5∪NoP, 防重) 的 cur_retail 分布')
    print(f'{"="*82}')

    all_b = pd.concat([df_v5[df_v5['ret_pct']>=100][['date','code','cur_retail','cur_mf','cur_trend','ret_pct']],
                        df_nop[df_nop['ret_pct']>=100][['date','code','cur_retail','cur_mf','cur_trend','ret_pct']]],
                       ignore_index=True)
    all_b['date_dt'] = pd.to_datetime(all_b['date'])
    # 防重: 同 code, 60 日内只留 ret 最大的
    all_b = all_b.sort_values(['code','date_dt'])
    keep_idx = []
    last_seen = {}
    for idx, r in all_b.iterrows():
        c = r['code']; d = r['date_dt']
        if c in last_seen and (d - last_seen[c][0]).days < 60:
            # 比较 ret, 若新 > 旧, 替换
            if r['ret_pct'] > last_seen[c][1]:
                # 移除旧的
                if last_seen[c][2] in keep_idx:
                    keep_idx.remove(last_seen[c][2])
                keep_idx.append(idx)
                last_seen[c] = (d, r['ret_pct'], idx)
            continue
        keep_idx.append(idx)
        last_seen[c] = (d, r['ret_pct'], idx)
    all_b = all_b.loc[keep_idx].reset_index(drop=True)
    print(f'\n  防重后总暴涨股: {len(all_b)}')

    # cur_retail 分布
    bins2 = [-1500, -700, -500, -350, -250, -150, -50, 50, 1500]
    labels2 = ['<-700', '-700~-500', '-500~-350', '-350~-250', '-250~-150', '-150~-50', '-50~50', '>50']
    all_b['bin'] = pd.cut(all_b['cur_retail'], bins2, labels=labels2)
    print(f'\n  {"cur_retail 区间":<16} {"n":>6}  {"占比":>7}')
    for label in labels2:
        cnt = (all_b['bin']==label).sum()
        bar = '█' * min(cnt, 60)
        pct = cnt / len(all_b) * 100
        print(f'  {label:<16} {cnt:>6}  {pct:>5.1f}%  {bar}')

    # 分位
    print(f'\n  ≥+100% 暴涨股 cur_retail 五数:')
    print(f'    min={all_b["cur_retail"].min():.0f}')
    print(f'    p10={all_b["cur_retail"].quantile(0.10):.0f}')
    print(f'    p25={all_b["cur_retail"].quantile(0.25):.0f}')
    print(f'    med={all_b["cur_retail"].median():.0f}')
    print(f'    p75={all_b["cur_retail"].quantile(0.75):.0f}')
    print(f'    p90={all_b["cur_retail"].quantile(0.90):.0f}')
    print(f'    max={all_b["cur_retail"].max():.0f}')

    pct_under_250 = (all_b['cur_retail'] < -250).mean() * 100
    pct_under_150 = (all_b['cur_retail'] < -150).mean() * 100
    pct_under_0 = (all_b['cur_retail'] < 0).mean() * 100
    print(f'\n  ≥+100% 暴涨股 入场时 retail 阈值占比:')
    print(f'    cur_retail < -250: {pct_under_250:.1f}%   (V5 入池条件实际是 pool_min<-250 但触发时 cur_retail 已上升)')
    print(f'    cur_retail < -150: {pct_under_150:.1f}%')
    print(f'    cur_retail <    0: {pct_under_0:.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
