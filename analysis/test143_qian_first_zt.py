# -*- coding: utf-8 -*-
"""test143 — 个股乾卦事件后第 1 个涨停板的表现

事件定义:
  - "乾卦事件" = 个股 gua_code=='111' 的每一段连续起点 (上一天非乾或换股)
  - 从乾卦事件起点 (含) 起, 向后找第 1 个涨停日 (close/prev_close >= 阈值)
  - 买入价 = 涨停日 close
  - 卖出: 涨停日次日起 30 个交易日内 high 的最高点 (上限收益)

输出: 整体收益分布, 按 lag (距乾卦起点天数) 分桶, 按年分桶
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'stock_bagua_daily.parquet')

ZT_THRESHOLDS = [1.095, 1.099]   # 涨停阈值, 9.5% 和 9.9% 两档
HOLD_DAYS = 30                    # 持有窗口
MAX_LAG = 60                      # 起点到涨停的最大间隔; 超过视为"非乾卦相关"


def load():
    print(f'[load] {PATH}')
    t0 = time.time()
    df = pd.read_parquet(PATH, columns=['date','code','gua_code','open','high','low','close'])
    print(f'  shape={df.shape}  耗时 {time.time()-t0:.1f}s')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    return df


def find_events(df, zt_threshold):
    """对每只票按时间扫一遍, 输出事件列表."""
    events = []
    n_codes = 0
    n_qian_seg = 0
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n_codes += 1
        n = len(sub)
        if n < 5:
            continue
        # 转 numpy (stock_bagua_daily 是 Parquet, string 列必须 to_numpy)
        g  = sub['gua_code'].to_numpy()
        c  = sub['close'].to_numpy(dtype=float)
        h  = sub['high'].to_numpy(dtype=float)
        d  = sub['date'].to_numpy()

        # 乾卦段起点
        is_q = (g == '111')
        prev_q = np.zeros(n, dtype=bool)
        prev_q[1:] = is_q[:-1]
        starts = np.where(is_q & ~prev_q)[0]
        n_qian_seg += len(starts)

        if len(starts) == 0:
            continue

        # 涨停标记 (close / prev_close >= 阈值)
        pc = np.empty(n, dtype=float)
        pc[0] = np.nan
        pc[1:] = c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt = ratio >= zt_threshold

        for s in starts:
            # 在 [s, s+MAX_LAG] 区间找第 1 个 zt
            end_search = min(s + MAX_LAG + 1, n)
            rest = np.where(zt[s:end_search])[0]
            if len(rest) == 0:
                continue
            zt_idx = s + rest[0]

            # 卖出窗口: zt_idx+1 .. zt_idx+HOLD_DAYS
            sell_end = min(zt_idx + HOLD_DAYS + 1, n)
            if sell_end <= zt_idx + 1:
                continue
            max_high = h[zt_idx+1:sell_end].max()
            buy = c[zt_idx]

            events.append({
                'code': code,
                'qian_start': d[s],
                'zt_date': d[zt_idx],
                'lag': int(zt_idx - s),
                'buy_close': float(buy),
                'max_high_30d': float(max_high),
                'ret_high': float(max_high / buy - 1.0),
            })
    print(f'  扫码 {n_codes} 只, 乾卦段 {n_qian_seg}, 命中事件 {len(events)}, 耗时 {time.time()-t0:.1f}s')
    return pd.DataFrame(events)


def find_baseline(df, zt_threshold):
    """Baseline: 全部涨停日 (不限乾卦), 同 30d max high 口径.

    每只票全部涨停日都纳入 (含连续涨停, 与乾卦事件每段 1 个不同, 但是个上限对照).
    """
    events = []
    n_codes = 0
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n_codes += 1
        n = len(sub)
        if n < 5:
            continue
        c = sub['close'].to_numpy(dtype=float)
        h = sub['high'].to_numpy(dtype=float)
        d = sub['date'].to_numpy()

        pc = np.empty(n, dtype=float)
        pc[0] = np.nan
        pc[1:] = c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt_idx_arr = np.where(ratio >= zt_threshold)[0]

        for zt_idx in zt_idx_arr:
            sell_end = min(zt_idx + HOLD_DAYS + 1, n)
            if sell_end <= zt_idx + 1:
                continue
            max_high = h[zt_idx+1:sell_end].max()
            buy = c[zt_idx]
            events.append({
                'code': code,
                'zt_date': d[zt_idx],
                'buy_close': float(buy),
                'max_high_30d': float(max_high),
                'ret_high': float(max_high / buy - 1.0),
            })
    print(f'  扫码 {n_codes} 只, baseline 涨停事件 {len(events)}, 耗时 {time.time()-t0:.1f}s')
    return pd.DataFrame(events)


def report(ev, label):
    print(f'\n=== {label} ===')
    if len(ev) == 0:
        print('  (空)')
        return
    n = len(ev)
    r = ev['ret_high']
    print(f'  事件数: {n}')
    print(f'  平均收益: {r.mean()*100:.2f}%')
    print(f'  中位收益: {r.median()*100:.2f}%')
    print(f'  胜率(>0):    {(r>0).mean()*100:.1f}%')
    print(f'  胜率(>5%):   {(r>0.05).mean()*100:.1f}%')
    print(f'  胜率(>10%):  {(r>0.10).mean()*100:.1f}%')
    print(f'  胜率(>20%):  {(r>0.20).mean()*100:.1f}%')
    print(f'  收益分位 25/50/75/90: {r.quantile(.25)*100:.2f}% / {r.quantile(.5)*100:.2f}% / '
          f'{r.quantile(.75)*100:.2f}% / {r.quantile(.9)*100:.2f}%')

    # 按 lag 分桶 (仅乾卦事件有 lag 列)
    if 'lag' in ev.columns:
        print('  按 lag (起点到涨停天数) 分桶:')
        bins = [(-1, 0, '当日涨停'),
                (0, 3, 'T+1~3'),
                (3, 7, 'T+4~7'),
                (7, 15, 'T+8~15'),
                (15, 30, 'T+16~30'),
                (30, 60, 'T+31~60')]
        for lo, hi, lab in bins:
            m = (ev['lag'] > lo) & (ev['lag'] <= hi)
            sub = ev[m]
            if len(sub) == 0:
                print(f'    {lab:8s}  N=0')
                continue
            sr = sub['ret_high']
            print(f'    {lab:8s}  N={len(sub):5d}  '
                  f'avg={sr.mean()*100:6.2f}%  win>0={(sr>0).mean()*100:5.1f}%  '
                  f'win>5={(sr>0.05).mean()*100:5.1f}%  win>10={(sr>0.10).mean()*100:5.1f}%')

    # 按年
    print('  按年:')
    ev['year'] = ev['zt_date'].astype(str).str[:4]
    for y, sub in ev.groupby('year'):
        sr = sub['ret_high']
        print(f'    {y}  N={len(sub):5d}  avg={sr.mean()*100:6.2f}%  '
              f'win>5={(sr>0.05).mean()*100:5.1f}%  win>10={(sr>0.10).mean()*100:5.1f}%')


def main():
    df = load()
    out_dir = os.path.join(ROOT, 'data_layer', 'data', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    for thr in ZT_THRESHOLDS:
        print(f'\n##### 涨停阈值 = {thr} (>={(thr-1)*100:.1f}%) #####')
        ev = find_events(df, thr)
        if len(ev) == 0:
            continue
        report(ev, f'[乾卦事件] 涨停>={(thr-1)*100:.1f}%, 持有 {HOLD_DAYS}d, lag<={MAX_LAG}d')
        out = os.path.join(out_dir, f'test143_qian_first_zt_{int((thr-1)*1000)}.parquet')
        ev.to_parquet(out, engine='pyarrow', compression='snappy')
        print(f'  写出: {out}')

        # Baseline: 全部涨停 (不限乾卦)
        bl = find_baseline(df, thr)
        report(bl, f'[Baseline 全部涨停] 涨停>={(thr-1)*100:.1f}%, 持有 {HOLD_DAYS}d')
        out_bl = os.path.join(out_dir, f'test143_baseline_zt_{int((thr-1)*1000)}.parquet')
        bl.to_parquet(out_bl, engine='pyarrow', compression='snappy')
        print(f'  写出: {out_bl}')

        # 对比: 乾卦事件 vs baseline (按年)
        print(f'\n=== 乾卦 vs Baseline 增量 (avg ret_high, 涨停>={(thr-1)*100:.1f}%) ===')
        ev['year'] = ev['zt_date'].astype(str).str[:4]
        bl['year'] = bl['zt_date'].astype(str).str[:4]
        ev_y = ev.groupby('year')['ret_high'].agg(['mean','count']).rename(columns={'mean':'qian_avg','count':'qian_n'})
        bl_y = bl.groupby('year')['ret_high'].agg(['mean','count']).rename(columns={'mean':'bl_avg','count':'bl_n'})
        cmp = ev_y.join(bl_y, how='outer').fillna(0)
        cmp['delta'] = cmp['qian_avg'] - cmp['bl_avg']
        print(f'  {"year":<6}{"qian_n":>8}{"qian_avg":>12}{"bl_n":>10}{"bl_avg":>12}{"delta":>10}')
        for y, row in cmp.iterrows():
            print(f'  {y:<6}{int(row["qian_n"]):>8}{row["qian_avg"]*100:>11.2f}%'
                  f'{int(row["bl_n"]):>10}{row["bl_avg"]*100:>11.2f}%{row["delta"]*100:>9.2f}%')
        print(f'  {"全期":<6}{len(ev):>8}{ev["ret_high"].mean()*100:>11.2f}%'
              f'{len(bl):>10}{bl["ret_high"].mean()*100:>11.2f}%'
              f'{(ev["ret_high"].mean()-bl["ret_high"].mean())*100:>9.2f}%')


if __name__ == '__main__':
    main()
