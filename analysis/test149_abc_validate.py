# -*- coding: utf-8 -*-
"""test149 — A/B/C 三类首板牛股规则验证

A 顺势二连型: 个股兑(011) + 大盘乾(111) + 前夜散户/主力线高
B 双坤底反弹: 个股坤(000) + 大盘坤(000)
C 趋势加速型: 个股乾(111) + trend > 60

每类算:
  - N 事件数
  - K 分布 (K=0/=1/=2/=3/≥4)
  - 连板率 (≥2连/≥3连/≥4连/≥5连)
  - T+5 / T+10 / T+30 close 持有收益 (无 lookahead)
  - T+30 胜率

跟 baseline (全部首板) 对比 lift.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCKS = os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet')
MARKET = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')

ZT = 1.099
HOLDS = [5, 10, 30]


def load():
    print('[load]')
    t0 = time.time()
    s = pd.read_parquet(STOCKS, columns=['code','date','close','trend','retail','main_force','gua'])
    s = s.dropna(subset=['close','retail','main_force','gua']).copy()
    s['date'] = s['date'].astype(str)
    s['gua'] = s['gua'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(3)
    s = s.sort_values(['code','date']).reset_index(drop=True)
    print(f'  stocks {s.shape}')

    mkt = pd.read_parquet(MARKET, columns=['date','d_gua'])
    mkt['date'] = mkt['date'].astype(str)
    mkt['d_gua'] = mkt['d_gua'].astype(str).str.zfill(3)
    mkt = mkt.rename(columns={'d_gua':'mkt_d_gua'})
    s = s.merge(mkt, on='date', how='left')
    print(f'  merged {s.shape}, 耗时 {time.time()-t0:.1f}s')
    return s


def scan(df):
    rows = []
    n_codes = 0
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n_codes += 1
        n = len(sub)
        if n < 40: continue
        c     = sub['close'].to_numpy(dtype=float)
        retail= sub['retail'].to_numpy(dtype=float)
        mf    = sub['main_force'].to_numpy(dtype=float)
        trend = sub['trend'].to_numpy(dtype=float)
        gua   = sub['gua'].to_numpy()
        mkt_g = sub['mkt_d_gua'].astype(str).to_numpy()
        d     = sub['date'].to_numpy()

        pc = np.empty(n); pc[0]=np.nan; pc[1:]=c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt = ratio >= ZT
        zt[0] = False
        prev_zt = np.zeros(n, dtype=bool); prev_zt[1:] = zt[:-1]
        first_zt = zt & ~prev_zt

        # 后续连板天数 K (从 T+1 起连续 zt 的天数)
        K = np.zeros(n, dtype=int)
        for i in range(n - 2, -1, -1):
            K[i] = K[i + 1] + 1 if zt[i + 1] else 0

        for t in np.where(first_zt)[0]:
            if t < 5 or t + 30 >= n: continue
            row = {
                'code': code, 'date': d[t], 'k': int(K[t]),
                'gua_t': gua[t], 'gua_prev': gua[t - 1],
                'mkt_t': mkt_g[t],
                'retail_t': retail[t], 'retail_prev': retail[t - 1],
                'mf_t': mf[t], 'mf_prev': mf[t - 1],
                'trend_t': trend[t],
            }
            for h in HOLDS:
                row[f't{h}'] = c[t + h] / c[t] - 1.0
            rows.append(row)
    print(f'  扫码 {n_codes}, 首板事件 {len(rows)}, 耗时 {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


def stats(sub, name):
    n = len(sub)
    if n == 0:
        return None
    return {
        'name': name,
        'n': n,
        'k0_pct':   (sub['k'] == 0).mean(),
        'ge1_pct':  (sub['k'] >= 1).mean(),
        'ge2_pct':  (sub['k'] >= 2).mean(),
        'ge3_pct':  (sub['k'] >= 3).mean(),
        'ge4_pct':  (sub['k'] >= 4).mean(),
        'k_avg':    sub['k'].mean(),
        't5_avg':   sub['t5'].mean(),
        't10_avg':  sub['t10'].mean(),
        't30_avg':  sub['t30'].mean(),
        't30_win':  (sub['t30'] > 0).mean(),
        't30_med':  sub['t30'].median(),
    }


def fmt(x, pct=True): return f'{x*100:6.2f}%' if pct else f'{x:6.2f}'


def report(rows, baseline):
    bl = baseline
    print('\n' + '='*120)
    print(f'{"规则":<28}{"N":>8}{"avg连板":>10}'
          f'{"≥2连":>9}{"≥3连":>9}{"≥4连":>9}{"≥5连":>9}'
          f'{"T+5":>9}{"T+10":>9}{"T+30":>9}{"胜率T30":>9}{"ΔT30":>9}')
    print('='*120)
    print(f'  {"baseline 全部首板":<26}{bl["n"]:>8}'
          f'{bl["k_avg"]:>9.2f}'
          f'{fmt(bl["ge1_pct"]):>10}{fmt(bl["ge2_pct"]):>10}{fmt(bl["ge3_pct"]):>10}{fmt(bl["ge4_pct"]):>10}'
          f'{fmt(bl["t5_avg"]):>10}{fmt(bl["t10_avg"]):>10}{fmt(bl["t30_avg"]):>10}{fmt(bl["t30_win"]):>10}'
          f'  {"--":>7}')
    print('-'*120)
    for r in rows:
        if r is None: continue
        delta_t30 = (r['t30_avg'] - bl['t30_avg']) * 100
        print(f'  {r["name"]:<26}{r["n"]:>8}'
              f'{r["k_avg"]:>9.2f}'
              f'{fmt(r["ge1_pct"]):>10}{fmt(r["ge2_pct"]):>10}{fmt(r["ge3_pct"]):>10}{fmt(r["ge4_pct"]):>10}'
              f'{fmt(r["t5_avg"]):>10}{fmt(r["t10_avg"]):>10}{fmt(r["t30_avg"]):>10}{fmt(r["t30_win"]):>10}'
              f'{delta_t30:>+8.2f}pp')


def main():
    df = load()
    print('\n[scan] 找首板事件 + 后续连板数 + T+N 收益')
    ev = scan(df)
    if len(ev) == 0:
        print('无事件'); return

    # baseline
    bl = stats(ev, 'baseline')
    print(f'\nbaseline: N={bl["n"]}, avg连板={bl["k_avg"]:.2f}, 二连率={bl["ge1_pct"]*100:.1f}%, '
          f'T+30 avg={bl["t30_avg"]*100:.2f}%')

    # A / B / C 规则
    rules = [
        # A 类
        ('A1 顺势二连 严版',
            (ev['gua_t']=='011') & (ev['mkt_t']=='111')
            & (ev['retail_prev']>50) & (ev['mf_prev']>30)),
        ('A2 顺势二连 (无主散门槛)',
            (ev['gua_t']=='011') & (ev['mkt_t']=='111')),
        ('A3 兑日 (无大盘门槛)',
            (ev['gua_t']=='011')),
        # B 类
        ('B1 双坤底 严版',
            (ev['gua_t']=='000') & (ev['mkt_t']=='000')),
        ('B2 双坤底 + 前夜也坤',
            (ev['gua_t']=='000') & (ev['mkt_t']=='000') & (ev['gua_prev']=='000')),
        ('B3 单·个股坤',
            (ev['gua_t']=='000')),
        # C 类
        ('C1 个股乾 + trend>60',
            (ev['gua_t']=='111') & (ev['trend_t']>60)),
        ('C2 个股乾 + 前夜也乾',
            (ev['gua_t']=='111') & (ev['gua_prev']=='111')),
        ('C3 个股乾 + 前夜乾或巽',
            (ev['gua_t']=='111') & ev['gua_prev'].isin(['111','110'])),
    ]

    rows = []
    for name, mask in rules:
        rows.append(stats(ev[mask], name))

    report(rows, bl)

    # 加一个总结: 三类各自最优
    print('\n[小结] 三类规则横向比较 (按 T+30 avg 收益排序)')
    valid = [r for r in rows if r is not None]
    valid.sort(key=lambda x: -x['t30_avg'])
    for i, r in enumerate(valid, 1):
        delta = (r['t30_avg'] - bl['t30_avg']) * 100
        print(f'  {i}. {r["name"]:<26} N={r["n"]:>6}  T+30={r["t30_avg"]*100:+6.2f}% '
              f'(Δ{delta:+.2f}pp)  ≥2连率={r["ge1_pct"]*100:.1f}%')


if __name__ == '__main__':
    main()
