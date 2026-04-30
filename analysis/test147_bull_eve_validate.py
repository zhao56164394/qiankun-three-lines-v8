# -*- coding: utf-8 -*-
"""test147 — 牛股前夜规则的真实命中率 + 持有收益验证

基于 test146 指纹结果, 检验候选规则:
  R0  全市场 baseline
  R1  兑日 (d_gua='011')
  R2  R1 + main_force > 20
  R3  R2 + retail < -50
  R4  R2 + 大盘日卦 ∈ {010坎, 011兑, 111乾}
  R5  R4 + retail < -50
  R6  R4 + retail < -100
  R7  R4 + retail < -250

每条规则算:
  - 事件数 N
  - 主升触发率 (T+1 就开启 d_gua='111' 连续 ≥10 日的比例)
  - 30 日内触发率 (T+1..T+30 内出现主升浪段起点)
  - T+30 close 持有平均收益
  - T+30 胜率
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCKS = os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet')
MARKET = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')

MIN_RUN = 10        # 主升浪连续乾卦门槛
HOLD = 30           # 持有窗口
FUT_WIN = 30        # 30 日内触发判定窗口


def load():
    print('[load]')
    t0 = time.time()
    s = pd.read_parquet(STOCKS, columns=['code','date','close','trend','retail','main_force','gua'])
    s = s.dropna(subset=['close','retail','main_force','gua']).copy()
    s['date'] = s['date'].astype(str)
    # gua 是 '0.0' / '11.0' / '111.0' 浮点字符串 → 标准化为 3 位 ('000','011','111')
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


def aggregate(df, rules):
    agg = {name: {'n':0, 'eve_n':0, 'fut_n':0, 'ret_sum':0.0, 'win_n':0}
           for name, _ in rules}
    n_codes = 0
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n_codes += 1
        n = len(sub)
        if n < HOLD + 5:
            continue
        # 转 numpy (Parquet → to_numpy)
        gua    = sub['gua'].to_numpy()
        retail = sub['retail'].to_numpy(dtype=float)
        mf     = sub['main_force'].to_numpy(dtype=float)
        mkt_dg = sub['mkt_d_gua'].astype(str).to_numpy()
        c      = sub['close'].to_numpy(dtype=float)

        # 主升浪段起点: d_gua='111' 连续 >= MIN_RUN
        is_q = (gua == '111')
        prev_q = np.zeros(n, dtype=bool); prev_q[1:] = is_q[:-1]
        next_q = np.zeros(n, dtype=bool); next_q[:-1] = is_q[1:]
        starts = np.where(is_q & ~prev_q)[0]
        ends   = np.where(is_q & ~next_q)[0]
        L = min(len(starts), len(ends))
        bull_start = np.zeros(n, dtype=bool)
        for i in range(L):
            s, e = starts[i], ends[i]
            if e - s + 1 >= MIN_RUN:
                bull_start[s] = True

        # is_eve: t+1 是 bull_start
        is_eve = np.zeros(n, dtype=bool)
        is_eve[:-1] = bull_start[1:]

        # future_trigger: t+1..t+FUT_WIN 内有 bull_start (用累积和)
        bs_int = bull_start.astype(np.int32)
        csum = np.concatenate([[0], np.cumsum(bs_int)])
        idx = np.arange(n)
        end_idx = np.minimum(idx + 1 + FUT_WIN, n)
        start_idx = np.minimum(idx + 1, n)
        fut = (csum[end_idx] - csum[start_idx]) > 0

        # T+30 收益
        ret = np.full(n, np.nan)
        ret[:n-HOLD] = c[HOLD:] / c[:n-HOLD] - 1.0
        valid = ~np.isnan(ret) & (idx < n - HOLD)

        arrs = {'gua':gua, 'retail':retail, 'main_force':mf, 'mkt_d_gua':mkt_dg}

        for name, mask_fn in rules:
            mask = mask_fn(arrs) & valid
            cnt = int(mask.sum())
            if cnt == 0: continue
            agg[name]['n']       += cnt
            agg[name]['eve_n']   += int((mask & is_eve).sum())
            agg[name]['fut_n']   += int((mask & fut).sum())
            agg[name]['ret_sum'] += float(ret[mask].sum())
            agg[name]['win_n']   += int(((ret > 0) & mask).sum())

    print(f'  扫码 {n_codes}, 耗时 {time.time()-t0:.1f}s')
    return agg


def report(agg, rules, baseline_name='R0_全市场baseline'):
    bl = agg[baseline_name]
    bl_eve_rate = bl['eve_n'] / bl['n'] if bl['n'] else 0
    bl_fut_rate = bl['fut_n'] / bl['n'] if bl['n'] else 0
    bl_avg = bl['ret_sum'] / bl['n'] if bl['n'] else 0
    bl_win = bl['win_n'] / bl['n'] if bl['n'] else 0

    print('\n' + '='*108)
    print(f'{"规则":<32}{"N":>10}{"主升次日率":>13}{"30d触发率":>12}{"T+30 avg":>11}{"胜率":>9}{"vs bl Δavg":>13}')
    print('='*108)
    for name, _ in rules:
        a = agg[name]
        if a['n'] == 0:
            print(f'  {name:<30}  N=0'); continue
        eve = a['eve_n'] / a['n']
        fut = a['fut_n'] / a['n']
        avg = a['ret_sum'] / a['n']
        win = a['win_n'] / a['n']
        delta = (avg - bl_avg) * 100
        eve_x = eve / bl_eve_rate if bl_eve_rate > 0 else 0
        fut_x = fut / bl_fut_rate if bl_fut_rate > 0 else 0
        print(f'  {name:<30}{a["n"]:>10}'
              f'{eve*100:>11.2f}% ({eve_x:.0f}x)'
              f'{fut*100:>9.1f}%'
              f'{avg*100:>10.2f}%'
              f'{win*100:>8.1f}%'
              f'{delta:>+11.2f}pp')


def main():
    df = load()

    # main_force / retail 分布参考
    print('\n[main_force / retail 分布参考]')
    print(f'  main_force: median={df["main_force"].median():.1f}, '
          f'q25={df["main_force"].quantile(.25):.1f}, q75={df["main_force"].quantile(.75):.1f}')
    print(f'  retail:     median={df["retail"].median():.1f}, '
          f'q25={df["retail"].quantile(.25):.1f}, q75={df["retail"].quantile(.75):.1f}')

    rules = [
        ('R0_全市场baseline',
            lambda a: np.ones(len(a['gua']), dtype=bool)),
        ('R1_兑日(011)',
            lambda a: a['gua']=='011'),
        ('R2_R1+mf>20',
            lambda a: (a['gua']=='011') & (a['main_force']>20)),
        ('R3_R2+retail<-50',
            lambda a: (a['gua']=='011') & (a['main_force']>20) & (a['retail']<-50)),
        ('R4_R2+大盘∈{坎兑乾}',
            lambda a: (a['gua']=='011') & (a['main_force']>20)
                      & np.isin(a['mkt_d_gua'], ['010','011','111'])),
        ('R5_R4+retail<-50',
            lambda a: (a['gua']=='011') & (a['main_force']>20)
                      & np.isin(a['mkt_d_gua'], ['010','011','111'])
                      & (a['retail']<-50)),
        ('R6_R4+retail<-100',
            lambda a: (a['gua']=='011') & (a['main_force']>20)
                      & np.isin(a['mkt_d_gua'], ['010','011','111'])
                      & (a['retail']<-100)),
        ('R7_R4+retail<-250',
            lambda a: (a['gua']=='011') & (a['main_force']>20)
                      & np.isin(a['mkt_d_gua'], ['010','011','111'])
                      & (a['retail']<-250)),
        # 严格门槛: 项目里散户线<-250 是入池硬条件
        ('R8_单retail<-250',
            lambda a: a['retail']<-250),
        ('R9_兑日+retail<-250',
            lambda a: (a['gua']=='011') & (a['retail']<-250)),
        ('R10_仅mf>50+retail<-150',
            lambda a: (a['main_force']>50) & (a['retail']<-150)),
    ]

    agg = aggregate(df, rules)

    report(agg, rules)

    print('\n[图例]')
    print('  N            = 满足规则且有 T+30 数据的事件数')
    print('  主升次日率    = 入场日 T 后, T+1 就开启 d_gua=111 连续 ≥10 日的比例 (Nx = vs baseline 倍数)')
    print('  30d触发率    = T+1..T+30 内任意时刻开启主升浪的比例')
    print('  T+30 avg     = 入场日 close 买入, T+30 close 卖出的平均收益 (无 lookahead)')
    print('  胜率          = T+30 收益 > 0 的比例')
    print('  Δavg vs bl   = 跟全市场 baseline 的 T+30 avg 差值')


if __name__ == '__main__':
    main()
