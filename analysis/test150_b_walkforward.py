# -*- coding: utf-8 -*-
"""test150 — B 类双坤底反弹规则的 walk-forward + IS/OOS 验证

目的: B 类规则全样本 T+30 +17.7% 是否切片福利?
  - 按年切 7 段 (含 2015 牛市单独)
  - 看每段 B 类 T+30 是否都正
  - IS = {2018, 2019, 2021, 2022} 跨牛熊震荡
  - OOS = {2023_24} 不参与决策, 只看是否一致

通过标准:
  - 每段 T+30 > 0 (方向一致)
  - 每段 vs baseline Δ > 0 (lift 一致)
  - IS 平均 lift ≥ +10pp 且 OOS lift ≥ +5pp
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

# 时间窗口
def window_of(date_str):
    y = int(date_str[:4])
    if y in (2014, 2015): return 'w14_15_牛末'
    if y in (2016, 2017): return 'w16_17_震荡'
    if y == 2018:         return 'w18_熊市'
    if y == 2019:         return 'w19_反弹'
    if y == 2020:         return 'w20_疫情牛'
    if y == 2021:         return 'w21_抱团'
    if y == 2022:         return 'w22_杀跌'
    if y in (2023, 2024): return 'w23_24_震荡'
    if y >= 2025:         return 'w25_26_新轮'
    return 'other'

WIN_ORDER = [
    'w14_15_牛末','w16_17_震荡','w18_熊市','w19_反弹','w20_疫情牛',
    'w21_抱团','w22_杀跌','w23_24_震荡','w25_26_新轮']

IS_WINS  = ['w18_熊市', 'w19_反弹', 'w21_抱团', 'w22_杀跌']
OOS_WIN  = 'w23_24_震荡'


def load():
    print('[load]')
    t0 = time.time()
    s = pd.read_parquet(STOCKS, columns=['code','date','close','trend','retail','main_force','gua'])
    s = s.dropna(subset=['close','retail','main_force','gua']).copy()
    s['date'] = s['date'].astype(str)
    s['gua'] = s['gua'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(3)
    s = s.sort_values(['code','date']).reset_index(drop=True)

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

        K = np.zeros(n, dtype=int)
        for i in range(n - 2, -1, -1):
            K[i] = K[i + 1] + 1 if zt[i + 1] else 0

        for t in np.where(first_zt)[0]:
            if t < 5 or t + 30 >= n: continue
            rows.append({
                'date': d[t], 'k': int(K[t]),
                'gua_t': gua[t], 'gua_prev': gua[t - 1], 'mkt_t': mkt_g[t],
                't5':  c[t + 5]  / c[t] - 1.0,
                't10': c[t + 10] / c[t] - 1.0,
                't30': c[t + 30] / c[t] - 1.0,
            })
    ev = pd.DataFrame(rows)
    ev['win'] = ev['date'].apply(window_of)
    print(f'  扫码 {n_codes}, 首板事件 {len(ev)}, 耗时 {time.time()-t0:.1f}s')
    return ev


def stats(sub):
    if len(sub) == 0:
        return dict(n=0, t30_avg=np.nan, t30_win=np.nan, ge1=np.nan, ge2=np.nan)
    return dict(
        n=len(sub),
        t30_avg=float(sub['t30'].mean()),
        t30_win=float((sub['t30'] > 0).mean()),
        ge1=float((sub['k'] >= 1).mean()),
        ge2=float((sub['k'] >= 2).mean()),
    )


def fmt_pct(x):
    if pd.isna(x): return '   --'
    return f'{x*100:6.2f}%'


def report_walk(ev, rule_mask, rule_name, baseline_name='全部首板'):
    print(f'\n{"="*112}')
    print(f'{rule_name}  vs  {baseline_name}  — 按段 walk-forward')
    print('='*112)
    head = (f'  {"段":<14}{"bl_N":>6}{"bl_T30":>10}{"bl_胜":>9}{"  | "}'
            f'{"R_N":>6}{"R_T30":>10}{"R_胜":>9}{"R_≥2连":>10}{"R_≥3连":>10}{"ΔT30":>10}')
    print(head)
    print('-'*112)
    is_lifts, oos_lift = [], None
    for w in WIN_ORDER:
        bl = stats(ev[ev['win'] == w])
        rl = stats(ev[(ev['win'] == w) & rule_mask])
        if bl['n'] == 0: continue
        delta = (rl['t30_avg'] - bl['t30_avg']) if not pd.isna(rl['t30_avg']) else np.nan
        tag = ''
        if w in IS_WINS:
            tag = ' [IS]'
            if not pd.isna(delta): is_lifts.append(delta)
        elif w == OOS_WIN:
            tag = ' [OOS]'
            if not pd.isna(delta): oos_lift = delta
        print(f'  {w:<14}{bl["n"]:>6}{fmt_pct(bl["t30_avg"]):>10}{fmt_pct(bl["t30_win"]):>9}'
              f'   | {rl["n"]:>5}{fmt_pct(rl["t30_avg"]):>10}{fmt_pct(rl["t30_win"]):>9}'
              f'{fmt_pct(rl["ge1"]):>10}{fmt_pct(rl["ge2"]):>10}'
              f'{(delta*100 if not pd.isna(delta) else 0):>+8.2f}pp{tag}')
    # 总结
    is_avg = np.mean(is_lifts) if is_lifts else np.nan
    print('-'*112)
    print(f'  IS 平均 ΔT30 = {is_avg*100:+.2f}pp ({len(is_lifts)} 段)')
    if oos_lift is not None:
        print(f'  OOS ΔT30     = {oos_lift*100:+.2f}pp')

    # 通过判定
    pos_segs  = sum(1 for x in is_lifts if x > 0)
    pass_dir  = pos_segs == len(is_lifts) and (oos_lift is None or oos_lift > 0)
    pass_size = (not pd.isna(is_avg) and is_avg >= 0.10) and (oos_lift is None or oos_lift >= 0.05)
    print(f'  方向一致 (每段 lift>0): {"✅" if pass_dir else "❌"}  ({pos_segs}/{len(is_lifts)} IS 段)')
    print(f'  幅度达标 (IS ≥+10pp, OOS ≥+5pp): {"✅" if pass_size else "❌"}')


def main():
    df = load()
    print('\n[scan] 找首板事件 + T+30 收益')
    ev = scan(df)
    if len(ev) == 0:
        print('无事件'); return

    # 段分布
    print('\n[段分布]')
    for w in WIN_ORDER:
        n = (ev['win'] == w).sum()
        if n: print(f'  {w}: N={n}')

    # 三个 B 规则
    rules = [
        ('B1 双坤底', (ev['gua_t']=='000') & (ev['mkt_t']=='000')),
        ('B2 双坤底+前夜也坤', (ev['gua_t']=='000') & (ev['mkt_t']=='000') & (ev['gua_prev']=='000')),
        ('B3 单·个股坤', ev['gua_t']=='000'),
    ]
    for name, mask in rules:
        report_walk(ev, mask, name)


if __name__ == '__main__':
    main()
