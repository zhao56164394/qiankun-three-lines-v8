# -*- coding: utf-8 -*-
"""test145 — 各卦后第 1 个涨停板之后的真实价格路径 (是否持续上涨?)

之前 test144 用 30d max high (lookahead) 判定收益, 看不出"是不是一直涨".
这次换口径:
  - T+1/5/10/20/30 收盘价收益 (真实持有 N 日的结果)
  - 期间最低收盘 (min_close): 看是否曾跌破买入价
  - 期间最低 low (min_low): 看盘中最大回撤
  - 期间最高 high (max_high): 上限收益, 跟之前对照

回答的问题:
  - 乾卦后第 1 个涨停, 涨停后 close 是否单调向上? (T+1 < T+5 < T+10 < T+20 < T+30 ?)
  - 多少比例的事件 T+30 收盘还在买入价之上?
  - 跟 baseline (全部涨停板) 比, 路径形态有没有更优?
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'stock_bagua_daily.parquet')

ZT = 1.099
MAX_LAG = 60
HOLDS = [1, 5, 10, 20, 30]

GUAS = [
    ('111', '乾'), ('110', '巽'), ('101', '离'), ('100', '艮'),
    ('011', '兑'), ('010', '坎'), ('001', '震'), ('000', '坤'),
]


def load():
    print(f'[load] {PATH}')
    t0 = time.time()
    df = pd.read_parquet(PATH, columns=['date','code','gua_code','open','high','low','close'])
    df = df.sort_values(['code','date']).reset_index(drop=True)
    print(f'  shape={df.shape}  耗时 {time.time()-t0:.1f}s')
    return df


def scan(df, target_gua=None):
    rets = {f't{t}': [] for t in HOLDS}
    rets['max_high'] = []
    rets['min_close'] = []
    rets['min_low']   = []
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n = len(sub)
        if n < 35: continue
        g = sub['gua_code'].to_numpy()
        c = sub['close'].to_numpy(dtype=float)
        h = sub['high'].to_numpy(dtype=float)
        l = sub['low'].to_numpy(dtype=float)

        pc = np.empty(n, dtype=float); pc[0]=np.nan; pc[1:]=c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt = ratio >= ZT

        if target_gua is None:
            zt_idxs = np.where(zt)[0]
        else:
            is_g = (g == target_gua)
            prev_g = np.zeros(n, dtype=bool); prev_g[1:] = is_g[:-1]
            starts = np.where(is_g & ~prev_g)[0]
            zt_idxs = []
            for s in starts:
                e = min(s + MAX_LAG + 1, n)
                rest = np.where(zt[s:e])[0]
                if len(rest) == 0: continue
                zt_idxs.append(s + rest[0])
            zt_idxs = np.array(zt_idxs, dtype=int)

        for zt_idx in zt_idxs:
            buy = c[zt_idx]
            if buy <= 0: continue
            # 必须有完整 30d 后续, 不然丢弃 (避免短样本拖累统计)
            if zt_idx + 30 >= n:
                continue
            for t in HOLDS:
                rets[f't{t}'].append(c[zt_idx + t] / buy - 1.0)
            rets['max_high'].append(h[zt_idx+1:zt_idx+31].max() / buy - 1.0)
            rets['min_close'].append(c[zt_idx+1:zt_idx+31].min() / buy - 1.0)
            rets['min_low'].append(l[zt_idx+1:zt_idx+31].min() / buy - 1.0)

    n_ev = len(rets['t1'])
    print(f'  事件数 {n_ev}, 耗时 {time.time()-t0:.1f}s')
    return {k: np.array(v, dtype=float) for k, v in rets.items()}, n_ev


def stats(d, n_ev):
    s = {'n': n_ev}
    for t in HOLDS:
        a = d[f't{t}']
        s[f't{t}_avg'] = float(a.mean()) if len(a) else 0
        s[f't{t}_win'] = float((a > 0).mean()) if len(a) else 0
    s['max_high_avg'] = float(d['max_high'].mean()) if len(d['max_high']) else 0
    s['min_close_avg'] = float(d['min_close'].mean()) if len(d['min_close']) else 0
    s['min_low_avg'] = float(d['min_low'].mean()) if len(d['min_low']) else 0
    # 期间是否曾跌破买入价
    s['ever_below'] = float((d['min_close'] < 0).mean()) if len(d['min_close']) else 0
    return s


def fmt_pct(x): return f'{x*100:6.2f}%'
def fmt_pct_s(x): return f'{x*100:5.1f}%'


def print_path(label, s):
    print(f'\n[{label}]  N={s["n"]}')
    print(f'  T+1   close avg={fmt_pct(s["t1_avg"])}  win>0={fmt_pct_s(s["t1_win"])}')
    print(f'  T+5   close avg={fmt_pct(s["t5_avg"])}  win>0={fmt_pct_s(s["t5_win"])}')
    print(f'  T+10  close avg={fmt_pct(s["t10_avg"])}  win>0={fmt_pct_s(s["t10_win"])}')
    print(f'  T+20  close avg={fmt_pct(s["t20_avg"])}  win>0={fmt_pct_s(s["t20_win"])}')
    print(f'  T+30  close avg={fmt_pct(s["t30_avg"])}  win>0={fmt_pct_s(s["t30_win"])}')
    print(f'  期间最高 high avg={fmt_pct(s["max_high_avg"])}')
    print(f'  期间最低 close avg={fmt_pct(s["min_close_avg"])}  曾跌破买入价比例={fmt_pct_s(s["ever_below"])}')
    print(f'  期间最低 low   avg={fmt_pct(s["min_low_avg"])}')


def main():
    df = load()

    print(f'\n========== Baseline 全部涨停板 ==========')
    bl_d, bl_n = scan(df, target_gua=None)
    bl_s = stats(bl_d, bl_n)
    print_path('Baseline', bl_s)

    rows = []
    for code, name in GUAS:
        print(f'\n========== {name}卦 ({code}) ==========')
        d, n = scan(df, target_gua=code)
        s = stats(d, n); s['gua'] = name; s['code'] = code
        print_path(f'{name}卦', s)
        rows.append(s)

    # 横向汇总: T+N close 平均收益
    print('\n' + '='*100)
    print('8 卦 vs Baseline — T+N 持有收盘价收益对比 (单调上升 ↔ T+1<T+5<T+10<T+20<T+30)')
    print('='*100)
    print(f'  {"卦":<4}{"code":<6}{"N":>8}'
          f'{"T+1":>8}{"T+5":>9}{"T+10":>9}{"T+20":>9}{"T+30":>9}'
          f'{"min_cls":>9}{"曾破":>7}{"max_h":>9}')
    print('-'*100)
    for s in sorted(rows, key=lambda x: -x['t30_avg']):
        print(f'  {s["gua"]:<4}{s["code"]:<6}{s["n"]:>8}'
              f'{fmt_pct(s["t1_avg"])}{fmt_pct(s["t5_avg"])}{fmt_pct(s["t10_avg"])}'
              f'{fmt_pct(s["t20_avg"])}{fmt_pct(s["t30_avg"])}'
              f'{fmt_pct(s["min_close_avg"])}{fmt_pct_s(s["ever_below"])}'
              f'{fmt_pct(s["max_high_avg"])}')
    print('-'*100)
    print(f'  {"baseline":<10}{bl_s["n"]:>8}'
          f'{fmt_pct(bl_s["t1_avg"])}{fmt_pct(bl_s["t5_avg"])}{fmt_pct(bl_s["t10_avg"])}'
          f'{fmt_pct(bl_s["t20_avg"])}{fmt_pct(bl_s["t30_avg"])}'
          f'{fmt_pct(bl_s["min_close_avg"])}{fmt_pct_s(bl_s["ever_below"])}'
          f'{fmt_pct(bl_s["max_high_avg"])}')

    print('\n说明:')
    print('  - T+N close avg = 涨停日收盘买入, 持有 N 日的收盘价平均收益')
    print('  - 单调上升 ↔ T+1 < T+5 < T+10 < T+20 < T+30 (后面比前面持续走高)')
    print('  - min_close = 涨停日次日起 30d 内的最低收盘价 / 买入价 - 1')
    print('  - 曾破 = 30d 内某天收盘价 < 买入价的事件比例')


if __name__ == '__main__':
    main()
