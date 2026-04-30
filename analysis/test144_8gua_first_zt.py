# -*- coding: utf-8 -*-
"""test144 — 个股 8 卦事件后第 1 个涨停板的表现 (横向对比)

事件: gua_code==X 段起点 → 起点(含)起 lag<=60d 内第 1 个涨停 (close/prev_close >= 1.099)
口径: 涨停日 close 买入, 之后 30d 内 high 最高点卖 (lookahead 上限收益)
对比: 跟 "全部涨停板" baseline 算 delta
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'stock_bagua_daily.parquet')

ZT = 1.099
HOLD = 30
MAX_LAG = 60

# 8 卦: code -> 名
GUAS = [
    ('111', '乾'),
    ('110', '巽'),
    ('101', '离'),
    ('100', '艮'),
    ('011', '兑'),
    ('010', '坎'),
    ('001', '震'),
    ('000', '坤'),
]


def load():
    print(f'[load] {PATH}')
    t0 = time.time()
    df = pd.read_parquet(PATH, columns=['date','code','gua_code','high','close'])
    df = df.sort_values(['code','date']).reset_index(drop=True)
    print(f'  shape={df.shape}  耗时 {time.time()-t0:.1f}s')
    return df


def scan(df, target_gua=None):
    """target_gua=None → baseline (全部涨停日)
       target_gua='111' → 该卦段起点起 lag<=60 内第 1 个涨停
    """
    rets = []
    t0 = time.time()
    for code, sub in df.groupby('code', sort=False):
        n = len(sub)
        if n < 5: continue
        g = sub['gua_code'].to_numpy()
        c = sub['close'].to_numpy(dtype=float)
        h = sub['high'].to_numpy(dtype=float)

        pc = np.empty(n, dtype=float); pc[0]=np.nan; pc[1:]=c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt = ratio >= ZT

        if target_gua is None:
            zt_idxs = np.where(zt)[0]
        else:
            is_g = (g == target_gua)
            prev_g = np.zeros(n, dtype=bool)
            prev_g[1:] = is_g[:-1]
            starts = np.where(is_g & ~prev_g)[0]
            zt_idxs = []
            for s in starts:
                e = min(s + MAX_LAG + 1, n)
                rest = np.where(zt[s:e])[0]
                if len(rest) == 0: continue
                zt_idxs.append(s + rest[0])
            zt_idxs = np.array(zt_idxs, dtype=int)

        for zt_idx in zt_idxs:
            sell_end = min(zt_idx + HOLD + 1, n)
            if sell_end <= zt_idx + 1: continue
            max_h = h[zt_idx+1:sell_end].max()
            buy = c[zt_idx]
            rets.append(max_h / buy - 1.0)

    arr = np.array(rets, dtype=float)
    print(f'  扫完, 事件数 {len(arr)}, 耗时 {time.time()-t0:.1f}s')
    return arr


def stats(arr):
    if len(arr) == 0:
        return dict(n=0, avg=0, med=0, win5=0, win10=0, win20=0)
    return dict(
        n=len(arr),
        avg=float(arr.mean()),
        med=float(np.median(arr)),
        win5=float((arr>0.05).mean()),
        win10=float((arr>0.10).mean()),
        win20=float((arr>0.20).mean()),
    )


def main():
    df = load()

    print(f'\n[baseline] 全部涨停板 (>={(ZT-1)*100:.1f}%, hold {HOLD}d)')
    bl = scan(df, target_gua=None)
    bl_s = stats(bl)
    print(f'  n={bl_s["n"]}, avg={bl_s["avg"]*100:.2f}%, med={bl_s["med"]*100:.2f}%, '
          f'win>5={bl_s["win5"]*100:.1f}%, win>10={bl_s["win10"]*100:.1f}%, win>20={bl_s["win20"]*100:.1f}%')

    rows = []
    for code, name in GUAS:
        print(f'\n[{name}卦 {code}]')
        arr = scan(df, target_gua=code)
        s = stats(arr)
        s['gua'] = name; s['code'] = code
        s['delta_avg'] = s['avg'] - bl_s['avg']
        s['delta_win5'] = s['win5'] - bl_s['win5']
        s['delta_win10'] = s['win10'] - bl_s['win10']
        rows.append(s)

    print('\n' + '='*88)
    print('8 卦事件后第 1 个涨停板 vs Baseline 全部涨停 (涨停>=9.9%, 30d max high)')
    print('='*88)
    print(f'  {"卦":<4}{"code":<6}{"事件数":>8}{"avg":>10}{"med":>10}{"win>5%":>10}{"win>10%":>10}'
          f'{"Δavg":>10}{"Δwin5":>9}{"Δwin10":>10}')
    print('-'*88)
    for s in sorted(rows, key=lambda x: -x['delta_avg']):
        print(f'  {s["gua"]:<4}{s["code"]:<6}{s["n"]:>8}'
              f'{s["avg"]*100:>9.2f}%{s["med"]*100:>9.2f}%'
              f'{s["win5"]*100:>9.1f}%{s["win10"]*100:>9.1f}%'
              f'{s["delta_avg"]*100:>9.2f}%{s["delta_win5"]*100:>8.1f}%{s["delta_win10"]*100:>9.1f}%')
    print('-'*88)
    print(f'  {"baseline":<10}{bl_s["n"]:>8}'
          f'{bl_s["avg"]*100:>9.2f}%{bl_s["med"]*100:>9.2f}%'
          f'{bl_s["win5"]*100:>9.1f}%{bl_s["win10"]*100:>9.1f}%')


if __name__ == '__main__':
    main()
