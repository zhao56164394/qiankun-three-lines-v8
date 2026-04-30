# -*- coding: utf-8 -*-
"""test148 — 涨停之后连续涨停的牛股特征

事件: 首板日 T (T 涨停, T-1 未涨停)
标签: 后续连板天数 K (从 T+1 起连续涨停的天数, K=0 即单板炸板)

分组:
  K=0 单板炸板        — 涨停一次就破
  K=1 二连 (T,T+1)
  K=2 三连
  K=3 四连
  K≥4 五连+ (大牛)

对每组分析当日 (T) 的特征:
  卦象: 个股日卦 / 大盘日卦
  数值: retail / main_force / trend / 前 5 日累计涨幅
  前夜: T-1 日个股卦 / retail
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
GUA_NAME = {'111':'乾','110':'巽','101':'离','100':'艮',
            '011':'兑','010':'坎','001':'震','000':'坤'}


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
        if n < 10: continue
        c     = sub['close'].to_numpy(dtype=float)
        retail= sub['retail'].to_numpy(dtype=float)
        mf    = sub['main_force'].to_numpy(dtype=float)
        trend = sub['trend'].to_numpy(dtype=float)
        gua   = sub['gua'].to_numpy()
        mkt_g = sub['mkt_d_gua'].astype(str).to_numpy()
        d     = sub['date'].to_numpy()

        pc = np.empty(n); pc[0] = np.nan; pc[1:] = c[:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = c / pc
        zt = ratio >= ZT
        zt[0] = False

        # 首板 = zt[t] & ~zt[t-1]
        prev_zt = np.zeros(n, dtype=bool); prev_zt[1:] = zt[:-1]
        first_zt = zt & ~prev_zt

        # 计算每个 t 之后连续涨停的天数 K (向量化: 反向累计 reset)
        # K[t] = 从 T+1 起连续 zt 的天数. 用 reverse iter:
        K = np.zeros(n, dtype=int)
        for i in range(n - 2, -1, -1):
            if zt[i + 1]:
                K[i] = K[i + 1] + 1
            else:
                K[i] = 0

        first_idxs = np.where(first_zt)[0]
        for t in first_idxs:
            if t < 5: continue  # 需要前 5 日数据
            ret_5d = c[t] / c[t - 5] - 1.0
            rows.append({
                'code': code, 'date': d[t],
                'k': int(K[t]),
                'gua_t': gua[t],
                'retail_t': retail[t],
                'mf_t': mf[t],
                'trend_t': trend[t],
                'mkt_t': mkt_g[t],
                'gua_prev': gua[t - 1],
                'retail_prev': retail[t - 1],
                'mf_prev': mf[t - 1],
                'ret_5d': ret_5d,
            })
    print(f'  扫码 {n_codes}, 首板事件 {len(rows)}, 耗时 {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


def make_groups(ev):
    bins = [
        ('K=0 单板炸板',  ev['k'] == 0),
        ('K=1 二连板',    ev['k'] == 1),
        ('K=2 三连板',    ev['k'] == 2),
        ('K=3 四连板',    ev['k'] == 3),
        ('K≥4 五连+',     ev['k'] >= 4),
    ]
    return bins


def cat_table(ev, bins, col, label):
    print(f'\n--- {label} ({col}) ---  各组占比 (单位 %)')
    head = f'  {"卦":<4}{"全样":>9}'
    for name, _ in bins:
        head += f'{name:>14}'
    print(head)
    all_dist = ev[col].astype(str).value_counts(normalize=True) * 100
    for code in ['111','110','101','100','011','010','001','000']:
        nm = GUA_NAME.get(code, code)
        line = f'  {code} {nm}{all_dist.get(code, 0):>8.1f}'
        for name, mask in bins:
            sub = ev[mask][col].astype(str).value_counts(normalize=True) * 100
            line += f'{sub.get(code, 0):>13.1f}'
        print(line)


def num_table(ev, bins, cols):
    print(f'\n--- 数值特征中位数 (各组对比) ---')
    head = f'  {"字段":<14}{"全样":>10}'
    for name, _ in bins:
        head += f'{name:>14}'
    print(head)
    for col in cols:
        line = f'  {col:<14}{ev[col].median():>10.2f}'
        for name, mask in bins:
            v = ev[mask][col].median()
            line += f'{v:>14.2f}'
        print(line)


def k_summary(ev, bins):
    total = len(ev)
    print('\n========== 首板事件分布 ==========')
    print(f'  全样: {total}')
    for name, mask in bins:
        n = int(mask.sum())
        pct = n / total * 100 if total else 0
        print(f'  {name:<14} N={n:>6} ({pct:5.2f}%)')
    print(f'\n  连板天数 K 分位: Q50={int(ev["k"].median())}, '
          f'Q75={int(ev["k"].quantile(.75))}, Q90={int(ev["k"].quantile(.9))}, '
          f'Q95={int(ev["k"].quantile(.95))}, Q99={int(ev["k"].quantile(.99))}, max={int(ev["k"].max())}')


def main():
    df = load()
    print('\n[scan] 找首板事件 + 后续连板数')
    ev = scan(df)
    if len(ev) == 0:
        print('无事件'); return

    bins = make_groups(ev)
    k_summary(ev, bins)

    # 卦象分布
    print('\n========== 卦象分布 (单位 %, 各组内占比) ==========')
    cat_table(ev, bins, 'gua_t', '当日个股日卦')
    cat_table(ev, bins, 'gua_prev', '前夜 (T-1) 个股日卦')
    cat_table(ev, bins, 'mkt_t', '当日大盘日卦')

    # 数值特征
    num_table(ev, bins, ['retail_t','retail_prev','mf_t','mf_prev','trend_t','ret_5d'])

    # 落地
    out_dir = os.path.join(ROOT, 'data_layer', 'data', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    ev.to_parquet(os.path.join(out_dir, 'test148_first_zt.parquet'),
                  engine='pyarrow', compression='snappy')
    print(f'\n落地: data_layer/data/analysis/test148_first_zt.parquet (N={len(ev)})')


if __name__ == '__main__':
    main()
