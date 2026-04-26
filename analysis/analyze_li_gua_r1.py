# -*- coding: utf-8 -*-
"""离卦 (101) 第一轮因子分析 · 池深 × 池天 · 双视角

数据源: data_layer/data/backtest_8gua_naked_result.json (裸跑综合回测, 离卦 489 个候选信号)

双视角:
  - 全量视角 (signal): 所有候选信号, 假设全都被买入, 统计 actual_ret
  - 买入视角 (trade): 实际通过资金约束买入的 22 笔, 统计 ret_pct (扣佣后)

两维因子:
  - 池深 (pool_retail): 入池后散户线最深值 (负数, 越负表示抛售越重)
  - 池天 (pool_days): 入池到信号触发的天数

从"八卦理论 — 离卦 101 高位护盘"的视角解读规律.
"""
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED_RESULT = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
GUA = '101'
GUA_NAME = '离'

DEPTH_BUCKETS = [
    (-float('inf'), -500, '≤-500 极深'),
    (-500, -400, '-500~-400 深'),
    (-400, -350, '-400~-350 较深'),
    (-350, -300, '-350~-300 中'),
    (-300, -250, '-300~-250 浅'),
]

DAYS_BUCKETS = [
    (0, 3, '0-3 天 极而反'),
    (4, 7, '4-7 天 底部'),
    (8, 15, '8-15 天 磨底'),
    (16, 30, '16-30 天 久磨'),
    (31, 10000, '31+ 天 慢牛底'),
]


def bucket_depth(x):
    x = float(x)
    for lo, hi, label in DEPTH_BUCKETS:
        if lo < x <= hi:
            return label
    return '外'


def bucket_days(x):
    x = int(x) if pd.notna(x) else -1
    for lo, hi, label in DAYS_BUCKETS:
        if lo <= x <= hi:
            return label
    return '外'


def load_gua_data():
    with open(NAKED_RESULT, encoding='utf-8') as f:
        d = json.load(f)

    sig = pd.DataFrame(d['signal_detail'])
    trd = pd.DataFrame(d['trade_log'])

    for df in (sig, trd):
        for col in ('tian_gua', 'gua', 'ren_gua', 'di_gua'):
            if col in df.columns:
                df[col] = df[col].astype(str).str.zfill(3)

    li_s = sig[sig['tian_gua'] == GUA].copy()
    li_t = trd[trd['gua'] == GUA].copy()

    # trade 反查 signal 拿到 pool_retail / pool_days
    key_sig = li_s.set_index(['code', 'buy_date'])[['pool_retail', 'pool_days']]
    li_t = li_t.merge(
        key_sig.reset_index(), on=['code', 'buy_date'], how='left', suffixes=('', '_sig')
    )

    li_s['depth_bucket'] = li_s['pool_retail'].apply(bucket_depth)
    li_s['days_bucket'] = li_s['pool_days'].apply(bucket_days)
    li_t['depth_bucket'] = li_t['pool_retail'].apply(bucket_depth)
    li_t['days_bucket'] = li_t['pool_days'].apply(bucket_days)
    return li_s, li_t


def _fmt(v, kind='ret'):
    if pd.isna(v):
        return '  -'
    if kind == 'ret':
        return f'{v:+6.2f}%'
    if kind == 'pct':
        return f'{v:5.1f}%'
    if kind == 'int':
        return f'{int(v):>4}'
    return str(v)


def agg_group_1d(df, bucket_col, ret_col, bucket_order):
    rows = []
    for b in bucket_order:
        sub = df[df[bucket_col] == b]
        if len(sub) == 0:
            rows.append({'bucket': b, 'n': 0, 'mean': np.nan, 'median': np.nan, 'win': np.nan, 'sum': 0.0})
            continue
        rets = sub[ret_col].dropna()
        rows.append({
            'bucket': b,
            'n': len(sub),
            'mean': rets.mean() if len(rets) else np.nan,
            'median': rets.median() if len(rets) else np.nan,
            'win': (rets > 0).mean() * 100 if len(rets) else np.nan,
            'sum': rets.sum() if len(rets) else 0.0,
        })
    return pd.DataFrame(rows)


def print_1d(title, sig_df, trd_df, bucket_col, bucket_order):
    print('\n' + '=' * 100)
    print(f'  {title}')
    print('=' * 100)
    s = agg_group_1d(sig_df, bucket_col, 'actual_ret', bucket_order)
    t = agg_group_1d(trd_df, bucket_col, 'ret_pct', bucket_order)

    hdr = f'{"bucket":<22} | {"全量 n":>6} {"均收":>7} {"中位":>7} {"胜率":>6} {"累积":>8}  | {"买入 n":>6} {"均收":>7} {"中位":>7} {"胜率":>6} {"累积":>8}'
    print(hdr)
    print('-' * len(hdr))
    for b in bucket_order:
        rs = s[s['bucket'] == b].iloc[0]
        rt = t[t['bucket'] == b].iloc[0]
        print(f'{b:<22} | {_fmt(rs["n"], "int")} {_fmt(rs["mean"])} {_fmt(rs["median"])} {_fmt(rs["win"], "pct")} {_fmt(rs["sum"])}  '
              f'| {_fmt(rt["n"], "int")} {_fmt(rt["mean"])} {_fmt(rt["median"])} {_fmt(rt["win"], "pct")} {_fmt(rt["sum"])}')


def print_2d_heatmap(title, df, ret_col, value_label, depth_order, days_order):
    print('\n' + '=' * 100)
    print(f'  {title}')
    print('=' * 100)

    # count pivot
    cnt = df.pivot_table(index='depth_bucket', columns='days_bucket', values=ret_col, aggfunc='count', fill_value=0)
    mean = df.pivot_table(index='depth_bucket', columns='days_bucket', values=ret_col, aggfunc='mean')

    cnt = cnt.reindex(index=depth_order, columns=days_order, fill_value=0)
    mean = mean.reindex(index=depth_order, columns=days_order)

    # 打印 count
    hdr_label = 'depth × days'
    print('\n  [count] 样本数分布')
    print(f'  {hdr_label:<22} ' + ' '.join(f'{d:<18}' for d in days_order))
    for depth in depth_order:
        row = cnt.loc[depth]
        print(f'  {depth:<22} ' + ' '.join(f'{int(row[d]):>18}' for d in days_order))

    # 打印 mean
    print(f'\n  [{value_label}] 各组均收益 (%)')
    print(f'  {hdr_label:<22} ' + ' '.join(f'{d:<18}' for d in days_order))
    for depth in depth_order:
        row = mean.loc[depth]
        print(f'  {depth:<22} ' + ' '.join(f'{("  -" if pd.isna(row[d]) else f"{row[d]:+17.2f}%"):>18}' for d in days_order))


def main():
    li_s, li_t = load_gua_data()

    print(f'\n  离卦 ({GUA} {GUA_NAME}) 第一轮因子分析 · 池深 × 池天')
    print(f'  数据源: {NAKED_RESULT}')
    print(f'  全量 (signal) 候选: {len(li_s)} 条   |   买入 (trade) 实际: {len(li_t)} 条')
    print(f'  池深范围: {li_s["pool_retail"].min():.0f} ~ {li_s["pool_retail"].max():.0f}   '
          f'均 {li_s["pool_retail"].mean():.0f}   中位 {li_s["pool_retail"].median():.0f}')
    print(f'  池天范围: {li_s["pool_days"].min()} ~ {li_s["pool_days"].max()} 天   '
          f'均 {li_s["pool_days"].mean():.1f}   中位 {li_s["pool_days"].median():.0f}')

    depth_order = [b[2] for b in DEPTH_BUCKETS]
    days_order = [b[2] for b in DAYS_BUCKETS]

    # 1D: 池深
    print_1d('一维 · 池深分桶 (pool_retail)', li_s, li_t, 'depth_bucket', depth_order)
    # 1D: 池天
    print_1d('一维 · 池天分桶 (pool_days)', li_s, li_t, 'days_bucket', days_order)

    # 2D: 池深 × 池天
    print_2d_heatmap('二维 · 池深 × 池天 · 全量 (actual_ret)', li_s, 'actual_ret', '全量均收', depth_order, days_order)
    print_2d_heatmap('二维 · 池深 × 池天 · 买入 (ret_pct)', li_t, 'ret_pct', '买入均收', depth_order, days_order)


if __name__ == '__main__':
    main()
