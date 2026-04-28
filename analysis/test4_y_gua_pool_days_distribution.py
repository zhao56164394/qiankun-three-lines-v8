# -*- coding: utf-8 -*-
"""Phase 3 Step 0 — 池天 (pool_days) 在 y_gua 主分治下的分布勘察.

目的: 看清 8 个 y_gua 桶下 pool_days 的真实分布, 再决定怎么切档位
      (避免凭直觉划档导致 cell 稀疏 → 噪声主导的 Phase 2 教训).

数据源: IS_baseline.json (test3 cfg 在 2014-2022 跑出的 signal_detail)
  - sig 全量 (~11k 条) = 通过 cfg 过滤后的潜在买入候选信号
  - 字段含 pool_days, pool_retail, buy_date, code, tian_gua

y_gua join: 按 buy_date 查 multi_scale_gua_daily.parquet (12 月窗口版)

输出:
  1. 各 y_gua 桶 sig 样本量 + pool_days 分位数 (p10/p25/p50/p75/p90/p95) + mean
  2. 全局 pool_days 分位数 (作对照)
  3. 候选档位建议 (4 档, 按全局分位均切)
  4. 同样对 trd (实买) 作辅助参考
"""
import os
import sys
import json
import io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3')

GUA_NAME = {
    '000': '坤(深熊)', '001': '艮(吸筹)', '010': '坎(乏力)', '011': '巽(底爆)',
    '100': '震(出货)', '101': '离(护盘)', '110': '兑(末减)', '111': '乾(疯牛)',
}

QS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def load_y_map():
    """加载日期 → y_gua 映射 (12 月窗口版)"""
    path = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    df = pd.read_parquet(path, columns=['date', 'y_gua'])
    df['date'] = df['date'].astype(str)
    df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
    return dict(zip(df['date'], df['y_gua']))


def describe_pool_days(arr, name):
    """输出 pool_days 数组的分位数 + 均值"""
    if len(arr) == 0:
        return None
    a = np.asarray(arr)
    qs = np.quantile(a, QS)
    return {
        'name': name,
        'n': len(a),
        'mean': a.mean(),
        'p10': qs[0], 'p25': qs[1], 'p50': qs[2],
        'p75': qs[3], 'p90': qs[4], 'p95': qs[5],
    }


def print_table(rows, header):
    """简易表格打印"""
    if not rows:
        print('(空)')
        return
    print('  ' + header)
    print('  ' + '-' * (len(header) - 2))
    for r in rows:
        print('  ' + r)


def main():
    # 1. 数据加载
    y_map = load_y_map()
    print(f'\n=== 加载 y_gua 映射 (12 月窗口版): {len(y_map)} 天 ===')

    is_path = os.path.join(ABL, 'IS_baseline.json')
    with open(is_path, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    trades = pd.DataFrame(d['trade_log'])
    print(f'IS_baseline: sig={len(sigs)}, trd={len(trades)}')
    print(f'IS 终值: {d["meta"]["final_capital"]/10000:.1f} 万')

    # 2. join y_gua
    sigs['buy_date'] = sigs['buy_date'].astype(str)
    sigs['y_gua'] = sigs['buy_date'].map(y_map)
    trades['buy_date'] = trades['buy_date'].astype(str)
    trades['y_gua'] = trades['buy_date'].map(y_map)

    # join pool_days 进 trades (via buy_date+code)
    sig_lookup = sigs[['buy_date', 'code', 'pool_days']].drop_duplicates(['buy_date', 'code'])
    trades = trades.merge(sig_lookup, on=['buy_date', 'code'], how='left')
    trades['profit_wan'] = trades['profit'] / 10000

    # 缺 y_gua 的 sig (买入日早于 y_gua 起点)
    miss_sig = sigs['y_gua'].isna().sum()
    miss_trd = trades['y_gua'].isna().sum()
    miss_pd_trd = trades['pool_days'].isna().sum()
    print(f'sig 缺 y_gua: {miss_sig}; trd 缺 y_gua: {miss_trd}; trd 缺 pool_days: {miss_pd_trd}')

    sigs_ok = sigs.dropna(subset=['y_gua', 'pool_days']).copy()
    trades_ok = trades.dropna(subset=['y_gua', 'pool_days']).copy()
    print(f'有效 sig: {len(sigs_ok)} / 有效 trd: {len(trades_ok)}\n')

    # 3. 全局分布
    print('=' * 90)
    print('维度 0: 全局 pool_days 分布 (全样本, 不分 y_gua)')
    print('=' * 90)
    gs = describe_pool_days(sigs_ok['pool_days'].values, 'sig')
    gt = describe_pool_days(trades_ok['pool_days'].values, 'trd')
    print(f'  {"":<8} {"n":>6} {"mean":>7} {"p10":>5} {"p25":>5} {"p50":>5} {"p75":>5} {"p90":>5} {"p95":>5}')
    for r in (gs, gt):
        if r:
            print(f'  {r["name"]:<8} {r["n"]:>6d} {r["mean"]:>7.2f} '
                  f'{r["p10"]:>5.0f} {r["p25"]:>5.0f} {r["p50"]:>5.0f} '
                  f'{r["p75"]:>5.0f} {r["p90"]:>5.0f} {r["p95"]:>5.0f}')

    # 4. 8 个 y_gua 桶下分布
    print('\n' + '=' * 90)
    print('维度 1: 按 y_gua 分组的 pool_days 分布 (sig 视角 = 候选信号)')
    print('=' * 90)
    print(f'  {"y_gua":<6} {"卦名":<10} {"n_sig":>6} {"mean":>7} '
          f'{"p10":>5} {"p25":>5} {"p50":>5} {"p75":>5} {"p90":>5} {"p95":>5}')
    print('  ' + '-' * 86)
    rows_sig = []
    for y in sorted(sigs_ok['y_gua'].unique()):
        sub = sigs_ok[sigs_ok['y_gua'] == y]
        r = describe_pool_days(sub['pool_days'].values, y)
        if r is None:
            continue
        rows_sig.append({'y_gua': y, **r})
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} {r["n"]:>6d} {r["mean"]:>7.2f} '
              f'{r["p10"]:>5.0f} {r["p25"]:>5.0f} {r["p50"]:>5.0f} '
              f'{r["p75"]:>5.0f} {r["p90"]:>5.0f} {r["p95"]:>5.0f}')

    # 5. trd 辅助视角
    print('\n' + '=' * 90)
    print('维度 2: 按 y_gua 分组的 pool_days 分布 (trd 视角 = 实际成交, 仅作参考)')
    print('=' * 90)
    print(f'  {"y_gua":<6} {"卦名":<10} {"n_trd":>6} {"利万":>8} {"mean":>7} '
          f'{"p10":>5} {"p25":>5} {"p50":>5} {"p75":>5} {"p90":>5} {"p95":>5}')
    print('  ' + '-' * 95)
    for y in sorted(trades_ok['y_gua'].unique()):
        sub = trades_ok[trades_ok['y_gua'] == y]
        r = describe_pool_days(sub['pool_days'].values, y)
        if r is None:
            continue
        profit = sub['profit_wan'].sum()
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} {r["n"]:>6d} {profit:>+8.1f} {r["mean"]:>7.2f} '
              f'{r["p10"]:>5.0f} {r["p25"]:>5.0f} {r["p50"]:>5.0f} '
              f'{r["p75"]:>5.0f} {r["p90"]:>5.0f} {r["p95"]:>5.0f}')

    # 6. 候选档位建议 — 用全局分位数等切
    print('\n' + '=' * 90)
    print('维度 3: 候选档位建议 (按全局 sig 分位数)')
    print('=' * 90)
    s = sigs_ok['pool_days'].values
    # 4 档建议: 按 25/50/75 等切
    q4 = np.quantile(s, [0.25, 0.50, 0.75])
    # 5 档建议: 按 20/40/60/80
    q5 = np.quantile(s, [0.20, 0.40, 0.60, 0.80])
    print(f'  4 档建议 (25/50/75 分位): [0,{int(q4[0])}] / ({int(q4[0])},{int(q4[1])}] / ({int(q4[1])},{int(q4[2])}] / ({int(q4[2])},+∞)')
    print(f'  5 档建议 (20/40/60/80 分位): [0,{int(q5[0])}] / ({int(q5[0])},{int(q5[1])}] / ({int(q5[1])},{int(q5[2])}] / ({int(q5[2])},{int(q5[3])}] / ({int(q5[3])},+∞)')

    # 各 y_gua 桶下样本量分布在候选档的 cell 量
    def bucket_count(arr, bins):
        edges = [-0.5] + list(bins) + [1e9]
        cuts = pd.cut(arr, bins=edges, labels=False)
        return np.bincount(cuts.astype(int), minlength=len(edges) - 1)

    print(f'\n  各 y_gua 桶在 4 档下的 sig cell 量 (bins={list(int(x) for x in q4)}):')
    print(f'  {"y_gua":<6} {"卦名":<10} {"档1":>6} {"档2":>6} {"档3":>6} {"档4":>6} {"min_cell":>9}')
    print('  ' + '-' * 60)
    cell_min_records = []
    for y in sorted(sigs_ok['y_gua'].unique()):
        sub = sigs_ok[sigs_ok['y_gua'] == y]
        cnts = bucket_count(sub['pool_days'].values, q4)
        cell_min_records.append((y, int(cnts.min())))
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} '
              f'{cnts[0]:>6d} {cnts[1]:>6d} {cnts[2]:>6d} {cnts[3]:>6d} {cnts.min():>9d}')

    print(f'\n  各 y_gua 桶在 5 档下的 sig cell 量 (bins={list(int(x) for x in q5)}):')
    print(f'  {"y_gua":<6} {"卦名":<10} {"档1":>6} {"档2":>6} {"档3":>6} {"档4":>6} {"档5":>6} {"min_cell":>9}')
    print('  ' + '-' * 70)
    for y in sorted(sigs_ok['y_gua'].unique()):
        sub = sigs_ok[sigs_ok['y_gua'] == y]
        cnts = bucket_count(sub['pool_days'].values, q5)
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} '
              f'{cnts[0]:>6d} {cnts[1]:>6d} {cnts[2]:>6d} {cnts[3]:>6d} {cnts[4]:>6d} {cnts.min():>9d}')

    # 7. 落地 json 供后续脚本读取
    out = {
        'meta': {
            'data_source': 'IS_baseline.json (2014-2022)',
            'y_gua_window': '12 month',
            'sig_n_total': len(sigs_ok),
            'trd_n_total': len(trades_ok),
        },
        'global_sig_pool_days': gs,
        'global_trd_pool_days': gt,
        'sig_by_y_gua': rows_sig,
        'q4_bins': [int(x) for x in q4],
        'q5_bins': [int(x) for x in q5],
        'q4_min_cell_per_y_gua': cell_min_records,
    }
    out_path = os.path.join(ABL, 'phase3_pool_days_distribution.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n落地: {out_path}')


if __name__ == '__main__':
    main()
