# -*- coding: utf-8 -*-
"""Phase 3 坤桶研究 Step 3 — 主力散户背离形态挖掘.

业务假设 (用户提出):
  散户恐慌后股价跌到一定程度, 若主力悄悄吸筹, 后续才会涨.
  → 真反弹 = 散户在跑 + 主力在买 (背离)
  → 假反弹 = 都在跑

挖掘维度 (不预设阈值, 让数据涌现):
  1. 5 天 retail 变化 × 5 天 main_force 变化 (4-cell + 极端)
  2. 20 天 retail 变化 × 20 天 main_force 变化
  3. retail 60d 百分位 × main_force 60d 百分位 (5×5 极端背离)
  4. 主力线过去 20 天"为正天数" (吸筹持续性)
  5. 价格 20 天变化 vs main_force 变化 (价格-资金背离)
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = None


def load_signals():
    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test4',
                     'kun_naked_t11_t89.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    df = pd.DataFrame(d['results'])
    df['entry_date'] = df['entry_date'].astype(str)
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def load_main_codes():
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                                       'main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    return set(uni[uni['board'] == '主板']['code'].unique())


def build_features():
    """对全主板 stocks 数据预算所有衍生特征, 返回 DataFrame (code, date, ...)."""
    main_codes = load_main_codes()
    df = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet'),
                         columns=['code', 'date', 'close', 'retail', 'main_force'])
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = df['date'].astype(str).str[:10]
    df = df[df['code'].isin(main_codes)].copy()
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    print(f'[特征] 加载 {len(df)} 行 / {df["code"].nunique()} 股票')
    g = df.groupby('code', sort=False)

    # 变化值
    df['retail_chg_5d'] = g['retail'].diff(5)
    df['retail_chg_20d'] = g['retail'].diff(20)
    df['mf_chg_5d'] = g['main_force'].diff(5)
    df['mf_chg_20d'] = g['main_force'].diff(20)

    # 60 天滚动百分位 (当日值 - 60d min) / (60d max - 60d min)
    print('[特征] 算 60 天滚动 min/max...')
    for col in ['retail', 'main_force']:
        rmin = g[col].rolling(60, min_periods=20).min().reset_index(0, drop=True)
        rmax = g[col].rolling(60, min_periods=20).max().reset_index(0, drop=True)
        rng = (rmax - rmin).replace(0, np.nan)
        df[f'{col}_60d_pct'] = (df[col] - rmin) / rng * 100

    # 主力线过去 20 天 > 0 的天数
    print('[特征] 算主力 20 天为正天数...')
    pos = (df['main_force'] > 0).astype(int)
    df['mf_pos_20d_n'] = pos.groupby(df['code']).rolling(20, min_periods=10).sum().reset_index(0, drop=True)

    # 价格 20 天变化
    df['close_chg_20d_pct'] = g['close'].pct_change(20) * 100

    return df[['code', 'date',
               'retail_chg_5d', 'retail_chg_20d',
               'mf_chg_5d', 'mf_chg_20d',
               'retail_60d_pct', 'main_force_60d_pct',
               'mf_pos_20d_n', 'close_chg_20d_pct']]


def matrix_2d(df, fx, fy, x_bins, x_labels, y_bins, y_labels, title):
    """二维矩阵: success_rate by (fx 档, fy 档)"""
    sub = df.dropna(subset=[fx, fy]).copy()
    sub['x_bin'] = pd.cut(sub[fx], bins=x_bins, labels=x_labels, include_lowest=True)
    sub['y_bin'] = pd.cut(sub[fy], bins=y_bins, labels=y_labels, include_lowest=True)

    pivot_n = sub.pivot_table(index='y_bin', columns='x_bin', values='success',
                              aggfunc='count', fill_value=0, observed=False)
    pivot_s = sub.pivot_table(index='y_bin', columns='x_bin', values='success',
                              aggfunc='sum', fill_value=0, observed=False)
    pivot_r = (pivot_s / pivot_n.replace(0, np.nan)) * 100

    print(f'\n{"=" * 100}')
    print(f'矩阵: {title}')
    print(f'  行={fy}, 列={fx}')
    print('=' * 100)

    # 成功率热表
    print(f'\n  [成功率%] (cell 样本 < 50 标 .)')
    header = f'  {"":<22} ' + ' '.join(f'{c!s:>10}' for c in pivot_r.columns)
    print(header)
    for y in pivot_r.index:
        row = [f'  {y!s:<22} ']
        for x in pivot_r.columns:
            n = pivot_n.loc[y, x]
            r = pivot_r.loc[y, x]
            if n < 50 or pd.isna(r):
                row.append(f'{".":>10}')
            else:
                bias = r - BASELINE
                marker = '★' if bias > 8 else ('+' if bias > 3 else
                          ('-' if bias < -3 else (' ✗' if bias < -8 else ' ')))
                row.append(f'{r:>5.1f}({marker:<2}')
        print(''.join(row))

    # 样本量
    print(f'\n  [样本量]')
    print(header)
    for y in pivot_n.index:
        row = [f'  {y!s:<22} ']
        for x in pivot_n.columns:
            n = pivot_n.loc[y, x]
            row.append(f'{int(n):>10d}')
        print(''.join(row))

    # 找出最强/最弱 cell (n>=200)
    valid_mask = pivot_n >= 200
    valid = pivot_r[valid_mask]
    if valid.notna().any().any():
        max_v = valid.stack().idxmax()
        min_v = valid.stack().idxmin()
        print(f'\n  [最强 cell] {max_v[0]} × {max_v[1]}: 成功率 {valid.loc[max_v[0], max_v[1]]:.1f}% '
              f'(偏离 {valid.loc[max_v[0], max_v[1]] - BASELINE:+.1f}pp), '
              f'样本 {int(pivot_n.loc[max_v[0], max_v[1]])}')
        print(f'  [最弱 cell] {min_v[0]} × {min_v[1]}: 成功率 {valid.loc[min_v[0], min_v[1]]:.1f}% '
              f'(偏离 {valid.loc[min_v[0], min_v[1]] - BASELINE:+.1f}pp), '
              f'样本 {int(pivot_n.loc[min_v[0], min_v[1]])}')


def cont_table_1d(df, factor, name, bins, labels):
    """1D 连续因子分档"""
    sub = df[df[factor].notna()].copy()
    sub['bucket'] = pd.cut(sub[factor], bins=bins, labels=labels, include_lowest=True)
    g = sub.groupby('bucket', observed=True)['success'].agg(['count', 'sum'])
    g['rate'] = g['sum'] / g['count'] * 100
    g['bias'] = g['rate'] - BASELINE

    print(f'\n{"=" * 80}')
    print(f'1D 因子: {name} ({factor})')
    print('=' * 80)
    print(f'  {"档位":<24} {"n":>8} {"n_succ":>7} {"成功率":>8} {"偏离":>10}')
    print('  ' + '-' * 60)
    for v, row in g.iterrows():
        n = int(row['count']) if row['count'] > 0 else 0
        if n == 0:
            continue
        ns = int(row['sum'])
        rate = row['rate']
        bias = row['bias']
        sign = '★' if bias > 5 else ('+' if bias > 2 else
               ('-' if bias < -2 else ('✗' if bias < -5 else ' ')))
        print(f'  {v!s:<24} {n:>8d} {ns:>7d} {rate:>7.1f}% {bias:>+8.1f}pp  {sign}')


def main():
    global BASELINE
    print('=== 加载信号 + 构造衍生特征 ===')
    sigs = load_signals()
    BASELINE = sigs['success'].mean() * 100
    print(f'信号 {len(sigs)}, 基线成功率 {BASELINE:.2f}%')

    feat = build_features()

    # join 到信号
    df = sigs.merge(feat, left_on=['code', 'entry_date'], right_on=['code', 'date'],
                    how='left').drop(columns='date')
    miss = df[['retail_chg_5d', 'mf_chg_5d', 'retail_60d_pct', 'main_force_60d_pct']].isna().sum()
    print(f'[特征 join] 缺失统计: {miss.to_dict()}')

    # === 1. 5 天主力散户变化矩阵 ===
    matrix_2d(df, 'retail_chg_5d', 'mf_chg_5d',
              x_bins=[-np.inf, -100, -30, 0, 30, 100, np.inf],
              x_labels=['<-100','-100~-30','-30~0','0~30','30~100','>100'],
              y_bins=[-np.inf, -50, 0, 50, np.inf],
              y_labels=['mf<-50','mf-50~0','mf0~50','mf>50'],
              title='5天 retail变化 × 5天 main_force变化')

    # === 2. 20 天主力散户变化矩阵 ===
    matrix_2d(df, 'retail_chg_20d', 'mf_chg_20d',
              x_bins=[-np.inf, -200, -100, -30, 30, 100, np.inf],
              x_labels=['<-200','-200~-100','-100~-30','-30~30','30~100','>100'],
              y_bins=[-np.inf, -100, 0, 100, np.inf],
              y_labels=['mf<-100','mf-100~0','mf0~100','mf>100'],
              title='20天 retail变化 × 20天 main_force变化')

    # === 3. 60 天百分位矩阵 (相对极端) ===
    matrix_2d(df, 'retail_60d_pct', 'main_force_60d_pct',
              x_bins=[-1, 20, 40, 60, 80, 101],
              x_labels=['ret_pct 0-20','20-40','40-60','60-80','80-100'],
              y_bins=[-1, 20, 40, 60, 80, 101],
              y_labels=['mf_pct 0-20','20-40','40-60','60-80','80-100'],
              title='retail 60d百分位 × mf 60d百分位 (相对极端)')

    # === 4. 主力 20 天为正天数 ===
    cont_table_1d(df, 'mf_pos_20d_n', '主力线过去20天为正天数 (吸筹持续性)',
                  bins=[-1, 4, 9, 14, 19, 21],
                  labels=['0-4天','5-9天','10-14天','15-19天','20天'])

    # === 5. 价格 20 天变化 ===
    cont_table_1d(df, 'close_chg_20d_pct', '价格20天变化%',
                  bins=[-np.inf, -20, -10, -5, 0, 5, 10, 20, np.inf],
                  labels=['<-20%','-20~-10%','-10~-5%','-5~0%','0~5%','5~10%','10~20%','>20%'])

    # === 6. 价格变化 × main_force 变化 (背离) ===
    matrix_2d(df, 'close_chg_20d_pct', 'mf_chg_20d',
              x_bins=[-np.inf, -10, -5, 0, 5, 10, np.inf],
              x_labels=['价<-10%','-10~-5%','-5~0%','0~5%','5~10%','>10%'],
              y_bins=[-np.inf, -100, 0, 100, np.inf],
              y_labels=['mf<-100','mf-100~0','mf0~100','mf>100'],
              title='[关键背离] 价格20天变化 × 主力20天变化')

    print(f'\n基线成功率: {BASELINE:.2f}% — 所有偏离值都以此为参照')


if __name__ == '__main__':
    main()
