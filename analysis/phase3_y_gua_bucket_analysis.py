# -*- coding: utf-8 -*-
"""y_gua 桶内 池深×池天 双视角分析 (Phase 2 模式, 按 y_gua 主分治)

输入: test6 真裸基线下的 IS sig + trd 数据 (按 y_gua 切片)
输出: 每个 y_gua 桶完整报告 (sig 矩阵 / trd 矩阵 / v1/v2/v3 三版本)

用法:
  python analysis/phase3_y_gua_bucket_analysis.py 000   # 跑坤桶
  python analysis/phase3_y_gua_bucket_analysis.py all   # 跑全 8 桶

数据源: test6 IS run 输出的 sig + trd
  - 需要先跑一次 test6 IS, 把 sig_log 和 trade_log 落地到 ablation/test6_pool_depth/baseline_IS.json
"""
import os
import sys
import io
import json
import argparse
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEPTH_BINS = [-np.inf, -400, -350, -300, -250]
DEPTH_LABELS = ['≤-400 极深', '(-400,-350] 深', '(-350,-300] 中', '(-300,-250] 浅']
DAYS_BINS = [-1, 3, 10, 30, 1e9]
DAYS_LABELS = ['[0-3] 极反', '[4-10] 磨底', '[11-30] 物极', '[31+] 久磨']

GUA_NAMES = {'000':'坤(深熊探底)','001':'艮(底部吸筹)','010':'坎(反弹乏力)',
             '011':'巽(底部爆发)','100':'震(高位出货)','101':'离(高位护盘)',
             '110':'兑(牛末减仓)','111':'乾(疯牛主升)'}


def load_baseline_data():
    """读 test6 IS baseline 的 sig + trd"""
    base_path = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth',
                             'baseline_IS.json')
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f'缺数据 {base_path}\n'
            '请先在 ablation runner 里把 baseline 的 sig_log + trade_log 落地'
        )
    with open(base_path, encoding='utf-8') as f:
        d = json.load(f)
    return pd.DataFrame(d['signal_detail']), pd.DataFrame(d['trade_log'])


def load_y_gua_map():
    df = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                                      'multi_scale_gua_daily.parquet'),
                         columns=['date', 'y_gua'])
    df['date'] = df['date'].astype(str)
    df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
    return dict(zip(df['date'], df['y_gua']))


def matrix_str(matrix, fmt='+5.1f'):
    """按 4×4 矩阵输出 markdown 表"""
    s = '|  | ' + ' | '.join(f'**{c}**' for c in DAYS_LABELS) + ' |\n'
    s += '|---|' + '|'.join(['---']*len(DAYS_LABELS)) + '|\n'
    for d_label in DEPTH_LABELS:
        row = [f'**{d_label}**']
        for t_label in DAYS_LABELS:
            cell = matrix.get((d_label, t_label))
            if cell is None or cell['n'] == 0:
                row.append(' - ')
            else:
                row.append(f'{cell["n"]} / {cell["val"]:{fmt}}')
        s += '| ' + ' | '.join(row) + ' |\n'
    return s


def build_sig_matrix(sigs):
    """sig 矩阵: 每个 (depth, days) cell 的 (n, mean%)"""
    sigs = sigs.copy()
    sigs['depth_b'] = pd.cut(sigs['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS,
                             include_lowest=True)
    sigs['days_b'] = pd.cut(sigs['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)
    matrix = {}
    for d in DEPTH_LABELS:
        for t in DAYS_LABELS:
            cell = sigs[(sigs['depth_b']==d) & (sigs['days_b']==t)]
            matrix[(d, t)] = {'n': len(cell),
                              'val': cell['actual_ret'].mean() if len(cell) else np.nan}
    return matrix


def build_trd_matrix(trades, sigs):
    """trd 矩阵: 每个 cell 的 (n, 利万). 需要 join sig 拿 pool_retail/pool_days"""
    sig_lookup = sigs[['buy_date','code','pool_retail','pool_days']].drop_duplicates(['buy_date','code'])
    t = trades.merge(sig_lookup, on=['buy_date','code'], how='left').copy()
    t['profit_wan'] = t['profit'] / 10000
    t['depth_b'] = pd.cut(t['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS,
                          include_lowest=True)
    t['days_b'] = pd.cut(t['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)
    matrix = {}
    for d in DEPTH_LABELS:
        for tlabel in DAYS_LABELS:
            cell = t[(t['depth_b']==d) & (t['days_b']==tlabel)]
            matrix[(d, tlabel)] = {'n': len(cell),
                                   'val': cell['profit_wan'].sum() if len(cell) else 0}
    return matrix


def analyze_bucket(y_gua, sigs, trades, y_map):
    """分析单个 y_gua 桶, 输出报告"""
    sigs['y_gua'] = sigs['buy_date'].astype(str).map(y_map)
    trades['y_gua'] = trades['buy_date'].astype(str).map(y_map)
    sub_s = sigs[sigs['y_gua']==y_gua].copy()
    sub_t = trades[trades['y_gua']==y_gua].copy()

    print(f'\n# {y_gua} {GUA_NAMES.get(y_gua,"?")} — {len(sub_s)} 信号 / {len(sub_t)} 笔\n')

    sig_matrix = build_sig_matrix(sub_s)
    trd_matrix = build_trd_matrix(sub_t, sub_s)

    print('## 全量 (sig) 矩阵\n')
    print(matrix_str(sig_matrix, fmt='+5.1f'))

    print('## 买入 (trd) 矩阵\n')
    print(matrix_str(trd_matrix, fmt='+5.1f'))

    # 给后续人工分析提供原始字典数据 (打印行/列均值)
    print('\n## 行/列均值参考\n')
    print('### sig 行均值 (横向, 按池深档):')
    for d in DEPTH_LABELS:
        vals = [sig_matrix[(d, t)]['val'] for t in DAYS_LABELS if not np.isnan(sig_matrix[(d, t)]['val'])]
        ns = [sig_matrix[(d, t)]['n'] for t in DAYS_LABELS]
        if vals:
            print(f'  {d}: 行均 {np.mean(vals):+5.2f}% (n_total={sum(ns)})')

    print('### sig 列均值 (纵向, 按池天档):')
    for t in DAYS_LABELS:
        vals = [sig_matrix[(d, t)]['val'] for d in DEPTH_LABELS if not np.isnan(sig_matrix[(d, t)]['val'])]
        ns = [sig_matrix[(d, t)]['n'] for d in DEPTH_LABELS]
        if vals:
            print(f'  {t}: 列均 {np.mean(vals):+5.2f}% (n_total={sum(ns)})')

    print('\n### trd 行均值 (横向, 按池深档):')
    for d in DEPTH_LABELS:
        vals = [trd_matrix[(d, t)]['val'] for t in DAYS_LABELS if trd_matrix[(d, t)]['n'] > 0]
        ns = [trd_matrix[(d, t)]['n'] for t in DAYS_LABELS]
        if vals:
            total = sum(vals)
            print(f'  {d}: 行总利 {total:+5.1f}万 (n_total={sum(ns)})')

    print('### trd 列均值 (纵向, 按池天档):')
    for t in DAYS_LABELS:
        vals = [trd_matrix[(d, t)]['val'] for d in DEPTH_LABELS if trd_matrix[(d, t)]['n'] > 0]
        ns = [trd_matrix[(d, t)]['n'] for d in DEPTH_LABELS]
        if vals:
            total = sum(vals)
            print(f'  {t}: 列总利 {total:+5.1f}万 (n_total={sum(ns)})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('y_gua', help='000-111 或 all')
    args = ap.parse_args()

    sigs, trades = load_baseline_data()
    print(f'baseline: sig {len(sigs)}, trd {len(trades)}')
    y_map = load_y_gua_map()

    if args.y_gua == 'all':
        for y in ['000','001','010','011','100','101','110','111']:
            analyze_bucket(y, sigs.copy(), trades.copy(), y_map)
    else:
        analyze_bucket(args.y_gua, sigs, trades, y_map)


if __name__ == '__main__':
    main()
