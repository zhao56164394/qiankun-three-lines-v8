# -*- coding: utf-8 -*-
"""Phase 2 Step 1 — 各卦 (池深 × 池天) 二元矩阵 + 双视角

每卦输出两个 4×4 矩阵:
  sig 视角: n / mean% (cell 性能 — 信号纯 alpha)
  trd 视角: n / 利万   (cell 实战 — 资金分配后兑现)

不做候选筛选 / 不做 LOO. 这一步只看二元结构, 让人观察各卦的"最优环境格"
和"最劣环境格", 给出业务解释.

输出:
  ablation/test3/phase2_2d_matrix.csv   (每行 gua×depth×days, 含 sig/trd 双视角)
  控制台: 8 卦 × 2 视角 = 16 个矩阵
"""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3')

DEPTH_BINS = [-np.inf, -400, -350, -300, -250]
DEPTH_LABELS = ['<=-400', '(-400,-350]', '(-350,-300]', '(-300,-250]']

DAYS_BINS = [-1, 3, 10, 30, 1e9]
DAYS_LABELS = ['[0-3]', '[4-10]', '[11-30]', '[31+]']

GUA_NAME = {
    '000': '坤(深熊探底)', '001': '艮(底部吸筹)', '010': '坎(反弹乏力)', '011': '巽(底部爆发)',
    '100': '震(高位出货)', '101': '离(高位护盘)', '110': '兑(牛末减仓)', '111': '乾(疯牛主升)',
}


def main():
    # 用完全裸跑 baseline (解除 cfg 起点 tier/pool_days 约束), 暴露完整 4×4
    with open(os.path.join(ABL, 'IS_naked_baseline.json'), encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    sigs['tian_gua'] = sigs['tian_gua'].astype(str).str.zfill(3)
    sigs['depth_bucket'] = pd.cut(sigs['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    sigs['days_bucket'] = pd.cut(sigs['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)

    trades = pd.DataFrame(d['trade_log'])
    trades['tian_gua'] = trades['tian_gua'].astype(str).str.zfill(3)
    sig_lookup = sigs[['buy_date', 'code', 'pool_retail', 'pool_days']].drop_duplicates(['buy_date', 'code'])
    trades = trades.merge(sig_lookup, on=['buy_date', 'code'], how='left')
    trades['depth_bucket'] = pd.cut(trades['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    trades['days_bucket'] = pd.cut(trades['pool_days'], bins=DAYS_BINS, labels=DAYS_LABELS)
    trades['profit_wan'] = trades['profit'] / 10000

    print(f'\nIS sig: {len(sigs)} / trd: {len(trades)} / IS baseline: {d["meta"]["final_capital"]/10000:.1f}万\n')

    # 落地全表
    rows = []
    for gua in sorted(sigs['tian_gua'].unique()):
        sub_s = sigs[sigs['tian_gua'] == gua]
        sub_t = trades[trades['tian_gua'] == gua]
        for depth in DEPTH_LABELS:
            for days in DAYS_LABELS:
                cell_s = sub_s[(sub_s['depth_bucket'] == depth) & (sub_s['days_bucket'] == days)]
                cell_t = sub_t[(sub_t['depth_bucket'] == depth) & (sub_t['days_bucket'] == days)]
                rows.append({
                    'gua': gua, 'gua_name': GUA_NAME.get(gua, ''),
                    'depth': depth, 'days': days,
                    'sig_n': len(cell_s),
                    'sig_mean%': cell_s['actual_ret'].mean() if len(cell_s) else np.nan,
                    'trd_n': len(cell_t),
                    'trd_利万': cell_t['profit_wan'].sum() if len(cell_t) else 0,
                })
    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(ABL, 'phase2_2d_matrix.csv'), index=False, encoding='utf-8-sig',
                  float_format='%.2f')

    # 分卦打印 4×4 矩阵
    for gua in sorted(sigs['tian_gua'].unique()):
        sub_s = sigs[sigs['tian_gua'] == gua]
        sub_t = trades[trades['tian_gua'] == gua]
        print('=' * 100)
        print(f'  {gua}  {GUA_NAME[gua]}    sig: {len(sub_s)} 条   trd: {len(sub_t)} 笔')
        print('=' * 100)

        # sig 视角矩阵: 每 cell 显示 "n/mean"
        sig_cells = {}
        for depth in DEPTH_LABELS:
            for days in DAYS_LABELS:
                cell = sub_s[(sub_s['depth_bucket'] == depth) & (sub_s['days_bucket'] == days)]
                if len(cell) == 0:
                    sig_cells[(depth, days)] = '   -    '
                else:
                    n = len(cell)
                    m = cell['actual_ret'].mean()
                    sig_cells[(depth, days)] = f'{n:>4}/{m:>+5.1f}'

        print('\n  sig 视角 (信号数 / 平均收益%):')
        header = '  ' + ' '*14 + ''.join(f'{d:>13}' for d in DAYS_LABELS)
        print(header)
        for depth in DEPTH_LABELS:
            row_str = f'  {depth:<14}' + ''.join(f'{sig_cells[(depth, d)]:>13}' for d in DAYS_LABELS)
            print(row_str)

        # trd 视角矩阵: 每 cell 显示 "n/利万"
        trd_cells = {}
        for depth in DEPTH_LABELS:
            for days in DAYS_LABELS:
                cell = sub_t[(sub_t['depth_bucket'] == depth) & (sub_t['days_bucket'] == days)]
                if len(cell) == 0:
                    trd_cells[(depth, days)] = '   -    '
                else:
                    n = len(cell)
                    p = cell['profit_wan'].sum()
                    trd_cells[(depth, days)] = f'{n:>3}/{p:>+6.1f}'

        print('\n  trd 视角 (实买笔数 / 总利润万):')
        header = '  ' + ' '*14 + ''.join(f'{d:>13}' for d in DAYS_LABELS)
        print(header)
        for depth in DEPTH_LABELS:
            row_str = f'  {depth:<14}' + ''.join(f'{trd_cells[(depth, d)]:>13}' for d in DAYS_LABELS)
            print(row_str)

        print()

    print(f'  落地: {os.path.join(ABL, "phase2_2d_matrix.csv")}')


if __name__ == '__main__':
    main()
