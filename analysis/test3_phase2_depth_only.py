# -*- coding: utf-8 -*-
"""Phase 2 Step 1 — 池深单维扰动表 (双视角)

只按 pool_retail 分 4 档, 不切 pool_days.
每卦 4 行 × 2 视角 = sig / trd 各 4 cell, 比 4×4 矩阵更易看出规律.

输出:
  ablation/test3/phase2_pool_depth_only.csv
"""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3')

DEPTH_BINS = [-np.inf, -400, -350, -300, -250]
DEPTH_LABELS = ['<=-400 极深', '(-400,-350] 深', '(-350,-300] 中', '(-300,-250] 浅']

GUA_NAME = {
    '000': '坤(深熊探底)', '001': '艮(底部吸筹)', '010': '坎(反弹乏力)', '011': '巽(底部爆发)',
    '100': '震(高位出货)', '101': '离(高位护盘)', '110': '兑(牛末减仓)', '111': '乾(疯牛主升)',
}


def main():
    with open(os.path.join(ABL, 'IS_naked_baseline.json'), encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    sigs['tian_gua'] = sigs['tian_gua'].astype(str).str.zfill(3)
    sigs['depth_bucket'] = pd.cut(sigs['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)

    trades = pd.DataFrame(d['trade_log'])
    trades['tian_gua'] = trades['tian_gua'].astype(str).str.zfill(3)
    sig_lookup = sigs[['buy_date', 'code', 'pool_retail']].drop_duplicates(['buy_date', 'code'])
    trades = trades.merge(sig_lookup, on=['buy_date', 'code'], how='left')
    trades['depth_bucket'] = pd.cut(trades['pool_retail'], bins=DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    trades['profit_wan'] = trades['profit'] / 10000

    print(f'\nIS sig: {len(sigs)} / trd: {len(trades)}\n')

    rows = []
    for gua in sorted(sigs['tian_gua'].unique()):
        sub_s = sigs[sigs['tian_gua'] == gua]
        sub_t = trades[trades['tian_gua'] == gua]
        for depth in DEPTH_LABELS:
            cell_s = sub_s[sub_s['depth_bucket'] == depth]
            cell_t = sub_t[sub_t['depth_bucket'] == depth]
            rows.append({
                'gua': gua,
                'gua_name': GUA_NAME.get(gua, ''),
                'depth': depth,
                'sig_n': len(cell_s),
                'sig_mean%': cell_s['actual_ret'].mean() if len(cell_s) else np.nan,
                'sig_win%': (cell_s['actual_ret'] > 0).mean() * 100 if len(cell_s) else np.nan,
                'trd_n': len(cell_t),
                'trd_利万': cell_t['profit_wan'].sum() if len(cell_t) else 0,
                'trd_mean%': cell_t['ret_pct'].mean() if len(cell_t) else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ABL, 'phase2_pool_depth_only.csv'), index=False, encoding='utf-8-sig',
              float_format='%.2f')

    # 分卦打印
    for gua in sorted(sigs['tian_gua'].unique()):
        sub = df[df['gua'] == gua]
        print('=' * 100)
        print(f'  {gua}  {GUA_NAME[gua]}')
        print('=' * 100)
        print(f'  {"depth":<14} {"sig_n":>6} {"sig_mean%":>10} {"sig_win%":>9}  |  {"trd_n":>5} {"trd_利万":>8} {"trd_mean%":>10}')
        for _, r in sub.iterrows():
            print(f'  {r["depth"]:<14} {r["sig_n"]:>6} {r["sig_mean%"]:>+10.2f} {r["sig_win%"]:>8.1f}%  |  '
                  f'{r["trd_n"]:>5} {r["trd_利万"]:>+8.1f} {r["trd_mean%"]:>+9.1f}%')
        print()

    print(f'  落地: {os.path.join(ABL, "phase2_pool_depth_only.csv")}')


if __name__ == '__main__':
    main()
