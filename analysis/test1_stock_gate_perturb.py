# -*- coding: utf-8 -*-
"""分析 test1 -gen + max_pos=3 baseline 的信号在 stock m/y_gua 上的双视角分布
找出双视角差/好的 (d_gua, stock_y_gua) 和 (d_gua, stock_m_gua) 候选 cell
分别在 std 版和 const 版 stock_multi_scale_gua_daily 上做.
"""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUAS = ['000','001','010','011','100','101','110','111']
NAMES = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}


def load_baseline():
    path = os.path.join(ROOT, 'data_layer/data/ablation/test1/test1_no_gen_max3.json')
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    trd = pd.DataFrame(d['trade_log'])
    sig = sig[~sig['is_skip']].copy()
    sig['tian_gua'] = sig['tian_gua'].astype(str).str.zfill(3)
    sig['code'] = sig['code'].astype(str).str.zfill(6)
    sig['signal_date'] = sig['signal_date'].astype(str)
    trd['gua'] = trd['gua'].astype(str).str.zfill(3)
    trd['code'] = trd['code'].astype(str).str.zfill(6)
    trd['buy_date'] = trd['buy_date'].astype(str)
    return sig, trd


def merge_stock_gua(sig, trd, ver_label, parquet_path):
    sg = pd.read_parquet(parquet_path, columns=['date', 'code', 'm_gua', 'y_gua'])
    sg['code'] = sg['code'].astype(str).str.zfill(6)
    sg['date'] = sg['date'].astype(str)
    sg = sg.dropna(subset=['m_gua', 'y_gua'])
    sg['m_gua'] = sg['m_gua'].astype(str).str.zfill(3)
    sg['y_gua'] = sg['y_gua'].astype(str).str.zfill(3)
    sg = sg.rename(columns={'m_gua': 'stk_m', 'y_gua': 'stk_y'})

    sig_v = sig.merge(sg, left_on=['signal_date', 'code'], right_on=['date', 'code'], how='left')
    trd_v = trd.merge(sg, left_on=['buy_date', 'code'], right_on=['date', 'code'], how='left')
    return sig_v, trd_v


def analyze(sig, trd, label):
    print(f'\n{"="*70}')
    print(f'  {label}')
    print(f'{"="*70}')
    print(f'  total: 全量 {len(sig)} 条, 实买 {len(trd)} 笔, 利润 {trd["profit"].sum()/10000:+.1f}万')
    print(f'  覆盖率: 全量有 stk 卦 {sig["stk_y"].notna().sum()}/{len(sig)} = {sig["stk_y"].notna().mean()*100:.1f}%')
    print(f'         实买有 stk 卦 {trd["stk_y"].notna().sum()}/{len(trd)} = {trd["stk_y"].notna().mean()*100:.1f}%')

    # 8 卦 × 8 stk_y 双视角扰动表
    print(f'\n  双视角扰动表 (d × stk_y_gua), n>=20 才显示:')
    print(f'  {"d":<6} {"stk_y":<6} {"sig_n":>5} {"sig_mean%":>10} {"trd_n":>4} {"trd_利万":>9}')
    bad, good = [], []
    for d in GUAS:
        for sy in GUAS:
            ssub = sig[(sig['tian_gua']==d) & (sig['stk_y']==sy)]
            tsub = trd[(trd['gua']==d) & (trd['stk_y']==sy)]
            if len(ssub) < 20:
                continue
            sm = ssub['actual_ret'].mean()
            tn = len(tsub)
            tp = tsub['profit'].sum()/10000 if tn else 0
            flag = ''
            if sm < -3 and (tn < 5 or tp < -3):
                flag = ' BAD'
                bad.append((d, sy, sm, tp, len(ssub), tn))
            elif sm > 3 and (tn >= 3 and tp > 5):
                flag = ' GOOD'
                good.append((d, sy, sm, tp, len(ssub), tn))
            print(f'  {d}{NAMES[d]:<2} {sy}{NAMES[sy]:<2} {len(ssub):>5} {sm:>+9.2f}% {tn:>4} {tp:>+8.1f}{flag}')
    return bad, good


def main():
    sig, trd = load_baseline()
    print(f'baseline: test1 -gen + max_pos=3 = 4425 万')
    print(f'  {len(sig)} 全量信号, {len(trd)} 实买笔')

    # 实验 A: const 版
    sig_c, trd_c = merge_stock_gua(sig, trd, 'const',
        os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily_const.parquet'))
    bad_c, good_c = analyze(sig_c, trd_c, 'const 版 (固定 SPD=3.5 ACC=93)')

    # 实验 B: std 版
    sig_s, trd_s = merge_stock_gua(sig, trd, 'std',
        os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'))
    bad_s, good_s = analyze(sig_s, trd_s, 'std 版 (252 滑动 std)')

    print(f'\n{"="*70}')
    print(f'  双视角差格 候选汇总')
    print(f'{"="*70}')
    print(f'  const 版差格: {len(bad_c)}')
    for d, sy, sm, tp, sn, tn in bad_c:
        print(f'    d={d}{NAMES[d]} stk_y={sy}{NAMES[sy]}  sig_n={sn} mean={sm:+.2f}%  trd_n={tn} 利={tp:+.1f}万')
    print(f'  std 版差格: {len(bad_s)}')
    for d, sy, sm, tp, sn, tn in bad_s:
        print(f'    d={d}{NAMES[d]} stk_y={sy}{NAMES[sy]}  sig_n={sn} mean={sm:+.2f}%  trd_n={tn} 利={tp:+.1f}万')

    # 保存候选
    out = os.path.join(ROOT, 'data_layer/data/ablation/test1/stock_gate_candidates.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'const_bad': bad_c, 'const_good': good_c,
            'std_bad': bad_s, 'std_good': good_s,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n候选保存: {out}')


if __name__ == '__main__':
    main()
