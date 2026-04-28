# -*- coding: utf-8 -*-
"""Phase 2 池深/池天 IS 双视角扰动表 (按 strategy-ablation skill 范式)

视角 1 (sig 全量): 11024 个通过 cfg 过滤的潜在信号
  - sig_n, sig_mean%, sig_win%
  - bootstrap 95% CI of mean (n_boot=1000)
  - 反映"信号纯 alpha"

视角 2 (trd 实买): 322 个经 max_pos+rank 筛选后实际成交
  - trd_n, trd_利万 (元转万), trd_mean%
  - 反映"资金分配后实战兑现"

判定矩阵 (来自 SKILL.md):
  sig_n>=20 + CI 全负 + trd_n>=5 + trd_利<0  → ✗ 真有害
  sig_n>=20 + CI 全正                       → ★ 真有益
  sig_n>=20 + CI 跨0 + trd_n>=8 + trd_利>+10 → ★ 实战有益
  sig_n>=20 + CI 跨0 + trd_n<5              → ○ 灰区
  sig_n<20                                  → 不下结论 (skip LOO)

输出:
  ablation/test3/phase2_perturb_depth.csv
  ablation/test3/phase2_perturb_days.csv
  ablation/test3/phase2_verdict.json
"""
import os, sys, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABL = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3')

DEPTH_BINS = [-np.inf, -400, -350, -300, -250]
DEPTH_LABELS = ['<=-400 极深', '(-400,-350] 深', '(-350,-300] 中', '(-300,-250] 浅']

DAYS_BINS = [-1, 3, 10, 30, 1e9]
DAYS_LABELS = ['[0-3] 极反', '[4-10] 磨底', '[11-30] 物极', '[31+] 久磨']

GUA_NAME = {
    '000': '坤(深熊)', '001': '艮(吸筹)', '010': '坎(乏力)', '011': '巽(底爆)',
    '100': '震(出货)', '101': '离(护盘)', '110': '兑(末减)', '111': '乾(疯牛)',
}

# 判定门槛
N_MIN_LOO = 20      # SKILL: n<20 不做 LOO
N_MIN_CONCL = 10    # SKILL: n<10 不下结论
N_TRD_HARM = 5      # ✗ 判定的最低 trd_n
N_TRD_BENEFIT = 8   # ★ 实战有益判定的最低 trd_n
TRD_HARM_THR = 0    # trd_利万 < 此值才算"实买亏"
TRD_BENEFIT_THR = 10  # trd_利万 > 此值才算"实买盈"
N_BOOT = 1000


def bootstrap_ci(arr, n_boot=N_BOOT, ci=95):
    """对样本均值做 bootstrap 置信区间"""
    if len(arr) < N_MIN_CONCL:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    a = np.asarray(arr)
    n = len(a)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = a[idx].mean()
    return float(np.percentile(boots, (100-ci)/2)), float(np.percentile(boots, 100-(100-ci)/2))


def verdict(row):
    """按 SKILL 判定矩阵给出 verdict"""
    n_s, n_t = row['sig_n'], row['trd_n']
    ci_lo, ci_hi = row['ci_lo'], row['ci_hi']
    trd_li = row['trd_利万']
    if n_s < N_MIN_CONCL:
        return '— 不下结论'
    if n_s < N_MIN_LOO:
        return '○ 灰(样本<20)'
    # n_s >= 20
    if not np.isnan(ci_hi) and ci_hi < 0:
        # CI 全负
        if n_t >= N_TRD_HARM and trd_li < TRD_HARM_THR:
            return '✗ 真有害'
        else:
            return '○ sig负但trd未兑'
    if not np.isnan(ci_lo) and ci_lo > 0:
        return '★ 真有益'
    # CI 跨 0
    if n_t >= N_TRD_BENEFIT and trd_li > TRD_BENEFIT_THR:
        return '★ 实战有益'
    if n_t < N_TRD_HARM:
        return '○ 灰区(trd样本少)'
    return '○ 中性'


def main():
    with open(os.path.join(ABL, 'IS_baseline.json'), encoding='utf-8') as f:
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

    print(f'\nIS sig: {len(sigs)} 条 / trd: {len(trades)} 笔')
    print(f'IS baseline 终值: {d["meta"]["final_capital"]/10000:.1f}万')
    matched = trades['pool_retail'].notna().sum()
    print(f'trade↔sig join 匹配: {matched}/{len(trades)}\n')

    def perturb(label_col, label_list, out_csv):
        rows = []
        for gua in sorted(sigs['tian_gua'].unique()):
            sub_s = sigs[sigs['tian_gua'] == gua]
            sub_t = trades[trades['tian_gua'] == gua]
            for lab in label_list:
                cell_s = sub_s[sub_s[label_col] == lab]
                cell_t = sub_t[sub_t[label_col] == lab]
                if len(cell_s) == 0 and len(cell_t) == 0:
                    continue
                ci_lo, ci_hi = bootstrap_ci(cell_s['actual_ret'].values) if len(cell_s) else (np.nan, np.nan)
                row = {
                    'gua': gua,
                    'gua_name': GUA_NAME.get(gua, ''),
                    label_col: lab,
                    'sig_n': len(cell_s),
                    'sig_mean%': cell_s['actual_ret'].mean() if len(cell_s) else np.nan,
                    'sig_win%': (cell_s['actual_ret'] > 0).mean() * 100 if len(cell_s) else np.nan,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'trd_n': len(cell_t),
                    'trd_利万': cell_t['profit_wan'].sum() if len(cell_t) else 0,
                    'trd_mean%': cell_t['ret_pct'].mean() if len(cell_t) else np.nan,
                    'trd_win%': (cell_t['ret_pct'] > 0).mean() * 100 if len(cell_t) else np.nan,
                }
                row['verdict'] = verdict(row)
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(ABL, out_csv), index=False, encoding='utf-8-sig',
                  float_format='%.2f')
        return df

    print('=' * 130)
    print('维度 A: (d_gua × pool_depth) — sig 全量+CI vs trd 实买')
    print('=' * 130)
    df_d = perturb('depth_bucket', DEPTH_LABELS, 'phase2_perturb_depth.csv')
    show_cols = ['gua_name', 'depth_bucket', 'sig_n', 'sig_mean%', 'ci_lo', 'ci_hi',
                 'trd_n', 'trd_利万', 'trd_mean%', 'verdict']
    print(df_d[show_cols].to_string(index=False))

    print('\n' + '=' * 130)
    print('维度 B: (d_gua × pool_days) — sig 全量+CI vs trd 实买')
    print('=' * 130)
    df_t = perturb('days_bucket', DAYS_LABELS, 'phase2_perturb_days.csv')
    show_cols = ['gua_name', 'days_bucket', 'sig_n', 'sig_mean%', 'ci_lo', 'ci_hi',
                 'trd_n', 'trd_利万', 'trd_mean%', 'verdict']
    print(df_t[show_cols].to_string(index=False))

    # 按 verdict 分类落地
    def collect(df, axis):
        out = {}
        for vd in df['verdict'].unique():
            cells = df[df['verdict'] == vd]
            out[vd] = cells.to_dict('records')
        return out

    summary = {
        'IS_baseline_wan': d['meta']['final_capital']/10000,
        'IS_sig_n': len(sigs),
        'IS_trd_n': len(trades),
        'depth_verdicts': collect(df_d, 'depth'),
        'days_verdicts': collect(df_t, 'days'),
    }

    print('\n' + '=' * 130)
    print('verdict 分类汇总')
    print('=' * 130)
    for axis, df in [('depth', df_d), ('days', df_t)]:
        print(f'\n[{axis}]')
        print(df['verdict'].value_counts().to_string())

    print('\n=== ✗ 真有害 cell (送 add-one + LOO) ===')
    bad_d = df_d[df_d['verdict'] == '✗ 真有害']
    bad_t = df_t[df_t['verdict'] == '✗ 真有害']
    print(f'depth: {len(bad_d)}; days: {len(bad_t)}')
    for axis, df in [('depth', bad_d), ('days', bad_t)]:
        if len(df):
            cols = ['gua_name', f'{axis}_bucket', 'sig_n', 'sig_mean%', 'ci_lo', 'ci_hi',
                    'trd_n', 'trd_利万']
            print(f'\n  {axis}:')
            print(df[cols].to_string(index=False))

    out_path = os.path.join(ABL, 'phase2_verdict.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n  落地: {out_path}')


if __name__ == '__main__':
    main()
