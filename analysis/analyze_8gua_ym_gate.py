# -*- coding: utf-8 -*-
"""8 卦 × 64 (y_gua, m_gua) cell 双视角扰动分析

回答两个问题:
  Q1. 当前 y 单维 gate 的"误关": 关掉的 y_gua 里是否有 m=金矿月被一起牺牲?
  Q2. 当前 y 单维 gate 的"漏关": 没关的 y_gua 里是否有 m=明确坏月被漏掉?

数据源: backtest_8gua_naked_result.json (这版已含 5 卦 gate)
        + multi_scale_gua_daily.csv (m_gua / y_gua)

注意: 现在数据里 y=101/110 等已被部分 gate 关掉, 信号偏少;
要看完整的 (y, m) 响应, 需要拿一个"无 gate"的 baseline. 这里先看现有数据,
对显示"被关"的格子用 ⚠ 标记 (没法重新评估 — 需重跑无 gate baseline 才能看)
"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.no_gate.json')
MULTI = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA = ['000','001','010','011','100','101','110','111']
NAME = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}


def load():
    with open(NAKED, encoding='utf-8') as f: d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    trd = pd.DataFrame(d['trade_log'])
    sig = sig[~sig['is_skip']].copy()
    sig['tian_gua'] = sig['tian_gua'].astype(str).str.zfill(3)
    trd['gua'] = trd['gua'].astype(str).str.zfill(3)
    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'y_gua':str,'m_gua':str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
    y_map = dict(zip(ms['date'], ms['y_gua'].fillna('').astype(str).str.zfill(3)))
    m_map = dict(zip(ms['date'], ms['m_gua'].fillna('').astype(str).str.zfill(3)))
    sig['y_gua'] = sig['signal_date'].map(y_map).fillna('')
    sig['m_gua'] = sig['signal_date'].map(m_map).fillna('')
    trd['y_gua'] = trd['buy_date'].map(y_map).fillna('')
    trd['m_gua'] = trd['buy_date'].map(m_map).fillna('')
    return sig, trd


def boot_ci(arr, n=2000, seed=42):
    arr = np.asarray(arr, dtype=float); arr = arr[~np.isnan(arr)]
    if len(arr) < 5: return (np.mean(arr) if len(arr) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0,len(arr),len(arr))].mean() for _ in range(n)])
    return (arr.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975))


def cell_summary(sig_df, trd_df, d, y, m):
    s = sig_df[(sig_df['tian_gua']==d) & (sig_df['y_gua']==y) & (sig_df['m_gua']==m)]
    t = trd_df[(trd_df['gua']==d) & (trd_df['y_gua']==y) & (trd_df['m_gua']==m)]
    sn = len(s); tn = len(t)
    sm, slo, shi = boot_ci(s['actual_ret'].values) if sn else (np.nan, np.nan, np.nan)
    tp = t['profit'].sum() / 10000 if tn else 0.0
    tm = t['ret_pct'].mean() if tn else np.nan
    return sn, sm, slo, shi, tn, tp, tm


def classify(sn, sm, slo, shi, tn, tp):
    """复用 stage 2 的判定 — 双验证为负才关"""
    sig_neg = (sn >= 20 and not np.isnan(shi) and shi < 0)
    sig_pos = (sn >= 20 and not np.isnan(slo) and slo > 0)
    trd_pos = (tn >= 5 and tp > 1)  # >1 万
    trd_neg = (tn >= 5 and tp < 0)
    if sig_neg and (trd_neg or tn < 5):
        return 'disable'
    if sig_pos or trd_pos:
        return 'keep'
    if sn >= 30 and not np.isnan(sm) and sm < -3 and not np.isnan(shi) and shi < 0:
        return 'disable'
    return 'gray'


# 当前 5 卦 gate
CURRENT_GATE = {
    '000': {'101','110'},
    '001': {'011','101'},
    '010': {'101'},
    '011': {'101'},
    '101': {'101','110','111'},
}


def main():
    sig, trd = load()
    print(f'\n  数据: 全量 sig={len(sig)}, 实买 trd={len(trd)}')
    print(f'  注意: 此 baseline 已含 5 卦 y_gua gate, 部分 cell 已被关 (n=0 也可能是被关导致)')

    print('\n' + '='*120)
    print('  各 d_gua 分支的 (y, m) cell 双视角扰动表')
    print('  ⊘ = 当前 y_gua gate 已关; ✗ = 推荐关 (双验证负); ★ = 推荐留 (显著正)')
    print('='*120)

    new_recommend = {}
    for d in GUA:
        print(f'\n  ====  d_gua = {d} {NAME[d]}  ====')
        cur_gate_y = CURRENT_GATE.get(d, set())
        # 排序: n 多在前
        all_cells = []
        for y in GUA:
            for m in GUA:
                sn, sm, slo, shi, tn, tp, tm = cell_summary(sig, trd, d, y, m)
                if sn == 0 and tn == 0:
                    continue
                v = classify(sn, sm, slo, shi, tn, tp)
                all_cells.append((y, m, sn, sm, slo, shi, tn, tp, tm, v))
        all_cells.sort(key=lambda x: -x[2])

        disable_ym = set()
        keep_ym = set()
        for y, m, sn, sm, slo, shi, tn, tp, tm, v in all_cells:
            already_gated = y in cur_gate_y
            mark = ''
            if already_gated and sn == 0:
                mark = ' ⊘ 当前已关'
            elif v == 'disable':
                mark = ' ✗ 推荐关'
                disable_ym.add((y, m))
            elif v == 'keep':
                mark = ' ★ 显著留'
                keep_ym.add((y, m))
            ci_str = (f'[{slo:+5.1f},{shi:+5.1f}]' if not np.isnan(slo) else '   ----   ')
            sig_str = f'{sn:>4}/{sm:+5.1f}%' if sn else '   0/  -  '
            trd_str = f'{tn:>2}笔/{tp:+5.1f}万' if tn else '0笔/  -  '
            print(f'    y={y}{NAME[y]} m={m}{NAME[m]} | 全 {sig_str} {ci_str} | 实 {trd_str}{mark}')

        new_recommend[d] = {'disable_ym': disable_ym, 'keep_ym': keep_ym, 'cur_gate_y': cur_gate_y}

    print('\n' + '='*120)
    print('  对比: 当前 y 单维 gate vs 推荐 (y, m) 联合 gate')
    print('='*120)
    for d in GUA:
        rec = new_recommend[d]
        cur = rec['cur_gate_y']
        new_ym = rec['disable_ym']
        # 当前 y 关掉的所有 (y, *) cell 数
        cur_cells = [(y, m) for y in cur for m in GUA]
        # 推荐新增 (除了当前 y 已关的)
        new_extra = {(y, m) for (y, m) in new_ym if y not in cur}
        new_remove = set()  # 当前 y 关里, 联合视角发现的 keep 格 (即 y 误关)
        for (y, m) in rec['keep_ym']:
            if y in cur:
                new_remove.add((y, m))

        print(f'\n  d={d} {NAME[d]}:')
        print(f'    当前 y 单维: {sorted(cur)} (覆盖 {len(cur)*8} 个 (y,m) cell)')
        print(f'    联合视角发现的新增坏格 (y 不在当前 gate): {sorted(new_extra)}')
        print(f'    联合视角发现的当前 y 误关的好格 (y, m): {sorted(new_remove)}')


if __name__ == '__main__':
    main()
