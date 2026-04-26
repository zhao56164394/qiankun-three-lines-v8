# -*- coding: utf-8 -*-
"""8 卦 × 8 y_gua 双视角扰动分析 — 标定每分支的 gate_disable_y_gua

数据源:
  - data_layer/data/backtest_8gua_naked_result.json (含 gate_y={101,110,111} for 离)
  - data_layer/data/foundation/multi_scale_gua_daily.csv

双视角:
  全量 (signal_detail): actual_ret + 95% bootstrap CI
  实买 (trade_log):     ret_pct + profit (受 5 仓资金挤压)

判定规则 (双验证):
  关火格 (d, y): 全量 CI 全负 AND 实买利润 < 0
  保留格:        全量 CI 全正 OR  实买利润 > 0 (任一为正就保)
  灰区:          全量 CI 触 0 / 实买利润接近 0 → 保留 (保守)

输出每个 d_gua 的 gate_disable_y_gua 推荐集合.
"""
import json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NAKED = os.path.join(ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')
MULTI = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

GUA = ['000','001','010','011','100','101','110','111']
NAME = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}
MEAN = {'000':'深熊探底','001':'熊底异动','010':'反弹乏力','011':'底部爆发',
        '100':'崩盘加速','101':'下跌护盘','110':'牛末滞涨','111':'疯牛主升'}


def load():
    with open(NAKED, encoding='utf-8') as f: d = json.load(f)
    sig = pd.DataFrame(d['signal_detail'])
    trd = pd.DataFrame(d['trade_log'])
    sig = sig[~sig['is_skip']].copy()
    sig['tian_gua'] = sig['tian_gua'].astype(str).str.zfill(3)
    trd['gua'] = trd['gua'].astype(str).str.zfill(3)
    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'y_gua':str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
    y_map = dict(zip(ms['date'], ms['y_gua'].fillna('').astype(str).str.zfill(3)))
    sig['y_gua'] = sig['signal_date'].map(y_map).fillna('')
    trd['y_gua'] = trd['buy_date'].map(y_map).fillna('')
    return sig, trd


def boot_ci(arr, n=2000, seed=42):
    arr = np.asarray(arr, dtype=float); arr = arr[~np.isnan(arr)]
    if len(arr) < 5: return (np.mean(arr) if len(arr) else np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0,len(arr),len(arr))].mean() for _ in range(n)])
    return (arr.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975))


def classify(sig_n, sig_mean, sig_lo, sig_hi, trd_n, trd_profit, trd_mean):
    """决定一个 (d,y) 格是 keep / disable / gray
    Rules (双验证):
      disable: 全量 n>=20 且 CI 全负   AND  (实买 n>=5 且 profit<0)
               OR  (全量 n>=30 且 mean<-3 且 hi<0)
      keep:    全量 mean>0 (且 CI 不全负)  OR  实买 profit>10000
      gray:    其他 (保留)
    """
    # 大样本全量 CI 全负
    sig_clear_neg = (sig_n >= 20 and not np.isnan(sig_hi) and sig_hi < 0)
    sig_clear_pos = (sig_n >= 20 and not np.isnan(sig_lo) and sig_lo > 0)
    # 实买视角
    trd_neg = (trd_n >= 5 and trd_profit < 0)
    trd_pos = (trd_n >= 5 and trd_profit > 10000)  # >1万才算"显著正"
    if sig_clear_neg and trd_neg:
        return 'disable'
    if sig_clear_neg and trd_n < 5:
        # 全量已显著负, 实买无样本 → 仍关 (因为旧 gate 关掉了实买)
        return 'disable'
    if sig_clear_pos or trd_pos:
        return 'keep'
    # 看绝对均值: 如果两边都 < 0 (任一样本足) 也关
    if sig_n >= 30 and sig_mean < -3 and not np.isnan(sig_hi) and sig_hi < 0:
        return 'disable'
    return 'gray'


def main():
    sig, trd = load()
    print(f'\n  全量 sig: {len(sig)},  实买 trd: {len(trd)}')

    print('\n' + '='*120)
    print('  d_gua × y_gua 双视角扰动表 (全量 sig + 实买 trd)')
    print('  每格: signal_n / sig_mean% [CI lo, hi] | trade_n / trade_profit万 / trade_mean%')
    print('='*120)

    recommend = {}
    for d in GUA:
        print(f'\n  ====  d_gua = {d} {NAME[d]} ({MEAN[d]})  ====')
        sub_s = sig[sig['tian_gua']==d]
        sub_t = trd[trd['gua']==d]
        disable_set = set()
        keep_set = set()
        gray_set = set()
        for y in GUA:
            ssub = sub_s[sub_s['y_gua']==y]
            tsub = sub_t[sub_t['y_gua']==y]
            sn = len(ssub); tn = len(tsub)
            sm, slo, shi = boot_ci(ssub['actual_ret'].values) if sn else (np.nan, np.nan, np.nan)
            tp = tsub['profit'].sum() if tn else 0
            tm = tsub['ret_pct'].mean() if tn else np.nan
            verdict = classify(sn, sm, slo, shi, tn, tp, tm)
            if verdict == 'disable': disable_set.add(y)
            elif verdict == 'keep':  keep_set.add(y)
            else:                    gray_set.add(y)
            ci_str = (f'[{slo:+5.2f}, {shi:+5.2f}]' if not np.isnan(slo) else '   ---  ')
            sig_str = f'{sn:>4}/{sm:+5.1f}% {ci_str}' if sn else f'{0:>4}/   -                  '
            trd_str = f'{tn:>3}/{tp/10000:+5.1f}万/{tm:+5.1f}%' if tn else f'{0:>3}/  -    /  -    '
            mark = {'disable':' ✗ 关', 'keep':' ★ 留', 'gray':'   '}[verdict]
            print(f'    y={y} {NAME[y]} ({MEAN[y]:<4})  全 {sig_str}  |  实 {trd_str}{mark}')
        recommend[d] = {'disable': disable_set, 'keep': keep_set, 'gray': gray_set}
        if disable_set:
            print(f'  → 推荐 gate_disable_y_gua = {sorted(disable_set)}')
        else:
            print(f'  → 不需要 y_gua gate')

    print('\n' + '='*100)
    print('  汇总建议 (复制到 GUA_STRATEGY)')
    print('='*100)
    for d in GUA:
        rec = recommend[d]
        if rec['disable']:
            ds = "{" + ', '.join(f"'{x}'" for x in sorted(rec['disable'])) + "}"
            print(f"  '{d}': # {NAME[d]} ({MEAN[d]})")
            print(f"      'gate_disable_y_gua': {ds},")
        else:
            print(f"  '{d}': # {NAME[d]} — 无需 gate")


if __name__ == '__main__':
    main()
