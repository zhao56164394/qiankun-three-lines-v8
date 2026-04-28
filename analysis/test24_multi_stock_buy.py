# -*- coding: utf-8 -*-
"""多股 case study — 验证 000001 找到的"坤↔巽"模式是否普遍

测试股 (5 只代表大票, 跨行业):
  000001 平安银行 (银行)
  600519 贵州茅台 (白酒)
  600036 招商银行 (银行)
  000858 五粮液   (白酒)
  601318 中国平安 (保险)

每股算:
  - baseline (任意日入场 60日 +5% 概率)
  - 坤→巽 (000→011) hit_rate
  - 巽→坤 (011→000) hit_rate
  - 其他 lift ≥ +10% 且 n ≥ 10 的候选

判定:
  ★ 普遍模式: 该变卦在 ≥4/5 只股 lift ≥ +5% (低门槛留余地)
  ○ 部分: 2-3 只
  ✗ 噪音: 0-1 只
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODES = [
    ('000001', '平安银行'),  # 银行
    ('601398', '工商银行'),  # 银行
    ('600028', '中国石化'),  # 能源
    ('000002', '万科A'),     # 地产
    ('600276', '恒瑞医药'),  # 医药
]
HOLD = 60
THRESH = 0.05

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def gua_label(g):
    return f'{g}{GUA_NAMES.get(g, "?")}'


def analyze_one(g_full, p_full, code):
    """对单只股票算 baseline + 各 X→Y hit_rate. 返回 dict {(f,t): {hit_rate, n, lift, ret}}"""
    g = g_full[g_full['code'] == code].copy().sort_values('date').reset_index(drop=True)
    p = p_full[p_full['code'] == code].copy().sort_values('date').reset_index(drop=True)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values('date').reset_index(drop=True)
    if len(df) < HOLD + 250:
        return None, len(df)

    closes = df['close'].to_numpy().astype(np.float64)
    guas = df['d_gua'].to_numpy()
    n = len(df)

    # baseline
    valid = np.arange(n - HOLD)
    base_c0 = closes[valid]; base_c60 = closes[valid + HOLD]
    base_hit_rate = (base_c60 >= base_c0 * (1 + THRESH)).mean() * 100

    # 事件
    is_change = np.r_[False, guas[1:] != guas[:-1]]
    event_idx = np.where(is_change)[0]
    event_idx = event_idx[event_idx + HOLD < n]
    event_idx = event_idx[event_idx > 0]

    from_arr = guas[event_idx - 1]
    to_arr = guas[event_idx]
    valid_pair = np.array([
        isinstance(f, str) and len(f) == 3 and isinstance(t, str) and len(t) == 3
        for f, t in zip(from_arr, to_arr)
    ])
    event_idx = event_idx[valid_pair]
    from_arr = from_arr[valid_pair]; to_arr = to_arr[valid_pair]
    c0_arr = closes[event_idx]; c60_arr = closes[event_idx + HOLD]
    hit_arr = (c60_arr >= c0_arr * (1 + THRESH))
    ret_arr = (c60_arr / c0_arr - 1) * 100

    stats = {'baseline': base_hit_rate, 'changes': {}, 'n_events_total': len(event_idx)}
    for f in set(from_arr):
        for t in set(to_arr):
            if f == t:
                continue
            mask = (from_arr == f) & (to_arr == t)
            n_ev = int(mask.sum())
            if n_ev < 1:
                continue
            stats['changes'][(f, t)] = {
                'n': n_ev,
                'hit_rate': hit_arr[mask].mean() * 100,
                'mean_ret': ret_arr[mask].mean(),
                'lift': hit_arr[mask].mean() * 100 - base_hit_rate,
            }
    return stats, n


def main():
    print('=== 加载数据 ===')
    g_full = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                              columns=['date', 'code', 'd_gua'])
    g_full['code'] = g_full['code'].astype(str).str.zfill(6)
    target_codes = [c for c, _ in CODES]
    g_full = g_full[g_full['code'].isin(target_codes)].copy()
    g_full['date'] = g_full['date'].astype(str)
    g_full['d_gua'] = g_full['d_gua'].astype(str).str.zfill(3)

    p_full = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                              columns=['date', 'code', 'close'])
    p_full['code'] = p_full['code'].astype(str).str.zfill(6)
    p_full = p_full[p_full['code'].isin(target_codes)].copy()
    p_full['date'] = p_full['date'].astype(str)

    # 跑各股
    results = {}
    print(f'\n## 逐股 baseline + 候选信号 (HOLD={HOLD}日, +{THRESH*100:.0f}% 阈值)\n')
    print(f'  {"股票":<14} {"行":>5} {"baseline":>9} {"事件":>5}')
    print('  ' + '-' * 40)
    for code, name in CODES:
        stats, n_rows = analyze_one(g_full, p_full, code)
        if stats is None:
            print(f'  {code} {name:<10} {n_rows:>5} 数据不足')
            continue
        results[code] = (name, stats)
        print(f'  {code} {name:<10} {n_rows:>5} {stats["baseline"]:>7.1f}% {stats["n_events_total"]:>5}')

    # === 焦点变卦: 坤↔巽 ===
    print(f'\n## 焦点 1: 坤→巽 (000→011)\n')
    print(f'  {"股票":<14} {"n":>4} {"hit%":>6} {"lift":>6} {"均ret%":>7}  {"判定":>4}')
    print('  ' + '-' * 50)
    n_strong = 0
    for code, (name, stats) in results.items():
        c = stats['changes'].get(('000', '011'))
        if c is None or c['n'] < 5:
            print(f'  {code} {name:<10} {(c["n"] if c else 0):>4}  样本不足')
            continue
        mark = '★' if c['lift'] >= 5 else ('✗' if c['lift'] <= -5 else '○')
        if c['lift'] >= 5: n_strong += 1
        print(f'  {code} {name:<10} {c["n"]:>4} {c["hit_rate"]:>5.1f}% {c["lift"]:>+5.1f} {c["mean_ret"]:>+6.2f}%  {mark:>4}')
    print(f'\n  → {n_strong}/{len(results)} 只股 lift ≥ +5%')

    print(f'\n## 焦点 2: 巽→坤 (011→000)\n')
    print(f'  {"股票":<14} {"n":>4} {"hit%":>6} {"lift":>6} {"均ret%":>7}  {"判定":>4}')
    print('  ' + '-' * 50)
    n_strong2 = 0
    for code, (name, stats) in results.items():
        c = stats['changes'].get(('011', '000'))
        if c is None or c['n'] < 5:
            print(f'  {code} {name:<10} {(c["n"] if c else 0):>4}  样本不足')
            continue
        mark = '★' if c['lift'] >= 5 else ('✗' if c['lift'] <= -5 else '○')
        if c['lift'] >= 5: n_strong2 += 1
        print(f'  {code} {name:<10} {c["n"]:>4} {c["hit_rate"]:>5.1f}% {c["lift"]:>+5.1f} {c["mean_ret"]:>+6.2f}%  {mark:>4}')
    print(f'\n  → {n_strong2}/{len(results)} 只股 lift ≥ +5%')

    # === 全市场: 哪些变卦在 ≥3 只股都 lift ≥ +5% ===
    print(f'\n## 全部 X→Y 普遍性扫描 (在 ≥{len(results)} 只股出现, 算共识 lift)\n')
    # 收集每个 (f,t) 在各股的 (lift, n)
    all_pairs = {}
    for code, (name, stats) in results.items():
        for ft, c in stats['changes'].items():
            if c['n'] < 5:
                continue
            all_pairs.setdefault(ft, []).append((code, c['lift'], c['hit_rate'], c['n']))

    # 只留出现在 ≥3 只股的
    rows = []
    for ft, lst in all_pairs.items():
        if len(lst) < 3:
            continue
        lifts = [x[1] for x in lst]
        n_pos = sum(1 for L in lifts if L >= 5)
        n_neg = sum(1 for L in lifts if L <= -5)
        rows.append({
            'from': ft[0], 'to': ft[1],
            'n_stocks': len(lst),
            'mean_lift': np.mean(lifts),
            'n_pos': n_pos, 'n_neg': n_neg,
            'detail': lst,
        })
    if not rows:
        print('  无候选 (≥3 只股出现的变卦) — 单股模式不普遍')
        return
    rdf = pd.DataFrame(rows).sort_values('mean_lift', ascending=False)
    print(f'  {"变卦":<14} {"n股":>4} {"+5%":>3} {"-5%":>3} {"均lift":>7}')
    print('  ' + '-' * 40)
    for _, r in rdf.iterrows():
        arrow = f'{gua_label(r["from"])}→{gua_label(r["to"])}'
        print(f'  {arrow:<14} {int(r["n_stocks"]):>4} {int(r["n_pos"]):>3} {int(r["n_neg"]):>3} {r["mean_lift"]:>+6.1f}')

    # 真普遍模式: ≥4 只 lift ≥ +5%
    real = rdf[rdf['n_pos'] >= 4]
    print(f'\n## ★ 真普遍买点 (≥4 只股 lift ≥ +5%): {len(real)} 个')
    for _, r in real.iterrows():
        arrow = f'{gua_label(r["from"])}→{gua_label(r["to"])}'
        print(f'\n  {arrow}: {int(r["n_pos"])}/{int(r["n_stocks"])} 只 lift ≥ +5%, 均 lift {r["mean_lift"]:+.1f}')
        for code, lift, hr, n in r['detail']:
            name = dict(CODES).get(code, code)
            mark = '★' if lift >= 5 else ('✗' if lift <= -5 else '○')
            print(f'    {code} {name:<10} n={n:>3} hit={hr:>5.1f}% lift={lift:>+5.1f}  {mark}')


if __name__ == '__main__':
    main()
