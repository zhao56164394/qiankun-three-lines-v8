# -*- coding: utf-8 -*-
"""单股 case study — 000001 平安银行 个股日卦买点

研究问题: 哪些日卦变化 X→Y, 后 60 日 close 上涨 ≥ 5% 的概率高于 baseline?

判定:
  hit = close[i+60] >= close[i] * 1.05
  baseline = 任意一日入场, 60 日后 +5% 的概率

输出:
  - 全期 baseline hit_rate
  - 每个 X→Y 变卦的: n_events, hit_rate, mean_60d_return
  - 高于 baseline 显著的, 列出全部事件日期 + 60d return
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = '000001'
HOLD = 60
THRESH = 0.05

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def gua_label(g):
    return f'{g}{GUA_NAMES.get(g, "?")}'


def main():
    # 读个股日卦
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua'])
    g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'] == CODE].copy()
    g['date'] = g['date'].astype(str)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)

    # 读 close
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'] == CODE].copy()
    p['date'] = p['date'].astype(str)

    df = g.merge(p, on=['date', 'code'], how='inner').sort_values('date').reset_index(drop=True)
    print(f'=== {CODE} 平安银行 ===')
    print(f'数据范围: {df["date"].iloc[0]} ~ {df["date"].iloc[-1]}, {len(df):,} 行\n')

    dates = df['date'].to_numpy()
    closes = df['close'].to_numpy().astype(np.float64)
    guas = df['d_gua'].to_numpy()
    n = len(df)

    # === Baseline: 任意一天入场, 60 日后 +5% 的概率 ===
    valid = np.arange(n - HOLD)
    base_c0 = closes[valid]
    base_c60 = closes[valid + HOLD]
    base_hit = (base_c60 >= base_c0 * (1 + THRESH))
    base_ret = (base_c60 / base_c0 - 1) * 100
    base_hit_rate = base_hit.mean() * 100
    base_mean_ret = base_ret.mean()
    print(f'## Baseline (任意日入场, 60日后 +{THRESH*100:.0f}%)')
    print(f'  样本: {len(base_c0):,}, hit_rate: {base_hit_rate:.1f}%, 均收益: {base_mean_ret:+.2f}%\n')

    # === 找日卦变化事件 ===
    # i 处发生变化: guas[i] != guas[i-1]
    is_change = np.r_[False, guas[1:] != guas[:-1]]
    event_idx_all = np.where(is_change)[0]
    # 必须 i+60 < n
    event_idx = event_idx_all[event_idx_all + HOLD < n]
    print(f'日卦变化事件: {len(event_idx_all):,} (有 {HOLD} 日后续: {len(event_idx):,})\n')

    # 算每个事件的 from→to, c0, c60, hit, ret
    from_arr = guas[event_idx - 1]
    to_arr = guas[event_idx]
    # 过滤 NaN/非 3 位卦码
    valid_pair = np.array([
        isinstance(f, str) and len(f) == 3 and isinstance(t, str) and len(t) == 3
        for f, t in zip(from_arr, to_arr)
    ])
    event_idx = event_idx[valid_pair]
    from_arr = from_arr[valid_pair]
    to_arr = to_arr[valid_pair]
    c0_arr = closes[event_idx]
    c60_arr = closes[event_idx + HOLD]
    ret_arr = (c60_arr / c0_arr - 1) * 100
    hit_arr = (c60_arr >= c0_arr * (1 + THRESH))
    date_arr = dates[event_idx]
    print(f'  过滤后有效事件: {len(event_idx):,}\n')

    # === 按 from→to 聚合 ===
    rows = []
    for f in sorted(set(from_arr)):
        for t in sorted(set(to_arr)):
            if f == t:
                continue
            mask = (from_arr == f) & (to_arr == t)
            n_ev = int(mask.sum())
            if n_ev == 0:
                continue
            hits = int(hit_arr[mask].sum())
            mean_ret = float(ret_arr[mask].mean())
            rows.append({
                'from': f, 'to': t,
                'n_events': n_ev,
                'n_hits': hits,
                'hit_rate': hits / n_ev * 100,
                'mean_ret': mean_ret,
                'lift_vs_base': hits / n_ev * 100 - base_hit_rate,
            })
    rdf = pd.DataFrame(rows).sort_values('hit_rate', ascending=False)

    # === 全部 X→Y 表 ===
    print(f'## 全部 X→Y 变卦统计 (按 hit_rate 降序)')
    print(f'  {"变卦":<14} {"n_ev":>5} {"hit":>4} {"hit%":>6} {"lift":>6} {"均收益%":>8}')
    print('  ' + '-' * 55)
    for _, r in rdf.iterrows():
        arrow = f'{gua_label(r["from"])}→{gua_label(r["to"])}'
        print(f'  {arrow:<14} {int(r["n_events"]):>5} {int(r["n_hits"]):>4} '
              f'{r["hit_rate"]:>5.1f}% {r["lift_vs_base"]:>+5.1f} {r["mean_ret"]:>+7.2f}%')

    # === 候选: lift >= +10% 且 n_events >= 10 ===
    cands = rdf[(rdf['lift_vs_base'] >= 10) & (rdf['n_events'] >= 10)].copy()
    print(f'\n## 候选买点 (lift ≥ +10%, n ≥ 10): {len(cands)} 个')
    if len(cands) == 0:
        print('  无候选 — 该股 d_gua 变化无显著买点模式')
    else:
        for _, r in cands.iterrows():
            f, t = r['from'], r['to']
            mask = (from_arr == f) & (to_arr == t)
            evs = pd.DataFrame({
                'date': date_arr[mask],
                'c0': c0_arr[mask],
                'c60': c60_arr[mask],
                'ret%': ret_arr[mask],
                'hit': hit_arr[mask],
            }).sort_values('date')
            print(f'\n  ### {gua_label(f)}→{gua_label(t)}: n={int(r["n_events"])}, '
                  f'hit_rate={r["hit_rate"]:.1f}% (baseline {base_hit_rate:.1f}%, lift +{r["lift_vs_base"]:.1f}), '
                  f'均收益 {r["mean_ret"]:+.2f}%')
            print(f'  {"日期":<12} {"c0":>7} {"c60":>7} {"60d ret":>9} {"+5%?":>5}')
            for _, e in evs.iterrows():
                mark = '✅' if e['hit'] else '✗'
                print(f'  {e["date"]:<12} {e["c0"]:>7.2f} {e["c60"]:>7.2f} {e["ret%"]:>+8.2f}% {mark:>5}')


if __name__ == '__main__':
    main()
