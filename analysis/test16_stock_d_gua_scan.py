# -*- coding: utf-8 -*-
"""个股 d_gua 变卦扫描 (找真买卖点)

每只股 d_gua 切换 X→Y 的事件 → 该股 N 天后收益
对 56 种变卦 × 5 持有期 = 280 候选, 每个候选汇总所有股的事件

关键: 这是寻找"个股自己变卦"作买卖点的真信号 (用户架构: 买卖点=个股变卦, 大盘卦做分治/过滤)
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
HOLD_DAYS = [1, 3, 5, 10, 20]


def load_data():
    print('=== 加载数据 ===')
    t0 = time.time()
    stk_g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                             columns=['date', 'code', 'd_gua'])
    stk_g['date'] = stk_g['date'].astype(str)
    stk_g['code'] = stk_g['code'].astype(str).str.zfill(6)
    stk_g['d_gua'] = stk_g['d_gua'].astype(str).str.zfill(3)
    print(f'  stock gua: {len(stk_g):,} 行, {time.time()-t0:.1f}s')

    t0 = time.time()
    stk_p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                            columns=['date', 'code', 'close'])
    stk_p['date'] = stk_p['date'].astype(str)
    stk_p['code'] = stk_p['code'].astype(str).str.zfill(6)
    print(f'  stock close: {len(stk_p):,} 行, {time.time()-t0:.1f}s')

    # merge gua + close
    t0 = time.time()
    df = stk_g.merge(stk_p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  merged: {len(df):,} 行, {time.time()-t0:.1f}s')

    # 按 code 分组取 (date, gua, close) numpy 数组
    print('  building per-code arrays...')
    t0 = time.time()
    code_data = {}
    for code, g in df.groupby('code'):
        code_data[code] = (g['d_gua'].values, g['close'].values)
    print(f'  {len(code_data)} 只票, {time.time()-t0:.1f}s')
    return code_data


def baseline_random_returns(code_data, hold_days, n_samples=10000, seed=42):
    rng = np.random.RandomState(seed)
    codes = list(code_data.keys())
    rets = []
    while len(rets) < n_samples:
        code = rng.choice(codes)
        guas, closes = code_data[code]
        if len(closes) <= hold_days + 1:
            continue
        i = rng.randint(0, len(closes) - hold_days - 1)
        c0 = closes[i]; c1 = closes[i + hold_days]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return rets


def boot_alpha_ci(event_rets, base_rets, n_boot=1000, seed=42):
    if len(event_rets) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    n_event = len(event_rets); n_base = len(base_rets)
    boots = np.empty(n_boot)
    e = np.asarray(event_rets); b = np.asarray(base_rets)
    for i in range(n_boot):
        r1 = e[rng.randint(0, n_event, n_event)].mean()
        r2 = b[rng.randint(0, n_base, n_base)].mean()
        boots[i] = r1 - r2
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    code_data = load_data()
    print('\n=== 计算各持有期基线 ===')
    base_by_h = {}
    for h in HOLD_DAYS:
        base_by_h[h] = baseline_random_returns(code_data, h)
        print(f'  hold={h}d: 基线均值 {np.mean(base_by_h[h]):+.2f}%, n={len(base_by_h[h])}')

    print('\n=== 扫描个股 d_gua 变卦事件 ===')
    rows = []
    GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

    # 一次性: 找出每个 (from→to) 变卦下所有 (code, idx) 事件
    print('  收集变卦事件...')
    t0 = time.time()
    all_events = {}  # (from, to) -> list of (code, idx)
    for code, (guas, closes) in code_data.items():
        for i in range(1, len(guas)):
            f = guas[i - 1]; t = guas[i]
            if f == t or f not in GUAS or t not in GUAS:
                continue
            all_events.setdefault((f, t), []).append((code, i))
    print(f'  {sum(len(v) for v in all_events.values()):,} 总事件, '
          f'{len(all_events)} 种变卦, {time.time()-t0:.1f}s')

    for (f, t), events in all_events.items():
        if len(events) < 30:
            continue
        for h in HOLD_DAYS:
            rets = []
            for code, i in events:
                guas, closes = code_data[code]
                if i + h >= len(closes):
                    continue
                c0 = closes[i]; c1 = closes[i + h]
                if c0 > 0:
                    rets.append((c1 / c0 - 1) * 100)
            if len(rets) < 30:
                continue
            # 限制样本到 50000 加速 bootstrap
            if len(rets) > 50000:
                rng = np.random.RandomState(42)
                rets = rng.choice(rets, 50000, replace=False).tolist()
            base = base_by_h[h]
            alpha = np.mean(rets) - np.mean(base)
            ci_lo, ci_hi = boot_alpha_ci(rets, base)
            if ci_lo is None:
                continue
            win = (np.array(rets) > 0).mean() * 100
            verdict = '★买点' if ci_lo > 0 else ('✗卖点' if ci_hi < 0 else '○灰区')
            rows.append({
                'from': f, 'to': t, 'hold': h, 'n_events': len(events), 'n_rets': len(rets),
                'event_mean': np.mean(rets), 'alpha': alpha,
                'ci_lo': ci_lo, 'ci_hi': ci_hi, 'win_rate': win, 'verdict': verdict,
            })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, 'data_layer/data/ablation/test16_stock_d_gua_scan')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'stock_d_scan.csv'), index=False, encoding='utf-8-sig',
              float_format='%.3f')
    print(f'\n落地: {out_dir}/stock_d_scan.csv ({len(df)} 行)')

    # 跨持有期一致性: 同一个 (from, to), 5 个 hold 全部 ★ 或 全部 ✗
    print('\n' + '=' * 100)
    print('# 跨持有期 5/5 一致 真信号 (按个股 d_gua)')
    print('=' * 100)

    grouped = df.groupby(['from', 'to'])
    consistent = []
    for (f, t), g in grouped:
        if len(g) < 5: continue  # 不足 5 个 hold 的不评估
        verdicts = g['verdict'].tolist()
        n_buy = sum(1 for v in verdicts if v == '★买点')
        n_sell = sum(1 for v in verdicts if v == '✗卖点')
        if n_buy >= 4:  # 5/5 或 4/5 ★
            consistent.append({'from': f, 'to': t, 'cons_type': '★买',
                                'n_consistent': n_buy, 'avg_alpha': g['alpha'].mean(),
                                'group_data': g})
        elif n_sell >= 4:
            consistent.append({'from': f, 'to': t, 'cons_type': '✗卖',
                                'n_consistent': n_sell, 'avg_alpha': g['alpha'].mean(),
                                'group_data': g})

    # 真买点
    buys = [c for c in consistent if c['cons_type'] == '★买']
    sells = [c for c in consistent if c['cons_type'] == '✗卖']
    buys.sort(key=lambda x: -x['avg_alpha'])
    sells.sort(key=lambda x: x['avg_alpha'])

    print(f'\n## ★ 真买点候选 (n={len(buys)})')
    print(f'  {"变卦":<10} {"事件 N":>8} {"一致":>5} {"均α":>7}  {"1d":>7} {"3d":>7} {"5d":>7} {"10d":>7} {"20d":>7}')
    print('  ' + '-' * 90)
    for c in buys:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        g = c['group_data']
        events_n = g.iloc[0]['n_events']
        cells = []
        for h in HOLD_DAYS:
            row = g[g['hold'] == h]
            if len(row) == 0:
                cells.append(f'{"-":>7}')
            else:
                v = row.iloc[0]
                ci_lo = v['ci_lo']
                marker = '★' if ci_lo > 0 else ('✗' if v['ci_hi'] < 0 else '○')
                cells.append(f'{v["alpha"]:>+5.2f}{marker}')
        print(f'  {ct:<12} {int(events_n):>8} {c["n_consistent"]:>5}/5 {c["avg_alpha"]:>+6.2f} '
              f'  {" ".join(cells)}')

    print(f'\n## ✗ 真卖点候选 (n={len(sells)})')
    print(f'  {"变卦":<10} {"事件 N":>8} {"一致":>5} {"均α":>7}  {"1d":>7} {"3d":>7} {"5d":>7} {"10d":>7} {"20d":>7}')
    print('  ' + '-' * 90)
    for c in sells:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        g = c['group_data']
        events_n = g.iloc[0]['n_events']
        cells = []
        for h in HOLD_DAYS:
            row = g[g['hold'] == h]
            if len(row) == 0:
                cells.append(f'{"-":>7}')
            else:
                v = row.iloc[0]
                ci_lo = v['ci_lo']
                marker = '★' if ci_lo > 0 else ('✗' if v['ci_hi'] < 0 else '○')
                cells.append(f'{v["alpha"]:>+5.2f}{marker}')
        print(f'  {ct:<12} {int(events_n):>8} {c["n_consistent"]:>5}/5 {c["avg_alpha"]:>+6.2f} '
              f'  {" ".join(cells)}')


if __name__ == '__main__':
    main()
