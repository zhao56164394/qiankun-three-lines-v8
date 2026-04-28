# -*- coding: utf-8 -*-
"""6 卦流变卦全扫 — 全向量化 (高性能版)

6 流: 大盘 d/m/y_gua, 个股 d/m/y_gua
56 变卦 × 5 持有期 × 6 流 = 1680 候选

全向量化关键:
  - 个股流: 一次性把每只票按 code 拼成连续 numpy 数组, vectorized fwd return
  - 大盘流: 大盘事件日期 → 用 searchsorted 一次性匹配所有股票的同日索引
  - bootstrap: 一次 1000×n_event 矩阵抽样, 一次 .mean(axis=1)
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
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']
HOLD_DAYS = [1, 3, 5, 10, 20]


def boot_alpha_ci_vec(event_rets, base_rets, n_boot=1000, seed=42):
    """向量化 bootstrap. 一次抽样 n_boot×N 矩阵."""
    if len(event_rets) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    e = np.asarray(event_rets, dtype=np.float32)
    b = np.asarray(base_rets, dtype=np.float32)
    n_event = len(e); n_base = len(b)
    # 一次性矩阵抽样
    idx_e = rng.randint(0, n_event, size=(n_boot, n_event))
    idx_b = rng.randint(0, n_base, size=(n_boot, n_base))
    boots = e[idx_e].mean(axis=1) - b[idx_b].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_market():
    print('=== 加载大盘卦 ===')
    m = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                        columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    m['date'] = m['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        m[c] = m[c].astype(str).str.zfill(3)
    m = m.sort_values('date').reset_index(drop=True)
    return m


def load_stocks():
    print('=== 加载个股卦+close ===')
    t0 = time.time()
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    print(f'  gua: {len(g):,} 行, {time.time()-t0:.1f}s')
    t0 = time.time()
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  merged: {len(df):,} 行, {time.time()-t0:.1f}s')
    return df


def build_baseline(close_arr, hold, n=10000, seed=42):
    """从 close_arr (1D 拼接所有股, 但这样跨股, 不对).
    简化: 随机抽 (code, idx), 但全表已按 code 拼接, idx 不能跨 code"""
    pass  # use stk df group


def fwd_returns_vectorized(closes_arr, idx_arr, hold):
    """向量化算 idx_arr 中每个 i 的 N 天后收益.
    closes_arr: float32 numpy array (一只股的 close 序列)
    idx_arr: 事件位置 indices (在 closes_arr 中)
    返回: rets numpy array (跳过越界)
    """
    n = len(closes_arr)
    valid = idx_arr[idx_arr + hold < n]
    if len(valid) == 0:
        return np.array([], dtype=np.float32)
    c0 = closes_arr[valid]
    c1 = closes_arr[valid + hold]
    mask = c0 > 0
    return ((c1[mask] / c0[mask] - 1.0) * 100.0).astype(np.float32)


def baseline_random_per_stock(stk_df, hold, n_samples=10000, seed=42):
    """从随机抽 (code, idx) 算 baseline forward returns"""
    rng = np.random.RandomState(seed)
    codes = stk_df['code'].unique()
    rets = []
    code_to_arr = {c: g['close'].values.astype(np.float32) for c, g in stk_df.groupby('code')}
    while len(rets) < n_samples:
        code = rng.choice(codes)
        arr = code_to_arr[code]
        if len(arr) <= hold + 1:
            continue
        i = rng.randint(0, len(arr) - hold - 1)
        c0 = arr[i]; c1 = arr[i + hold]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return np.array(rets, dtype=np.float32)


def baseline_market(market, hold, n_samples=10000, seed=42):
    """大盘流 baseline: 随机抽 (date) → 该日所有股的 forward return"""
    pass  # 直接复用 individual stock baseline (上面)


def scan_stock_self_streams(stk_df, baseline_by_h):
    """个股自己的 d/m/y_gua 变卦 → 该股自己的 forward return"""
    print('\n=== 扫描 个股自变卦流 (3 流: d/m/y_gua) ===')
    rows = []

    for stream_name, gua_col in [('个股日卦', 'd_gua'), ('个股月卦', 'm_gua'), ('个股年卦', 'y_gua')]:
        print(f'\n  -- {stream_name} ({gua_col}) --')
        t0 = time.time()
        # 一次性收集全部 56 变卦的事件 (idx, code) tuple
        # 思路: 把每只票的 (gua序列, close序列) 拿出, vectorized 找变卦事件
        events_by_change = {}  # (from, to) -> list of (code, idx_in_code)
        for code, g in stk_df.groupby('code', sort=False):
            guas = g[gua_col].values
            # find boundaries: i where guas[i] != guas[i-1]
            for i in range(1, len(guas)):
                if guas[i] != guas[i - 1] and guas[i] in GUAS and guas[i - 1] in GUAS:
                    events_by_change.setdefault((guas[i - 1], guas[i]), []).append((code, i))
        print(f'    收集事件: {sum(len(v) for v in events_by_change.values()):,} '
              f'({len(events_by_change)} 种变卦), {time.time()-t0:.1f}s')

        # vectorize: 对每个变卦, 一次性算所有事件的 N 天后收益
        # 把事件按 code 分组, 用 code-level numpy arrays 算
        code_to_arr = {c: g['close'].values.astype(np.float32) for c, g in stk_df.groupby('code', sort=False)}

        for (f, t), evts in events_by_change.items():
            if len(evts) < 30:
                continue
            # 按 code 聚合 idx
            code_idx = {}
            for code, i in evts:
                code_idx.setdefault(code, []).append(i)
            for h in HOLD_DAYS:
                rets_all = []
                for code, idx_list in code_idx.items():
                    arr = code_to_arr[code]
                    rets_all.append(fwd_returns_vectorized(arr, np.array(idx_list), h))
                rets_all = np.concatenate(rets_all) if rets_all else np.array([])
                if len(rets_all) < 30:
                    continue
                # cap 到 50000 加速
                if len(rets_all) > 50000:
                    rng = np.random.RandomState(42)
                    rets_all = rng.choice(rets_all, 50000, replace=False)
                base = baseline_by_h[h]
                alpha = float(rets_all.mean() - base.mean())
                ci_lo, ci_hi = boot_alpha_ci_vec(rets_all, base)
                if ci_lo is None: continue
                win = float((rets_all > 0).mean() * 100)
                verdict = '★买点' if ci_lo > 0 else ('✗卖点' if ci_hi < 0 else '○灰区')
                rows.append({
                    'stream': stream_name, 'gua_col': gua_col,
                    'from': f, 'to': t, 'hold': h, 'n_events': len(evts),
                    'n_rets': len(rets_all), 'event_mean': float(rets_all.mean()),
                    'alpha': alpha, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    'win_rate': win, 'verdict': verdict,
                })
        print(f'    扫描完成: {time.time()-t0:.1f}s')
    return rows


def scan_market_streams(market, stk_df, baseline_by_h):
    """大盘 d/m/y_gua 变卦 → 当日所有股 N 天后收益"""
    print('\n=== 扫描 大盘流 (3 流: d/m/y_gua) ===')
    rows = []
    code_to_arr = {c: (g['date'].values, g['close'].values.astype(np.float32))
                   for c, g in stk_df.groupby('code', sort=False)}

    for stream_name, gua_col in [('大盘日卦', 'd_gua'), ('大盘月卦', 'm_gua'), ('大盘年卦', 'y_gua')]:
        print(f'\n  -- {stream_name} ({gua_col}) --')
        t0 = time.time()
        m = market.sort_values('date').reset_index(drop=True)
        m['prev'] = m[gua_col].shift(1)

        # 收集每个 (from, to) 的事件日期
        events_by_change = {}
        for f in GUAS:
            for t in GUAS:
                if f == t: continue
                ev_dates = m.loc[(m['prev'] == f) & (m[gua_col] == t), 'date'].values
                if len(ev_dates) >= 5:
                    events_by_change[(f, t)] = ev_dates
        print(f'    收集 {sum(len(v) for v in events_by_change.values())} 大盘事件日, '
              f'{len(events_by_change)} 种变卦, {time.time()-t0:.1f}s')

        for (f, t), ev_dates in events_by_change.items():
            ev_set = set(ev_dates.tolist())
            for h in HOLD_DAYS:
                rets_all = []
                for code, (dates, closes) in code_to_arr.items():
                    # 找 dates 中匹配 ev_set 的位置
                    idx_match = np.array([i for i, d in enumerate(dates) if d in ev_set])
                    if len(idx_match) == 0: continue
                    rets_all.append(fwd_returns_vectorized(closes, idx_match, h))
                rets_all = np.concatenate(rets_all) if rets_all else np.array([])
                if len(rets_all) < 30: continue
                if len(rets_all) > 50000:
                    rng = np.random.RandomState(42)
                    rets_all = rng.choice(rets_all, 50000, replace=False)
                base = baseline_by_h[h]
                alpha = float(rets_all.mean() - base.mean())
                ci_lo, ci_hi = boot_alpha_ci_vec(rets_all, base)
                if ci_lo is None: continue
                win = float((rets_all > 0).mean() * 100)
                verdict = '★买点' if ci_lo > 0 else ('✗卖点' if ci_hi < 0 else '○灰区')
                rows.append({
                    'stream': stream_name, 'gua_col': gua_col,
                    'from': f, 'to': t, 'hold': h, 'n_events': len(ev_dates),
                    'n_rets': len(rets_all), 'event_mean': float(rets_all.mean()),
                    'alpha': alpha, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    'win_rate': win, 'verdict': verdict,
                })
        print(f'    扫描完成: {time.time()-t0:.1f}s')
    return rows


def report_consistent(df, stream_filter=None):
    """跨持有期一致性: 同 (stream, from, to) 至少 4/5 同向"""
    df_use = df if stream_filter is None else df[df['stream'].isin(stream_filter)]
    grouped = df_use.groupby(['stream', 'from', 'to'])
    consistent = []
    for (s, f, t), g in grouped:
        if len(g) < 4: continue
        n_buy = (g['verdict'] == '★买点').sum()
        n_sell = (g['verdict'] == '✗卖点').sum()
        if n_buy >= 4:
            consistent.append({'stream': s, 'from': f, 'to': t, 'cons_type': '★买',
                                'n_consistent': n_buy, 'avg_alpha': g['alpha'].mean(),
                                'group_data': g})
        elif n_sell >= 4:
            consistent.append({'stream': s, 'from': f, 'to': t, 'cons_type': '✗卖',
                                'n_consistent': n_sell, 'avg_alpha': g['alpha'].mean(),
                                'group_data': g})
    return consistent


def main():
    t_all = time.time()
    market = load_market()
    stk = load_stocks()
    print(f'\n=== 计算各持有期基线 ===')
    base_by_h = {}
    for h in HOLD_DAYS:
        base_by_h[h] = baseline_random_per_stock(stk, h)
        print(f'  hold={h}d: 基线均值 {base_by_h[h].mean():+.2f}%')

    rows = []
    rows.extend(scan_stock_self_streams(stk, base_by_h))
    rows.extend(scan_market_streams(market, stk, base_by_h))

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, 'data_layer/data/ablation/test17_yao_change_full_scan')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'full_scan.csv'), index=False, encoding='utf-8-sig',
              float_format='%.3f')
    print(f'\n落地: {out_dir}/full_scan.csv ({len(df)} 行)')

    # 跨期一致性报告
    print('\n' + '=' * 100)
    print('# 跨持有期一致性 - 个股自变卦流 (买卖点核心)')
    print('=' * 100)
    cons_stk = report_consistent(df, stream_filter=['个股日卦', '个股月卦', '个股年卦'])
    buys = sorted([c for c in cons_stk if c['cons_type'] == '★买'], key=lambda x: -x['avg_alpha'])
    sells = sorted([c for c in cons_stk if c['cons_type'] == '✗卖'], key=lambda x: x['avg_alpha'])
    print(f'\n## 个股 ★ 真买点 (n={len(buys)})')
    for c in buys[:20]:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        print(f'  {c["stream"]:<8} {ct:<10} 一致 {c["n_consistent"]}/5  均α {c["avg_alpha"]:+.2f}%')
    print(f'\n## 个股 ✗ 真卖点 (n={len(sells)})')
    for c in sells[:20]:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        print(f'  {c["stream"]:<8} {ct:<10} 一致 {c["n_consistent"]}/5  均α {c["avg_alpha"]:+.2f}%')

    print('\n' + '=' * 100)
    print('# 跨持有期一致性 - 大盘流 (分治/过滤维度)')
    print('=' * 100)
    cons_mkt = report_consistent(df, stream_filter=['大盘日卦', '大盘月卦', '大盘年卦'])
    buys_m = sorted([c for c in cons_mkt if c['cons_type'] == '★买'], key=lambda x: -x['avg_alpha'])
    sells_m = sorted([c for c in cons_mkt if c['cons_type'] == '✗卖'], key=lambda x: x['avg_alpha'])
    print(f'\n## 大盘 ★ 利好信号 (n={len(buys_m)})')
    for c in buys_m[:15]:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        print(f'  {c["stream"]:<8} {ct:<10} 一致 {c["n_consistent"]}/5  均α {c["avg_alpha"]:+.2f}%')
    print(f'\n## 大盘 ✗ 利空信号 (n={len(sells_m)})')
    for c in sells_m[:15]:
        ct = f'{c["from"]}{GUA_NAMES[c["from"]]}→{c["to"]}{GUA_NAMES[c["to"]]}'
        print(f'  {c["stream"]:<8} {ct:<10} 一致 {c["n_consistent"]}/5  均α {c["avg_alpha"]:+.2f}%')

    print(f'\n=== 总耗时: {time.time()-t_all:.1f}s ===')


if __name__ == '__main__':
    main()
