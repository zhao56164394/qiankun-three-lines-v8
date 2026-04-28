# -*- coding: utf-8 -*-
"""6 卦流变卦扫描 — 极致优化版

优化点:
  1. 整张表 → contiguous numpy 数组 (避免 groupby 重复)
  2. code 边界数组 (start_idx, end_idx) 替代 groupby
  3. 大盘流: 大盘事件日期 → searchsorted 找全局索引一次性匹配
  4. 全部 bootstrap 向量化 (n_boot × N 矩阵抽样)
  5. 数据共享: 6 流复用同一份股票 close array

预期: <5 分钟跑完 6 流
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
    if len(event_rets) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    e = np.asarray(event_rets, dtype=np.float32)
    b = np.asarray(base_rets, dtype=np.float32)
    n_event = len(e); n_base = len(b)
    idx_e = rng.randint(0, n_event, size=(n_boot, n_event))
    idx_b = rng.randint(0, n_base, size=(n_boot, n_base))
    boots = e[idx_e].mean(axis=1) - b[idx_b].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_all():
    """一次性加载 + 准备所有 numpy 数组"""
    print('=== 加载数据 ===')
    t0 = time.time()
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        market[c] = market[c].astype(str).str.zfill(3)
    market = market.sort_values('date').reset_index(drop=True)
    print(f'  market: {len(market)} 天, {time.time()-t0:.1f}s')

    t0 = time.time()
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                         columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                         columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  merged: {len(df):,} 行, {time.time()-t0:.1f}s')

    t0 = time.time()
    # 整表 numpy 数组
    code_arr = df['code'].values
    date_arr = df['date'].values
    close_arr = df['close'].values.astype(np.float32)
    d_arr = df['d_gua'].values
    m_arr = df['m_gua'].values
    y_arr = df['y_gua'].values

    # 找 code 边界 (cumcount + boundaries)
    # 因为按 (code, date) 排序, code 段连续
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    print(f'  numpy arrays + 边界: {len(code_starts)} 只票, {time.time()-t0:.1f}s')

    # 全局 (date, code) → row_idx hash, 给大盘流用
    t0 = time.time()
    date_code_to_idx = {}
    for i in range(len(date_arr)):
        date_code_to_idx[(date_arr[i], code_arr[i])] = i
    print(f'  (date,code)→row hash: {len(date_code_to_idx)} entries, {time.time()-t0:.1f}s')

    return {
        'market': market, 'df': df,
        'code_arr': code_arr, 'date_arr': date_arr, 'close_arr': close_arr,
        'd_arr': d_arr, 'm_arr': m_arr, 'y_arr': y_arr,
        'code_starts': code_starts, 'code_ends': code_ends,
        'date_code_to_idx': date_code_to_idx,
    }


def build_baseline(data, hold, n=10000, seed=42):
    """从 close_arr 上随机抽 (idx, idx+hold) 对; 不能跨 code 边界"""
    rng = np.random.RandomState(seed)
    close_arr = data['close_arr']
    code_starts = data['code_starts']; code_ends = data['code_ends']
    n_codes = len(code_starts)
    rets = []
    while len(rets) < n:
        c = rng.randint(0, n_codes)
        s = code_starts[c]; e = code_ends[c]
        if e - s <= hold + 1: continue
        i = rng.randint(s, e - hold - 1)
        c0 = close_arr[i]; c1 = close_arr[i + hold]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return np.array(rets, dtype=np.float32)


def find_stock_self_changes(data, gua_arr_name):
    """找出 个股自变卦 X→Y 事件位置 (全局 row_idx). 跨 code 边界忽略."""
    g = data[gua_arr_name]  # 'd_arr' / 'm_arr' / 'y_arr'
    code_arr = data['code_arr']
    n = len(g)
    # 变卦: 当前位置跟前一位置不同, 且 code 一致
    same_code = np.r_[False, code_arr[1:] == code_arr[:-1]]
    diff_gua = np.r_[False, g[1:] != g[:-1]]
    valid_g_prev = np.r_[False, np.isin(g[:-1], GUAS)]
    valid_g_now = np.isin(g, GUAS)
    is_event = same_code & diff_gua & valid_g_prev & valid_g_now

    event_idx = np.where(is_event)[0]
    # 对每个 event_idx 取 (from, to)
    from_arr = g[event_idx - 1]
    to_arr = g[event_idx]
    return event_idx, from_arr, to_arr


def fwd_returns_global(data, event_idx, hold):
    """计算 event_idx 中每个位置 N 天后的收益 (受 code 边界限制).
    向量化: 用 code_starts/ends 找每个 event 所属 code 的 e_end, 过滤越界
    """
    close_arr = data['close_arr']
    code_starts = data['code_starts']
    code_ends = data['code_ends']
    # 给每个 event 找其 code 的 end
    # 先用 searchsorted 找 event_idx 落在哪个 code 段
    code_seg = np.searchsorted(code_starts, event_idx, side='right') - 1
    end_of_code = code_ends[code_seg]
    valid_mask = (event_idx + hold) < end_of_code
    valid_idx = event_idx[valid_mask]
    if len(valid_idx) == 0:
        return np.array([], dtype=np.float32)
    c0 = close_arr[valid_idx]
    c1 = close_arr[valid_idx + hold]
    rets_mask = c0 > 0
    return ((c1[rets_mask] / c0[rets_mask] - 1.0) * 100.0).astype(np.float32)


def scan_stock_self_streams_fast(data, base_by_h):
    """优化版: 一次找全部事件, 按 (from, to) 分组, vectorized fwd return"""
    print('\n=== 扫描 个股自变卦 (3 流) - FAST ===')
    rows = []
    for stream_name, arr_name in [('个股日卦', 'd_arr'), ('个股月卦', 'm_arr'), ('个股年卦', 'y_arr')]:
        print(f'\n  -- {stream_name} --')
        t0 = time.time()
        event_idx, from_arr, to_arr = find_stock_self_changes(data, arr_name)
        print(f'    事件: {len(event_idx):,}, {time.time()-t0:.1f}s')

        # 按 (from, to) 分组
        t0 = time.time()
        # 用 string concat 做组 key
        ft = np.char.add(from_arr.astype('U3'), to_arr.astype('U3'))
        unique_ft = np.unique(ft)
        for ft_key in unique_ft:
            mask = ft == ft_key
            idx_this = event_idx[mask]
            if len(idx_this) < 30:
                continue
            f = ft_key[:3]; t = ft_key[3:]
            for h in HOLD_DAYS:
                rets = fwd_returns_global(data, idx_this, h)
                if len(rets) < 30: continue
                if len(rets) > 50000:
                    rng = np.random.RandomState(42)
                    rets = rng.choice(rets, 50000, replace=False)
                base = base_by_h[h]
                alpha = float(rets.mean() - base.mean())
                ci_lo, ci_hi = boot_alpha_ci_vec(rets, base)
                if ci_lo is None: continue
                win = float((rets > 0).mean() * 100)
                verdict = '★买点' if ci_lo > 0 else ('✗卖点' if ci_hi < 0 else '○灰区')
                rows.append({
                    'stream': stream_name, 'from': f, 'to': t, 'hold': h,
                    'n_events': int(mask.sum()), 'n_rets': len(rets),
                    'event_mean': float(rets.mean()), 'alpha': alpha,
                    'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    'win_rate': win, 'verdict': verdict,
                })
        print(f'    扫完: {time.time()-t0:.1f}s')
    return rows


def find_market_change_events(market, gua_col):
    """大盘 X→Y 事件日期"""
    m = market.sort_values('date').reset_index(drop=True)
    m['prev'] = m[gua_col].shift(1)
    events = m[(m['prev'].notna()) & (m['prev'] != m[gua_col])]
    return events[['date', 'prev', gua_col]].rename(columns={'prev': 'from', gua_col: 'to'}).reset_index(drop=True)


def scan_market_streams_fast(data, base_by_h):
    """大盘流 - FAST: 用全局 (date,code) 哈希一次定位"""
    print('\n=== 扫描 大盘流 (3 流) - FAST ===')
    rows = []
    market = data['market']
    date_arr = data['date_arr']
    close_arr = data['close_arr']
    code_starts = data['code_starts']
    code_ends = data['code_ends']

    # 预备: 全局 date 排序索引 (date_arr 在按 (code, date) 排序的表里, 同一 date 在不同 code 处)
    # 这里改用 sorted 个股索引 group by date
    # 更高效: 按 date 分组每个 (date) 的 row_idx 列表
    print('  building date → row indices...')
    t0 = time.time()
    date_to_rows = {}
    for i in range(len(date_arr)):
        d = date_arr[i]
        if d not in date_to_rows:
            date_to_rows[d] = []
        date_to_rows[d].append(i)
    # 转 numpy
    for d in date_to_rows:
        date_to_rows[d] = np.array(date_to_rows[d], dtype=np.int64)
    print(f'  {len(date_to_rows)} 天, {time.time()-t0:.1f}s')

    for stream_name, gua_col in [('大盘日卦', 'd_gua'), ('大盘月卦', 'm_gua'), ('大盘年卦', 'y_gua')]:
        print(f'\n  -- {stream_name} --')
        t0 = time.time()
        evts = find_market_change_events(market, gua_col)
        # 按 (from, to) 分组
        for (f, t), g in evts.groupby(['from', 'to']):
            if len(g) < 5: continue
            ev_dates = g['date'].values
            # 取每个事件日期上的所有 row_idx
            all_idx = []
            for d in ev_dates:
                if d in date_to_rows:
                    all_idx.append(date_to_rows[d])
            if not all_idx: continue
            all_idx = np.concatenate(all_idx)
            for h in HOLD_DAYS:
                # 用 fwd_returns_global 算 (受 code 边界限制)
                rets = fwd_returns_global(data, all_idx, h)
                if len(rets) < 30: continue
                if len(rets) > 50000:
                    rng = np.random.RandomState(42)
                    rets = rng.choice(rets, 50000, replace=False)
                base = base_by_h[h]
                alpha = float(rets.mean() - base.mean())
                ci_lo, ci_hi = boot_alpha_ci_vec(rets, base)
                if ci_lo is None: continue
                win = float((rets > 0).mean() * 100)
                verdict = '★买点' if ci_lo > 0 else ('✗卖点' if ci_hi < 0 else '○灰区')
                rows.append({
                    'stream': stream_name, 'from': f, 'to': t, 'hold': h,
                    'n_events': len(ev_dates), 'n_rets': len(rets),
                    'event_mean': float(rets.mean()), 'alpha': alpha,
                    'ci_lo': ci_lo, 'ci_hi': ci_hi,
                    'win_rate': win, 'verdict': verdict,
                })
        print(f'    扫完: {time.time()-t0:.1f}s')
    return rows


def report_consistent(df, stream_filter=None):
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
    data = load_all()
    print(f'\n=== 计算各持有期基线 ===')
    base_by_h = {}
    for h in HOLD_DAYS:
        base_by_h[h] = build_baseline(data, h)
        print(f'  hold={h}d: 基线均值 {base_by_h[h].mean():+.2f}%')

    rows = []
    rows.extend(scan_stock_self_streams_fast(data, base_by_h))
    rows.extend(scan_market_streams_fast(data, base_by_h))

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, 'data_layer/data/ablation/test18_yao_change_fast')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'full_scan.csv'), index=False, encoding='utf-8-sig',
              float_format='%.3f')
    print(f'\n落地: {out_dir}/full_scan.csv ({len(df)} 行)')
    print(f'\n=== 总耗时: {time.time()-t_all:.1f}s ===')


if __name__ == '__main__':
    main()
