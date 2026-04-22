# -*- coding: utf-8 -*-
"""
scanner.py — Stage1 预扫描: pandas 快速筛选候选股

扫描全部个股 CSV, 找出回测期间内散户线曾低于 POOL_THRESHOLD 的股票,
作为 Backtrader Stage2 的候选集 (减少从 ~5000 只到 ~200-500 只)。
"""
import os
import pickle
import hashlib
import time
import pandas as pd
import numpy as np
from .config import (STOCKS_DIR, POOL_THRESHOLD, BACKTEST_START, BACKTEST_END,
                     SEG_EVENTS_PATH, MIN_512_SAMPLES)


def get_cache_path(start_date, end_date, threshold):
    """生成扫描缓存文件路径"""
    key = f"{start_date}_{end_date}_{threshold}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache_dir = os.path.join(os.path.dirname(STOCKS_DIR), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'scanner_{h}.pkl')


def scan_candidates(start_date=None, end_date=None, threshold=None,
                    use_cache=True, verbose=True):
    """扫描候选股: 散户线曾低于阈值的股票

    Args:
        start_date: 起始日期, 默认 BACKTEST_START
        end_date: 结束日期, 默认 BACKTEST_END
        threshold: 散户线阈值, 默认 POOL_THRESHOLD
        use_cache: 是否使用缓存
        verbose: 是否打印进度

    Returns:
        dict: {code: DataFrame} 候选股数据 (index=DatetimeIndex)
    """
    start_date = start_date or BACKTEST_START
    end_date = end_date or BACKTEST_END
    threshold = threshold or POOL_THRESHOLD

    cache_path = get_cache_path(start_date, end_date, threshold)

    # 尝试加载缓存
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                candidates = pickle.load(f)
            if verbose:
                print(f"  [scanner] 使用缓存: {len(candidates)} 只候选股")
            return candidates
        except Exception as e:
            if verbose:
                print(f"  [scanner] 缓存加载失败: {e}")

    if verbose:
        print(f"  [scanner] 开始扫描, 阈值={threshold}, "
              f"区间={start_date}~{end_date}")

    t0 = time.time()
    candidates = {}
    total = 0
    skipped = 0

    files = [f for f in os.listdir(STOCKS_DIR) if f.endswith('.csv')]

    for i, fname in enumerate(files):
        code = fname.replace('.csv', '')
        fpath = os.path.join(STOCKS_DIR, fname)
        total += 1

        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig',
                             usecols=['date', 'open', 'close', 'high', 'low',
                                      'trend', 'retail', 'main_force',
                                      'year_gua', 'month_gua', 'day_gua'])
            df['date'] = pd.to_datetime(df['date'])

            # 日期过滤
            mask = (df['date'] >= pd.to_datetime(start_date)) & \
                   (df['date'] <= pd.to_datetime(end_date))
            df_period = df[mask]

            if len(df_period) == 0:
                skipped += 1
                continue

            # 检查散户线是否曾低于阈值
            retail = df_period['retail'].dropna()
            if len(retail) == 0 or retail.min() >= threshold:
                skipped += 1
                continue

            # 通过筛选! 保存完整数据 (回测需要)
            df = df.set_index('date').sort_index()
            # NaN 前向填充
            for col in ['trend', 'retail', 'main_force']:
                df[col] = df[col].ffill()
            # 卦列: 字符串→整数 (与 feeds.py 一致)
            for col in ['year_gua', 'month_gua', 'day_gua']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            candidates[code] = df

        except Exception as e:
            skipped += 1
            continue

        if verbose and (i + 1) % 500 == 0:
            print(f"    扫描进度: {i+1}/{len(files)}, "
                  f"已找到 {len(candidates)} 只候选股")

    elapsed = time.time() - t0

    if verbose:
        print(f"  [scanner] 扫描完成: "
              f"总计 {total} 只, 跳过 {skipped} 只, "
              f"候选 {len(candidates)} 只, 耗时 {elapsed:.1f}s")

    # 保存缓存
    if use_cache:
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(candidates, f)
            if verbose:
                print(f"  [scanner] 缓存已保存: {cache_path}")
        except Exception as e:
            if verbose:
                print(f"  [scanner] 缓存保存失败: {e}")

    return candidates


if __name__ == '__main__':
    candidates = scan_candidates(verbose=True)
    print(f"\n候选股列表 ({len(candidates)} 只):")
    for code in sorted(candidates.keys())[:20]:
        df = candidates[code]
        min_retail = df['retail'].min()
        print(f"  {code}: 数据 {len(df)} 行, 散户线最低 {min_retail:.0f}")
    if len(candidates) > 20:
        print(f"  ... 还有 {len(candidates)-20} 只")


# ============================================================
# 512分级系统
# ============================================================
def load_seg_events():
    """加载个股段首事件表 (用于512分级)"""
    if not os.path.exists(SEG_EVENTS_PATH):
        raise FileNotFoundError(f"段首事件文件不存在: {SEG_EVENTS_PATH}")
    df = pd.read_csv(SEG_EVENTS_PATH, encoding='utf-8-sig')
    df['event_date'] = df['event_date'].astype(str)
    df['avail_date'] = df['avail_date'].astype(str)
    for col in ['year_gua', 'month_gua', 'day_gua']:
        df[col] = df[col].astype(str).str.zfill(3)
    return df


def build_daily_512_snapshot(seg_events, trade_dates):
    """为每个交易日构建512快照 (无未来函数)

    与 backtest_capital.py 的 build_daily_512_snapshot 逻辑完全一致:
    滑动指针, 只使用 avail_date <= 当天的历史事件。

    Args:
        seg_events: 段首事件 DataFrame (含 avail_date, year/month/day_gua, excess_ret)
        trade_dates: 所有交易日列表 (str)

    Returns:
        dict: {date_str: {combo: mean_excess_ret}}
    """
    seg_events = seg_events.sort_values('avail_date').reset_index(drop=True)
    trade_dates = sorted(set(trade_dates))

    combos = (seg_events['year_gua'] + '_' +
              seg_events['month_gua'] + '_' +
              seg_events['day_gua']).values
    avail_dates = seg_events['avail_date'].values
    excess_rets = seg_events['excess_ret'].values

    snapshots = {}
    evt_ptr = 0
    n_events = len(seg_events)
    combo_rets = {}

    for dt in trade_dates:
        while evt_ptr < n_events and avail_dates[evt_ptr] <= dt:
            c = combos[evt_ptr]
            r = excess_rets[evt_ptr]
            if not np.isnan(r):
                combo_rets.setdefault(c, []).append(r)
            evt_ptr += 1

        snap = {}
        for c, rets in combo_rets.items():
            if len(rets) >= MIN_512_SAMPLES:
                snap[c] = np.mean(rets)
        snapshots[dt] = snap

    return snapshots
