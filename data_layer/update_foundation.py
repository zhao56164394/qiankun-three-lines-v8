# -*- coding: utf-8 -*-
"""
update_foundation.py — foundation 数据增量更新

每日收盘后运行，只追加新日期数据，不全量重算。
全量重算仍用各 prepare_*.py 脚本。

用法:
  python data_layer/update_foundation.py          # 增量更新所有
  python data_layer/update_foundation.py --check   # 只检查各文件当前日期
  python data_layer/update_foundation.py --rebuild  # 重建所有过期的派生文件
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.foundation_sources import normalize_date


DATA_DEPENDENCIES = {
    'daily_5d_scores.csv':        ['daily_cross_section.csv'],
    'daily_forward_returns.csv':  ['daily_cross_section.csv'],
    'daily_3yao.csv':             ['daily_5d_scores.csv'],
    'daily_bagua_sequence.csv':   ['daily_5d_scores.csv'],
    'market_bagua_daily.csv':     ['daily_cross_section.csv', 'daily_5d_scores.csv'],
    'stock_bagua_daily.csv':      ['daily_cross_section.csv', 'daily_5d_scores.csv'],
    # 日/月/年 三尺度卦 — 分治主底座, 依赖 zz1000_daily.csv (外部维护)
    'multi_scale_gua_daily.csv':  [],
}

REBUILD_MAP = {
    'daily_5d_scores.csv':        ('data_layer.prepare_daily_5d_scores',    'build_daily_5d_scores'),
    'daily_forward_returns.csv':  ('data_layer.prepare_daily_forward_returns', 'build_daily_forward_returns'),
    'daily_3yao.csv':             ('data_layer.prepare_daily_bagua',        'update_daily_bagua'),
    'daily_bagua_sequence.csv':   ('data_layer.prepare_daily_bagua',        'update_daily_bagua'),
    'market_bagua_daily.csv':     ('data_layer.prepare_market_bagua',       'build_market_bagua'),
    'stock_bagua_daily.csv':      ('data_layer.prepare_stock_bagua',        'build_stock_bagua'),
    'multi_scale_gua_daily.csv':  ('data_layer.prepare_multi_scale_gua',    'main'),
}


def _get_last_date(csv_path, date_col='date'):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=[date_col])
    if df.empty:
        return None
    return str(df[date_col].iloc[-1])


def _sync_parquet(csv_path):
    """CSV 更新后同步重写对应 Parquet（迁移期双轨支持）。"""
    if not os.path.exists(csv_path):
        return
    pq_path = os.path.splitext(csv_path)[0] + '.parquet'
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        if 'code' in df.columns:
            df['code'] = df['code'].astype('string').str.zfill(6)
        if 'gua_code' in df.columns:
            df['gua_code'] = df['gua_code'].astype('string').str.zfill(3)
        for col in ['d_gua', 'm_gua', 'y_gua']:
            if col in df.columns:
                df[col] = df[col].astype('string').str.zfill(3)
        df.to_parquet(pq_path, engine='pyarrow', compression='snappy', index=False)
    except Exception as e:
        print(f'  ⚠ Parquet 同步失败 ({os.path.basename(pq_path)}): {e}')


def _next_day(date_str):
    return (pd.to_datetime(date_str) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')


def _available_dates_from_metrics():
    from data_layer.foundation_sources import list_csv_files_under, load_daily_metrics
    from data_layer.foundation_config import PATHS
    files = list_csv_files_under(PATHS['stock_daily_metrics_root'])
    dates = set()
    for path in files:
        try:
            m = load_daily_metrics(file_path=path)
            if not m.empty:
                dates.add(normalize_date(m['date'].max()))
        except Exception:
            continue
    return sorted(dates)


# ============================================================
# 1. 主板样本池
# ============================================================
def update_universe():
    path = foundation_file('main_board_universe.csv')
    last = _get_last_date(path)
    if last is None:
        print('  主板样本池不存在，需先运行: python data_layer/prepare_main_board_universe.py')
        return None

    from data_layer.prepare_main_board_universe import build_main_board_universe
    start = _next_day(last)
    print(f'  主板样本池增量: {last} → {start}起...')

    try:
        new_df = build_main_board_universe(start_date=start)
    except FileNotFoundError:
        print('  无新数据')
        return None

    if new_df.empty or str(new_df['date'].max()) <= last:
        print('  无新数据')
        return None

    new_df = new_df[new_df['date'].astype(str) > last].copy()
    if new_df.empty:
        print('  无新数据')
        return None

    new_df.to_csv(path, mode='a', index=False, header=False, encoding='utf-8-sig')
    new_last = _get_last_date(path)
    print(f'  OK 主板样本池: → {new_last} (+{len(new_df)}行)')
    _sync_parquet(path)
    return new_df


# ============================================================
# 2. 每日截面
# ============================================================
def update_cross_section():
    path = foundation_file('daily_cross_section.csv')
    last = _get_last_date(path)
    if last is None:
        print('  截面数据不存在，需先运行: python data_layer/prepare_daily_cross_section.py')
        return None

    start = _next_day(last)
    print(f'  截面数据增量: {last} → {start}起...')

    from data_layer.prepare_daily_cross_section import build_daily_cross_section
    tmp_name = '_incremental_cross_section_tmp.csv'
    try:
        new_df = build_daily_cross_section(start_date=start, output_name=tmp_name)
    except (FileNotFoundError, ValueError):
        print('  无新数据')
        return None

    if new_df.empty or str(new_df['date'].max()) <= last:
        print('  无新数据')
        tmp_path = foundation_file(tmp_name)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None

    new_df = new_df[new_df['date'].astype(str) > last].copy()
    tmp_path = foundation_file(tmp_name)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if new_df.empty:
        print('  无新数据')
        return None

    new_df.to_csv(path, mode='a', index=False, header=False, encoding='utf-8-sig')
    new_last = _get_last_date(path)
    print(f'  OK 截面数据: → {new_last} (+{len(new_df)}行)')
    _sync_parquet(path)
    return new_df


# ============================================================
# 3. 5维分数
# ============================================================
def update_5d_scores(new_cross_df=None):
    path = foundation_file('daily_5d_scores.csv')
    last = _get_last_date(path)
    if last is None:
        print('  5维分数不存在，需先运行: python data_layer/prepare_daily_5d_scores.py')
        return None

    cross_path = foundation_file('daily_cross_section.csv')
    cross_last = _get_last_date(cross_path)
    if cross_last is None or cross_last <= last:
        print(f'  5维分数已是最新 ({last})')
        return None

    print(f'  5维分数增量: {last} → {cross_last}...')

    if new_cross_df is not None and not new_cross_df.empty:
        cross_new = new_cross_df.copy()
    else:
        cross_new = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
        cross_new = cross_new[cross_new['date'].astype(str) > last].copy()

    if cross_new.empty:
        print('  无新数据')
        return None

    from data_layer.prepare_daily_5d_scores import _build_scores_for_day
    parts = []
    for _, day_df in cross_new.groupby('date', sort=True):
        parts.append(_build_scores_for_day(day_df))
    new_scores = pd.concat(parts, ignore_index=True)

    new_scores.to_csv(path, mode='a', index=False, header=False, encoding='utf-8-sig')
    new_last = _get_last_date(path)
    print(f'  OK 5维分数: → {new_last} (+{len(new_scores)}行)')
    _sync_parquet(path)
    return new_scores


# ============================================================
# 4. 前瞻收益
# ============================================================
def update_forward_returns(new_cross_df=None):
    path = foundation_file('daily_forward_returns.csv')
    cross_path = foundation_file('daily_cross_section.csv')
    last = _get_last_date(path)
    if last is None:
        print('  前瞻收益不存在，需先运行: python data_layer/prepare_daily_forward_returns.py')
        return None

    cross_last = _get_last_date(cross_path)
    if cross_last is None or cross_last <= last:
        print(f'  前瞻收益已是最新 ({last})')
        return None

    print(f'  前瞻收益增量: {last} → {cross_last}...')
    from data_layer.prepare_daily_forward_returns import HORIZONS

    cross_close = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str},
                              usecols=['date', 'code', 'close'])
    cross_close['date'] = cross_close['date'].astype(str)
    cross_close['close'] = pd.to_numeric(cross_close['close'], errors='coerce')
    cross_close = cross_close.sort_values(['code', 'date']).reset_index(drop=True)

    existing = pd.read_csv(path, encoding='utf-8-sig', dtype={'code': str})
    existing['date'] = existing['date'].astype(str)

    backfill_start = (pd.to_datetime(last) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
    need_update = cross_close[cross_close['date'] >= backfill_start].copy()

    out = need_update[['date', 'code']].copy()
    for horizon in HORIZONS:
        future_close = need_update.groupby('code')['close'].shift(-horizon)
        future_date = need_update.groupby('code')['date'].shift(-horizon)
        ret = (future_close / need_update['close'] - 1) * 100
        out[f'ret_fwd_{horizon}d'] = ret.round(4)
        out[f'avail_date_{horizon}d'] = future_date.astype(str)
        out.loc[future_date.isna(), f'avail_date_{horizon}d'] = pd.NA

    old_keep = existing[existing['date'] < backfill_start].copy()
    combined = pd.concat([old_keep, out], ignore_index=True)
    combined = combined.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code']).reset_index(drop=True)
    combined.to_csv(path, index=False, encoding='utf-8-sig')

    new_last = _get_last_date(path)
    print(f'  OK 前瞻收益: → {new_last} (回填+新增)')
    _sync_parquet(path)
    return combined


# ============================================================
# 5. 市场卦 (输出仅2MB，重算指标很快)
# ============================================================
def update_market_bagua(new_cross_df=None, new_scores_df=None):
    path = foundation_file('market_bagua_daily.csv')
    last = _get_last_date(path)
    if last is None:
        print('  市场卦不存在，需先运行: python data_layer/prepare_market_bagua.py')
        return None

    cross_path = foundation_file('daily_cross_section.csv')
    cross_last = _get_last_date(cross_path)
    if cross_last is None or cross_last <= last:
        print(f'  市场卦已是最新 ({last})')
        return None

    print(f'  市场卦增量: {last} → {cross_last}...')

    from data_layer.prepare_market_bagua import (
        _build_index_anchor, _calc_market_features, _mark_segments,
        OUTPUT_COLUMNS, SCORE_COLUMNS, INDEX_CLOSE_COLUMNS,
    )

    existing = pd.read_csv(path, encoding='utf-8-sig')
    existing['date'] = existing['date'].astype(str)

    raw_agg_cols = [
        'date', 'stock_count',
        'market_open_proxy', 'market_high_proxy', 'market_low_proxy', 'market_close_proxy',
        'up_ratio', 'zt_ratio', 'dt_ratio', 'turnover_median',
        'zt_count', 'dt_count', 'zb_count', 'limit_heat', 'limit_quality', 'ladder_heat',
        'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
        'new_high_20_ratio', 'new_low_20_ratio',
    ] + SCORE_COLUMNS
    keep_cols = [c for c in raw_agg_cols if c in existing.columns]
    old_agg = existing[keep_cols].copy()

    if new_cross_df is not None and not new_cross_df.empty:
        cross_new = new_cross_df.copy()
    else:
        cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
        cross_new = cross[cross['date'].astype(str) > last].copy()

    if cross_new.empty:
        print('  无新截面数据')
        return None

    index_anchor = _build_index_anchor(cross_new) if all(c in cross_new.columns for c in INDEX_CLOSE_COLUMNS) else pd.DataFrame()

    for col in ['zt_count', 'dt_count', 'zb_count', 'lb_count']:
        if col not in cross_new.columns:
            cross_new[col] = pd.NA

    base_cols = [
        'date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate', 'is_zt', 'is_dt',
        'zt_count', 'dt_count', 'zb_count', 'lb_count',
        'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
        'new_high_20_ratio', 'new_low_20_ratio',
    ]
    base_cols = [c for c in base_cols if c in cross_new.columns]
    base = cross_new[base_cols].copy()
    for col in ['open', 'high', 'low', 'close', 'turnover_rate', 'zt_count', 'dt_count', 'zb_count', 'lb_count']:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors='coerce')
    if 'is_zt' in base.columns:
        base['is_zt'] = pd.to_numeric(base['is_zt'], errors='coerce').fillna(0)
    if 'is_dt' in base.columns:
        base['is_dt'] = pd.to_numeric(base['is_dt'], errors='coerce').fillna(0)
    if 'lb_count' in base.columns:
        base['lb_count'] = base['lb_count'].fillna(0)

    prev_close = base.groupby('code')['close'].shift(1)
    no_prev = prev_close.isna() & base['open'].notna()
    prev_close.loc[no_prev] = base.loc[no_prev, 'open']
    base['is_up'] = (base['close'] > prev_close).astype(float)
    base.loc[prev_close.isna(), 'is_up'] = pd.NA

    new_market = base.groupby('date', sort=True).agg({
        'code': 'count',
        'open': 'mean', 'high': 'mean', 'low': 'mean', 'close': 'mean',
        'is_up': 'mean', 'is_zt': 'mean', 'is_dt': 'mean',
        'turnover_rate': 'median',
        'zt_count': 'first', 'dt_count': 'first', 'zb_count': 'first',
        'lb_count': lambda s: s.clip(upper=3).mean() / 3.0,
        'above_ma5_ratio': 'mean', 'above_ma10_ratio': 'mean',
        'above_ma20_ratio': 'mean', 'above_ma60_ratio': 'mean',
        'new_high_20_ratio': 'mean', 'new_low_20_ratio': 'mean',
    }).reset_index().rename(columns={
        'code': 'stock_count',
        'open': 'market_open_proxy', 'high': 'market_high_proxy',
        'low': 'market_low_proxy', 'close': 'market_close_proxy',
        'is_up': 'up_ratio', 'is_zt': 'zt_ratio', 'is_dt': 'dt_ratio',
        'turnover_rate': 'turnover_median', 'lb_count': 'ladder_heat',
    })

    if not index_anchor.empty:
        new_market = new_market.merge(index_anchor, on='date', how='left')
        for col in ['market_close_proxy', 'market_open_proxy', 'market_high_proxy', 'market_low_proxy']:
            if 'market_index_anchor' in new_market.columns:
                new_market[col] = new_market['market_index_anchor'].combine_first(new_market[col])

    if new_scores_df is not None and not new_scores_df.empty:
        scores_new = new_scores_df[new_scores_df['date'].astype(str) > last].copy()
    else:
        score_path = foundation_file('daily_5d_scores.csv')
        if os.path.exists(score_path):
            scores = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str})
            scores_new = scores[scores['date'].astype(str) > last].copy()
        else:
            scores_new = pd.DataFrame()

    if not scores_new.empty:
        for col in SCORE_COLUMNS:
            if col in scores_new.columns:
                scores_new[col] = pd.to_numeric(scores_new[col], errors='coerce')
        score_market = scores_new.groupby('date', sort=True)[SCORE_COLUMNS].mean().reset_index()
        new_market = new_market.merge(score_market, on='date', how='left')
    else:
        for col in SCORE_COLUMNS:
            new_market[col] = pd.NA

    for col in keep_cols:
        if col not in new_market.columns:
            new_market[col] = pd.NA
    new_agg = new_market[keep_cols].copy()

    combined_agg = pd.concat([old_agg, new_agg], ignore_index=True)
    combined_agg = combined_agg.drop_duplicates('date', keep='last').sort_values('date').reset_index(drop=True)

    market = _calc_market_features(combined_agg)
    market = _mark_segments(market)
    for col in OUTPUT_COLUMNS:
        if col not in market.columns:
            market[col] = pd.NA
    market = market[OUTPUT_COLUMNS].sort_values('date').reset_index(drop=True)

    market.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  OK 市场卦: → {market["date"].iloc[-1]} ({len(market)}行)')
    _sync_parquet(path)
    return market


# ============================================================
# 6. 个股卦 (1.3GB，用现有输出做warmup)
# ============================================================
def update_stock_bagua(new_cross_df=None, new_scores_df=None):
    path = foundation_file('stock_bagua_daily.csv')
    last = _get_last_date(path)
    if last is None:
        print('  个股卦不存在，需先运行: python data_layer/prepare_stock_bagua.py')
        return None

    cross_path = foundation_file('daily_cross_section.csv')
    cross_last = _get_last_date(cross_path)
    if cross_last is None or cross_last <= last:
        print(f'  个股卦已是最新 ({last})')
        return None

    print(f'  个股卦增量: {last} → {cross_last}...')

    from data_layer.prepare_stock_bagua import (
        _build_stock_features, _mark_segments, OUTPUT_COLUMNS, NUMERIC_COLS,
    )

    print('    加载现有个股卦...')
    existing = pd.read_csv(path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    existing['date'] = existing['date'].astype(str)

    warmup_start = (pd.to_datetime(last) - pd.Timedelta(days=300)).strftime('%Y-%m-%d')
    warmup = existing[existing['date'] >= warmup_start].copy()

    if new_cross_df is not None and not new_cross_df.empty:
        cross_new = new_cross_df.copy()
    else:
        cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
        cross_new = cross[cross['date'].astype(str) > last].copy()

    if cross_new.empty:
        print('  无新截面数据')
        return None

    if new_scores_df is not None and not new_scores_df.empty:
        scores_new = new_scores_df[new_scores_df['date'].astype(str) > last].copy()
    else:
        score_path = foundation_file('daily_5d_scores.csv')
        scores = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str})
        scores_new = scores[scores['date'].astype(str) > last].copy()

    keep_cross = ['date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate', 'lb_count', 'is_zt']
    keep_scores = ['date', 'code', 'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi']
    for col in keep_cross:
        if col not in cross_new.columns:
            cross_new[col] = pd.NA
    for col in keep_scores:
        if col not in scores_new.columns:
            scores_new[col] = pd.NA

    new_raw = cross_new[keep_cross].merge(scores_new[keep_scores], on=['date', 'code'], how='left')
    new_raw['code'] = new_raw['code'].astype(str).str.zfill(6)
    new_raw['date'] = new_raw['date'].astype(str)
    for col in NUMERIC_COLS:
        if col in new_raw.columns:
            new_raw[col] = pd.to_numeric(new_raw[col], errors='coerce')
    if 'lb_count' in new_raw.columns:
        new_raw['lb_count'] = new_raw['lb_count'].fillna(0)
    if 'is_zt' in new_raw.columns:
        new_raw['is_zt'] = new_raw['is_zt'].fillna(0)

    warmup_input_cols = ['date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate',
                         'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi']
    warmup_for_merge = warmup[warmup_input_cols].copy()
    warmup_for_merge['lb_count'] = 0.0
    warmup_for_merge['is_zt'] = 0.0
    for col in NUMERIC_COLS:
        if col in warmup_for_merge.columns:
            warmup_for_merge[col] = pd.to_numeric(warmup_for_merge[col], errors='coerce')

    combined_input = pd.concat([warmup_for_merge, new_raw], ignore_index=True)
    combined_input = combined_input.drop_duplicates(['date', 'code'], keep='last')
    combined_input = combined_input.sort_values(['code', 'date']).reset_index(drop=True)

    new_codes = set(new_raw['code'].unique())
    print(f'    处理 {len(new_codes)} 只股票...')

    new_parts = []
    for code, group in combined_input[combined_input['code'].isin(new_codes)].groupby('code', sort=True):
        out = _build_stock_features(group)
        if out is not None and not out.empty:
            out_new = out[out['date'] > last].copy()
            if not out_new.empty:
                new_parts.append(out_new)

    if not new_parts:
        print('  无有效新数据')
        return None

    new_stock = pd.concat(new_parts, ignore_index=True)

    full = pd.concat([existing, new_stock], ignore_index=True)
    full = full.drop_duplicates(['date', 'code'], keep='last')
    full = _mark_segments(full)

    for col in OUTPUT_COLUMNS:
        if col not in full.columns:
            full[col] = pd.NA
    full = full[OUTPUT_COLUMNS].sort_values(['date', 'code']).reset_index(drop=True)

    full.to_csv(path, index=False, encoding='utf-8-sig')
    new_last = str(full['date'].max())
    print(f'  OK 个股卦: → {new_last} (+{len(new_stock)}行, 共{len(full)}行)')
    _sync_parquet(path)
    return new_stock


# ============================================================
# 新鲜度校验
# ============================================================
def verify_freshness():
    """比对派生文件与上游文件的修改时间，返回过期列表。"""
    stale = []
    for downstream, upstreams in DATA_DEPENDENCIES.items():
        down_path = foundation_file(downstream)
        if not os.path.exists(down_path):
            continue
        down_mtime = os.path.getmtime(down_path)
        for up_name in upstreams:
            up_path = foundation_file(up_name)
            if not os.path.exists(up_path):
                continue
            if os.path.getmtime(up_path) > down_mtime:
                stale.append((downstream, up_name))
                break
    return stale


def print_freshness(stale):
    if stale:
        print('\n  [!] 新鲜度警告 — 以下派生文件比上游旧，数据可能不一致:')
        for down, up in stale:
            print(f'      {down}  <-  {up} (上游更新)')
        print('  --> 运行 python data_layer/update_foundation.py --rebuild 修复')
    else:
        print('  [ok] 派生文件新鲜度检查通过')


def rebuild_stale():
    """检测过期的派生文件并按依赖顺序重建。"""
    stale = verify_freshness()
    if not stale:
        print('  所有派生文件均为最新，无需重建。')
        return

    stale_files = {item[0] for item in stale}
    rebuild_order = [
        'daily_5d_scores.csv',
        'daily_forward_returns.csv',
        'daily_3yao.csv',
        'daily_bagua_sequence.csv',
        'market_bagua_daily.csv',
        'stock_bagua_daily.csv',
        'multi_scale_gua_daily.csv',
    ]
    done_modules = set()

    print('=' * 60)
    print('  重建过期派生文件')
    print('=' * 60)
    t0 = time.time()

    for fname in rebuild_order:
        if fname not in stale_files:
            continue
        mod_name, func_name = REBUILD_MAP[fname]
        if mod_name in done_modules:
            continue
        done_modules.add(mod_name)
        import importlib
        print(f'\n  重建 {fname} ...')
        try:
            mod = importlib.import_module(mod_name)
            getattr(mod, func_name)()
        except Exception as e:
            print(f'    重建失败: {e}')

    elapsed = time.time() - t0
    print(f'\n  重建完成 (耗时 {elapsed:.0f}秒)')

    remaining = verify_freshness()
    print_freshness(remaining)


# ============================================================
# 状态检查
# ============================================================
def check_status():
    files = [
        ('主板样本池', 'main_board_universe.csv'),
        ('每日截面', 'daily_cross_section.csv'),
        ('5维分数', 'daily_5d_scores.csv'),
        ('前瞻收益', 'daily_forward_returns.csv'),
        ('天卦 (市场)', 'market_bagua_daily.csv'),
        ('地卦 (个股)', 'stock_bagua_daily.csv'),
        ('日/月/年卦', 'multi_scale_gua_daily.csv'),
    ]
    print('=' * 60)
    print('  Foundation 数据状态')
    print('=' * 60)
    for label, fname in files:
        path = foundation_file(fname)
        last = _get_last_date(path)
        if last:
            sz = os.path.getsize(path) / 1024 / 1024
            print(f'  {label:10s}  最新={last}  {sz:>8.1f}MB')
        else:
            print(f'  {label:10s}  不存在')
    print('=' * 60)
    print_freshness(verify_freshness())


def _patch_cross_section_index():
    """用 zz1000_daily.csv 回补 cross_section 中 csi1000_close 为 NaN 的日期"""
    cross_path = foundation_file('daily_cross_section.csv')
    zz_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data_layer', 'data', 'zz1000_daily.csv')
    if not os.path.exists(zz_path):
        return

    zz = pd.read_csv(zz_path, encoding='utf-8-sig', usecols=['date', 'close'])
    zz['date'] = zz['date'].astype(str)
    zz_map = dict(zip(zz['date'], zz['close']))

    cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    cross['date'] = cross['date'].astype(str)
    cross['csi1000_close'] = pd.to_numeric(cross['csi1000_close'], errors='coerce')

    nan_dates = cross.loc[cross['csi1000_close'].isna(), 'date'].unique()
    patched = []
    for d in nan_dates:
        if d in zz_map:
            cross.loc[cross['date'] == d, 'csi1000_close'] = zz_map[d]
            patched.append(d)

    if patched:
        cross.to_csv(cross_path, index=False, encoding='utf-8-sig')
        print(f'  csi1000_close 已补丁: {patched}')
        _sync_parquet(cross_path)
    else:
        print(f'  csi1000_close 无需补丁')


# ============================================================
# 入口
# ============================================================
def _rebuild_baseline_snapshot_safe():
    """重建 dashboard 裸跑快照；失败不影响数据更新本身。"""
    try:
        import rebuild_baseline_snapshot as rbs
        print('\n  重建 dashboard 裸跑快照...')
        rbs.main(verbose=True)
    except Exception as e:
        print(f'  !! 裸跑快照重建失败: {e}')


def update_multi_scale_gua():
    """日/月/年 三尺度卦 (分治主底座)
    依赖 zz1000_daily.csv (外部维护), 每次全量重算 (~5000 天, 秒级).
    """
    path = foundation_file('multi_scale_gua_daily.csv')
    last = _get_last_date(path) if os.path.exists(path) else None
    from data_layer.prepare_multi_scale_gua import main as build_ms
    build_ms()
    new_last = _get_last_date(path)
    print(f'  日/月/年卦: {last} -> {new_last}')
    _sync_parquet(path)


def update_all():
    print('=' * 60)
    print(f'  Foundation 增量更新')
    print('=' * 60)
    t0 = time.time()

    print('\n[1/7] 主板样本池')
    update_universe()

    print('\n[2/7] 每日截面')
    new_cross = update_cross_section()
    _patch_cross_section_index()

    print('\n[3/7] 5维分数')
    new_scores = update_5d_scores(new_cross)

    print('\n[4/7] 前瞻收益')
    update_forward_returns(new_cross)

    print('\n[5/7] 天卦')
    update_market_bagua(new_cross, new_scores)

    print('\n[6/7] 地卦')
    update_stock_bagua(new_cross, new_scores)

    print('\n[7/7] 日/月/年卦 (分治底座)')
    update_multi_scale_gua()

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'  全部完成 (耗时 {elapsed:.0f}秒)')
    print(f'{"=" * 60}')
    check_status()

    stale = verify_freshness()
    if stale:
        print('\n  检测到过期派生文件，自动重建...')
        rebuild_stale()

    _rebuild_baseline_snapshot_safe()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Foundation 数据增量更新')
    parser.add_argument('--check', action='store_true', help='只检查状态，不更新')
    parser.add_argument('--rebuild', action='store_true', help='重建所有过期的派生文件')
    args = parser.parse_args()

    if args.check:
        check_status()
    elif args.rebuild:
        rebuild_stale()
    else:
        update_all()
