# -*- coding: utf-8 -*-
"""
prepare_daily_cross_section.py

构建新底座日度横截面宽表。
支持单日快照与全历史输出。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import DEFAULT_COLUMNS, STOCKS_DATA_DIR, PATHS, foundation_file
from data_layer.foundation_sources import (
    load_daily_metrics,
    load_daily_metrics_history,
    load_moneyflow,
    load_moneyflow_history,
    load_chip_distribution,
    load_limit_ladder,
    load_limit_summary,
    load_industry_components,
    load_concept_components,
    load_index_closes_from_stock_metrics,
    load_index_closes_history,
    normalize_date,
    normalize_code,
    list_csv_files_under,
)




def _build_market_structure_features(metrics_hist: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    if metrics_hist.empty or universe.empty:
        return pd.DataFrame(columns=[
            'date',
            'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
            'new_high_20_ratio', 'new_low_20_ratio',
        ])

    hist = metrics_hist.copy()
    hist['date'] = hist['date'].astype(str)
    hist['code'] = hist['code'].astype(str).str.zfill(6)
    hist = hist.sort_values(['code', 'date']).reset_index(drop=True)

    for col in ['close', 'high', 'low']:
        hist[col] = pd.to_numeric(hist[col], errors='coerce')

    for window in [5, 10, 20, 60]:
        ma = hist.groupby('code')['close'].transform(lambda s: s.rolling(window, min_periods=window).mean())
        hist[f'above_ma{window}'] = ((hist['close'] > ma) & ma.notna()).astype(float)

    high20 = hist.groupby('code')['high'].transform(lambda s: s.rolling(20, min_periods=20).max())
    low20 = hist.groupby('code')['low'].transform(lambda s: s.rolling(20, min_periods=20).min())
    hist['new_high_20'] = ((hist['high'] >= high20) & high20.notna()).astype(float)
    hist['new_low_20'] = ((hist['low'] <= low20) & low20.notna()).astype(float)

    keep = [
        'date', 'code',
        'above_ma5', 'above_ma10', 'above_ma20', 'above_ma60',
        'new_high_20', 'new_low_20',
    ]
    hist = hist[keep]

    active_universe = universe[universe['in_universe'] == 1].copy()
    active_universe['date'] = active_universe['date'].astype(str)
    active_universe['code'] = active_universe['code'].astype(str).str.zfill(6)

    merged = active_universe[['date', 'code']].merge(hist, on=['date', 'code'], how='left')
    features = merged.groupby('date', as_index=False).agg({
        'above_ma5': 'mean',
        'above_ma10': 'mean',
        'above_ma20': 'mean',
        'above_ma60': 'mean',
        'new_high_20': 'mean',
        'new_low_20': 'mean',
    })
    features = features.rename(columns={
        'above_ma5': 'above_ma5_ratio',
        'above_ma10': 'above_ma10_ratio',
        'above_ma20': 'above_ma20_ratio',
        'above_ma60': 'above_ma60_ratio',
        'new_high_20': 'new_high_20_ratio',
        'new_low_20': 'new_low_20_ratio',
    })
    return features.sort_values('date').reset_index(drop=True)



def _build_cross_section_for_date(trade_date: str, universe: pd.DataFrame, limit_summary: pd.DataFrame, moneyflow_hist: pd.DataFrame, index_hist: pd.DataFrame, market_structure_features: pd.DataFrame) -> pd.DataFrame:
    metrics = load_daily_metrics(trade_date)
    if metrics.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS['cross_section'])
    metrics = metrics[metrics['date'] == trade_date].drop_duplicates(['date', 'code'])

    day_universe = universe[universe['date'].astype(str) == trade_date].copy()
    day_universe = day_universe[day_universe['in_universe'] == 1].copy()
    if day_universe.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS['cross_section'])

    base = day_universe.merge(metrics, on=['date', 'code'], how='left')

    moneyflow = moneyflow_hist[moneyflow_hist['date'] == trade_date].copy()
    if not moneyflow.empty:
        keep = [c for c in ['date', 'code', 'small_net', 'large_net', 'super_large_net'] if c in moneyflow.columns]
        base = base.merge(moneyflow[keep], on=['date', 'code'], how='left')

    chip = load_chip_distribution(trade_date)
    if not chip.empty:
        keep = [c for c in ['date', 'code', 'cost_50', 'cost_85', 'avg_cost', 'winner_ratio'] if c in chip.columns]
        base = base.merge(chip[keep], on=['date', 'code'], how='left')

    ladder = load_limit_ladder(trade_date)
    if not ladder.empty:
        base = base.merge(ladder[['date', 'code', 'lb_count']], on=['date', 'code'], how='left')

    day_limit = limit_summary[limit_summary['date'] == trade_date].copy() if not limit_summary.empty else pd.DataFrame()
    if len(day_limit) > 0:
        for col in ['zt_count', 'dt_count', 'zb_count']:
            base[col] = day_limit.iloc[0].get(col)

    if 'lb_count' not in base.columns:
        base['lb_count'] = 0
    base['lb_count'] = pd.to_numeric(base['lb_count'], errors='coerce').fillna(0)
    base['is_zt'] = (base['lb_count'] >= 1).astype(int)
    base['is_dt'] = 0
    base['is_zb'] = 0

    industry = load_industry_components(trade_date)
    if not industry.empty:
        if 'industry_name_x' in base.columns:
            base = base.drop(columns=['industry_name_x'])
        base = base.merge(industry[['code', 'industry_name']], on='code', how='left', suffixes=('', '_new'))
        if 'industry_name_new' in base.columns:
            base['industry_name'] = base['industry_name'].fillna(base['industry_name_new'])
            base = base.drop(columns=['industry_name_new'])

    concept = load_concept_components(trade_date)
    if not concept.empty:
        base = base.merge(concept, on='code', how='left')
    if 'concept_count' not in base.columns:
        base['concept_count'] = 0

    index_close = index_hist[index_hist['date'] == trade_date].copy()
    if not index_close.empty:
        base = base.merge(index_close, on='date', how='left')

    day_market_structure = market_structure_features[market_structure_features['date'] == trade_date].copy()
    if not day_market_structure.empty:
        base = base.merge(day_market_structure, on='date', how='left')

    for col in DEFAULT_COLUMNS['cross_section']:
        if col not in base.columns:
            base[col] = pd.NA

    return base[DEFAULT_COLUMNS['cross_section']].copy().sort_values(['date', 'code']).reset_index(drop=True)


def _build_cross_section_range(dates, universe: pd.DataFrame, limit_summary: pd.DataFrame, moneyflow_hist: pd.DataFrame, index_hist: pd.DataFrame, market_structure_features: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for trade_date in dates:
        part = _build_cross_section_for_date(trade_date, universe, limit_summary, moneyflow_hist, index_hist, market_structure_features)
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=DEFAULT_COLUMNS['cross_section'])
    output = pd.concat(parts, ignore_index=True)
    output = output.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code']).reset_index(drop=True)
    return output


def build_daily_cross_section(date=None, start_date=None, end_date=None, output_name='daily_cross_section.csv'):
    if start_date is None:
        from data_layer.foundation_config import UNIVERSE_CONFIG
        start_date = UNIVERSE_CONFIG.get('history_start_date')
    requested_start_date = normalize_date(start_date) if start_date is not None else None

    universe_path = foundation_file('main_board_universe.csv')
    if os.path.exists(universe_path):
        universe = pd.read_csv(universe_path, encoding='utf-8-sig', dtype={'code': str})
    else:
        from data_layer.prepare_main_board_universe import build_main_board_universe
        universe = build_main_board_universe(start_date=start_date)

    if universe.empty:
        raise ValueError('main_board_universe.csv 为空，无法构建横截面')

    limit_summary = load_limit_summary()
    moneyflow_hist = load_moneyflow_history(start_date=start_date)
    index_hist = load_index_closes_history(start_date=start_date)
    metrics_history_start = None
    if requested_start_date is not None:
        metrics_history_start = (pd.to_datetime(requested_start_date) - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
    metrics_hist = load_daily_metrics_history(start_date=metrics_history_start)
    market_structure_features = _build_market_structure_features(metrics_hist, universe)
    dates = sorted(universe['date'].astype(str).unique())
    if requested_start_date is not None:
        dates = [d for d in dates if d >= requested_start_date]
    if date is not None:
        trade_date = normalize_date(date)
        dates = [d for d in dates if d == trade_date]
    if end_date is not None:
        end_date = normalize_date(end_date)
        dates = [d for d in dates if d <= end_date]
    if not dates:
        raise ValueError('没有可构建的交易日')

    output = _build_cross_section_range(dates, universe, limit_summary, moneyflow_hist, index_hist, market_structure_features)
    if output.empty:
        raise FileNotFoundError('找不到可用横截面数据')

    out_path = foundation_file(output_name)
    output.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('日度横截面构建完成')
    print('=' * 80)
    print(f'日期范围: {output["date"].min()} ~ {output["date"].max()}')
    print(f'样本数: {len(output)}')
    print(f'输出: {out_path}')
    return output


def build_daily_cross_section_segmented(segments):
    built_files = []
    for start_date, end_date, output_name in segments:
        build_daily_cross_section(start_date=start_date, end_date=end_date, output_name=output_name)
        built_files.append(foundation_file(output_name))

    parts = []
    for path in built_files:
        if os.path.exists(path):
            parts.append(pd.read_csv(path, encoding='utf-8-sig', dtype={'code': str}))
    if not parts:
        raise FileNotFoundError('分段横截面未生成成功')

    output = pd.concat(parts, ignore_index=True)
    output = output.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code']).reset_index(drop=True)
    out_path = foundation_file('daily_cross_section.csv')
    output.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('分段横截面合并完成')
    print('=' * 80)
    print(f'日期范围: {output["date"].min()} ~ {output["date"].max()}')
    print(f'样本数: {len(output)}')
    print(f'输出: {out_path}')
    return output


if __name__ == '__main__':
    build_daily_cross_section()
