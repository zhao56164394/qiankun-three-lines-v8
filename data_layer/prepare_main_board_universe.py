# -*- coding: utf-8 -*-
"""
prepare_main_board_universe.py

生成主板样本池：
- 仅沪深主板
- 保留 ST（是否排除由配置控制）
- 剔除上市未满指定天数样本
- 支持单日与全历史输出
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import UNIVERSE_CONFIG, DEFAULT_COLUMNS, PATHS, foundation_file
from data_layer.foundation_sources import load_stock_basic, load_daily_metrics, normalize_date, list_csv_files_under


def _build_universe_for_metrics(metrics: pd.DataFrame, stock_basic: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS['universe'])

    metrics = metrics[['date', 'code']].drop_duplicates().copy()
    df = metrics.merge(stock_basic, on='code', how='left')
    df['date'] = df['date'].map(normalize_date)
    df['is_st'] = df['is_st'].fillna(False)
    df['list_date'] = df['list_date'].fillna('')

    listed_days = (pd.to_datetime(df['date']) - pd.to_datetime(df['list_date'], errors='coerce')).dt.days
    df['listed_days'] = listed_days

    allowed_market = df['board'].isin(UNIVERSE_CONFIG['allowed_market_types'])
    allowed_exchange = df['exchange'].isin(UNIVERSE_CONFIG['allowed_exchanges'])
    no_st = (~df['is_st']) if UNIVERSE_CONFIG['exclude_st'] else True
    enough_days = df['listed_days'].fillna(-1) >= UNIVERSE_CONFIG['min_list_days']
    no_excluded_prefix = ~df['code'].astype(str).str.startswith(tuple(UNIVERSE_CONFIG['exclude_prefixes']))

    df['in_universe'] = (allowed_market & allowed_exchange & no_st & enough_days & no_excluded_prefix).astype(int)
    output = df[DEFAULT_COLUMNS['universe']].copy()
    return output.sort_values(['date', 'code']).reset_index(drop=True)


def build_main_board_universe(date=None, start_date=None):
    stock_basic = load_stock_basic()
    if start_date is None:
        start_date = UNIVERSE_CONFIG.get('history_start_date')

    if date is not None:
        metrics = load_daily_metrics(date=date)
        if metrics.empty:
            raise FileNotFoundError('找不到每日指标数据，无法构建主板样本池')
        trade_date = normalize_date(date or metrics['date'].max())
        metrics = metrics[metrics['date'] == trade_date].copy()
        output = _build_universe_for_metrics(metrics, stock_basic)
    else:
        all_parts = []
        file_paths = list_csv_files_under(PATHS['stock_daily_metrics_root'])
        if start_date is not None:
            start_date = normalize_date(start_date)
        for path in file_paths:
            try:
                metrics = load_daily_metrics(file_path=path)
            except Exception:
                continue
            if metrics.empty:
                continue
            trade_date = normalize_date(metrics['date'].max())
            if start_date is not None and trade_date < start_date:
                continue
            part = _build_universe_for_metrics(metrics, stock_basic)
            if not part.empty:
                all_parts.append(part)
        if not all_parts:
            raise FileNotFoundError('找不到每日指标历史数据，无法构建主板样本池')
        output = pd.concat(all_parts, ignore_index=True)
        output = output.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code']).reset_index(drop=True)
        trade_date = str(output['date'].max())

    out_path = foundation_file('main_board_universe.csv')
    output.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('主板样本池构建完成')
    print('=' * 80)
    print(f'最新日期: {trade_date}')
    print(f'总样本: {len(output)}')
    print(f'入池样本: {int(output["in_universe"].sum())}')
    print(f'输出: {out_path}')
    return output


if __name__ == '__main__':
    build_main_board_universe()
