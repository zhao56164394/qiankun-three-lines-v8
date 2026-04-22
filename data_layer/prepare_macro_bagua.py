# -*- coding: utf-8 -*-
"""
prepare_macro_bagua.py

基于新底座日度横截面，生成更慢周期的市场级八卦序列。
用于判断牛熊切换与大周期状态，不替代现有日线 market_bagua。
"""
import os
import sys
from itertools import product

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_bagua import GUA_NAME_MAP
from data_layer.prepare_market_bagua import _build_index_anchor, _mark_segments
from strategy.indicator import calc_trend_line


OUTPUT_COLUMNS = [
    'date', 'stock_count',
    'market_open_proxy', 'market_high_proxy', 'market_low_proxy', 'market_close_proxy',
    'up_ratio', 'zt_ratio', 'dt_ratio', 'turnover_median',
    'zt_count', 'dt_count', 'zb_count', 'limit_heat', 'limit_quality', 'ladder_heat',
    'above_ma20_ratio', 'above_ma60_ratio', 'new_high_20_ratio', 'new_low_20_ratio',
    'market_trend_slow', 'market_trend_anchor_slow', 'market_speed_slow', 'macro_breadth_slow',
    'yao_1', 'yao_2', 'yao_3', 'gua_code', 'gua_name',
    'prev_gua', 'changed', 'seg_id', 'seg_day',
]

DEFAULT_MACRO_PARAMS = {
    'trend_period': 180,
    'yao1_window': 250,
    'yao1_min_periods': 160,
    'yao1_q_low': 0.4,
    'yao1_q_high': 0.6,
    'speed_lookback': 80,
    'breadth_fast_window': 30,
    'breadth_slow_window': 60,
    'breadth_weight_ma60': 0.75,
    'breadth_weight_ma20': 0.25,
    'breakout_weight': 0.9,
    'heat_weight': 0.25,
}


def get_macro_params(overrides=None):
    params = DEFAULT_MACRO_PARAMS.copy()
    if overrides:
        params.update(overrides)
    return params


def iter_macro_param_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _calc_dynamic_macro_yao1(trend: pd.Series, params: dict):
    window = int(params['yao1_window'])
    min_periods = int(params['yao1_min_periods'])
    q_low = float(params['yao1_q_low'])
    q_high = float(params['yao1_q_high'])
    q40 = trend.rolling(window, min_periods=min_periods).quantile(q_low).shift(1)
    q60 = trend.rolling(window, min_periods=min_periods).quantile(q_high).shift(1)
    anchor = trend.rolling(window, min_periods=min_periods).median().shift(1)
    values = []
    prev = None
    for t, low, high, mid in zip(trend.tolist(), q40.tolist(), q60.tolist(), anchor.tolist()):
        if pd.isna(t) or pd.isna(low) or pd.isna(high) or pd.isna(mid):
            values.append(pd.NA)
            continue
        if t >= high:
            prev = 1
        elif t <= low:
            prev = 0
        elif prev is None:
            prev = 1 if t >= mid else 0
        values.append(prev)
    return pd.Series(values, index=trend.index), anchor


def _calc_macro_features(market: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = get_macro_params(params)
    market = market.sort_values('date').reset_index(drop=True).copy()
    numeric_cols = [
        'above_ma20_ratio', 'above_ma60_ratio', 'new_high_20_ratio', 'new_low_20_ratio',
        'zt_count', 'dt_count', 'zb_count', 'ladder_heat',
    ]
    for col in numeric_cols:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors='coerce')
        else:
            market[col] = pd.NA

    stock_count = pd.to_numeric(market['stock_count'], errors='coerce')
    safe_stock_count = stock_count.where(stock_count > 0)
    market['limit_heat'] = market['zt_count'] / safe_stock_count
    market['limit_quality'] = (market['zt_count'] - market['zb_count']) / safe_stock_count

    market['market_trend_slow'] = calc_trend_line(
        market['market_close_proxy'].values,
        market['market_high_proxy'].values,
        market['market_low_proxy'].values,
        period=int(params['trend_period']),
    )
    trend_series = pd.Series(market['market_trend_slow'])
    market['yao_1'], market['market_trend_anchor_slow'] = _calc_dynamic_macro_yao1(trend_series, params)
    market['market_speed_slow'] = market['market_trend_slow'] - market['market_trend_slow'].shift(int(params['speed_lookback']))

    breadth_core = (
        float(params['breadth_weight_ma60']) * pd.to_numeric(market['above_ma60_ratio'], errors='coerce')
        + float(params['breadth_weight_ma20']) * pd.to_numeric(market['above_ma20_ratio'], errors='coerce')
    )
    breakout_core = (
        pd.to_numeric(market['new_high_20_ratio'], errors='coerce')
        - pd.to_numeric(market['new_low_20_ratio'], errors='coerce')
    )
    heat_confirm = (
        0.5 * pd.to_numeric(market['limit_quality'], errors='coerce').fillna(0.0)
        + 0.25 * pd.to_numeric(market['limit_heat'], errors='coerce').fillna(0.0)
        + 0.25 * pd.to_numeric(market['ladder_heat'], errors='coerce').fillna(0.0)
    )
    fast_window = int(params['breadth_fast_window'])
    slow_window = int(params['breadth_slow_window'])
    market['macro_breadth_slow'] = (
        breadth_core.rolling(fast_window, min_periods=fast_window).mean()
        - breadth_core.rolling(slow_window, min_periods=slow_window).mean()
        + float(params['breakout_weight']) * breakout_core.rolling(fast_window, min_periods=fast_window).mean()
        + float(params['heat_weight']) * heat_confirm.rolling(fast_window, min_periods=fast_window).mean()
    )

    market = market.dropna(subset=['market_trend_slow', 'market_trend_anchor_slow', 'market_speed_slow', 'macro_breadth_slow']).copy()
    if market.empty:
        return market

    market['yao_1'] = pd.to_numeric(market['yao_1'], errors='coerce').astype(int)
    market['yao_2'] = (pd.to_numeric(market['market_speed_slow'], errors='coerce') > 0).astype(int)
    market['yao_3'] = (pd.to_numeric(market['macro_breadth_slow'], errors='coerce') > 0).astype(int)
    market['gua_code'] = market.apply(lambda row: f"{int(row['yao_1'])}{int(row['yao_2'])}{int(row['yao_3'])}", axis=1)
    market['gua_name'] = market['gua_code'].map(GUA_NAME_MAP)
    return market


def _build_macro_market_base():
    cross_path = foundation_file('daily_cross_section.csv')
    if not os.path.exists(cross_path):
        raise FileNotFoundError(f'daily_cross_section.csv 不存在: {cross_path}')

    cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    index_anchor = _build_index_anchor(cross)

    for col in ['zt_count', 'dt_count', 'zb_count', 'lb_count']:
        if col not in cross.columns:
            cross[col] = pd.NA

    base_cols = [
        'date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate', 'is_zt', 'is_dt',
        'zt_count', 'dt_count', 'zb_count', 'lb_count',
        'above_ma20_ratio', 'above_ma60_ratio', 'new_high_20_ratio', 'new_low_20_ratio',
    ]
    base = cross[base_cols].copy()
    base = base.sort_values(['code', 'date']).reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close', 'turnover_rate', 'zt_count', 'dt_count', 'zb_count', 'lb_count']:
        base[col] = pd.to_numeric(base[col], errors='coerce')
    base['is_zt'] = pd.to_numeric(base['is_zt'], errors='coerce').fillna(0)
    base['is_dt'] = pd.to_numeric(base['is_dt'], errors='coerce').fillna(0)
    base['lb_count'] = base['lb_count'].fillna(0)

    prev_close = base.groupby('code')['close'].shift(1)
    base['is_up'] = (base['close'] > prev_close).astype(float)
    base.loc[prev_close.isna(), 'is_up'] = pd.NA

    market = base.groupby('date', sort=True).agg({
        'code': 'count',
        'open': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean',
        'is_up': 'mean',
        'is_zt': 'mean',
        'is_dt': 'mean',
        'turnover_rate': 'median',
        'zt_count': 'first',
        'dt_count': 'first',
        'zb_count': 'first',
        'lb_count': lambda s: s.clip(upper=3).mean() / 3.0,
        'above_ma20_ratio': 'mean',
        'above_ma60_ratio': 'mean',
        'new_high_20_ratio': 'mean',
        'new_low_20_ratio': 'mean',
    }).reset_index().rename(columns={
        'code': 'stock_count',
        'open': 'market_open_proxy',
        'high': 'market_high_proxy',
        'low': 'market_low_proxy',
        'close': 'market_close_proxy',
        'is_up': 'up_ratio',
        'is_zt': 'zt_ratio',
        'is_dt': 'dt_ratio',
        'turnover_rate': 'turnover_median',
        'lb_count': 'ladder_heat',
    })

    market = market.merge(index_anchor, on='date', how='left')
    market['market_close_proxy'] = market['market_index_anchor'].combine_first(market['market_close_proxy'])
    market['market_open_proxy'] = market['market_index_anchor'].combine_first(market['market_open_proxy'])
    market['market_high_proxy'] = market['market_index_anchor'].combine_first(market['market_high_proxy'])
    market['market_low_proxy'] = market['market_index_anchor'].combine_first(market['market_low_proxy'])
    return market


def build_macro_bagua(params: dict | None = None, write_output: bool = True):
    params = get_macro_params(params)
    market = _build_macro_market_base()
    market = _calc_macro_features(market, params)
    if market.empty:
        raise ValueError('无法生成大周期八卦：慢周期特征为空，请检查 warmup 区间或市场代理序列')

    market = _mark_segments(market)

    for col in OUTPUT_COLUMNS:
        if col not in market.columns:
            market[col] = pd.NA
    market = market[OUTPUT_COLUMNS].sort_values('date').reset_index(drop=True)

    if write_output:
        out_path = foundation_file('macro_bagua_daily.csv')
        market.to_csv(out_path, index=False, encoding='utf-8-sig')

        print('=' * 80)
        print('大周期八卦生成完成')
        print('=' * 80)
        print(f'日期范围: {market["date"].min()} ~ {market["date"].max()}')
        print(f'交易日数: {len(market)}')
        print(f'卦类数: {market["gua_code"].nunique()}')
        print(f'输出: {out_path}')
    return market


if __name__ == '__main__':
    build_macro_bagua()
