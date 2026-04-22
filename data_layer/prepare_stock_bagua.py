# -*- coding: utf-8 -*-
"""
prepare_stock_bagua.py

基于新底座日度横截面 + 5维分数，生成与 market/macro 同语义的个股级八卦序列。
三爻定义：
1. 位置：个股趋势线相对自身动态锚
2. 速度：个股趋势线20日变化方向
3. 热度：个股换手/突破/涨停热度/短线分数动能的合成代理
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_bagua import GUA_NAME_MAP
from strategy.indicator import calc_trend_line


OUTPUT_COLUMNS = [
    'date', 'code', 'gua_code', 'gua_name',
    'yao_1', 'yao_2', 'yao_3',
    'open', 'high', 'low', 'close', 'turnover_rate',
    'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi',
    'trend_value', 'trend_anchor', 'speed_value', 'heat_momo',
    'prev_gua', 'changed', 'seg_id', 'seg_day',
]


HEALTH_SUMMARY_COLUMNS = [
    'sample_count', 'code_count', 'date_count', 'date_min', 'date_max',
    'valid_ratio', 'bagua_count', 'change_rate', 'segment_length_mean', 'segment_length_median',
    'short_segment_ratio_le_2', 'short_segment_ratio_le_5',
]


NUMERIC_COLS = [
    'open', 'high', 'low', 'close', 'turnover_rate',
    'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi',
    'lb_count', 'is_zt',
]


HEAT_FEATURE_COLUMNS = [
    'turnover_ma5', 'turnover_ma20', 'turnover_momo',
    'high_20_prev', 'low_20_prev', 'breakout_flag', 'breakdown_flag', 'breakout_bias',
    'lb_clip', 'score_qi_ma5', 'score_qi_ma20', 'score_qi_momo',
    'score_zhong_ma5', 'score_zhong_ma20', 'score_zhong_momo',
    'heat_momo_raw', 'heat_momo',
]


HEAT_ON_THRESHOLD = 0.08
HEAT_OFF_THRESHOLD = -0.08


def _calc_dynamic_stock_yao1(trend: pd.Series):
    q40 = trend.rolling(120, min_periods=60).quantile(0.4).shift(1)
    q60 = trend.rolling(120, min_periods=60).quantile(0.6).shift(1)
    anchor = trend.rolling(120, min_periods=60).median().shift(1)
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


def _calc_heat_yao3(heat_series: pd.Series):
    values = []
    prev = None
    for val in pd.to_numeric(heat_series, errors='coerce').tolist():
        if pd.isna(val):
            values.append(pd.NA)
            continue
        if val >= HEAT_ON_THRESHOLD:
            prev = 1
        elif val <= HEAT_OFF_THRESHOLD:
            prev = 0
        elif prev is None:
            prev = 1 if val > 0 else 0
        values.append(prev)
    return pd.Series(values, index=heat_series.index)


def _mark_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['code', 'date']).copy()
    df['prev_gua'] = df.groupby('code')['gua_code'].shift(1).fillna('')
    df['changed'] = (df['gua_code'] != df['prev_gua']).astype(int)
    df.loc[df['prev_gua'] == '', 'changed'] = 1
    df['seg_id'] = df.groupby('code')['changed'].cumsum().astype(int)
    df['seg_day'] = df.groupby(['code', 'seg_id']).cumcount() + 1
    return df


def _build_base_frame() -> pd.DataFrame:
    cross_path = foundation_file('daily_cross_section.csv')
    score_path = foundation_file('daily_5d_scores.csv')
    if not os.path.exists(cross_path):
        raise FileNotFoundError(f'daily_cross_section.csv 不存在: {cross_path}')
    if not os.path.exists(score_path):
        raise FileNotFoundError(f'daily_5d_scores.csv 不存在: {score_path}')

    cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    scores = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)

    keep_cross = ['date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate', 'lb_count', 'is_zt']
    keep_scores = ['date', 'code', 'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi']

    for col in keep_cross:
        if col not in cross.columns:
            cross[col] = pd.NA
    for col in keep_scores:
        if col not in scores.columns:
            scores[col] = pd.NA

    df = cross[keep_cross].merge(scores[keep_scores], on=['date', 'code'], how='left')
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = df['date'].astype(str)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['lb_count'] = df['lb_count'].fillna(0)
    df['is_zt'] = df['is_zt'].fillna(0)
    return df.sort_values(['code', 'date']).reset_index(drop=True)


def _build_stock_features(group: pd.DataFrame) -> pd.DataFrame:
    df = group.sort_values('date').copy()
    if len(df) < 60:
        return pd.DataFrame(columns=df.columns.tolist() + OUTPUT_COLUMNS + HEAT_FEATURE_COLUMNS)

    close = pd.to_numeric(df['close'], errors='coerce').values
    high = pd.to_numeric(df['high'], errors='coerce').values
    low = pd.to_numeric(df['low'], errors='coerce').values
    df['trend_value'] = calc_trend_line(close, high, low, period=55)

    trend_series = pd.Series(df['trend_value'], index=df.index)
    df['yao_1'], df['trend_anchor'] = _calc_dynamic_stock_yao1(trend_series)
    df['speed_value'] = df['trend_value'] - df['trend_value'].shift(20)

    df['turnover_ma5'] = df['turnover_rate'].rolling(5, min_periods=5).mean()
    df['turnover_ma20'] = df['turnover_rate'].rolling(20, min_periods=20).mean()
    df['turnover_momo'] = (df['turnover_ma5'] / df['turnover_ma20']) - 1.0
    df['turnover_momo'] = df['turnover_momo'].replace([np.inf, -np.inf], np.nan)

    df['high_20_prev'] = df['high'].rolling(20, min_periods=20).max().shift(1)
    df['low_20_prev'] = df['low'].rolling(20, min_periods=20).min().shift(1)
    df['breakout_flag'] = (df['close'] > df['high_20_prev']).astype(float)
    df.loc[df['high_20_prev'].isna(), 'breakout_flag'] = np.nan
    df['breakdown_flag'] = (df['close'] < df['low_20_prev']).astype(float)
    df.loc[df['low_20_prev'].isna(), 'breakdown_flag'] = np.nan
    df['breakout_bias'] = (df['close'] / df['high_20_prev']) - 1.0
    df['breakout_bias'] = df['breakout_bias'].replace([np.inf, -np.inf], np.nan)

    df['lb_clip'] = df['lb_count'].clip(lower=0, upper=3) / 3.0
    df['score_qi_ma5'] = df['score_qi'].rolling(5, min_periods=5).mean()
    df['score_qi_ma20'] = df['score_qi'].rolling(20, min_periods=20).mean()
    df['score_qi_momo'] = (df['score_qi_ma5'] - df['score_qi_ma20']) / 100.0
    df['score_zhong_ma5'] = df['score_zhong'].rolling(5, min_periods=5).mean()
    df['score_zhong_ma20'] = df['score_zhong'].rolling(20, min_periods=20).mean()
    df['score_zhong_momo'] = (df['score_zhong_ma5'] - df['score_zhong_ma20']) / 100.0

    df['heat_momo_raw'] = (
        0.30 * df['turnover_momo'].fillna(0.0)
        + 0.20 * df['breakout_flag'].fillna(0.0)
        - 0.15 * df['breakdown_flag'].fillna(0.0)
        + 0.08 * df['breakout_bias'].clip(lower=-0.3, upper=0.3).fillna(0.0)
        + 0.07 * df['lb_clip'].fillna(0.0)
        + 0.05 * df['is_zt'].fillna(0.0)
        + 0.10 * df['score_qi_momo'].fillna(0.0)
        + 0.10 * df['score_zhong_momo'].fillna(0.0)
    )
    df['heat_momo'] = df['heat_momo_raw'].rolling(5, min_periods=3).mean()

    df = df.dropna(subset=['yao_1', 'trend_anchor', 'speed_value']).copy()
    if df.empty:
        return df

    df['yao_1'] = pd.to_numeric(df['yao_1'], errors='coerce').astype(int)
    df['yao_2'] = (pd.to_numeric(df['speed_value'], errors='coerce') > 0).astype(int)
    df['yao_3'] = _calc_heat_yao3(df['heat_momo'])
    df = df.dropna(subset=['yao_3']).copy()
    df['yao_3'] = pd.to_numeric(df['yao_3'], errors='coerce').astype(int)
    df['gua_code'] = df['yao_1'].astype(str) + df['yao_2'].astype(str) + df['yao_3'].astype(str)
    df['gua_name'] = df['gua_code'].map(GUA_NAME_MAP)
    return df


def _build_health_summary(stock_bagua: pd.DataFrame) -> dict:
    if stock_bagua is None or stock_bagua.empty:
        return {}

    valid = stock_bagua.dropna(subset=['gua_code']).copy()
    seg_len = valid.groupby(['code', 'seg_id'])['seg_day'].max()
    out = {
        'sample_count': int(len(stock_bagua)),
        'code_count': int(stock_bagua['code'].nunique()),
        'date_count': int(stock_bagua['date'].nunique()),
        'date_min': str(stock_bagua['date'].min()),
        'date_max': str(stock_bagua['date'].max()),
        'valid_ratio': round(float(len(valid) / len(stock_bagua)), 6) if len(stock_bagua) else 0.0,
        'bagua_count': int(valid['gua_code'].nunique()) if not valid.empty else 0,
        'gua_distribution': {str(k): round(float(v), 6) for k, v in valid['gua_code'].value_counts(normalize=True).sort_index().to_dict().items()},
        'change_rate': round(float(valid['changed'].mean()), 6) if 'changed' in valid.columns and not valid.empty else None,
        'segment_length_mean': round(float(seg_len.mean()), 4) if len(seg_len) else None,
        'segment_length_median': round(float(seg_len.median()), 4) if len(seg_len) else None,
        'short_segment_ratio_le_2': round(float((seg_len <= 2).mean()), 6) if len(seg_len) else None,
        'short_segment_ratio_le_5': round(float((seg_len <= 5).mean()), 6) if len(seg_len) else None,
        'avg_daily_code_count': round(float(valid.groupby('date')['code'].nunique().mean()), 4) if not valid.empty else None,
        'avg_code_change_rate': round(float(valid.groupby('code')['changed'].mean().mean()), 6) if not valid.empty else None,
    }
    return out


def _build_yearly_health(stock_bagua: pd.DataFrame) -> pd.DataFrame:
    if stock_bagua is None or stock_bagua.empty:
        return pd.DataFrame()
    rows = []
    tmp = stock_bagua.copy()
    tmp['year'] = tmp['date'].astype(str).str[:4]
    for year, year_df in tmp.groupby('year', sort=True):
        stats = _build_health_summary(year_df)
        if not stats:
            continue
        row = {'year': str(year)}
        for key in HEALTH_SUMMARY_COLUMNS:
            row[key] = stats.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def build_stock_bagua(write_output: bool = True):
    base = _build_base_frame()
    parts = []
    for _, group in base.groupby('code', sort=True):
        out = _build_stock_features(group)
        if out is not None and not out.empty:
            parts.append(out)

    if not parts:
        raise ValueError('无法生成个股卦：有效样本为空，请检查底层数据是否齐全')

    stock = pd.concat(parts, ignore_index=True)
    stock = _mark_segments(stock)
    for col in OUTPUT_COLUMNS:
        if col not in stock.columns:
            stock[col] = pd.NA
    stock = stock[OUTPUT_COLUMNS].sort_values(['date', 'code']).reset_index(drop=True)

    summary = _build_health_summary(stock)
    yearly = _build_yearly_health(stock)

    if write_output:
        out_path = foundation_file('stock_bagua_daily.csv')
        health_json_path = foundation_file('stock_bagua_health_summary.json')
        health_yearly_path = foundation_file('stock_bagua_yearly_health.csv')
        stock.to_csv(out_path, index=False, encoding='utf-8-sig')
        yearly.to_csv(health_yearly_path, index=False, encoding='utf-8-sig')
        with open(health_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print('=' * 80)
        print('个股八卦生成完成')
        print('=' * 80)
        print(f"日期范围: {stock['date'].min()} ~ {stock['date'].max()}")
        print(f"样本数: {len(stock)}")
        print(f"股票数: {stock['code'].nunique()}")
        print(f"卦类数: {stock['gua_code'].nunique()}")
        print(f"输出: {out_path}")
        print(f"健康汇总: {health_json_path}")
        print(f"年度健康: {health_yearly_path}")

    return stock, summary, yearly


if __name__ == '__main__':
    build_stock_bagua()
