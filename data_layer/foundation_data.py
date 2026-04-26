# -*- coding: utf-8 -*-
"""
foundation_data.py

新底座统一读取接口。
风格参考 gua_data.py，但保持语义独立。

数据格式：Parquet 优先，CSV 兜底（迁移期双轨支持）。
"""
import os
from typing import Optional

import pandas as pd

from data_layer.foundation_config import foundation_file, foundation_parquet


_cache = {}


def _load_table(cache_key: str, filename: str, force_reload: bool = False) -> pd.DataFrame:
    """读取数据表：Parquet 优先，CSV 兜底。"""
    if cache_key in _cache and not force_reload:
        return _cache[cache_key]
    pq_path = foundation_parquet(filename)
    csv_path = foundation_file(filename)
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    else:
        raise FileNotFoundError(f'数据文件不存在: {pq_path} / {csv_path}')
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.zfill(6)
    if 'gua_code' in df.columns:
        df['gua_code'] = df['gua_code'].astype(str).str.zfill(3)
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    _cache[cache_key] = df
    return df


def _load_csv(cache_key: str, filename: str, force_reload: bool = False) -> pd.DataFrame:
    """兼容旧名，新代码请用 _load_table。"""
    return _load_table(cache_key, filename, force_reload)



def load_daily_cross_section(force_reload: bool = False) -> pd.DataFrame:
    return _load_table('foundation_cross', 'daily_cross_section.csv', force_reload)


def load_daily_5d_scores(force_reload: bool = False) -> pd.DataFrame:
    return _load_table('foundation_scores', 'daily_5d_scores.csv', force_reload)


def load_daily_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_table('foundation_bagua', 'daily_bagua_sequence.csv', force_reload)



def load_market_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_table('foundation_market_bagua', 'market_bagua_daily.csv', force_reload)


def load_stock_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_table('foundation_stock_bagua', 'stock_bagua_daily.csv', force_reload)


def load_stock_bagua_map(force_reload: bool = False) -> dict:
    df = load_stock_bagua(force_reload=force_reload)
    if df.empty:
        return {}
    cols = [
        'gua_code', 'gua_name', 'yao_1', 'yao_2', 'yao_3',
        'trend_value', 'trend_anchor', 'speed_value', 'heat_momo',
        'prev_gua', 'changed', 'seg_id', 'seg_day',
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA

    # 向量化构造 (避免 iterrows: 7.7M 行 144s -> ~10s)
    sub = df[['date', 'code'] + cols].copy()
    sub['date'] = sub['date'].astype(str)
    sub['code'] = sub['code'].astype(str).str.zfill(6)
    sub['gua_code'] = sub['gua_code'].astype(str).str.zfill(3)

    keys = list(zip(sub['date'].values, sub['code'].values))
    records = sub.to_dict('records')
    out = {}
    for k, r in zip(keys, records):
        out[k] = {
            'di_gua': r['gua_code'],
            'di_gua_name': r.get('gua_name', '') or '',
            'stock_yao_1': r.get('yao_1', pd.NA),
            'stock_yao_2': r.get('yao_2', pd.NA),
            'stock_yao_3': r.get('yao_3', pd.NA),
            'stock_trend_value': r.get('trend_value', pd.NA),
            'stock_trend_anchor': r.get('trend_anchor', pd.NA),
            'stock_speed_value': r.get('speed_value', pd.NA),
            'stock_heat_momo': r.get('heat_momo', pd.NA),
            'stock_prev_gua': r.get('prev_gua', '') or '',
            'stock_changed': r.get('changed', pd.NA),
            'stock_seg_id': r.get('seg_id', pd.NA),
            'stock_seg_day': r.get('seg_day', pd.NA),
        }
    return out



