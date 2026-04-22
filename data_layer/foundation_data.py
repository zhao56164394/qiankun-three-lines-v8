# -*- coding: utf-8 -*-
"""
foundation_data.py

新底座统一读取接口。
风格参考 gua_data.py，但保持语义独立。
"""
import os
from typing import Optional

import pandas as pd

from data_layer.foundation_config import foundation_file


_cache = {}


def _load_csv(cache_key: str, filename: str, force_reload: bool = False) -> pd.DataFrame:
    if cache_key in _cache and not force_reload:
        return _cache[cache_key]
    path = foundation_file(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'数据文件不存在: {path}')
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.zfill(6)
    if 'gua_code' in df.columns:
        df['gua_code'] = df['gua_code'].astype(str).str.zfill(3)
    if 'date' in df.columns:
        df['date'] = df['date'].astype(str)
    _cache[cache_key] = df
    return df


def load_main_board_universe(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_universe', 'main_board_universe.csv', force_reload)


def load_daily_cross_section(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_cross', 'daily_cross_section.csv', force_reload)


def load_daily_5d_scores(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_scores', 'daily_5d_scores.csv', force_reload)


def load_daily_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_bagua', 'daily_bagua_sequence.csv', force_reload)


def load_daily_3yao(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_3yao', 'daily_3yao.csv', force_reload)


def load_market_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_market_bagua', 'market_bagua_daily.csv', force_reload)


def load_macro_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_macro_bagua', 'macro_bagua_daily.csv', force_reload)


def load_stock_bagua(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_stock_bagua', 'stock_bagua_daily.csv', force_reload)


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
    out = {}
    for _, row in df[['date', 'code'] + cols].iterrows():
        out[(str(row['date']), str(row['code']).zfill(6))] = {
            'stock_gua': str(row['gua_code']).zfill(3),
            'stock_gua_name': row.get('gua_name', ''),
            'stock_yao_1': row.get('yao_1', pd.NA),
            'stock_yao_2': row.get('yao_2', pd.NA),
            'stock_yao_3': row.get('yao_3', pd.NA),
            'stock_trend_value': row.get('trend_value', pd.NA),
            'stock_trend_anchor': row.get('trend_anchor', pd.NA),
            'stock_speed_value': row.get('speed_value', pd.NA),
            'stock_heat_momo': row.get('heat_momo', pd.NA),
            'stock_prev_gua': row.get('prev_gua', ''),
            'stock_changed': row.get('changed', pd.NA),
            'stock_seg_id': row.get('seg_id', pd.NA),
            'stock_seg_day': row.get('seg_day', pd.NA),
        }
    return out


def load_daily_forward_returns(force_reload: bool = False) -> pd.DataFrame:
    return _load_csv('foundation_forward_returns', 'daily_forward_returns.csv', force_reload)


def get_foundation_snapshot(date: Optional[str] = None) -> pd.DataFrame:
    df = load_daily_cross_section()
    if date is None:
        date = df['date'].max()
    return df[df['date'] == str(date)].copy().reset_index(drop=True)


def get_stock_foundation_series(code: str) -> pd.DataFrame:
    cross = load_daily_cross_section()
    score = load_daily_5d_scores()
    bagua = load_daily_bagua()
    out = cross[cross['code'] == str(code)].copy()
    if len(out) == 0:
        return out
    out = out.merge(score[score['code'] == str(code)], on=['date', 'code'], how='left')
    out = out.merge(bagua[bagua['code'] == str(code)], on=['date', 'code'], how='left')
    return out.sort_values('date').reset_index(drop=True)


def get_daily_bagua_distribution(date: Optional[str] = None) -> pd.DataFrame:
    df = load_daily_bagua()
    if date is None:
        date = df['date'].max()
    day = df[df['date'] == str(date)].copy()
    if len(day) == 0:
        return pd.DataFrame(columns=['gua_code', 'gua_name', 'count', 'ratio'])
    out = day.groupby(['gua_code', 'gua_name']).size().reset_index(name='count')
    out['ratio'] = out['count'] / len(day)
    return out.sort_values(['count', 'gua_code'], ascending=[False, True]).reset_index(drop=True)
