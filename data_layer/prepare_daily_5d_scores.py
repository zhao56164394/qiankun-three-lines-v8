# -*- coding: utf-8 -*-
"""
prepare_daily_5d_scores.py

基于日度横截面宽表计算 5 维分数：位 / 势 / 变 / 众 / 气
方案A：完全摆脱旧三线依赖，只使用新底座可稳定取得的数据。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file


BAGUA_SCORE_COLUMNS = [
    'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi',
    'wei_cost_score', 'wei_winner_score', 'wei_pb_score',
    'shi_index_score', 'shi_turnover_score',
    'bian_flow_score', 'bian_chip_score',
    'zhong_amount_score', 'zhong_concept_score', 'zhong_event_score',
    'qi_flow_score', 'qi_turnover_score', 'qi_crowding_score',
]


INDEX_WEIGHTS = {
    'allA_close': 0.30,
    'hs300_close': 0.20,
    'csi500_close': 0.18,
    'csi1000_close': 0.18,
    'sh_close': 0.07,
    'sz_close': 0.07,
}


def _clip_0_100(series):
    return pd.to_numeric(series, errors='coerce').clip(0, 100)


def _safe_rank_pct(series, ascending=True):
    s = pd.to_numeric(series, errors='coerce')
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    ranked = s.rank(pct=True, ascending=ascending, method='average')
    return ranked * 100


def _safe_div(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    out = a / b
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _weighted_index_level(df):
    pieces = []
    weights = []
    for col, w in INDEX_WEIGHTS.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors='coerce')
            if s.notna().sum() > 0:
                pieces.append(s)
                weights.append(w)
    if not pieces:
        return pd.Series(50.0, index=df.index)

    stacked = pd.concat(pieces, axis=1)
    row_mean = stacked.mean(axis=1)
    total_weight = sum(weights)
    if total_weight <= 0:
        return pd.Series(50.0, index=df.index)

    norm = 0
    for s, w in zip(pieces, weights):
        norm = norm + s.fillna(row_mean) * (w / total_weight)
    return _safe_rank_pct(norm).fillna(50)


def _build_scores_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    df = day_df.copy()

    close = pd.to_numeric(df.get('close'), errors='coerce')
    avg_cost = pd.to_numeric(df.get('avg_cost'), errors='coerce')
    cost_50 = pd.to_numeric(df.get('cost_50'), errors='coerce')
    cost_85 = pd.to_numeric(df.get('cost_85'), errors='coerce')
    winner_ratio = pd.to_numeric(df.get('winner_ratio'), errors='coerce')
    pb = pd.to_numeric(df.get('pb'), errors='coerce')
    amount = pd.to_numeric(df.get('amount'), errors='coerce')
    turnover_rate = pd.to_numeric(df.get('turnover_rate'), errors='coerce')
    concept_count = pd.to_numeric(df.get('concept_count'), errors='coerce')
    small_net = pd.to_numeric(df.get('small_net'), errors='coerce')
    large_net = pd.to_numeric(df.get('large_net'), errors='coerce')
    super_large_net = pd.to_numeric(df.get('super_large_net'), errors='coerce')
    lb_count = pd.to_numeric(df.get('lb_count'), errors='coerce')
    is_zt = pd.to_numeric(df.get('is_zt', 0), errors='coerce').fillna(0)
    is_dt = pd.to_numeric(df.get('is_dt', 0), errors='coerce').fillna(0)

    cost_ratio = _safe_div(close, avg_cost)
    cost_band_ratio = _safe_div(close, cost_50)
    df['wei_cost_score'] = _clip_0_100(((cost_ratio - 0.85) / 0.35) * 100)
    df['wei_winner_score'] = _clip_0_100(winner_ratio)
    df['wei_pb_score'] = _safe_rank_pct(pb, ascending=False)
    df['score_wei'] = pd.concat([
        df['wei_cost_score'],
        df['wei_winner_score'],
        df['wei_pb_score'],
        _clip_0_100(((cost_band_ratio - 0.85) / 0.35) * 100),
    ], axis=1).mean(axis=1).round(4)

    df['shi_index_score'] = _weighted_index_level(df)
    df['shi_turnover_score'] = _safe_rank_pct(turnover_rate).fillna(50)
    df['score_shi'] = pd.concat([
        df['shi_index_score'],
        df['shi_turnover_score'],
    ], axis=1).mean(axis=1).round(4)

    chip_spread = _safe_div(cost_85 - cost_50, cost_50)
    net_combo = pd.concat([
        _safe_rank_pct(large_net),
        _safe_rank_pct(super_large_net),
        _safe_rank_pct(-small_net),
    ], axis=1).mean(axis=1)
    df['bian_flow_score'] = net_combo.fillna(50)
    df['bian_chip_score'] = _safe_rank_pct(chip_spread, ascending=False).fillna(50)
    df['score_bian'] = pd.concat([
        df['bian_flow_score'],
        df['bian_chip_score'],
    ], axis=1).mean(axis=1).round(4)

    event_heat = lb_count.fillna(0) * 20 + is_zt * 20 - is_dt * 20
    df['zhong_amount_score'] = _safe_rank_pct(amount).fillna(50)
    df['zhong_concept_score'] = _safe_rank_pct(concept_count).fillna(50)
    df['zhong_event_score'] = _clip_0_100(event_heat + 50).fillna(50)
    df['score_zhong'] = pd.concat([
        df['zhong_amount_score'],
        df['zhong_concept_score'],
        df['zhong_event_score'],
    ], axis=1).mean(axis=1).round(4)

    crowding = pd.concat([
        _safe_rank_pct(turnover_rate),
        _safe_rank_pct(winner_ratio),
        _safe_rank_pct(cost_ratio),
    ], axis=1).mean(axis=1)
    df['qi_flow_score'] = net_combo.fillna(50)
    df['qi_turnover_score'] = _safe_rank_pct(turnover_rate).fillna(50)
    df['qi_crowding_score'] = crowding.fillna(50)
    df['score_qi'] = pd.concat([
        df['qi_flow_score'],
        df['qi_turnover_score'],
        df['qi_crowding_score'],
    ], axis=1).mean(axis=1).round(4)

    for col in BAGUA_SCORE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce').round(4)

    keep_cols = ['date', 'code'] + [c for c in BAGUA_SCORE_COLUMNS if c in df.columns]
    return df[keep_cols].sort_values(['date', 'code']).reset_index(drop=True)


def build_daily_5d_scores(date=None):
    cross_path = foundation_file('daily_cross_section.csv')
    if not os.path.exists(cross_path):
        from data_layer.prepare_daily_cross_section import build_daily_cross_section
        build_daily_cross_section(date=date)

    df = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str})
    if df.empty:
        raise ValueError('daily_cross_section.csv 为空，无法计算 5 维分数')

    if date is not None:
        df = df[df['date'].astype(str) == str(date)].copy()
    if df.empty:
        raise ValueError('指定日期无横截面数据，无法计算 5 维分数')

    parts = []
    for _, day_df in df.groupby('date', sort=True):
        parts.append(_build_scores_for_day(day_df))
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code']).reset_index(drop=True)

    out_path = foundation_file('daily_5d_scores.csv')
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('5维分数构建完成')
    print('=' * 80)
    print(f'日期范围: {out["date"].min()} ~ {out["date"].max()}')
    print(f'样本数: {len(out)}')
    print(f'输出: {out_path}')
    return out


if __name__ == '__main__':
    build_daily_5d_scores()
