# -*- coding: utf-8 -*-
"""
verify_bagua_regime.py

v6.0 新底座下的八卦分治第一轮验证：
1. 生成市场卦日表
2. 生成个股未来收益标签
3. 验证不同市场卦下，主板股票未来收益分布是否存在稳定差异
"""
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_forward_returns import build_daily_forward_returns
from data_layer.prepare_market_bagua import build_market_bagua


HORIZONS = [1, 3, 5, 10, 20]
ROLLING_WINDOWS = [120, 250]
AUX_PERIODS = {
    '2014_2017': ('2014-01-01', '2017-12-31'),
    '2018_2022': ('2018-01-01', '2022-12-31'),
    '2023_2025': ('2023-01-01', '2025-12-31'),
    '2026': ('2026-01-01', '2026-12-31'),
    'all': ('2014-01-01', '2099-12-31'),
}


def _market_health_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    gua_col = 'market_gua_code' if 'market_gua_code' in df.columns else 'gua_code'
    gua = df[gua_col].dropna().astype(str).str.zfill(3)
    if gua.empty:
        return None
    vc = gua.value_counts(normalize=True)
    stats = {
        'bagua_count': int(gua.nunique()),
        'top1_ratio': round(float(vc.iloc[0]), 6) if len(vc) >= 1 else None,
        'top2_ratio': round(float(vc.iloc[:2].sum()), 6) if len(vc) >= 2 else None,
        'sample_count': int(len(df)),
    }
    for col in ['yao_1', 'yao_2', 'yao_3', 'market_yao_1', 'market_yao_2', 'market_yao_3']:
        if col in df.columns:
            target = col.replace('market_', '')
            stats[f'{target}_ratio'] = round(float(pd.to_numeric(df[col], errors='coerce').mean()), 6)
    seg_id_col = 'market_seg_id' if 'market_seg_id' in df.columns else 'seg_id'
    seg_day_col = 'market_seg_day' if 'market_seg_day' in df.columns else 'seg_day'
    if {seg_id_col, seg_day_col}.issubset(df.columns):
        seg = df[['date', seg_id_col, seg_day_col]].drop_duplicates(subset=['date'])
        if not seg.empty:
            seg_len = seg.groupby(seg_id_col)[seg_day_col].max()
            stats['segment_length_mean'] = round(float(seg_len.mean()), 4)
            stats['segment_length_median'] = round(float(seg_len.median()), 4)
            stats['change_rate'] = round(float(seg[seg_day_col].eq(1).mean()), 6)
    return stats


def _build_yearly_health(market: pd.DataFrame):
    rows = []
    tmp = market.copy()
    tmp['year'] = tmp['date'].astype(str).str[:4]
    for year, year_df in tmp.groupby('year', sort=True):
        stats = _market_health_stats(year_df)
        if stats:
            rows.append({'year': year, **stats})
    return pd.DataFrame(rows)


def _build_rolling_health(market: pd.DataFrame, windows):
    rows = []
    base = market[['date', 'gua_code', 'yao_1', 'yao_2', 'yao_3', 'seg_id', 'seg_day']].drop_duplicates('date').sort_values('date').reset_index(drop=True)
    for window in windows:
        if len(base) < window:
            continue
        for end_idx in range(window - 1, len(base)):
            sample = base.iloc[end_idx - window + 1:end_idx + 1].copy()
            stats = _market_health_stats(sample)
            if not stats:
                continue
            rows.append({
                'window': window,
                'start_date': sample['date'].iloc[0],
                'end_date': sample['date'].iloc[-1],
                **stats,
            })
    return pd.DataFrame(rows)


def _build_transition_events(market: pd.DataFrame):
    events = market[market['changed'] == 1].copy()
    if events.empty:
        return pd.DataFrame()
    events['prev_gua'] = events['prev_gua'].apply(_normalize_gua_code)
    events['to_gua'] = events['gua_code'].apply(_normalize_gua_code)
    events['move_type'] = '延续'
    events.loc[(events['market_speed_20'] > 0) & (events['breadth_momo'] > 0), 'move_type'] = '加速转强'
    events.loc[(events['market_speed_20'] > 0) & (events['breadth_momo'] <= 0), 'move_type'] = '趋势修复'
    events.loc[(events['market_speed_20'] <= 0) & (events['breadth_momo'] > 0), 'move_type'] = '高位分化或低位修复'
    events.loc[(events['market_speed_20'] <= 0) & (events['breadth_momo'] <= 0), 'move_type'] = '转弱或杀跌'
    cols = [
        'date', 'prev_gua', 'to_gua', 'gua_name', 'move_type', 'market_close_proxy',
        'market_trend_55', 'market_speed_20', 'breadth_momo', 'up_ratio', 'seg_id', 'seg_day'
    ]
    return events[cols].reset_index(drop=True)


def _normalize_gua_code(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return ''
    try:
        return str(int(float(text))).zfill(3)
    except ValueError:
        return text.zfill(3)


def _safe_t_stat(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    n = len(s)
    if n < 2:
        return None
    std = float(s.std(ddof=1))
    if std == 0:
        return None
    return round(float(s.mean()) / (std / math.sqrt(n)), 4)


def _aux_period_name(date_str: str) -> str:
    for name, (start, end) in AUX_PERIODS.items():
        if start <= str(date_str) <= end:
            return name
    return 'other'


def _build_yearly_effect(detail: pd.DataFrame):
    rows = []
    tmp = detail.copy()
    tmp['year'] = tmp['date'].astype(str).str[:4]
    for year, year_df in tmp.groupby('year', sort=True):
        for horizon in HORIZONS:
            col = f'ret_fwd_{horizon}d'
            valid = year_df[['market_gua_code', col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna(subset=[col, 'market_gua_code'])
            if valid.empty:
                continue
            grouped = valid.groupby('market_gua_code')[col].mean()
            if grouped.empty:
                continue
            rows.append({
                'year': year,
                'horizon': horizon,
                'overall_mean_ret': round(float(valid[col].mean()), 4),
                'best_mean_ret': round(float(grouped.max()), 4),
                'worst_mean_ret': round(float(grouped.min()), 4),
                'spread_best_worst': round(float(grouped.max() - grouped.min()), 4),
                'bagua_count': int(grouped.index.nunique()),
                'sample_count': int(len(valid)),
            })
    return pd.DataFrame(rows)


def _build_rolling_health_extremes(rolling_health: pd.DataFrame, windows):
    rows = []
    for window in windows:
        sample = rolling_health[rolling_health['window'] == window].copy()
        if sample.empty:
            continue
        worst = sample.sort_values(['bagua_count', 'top2_ratio', 'top1_ratio', 'end_date'], ascending=[True, False, False, True]).head(5).copy()
        worst['extreme_type'] = 'worst'
        best = sample.sort_values(['bagua_count', 'top2_ratio', 'top1_ratio', 'end_date'], ascending=[False, True, True, True]).head(5).copy()
        best['extreme_type'] = 'best'
        rows.extend(worst.to_dict(orient='records'))
        rows.extend(best.to_dict(orient='records'))
    return pd.DataFrame(rows)


def _build_transition_summary(transition_events: pd.DataFrame):
    if transition_events is None or transition_events.empty:
        return {}
    move_type_counts = transition_events['move_type'].value_counts().to_dict()
    to_gua_counts = transition_events['to_gua'].value_counts().to_dict()
    return {
        'move_type_counts': {str(k): int(v) for k, v in move_type_counts.items()},
        'to_gua_counts': {str(k): int(v) for k, v in to_gua_counts.items()},
        'latest_events': transition_events.tail(10).to_dict(orient='records'),
    }


def verify_bagua_regime():
    market_path = foundation_file('market_bagua_daily.csv')
    fwd_path = foundation_file('daily_forward_returns.csv')
    stock_bagua_path = foundation_file('daily_bagua_sequence.csv')

    if not os.path.exists(market_path):
        build_market_bagua()
    if not os.path.exists(fwd_path):
        build_daily_forward_returns()
    if not os.path.exists(stock_bagua_path):
        raise FileNotFoundError(f'daily_bagua_sequence.csv 不存在: {stock_bagua_path}')

    market = pd.read_csv(market_path, encoding='utf-8-sig', dtype={'gua_code': str}, low_memory=False)
    fwd = pd.read_csv(
        fwd_path,
        encoding='utf-8-sig',
        dtype={'code': str, 'avail_date_1d': str, 'avail_date_3d': str, 'avail_date_5d': str, 'avail_date_10d': str, 'avail_date_20d': str},
        low_memory=False,
    )
    stock_bagua = pd.read_csv(stock_bagua_path, encoding='utf-8-sig', dtype={'code': str, 'gua_code': str}, low_memory=False)

    market['date'] = market['date'].astype(str)
    fwd['date'] = fwd['date'].astype(str)
    stock_bagua['date'] = stock_bagua['date'].astype(str)
    stock_bagua['gua_code'] = stock_bagua['gua_code'].astype(str).str.zfill(3)
    market['gua_code'] = market['gua_code'].astype(str).str.zfill(3)

    detail = fwd.merge(
        market[['date', 'gua_code', 'gua_name', 'yao_1', 'yao_2', 'yao_3', 'seg_id', 'seg_day']],
        on='date',
        how='left',
        suffixes=('', '_market'),
    )
    detail = detail.rename(columns={
        'gua_code': 'market_gua_code',
        'gua_name': 'market_gua_name',
        'yao_1': 'market_yao_1',
        'yao_2': 'market_yao_2',
        'yao_3': 'market_yao_3',
        'seg_id': 'market_seg_id',
        'seg_day': 'market_seg_day',
    })
    detail = detail.merge(
        stock_bagua[['date', 'code', 'gua_code', 'gua_name']],
        on=['date', 'code'],
        how='left',
        suffixes=('', '_stock'),
    )
    detail = detail.rename(columns={
        'gua_code': 'stock_gua_code',
        'gua_name': 'stock_gua_name',
    })
    detail['aux_period'] = detail['date'].map(_aux_period_name)
    detail = detail[detail['aux_period'] != 'other'].copy()

    detail_path = foundation_file('bagua_regime_stock_returns_detail.csv')
    detail.to_csv(detail_path, index=False, encoding='utf-8-sig')

    yearly_health = _build_yearly_health(market)
    rolling_health = _build_rolling_health(market, ROLLING_WINDOWS)
    rolling_extremes = _build_rolling_health_extremes(rolling_health, ROLLING_WINDOWS)
    transition_events = _build_transition_events(market)
    transition_summary = _build_transition_summary(transition_events)
    yearly_effect = _build_yearly_effect(detail)

    summary_rows = []
    effect_rows = []
    health_rows = []

    for period_name in ['2014_2017', '2018_2022', '2023_2025', '2026', 'all']:
        period_df = detail if period_name == 'all' else detail[detail['aux_period'] == period_name]
        if period_df.empty:
            continue

        health = _market_health_stats(period_df.rename(columns={
            'market_yao_1': 'yao_1',
            'market_yao_2': 'yao_2',
            'market_yao_3': 'yao_3',
        }))
        if health:
            health_rows.append({'period': period_name, **health})

        for horizon in HORIZONS:
            col = f'ret_fwd_{horizon}d'
            valid = period_df[['market_gua_code', 'market_gua_name', col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna(subset=[col, 'market_gua_code'])
            if valid.empty:
                continue

            overall_mean = round(float(valid[col].mean()), 4)
            grouped = valid.groupby(['market_gua_code', 'market_gua_name'])[col]
            stats = grouped.agg(['count', 'mean', 'median', 'std']).reset_index()
            win_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').gt(0).mean()).reset_index(name='win_rate')
            pos_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').ge(2).mean()).reset_index(name='ge_2pct_rate')
            neg_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').le(-2).mean()).reset_index(name='le_minus_2pct_rate')
            t_stats = grouped.apply(_safe_t_stat).reset_index(name='t_stat')
            merged = stats.merge(win_rate, on=['market_gua_code', 'market_gua_name'], how='left')
            merged = merged.merge(pos_rate, on=['market_gua_code', 'market_gua_name'], how='left')
            merged = merged.merge(neg_rate, on=['market_gua_code', 'market_gua_name'], how='left')
            merged = merged.merge(t_stats, on=['market_gua_code', 'market_gua_name'], how='left')

            means = []
            for _, row in merged.iterrows():
                mean_value = round(float(row['mean']), 4) if pd.notna(row['mean']) else None
                means.append(mean_value if mean_value is not None else 0)
                summary_rows.append({
                    'period': period_name,
                    'horizon': horizon,
                    'market_gua_code': row['market_gua_code'],
                    'market_gua_name': row['market_gua_name'],
                    'count': int(row['count']),
                    'mean_ret': mean_value,
                    'median_ret': None if pd.isna(row['median']) else round(float(row['median']), 4),
                    'std_ret': None if pd.isna(row['std']) else round(float(row['std']), 4),
                    'win_rate': None if pd.isna(row['win_rate']) else round(float(row['win_rate']), 4),
                    'ge_2pct_rate': None if pd.isna(row['ge_2pct_rate']) else round(float(row['ge_2pct_rate']), 4),
                    'le_minus_2pct_rate': None if pd.isna(row['le_minus_2pct_rate']) else round(float(row['le_minus_2pct_rate']), 4),
                    't_stat': row['t_stat'],
                    'excess_vs_all': None if mean_value is None else round(mean_value - overall_mean, 4),
                })

            if means:
                effect_rows.append({
                    'period': period_name,
                    'horizon': horizon,
                    'overall_mean_ret': overall_mean,
                    'best_mean_ret': round(max(means), 4),
                    'worst_mean_ret': round(min(means), 4),
                    'spread_best_worst': round(max(means) - min(means), 4),
                    'bagua_count': int(merged['market_gua_code'].nunique()),
                    'sample_count': int(len(valid)),
                })

    summary = pd.DataFrame(summary_rows)
    effect = pd.DataFrame(effect_rows)
    health = pd.DataFrame(health_rows)
    if not summary.empty:
        summary = summary.sort_values(['period', 'horizon', 'mean_ret', 'market_gua_code'], ascending=[True, True, False, True]).reset_index(drop=True)
    if not effect.empty:
        effect = effect.sort_values(['period', 'horizon']).reset_index(drop=True)
    if not health.empty:
        health = health.sort_values(['period']).reset_index(drop=True)

    summary_path = foundation_file('bagua_regime_return_summary.csv')
    effect_path = foundation_file('bagua_regime_effect_tests.csv')
    health_path = foundation_file('bagua_regime_market_health.csv')
    yearly_health_path = foundation_file('bagua_regime_market_health_by_year.csv')
    rolling_health_path = foundation_file('bagua_regime_market_health_rolling.csv')
    rolling_extremes_path = foundation_file('bagua_regime_market_health_rolling_extremes.csv')
    transition_events_path = foundation_file('bagua_regime_transition_events.csv')
    yearly_effect_path = foundation_file('bagua_regime_effect_by_year.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    effect.to_csv(effect_path, index=False, encoding='utf-8-sig')
    health.to_csv(health_path, index=False, encoding='utf-8-sig')
    yearly_health.to_csv(yearly_health_path, index=False, encoding='utf-8-sig')
    rolling_health.to_csv(rolling_health_path, index=False, encoding='utf-8-sig')
    rolling_extremes.to_csv(rolling_extremes_path, index=False, encoding='utf-8-sig')
    transition_events.to_csv(transition_events_path, index=False, encoding='utf-8-sig')
    yearly_effect.to_csv(yearly_effect_path, index=False, encoding='utf-8-sig')

    report = {
        'files': {
            'market_bagua_daily.csv': os.path.exists(market_path),
            'daily_forward_returns.csv': os.path.exists(fwd_path),
            'bagua_regime_stock_returns_detail.csv': os.path.exists(detail_path),
            'bagua_regime_return_summary.csv': os.path.exists(summary_path),
            'bagua_regime_effect_tests.csv': os.path.exists(effect_path),
            'bagua_regime_market_health.csv': os.path.exists(health_path),
            'bagua_regime_market_health_by_year.csv': os.path.exists(yearly_health_path),
            'bagua_regime_market_health_rolling.csv': os.path.exists(rolling_health_path),
            'bagua_regime_market_health_rolling_extremes.csv': os.path.exists(rolling_extremes_path),
            'bagua_regime_transition_events.csv': os.path.exists(transition_events_path),
            'bagua_regime_effect_by_year.csv': os.path.exists(yearly_effect_path),
        },
        'market_gua_classes': sorted(detail['market_gua_code'].dropna().astype(str).str.zfill(3).unique().tolist()),
        'verification_focus': ['yearly', 'rolling', 'transition_events'],
        'aux_period_counts': detail.groupby('aux_period').size().to_dict(),
        'aux_period_market_health': health.set_index('period').to_dict(orient='index') if not health.empty else {},
        'yearly_market_health': yearly_health.set_index('year').to_dict(orient='index') if not yearly_health.empty else {},
        'yearly_effect_best_spread': yearly_effect.groupby('horizon')['spread_best_worst'].max().to_dict() if not yearly_effect.empty else {},
        'rolling_market_health_summary': {
            str(window): {
                'bagua_count_min': int(rolling_health.loc[rolling_health['window'] == window, 'bagua_count'].min()),
                'bagua_count_max': int(rolling_health.loc[rolling_health['window'] == window, 'bagua_count'].max()),
                'top1_ratio_max': round(float(rolling_health.loc[rolling_health['window'] == window, 'top1_ratio'].max()), 6),
                'top2_ratio_max': round(float(rolling_health.loc[rolling_health['window'] == window, 'top2_ratio'].max()), 6),
            }
            for window in ROLLING_WINDOWS if not rolling_health[rolling_health['window'] == window].empty
        },
        'rolling_health_extremes': rolling_extremes.to_dict(orient='records') if not rolling_extremes.empty else [],
        'transition_event_count': int(len(transition_events)),
        'transition_summary': transition_summary,
        'horizon_missing_rate': {
            f'ret_fwd_{h}d': round(float(pd.to_numeric(detail[f'ret_fwd_{h}d'], errors='coerce').isna().mean()), 6)
            for h in HORIZONS
        },
        'effect_spread_by_horizon': effect.groupby('horizon')['spread_best_worst'].max().to_dict() if not effect.empty else {},
    }
    report_path = foundation_file('bagua_regime_verify_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('=' * 80)
    print('八卦分治第一轮验证完成')
    print('=' * 80)
    print(f'detail: {detail_path}')
    print(f'summary: {summary_path}')
    print(f'effect: {effect_path}')
    print(f'health: {health_path}')
    print(f'yearly_health: {yearly_health_path}')
    print(f'rolling_health: {rolling_health_path}')
    print(f'rolling_extremes: {rolling_extremes_path}')
    print(f'transitions: {transition_events_path}')
    print(f'yearly_effect: {yearly_effect_path}')
    print(f'report: {report_path}')
    return detail, summary, effect, report


if __name__ == '__main__':
    verify_bagua_regime()
