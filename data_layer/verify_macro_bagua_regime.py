# -*- coding: utf-8 -*-
"""
verify_macro_bagua_regime.py

v1.0 大周期八卦第一轮验证：
1. 生成 macro_bagua 日表
2. 与个股未来收益标签结合，观察不同大周期卦下的中长期收益差异
3. 对比 macro 与 market 的切换频率、段长与卦分布稳定性
"""
import json
import math
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_forward_returns import build_daily_forward_returns
from data_layer.prepare_macro_bagua import build_macro_bagua
from data_layer.prepare_market_bagua import build_market_bagua


HORIZONS = [20, 60]
ROLLING_WINDOWS = [250, 500]
AUX_PERIODS = {
    '2014_2017': ('2014-01-01', '2017-12-31'),
    '2018_2022': ('2018-01-01', '2022-12-31'),
    '2023_2025': ('2023-01-01', '2025-12-31'),
    '2026': ('2026-01-01', '2026-12-31'),
    'all': ('2014-01-01', '2099-12-31'),
}

MACRO_GROUP_MAP = {
    '111': '进攻',
    '011': '进攻',
    '101': '进攻',
    '001': '观察',
    '010': '观察',
    '110': '观察',
    '100': '防守',
    '000': '防守',
}
GROUP_ORDER = ['进攻', '观察', '防守']
POSITION_SCHEMES = {
    'v1_base_100_50_0': {'进攻': 1.0, '观察': 0.5, '防守': 0.0},
    'v1_tight_100_30_0': {'进攻': 1.0, '观察': 0.3, '防守': 0.0},
    'v1_soft_80_30_0': {'进攻': 0.8, '观察': 0.3, '防守': 0.0},
}


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


def _market_health_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    gua_col = 'macro_gua_code' if 'macro_gua_code' in df.columns else 'gua_code'
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
    for col in ['yao_1', 'yao_2', 'yao_3', 'macro_yao_1', 'macro_yao_2', 'macro_yao_3']:
        if col in df.columns:
            target = col.replace('macro_', '')
            stats[f'{target}_ratio'] = round(float(pd.to_numeric(df[col], errors='coerce').mean()), 6)
    seg_id_col = 'macro_seg_id' if 'macro_seg_id' in df.columns else 'seg_id'
    seg_day_col = 'macro_seg_day' if 'macro_seg_day' in df.columns else 'seg_day'
    if {seg_id_col, seg_day_col}.issubset(df.columns):
        seg = df[['date', seg_id_col, seg_day_col]].drop_duplicates(subset=['date'])
        if not seg.empty:
            seg_len = seg.groupby(seg_id_col)[seg_day_col].max()
            stats['segment_length_mean'] = round(float(seg_len.mean()), 4)
            stats['segment_length_median'] = round(float(seg_len.median()), 4)
            stats['segment_length_p75'] = round(float(seg_len.quantile(0.75)), 4)
            stats['short_segment_ratio_le_2'] = round(float((seg_len <= 2).mean()), 6)
            stats['short_segment_ratio_le_5'] = round(float((seg_len <= 5).mean()), 6)
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
    events.loc[(events['market_speed_slow'] > 0) & (events['macro_breadth_slow'] > 0), 'move_type'] = '慢牛扩散'
    events.loc[(events['market_speed_slow'] > 0) & (events['macro_breadth_slow'] <= 0), 'move_type'] = '趋势修复'
    events.loc[(events['market_speed_slow'] <= 0) & (events['macro_breadth_slow'] > 0), 'move_type'] = '高位钝化或底部修复'
    events.loc[(events['market_speed_slow'] <= 0) & (events['macro_breadth_slow'] <= 0), 'move_type'] = '转弱或熊压制'
    cols = [
        'date', 'prev_gua', 'to_gua', 'gua_name', 'move_type', 'market_close_proxy',
        'market_trend_slow', 'market_trend_anchor_slow', 'market_speed_slow', 'macro_breadth_slow',
        'limit_heat', 'limit_quality', 'seg_id', 'seg_day'
    ]
    return events[cols].reset_index(drop=True)


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


def _build_yearly_effect(detail: pd.DataFrame):
    rows = []
    tmp = detail.copy()
    tmp['year'] = tmp['date'].astype(str).str[:4]
    for year, year_df in tmp.groupby('year', sort=True):
        for horizon in HORIZONS:
            col = f'ret_fwd_{horizon}d'
            valid = year_df[['macro_gua_code', col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna(subset=[col, 'macro_gua_code'])
            if valid.empty:
                continue
            grouped = valid.groupby('macro_gua_code')[col].mean()
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


def _build_compare_health(macro_market: pd.DataFrame, market: pd.DataFrame):
    macro_stats = _market_health_stats(macro_market)
    market_stats = _market_health_stats(market)
    if not macro_stats or not market_stats:
        return {}
    compare = {}
    for key in ['bagua_count', 'top1_ratio', 'top2_ratio', 'segment_length_mean', 'segment_length_median', 'short_segment_ratio_le_2', 'short_segment_ratio_le_5', 'change_rate']:
        compare[key] = {
            'macro': macro_stats.get(key),
            'market': market_stats.get(key),
        }
    return compare


def _build_macro_gua_profile(detail: pd.DataFrame, macro_market: pd.DataFrame):
    if detail is None or detail.empty:
        return pd.DataFrame()

    base = detail.copy()
    base['year'] = base['date'].astype(str).str[:4]
    base['macro_gua_code'] = base['macro_gua_code'].astype(str).str.zfill(3)
    base['macro_group'] = base['macro_gua_code'].map(MACRO_GROUP_MAP).fillna('未分组')

    market_base = macro_market.copy()
    market_base['gua_code'] = market_base['gua_code'].astype(str).str.zfill(3)
    seg_len = market_base.groupby('seg_id')['seg_day'].max().reset_index(name='segment_length')
    seg_len = market_base[['gua_code', 'seg_id']].drop_duplicates().merge(seg_len, on='seg_id', how='left')
    gua_seg = seg_len.groupby('gua_code')['segment_length']
    seg_stats = gua_seg.agg(['count', 'mean', 'median']).reset_index().rename(columns={
        'gua_code': 'macro_gua_code',
        'count': 'segment_count',
        'mean': 'segment_len_mean',
        'median': 'segment_len_median',
    })

    day_stats = market_base.groupby(['gua_code', 'gua_name']).agg(
        day_count=('date', 'count'),
        change_count=('changed', 'sum'),
        market_trend_slow_mean=('market_trend_slow', 'mean'),
        market_speed_slow_mean=('market_speed_slow', 'mean'),
        macro_breadth_slow_mean=('macro_breadth_slow', 'mean'),
        limit_heat_mean=('limit_heat', 'mean'),
        limit_quality_mean=('limit_quality', 'mean'),
        yao_1_ratio=('yao_1', 'mean'),
        yao_2_ratio=('yao_2', 'mean'),
        yao_3_ratio=('yao_3', 'mean'),
    ).reset_index().rename(columns={'gua_code': 'macro_gua_code', 'gua_name': 'macro_gua_name'})
    day_stats['change_rate'] = day_stats['change_count'] / day_stats['day_count']

    rows = []
    for gua_code, gua_df in base.groupby('macro_gua_code', sort=True):
        if not gua_code or gua_df.empty:
            continue
        row = {
            'macro_gua_code': gua_code,
            'macro_gua_name': gua_df['macro_gua_name'].dropna().iloc[0] if gua_df['macro_gua_name'].notna().any() else '',
            'macro_group': gua_df['macro_group'].dropna().iloc[0] if gua_df['macro_group'].notna().any() else '未分组',
            'sample_count_20d': int(pd.to_numeric(gua_df['ret_fwd_20d'], errors='coerce').notna().sum()) if 'ret_fwd_20d' in gua_df.columns else 0,
            'sample_count_60d': int(pd.to_numeric(gua_df['ret_fwd_60d'], errors='coerce').notna().sum()) if 'ret_fwd_60d' in gua_df.columns else 0,
            'year_count': int(gua_df.loc[gua_df['year'].notna(), 'year'].nunique()),
            'positive_year_ratio_20d': None,
            'positive_year_ratio_60d': None,
        }

        for horizon in HORIZONS:
            col = f'ret_fwd_{horizon}d'
            values = pd.to_numeric(gua_df[col], errors='coerce') if col in gua_df.columns else pd.Series(dtype=float)
            valid = values.dropna()
            row[f'avg_ret_{horizon}d'] = round(float(valid.mean()), 4) if len(valid) else None
            row[f'median_ret_{horizon}d'] = round(float(valid.median()), 4) if len(valid) else None
            row[f'win_rate_{horizon}d'] = round(float(valid.gt(0).mean()), 4) if len(valid) else None
            row[f'big_pos_ratio_{horizon}d'] = round(float(valid.ge(5).mean()), 4) if len(valid) else None
            row[f'big_loss_ratio_{horizon}d'] = round(float(valid.le(-5).mean()), 4) if len(valid) else None
            row[f'std_ret_{horizon}d'] = round(float(valid.std()), 4) if len(valid) >= 2 else None
            row[f't_stat_{horizon}d'] = _safe_t_stat(valid)

            yearly = gua_df[['year', col]].copy() if col in gua_df.columns else pd.DataFrame(columns=['year', col])
            yearly[col] = pd.to_numeric(yearly[col], errors='coerce')
            yearly = yearly.dropna(subset=['year', col])
            if not yearly.empty:
                yearly_mean = yearly.groupby('year')[col].mean()
                row[f'positive_year_ratio_{horizon}d'] = round(float(yearly_mean.gt(0).mean()), 4)
                row[f'yearly_ret_std_{horizon}d'] = round(float(yearly_mean.std()), 4) if len(yearly_mean) >= 2 else None
            else:
                row[f'yearly_ret_std_{horizon}d'] = None

        rows.append(row)

    profile = pd.DataFrame(rows)
    if profile.empty:
        return profile

    profile = profile.merge(day_stats, on=['macro_gua_code', 'macro_gua_name'], how='left')
    profile = profile.merge(seg_stats, on='macro_gua_code', how='left')
    profile['sample_count'] = profile[['sample_count_20d', 'sample_count_60d']].max(axis=1)
    profile['signal_to_trade_ratio'] = 1.0
    profile = profile.sort_values(['macro_group', 'avg_ret_60d', 'avg_ret_20d', 'macro_gua_code'], ascending=[True, False, False, True]).reset_index(drop=True)
    return profile


def _build_macro_group_summary(profile: pd.DataFrame):
    if profile is None or profile.empty:
        return pd.DataFrame()

    rows = []
    for group_name in GROUP_ORDER:
        grp = profile[profile['macro_group'] == group_name].copy()
        if grp.empty:
            continue
        row = {
            'macro_group': group_name,
            'gua_count': int(grp['macro_gua_code'].nunique()),
            'guas': ' / '.join(grp['macro_gua_code'].tolist()),
            'gua_names': ' / '.join(grp['macro_gua_name'].tolist()),
            'sample_count_20d': int(pd.to_numeric(grp['sample_count_20d'], errors='coerce').sum()),
            'sample_count_60d': int(pd.to_numeric(grp['sample_count_60d'], errors='coerce').sum()),
            'day_count': int(pd.to_numeric(grp['day_count'], errors='coerce').sum()),
            'segment_count': int(pd.to_numeric(grp['segment_count'], errors='coerce').sum()),
        }
        for col in [
            'avg_ret_20d', 'median_ret_20d', 'win_rate_20d', 'big_pos_ratio_20d', 'big_loss_ratio_20d', 'positive_year_ratio_20d', 'yearly_ret_std_20d',
            'avg_ret_60d', 'median_ret_60d', 'win_rate_60d', 'big_pos_ratio_60d', 'big_loss_ratio_60d', 'positive_year_ratio_60d', 'yearly_ret_std_60d',
            'segment_len_mean', 'segment_len_median', 'change_rate', 'limit_heat_mean', 'limit_quality_mean',
            'market_speed_slow_mean', 'macro_breadth_slow_mean', 'yao_1_ratio', 'yao_2_ratio', 'yao_3_ratio'
        ]:
            series = pd.to_numeric(grp[col], errors='coerce').dropna() if col in grp.columns else pd.Series(dtype=float)
            row[col] = round(float(series.mean()), 4) if len(series) else None
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out['macro_group'] = pd.Categorical(out['macro_group'], categories=GROUP_ORDER, ordered=True)
    out = out.sort_values('macro_group').reset_index(drop=True)
    return out


def _build_macro_group_decision(profile: pd.DataFrame):
    if profile is None or profile.empty:
        return pd.DataFrame()

    decide = profile.copy()
    decide['suggested_group'] = '观察'
    decide.loc[(pd.to_numeric(decide['avg_ret_20d'], errors='coerce') > 0) & (pd.to_numeric(decide['avg_ret_60d'], errors='coerce') > 0), 'suggested_group'] = '进攻'
    decide.loc[(pd.to_numeric(decide['avg_ret_20d'], errors='coerce') <= 0) & (pd.to_numeric(decide['avg_ret_60d'], errors='coerce') <= 0), 'suggested_group'] = '防守'
    decide['group_changed'] = decide['macro_group'] != decide['suggested_group']
    decide['decision_reason'] = '20/60日前瞻收益同正=进攻，同负=防守，其余=观察'
    cols = [
        'macro_gua_code', 'macro_gua_name', 'macro_group', 'suggested_group', 'group_changed', 'decision_reason',
        'avg_ret_20d', 'avg_ret_60d', 'win_rate_20d', 'win_rate_60d',
        'positive_year_ratio_20d', 'positive_year_ratio_60d',
        'segment_len_mean', 'change_rate', 'limit_heat_mean', 'market_speed_slow_mean', 'macro_breadth_slow_mean'
    ]
    return decide[cols].sort_values(['suggested_group', 'avg_ret_60d', 'avg_ret_20d', 'macro_gua_code'], ascending=[True, False, False, True]).reset_index(drop=True)


def _build_position_scheme_comparison(profile: pd.DataFrame, decision: pd.DataFrame):
    if profile is None or profile.empty or decision is None or decision.empty:
        return pd.DataFrame()

    merged = profile.merge(decision[['macro_gua_code', 'suggested_group']], on='macro_gua_code', how='left')
    rows = []
    total_all_day_count = float(pd.to_numeric(merged['day_count'], errors='coerce').fillna(0).sum())
    for scheme_name, weight_map in POSITION_SCHEMES.items():
        row = {'scheme': scheme_name}
        total_weighted_day = 0.0
        total_day = 0.0
        for horizon in HORIZONS:
            numerator = 0.0
            denominator = 0.0
            for _, item in merged.iterrows():
                group_name = item['suggested_group']
                weight = float(weight_map.get(group_name, 0.0))
                sample_count = pd.to_numeric(item.get(f'sample_count_{horizon}d'), errors='coerce')
                avg_ret = pd.to_numeric(item.get(f'avg_ret_{horizon}d'), errors='coerce')
                if pd.notna(sample_count) and pd.notna(avg_ret) and sample_count > 0:
                    numerator += weight * float(sample_count) * float(avg_ret)
                    denominator += float(sample_count)
            row[f'weighted_ret_{horizon}d'] = round(numerator / denominator, 4) if denominator > 0 else None

        for group_name in GROUP_ORDER:
            weight = float(weight_map.get(group_name, 0.0))
            grp = merged[merged['suggested_group'] == group_name]
            day_count = pd.to_numeric(grp['day_count'], errors='coerce').fillna(0).sum()
            row[f'{group_name}_weight'] = weight
            row[f'{group_name}_day_ratio'] = round(float(day_count) / total_all_day_count, 4) if total_all_day_count > 0 else None
            total_weighted_day += weight * float(day_count)
            total_day += float(day_count)
        row['avg_position'] = round(total_weighted_day / total_day, 4) if total_day > 0 else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values('weighted_ret_60d', ascending=False).reset_index(drop=True)


def _build_v1_final_group_table(profile: pd.DataFrame, decision: pd.DataFrame):
    if profile is None or profile.empty or decision is None or decision.empty:
        return pd.DataFrame()

    merged = decision.merge(
        profile[[
            'macro_gua_code', 'macro_gua_name', 'avg_ret_20d', 'avg_ret_60d',
            'win_rate_20d', 'win_rate_60d', 'positive_year_ratio_20d', 'positive_year_ratio_60d',
            'segment_len_mean', 'change_rate', 'market_speed_slow_mean', 'macro_breadth_slow_mean'
        ]],
        on=['macro_gua_code', 'macro_gua_name', 'avg_ret_20d', 'avg_ret_60d', 'win_rate_20d', 'win_rate_60d',
           'positive_year_ratio_20d', 'positive_year_ratio_60d', 'segment_len_mean', 'change_rate',
           'market_speed_slow_mean', 'macro_breadth_slow_mean'],
        how='left'
    )
    merged = merged.copy()
    merged['final_group'] = merged['suggested_group']
    merged['final_reason'] = merged['decision_reason']
    order_map = {'进攻': 0, '观察': 1, '防守': 2}
    merged['group_order'] = merged['final_group'].map(order_map).fillna(9)
    cols = [
        'macro_gua_code', 'macro_gua_name', 'final_group', 'final_reason', 'group_changed',
        'avg_ret_20d', 'avg_ret_60d', 'win_rate_20d', 'win_rate_60d',
        'positive_year_ratio_20d', 'positive_year_ratio_60d',
        'segment_len_mean', 'change_rate', 'market_speed_slow_mean', 'macro_breadth_slow_mean'
    ]
    return merged.sort_values(['group_order', 'avg_ret_60d', 'avg_ret_20d', 'macro_gua_code'], ascending=[True, False, False, True])[cols].reset_index(drop=True)


def _build_v1_position_decision_table(final_group_table: pd.DataFrame, scheme_name: str = 'v1_tight_100_30_0'):
    if final_group_table is None or final_group_table.empty:
        return pd.DataFrame()
    weight_map = POSITION_SCHEMES.get(scheme_name, {})
    out = final_group_table.copy()
    out['scheme'] = scheme_name
    out['target_position'] = out['final_group'].map(weight_map)
    out['action'] = out['target_position'].map({1.0: '正常进攻', 0.3: '轻仓观察', 0.0: '空仓防守'})
    cols = [
        'scheme', 'macro_gua_code', 'macro_gua_name', 'final_group', 'target_position', 'action',
        'final_reason', 'avg_ret_20d', 'avg_ret_60d', 'win_rate_20d', 'win_rate_60d',
        'positive_year_ratio_20d', 'positive_year_ratio_60d', 'segment_len_mean', 'change_rate'
    ]
    return out[cols].reset_index(drop=True)


def verify_macro_bagua_regime():
    macro_path = foundation_file('macro_bagua_daily.csv')
    market_path = foundation_file('market_bagua_daily.csv')
    fwd_path = foundation_file('daily_forward_returns.csv')

    if not os.path.exists(macro_path):
        build_macro_bagua()
    if not os.path.exists(market_path):
        build_market_bagua()
    if not os.path.exists(fwd_path):
        build_daily_forward_returns()

    macro_market = pd.read_csv(macro_path, encoding='utf-8-sig', dtype={'gua_code': str}, low_memory=False)
    market = pd.read_csv(market_path, encoding='utf-8-sig', dtype={'gua_code': str}, low_memory=False)
    fwd = pd.read_csv(
        fwd_path,
        encoding='utf-8-sig',
        dtype={'code': str, 'avail_date_20d': str, 'avail_date_60d': str},
        low_memory=False,
    )

    macro_market['date'] = macro_market['date'].astype(str)
    market['date'] = market['date'].astype(str)
    fwd['date'] = fwd['date'].astype(str)
    macro_market['gua_code'] = macro_market['gua_code'].astype(str).str.zfill(3)
    market['gua_code'] = market['gua_code'].astype(str).str.zfill(3)

    detail = fwd.merge(
        macro_market[['date', 'gua_code', 'gua_name', 'yao_1', 'yao_2', 'yao_3', 'seg_id', 'seg_day']],
        on='date',
        how='left',
        suffixes=('', '_macro'),
    )
    detail = detail.rename(columns={
        'gua_code': 'macro_gua_code',
        'gua_name': 'macro_gua_name',
        'yao_1': 'macro_yao_1',
        'yao_2': 'macro_yao_2',
        'yao_3': 'macro_yao_3',
        'seg_id': 'macro_seg_id',
        'seg_day': 'macro_seg_day',
    })
    detail = detail.merge(
        market[['date', 'gua_code', 'gua_name']].rename(columns={'gua_code': 'market_gua_code', 'gua_name': 'market_gua_name'}),
        on='date',
        how='left',
    )
    detail['aux_period'] = detail['date'].map(_aux_period_name)
    detail = detail[detail['aux_period'] != 'other'].copy()

    detail_path = foundation_file('macro_bagua_regime_stock_returns_detail.csv')
    detail.to_csv(detail_path, index=False, encoding='utf-8-sig')

    yearly_health = _build_yearly_health(macro_market)
    rolling_health = _build_rolling_health(macro_market, ROLLING_WINDOWS)
    transition_events = _build_transition_events(macro_market)
    transition_summary = _build_transition_summary(transition_events)
    yearly_effect = _build_yearly_effect(detail)
    compare_health = _build_compare_health(macro_market, market)

    summary_rows = []
    effect_rows = []
    health_rows = []

    for period_name in ['2014_2017', '2018_2022', '2023_2025', '2026', 'all']:
        period_df = detail if period_name == 'all' else detail[detail['aux_period'] == period_name]
        if period_df.empty:
            continue

        health = _market_health_stats(period_df.rename(columns={
            'macro_yao_1': 'yao_1',
            'macro_yao_2': 'yao_2',
            'macro_yao_3': 'yao_3',
        }))
        if health:
            health_rows.append({'period': period_name, **health})

        for horizon in HORIZONS:
            col = f'ret_fwd_{horizon}d'
            valid = period_df[['macro_gua_code', 'macro_gua_name', col]].copy()
            valid[col] = pd.to_numeric(valid[col], errors='coerce')
            valid = valid.dropna(subset=[col, 'macro_gua_code'])
            if valid.empty:
                continue

            overall_mean = round(float(valid[col].mean()), 4)
            grouped = valid.groupby(['macro_gua_code', 'macro_gua_name'])[col]
            stats = grouped.agg(['count', 'mean', 'median', 'std']).reset_index()
            win_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').gt(0).mean()).reset_index(name='win_rate')
            pos_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').ge(5).mean()).reset_index(name='ge_5pct_rate')
            neg_rate = grouped.apply(lambda x: pd.to_numeric(x, errors='coerce').le(-5).mean()).reset_index(name='le_minus_5pct_rate')
            t_stats = grouped.apply(_safe_t_stat).reset_index(name='t_stat')
            merged = stats.merge(win_rate, on=['macro_gua_code', 'macro_gua_name'], how='left')
            merged = merged.merge(pos_rate, on=['macro_gua_code', 'macro_gua_name'], how='left')
            merged = merged.merge(neg_rate, on=['macro_gua_code', 'macro_gua_name'], how='left')
            merged = merged.merge(t_stats, on=['macro_gua_code', 'macro_gua_name'], how='left')

            means = []
            for _, row in merged.iterrows():
                mean_value = round(float(row['mean']), 4) if pd.notna(row['mean']) else None
                means.append(mean_value if mean_value is not None else 0)
                summary_rows.append({
                    'period': period_name,
                    'horizon': horizon,
                    'macro_gua_code': row['macro_gua_code'],
                    'macro_gua_name': row['macro_gua_name'],
                    'count': int(row['count']),
                    'mean_ret': mean_value,
                    'median_ret': None if pd.isna(row['median']) else round(float(row['median']), 4),
                    'std_ret': None if pd.isna(row['std']) else round(float(row['std']), 4),
                    'win_rate': None if pd.isna(row['win_rate']) else round(float(row['win_rate']), 4),
                    'ge_5pct_rate': None if pd.isna(row['ge_5pct_rate']) else round(float(row['ge_5pct_rate']), 4),
                    'le_minus_5pct_rate': None if pd.isna(row['le_minus_5pct_rate']) else round(float(row['le_minus_5pct_rate']), 4),
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
                    'bagua_count': int(merged['macro_gua_code'].nunique()),
                    'sample_count': int(len(valid)),
                })

    summary = pd.DataFrame(summary_rows)
    effect = pd.DataFrame(effect_rows)
    health = pd.DataFrame(health_rows)
    macro_profile = _build_macro_gua_profile(detail, macro_market)
    macro_group_summary = _build_macro_group_summary(macro_profile)
    macro_group_decision = _build_macro_group_decision(macro_profile)
    position_scheme_comparison = _build_position_scheme_comparison(macro_profile, macro_group_decision)
    v1_final_group_table = _build_v1_final_group_table(macro_profile, macro_group_decision)
    v1_position_decision_table = _build_v1_position_decision_table(v1_final_group_table, scheme_name='v1_tight_100_30_0')
    if not summary.empty:
        summary = summary.sort_values(['period', 'horizon', 'mean_ret', 'macro_gua_code'], ascending=[True, True, False, True]).reset_index(drop=True)
    if not effect.empty:
        effect = effect.sort_values(['period', 'horizon']).reset_index(drop=True)
    if not health.empty:
        health = health.sort_values(['period']).reset_index(drop=True)

    summary_path = foundation_file('macro_bagua_regime_return_summary.csv')
    effect_path = foundation_file('macro_bagua_regime_effect_tests.csv')
    health_path = foundation_file('macro_bagua_regime_market_health.csv')
    yearly_health_path = foundation_file('macro_bagua_regime_market_health_by_year.csv')
    rolling_health_path = foundation_file('macro_bagua_regime_market_health_rolling.csv')
    transition_events_path = foundation_file('macro_bagua_regime_transition_events.csv')
    yearly_effect_path = foundation_file('macro_bagua_regime_effect_by_year.csv')
    macro_profile_path = foundation_file('macro_bagua_profile.csv')
    macro_group_summary_path = foundation_file('macro_bagua_group_summary.csv')
    macro_group_decision_path = foundation_file('macro_bagua_group_decision.csv')
    position_scheme_comparison_path = foundation_file('macro_bagua_position_scheme_comparison.csv')
    v1_final_group_table_path = foundation_file('macro_bagua_v1_final_group_table.csv')
    v1_position_decision_table_path = foundation_file('macro_bagua_v1_position_decision_table.csv')
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    effect.to_csv(effect_path, index=False, encoding='utf-8-sig')
    health.to_csv(health_path, index=False, encoding='utf-8-sig')
    yearly_health.to_csv(yearly_health_path, index=False, encoding='utf-8-sig')
    rolling_health.to_csv(rolling_health_path, index=False, encoding='utf-8-sig')
    transition_events.to_csv(transition_events_path, index=False, encoding='utf-8-sig')
    yearly_effect.to_csv(yearly_effect_path, index=False, encoding='utf-8-sig')
    macro_profile.to_csv(macro_profile_path, index=False, encoding='utf-8-sig')
    macro_group_summary.to_csv(macro_group_summary_path, index=False, encoding='utf-8-sig')
    macro_group_decision.to_csv(macro_group_decision_path, index=False, encoding='utf-8-sig')
    position_scheme_comparison.to_csv(position_scheme_comparison_path, index=False, encoding='utf-8-sig')
    v1_final_group_table.to_csv(v1_final_group_table_path, index=False, encoding='utf-8-sig')
    v1_position_decision_table.to_csv(v1_position_decision_table_path, index=False, encoding='utf-8-sig')

    report = {
        'files': {
            'macro_bagua_daily.csv': os.path.exists(macro_path),
            'market_bagua_daily.csv': os.path.exists(market_path),
            'daily_forward_returns.csv': os.path.exists(fwd_path),
            'macro_bagua_regime_stock_returns_detail.csv': os.path.exists(detail_path),
            'macro_bagua_regime_return_summary.csv': os.path.exists(summary_path),
            'macro_bagua_regime_effect_tests.csv': os.path.exists(effect_path),
            'macro_bagua_regime_market_health.csv': os.path.exists(health_path),
            'macro_bagua_regime_market_health_by_year.csv': os.path.exists(yearly_health_path),
            'macro_bagua_regime_market_health_rolling.csv': os.path.exists(rolling_health_path),
            'macro_bagua_regime_transition_events.csv': os.path.exists(transition_events_path),
            'macro_bagua_regime_effect_by_year.csv': os.path.exists(yearly_effect_path),
            'macro_bagua_profile.csv': os.path.exists(macro_profile_path),
            'macro_bagua_group_summary.csv': os.path.exists(macro_group_summary_path),
            'macro_bagua_group_decision.csv': os.path.exists(macro_group_decision_path),
            'macro_bagua_position_scheme_comparison.csv': os.path.exists(position_scheme_comparison_path),
            'macro_bagua_v1_final_group_table.csv': os.path.exists(v1_final_group_table_path),
            'macro_bagua_v1_position_decision_table.csv': os.path.exists(v1_position_decision_table_path),
        },
        'macro_gua_classes': sorted(detail['macro_gua_code'].dropna().astype(str).str.zfill(3).unique().tolist()),
        'macro_group_map': MACRO_GROUP_MAP,
        'position_schemes': POSITION_SCHEMES,
        'verification_focus': ['mid_long_horizon', 'segment_length', 'market_vs_macro'],
        'aux_period_counts': detail.groupby('aux_period').size().to_dict(),
        'aux_period_macro_health': health.set_index('period').to_dict(orient='index') if not health.empty else {},
        'yearly_macro_health': yearly_health.set_index('year').to_dict(orient='index') if not yearly_health.empty else {},
        'yearly_effect_best_spread': yearly_effect.groupby('horizon')['spread_best_worst'].max().to_dict() if not yearly_effect.empty else {},
        'rolling_macro_health_summary': {
            str(window): {
                'bagua_count_min': int(rolling_health.loc[rolling_health['window'] == window, 'bagua_count'].min()),
                'bagua_count_max': int(rolling_health.loc[rolling_health['window'] == window, 'bagua_count'].max()),
                'top1_ratio_max': round(float(rolling_health.loc[rolling_health['window'] == window, 'top1_ratio'].max()), 6),
                'top2_ratio_max': round(float(rolling_health.loc[rolling_health['window'] == window, 'top2_ratio'].max()), 6),
            }
            for window in ROLLING_WINDOWS if not rolling_health[rolling_health['window'] == window].empty
        },
        'transition_event_count': int(len(transition_events)),
        'transition_summary': transition_summary,
        'horizon_missing_rate': {
            f'ret_fwd_{h}d': round(float(pd.to_numeric(detail[f'ret_fwd_{h}d'], errors='coerce').isna().mean()), 6)
            for h in HORIZONS if f'ret_fwd_{h}d' in detail.columns
        },
        'effect_spread_by_horizon': effect.groupby('horizon')['spread_best_worst'].max().to_dict() if not effect.empty else {},
        'market_vs_macro_health_compare': compare_health,
        'macro_profile_preview': macro_profile.head(8).to_dict(orient='records') if not macro_profile.empty else [],
        'macro_group_summary_preview': macro_group_summary.to_dict(orient='records') if not macro_group_summary.empty else [],
        'macro_group_decision_preview': macro_group_decision.to_dict(orient='records') if not macro_group_decision.empty else [],
        'position_scheme_comparison_preview': position_scheme_comparison.to_dict(orient='records') if not position_scheme_comparison.empty else [],
        'v1_final_group_table_preview': v1_final_group_table.to_dict(orient='records') if not v1_final_group_table.empty else [],
        'v1_position_decision_table_preview': v1_position_decision_table.to_dict(orient='records') if not v1_position_decision_table.empty else [],
    }
    report_path = foundation_file('macro_bagua_regime_verify_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('=' * 80)
    print('大周期八卦第一轮验证完成')
    print('=' * 80)
    print(f'detail: {detail_path}')
    print(f'summary: {summary_path}')
    print(f'effect: {effect_path}')
    print(f'health: {health_path}')
    print(f'yearly_health: {yearly_health_path}')
    print(f'rolling_health: {rolling_health_path}')
    print(f'transitions: {transition_events_path}')
    print(f'yearly_effect: {yearly_effect_path}')
    print(f'macro_profile: {macro_profile_path}')
    print(f'macro_group_summary: {macro_group_summary_path}')
    print(f'macro_group_decision: {macro_group_decision_path}')
    print(f'position_scheme_comparison: {position_scheme_comparison_path}')
    print(f'v1_final_group_table: {v1_final_group_table_path}')
    print(f'v1_position_decision_table: {v1_position_decision_table_path}')
    print(f'report: {report_path}')
    return detail, summary, effect, report


if __name__ == '__main__':
    verify_macro_bagua_regime()
