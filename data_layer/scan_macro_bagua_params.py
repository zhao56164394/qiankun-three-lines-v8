# -*- coding: utf-8 -*-
"""
scan_macro_bagua_params.py

扫描 macro_bagua 慢周期参数组合，优先筛出更像“大周期状态机”的候选，
再结合 20d / 60d 未来收益区分度做综合排序。
"""
import gc
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_forward_returns import build_daily_forward_returns
from data_layer.prepare_macro_bagua import (
    DEFAULT_MACRO_PARAMS,
    build_macro_bagua,
    get_macro_params,
    iter_macro_param_grid,
)
from data_layer.prepare_market_bagua import build_market_bagua
from data_layer.verify_macro_bagua_regime import (
    HORIZONS,
    ROLLING_WINDOWS,
    _build_compare_health,
    _build_rolling_health,
    _build_yearly_effect,
    _market_health_stats,
)


SCAN_GRID = {
    'trend_period': [120, 144, 180],
    'speed_lookback': [40, 60, 80],
    'breadth_fast_window': [15, 20, 30],
    'breadth_slow_window': [40, 60, 90],
    'breadth_weight_ma60': [0.55, 0.65, 0.75],
    'breakout_weight': [0.3, 0.6, 0.9],
}
TOP_N = 20
DETAIL_TOP_N = 5
CHECKPOINT_EVERY = 10


def _flush_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)


def _safe_float(value, digits=6):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _safe_int(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _config_to_label(params: dict) -> str:
    return (
        f"tp{int(params['trend_period'])}"
        f"_sp{int(params['speed_lookback'])}"
        f"_bf{int(params['breadth_fast_window'])}"
        f"_bs{int(params['breadth_slow_window'])}"
        f"_ma60{int(round(float(params['breadth_weight_ma60']) * 100))}"
        f"_bo{int(round(float(params['breakout_weight']) * 100))}"
    )


def _load_market_and_forward_data():
    market_path = foundation_file('market_bagua_daily.csv')
    fwd_path = foundation_file('daily_forward_returns.csv')

    if not os.path.exists(market_path):
        build_market_bagua()
    if not os.path.exists(fwd_path):
        build_daily_forward_returns()

    market = pd.read_csv(market_path, encoding='utf-8-sig', dtype={'gua_code': str}, low_memory=False)
    fwd = pd.read_csv(
        fwd_path,
        encoding='utf-8-sig',
        dtype={'code': str, 'avail_date_20d': str, 'avail_date_60d': str},
        low_memory=False,
    )
    market['date'] = market['date'].astype(str)
    market['gua_code'] = market['gua_code'].astype(str).str.zfill(3)
    fwd['date'] = fwd['date'].astype(str)
    return market, fwd


def _build_detail(macro_market: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    detail = fwd.merge(
        macro_market[['date', 'gua_code', 'gua_name', 'yao_1', 'yao_2', 'yao_3', 'seg_id', 'seg_day']],
        on='date',
        how='left',
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
    return detail


def _extract_effect_metrics(detail: pd.DataFrame):
    rows = {}
    for horizon in HORIZONS:
        col = f'ret_fwd_{horizon}d'
        valid = detail[['macro_gua_code', 'macro_gua_name', col]].copy()
        valid[col] = pd.to_numeric(valid[col], errors='coerce')
        valid = valid.dropna(subset=['macro_gua_code', col])
        if valid.empty:
            rows[horizon] = {}
            continue
        grouped = valid.groupby(['macro_gua_code', 'macro_gua_name'])[col].agg(['count', 'mean']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False).reset_index(drop=True)
        best = grouped.iloc[0]
        worst = grouped.iloc[-1]
        rows[horizon] = {
            'overall_mean_ret': _safe_float(valid[col].mean(), 4),
            'best_gua_code': str(best['macro_gua_code']).zfill(3),
            'best_gua_name': best['macro_gua_name'],
            'best_mean_ret': _safe_float(best['mean'], 4),
            'best_count': _safe_int(best['count']),
            'worst_gua_code': str(worst['macro_gua_code']).zfill(3),
            'worst_gua_name': worst['macro_gua_name'],
            'worst_mean_ret': _safe_float(worst['mean'], 4),
            'worst_count': _safe_int(worst['count']),
            'spread_best_worst': _safe_float(best['mean'] - worst['mean'], 4),
            'bagua_count': int(grouped['macro_gua_code'].nunique()),
            'sample_count': int(len(valid)),
        }
    return rows


def _extract_yearly_effect_metrics(detail: pd.DataFrame):
    yearly = _build_yearly_effect(detail)
    metrics = {}
    for horizon in HORIZONS:
        sub = yearly[yearly['horizon'] == horizon].copy()
        if sub.empty:
            metrics[horizon] = {}
            continue
        spread = pd.to_numeric(sub['spread_best_worst'], errors='coerce').dropna()
        best_ret = pd.to_numeric(sub['best_mean_ret'], errors='coerce').dropna()
        worst_ret = pd.to_numeric(sub['worst_mean_ret'], errors='coerce').dropna()
        metrics[horizon] = {
            'year_count': int(sub['year'].nunique()),
            'spread_min': _safe_float(spread.min(), 4) if not spread.empty else None,
            'spread_median': _safe_float(spread.median(), 4) if not spread.empty else None,
            'spread_max': _safe_float(spread.max(), 4) if not spread.empty else None,
            'best_mean_ret_median': _safe_float(best_ret.median(), 4) if not best_ret.empty else None,
            'worst_mean_ret_median': _safe_float(worst_ret.median(), 4) if not worst_ret.empty else None,
        }
    return yearly, metrics


def _extract_rolling_metrics(macro_market: pd.DataFrame):
    rolling = _build_rolling_health(macro_market, ROLLING_WINDOWS)
    metrics = {}
    for window in ROLLING_WINDOWS:
        sub = rolling[rolling['window'] == window].copy()
        if sub.empty:
            metrics[window] = {}
            continue
        metrics[window] = {
            'bagua_count_min': _safe_int(pd.to_numeric(sub['bagua_count'], errors='coerce').min()),
            'bagua_count_max': _safe_int(pd.to_numeric(sub['bagua_count'], errors='coerce').max()),
            'top1_ratio_max': _safe_float(pd.to_numeric(sub['top1_ratio'], errors='coerce').max(), 6),
            'top2_ratio_max': _safe_float(pd.to_numeric(sub['top2_ratio'], errors='coerce').max(), 6),
            'change_rate_max': _safe_float(pd.to_numeric(sub['change_rate'], errors='coerce').max(), 6),
            'segment_length_median_min': _safe_float(pd.to_numeric(sub['segment_length_median'], errors='coerce').min(), 4),
        }
    return rolling, metrics


def _hard_filter_row(row: dict, market_baseline: dict, macro_baseline: dict):
    reasons = []
    if (row.get('bagua_count') or 0) < 6:
        reasons.append('bagua_count<6')
    if row.get('change_rate') is None or row['change_rate'] >= (market_baseline.get('change_rate') or 1):
        reasons.append('change_rate不优于market')
    if row.get('segment_length_median') is None or row['segment_length_median'] <= (market_baseline.get('segment_length_median') or 0):
        reasons.append('segment_median不优于market')

    short2_limit = macro_baseline.get('short_segment_ratio_le_2')
    if short2_limit is not None:
        short2_limit = min(float(short2_limit) * 1.15, 0.35)
        if row.get('short_segment_ratio_le_2') is None or row['short_segment_ratio_le_2'] > short2_limit:
            reasons.append('短段占比过高')

    top1_limit = macro_baseline.get('top1_ratio')
    if top1_limit is not None:
        top1_limit = max(float(top1_limit) * 1.05, 0.42)
        if row.get('top1_ratio') is None or row['top1_ratio'] > top1_limit:
            reasons.append('top1占比过高')

    top2_limit = macro_baseline.get('top2_ratio')
    if top2_limit is not None:
        top2_limit = max(float(top2_limit) * 1.05, 0.72)
        if row.get('top2_ratio') is None or row['top2_ratio'] > top2_limit:
            reasons.append('top2占比过高')

    return len(reasons) == 0, '|'.join(reasons)


def _apply_rank_score(summary_df: pd.DataFrame):
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.copy()
    candidate_mask = summary_df['is_candidate'].fillna(0).astype(int).eq(1)
    rank_df = summary_df.loc[candidate_mask].copy()
    if rank_df.empty:
        summary_df['rank_score'] = None
        summary_df['candidate_rank'] = None
        summary_df['overall_rank'] = summary_df.index + 1
        return summary_df

    metric_specs = [
        ('spread_60d_all', False, 0.30),
        ('spread_20d_all', False, 0.20),
        ('yearly_spread_60d_min', False, 0.15),
        ('yearly_spread_20d_min', False, 0.10),
        ('segment_length_median', False, 0.10),
        ('change_rate', True, 0.08),
        ('short_segment_ratio_le_2', True, 0.05),
        ('rolling_top1_ratio_max', True, 0.02),
    ]

    n = len(rank_df)
    total = pd.Series(0.0, index=rank_df.index)
    for col, lower_better, weight in metric_specs:
        values = pd.to_numeric(rank_df[col], errors='coerce')
        if values.notna().sum() == 0:
            continue
        ranks = values.rank(method='average', ascending=lower_better, na_option='bottom')
        if n == 1:
            normalized = pd.Series(1.0, index=rank_df.index)
        else:
            normalized = 1 - (ranks - 1) / (n - 1)
        total = total + normalized * weight

    summary_df['rank_score'] = None
    summary_df.loc[rank_df.index, 'rank_score'] = (total * 100).round(4)
    summary_df['candidate_rank'] = None
    candidate_order = summary_df.loc[rank_df.index].sort_values(
        ['rank_score', 'spread_60d_all', 'spread_20d_all'],
        ascending=[False, False, False],
    ).index.tolist()
    for rank, idx in enumerate(candidate_order, start=1):
        summary_df.at[idx, 'candidate_rank'] = rank

    summary_df = summary_df.sort_values(
        ['is_candidate', 'rank_score', 'spread_60d_all', 'spread_20d_all', 'segment_length_median'],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    summary_df['overall_rank'] = summary_df.index + 1
    return summary_df


def _build_top_details_from_candidates(candidate_details, top_labels):
    yearly_detail_rows = []
    rolling_detail_rows = []
    compare_detail_rows = []
    for item in candidate_details:
        if item['config_label'] not in top_labels:
            continue
        yearly_effect = item['yearly_effect'].copy()
        if not yearly_effect.empty:
            yearly_effect.insert(0, 'config_label', item['config_label'])
            yearly_detail_rows.append(yearly_effect)
        rolling_health = item['rolling_health'].copy()
        if not rolling_health.empty:
            rolling_health.insert(0, 'config_label', item['config_label'])
            rolling_detail_rows.append(rolling_health)
        compare = item['compare_health'] or {}
        if compare:
            compare_row = {'config_label': item['config_label']}
            for key, payload in compare.items():
                compare_row[f'{key}_macro'] = payload.get('macro')
                compare_row[f'{key}_market'] = payload.get('market')
            compare_detail_rows.append(compare_row)

    yearly_detail_df = pd.concat(yearly_detail_rows, ignore_index=True) if yearly_detail_rows else pd.DataFrame()
    rolling_detail_df = pd.concat(rolling_detail_rows, ignore_index=True) if rolling_detail_rows else pd.DataFrame()
    compare_detail_df = pd.DataFrame(compare_detail_rows)
    return yearly_detail_df, rolling_detail_df, compare_detail_df


def _write_outputs(summary_df: pd.DataFrame, candidate_details, report: dict, top_n=TOP_N, detail_top_n=DETAIL_TOP_N):
    summary_df = _apply_rank_score(summary_df)
    top_df = summary_df[summary_df['is_candidate'] == 1].copy()
    if top_df.empty:
        top_df = summary_df.head(top_n).copy()
    else:
        top_df = top_df.head(top_n).copy()

    detail_labels = set(top_df.head(detail_top_n)['config_label'].tolist())
    yearly_detail_df, rolling_detail_df, compare_detail_df = _build_top_details_from_candidates(candidate_details, detail_labels)

    summary_path = foundation_file('macro_bagua_param_scan_summary.csv')
    top_path = foundation_file('macro_bagua_param_scan_top.csv')
    yearly_path = foundation_file('macro_bagua_param_scan_yearly.csv')
    rolling_path = foundation_file('macro_bagua_param_scan_rolling.csv')
    compare_path = foundation_file('macro_bagua_param_scan_compare.csv')
    report_path = foundation_file('macro_bagua_param_scan_report.json')

    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    top_df.to_csv(top_path, index=False, encoding='utf-8-sig')
    yearly_detail_df.to_csv(yearly_path, index=False, encoding='utf-8-sig')
    rolling_detail_df.to_csv(rolling_path, index=False, encoding='utf-8-sig')
    compare_detail_df.to_csv(compare_path, index=False, encoding='utf-8-sig')

    report = dict(report)
    report['best_config'] = top_df.iloc[0].to_dict() if not top_df.empty else summary_df.iloc[0].to_dict()
    report['top_config_labels'] = top_df['config_label'].tolist()
    report['detail_config_labels'] = list(detail_labels)
    report['files'] = {
        'macro_bagua_param_scan_summary.csv': os.path.exists(summary_path),
        'macro_bagua_param_scan_top.csv': os.path.exists(top_path),
        'macro_bagua_param_scan_yearly.csv': os.path.exists(yearly_path),
        'macro_bagua_param_scan_rolling.csv': os.path.exists(rolling_path),
        'macro_bagua_param_scan_compare.csv': os.path.exists(compare_path),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return summary_df, top_df, report


def evaluate_single_config(params: dict, market: pd.DataFrame, fwd: pd.DataFrame, market_baseline: dict, macro_baseline: dict):
    final_params = get_macro_params(params)
    final_params['breadth_weight_ma20'] = round(1 - float(final_params['breadth_weight_ma60']), 6)

    macro_market = build_macro_bagua(params=final_params, write_output=False)
    if macro_market.empty:
        raise ValueError('macro_bagua 结果为空')

    detail = _build_detail(macro_market, fwd)
    health = _market_health_stats(macro_market) or {}
    effect_metrics = _extract_effect_metrics(detail)
    yearly_effect, yearly_metrics = _extract_yearly_effect_metrics(detail)
    rolling_health, rolling_metrics = _extract_rolling_metrics(macro_market)
    compare_health = _build_compare_health(macro_market, market)

    row = {
        'config_label': _config_to_label(final_params),
        'trend_period': int(final_params['trend_period']),
        'yao1_window': int(final_params['yao1_window']),
        'yao1_min_periods': int(final_params['yao1_min_periods']),
        'yao1_q_low': float(final_params['yao1_q_low']),
        'yao1_q_high': float(final_params['yao1_q_high']),
        'speed_lookback': int(final_params['speed_lookback']),
        'breadth_fast_window': int(final_params['breadth_fast_window']),
        'breadth_slow_window': int(final_params['breadth_slow_window']),
        'breadth_weight_ma60': float(final_params['breadth_weight_ma60']),
        'breadth_weight_ma20': float(final_params['breadth_weight_ma20']),
        'breakout_weight': float(final_params['breakout_weight']),
        'heat_weight': float(final_params['heat_weight']),
        'sample_count': _safe_int(health.get('sample_count')),
        'bagua_count': _safe_int(health.get('bagua_count')),
        'top1_ratio': _safe_float(health.get('top1_ratio'), 6),
        'top2_ratio': _safe_float(health.get('top2_ratio'), 6),
        'yao_1_ratio': _safe_float(health.get('yao_1_ratio'), 6),
        'yao_2_ratio': _safe_float(health.get('yao_2_ratio'), 6),
        'yao_3_ratio': _safe_float(health.get('yao_3_ratio'), 6),
        'segment_length_mean': _safe_float(health.get('segment_length_mean'), 4),
        'segment_length_median': _safe_float(health.get('segment_length_median'), 4),
        'segment_length_p75': _safe_float(health.get('segment_length_p75'), 4),
        'short_segment_ratio_le_2': _safe_float(health.get('short_segment_ratio_le_2'), 6),
        'short_segment_ratio_le_5': _safe_float(health.get('short_segment_ratio_le_5'), 6),
        'change_rate': _safe_float(health.get('change_rate'), 6),
        'spread_20d_all': _safe_float(effect_metrics.get(20, {}).get('spread_best_worst'), 4),
        'best_mean_ret_20d_all': _safe_float(effect_metrics.get(20, {}).get('best_mean_ret'), 4),
        'worst_mean_ret_20d_all': _safe_float(effect_metrics.get(20, {}).get('worst_mean_ret'), 4),
        'spread_60d_all': _safe_float(effect_metrics.get(60, {}).get('spread_best_worst'), 4),
        'best_mean_ret_60d_all': _safe_float(effect_metrics.get(60, {}).get('best_mean_ret'), 4),
        'worst_mean_ret_60d_all': _safe_float(effect_metrics.get(60, {}).get('worst_mean_ret'), 4),
        'yearly_spread_20d_min': _safe_float(yearly_metrics.get(20, {}).get('spread_min'), 4),
        'yearly_spread_20d_median': _safe_float(yearly_metrics.get(20, {}).get('spread_median'), 4),
        'yearly_spread_20d_max': _safe_float(yearly_metrics.get(20, {}).get('spread_max'), 4),
        'yearly_spread_60d_min': _safe_float(yearly_metrics.get(60, {}).get('spread_min'), 4),
        'yearly_spread_60d_median': _safe_float(yearly_metrics.get(60, {}).get('spread_median'), 4),
        'yearly_spread_60d_max': _safe_float(yearly_metrics.get(60, {}).get('spread_max'), 4),
        'rolling_top1_ratio_max': _safe_float(rolling_metrics.get(250, {}).get('top1_ratio_max'), 6),
        'rolling_top2_ratio_max': _safe_float(rolling_metrics.get(250, {}).get('top2_ratio_max'), 6),
        'rolling_bagua_count_min': _safe_int(rolling_metrics.get(250, {}).get('bagua_count_min')),
        'rolling_change_rate_max': _safe_float(rolling_metrics.get(250, {}).get('change_rate_max'), 6),
        'rolling500_top1_ratio_max': _safe_float(rolling_metrics.get(500, {}).get('top1_ratio_max'), 6),
        'rolling500_top2_ratio_max': _safe_float(rolling_metrics.get(500, {}).get('top2_ratio_max'), 6),
        'vs_market_change_rate_delta': None,
        'vs_market_segment_median_delta': None,
        'vs_market_short2_delta': None,
        'vs_default_macro_change_rate_delta': None,
        'vs_default_macro_segment_median_delta': None,
        'vs_default_macro_short2_delta': None,
    }

    if row['change_rate'] is not None and market_baseline.get('change_rate') is not None:
        row['vs_market_change_rate_delta'] = round(row['change_rate'] - float(market_baseline['change_rate']), 6)
    if row['segment_length_median'] is not None and market_baseline.get('segment_length_median') is not None:
        row['vs_market_segment_median_delta'] = round(row['segment_length_median'] - float(market_baseline['segment_length_median']), 4)
    if row['short_segment_ratio_le_2'] is not None and market_baseline.get('short_segment_ratio_le_2') is not None:
        row['vs_market_short2_delta'] = round(row['short_segment_ratio_le_2'] - float(market_baseline['short_segment_ratio_le_2']), 6)

    if row['change_rate'] is not None and macro_baseline.get('change_rate') is not None:
        row['vs_default_macro_change_rate_delta'] = round(row['change_rate'] - float(macro_baseline['change_rate']), 6)
    if row['segment_length_median'] is not None and macro_baseline.get('segment_length_median') is not None:
        row['vs_default_macro_segment_median_delta'] = round(row['segment_length_median'] - float(macro_baseline['segment_length_median']), 4)
    if row['short_segment_ratio_le_2'] is not None and macro_baseline.get('short_segment_ratio_le_2') is not None:
        row['vs_default_macro_short2_delta'] = round(row['short_segment_ratio_le_2'] - float(macro_baseline['short_segment_ratio_le_2']), 6)

    is_candidate, filter_reason = _hard_filter_row(row, market_baseline, macro_baseline)
    row['is_candidate'] = int(is_candidate)
    row['filter_reason'] = filter_reason
    return row, macro_market, yearly_effect, rolling_health, compare_health


def scan_macro_params(grid=None, top_n=TOP_N, detail_top_n=DETAIL_TOP_N, max_configs=None):
    grid = grid or SCAN_GRID
    market, fwd = _load_market_and_forward_data()
    market_baseline = _market_health_stats(market) or {}
    default_macro = build_macro_bagua(params=DEFAULT_MACRO_PARAMS, write_output=False)
    macro_baseline = _market_health_stats(default_macro) or {}

    summary_rows = []
    candidate_details = []
    total_configs = 1
    for values in grid.values():
        total_configs *= len(values)
    effective_total = min(total_configs, max_configs) if max_configs is not None else total_configs

    base_report = {
        'scan_grid': grid,
        'default_macro_params': DEFAULT_MACRO_PARAMS,
        'market_baseline': market_baseline,
        'default_macro_baseline': macro_baseline,
        'selection_logic': {
            'hard_filters': [
                'bagua_count >= 6',
                'change_rate < current_market_change_rate',
                'segment_length_median > current_market_segment_median',
                'short_segment_ratio_le_2 <= default_macro_short2 * 1.15 且不高于 0.35',
                'top1_ratio / top2_ratio 不明显塌缩',
            ],
            'rank_metrics': [
                '60d spread_best_worst 更高',
                '20d spread_best_worst 更高',
                '年度最差 spread 更高',
                'segment_length_median 更高',
                'change_rate 更低',
                'short_segment_ratio_le_2 更低',
                'rolling_top1_ratio_max 更低',
            ],
        },
    }

    for idx, params in enumerate(iter_macro_param_grid(grid), start=1):
        if max_configs is not None and idx > max_configs:
            break
        final_params = get_macro_params(params)
        final_params['breadth_weight_ma20'] = round(1 - float(final_params['breadth_weight_ma60']), 6)
        label = _config_to_label(final_params)
        _flush_print(f'[{idx}/{effective_total}] 扫描 {label}')
        row, macro_market, yearly_effect, rolling_health, compare_health = evaluate_single_config(
            final_params,
            market,
            fwd,
            market_baseline,
            macro_baseline,
        )
        summary_rows.append(row)

        if row['is_candidate'] == 1:
            candidate_details.append({
                'config_label': row['config_label'],
                'yearly_effect': yearly_effect.copy(),
                'rolling_health': rolling_health.copy(),
                'compare_health': compare_health,
            })
            ranked_preview = _apply_rank_score(pd.DataFrame(summary_rows))
            current_top = ranked_preview[ranked_preview['is_candidate'] == 1].head(detail_top_n)
            keep_labels = set(current_top['config_label'].tolist())
            candidate_details = [item for item in candidate_details if item['config_label'] in keep_labels]

        del macro_market, yearly_effect, rolling_health, compare_health
        gc.collect()

        should_checkpoint = idx == effective_total or idx % CHECKPOINT_EVERY == 0
        if should_checkpoint:
            summary_df = pd.DataFrame(summary_rows)
            checkpoint_report = dict(base_report)
            checkpoint_report['evaluated_config_count'] = int(len(summary_df))
            checkpoint_report['candidate_count'] = int((summary_df['is_candidate'] == 1).sum())
            checkpoint_report['status'] = 'running' if idx < effective_total else 'completed'
            checkpoint_report['progress'] = {
                'completed': int(idx),
                'total': int(effective_total),
                'ratio': round(idx / effective_total, 6) if effective_total else 1.0,
            }
            summary_df, top_df, checkpoint_report = _write_outputs(
                summary_df,
                candidate_details,
                checkpoint_report,
                top_n=top_n,
                detail_top_n=detail_top_n,
            )
            top_count = int((summary_df['is_candidate'] == 1).sum())
            if not top_df.empty:
                head = top_df.iloc[0]
                _flush_print(
                    f'checkpoint {idx}/{effective_total} | 候选 {top_count} | '
                    f'当前第一 {head["config_label"]} | 60d spread {head["spread_60d_all"]} | '
                    f'change_rate {head["change_rate"]}'
                )
            else:
                _flush_print(f'checkpoint {idx}/{effective_total} | 暂无候选')

    summary_df = pd.DataFrame(summary_rows)
    final_report = dict(base_report)
    final_report['evaluated_config_count'] = int(len(summary_df))
    final_report['candidate_count'] = int((summary_df['is_candidate'] == 1).sum())
    final_report['status'] = 'completed'
    final_report['progress'] = {
        'completed': int(len(summary_df)),
        'total': int(effective_total),
        'ratio': 1.0,
    }
    summary_df, top_df, report = _write_outputs(
        summary_df,
        candidate_details,
        final_report,
        top_n=top_n,
        detail_top_n=detail_top_n,
    )

    summary_path = foundation_file('macro_bagua_param_scan_summary.csv')
    top_path = foundation_file('macro_bagua_param_scan_top.csv')
    yearly_path = foundation_file('macro_bagua_param_scan_yearly.csv')
    rolling_path = foundation_file('macro_bagua_param_scan_rolling.csv')
    compare_path = foundation_file('macro_bagua_param_scan_compare.csv')
    report_path = foundation_file('macro_bagua_param_scan_report.json')

    _flush_print('=' * 100)
    _flush_print('macro_bagua 参数扫描完成')
    _flush_print('=' * 100)
    _flush_print(f'总组合数: {len(summary_df)}')
    _flush_print(f'候选数: {(summary_df["is_candidate"] == 1).sum()}')
    _flush_print(f'summary: {summary_path}')
    _flush_print(f'top: {top_path}')
    _flush_print(f'yearly: {yearly_path}')
    _flush_print(f'rolling: {rolling_path}')
    _flush_print(f'compare: {compare_path}')
    _flush_print(f'report: {report_path}')
    if not top_df.empty:
        _flush_print('-' * 100)
        _flush_print('Top 10 候选:')
        show_cols = [
            'overall_rank', 'candidate_rank', 'config_label',
            'spread_60d_all', 'spread_20d_all',
            'segment_length_median', 'change_rate', 'short_segment_ratio_le_2',
            'rolling_top1_ratio_max', 'rank_score',
        ]
        _flush_print(top_df[show_cols].head(10).to_string(index=False))

    return summary_df, top_df, report


def main():
    max_configs_env = os.getenv('MACRO_SCAN_MAX_CONFIGS')
    max_configs = int(max_configs_env) if max_configs_env else None
    top_n = int(os.getenv('MACRO_SCAN_TOP_N', str(TOP_N)))
    detail_top_n = int(os.getenv('MACRO_SCAN_DETAIL_TOP_N', str(DETAIL_TOP_N)))
    scan_macro_params(top_n=top_n, detail_top_n=detail_top_n, max_configs=max_configs)


if __name__ == '__main__':
    main()
