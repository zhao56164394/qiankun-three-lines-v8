# -*- coding: utf-8 -*-
"""
verify_foundation.py

新底座第一轮验证脚本：
- 文件存在性
- 行数与日期
- 主键唯一性
- 关键字段缺失率
- 八卦分布
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file, path_exists_map


VERIFY_FILES = [
    'main_board_universe.csv',
    'daily_cross_section.csv',
    'daily_5d_scores.csv',
    'daily_3yao.csv',
    'daily_bagua_sequence.csv',
    'market_bagua_daily.csv',
    'macro_bagua_daily.csv',
    'daily_forward_returns.csv',
]
MARKET_PERIODS = {
    '2014_2017': ('2014-01-01', '2017-12-31'),
    '2018_2022': ('2018-01-01', '2022-12-31'),
    '2023_2025': ('2023-01-01', '2025-12-31'),
    '2026': ('2026-01-01', '2026-12-31'),
}


MARKET_STRUCTURE_FIELDS = [
    'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
    'new_high_20_ratio', 'new_low_20_ratio',
]


def _calc_market_structure_feature_checks(cross: pd.DataFrame):
    if cross is None or cross.empty:
        return None

    checks = {
        'missing_rate': {},
        'distribution': {},
    }

    for col in MARKET_STRUCTURE_FIELDS:
        if col not in cross.columns:
            continue
        s = pd.to_numeric(cross[col], errors='coerce')
        checks['missing_rate'][col] = round(float(s.isna().mean()), 6)
        valid = s.dropna()
        if valid.empty:
            checks['distribution'][col] = {
                'min': None,
                'p25': None,
                'median': None,
                'p75': None,
                'max': None,
                'mean': None,
                'out_of_range_ratio': None,
            }
            continue
        checks['distribution'][col] = {
            'min': round(float(valid.min()), 6),
            'p25': round(float(valid.quantile(0.25)), 6),
            'median': round(float(valid.median()), 6),
            'p75': round(float(valid.quantile(0.75)), 6),
            'max': round(float(valid.max()), 6),
            'mean': round(float(valid.mean()), 6),
            'out_of_range_ratio': round(float(((valid < 0) | (valid > 1)).mean()), 6),
        }

    available = [c for c in MARKET_STRUCTURE_FIELDS if c in cross.columns]
    if available:
        tmp = cross[['date'] + available].copy()
        tmp['year'] = tmp['date'].astype(str).str[:4]
        yearly = {}
        for year, year_df in tmp.groupby('year', sort=True):
            yearly[year] = {}
            for col in available:
                s = pd.to_numeric(year_df[col], errors='coerce').dropna()
                yearly[year][col] = None if s.empty else round(float(s.mean()), 6)
        checks['yearly_mean'] = yearly

    mono_cols = ['above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio']
    if all(col in cross.columns for col in mono_cols):
        tmp = cross[mono_cols].apply(pd.to_numeric, errors='coerce')
        valid = tmp.dropna()
        if not valid.empty:
            checks['ma_monotonicity_violation_ratio'] = round(float((
                (valid['above_ma5_ratio'] < valid['above_ma10_ratio']) |
                (valid['above_ma10_ratio'] < valid['above_ma20_ratio']) |
                (valid['above_ma20_ratio'] < valid['above_ma60_ratio'])
            ).mean()), 6)

    if all(col in cross.columns for col in ['new_high_20_ratio', 'new_low_20_ratio']):
        tmp = cross[['new_high_20_ratio', 'new_low_20_ratio']].apply(pd.to_numeric, errors='coerce').dropna()
        if not tmp.empty:
            checks['breakout_extremes'] = {
                'both_near_zero_ratio': round(float(((tmp['new_high_20_ratio'] <= 0.01) & (tmp['new_low_20_ratio'] <= 0.01)).mean()), 6),
                'both_high_ratio': round(float(((tmp['new_high_20_ratio'] >= 0.05) & (tmp['new_low_20_ratio'] >= 0.05)).mean()), 6),
            }

    return checks



def _calc_market_structure_stats(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    gua = df['gua_code'].astype(str).str.zfill(3)
    vc = gua.value_counts(normalize=True)
    top1_ratio = round(float(vc.iloc[0]), 6) if len(vc) >= 1 else None
    top2_ratio = round(float(vc.iloc[:2].sum()), 6) if len(vc) >= 2 else top1_ratio
    stats = {
        'market_gua_classes': sorted(gua.unique().tolist()),
        'market_gua_class_count': int(gua.nunique()),
        'market_top1_ratio': top1_ratio,
        'market_top2_ratio': top2_ratio,
    }
    for col in ['yao_1', 'yao_2', 'yao_3']:
        if col in df.columns:
            stats[f'{col}_ratio'] = round(float(pd.to_numeric(df[col], errors='coerce').mean()), 6)
    if {'seg_id', 'seg_day', 'changed'}.issubset(df.columns):
        seg_len = df.groupby('seg_id')['seg_day'].max()
        stats['market_segment_length'] = {
            'min': int(seg_len.min()),
            'max': int(seg_len.max()),
            'mean': round(float(seg_len.mean()), 4),
            'median': round(float(seg_len.median()), 4),
        }
        stats['market_bagua_change_rate'] = round(float(pd.to_numeric(df['changed'], errors='coerce').fillna(0).mean()), 6)
    return stats


def _file_stats(path):
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    if 'code' in df.columns:
        df['code'] = df['code'].astype(str).str.zfill(6)
    if 'gua_code' in df.columns:
        df['gua_code'] = df['gua_code'].astype(str).str.zfill(3)
    stats = {
        'path': path,
        'exists': True,
        'rows': int(len(df)),
        'columns': list(df.columns),
    }
    if 'date' in df.columns and len(df) > 0:
        stats['min_date'] = str(df['date'].astype(str).min())
        stats['max_date'] = str(df['date'].astype(str).max())
    if {'date', 'code'}.issubset(df.columns):
        stats['duplicate_date_code'] = int(df.duplicated(['date', 'code']).sum())
    return df, stats


def verify_foundation():
    report = {
        'source_paths': path_exists_map(),
        'files': {},
        'checks': {},
    }

    loaded = {}
    for filename in VERIFY_FILES:
        path = foundation_file(filename)
        if not os.path.exists(path):
            report['files'][filename] = {'exists': False}
            continue
        df, stats = _file_stats(path)
        loaded[filename] = df
        report['files'][filename] = stats

    cross = loaded.get('daily_cross_section.csv')
    score = loaded.get('daily_5d_scores.csv')
    bagua = loaded.get('daily_bagua_sequence.csv')
    market_bagua = loaded.get('market_bagua_daily.csv')
    forward_returns = loaded.get('daily_forward_returns.csv')
    universe = loaded.get('main_board_universe.csv')

    if universe is not None and len(universe) > 0:
        report['checks']['universe_count'] = int(universe['in_universe'].fillna(0).sum())
        report['checks']['st_in_universe'] = int(((universe['in_universe'] == 1) & (universe['is_st'].astype(str).isin(['True', '1', 'true']))).sum())
        daily_counts = universe.groupby('date')['in_universe'].sum()
        report['checks']['universe_daily_count'] = {
            'min': int(daily_counts.min()),
            'max': int(daily_counts.max()),
            'mean': round(float(daily_counts.mean()), 2),
        }

    if cross is not None and len(cross) > 0:
        key_fields = ['close', 'turnover_rate', 'cost_50', 'winner_ratio', 'small_net', 'allA_close']
        miss = {}
        for col in key_fields:
            if col in cross.columns:
                miss[col] = round(float(cross[col].isna().mean()), 6)
        report['checks']['cross_section_missing_rate'] = miss

        market_structure_checks = _calc_market_structure_feature_checks(cross)
        if market_structure_checks:
            report['checks']['market_structure_feature_checks'] = market_structure_checks

    if score is not None and len(score) > 0:
        dist = {}
        for col in ['score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi']:
            if col in score.columns:
                s = pd.to_numeric(score[col], errors='coerce')
                dist[col] = {
                    'min': None if s.dropna().empty else round(float(s.min()), 4),
                    'max': None if s.dropna().empty else round(float(s.max()), 4),
                    'mean': None if s.dropna().empty else round(float(s.mean()), 4),
                }
        report['checks']['score_distribution'] = dist

    if bagua is not None and len(bagua) > 0:
        bagua['gua_code'] = bagua['gua_code'].astype(str).str.zfill(3)
        vc = bagua['gua_code'].astype(str).value_counts().to_dict()
        report['checks']['bagua_distribution'] = {k: int(v) for k, v in vc.items()}
        report['checks']['bagua_classes'] = sorted(list(vc.keys()))
        if {'code', 'seg_id', 'seg_day', 'changed'}.issubset(bagua.columns):
            seg_len = bagua.groupby(['code', 'seg_id'])['seg_day'].max()
            report['checks']['segment_length'] = {
                'min': int(seg_len.min()),
                'max': int(seg_len.max()),
                'mean': round(float(seg_len.mean()), 4),
                'median': round(float(seg_len.median()), 4),
            }
            change_rate = pd.to_numeric(bagua['changed'], errors='coerce').fillna(0)
            report['checks']['bagua_change_rate'] = round(float(change_rate.mean()), 6)

    if market_bagua is not None and len(market_bagua) > 0:
        market_bagua['date'] = market_bagua['date'].astype(str)
        market_bagua['gua_code'] = market_bagua['gua_code'].astype(str).str.zfill(3)
        market_vc = market_bagua['gua_code'].value_counts().to_dict()
        report['checks']['market_bagua_distribution'] = {k: int(v) for k, v in market_vc.items()}

        overall_market_stats = _calc_market_structure_stats(market_bagua)
        if overall_market_stats:
            report['checks'].update(overall_market_stats)

        period_stats = {}
        for period_name, (start, end) in MARKET_PERIODS.items():
            period_df = market_bagua[(market_bagua['date'] >= start) & (market_bagua['date'] <= end)].copy()
            stats = _calc_market_structure_stats(period_df)
            if stats:
                period_stats[period_name] = stats
        report['checks']['market_period_structure'] = period_stats

    if forward_returns is not None and len(forward_returns) > 0:
        fwd_miss = {}
        for col in ['ret_fwd_1d', 'ret_fwd_3d', 'ret_fwd_5d', 'ret_fwd_10d', 'ret_fwd_20d']:
            if col in forward_returns.columns:
                fwd_miss[col] = round(float(pd.to_numeric(forward_returns[col], errors='coerce').isna().mean()), 6)
        report['checks']['forward_returns_missing_rate'] = fwd_miss

    report_path = foundation_file('verify_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('=' * 80)
    print('foundation 验证完成')
    print('=' * 80)
    for filename, stats in report['files'].items():
        print(f'{filename}: {"存在" if stats.get("exists") else "缺失"}')
        if stats.get('exists'):
            print(f'  rows={stats.get("rows", 0)} range={stats.get("min_date", "-")}~{stats.get("max_date", "-")}')
            if 'duplicate_date_code' in stats:
                print(f'  duplicate(date,code)={stats["duplicate_date_code"]}')
    print(f'验证报告: {report_path}')
    return report


if __name__ == '__main__':
    verify_foundation()
