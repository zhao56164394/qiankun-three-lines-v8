# -*- coding: utf-8 -*-
"""
prepare_market_bagua.py

基于新底座日度横截面，生成市场级八卦序列。
市场卦直接由市场时序状态编码，不再由个股五维分数均值直接二值化。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bagua_engine import encode_market_state_dynamic
from data_layer.foundation_config import foundation_file
from data_layer.prepare_daily_bagua import GUA_NAME_MAP
from strategy.indicator import calc_trend_line


OUTPUT_COLUMNS = [
    'date', 'stock_count',
    'market_open_proxy', 'market_high_proxy', 'market_low_proxy', 'market_close_proxy',
    'score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi',
    'up_ratio', 'zt_ratio', 'dt_ratio', 'turnover_median',
    'zt_count', 'dt_count', 'zb_count', 'limit_heat', 'limit_quality', 'ladder_heat',
    'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
    'new_high_20_ratio', 'new_low_20_ratio',
    'ma_breadth_score', 'breakout_score', 'enhanced_breadth_momo',
    'market_trend_55', 'market_trend_anchor_120', 'market_speed_10', 'market_speed_20', 'market_speed_30', 'market_speed_combo',
    'up_ratio_ma5', 'up_ratio_ma20', 'zt_ratio_ma5', 'zt_ratio_ma20',
    'turnover_median_ma5', 'turnover_median_ma20', 'breadth_momo',
    'yao_1', 'yao_2', 'yao_3', 'gua_code', 'gua_name',
    'prev_gua', 'changed', 'seg_id', 'seg_day',
]


SCORE_COLUMNS = ['score_wei', 'score_shi', 'score_bian', 'score_zhong', 'score_qi']
INDEX_CLOSE_COLUMNS = ['allA_close', 'hs300_close', 'csi500_close', 'csi1000_close', 'sh_close', 'sz_close']


def _build_index_anchor(cross: pd.DataFrame) -> pd.DataFrame:
    index_base = cross[['date'] + INDEX_CLOSE_COLUMNS].drop_duplicates('date').sort_values('date').reset_index(drop=True)
    for col in INDEX_CLOSE_COLUMNS:
        index_base[col] = pd.to_numeric(index_base[col], errors='coerce')
        first_valid = index_base[col].dropna()
        if first_valid.empty:
            index_base[col] = pd.NA
            continue
        anchor0 = float(first_valid.iloc[0])
        index_base[col] = index_base[col] / anchor0 * 100.0
    index_base['market_index_anchor'] = index_base[INDEX_CLOSE_COLUMNS].mean(axis=1)
    return index_base[['date', 'market_index_anchor']]


def _calc_dynamic_yao1(trend: pd.Series, ma_breadth_score: pd.Series | None = None) -> pd.Series:
    q40 = trend.rolling(120, min_periods=60).quantile(0.4).shift(1)
    q60 = trend.rolling(120, min_periods=60).quantile(0.6).shift(1)
    anchor = trend.rolling(120, min_periods=60).median().shift(1)
    ma_confirm = None
    if ma_breadth_score is not None:
        ma_confirm = pd.to_numeric(ma_breadth_score, errors='coerce')
    values = []
    prev = None
    for idx, (t, low, high, mid) in enumerate(zip(trend.tolist(), q40.tolist(), q60.tolist(), anchor.tolist())):
        if pd.isna(t) or pd.isna(low) or pd.isna(high) or pd.isna(mid):
            values.append(pd.NA)
            continue
        confirm_val = None if ma_confirm is None else ma_confirm.iloc[idx]
        if t >= high:
            prev = 1
        elif t <= low:
            prev = 0
        elif prev is None:
            prev = 1 if t >= mid else 0
        else:
            near_mid = abs(t - mid) <= 2.0
            if near_mid and not pd.isna(confirm_val):
                if prev == 0 and confirm_val >= 0.6:
                    prev = 1
                elif prev == 1 and confirm_val <= 0.4:
                    prev = 0
        values.append(prev)
    return pd.Series(values, index=trend.index), anchor



def _mark_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('date').copy()
    df['prev_gua'] = df['gua_code'].shift(1).fillna('')
    df['changed'] = (df['gua_code'] != df['prev_gua']).astype(int)
    df.loc[df['prev_gua'] == '', 'changed'] = 1
    df['seg_id'] = df['changed'].cumsum().astype(int)
    df['seg_day'] = df.groupby('seg_id').cumcount() + 1
    return df


def _calc_market_features(market: pd.DataFrame) -> pd.DataFrame:
    market = market.sort_values('date').reset_index(drop=True).copy()
    for col in [
        'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
        'new_high_20_ratio', 'new_low_20_ratio',
        'zt_count', 'dt_count', 'zb_count', 'ladder_heat',
    ]:
        if col in market.columns:
            market[col] = pd.to_numeric(market[col], errors='coerce')
        else:
            market[col] = pd.NA

    stock_count = pd.to_numeric(market['stock_count'], errors='coerce')
    safe_stock_count = stock_count.where(stock_count > 0)
    market['limit_heat'] = market['zt_count'] / safe_stock_count
    market['limit_quality'] = (market['zt_count'] - market['zb_count']) / safe_stock_count

    market['ma_breadth_score'] = (
        0.1 * market['above_ma5_ratio']
        + 0.2 * market['above_ma10_ratio']
        + 0.3 * market['above_ma20_ratio']
        + 0.4 * market['above_ma60_ratio']
    )
    market['breakout_score'] = market['new_high_20_ratio'] - market['new_low_20_ratio']

    market['market_trend_55'] = calc_trend_line(
        market['market_close_proxy'].values,
        market['market_high_proxy'].values,
        market['market_low_proxy'].values,
        period=55,
    )
    trend_series = pd.Series(market['market_trend_55'])
    market['yao_1'], market['market_trend_anchor_120'] = _calc_dynamic_yao1(trend_series, market['ma_breadth_score'])
    market['market_speed_10'] = market['market_trend_55'] - market['market_trend_55'].shift(10)
    market['market_speed_20'] = market['market_trend_55'] - market['market_trend_55'].shift(20)
    market['market_speed_30'] = market['market_trend_55'] - market['market_trend_55'].shift(30)
    market['market_speed_combo'] = 0.5 * market['market_speed_10'] + 0.3 * market['market_speed_20'] + 0.2 * market['market_speed_30']
    market['up_ratio_ma5'] = market['up_ratio'].rolling(5, min_periods=5).mean()
    market['up_ratio_ma20'] = market['up_ratio'].rolling(20, min_periods=20).mean()
    market['zt_ratio_ma5'] = market['zt_ratio'].rolling(5, min_periods=5).mean()
    market['zt_ratio_ma20'] = market['zt_ratio'].rolling(20, min_periods=20).mean()
    market['turnover_median_ma5'] = market['turnover_median'].rolling(5, min_periods=5).mean()
    market['turnover_median_ma20'] = market['turnover_median'].rolling(20, min_periods=20).mean()
    market['breadth_momo'] = (
        (market['up_ratio_ma5'] - market['up_ratio_ma20'])
        + 2.0 * (market['zt_ratio_ma5'] - market['zt_ratio_ma20'])
        + 0.2 * ((market['turnover_median_ma5'] / market['turnover_median_ma20']) - 1.0)
    )
    market['enhanced_breadth_momo'] = market['breadth_momo']
    near_zero = market['breadth_momo'].abs() <= 0.03
    market.loc[near_zero, 'enhanced_breadth_momo'] = (
        market.loc[near_zero, 'breadth_momo']
        + 0.2 * market.loc[near_zero, 'breakout_score']
        + 0.12 * market.loc[near_zero, 'limit_heat'].fillna(0.0)
        + 0.22 * market.loc[near_zero, 'limit_quality'].fillna(0.0)
        + 0.10 * market.loc[near_zero, 'ladder_heat'].fillna(0.0)
    )

    encoded = market.apply(
        lambda row: encode_market_state_dynamic(
            row['market_trend_55'],
            row['market_speed_20'],
            row['enhanced_breadth_momo'],
            row['market_trend_anchor_120'],
        ),
        axis=1,
        result_type='expand',
    )
    encoded.columns = ['yao_1_tmp', 'yao_2', 'yao_3', 'gua_code']
    market[['yao_2', 'yao_3', 'gua_code']] = encoded[['yao_2', 'yao_3', 'gua_code']]
    market = market.dropna(subset=['gua_code', 'yao_1']).copy()
    market['yao_1'] = pd.to_numeric(market['yao_1'], errors='coerce').astype(int)
    market['yao_2'] = market['yao_2'].astype(int)
    market['yao_3'] = market['yao_3'].astype(int)
    market['gua_code'] = market.apply(lambda row: f"{int(row['yao_1'])}{int(row['yao_2'])}{int(row['yao_3'])}", axis=1)
    market['gua_name'] = market['gua_code'].map(GUA_NAME_MAP)
    if 'yao_1_tmp' in market.columns:
        market = market.drop(columns=['yao_1_tmp'])
    return market


def build_market_bagua():
    cross_path = foundation_file('daily_cross_section.csv')
    score_path = foundation_file('daily_5d_scores.csv')
    if not os.path.exists(cross_path):
        raise FileNotFoundError(f'daily_cross_section.csv 不存在: {cross_path}')
    if not os.path.exists(score_path):
        raise FileNotFoundError(f'daily_5d_scores.csv 不存在: {score_path}')

    cross = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    scores = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str}, low_memory=False)
    index_anchor = _build_index_anchor(cross)

    for col in ['zt_count', 'dt_count', 'zb_count', 'lb_count']:
        if col not in cross.columns:
            cross[col] = pd.NA

    base_cols = [
        'date', 'code', 'open', 'high', 'low', 'close', 'turnover_rate', 'is_zt', 'is_dt',
        'zt_count', 'dt_count', 'zb_count', 'lb_count',
        'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
        'new_high_20_ratio', 'new_low_20_ratio',
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
        'above_ma5_ratio': 'mean',
        'above_ma10_ratio': 'mean',
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

    score_base = scores[['date', 'code'] + SCORE_COLUMNS].copy()
    score_base = score_base.merge(base[['date', 'code']], on=['date', 'code'], how='inner')
    if not score_base.empty:
        for col in SCORE_COLUMNS:
            score_base[col] = pd.to_numeric(score_base[col], errors='coerce')
        score_market = score_base.groupby('date', sort=True)[SCORE_COLUMNS].mean().reset_index()
        market = market.merge(score_market, on='date', how='left')
    else:
        for col in SCORE_COLUMNS:
            market[col] = pd.NA

    market = _calc_market_features(market)
    if market.empty:
        raise ValueError('无法生成市场卦：市场时序特征全部为空，请检查市场代理序列或 warmup 窗口')

    market = _mark_segments(market)

    for col in OUTPUT_COLUMNS:
        if col not in market.columns:
            market[col] = pd.NA
    market = market[OUTPUT_COLUMNS].sort_values('date').reset_index(drop=True)

    out_path = foundation_file('market_bagua_daily.csv')
    market.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('市场八卦生成完成')
    print('=' * 80)
    print(f'日期范围: {market["date"].min()} ~ {market["date"].max()}')
    print(f'交易日数: {len(market)}')
    print(f'卦类数: {market["gua_code"].nunique()}')
    print(f'输出: {out_path}')
    return market


if __name__ == '__main__':
    build_market_bagua()
