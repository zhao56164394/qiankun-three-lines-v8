# -*- coding: utf-8 -*-
"""
prepare_daily_bagua.py

将 5 维评分压缩为 3 爻，并生成 8 卦结果。
支持全历史真实序列与连续段标记。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file


GUA_NAME_MAP = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def _to_yao(score, threshold=50):
    return (pd.to_numeric(score, errors='coerce').fillna(50) >= threshold).astype(int)


def _mark_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['code', 'date']).copy()
    df['prev_gua'] = df.groupby('code')['gua_code'].shift(1).fillna('')
    df['changed'] = (df['gua_code'] != df['prev_gua']).astype(int)
    df.loc[df['prev_gua'] == '', 'changed'] = 1
    df['seg_id'] = df.groupby('code')['changed'].cumsum().astype(int)
    df['seg_day'] = df.groupby(['code', 'seg_id']).cumcount() + 1
    return df


def build_daily_bagua(date=None):
    score_path = foundation_file('daily_5d_scores.csv')
    if not os.path.exists(score_path):
        from data_layer.prepare_daily_5d_scores import build_daily_5d_scores
        build_daily_5d_scores(date=date)

    df = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str})
    if date is not None:
        df = df[df['date'].astype(str) == str(date)].copy()
    if df.empty:
        raise ValueError('daily_5d_scores.csv 为空，无法生成八卦')

    df['yao_1'] = _to_yao(df['score_wei'])
    df['yao_2'] = _to_yao((pd.to_numeric(df['score_shi'], errors='coerce') + pd.to_numeric(df['score_bian'], errors='coerce')) / 2)
    df['yao_3'] = _to_yao((pd.to_numeric(df['score_zhong'], errors='coerce') + pd.to_numeric(df['score_qi'], errors='coerce')) / 2)

    df['gua_code'] = df['yao_1'].astype(str) + df['yao_2'].astype(str) + df['yao_3'].astype(str)
    df['gua_name'] = df['gua_code'].map(GUA_NAME_MAP)
    df = _mark_segments(df)

    yao_out = df[['date', 'code', 'yao_1', 'yao_2', 'yao_3']].copy().sort_values(['date', 'code'])
    bagua_out = df[['date', 'code', 'gua_code', 'gua_name', 'prev_gua', 'changed', 'seg_id', 'seg_day']].copy().sort_values(['date', 'code'])

    yao_path = foundation_file('daily_3yao.csv')
    bagua_path = foundation_file('daily_bagua_sequence.csv')
    yao_out.to_csv(yao_path, index=False, encoding='utf-8-sig')
    bagua_out.to_csv(bagua_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('3爻与8卦生成完成')
    print('=' * 80)
    print(f'日期范围: {bagua_out["date"].min()} ~ {bagua_out["date"].max()}')
    print(f'样本数: {len(bagua_out)}')
    print(f'3爻输出: {yao_path}')
    print(f'八卦输出: {bagua_path}')
    return yao_out, bagua_out


if __name__ == '__main__':
    build_daily_bagua()
