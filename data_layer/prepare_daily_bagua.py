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
from data_layer.gua_data import GUA_NAMES as GUA_NAME_MAP


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
    """全量重建 daily_bagua_sequence.csv（慢，770万行）"""
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
    print('3爻与8卦生成完成 (全量)')
    print('=' * 80)
    print(f'日期范围: {bagua_out["date"].min()} ~ {bagua_out["date"].max()}')
    print(f'样本数: {len(bagua_out)}')
    return yao_out, bagua_out


def update_daily_bagua():
    """增量更新 daily_bagua_sequence.csv — 只追加新日期，自动接续段信息。"""
    bagua_path = foundation_file('daily_bagua_sequence.csv')
    yao_path = foundation_file('daily_3yao.csv')
    score_path = foundation_file('daily_5d_scores.csv')

    if not os.path.exists(bagua_path) or not os.path.exists(score_path):
        print('文件不存在，执行全量构建...')
        return build_daily_bagua()

    existing = pd.read_csv(bagua_path, encoding='utf-8-sig', dtype={'code': str})
    last_date = str(existing['date'].max())

    scores = pd.read_csv(score_path, encoding='utf-8-sig', dtype={'code': str})
    new_scores = scores[scores['date'].astype(str) > last_date].copy()
    if new_scores.empty:
        print(f'daily_bagua_sequence 已是最新 ({last_date})，无需更新')
        return None, None

    new_dates = sorted(new_scores['date'].unique())
    print(f'增量更新: {len(new_dates)} 个新日期 ({new_dates[0]} ~ {new_dates[-1]})')

    new_scores['yao_1'] = _to_yao(new_scores['score_wei'])
    new_scores['yao_2'] = _to_yao((pd.to_numeric(new_scores['score_shi'], errors='coerce') + pd.to_numeric(new_scores['score_bian'], errors='coerce')) / 2)
    new_scores['yao_3'] = _to_yao((pd.to_numeric(new_scores['score_zhong'], errors='coerce') + pd.to_numeric(new_scores['score_qi'], errors='coerce')) / 2)
    new_scores['gua_code'] = new_scores['yao_1'].astype(str) + new_scores['yao_2'].astype(str) + new_scores['yao_3'].astype(str)
    new_scores['gua_name'] = new_scores['gua_code'].map(GUA_NAME_MAP)

    last_state = existing.groupby('code').last()[['gua_code', 'seg_id']].to_dict('index')

    new_rows = new_scores[['date', 'code', 'gua_code', 'gua_name']].copy().sort_values(['code', 'date'])
    prev_gua_list = []
    changed_list = []
    seg_id_list = []
    seg_day_list = []

    current_stock = None
    cur_gua = ''
    cur_seg_id = 0
    cur_seg_day = 0

    for _, row in new_rows.iterrows():
        code = row['code']
        gua = row['gua_code']

        if code != current_stock:
            current_stock = code
            state = last_state.get(code, {})
            cur_gua = str(state.get('gua_code', ''))
            cur_seg_id = int(state.get('seg_id', 0))
            cur_seg_day = 0

        prev_gua_list.append(cur_gua)
        if gua != cur_gua:
            changed_list.append(1)
            cur_seg_id += 1
            cur_seg_day = 1
        else:
            changed_list.append(0)
            cur_seg_day += 1

        seg_id_list.append(cur_seg_id)
        seg_day_list.append(cur_seg_day)
        cur_gua = gua

    new_rows['prev_gua'] = prev_gua_list
    new_rows['changed'] = changed_list
    new_rows['seg_id'] = seg_id_list
    new_rows['seg_day'] = seg_day_list

    new_bagua = new_rows[['date', 'code', 'gua_code', 'gua_name', 'prev_gua', 'changed', 'seg_id', 'seg_day']]
    combined = pd.concat([existing, new_bagua], ignore_index=True).sort_values(['date', 'code'])
    combined.to_csv(bagua_path, index=False, encoding='utf-8-sig')

    new_yao = new_scores[['date', 'code', 'yao_1', 'yao_2', 'yao_3']].sort_values(['date', 'code'])
    if os.path.exists(yao_path):
        existing_yao = pd.read_csv(yao_path, encoding='utf-8-sig', dtype={'code': str})
        combined_yao = pd.concat([existing_yao, new_yao], ignore_index=True).sort_values(['date', 'code'])
    else:
        combined_yao = new_yao
    combined_yao.to_csv(yao_path, index=False, encoding='utf-8-sig')

    print(f'增量更新完成: 新增 {len(new_bagua)} 行')
    print(f'日期范围: {combined["date"].min()} ~ {combined["date"].max()}')
    print(f'总样本数: {len(combined)}')

    # 同步 Parquet（迁移期双轨支持）
    try:
        from data_layer.update_foundation import _sync_parquet
        _sync_parquet(bagua_path)
        _sync_parquet(yao_path)
    except Exception as e:
        print(f'  ⚠ Parquet 同步失败: {e}')
    return new_yao, new_bagua


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='全量重建')
    args = parser.parse_args()
    if args.full:
        build_daily_bagua()
    else:
        update_daily_bagua()
