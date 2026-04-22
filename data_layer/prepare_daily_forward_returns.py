# -*- coding: utf-8 -*-
"""
prepare_daily_forward_returns.py

基于新底座横截面生成个股未来收益标签。
标签与八卦链路分离，避免在生成状态时混入未来信息。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file


HORIZONS = [1, 3, 5, 10, 20, 60]


def build_daily_forward_returns():
    cross_path = foundation_file('daily_cross_section.csv')
    if not os.path.exists(cross_path):
        raise FileNotFoundError(f'daily_cross_section.csv 不存在: {cross_path}')

    df = pd.read_csv(cross_path, encoding='utf-8-sig', dtype={'code': str}, usecols=['date', 'code', 'close'])
    if df.empty:
        raise ValueError('daily_cross_section.csv 为空，无法生成未来收益标签')

    df['date'] = df['date'].astype(str)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    out = df[['date', 'code']].copy()
    for horizon in HORIZONS:
        future_close = df.groupby('code')['close'].shift(-horizon)
        future_date = df.groupby('code')['date'].shift(-horizon)
        ret = (future_close / df['close'] - 1) * 100
        out[f'ret_fwd_{horizon}d'] = ret.round(4)
        out[f'avail_date_{horizon}d'] = future_date.astype(str)
        out.loc[future_date.isna(), f'avail_date_{horizon}d'] = pd.NA

    out = out.sort_values(['date', 'code']).reset_index(drop=True)
    out_path = foundation_file('daily_forward_returns.csv')
    out.to_csv(out_path, index=False, encoding='utf-8-sig')

    print('=' * 80)
    print('未来收益标签生成完成')
    print('=' * 80)
    print(f'日期范围: {out["date"].min()} ~ {out["date"].max()}')
    print(f'样本数: {len(out)}')
    print(f'输出: {out_path}')
    return out


if __name__ == '__main__':
    build_daily_forward_returns()
