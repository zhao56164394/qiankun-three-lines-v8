# -*- coding: utf-8 -*-
"""
migrate_to_parquet.py — 一次性把核心 CSV 转成 Parquet

转换清单：
  1. foundation/ 下 9 个 GB 级核心文件 → 同名 .parquet
  2. data_layer/data/zz1000_daily.csv → zz1000_daily.parquet
  3. data_layer/data/stocks/*.csv (5102 只) → 单一 stocks.parquet
  4. data_layer/data/stock_seg_events.csv → stock_seg_events.parquet

dtype 约束：
  - code 列：string (防前导 0 丢失)
  - 所有 *_gua / gua_code 列：string (防 '001' 变成 1)
  - date 列：string (与现有代码一致)
  - 数值列：保持 float/int 自动推断
"""
import os
import sys
import time
import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import (
    FOUNDATION_DATA_DIR, STOCKS_DATA_DIR, DATA_LAYER_ROOT,
    foundation_file, foundation_parquet, STOCKS_PARQUET, ZZ1000_PARQUET,
)


# 必须强制为字符串的列名（防前导 0/三位卦码丢失）
STRING_COLUMNS = {
    'code', 'date', 'gua_code', 'gua_name',
    'd_gua', 'm_gua', 'y_gua', 'd_name', 'm_name', 'y_name',
    'gua', 'zz_gua', 'prev_gua',
    'name', 'exchange', 'board', 'industry_name', 'avail_date',
    'avail_date_1d', 'avail_date_3d', 'avail_date_5d',
    'avail_date_10d', 'avail_date_20d', 'avail_date_60d',
    'list_date', 'event_date', 'sell_method',
}


def _coerce_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """对已知字符串列强制 string 类型，code 列额外补前导 0。"""
    for col in df.columns:
        if col in STRING_COLUMNS:
            df[col] = df[col].astype('string')
    if 'code' in df.columns:
        df['code'] = df['code'].astype('string').str.zfill(6)
    if 'gua_code' in df.columns:
        df['gua_code'] = df['gua_code'].astype('string').str.zfill(3)
    for col in ['d_gua', 'm_gua', 'y_gua']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.zfill(3)
    return df


def _convert_one(csv_path: str, parquet_path: str, label: str) -> None:
    """转换单个 CSV → Parquet。"""
    if not os.path.exists(csv_path):
        print(f'  [跳过] {label}: CSV 不存在')
        return
    csv_size = os.path.getsize(csv_path) / 1024 / 1024
    t0 = time.time()
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    df = _coerce_string_columns(df)
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
    dt = time.time() - t0
    pq_size = os.path.getsize(parquet_path) / 1024 / 1024
    print(f'  [OK] {label}: {csv_size:.1f}MB → {pq_size:.1f}MB  '
          f'(压缩 {csv_size/pq_size:.1f}x, {dt:.1f}s)')


# ============================================================
# 1. foundation/ 大文件
# ============================================================
FOUNDATION_FILES = [
    'daily_cross_section.csv',
    'stock_bagua_daily.csv',
    'daily_5d_scores.csv',
    'daily_forward_returns.csv',
    'main_board_universe.csv',
    'daily_bagua_sequence.csv',
    'stock_daily_gua.csv',
    'multi_scale_gua_daily.csv',
    'market_bagua_daily.csv',
]


def migrate_foundation():
    print('\n=== 1. foundation/ 核心大文件 ===')
    for fname in FOUNDATION_FILES:
        csv_p = foundation_file(fname)
        pq_p = foundation_parquet(fname)
        _convert_one(csv_p, pq_p, fname)


# ============================================================
# 2. zz1000 + stock_seg_events
# ============================================================
def migrate_zz1000():
    print('\n=== 2. zz1000_daily ===')
    csv_p = os.path.join(DATA_LAYER_ROOT, 'data', 'zz1000_daily.csv')
    _convert_one(csv_p, ZZ1000_PARQUET, 'zz1000_daily.csv')


def migrate_stock_seg_events():
    print('\n=== 3. stock_seg_events ===')
    csv_p = os.path.join(DATA_LAYER_ROOT, 'data', 'stock_seg_events.csv')
    pq_p = os.path.join(DATA_LAYER_ROOT, 'data', 'stock_seg_events.parquet')
    _convert_one(csv_p, pq_p, 'stock_seg_events.csv')


# ============================================================
# 4. stocks/ 5102 个 CSV → 单一 stocks.parquet
# ============================================================
def migrate_stocks():
    print('\n=== 4. stocks/ 5102 个个股 CSV ===')
    csv_files = sorted(glob.glob(os.path.join(STOCKS_DATA_DIR, '*.csv')))
    print(f'  发现 {len(csv_files)} 个个股 CSV')

    if not csv_files:
        print('  [跳过] stocks 目录为空')
        return

    t0 = time.time()
    parts = []
    total_size = 0
    for i, fpath in enumerate(csv_files):
        if (i + 1) % 500 == 0:
            print(f'  进度 {i+1}/{len(csv_files)}')
        code = os.path.splitext(os.path.basename(fpath))[0]
        total_size += os.path.getsize(fpath)
        df = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False)
        df.insert(0, 'code', code)
        parts.append(df)

    print(f'  合并 {len(parts)} 个 DataFrame...')
    full = pd.concat(parts, ignore_index=True)
    del parts

    full = _coerce_string_columns(full)
    full = full.sort_values(['code', 'date']).reset_index(drop=True)

    print(f'  写入 {STOCKS_PARQUET} ...')
    full.to_parquet(STOCKS_PARQUET, engine='pyarrow', compression='snappy',
                    index=False)
    dt = time.time() - t0
    pq_size = os.path.getsize(STOCKS_PARQUET) / 1024 / 1024
    csv_total_mb = total_size / 1024 / 1024
    print(f'  [OK] stocks: {csv_total_mb:.0f}MB ({len(csv_files)}文件) '
          f'→ {pq_size:.1f}MB (1文件)  压缩 {csv_total_mb/pq_size:.1f}x, {dt:.0f}s')
    print(f'  行数 {len(full):,}, 唯一 code {full["code"].nunique()}')


# ============================================================
# 入口
# ============================================================
def main():
    print('=' * 60)
    print('  CSV → Parquet 一次性转换')
    print('=' * 60)
    t0 = time.time()

    migrate_foundation()
    migrate_zz1000()
    migrate_stock_seg_events()
    migrate_stocks()

    print('\n' + '=' * 60)
    print(f'  全部完成 ({time.time()-t0:.0f}s)')
    print('=' * 60)


if __name__ == '__main__':
    main()
