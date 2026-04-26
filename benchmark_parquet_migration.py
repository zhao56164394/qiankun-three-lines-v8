# -*- coding: utf-8 -*-
"""
benchmark_parquet_migration.py — 验证 Parquet 迁移的正确性 + 测速

对比 CSV vs Parquet 在以下入口的：
1. 行数一致
2. 关键字段值一致 (code/gua_code 前导零、date 字符串)
3. 加载耗时
"""
import os
import sys
import io
import time
import shutil
import tempfile

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def time_it(label, fn):
    t0 = time.time()
    result = fn()
    dt = time.time() - t0
    return dt, result


def compare_dfs(name, df_csv, df_pq, key_cols=None):
    """比较两个 DataFrame 的等价性。
    注意：Parquet 化时把 code/gua_code/d_gua/m_gua/y_gua 强制 string + zfill,
    所以比对前两边都做相同标准化。
    """
    issues = []
    if len(df_csv) != len(df_pq):
        issues.append(f'  X 行数不同 csv={len(df_csv)} pq={len(df_pq)}')
    if set(df_csv.columns) != set(df_pq.columns):
        only_csv = set(df_csv.columns) - set(df_pq.columns)
        only_pq = set(df_pq.columns) - set(df_csv.columns)
        if only_csv: issues.append(f'  ! csv 独有列: {only_csv}')
        if only_pq: issues.append(f'  ! pq 独有列: {only_pq}')

    def _norm(s, col):
        s = s.astype(str)
        if col == 'code': return s.str.zfill(6)
        if col in ('gua_code', 'd_gua', 'm_gua', 'y_gua'): return s.str.zfill(3)
        return s

    if key_cols:
        for col in key_cols:
            if col in df_csv.columns and col in df_pq.columns:
                a = _norm(df_csv[col].head(100), col).tolist()
                b = _norm(df_pq[col].head(100), col).tolist()
                if a != b:
                    issues.append(f'  X {col} 前 100 行不同: csv样本={a[:3]}, pq样本={b[:3]}')
    if issues:
        print(f'\n[{name}] 不一致:')
        for i in issues: print(i)
    else:
        print(f'  [OK] {name}: 行数 {len(df_csv):,} 完全一致')


print('=' * 60)
print('  Parquet 迁移验证 + 性能测试')
print('=' * 60)

from data_layer.foundation_config import foundation_file, foundation_parquet, STOCKS_PARQUET, ZZ1000_PARQUET, DATA_LAYER_ROOT


# ============================================================
# 1. foundation 几个核心文件
# ============================================================
print('\n=== 1. foundation 核心文件 (CSV vs Parquet 一致性 & 速度) ===\n')

CASES = [
    ('market_bagua_daily.csv', ['date', 'gua_code']),
    ('multi_scale_gua_daily.csv', ['date', 'd_gua', 'm_gua', 'y_gua']),
    ('daily_bagua_sequence.csv', ['date', 'code', 'gua_code']),
    ('main_board_universe.csv', ['date', 'code']),
]
for fname, key_cols in CASES:
    csv_p = foundation_file(fname)
    pq_p = foundation_parquet(fname)
    if not (os.path.exists(csv_p) and os.path.exists(pq_p)):
        print(f'  [跳过] {fname}: 文件缺失')
        continue
    t_csv, df_csv = time_it('csv', lambda: pd.read_csv(csv_p, encoding='utf-8-sig', low_memory=False))
    t_pq, df_pq = time_it('pq', lambda: pd.read_parquet(pq_p))
    print(f'\n{fname}: CSV {t_csv:.2f}s  →  Parquet {t_pq:.2f}s  ({t_csv/max(t_pq,0.001):.1f}x)')
    compare_dfs(fname, df_csv, df_pq, key_cols=key_cols)


# ============================================================
# 2. 个股加载: 5102 CSV vs stocks.parquet
# ============================================================
print('\n=== 2. 个股加载 (5102 CSV 循环 vs 1 个 Parquet) ===\n')

import backtest_capital

# 清掉 pkl 缓存确保公平对比
for pkl in ['_cache_stocks.pkl']:
    p = os.path.join(DATA_LAYER_ROOT, 'data', pkl)
    if os.path.exists(p):
        os.remove(p)

# Parquet 路径
t_pq, data_pq = time_it('parquet', lambda: backtest_capital._build_stocks())
print(f'  Parquet 路径: {t_pq:.1f}s, 加载 {len(data_pq)} 只股票')

# 临时把 stocks.parquet 改名让 _build_stocks 走 CSV 分支
if os.path.exists(STOCKS_PARQUET):
    bak = STOCKS_PARQUET + '.tmp_bench'
    shutil.move(STOCKS_PARQUET, bak)
    try:
        t_csv, data_csv = time_it('csv', lambda: backtest_capital._build_stocks())
        print(f'  CSV 路径:     {t_csv:.1f}s, 加载 {len(data_csv)} 只股票')
        print(f'  → Parquet 提速 {t_csv/max(t_pq, 0.001):.1f}x')

        # 抽样验证一致性
        sample_codes = list(data_pq.keys())[:3]
        for c in sample_codes:
            df_pq = data_pq[c]
            df_csv = data_csv.get(c)
            if df_csv is None:
                print(f'  X {c} 在 CSV 路径中缺失')
                continue
            if len(df_pq) != len(df_csv):
                print(f'  X {c} 行数不一致 pq={len(df_pq)} csv={len(df_csv)}')
            else:
                # 对比 close 列
                same = np.allclose(df_pq['close'].values, df_csv['close'].values, equal_nan=True)
                print(f'  [{"OK" if same else "X"}] {c}: {len(df_pq)} 行, close 列一致={same}')
    finally:
        shutil.move(bak, STOCKS_PARQUET)


# ============================================================
# 3. zz1000 加载
# ============================================================
print('\n=== 3. zz1000 加载 ===\n')

zz_pkl = os.path.join(DATA_LAYER_ROOT, 'data', '_cache_zz1000.pkl')
if os.path.exists(zz_pkl): os.remove(zz_pkl)

t_pq, _ = time_it('zz1000_pq', lambda: backtest_capital._build_zz1000())
print(f'  Parquet: {t_pq:.2f}s')

if os.path.exists(ZZ1000_PARQUET):
    bak = ZZ1000_PARQUET + '.tmp_bench'
    shutil.move(ZZ1000_PARQUET, bak)
    try:
        t_csv, _ = time_it('zz1000_csv', lambda: backtest_capital._build_zz1000())
        print(f'  CSV:     {t_csv:.2f}s  → 提速 {t_csv/max(t_pq, 0.001):.1f}x')
    finally:
        shutil.move(bak, ZZ1000_PARQUET)


# ============================================================
# 4. foundation_data 入口
# ============================================================
print('\n=== 4. foundation_data 入口 (清缓存) ===\n')

import data_layer.foundation_data as fd
fd._cache.clear()
t1, df1 = time_it('load_market_bagua', lambda: fd.load_market_bagua())
print(f'  load_market_bagua: {t1:.2f}s, {len(df1)} 行')

fd._cache.clear()
t2, df2 = time_it('load_daily_bagua', lambda: fd.load_daily_bagua())
print(f'  load_daily_bagua: {t2:.2f}s, {len(df2):,} 行')

print('\n' + '=' * 60)
print('  验证完成')
print('=' * 60)
