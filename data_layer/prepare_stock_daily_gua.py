# -*- coding: utf-8 -*-
"""个股日卦 (v10 规则, 无未来函数)

对每只股逐日套用 v10 三爻:
  位 (pos) = 个股 trend >= 50
  势 (spd) = 个股 trend 日差 ±2 滞后带 + >=89 高位保护
  变 (acc) = 个股 main_force ±30 滞后带

个股 K 线文件 (data/stocks/{code}.csv) 里已有 trend / main_force 列
(由 strategy/indicator.py 的 calc_trend_line / calc_main_force_line 预计算).
这里只做 v10 规则 + 码合成.

输出: data_layer/data/foundation/stock_daily_gua.csv
列: date, code, trend, mf, pos, spd, acc, gua_code, gua_name
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.prepare_multi_scale_gua import apply_v10_rules, GUA_NAMES  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCKS_DIR = os.path.join(ROOT, 'data_layer', 'data', 'stocks')
OUTPUT = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'stock_daily_gua.csv')


def process_one(code, df):
    df = df.sort_values('date').reset_index(drop=True)
    trend = df['trend'].astype(float).values
    mf = df['main_force'].astype(float).values

    yao_pos, yao_spd, yao_acc, gua_codes = apply_v10_rules(trend, mf)

    return pd.DataFrame({
        'date': df['date'].values,
        'code': code,
        'trend': trend,
        'mf': mf,
        'pos': yao_pos,
        'spd': yao_spd,
        'acc': yao_acc,
        'gua_code': gua_codes,
    })


def main():
    files = sorted(Path(STOCKS_DIR).glob('*.csv'))
    n = len(files)
    print(f'个股文件数: {n}')

    chunks = []
    skipped = 0
    for i, f in enumerate(files):
        code = f.stem
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
        except Exception:
            skipped += 1
            continue
        if 'trend' not in df.columns or 'main_force' not in df.columns:
            skipped += 1
            continue
        df = df.dropna(subset=['trend', 'main_force']).copy()
        if len(df) < 20:
            skipped += 1
            continue
        out = process_one(code, df)
        chunks.append(out)
        if (i + 1) % 500 == 0:
            print(f'  处理 {i + 1}/{n} ...')

    print(f'合并 {len(chunks)} 只股, 跳过 {skipped}')
    result = pd.concat(chunks, ignore_index=True)

    # 去空卦 (早期无值)
    result = result[result['gua_code'].astype(str) != ''].copy()
    result['gua_name'] = result['gua_code'].map(GUA_NAMES)
    result = result.dropna(subset=['gua_name'])

    # 保留有效列
    result = result[['date', 'code', 'trend', 'mf', 'pos', 'spd', 'acc', 'gua_code', 'gua_name']]
    result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')

    print(f'\n保存 {OUTPUT}')
    print(f'  总行数: {len(result):,}')
    print(f'  日期范围: {result["date"].min()} ~ {result["date"].max()}')
    print(f'  股票数: {result["code"].nunique()}')
    print(f'\n8 态分布:')
    dist = result['gua_code'].value_counts(normalize=True).sort_index() * 100
    for code, pct in dist.items():
        print(f'  {code}{GUA_NAMES.get(code, "?")}: {pct:.2f}%')


if __name__ == '__main__':
    main()
