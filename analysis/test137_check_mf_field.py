# -*- coding: utf-8 -*-
"""检查 main_force 字段的真实含义

通达信看到入池时主力线是负数, 但脚本里 mf=+71. 必有一个不对.

候选:
  1. main_force 字段定义反了 (正的实际是流出, 负的是流入)
  2. main_force 不是日值, 是某个聚合
  3. 数据列名错了
"""
import os, sys, io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print('=== 检查 main_force 数据 ===\n')

    # 1. 看原始 stocks.parquet 都有什么字段
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'))
    print('字段列表:')
    print(p.columns.tolist())
    print(f'\n  形状: {p.shape}')
    print(f'  样例 5 行:')
    print(p.head().to_string())

    # 2. 顺丰 002352 在 2016-01-19 当天的所有字段
    print('\n\n=== 002352 顺丰控股 2016-01-19 当天 ===')
    sub = p[(p['code'].astype(str).str.zfill(6) == '002352') &
                (p['date'].astype(str) == '2016-01-19')]
    if len(sub):
        print(sub.T.to_string())
    else:
        # 也许字段是 datetime
        sub = p[(p['code'].astype(str).str.zfill(6) == '002352')]
        sub = sub[sub['date'].astype(str).str.contains('2016-01')]
        print(sub.head(20).to_string())

    # 3. 顺丰 2016-01 整月所有数据
    print('\n=== 002352 顺丰控股 2016-01 整月 main_force / retail ===')
    sub = p[(p['code'].astype(str).str.zfill(6) == '002352')]
    sub = sub[sub['date'].astype(str).between('2016-01-01', '2016-02-01')]
    if 'main_force' in sub.columns and 'retail' in sub.columns:
        print(sub[['date', 'close', 'main_force', 'retail']].to_string())

    # 4. 字段有没有别的资金流相关列
    print('\n=== 资金流相关字段 ===')
    for col in p.columns:
        if any(k in col.lower() for k in ['main', 'force', 'retail', 'flow', 'net', 'large', 'small', 'medium', 'super']):
            print(f'  {col}')

    # 5. 全市场 main_force 和 retail 各自总和 (理论上应该镜像 = 0)
    print('\n=== 数据校验: 同日全市场 main_force + retail 是否 ≈ 0 ===')
    if 'main_force' in p.columns and 'retail' in p.columns:
        # 取 100 个日期看
        for d in sorted(p['date'].astype(str).unique())[::500][:5]:
            sub = p[p['date'].astype(str) == d]
            mf_sum = sub['main_force'].sum()
            re_sum = sub['retail'].sum()
            print(f'  {d}: main_force_sum={mf_sum:>+12.0f}  retail_sum={re_sum:>+12.0f}  '
                  f'sum={mf_sum+re_sum:>+12.0f}')


if __name__ == '__main__':
    main()
