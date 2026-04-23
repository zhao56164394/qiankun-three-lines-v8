# -*- coding: utf-8 -*-
"""
数据层 - 个股段首事件表生成

为每只股票的每个象卦段首日记录:
  - 股票代码
  - 当日象卦 (gua)
  - 未来30日超额收益 (excess_ret)
  - 可用日期 (avail_date): 30个交易日后解锁

查询日期T时, 只取 avail_date <= T 的事件做统计
纯阳系统 — 不使用任何未来信息

输出:
  data_layer/data/stock_seg_events.csv
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def prepare_stock_events():
    """生成个股段首事件表（超额收益版 + 中证象卦）"""
    print("=" * 80)
    print("个股段首事件表生成（超额收益 + ren_gua）")
    print("=" * 80)

    # 加载中证1000日线，构建日期→ret_30映射 + 日期→ren_gua映射
    zz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'data', 'zz1000_daily.csv')
    zz = pd.read_csv(zz_path, encoding='utf-8-sig', usecols=['date', 'close', 'gua'])
    zz['date'] = zz['date'].astype(str)
    zz['zz_ret_30'] = (zz['close'].shift(-30) / zz['close'] - 1) * 100
    zz['gua'] = zz['gua'].astype(str).str.split('.').str[0].str.zfill(3)
    zz_ret_map = dict(zip(zz['date'], zz['zz_ret_30']))
    ren_gua_map = dict(zip(zz['date'], zz['gua']))
    print(f"中证1000日线: {len(zz)}天, 有ret_30: {zz['zz_ret_30'].notna().sum()}天")

    stock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'stocks')
    files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    print(f"股票文件数: {len(files)}")

    all_events = []
    n_processed = 0

    for fname in files:
        code = fname.replace('.csv', '')
        fpath = os.path.join(stock_dir, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig',
                             usecols=['date', 'close', 'gua'])
        except Exception:
            continue

        df['date'] = df['date'].astype(str)
        df['gua'] = df['gua'].astype(str).str.zfill(3)

        # 未来30日收益
        df['ret_30'] = (df['close'].shift(-30) / df['close'] - 1) * 100
        # 超额收益 = 个股ret_30 - 中证1000同期ret_30
        df['zz_ret_30'] = df['date'].map(zz_ret_map)
        df['excess_ret'] = df['ret_30'] - df['zz_ret_30']
        # 可用日期: 第i行的ret_30在第i+30行的日期才解锁
        df['avail_date'] = df['date'].shift(-30)

        # 象卦段首标记
        df['seg_start'] = df['gua'] != df['gua'].shift(1)

        # 筛选: 段首 + 有excess_ret + 有avail_date
        events = df[df['seg_start'] & df['excess_ret'].notna() & df['avail_date'].notna()]

        for _, row in events.iterrows():
            all_events.append({
                'code': code,
                'event_date': row['date'],
                'avail_date': row['avail_date'],
                'gua': row['gua'],
                'ren_gua': ren_gua_map.get(row['date'], ''),
                'excess_ret': round(row['excess_ret'], 4),
            })

        n_processed += 1
        if n_processed % 500 == 0:
            print(f"  已处理 {n_processed}/{len(files)}, 累计事件 {len(all_events)}")

    event_df = pd.DataFrame(all_events)
    event_df = event_df.sort_values('event_date').reset_index(drop=True)

    # 保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'stock_seg_events.csv')
    event_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    # 统计
    n_events = len(event_df)
    n_combos = event_df['gua'].nunique()
    n_stocks = event_df['code'].nunique()

    print(f"\n总事件数: {n_events}")
    print(f"覆盖股票: {n_stocks}")
    print(f"覆盖卦象: {n_combos} / 8")
    print(f"日期范围: {event_df['event_date'].iloc[0]} ~ {event_df['event_date'].iloc[-1]}")
    print(f"可用日期: {event_df['avail_date'].iloc[0]} ~ {event_df['avail_date'].iloc[-1]}")

    combo_counts = event_df.groupby('gua').size()
    print(f"\n各卦事件数: 最多{combo_counts.max()} 最少{combo_counts.min()} 中位{combo_counts.median():.0f}")
    print(f">=3次(可统计): {(combo_counts >= 3).sum()} 种卦象")

    # 交叉表统计
    cross_counts = event_df.groupby(['ren_gua', 'gua']).size()
    cross_mean = event_df.groupby(['ren_gua', 'gua'])['excess_ret'].mean()
    print(f"\n交叉表(ren_gua x stk_gua): {len(cross_counts)} 个组合, >=3次: {(cross_counts >= 3).sum()}")

    # 超额收益分布
    combo_mean = event_df.groupby('gua')['excess_ret'].mean()
    print(f"\n各卦超额收益均值:")
    for gua_code, ret in combo_mean.sort_values(ascending=False).items():
        print(f"  {gua_code}: {ret:+.2f}%")

    print(f"\n保存: {out_path}")
    print(f"文件大小: {os.path.getsize(out_path)/1024/1024:.1f} MB")
    return event_df


if __name__ == '__main__':
    prepare_stock_events()
