# -*- coding: utf-8 -*-
"""
数据层 - 阳系统段首事件表生成

为每个512组合的段首日记录:
  - 当日卦组合 (year_gua, month_gua, day_gua)
  - 未来30日收益 (ret_30)
  - 可用日期 (avail_date): 30个交易日后, 即该ret_30在实时中"解锁"的日期

查询日期T的512卦象图时, 只取 avail_date <= T 的事件做统计
这样就是纯阳系统 — 不使用任何未来信息

输出:
  data_layer/data/yang_seg_events.csv

新数据到来后重新运行即可更新
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from bagua_engine import BAGUA_TABLE


def prepare_yang_events():
    """生成段首事件表"""
    print("=" * 80)
    print("阳系统段首事件表生成")
    print("=" * 80)

    # ── 加载数据 ──
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'zz1000_daily.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['date'] = df['date'].astype(str)

    for col in ['year_gua', 'month_gua', 'day_gua']:
        df[col] = df[col].astype(str).str.zfill(3)

    # 未来30日收益
    df['ret_30'] = (df['close'].shift(-30) / df['close'] - 1) * 100

    # 可用日期: 第i行的ret_30在第i+30行的日期才"解锁"
    df['avail_date'] = df['date'].shift(-30)

    # 512组合段首标记
    df['combo'] = df['year_gua'] + '_' + df['month_gua'] + '_' + df['day_gua']
    df['combo_seg_start'] = df['combo'] != df['combo'].shift(1)

    print(f"数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({len(df)}个交易日)")

    # 筛选: 段首 + 有ret_30 + 有avail_date
    events = df[df['combo_seg_start'] & df['ret_30'].notna() & df['avail_date'].notna()].copy()

    # 构建事件表
    event_rows = []
    for _, row in events.iterrows():
        yg, mg, dg = row['year_gua'], row['month_gua'], row['day_gua']
        event_rows.append({
            'event_date': row['date'],
            'avail_date': row['avail_date'],
            'year_gua': yg,
            'month_gua': mg,
            'day_gua': dg,
            'year_name': BAGUA_TABLE[yg][0],
            'month_name': BAGUA_TABLE[mg][0],
            'day_name': BAGUA_TABLE[dg][0],
            'year_yy': BAGUA_TABLE[yg][3],
            'month_yy': BAGUA_TABLE[mg][3],
            'day_yy': BAGUA_TABLE[dg][3],
            'ret_30': round(row['ret_30'], 4),
            'close': round(row['close'], 4),
        })

    event_df = pd.DataFrame(event_rows)

    # 按事件日期排序
    event_df = event_df.sort_values('event_date').reset_index(drop=True)

    # 保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'yang_seg_events.csv')
    event_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    # ── 统计摘要 ──
    n_events = len(event_df)
    n_combos = event_df.groupby(['year_gua', 'month_gua', 'day_gua']).ngroups
    date_range = f"{event_df['event_date'].iloc[0]} ~ {event_df['event_date'].iloc[-1]}"
    avail_range = f"{event_df['avail_date'].iloc[0]} ~ {event_df['avail_date'].iloc[-1]}"

    print(f"\n段首事件数: {n_events}")
    print(f"覆盖组合数: {n_combos} / 512")
    print(f"事件日期范围: {date_range}")
    print(f"可用日期范围: {avail_range}")

    # 各组合事件数分布
    combo_counts = event_df.groupby(['year_gua', 'month_gua', 'day_gua']).size()
    print(f"\n各组合事件数分布:")
    print(f"  最多: {combo_counts.max()} 次")
    print(f"  最少: {combo_counts.min()} 次")
    print(f"  中位数: {combo_counts.median():.0f} 次")
    print(f"  >= 3次(可统计): {(combo_counts >= 3).sum()} 种组合")

    # 展示事件数最多的TOP10组合
    top10 = combo_counts.sort_values(ascending=False).head(10)
    print(f"\n  事件数TOP10组合:")
    print(f"  {'年':>4} {'月':>4} {'日':>4} {'事件数':>6}")
    print(f"  {'─' * 25}")
    for (yg, mg, dg), cnt in top10.items():
        print(f"  {BAGUA_TABLE[yg][0]:>4} {BAGUA_TABLE[mg][0]:>4} {BAGUA_TABLE[dg][0]:>4} {cnt:>6}")

    print(f"\n保存: {out_path}")
    print(f"{'=' * 80}")

    return event_df


if __name__ == '__main__':
    prepare_yang_events()
