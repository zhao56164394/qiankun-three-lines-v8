# -*- coding: utf-8 -*-
"""
数据层 - 预计算中证1000指标+象卦

象卦体系 v3.0 — 位置-速度-主力动向:
  爻1 = 位置: 趋势线(250日) >= 50
  爻2 = 速度: 趋势线20日变化 > 0
  爻3 = 主力动向: 主力线20日变化MA10 > 0
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import zipfile
from strategy.indicator import calc_trend_line, calc_main_force_line
from bagua_engine import calc_xiang_gua, BAGUA_TABLE


def prepare_zz1000():
    """预计算中证1000指标+象卦"""
    print("=" * 80)
    print("数据层 - 预计算中证1000指标+象卦 (v3.0)")
    print("=" * 80)

    # 加载中证1000数据
    print("\n加载中证1000数据...")
    with zipfile.ZipFile('E:/BaiduSyncdisk/A股数据_zip/指数/指数_日_kline.zip') as zf:
        names = [n for n in zf.namelist() if '000852' in n]
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, encoding='utf-8-sig')

    dates = df.iloc[:, 0].astype(str).values
    closes = df.iloc[:, 4].values.astype(float)   # 收盘价(第5列)
    highs = df.iloc[:, 5].values.astype(float)
    lows = df.iloc[:, 6].values.astype(float)

    print(f"数据范围: {dates[0]} ~ {dates[-1]}, 共{len(dates)}天")

    # 计算象卦
    print("\n计算象卦 (趋势线250 + 速度20 + 主力20日变化MA10)...")
    gua_list, trend, speed, main_force_dir = calc_xiang_gua(closes, highs, lows)

    # 主力线(供回测使用)
    main_force = calc_main_force_line(closes)

    # 组装DataFrame
    print("\n组装数据...")
    result = pd.DataFrame({
        'date': dates,
        'close': closes,
        'high': highs,
        'low': lows,
        'trend': trend,
        'main_force': main_force,
        'gua': gua_list,
    })

    # 扩展到 2005-01-04 源数据起点, 保留全部历史
    # (原 xiang_gua 250 日趋势需 warmup, 所以 gua 列在早期会为空字符串,
    #  下游统计/回测本来就按日期过滤, 无副作用)
    result = result[result['date'] >= '2005-01-04'].reset_index(drop=True)

    # 保存
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'zz1000_daily.csv')
    result.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n保存成功: {output_path}")
    print(f"  数据范围: {result['date'].iloc[0]} ~ {result['date'].iloc[-1]}")
    print(f"  总行数: {len(result)}")

    # 统计八卦分布
    print("\n象卦分布统计:")
    valid = result[result['gua'] != '']
    print(f"  有效{len(valid)}天 ({len(valid)/len(result)*100:.1f}%)")
    if len(valid) > 0:
        dist = valid['gua'].value_counts()
        for code in ['000', '001', '010', '011', '100', '101', '110', '111']:
            if code in dist.index:
                name = BAGUA_TABLE[code][0]
                print(f"    {name}({code}): {dist[code]}天 ({dist[code]/len(valid)*100:.1f}%)")

    # 卦变频率统计
    print("\n卦变频率:")
    if len(valid) > 1:
        changes = (valid['gua'] != valid['gua'].shift(1)).sum() - 1
        avg_days = len(valid) / max(changes, 1)
        print(f"  {changes}次卦变, 平均{avg_days:.0f}天/次")

    return result


if __name__ == '__main__':
    prepare_zz1000()
