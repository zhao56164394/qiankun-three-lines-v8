# -*- coding: utf-8 -*-
"""
数据层 - 预计算全市场个股指标+象卦

象卦体系 v5.1 — 个股独立参数(150/20/10):
  爻1 = 位置: 趋势线(150日) >= 50
  爻2 = 速度: 趋势线20日变化 > 0
  爻3 = 主力动向: 主力线20日变化MA10 > 0
  注: 中证1000仍用250/20/10, 个股用150更早识别趋势转折
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import zipfile
from strategy.indicator import calc_trend_line, calc_retail_line, calc_main_force_line
from bagua_engine import calc_xiang_gua


def process_one_stock(code, zf):
    """处理单只股票"""
    fname = f'{code}_daily_qfq.csv'
    if fname not in zf.namelist():
        return None

    try:
        with zf.open(fname) as f:
            df = pd.read_csv(f, encoding='utf-8-sig', dtype={'日期': str})

        # 只保留2014-06-01之后的数据
        mask = df['日期'].values >= '2014-06-01'
        if mask.sum() < 60:
            return None

        df = df[mask].reset_index(drop=True)

        opens = df['开盘'].values.astype(float)
        closes = df['收盘'].values.astype(float)
        highs = df['最高'].values.astype(float)
        lows = df['最低'].values.astype(float)

        if (closes <= 0).any():
            return None

        # 计算三线指标
        trend = calc_trend_line(closes, highs, lows)
        retail = calc_retail_line(closes)
        main_force = calc_main_force_line(closes)

        # 计算象卦 (个股参数: 150/20/主力20MA10, 比中证250更早识别转折)
        gua_list, _, _, _ = calc_xiang_gua(closes, highs, lows, trend_period=150)

        # 组装结果
        result = pd.DataFrame({
            'date': df['日期'].values,
            'open': opens,
            'close': closes,
            'high': highs,
            'low': lows,
            'trend': trend,
            'retail': retail,
            'main_force': main_force,
            'gua': gua_list,
        })

        return result

    except Exception as e:
        print(f"  错误 {code}: {e}")
        return None


def prepare_stocks():
    """预计算全市场个股指标+象卦"""
    print("=" * 80)
    print("数据层 - 预计算全市场个股指标+象卦 (v3.0)")
    print("=" * 80)

    # 加载股票列表
    df_list = pd.read_csv('E:/BaiduSyncdisk/A股数据_zip/股票列表.csv', encoding='utf-8-sig', dtype=str)
    active_codes = set(df_list['股票代码'].values)

    # 获取所有股票代码
    with zipfile.ZipFile('E:/BaiduSyncdisk/A股数据_zip/daily_qfq.zip') as zf:
        all_files = zf.namelist()

    all_codes = [f.replace('_daily_qfq.csv', '') for f in all_files
                 if f.endswith('_daily_qfq.csv') and f.startswith(('00', '60', '30', '68'))]
    all_codes = [c for c in all_codes if c in active_codes]

    print(f"\n待处理股票: {len(all_codes)}只")

    # 批量处理
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'stocks')
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    skip_count = 0

    with zipfile.ZipFile('E:/BaiduSyncdisk/A股数据_zip/daily_qfq.zip') as zf:
        for i, code in enumerate(all_codes):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(all_codes)} ({success_count}成功, {skip_count}跳过)")

            result = process_one_stock(code, zf)

            if result is None:
                skip_count += 1
                continue

            # 保存
            output_path = os.path.join(output_dir, f'{code}.csv')
            result.to_csv(output_path, index=False, encoding='utf-8-sig')
            success_count += 1

    print(f"\n完成!")
    print(f"  成功: {success_count}只")
    print(f"  跳过: {skip_count}只")
    print(f"  输出目录: {output_dir}")


if __name__ == '__main__':
    prepare_stocks()
