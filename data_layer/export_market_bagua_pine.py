# -*- coding: utf-8 -*-
"""
export_market_bagua_pine.py

将预计算好的 market_bagua_daily.csv 导出为可直接粘贴到 TradingView 的 Pine Script。
Pine 端只负责显示，不重算市场卦逻辑，避免与本地数据链路漂移。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.foundation_config import foundation_file
from data_layer.foundation_data import load_market_bagua


OUTPUT_FILENAME = 'market_bagua_tradingview_overlay.pine'

GUA_NAME_MAP = {
    0: '坤',
    1: '艮',
    10: '坎',
    11: '巽',
    100: '震',
    101: '离',
    110: '兑',
    111: '乾',
}

GUA_COLOR_MAP = {
    0: 'color.rgb(34, 197, 94)',
    1: 'color.rgb(134, 239, 172)',
    10: 'color.rgb(74, 222, 128)',
    11: 'color.rgb(245, 158, 11)',
    100: 'color.rgb(239, 68, 68)',
    101: 'color.rgb(251, 146, 60)',
    110: 'color.rgb(167, 139, 250)',
    111: 'color.rgb(220, 38, 38)',
}


def _load_export_base() -> pd.DataFrame:
    df = load_market_bagua(force_reload=True).copy()
    if df.empty:
        raise ValueError('market_bagua_daily.csv 为空，无法导出 TradingView 脚本')

    required_cols = ['date', 'gua_code', 'gua_name', 'changed']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f'market_bagua_daily.csv 缺少字段: {missing}')

    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df['gua_code'] = df['gua_code'].astype(str).str.zfill(3)
    df['changed'] = pd.to_numeric(df['changed'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    df['ymd'] = df['date'].dt.strftime('%Y%m%d').astype(int)
    df['gua_int'] = df['gua_code'].astype(int)
    return df[['date', 'ymd', 'gua_code', 'gua_int', 'gua_name', 'changed']].copy()


def _build_put_lines(df: pd.DataFrame) -> list[str]:
    lines = []
    for row in df.itertuples(index=False):
        lines.append(f'    guaMap.put({row.ymd}, {row.gua_int})')
        lines.append(f'    changedMap.put({row.ymd}, {row.changed})')
    return lines


def build_market_bagua_pine() -> str:
    df = _load_export_base()
    put_lines = '\n'.join(_build_put_lines(df))
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')
    sample_count = len(df)

    script = f'''//@version=6
indicator("市场卦可视化(预计算导入)", shorttitle="市场卦TV", overlay=true, max_labels_count=500)

// 本脚本由 data_layer/export_market_bagua_pine.py 自动生成
// 数据区间: {start_date} ~ {end_date}
// 交易日数: {sample_count}
// 使用说明:
// 1. 挂到 A 股日线图上使用
// 2. Pine 端只负责显示，不重算市场卦
// 3. 背景色=市场卦，标签=变卦点

show_bg = input.bool(true, "显示卦背景")
show_change_labels = input.bool(true, "显示变卦标签")
bg_alpha = input.int(84, "背景透明度", minval=0, maxval=100)

var map<int, int> guaMap = map.new<int, int>()
var map<int, int> changedMap = map.new<int, int>()

if barstate.isfirst
{put_lines}

f_gua_name(int gua) =>
    switch gua
        0 => "坤"
        1 => "艮"
        10 => "坎"
        11 => "巽"
        100 => "震"
        101 => "离"
        110 => "兑"
        111 => "乾"
        => "未知"

f_gua_code(int gua) =>
    switch gua
        0 => "000"
        1 => "001"
        10 => "010"
        11 => "011"
        100 => "100"
        101 => "101"
        110 => "110"
        111 => "111"
        => "---"

f_gua_color(int gua) =>
    switch gua
        0 => {GUA_COLOR_MAP[0]}
        1 => {GUA_COLOR_MAP[1]}
        10 => {GUA_COLOR_MAP[10]}
        11 => {GUA_COLOR_MAP[11]}
        100 => {GUA_COLOR_MAP[100]}
        101 => {GUA_COLOR_MAP[101]}
        110 => {GUA_COLOR_MAP[110]}
        111 => {GUA_COLOR_MAP[111]}
        => color.gray

currYmd = year * 10000 + month * 100 + dayofmonth
currGua = map.get(guaMap, currYmd)
currChanged = map.get(changedMap, currYmd)

hasGua = not na(currGua)
guaColor = hasGua ? f_gua_color(currGua) : na
guaName = hasGua ? f_gua_name(currGua) : ""
guaCode = hasGua ? f_gua_code(currGua) : ""

bgcolor(show_bg and hasGua ? color.new(guaColor, bg_alpha) : na, title="市场卦背景")

if show_change_labels and currChanged == 1 and hasGua and barstate.isconfirmed
    label.new(
         bar_index,
         high,
         "变卦\\n" + guaName + " " + guaCode,
         style=label.style_label_down,
         color=color.new(guaColor, 0),
         textcolor=color.white,
         size=size.tiny)

plotchar(hasGua ? currGua : na, title="市场卦数值", char="", location=location.top, display=display.data_window)
'''
    return script


def export_market_bagua_pine() -> str:
    script = build_market_bagua_pine()
    output_path = foundation_file(OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)

    df = _load_export_base()
    print('=' * 80)
    print('TradingView Pine 导出完成')
    print('=' * 80)
    print(f'日期范围: {df["date"].min().strftime("%Y-%m-%d")} ~ {df["date"].max().strftime("%Y-%m-%d")}')
    print(f'交易日数: {len(df)}')
    print(f'输出文件: {output_path}')
    return output_path


if __name__ == '__main__':
    export_market_bagua_pine()
