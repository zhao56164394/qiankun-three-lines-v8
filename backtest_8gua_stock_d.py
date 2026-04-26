# -*- coding: utf-8 -*-
"""八卦分治 · 裸跑 + 个股层改用"个股日卦"过滤 组合实验

在 backtest_8gua_naked.py (裸跑, +2077%) 基础上, 把个股层过滤的"地卦 di_gua"
数据源替换成"个股日卦 stock_d_gua" (stock_daily_gua.csv, v10 公式与市场日卦同源).

两个 patch 叠加:
  1. naked: GUA_STRATEGY = derive_naked_cfg(...) (过滤白/黑名单清空 + sell=bear)
  2. stock_d: load_stock_bagua_map 返回个股日卦 map (字段兼容, 值换源)

注意 naked 下的 di_gua 白/黑名单已清空, 换源在"过滤规则"上无效,
但 di_gua 字段的覆盖范围不同 (地卦 ~3167 股 / 个股日卦 5096 股),
且 signals/trades 记录里 di_gua 值来源变了, 会影响:
  - Matrix 的 ren × di 分布 (dashboard 展示口径)
  - 少数因 "di_gua == '???'" 导致的信号缺失场景

输出: data_layer/data/backtest_8gua_naked_stock_d_result.json
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_8gua as b8
import experiment_gua as eg


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'data_layer', 'data')
FORMAL_RESULT_PATH = os.path.join(RESULT_DIR, 'backtest_8gua_result.json')
NAKED_STOCK_D_RESULT_PATH = os.path.join(RESULT_DIR, 'backtest_8gua_naked_stock_d_result.json')
FORMAL_BACKUP_PATH = FORMAL_RESULT_PATH + '.formal_backup'
STOCK_D_GUA_PATH = os.path.join(RESULT_DIR, 'foundation', 'stock_daily_gua.csv')


def load_stock_bagua_map_as_stock_d_gua() -> dict:
    """替换 load_stock_bagua_map, 返回同结构 map, di_gua 字段装个股日卦 gua_code."""
    print('[PATCH stock_d] 加载 stock_daily_gua.csv 替换 di_gua 源')
    df = pd.read_csv(STOCK_D_GUA_PATH, encoding='utf-8-sig',
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['code'] = df['code'].str.zfill(6)
    df['gua_code'] = df['gua_code'].str.zfill(3)
    print(f'  源数据: {len(df):,} 行, {df["code"].nunique()} 只股, '
          f'{df["date"].min()} ~ {df["date"].max()}')

    out = {}
    for row in df.itertuples(index=False):
        out[(row.date, row.code)] = {
            'di_gua': row.gua_code,
            'di_gua_name': row.gua_name,
            'stock_yao_1': row.pos,
            'stock_yao_2': row.spd,
            'stock_yao_3': row.acc,
            'stock_trend_value': row.trend,
            'stock_trend_anchor': pd.NA,
            'stock_speed_value': pd.NA,
            'stock_heat_momo': pd.NA,
            'stock_prev_gua': '',
            'stock_changed': pd.NA,
            'stock_seg_id': pd.NA,
            'stock_seg_day': pd.NA,
        }
    print(f'  构建映射 {len(out):,} 条 (date, code) 键')
    return out


def patch_strategy_to_naked():
    """裸跑 patch: 8 卦全部用 naked_cfg"""
    print('[PATCH naked] GUA_STRATEGY 切换为裸跑配置')
    for gua in sorted(b8.GUA_STRATEGY.keys()):
        b8.GUA_STRATEGY[gua] = eg.derive_naked_cfg(gua)


def patch_di_gua_to_stock_d():
    """个股卦 patch: 换 load_stock_bagua_map 为个股日卦版"""
    b8.load_stock_bagua_map = load_stock_bagua_map_as_stock_d_gua


def main():
    if os.path.exists(FORMAL_RESULT_PATH):
        os.replace(FORMAL_RESULT_PATH, FORMAL_BACKUP_PATH)
        print(f'[备份] formal result → {os.path.basename(FORMAL_BACKUP_PATH)}')

    try:
        patch_strategy_to_naked()
        patch_di_gua_to_stock_d()
        result, stats = b8.run()

        if os.path.exists(FORMAL_RESULT_PATH):
            os.replace(FORMAL_RESULT_PATH, NAKED_STOCK_D_RESULT_PATH)
            print(f'\n[SAVE] 裸跑+个股日卦结果 → {NAKED_STOCK_D_RESULT_PATH}')
    finally:
        if os.path.exists(FORMAL_BACKUP_PATH):
            os.replace(FORMAL_BACKUP_PATH, FORMAL_RESULT_PATH)
            print(f'[还原] formal result ← backup')


if __name__ == '__main__':
    main()
