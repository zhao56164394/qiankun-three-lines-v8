# -*- coding: utf-8 -*-
"""
feeds.py — 自定义 Backtrader 数据源

1. StockData:   个股数据 (扩展 trend, retail, main_force)
2. ZZ1000Data:  中证1000 (扩展 trend, retail, main_force, ma30, ma120)
"""
import pandas as pd
import numpy as np
import backtrader as bt
from .config import STOCKS_DIR, ZZ1000_PATH


class StockData(bt.feeds.PandasData):
    """个股数据源 — 扩展三线指标 + 卦列

    CSV 列: date, open, close, high, low, trend, retail, main_force,
            year_gua, month_gua, day_gua
    卦列以 int 存储 (0~111), 策略中用 str(int).zfill(3) 还原
    """
    lines = ('trend', 'retail', 'main_force',
             'year_gua', 'month_gua', 'day_gua',)

    params = (
        ('datetime', None),     # 使用 index
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', -1),         # 个股CSV没有volume列
        ('openinterest', -1),
        ('trend', 'trend'),
        ('retail', 'retail'),
        ('main_force', 'main_force'),
        ('year_gua', 'year_gua'),
        ('month_gua', 'month_gua'),
        ('day_gua', 'day_gua'),
    )


class ZZ1000Data(bt.feeds.PandasData):
    """中证1000数据源 — 扩展趋势线 + 主力线 + 大象卦

    CSV 列: date, close, high, low, trend, main_force, year_gua, ...
    注意: zz1000 没有 open 列, 用 close 代替
    """
    lines = ('trend', 'retail', 'main_force', 'year_gua',)

    params = (
        ('datetime', None),     # 使用 index
        ('open', 'close'),      # 没有 open 列, 用 close 代替
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', -1),
        ('openinterest', -1),
        ('trend', 'trend'),
        ('retail', -1),         # zz1000 CSV 可能没有 retail
        ('main_force', 'main_force'),
        ('year_gua', 'year_gua'),
    )


def load_stock_df(code, start_date=None, end_date=None):
    """加载个股 DataFrame, 处理 NaN, 供 StockData 使用

    Args:
        code: 股票代码 (如 '002460')
        start_date: 起始日期 (str), 可选
        end_date: 结束日期 (str), 可选

    Returns:
        pandas DataFrame (index=DatetimeIndex)
    """
    import os
    fpath = os.path.join(STOCKS_DIR, f'{code}.csv')
    if not os.path.exists(fpath):
        return None

    # 读取所有需要的列
    df = pd.read_csv(fpath, encoding='utf-8-sig',
                     usecols=['date', 'open', 'close', 'high', 'low',
                              'trend', 'retail', 'main_force',
                              'year_gua', 'month_gua', 'day_gua'])

    df['date'] = pd.to_datetime(df['date'])

    # 日期过滤
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        return None

    # NaN 处理: 指标前向填充, 价格行保留(backtrader 会跳过)
    for col in ['trend', 'retail', 'main_force']:
        df[col] = df[col].ffill()

    # 卦列: 字符串→整数 (backtrader lines 只支持数值)
    # "000"→0, "101"→101, "111"→111
    for col in ['year_gua', 'month_gua', 'day_gua']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # 设置 DatetimeIndex (Backtrader 要求)
    df = df.set_index('date')
    df = df.sort_index()

    return df


def load_zz1000_df(start_date=None, end_date=None):
    """加载中证1000 DataFrame, 供 ZZ1000Data 使用

    Returns:
        pandas DataFrame (index=DatetimeIndex)
    """
    import os
    if not os.path.exists(ZZ1000_PATH):
        raise FileNotFoundError(f"中证1000数据文件不存在: {ZZ1000_PATH}")

    df = pd.read_csv(ZZ1000_PATH, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])

    # 日期过滤
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    # 确保有 high/low 列 (有些中证1000数据可能缺失)
    if 'high' not in df.columns:
        df['high'] = df['close']
    if 'low' not in df.columns:
        df['low'] = df['close']

    # NaN 处理
    for col in ['trend', 'main_force']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # year_gua: 字符串→整数 (backtrader lines 只支持数值)
    # "000"→0, "101"→101, "111"→111
    if 'year_gua' in df.columns:
        df['year_gua'] = pd.to_numeric(df['year_gua'], errors='coerce').fillna(-1).astype(int)
    else:
        df['year_gua'] = -1

    # 设置 DatetimeIndex
    df = df.set_index('date')
    df = df.sort_index()

    return df


def create_stock_feed(code, start_date=None, end_date=None):
    """创建个股数据 feed 实例"""
    df = load_stock_df(code, start_date, end_date)
    if df is None:
        return None
    feed = StockData(dataname=df, name=code)
    return feed


def create_zz1000_feed(start_date=None, end_date=None, zz_df=None):
    """创建中证1000数据 feed 实例

    Args:
        start_date: 起始日期
        end_date: 结束日期
        zz_df: 已加载的 DataFrame, 为 None 时自动加载 (避免重复读CSV)
    """
    if zz_df is None:
        zz_df = load_zz1000_df(start_date, end_date)
    feed = ZZ1000Data(dataname=zz_df, name='zz1000')
    return feed
