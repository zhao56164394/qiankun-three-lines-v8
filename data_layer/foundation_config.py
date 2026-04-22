# -*- coding: utf-8 -*-
"""
foundation_config.py

新底座配置中心：
- 统一原始数据路径
- 统一输出路径
- 统一字段命名
- 统一主板样本口径
"""
import os
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LAYER_ROOT = os.path.join(PROJECT_ROOT, 'data_layer')
FOUNDATION_DATA_DIR = os.path.join(DATA_LAYER_ROOT, 'data', 'foundation')
STOCKS_DATA_DIR = os.path.join(DATA_LAYER_ROOT, 'data', 'stocks')

# 原始数据根目录
BAIDU_ROOT = 'E:/BaiduSyncdisk'
LEGACY_ROOT = 'e:/A'
LEGACY_ZIP_ROOT = os.path.join(BAIDU_ROOT, 'A股数据_zip')

# 多源路径
PATHS = {
    'stock_basic': os.path.join(BAIDU_ROOT, 'A股数据_每日指标', '股票列表.csv'),
    'stock_daily_metrics_root': os.path.join(BAIDU_ROOT, 'A股数据_每日指标', '增量数据', '每日指标'),
    'stock_moneyflow_root': os.path.join(BAIDU_ROOT, 'A股数据_资金流向', '资金流向_每日更新', '个股资金流向'),
    'stock_moneyflow_year_root': os.path.join(BAIDU_ROOT, 'A股数据_资金流向', '个股资金流向_按年汇总'),
    'chip_root': os.path.join(BAIDU_ROOT, 'A股筹码数据', '每日筹码及胜率'),
    'limit_up_summary': os.path.join(BAIDU_ROOT, '榜单数据', '每日涨停家数去ST_2018至今.csv'),
    'limit_down_summary': os.path.join(BAIDU_ROOT, '榜单数据', '每日跌停家数去ST_2018至今.csv'),
    'limit_up_detail_root': os.path.join(BAIDU_ROOT, '榜单数据', '榜单数据', '涨停榜单_按年汇总'),
    'limit_broken_board_summary': os.path.join(BAIDU_ROOT, '榜单数据', '汇总_炸板_2023_2025.csv'),
    'limit_ladder_root': os.path.join(BAIDU_ROOT, '榜单数据', '连板天梯'),
    'index_daily_root': os.path.join(BAIDU_ROOT, '指数数据', '增量数据', '大盘指数每日指标'),
    'index_daily_zip': os.path.join(BAIDU_ROOT, '指数数据', '指数日线行情.zip'),
    'index_daily_zip_legacy': os.path.join(BAIDU_ROOT, 'A股数据_zip', '指数', '指数_日_kline.zip'),
    'index_basic_csi': os.path.join(BAIDU_ROOT, '指数数据', '指数基本信息_中证指数.csv'),
    'industry_component_root': os.path.join(BAIDU_ROOT, '行业概念板块', '板块成分_同花顺', '行业板块成分汇总_同花顺'),
    'concept_component_root': os.path.join(BAIDU_ROOT, '行业概念板块', '板块成分_同花顺', '概念板块成分汇总_同花顺'),
    'industry_component_full': os.path.join(BAIDU_ROOT, '行业概念板块', '行业板块成分汇总_同花顺.csv'),
    'concept_component_full': os.path.join(BAIDU_ROOT, '行业概念板块', '概念板块成分汇总_同花顺.csv'),
}

# 主板样本口径
UNIVERSE_CONFIG = {
    'allowed_market_types': ['主板'],
    'allowed_exchanges': ['SSE', 'SZSE'],
    'exclude_st': False,
    'min_list_days': 120,
    'exclude_prefixes': ['30', '68', '8', '4'],  # 创业板/科创板/北交所常见前缀
    'history_start_date': '2014-01-01',
}

# 第一轮核心指数映射
CORE_INDEX_CODES = {
    '000001.SH': 'sh_close',
    '399001.SZ': 'sz_close',
    '000300.SH': 'hs300_close',
    '000905.SH': 'csi500_close',
    '000852.SH': 'csi1000_close',
    '000985.CSI': 'allA_close',
}

CORE_INDEX_ALIASES = {
    '上证综指': 'sh_close',
    '深证成指': 'sz_close',
    '沪深300': 'hs300_close',
    '中证500': 'csi500_close',
    '中证1000': 'csi1000_close',
    '中证全指': 'allA_close',
}

# 字段映射
STOCK_BASIC_RENAME = {
    'TS代码': 'ts_code',
    '股票代码': 'code',
    '股票名称': 'name',
    '所属行业': 'industry_name',
    '市场类型': 'board',
    '交易所代码': 'exchange',
    '上市日期': 'list_date',
}

DAILY_METRICS_RENAME = {
    '股票代码': 'code',
    '交易日期': 'date',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量(手)': 'volume',
    '成交额(千元)': 'amount_k',
    '换手率': 'turnover_rate',
    '总市值(万元)': 'total_mv_wan',
    '流通市值(万元)': 'circ_mv_wan',
    '市盈率': 'pe',
    '市盈率TTM': 'pe_ttm',
    '市净率': 'pb',
}

MONEYFLOW_RENAME = {
    '股票代码': 'code',
    '交易日期': 'date',
    '小单买入金额(万元)': 'small_buy',
    '小单卖出金额(万元)': 'small_sell',
    '中单买入金额(万元)': 'medium_buy',
    '中单卖出金额(万元)': 'medium_sell',
    '大单买入金额(万元)': 'large_buy',
    '大单卖出金额(万元)': 'large_sell',
    '特大单买入金额(万元)': 'super_large_buy',
    '特大单卖出金额(万元)': 'super_large_sell',
    '净流入金额(万元)': 'net_inflow',
}

CHIP_RENAME = {
    '股票代码': 'code',
    '交易日期': 'date',
    '5分位成本': 'cost_5',
    '15分位成本': 'cost_15',
    '50分位成本': 'cost_50',
    '85分位成本': 'cost_85',
    '95分位成本': 'cost_95',
    '加权平均成本': 'avg_cost',
    '胜率': 'winner_ratio',
}

INDEX_DAILY_RENAME = {
    '指数代码': 'index_code',
    '交易日期': 'date',
    '换手率': 'turnover_rate',
    '市盈率': 'pe',
    '市盈率TTM': 'pe_ttm',
    '市净率': 'pb',
}

NUMERIC_COLUMNS = {
    'daily_metrics': ['open', 'high', 'low', 'close', 'volume', 'amount_k', 'turnover_rate', 'total_mv_wan', 'circ_mv_wan', 'pe', 'pe_ttm', 'pb'],
    'moneyflow': ['small_buy', 'small_sell', 'medium_buy', 'medium_sell', 'large_buy', 'large_sell', 'super_large_buy', 'super_large_sell', 'net_inflow'],
    'chip': ['cost_5', 'cost_15', 'cost_50', 'cost_85', 'cost_95', 'avg_cost', 'winner_ratio'],
    'index_daily': ['turnover_rate', 'pe', 'pe_ttm', 'pb'],
}

DEFAULT_COLUMNS = {
    'universe': ['date', 'code', 'name', 'exchange', 'board', 'is_st', 'list_date', 'industry_name', 'in_universe'],
    'cross_section': [
        'date', 'code', 'name', 'exchange', 'board', 'industry_name',
        'open', 'high', 'low', 'close',
        'amount', 'turnover_rate', 'total_mv', 'circ_mv', 'pe', 'pb',
        'small_net', 'large_net', 'super_large_net',
        'cost_50', 'cost_85', 'avg_cost', 'winner_ratio',
        'is_zt', 'is_dt', 'is_zb', 'lb_count', 'zt_count', 'dt_count', 'zb_count', 'concept_count',
        'above_ma5_ratio', 'above_ma10_ratio', 'above_ma20_ratio', 'above_ma60_ratio',
        'new_high_20_ratio', 'new_low_20_ratio',
        'allA_close', 'hs300_close', 'csi500_close', 'csi1000_close', 'sh_close', 'sz_close',
    ],
}


def ensure_foundation_data_dir() -> str:
    os.makedirs(FOUNDATION_DATA_DIR, exist_ok=True)
    return FOUNDATION_DATA_DIR


def foundation_file(name: str) -> str:
    ensure_foundation_data_dir()
    return os.path.join(FOUNDATION_DATA_DIR, name)


def get_required_source_paths() -> List[str]:
    return [
        PATHS['stock_basic'],
        PATHS['stock_daily_metrics_root'],
        PATHS['chip_root'],
        PATHS['index_daily_root'],
    ]


def path_exists_map() -> Dict[str, bool]:
    return {k: os.path.exists(v) for k, v in PATHS.items()}
