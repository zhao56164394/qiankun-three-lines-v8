# -*- coding: utf-8 -*-
"""
数据层 - 中证1000 象卦数据访问模块

提供统一的API加载中证1000八卦数据，供回测、分析脚本调用。
数据源: data_layer/data/zz1000_daily.csv (由 prepare_zz1000.py 生成)

象卦体系 v3.0: 单层象卦 (位置-速度-主力动向)
  大盘象卦 + 个股象卦 — 统一参数(250/20/主力20MA10)

用法:
    from data_layer.gua_data import (
        load_zz1000_gua, get_current_gua,
        get_market_state, print_market_state,
        load_zz1000_with_segments, get_gua_segments,
        get_buy_filter, print_buy_filter,
    )
"""
import os
import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_ZZ1000_PATH = os.path.join(_DATA_DIR, 'zz1000_daily.csv')

# ============================================================
# 八卦基础定义 (避免循环依赖, 此处内联)
# ============================================================
BAGUA_TABLE = {
    '000': ('坤', '☷', '深熊探底', '阴'),
    '001': ('艮', '☶', '底部吸筹', '阴'),
    '010': ('坎', '☵', '反弹乏力', '阴'),
    '011': ('巽', '☴', '底部爆发', '阳'),
    '100': ('震', '☳', '高位出货', '阴'),
    '101': ('离', '☲', '高位护盘', '阳'),
    '110': ('兑', '☱', '牛末减仓', '阳'),
    '111': ('乾', '☰', '疯牛主升', '阳'),
}

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']


def _clean_code(code):
    """清洗卦编码: 处理 '101.0' / 101 / '101' 等各种格式"""
    s = str(code).strip()
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(3)


def gua_name(code):
    """'111' -> '乾'"""
    code = _clean_code(code)
    return BAGUA_TABLE.get(code, ('?',))[0]


def gua_label(code):
    """'111' -> '乾☰(疯牛主升)'"""
    code = _clean_code(code)
    info = BAGUA_TABLE.get(code)
    if info:
        return f"{info[0]}{info[1]}({info[2]})"
    return '?'


def gua_yinyang(code):
    """'111' -> '阳'"""
    code = _clean_code(code)
    return BAGUA_TABLE.get(code, ('?', '?', '?', '?'))[3]


def is_yang(code):
    return gua_yinyang(code) == '阳'


# ============================================================
# 数据加载
# ============================================================
_cache = {}


def load_zz1000_gua(force_reload=False):
    """
    加载中证1000象卦数据

    Returns:
        pd.DataFrame: 包含 date, close, trend, main_force, gua 等列
    """
    if 'zz1000' in _cache and not force_reload:
        return _cache['zz1000']

    if not os.path.exists(_ZZ1000_PATH):
        raise FileNotFoundError(
            f"数据文件不存在: {_ZZ1000_PATH}\n"
            f"请先运行: python data_layer/prepare_zz1000.py"
        )

    df = pd.read_csv(_ZZ1000_PATH, encoding='utf-8-sig')
    df['date'] = df['date'].astype(str)
    df['gua'] = df['gua'].apply(_clean_code)

    _cache['zz1000'] = df
    return df


def get_current_gua(date=None):
    """
    获取指定日期(或最新)的象卦

    Args:
        date: 日期字符串如 '2026-03-26', 为None则取最新

    Returns:
        dict: {
            'date': str, 'close': float,
            'gua': str, 'gua_name': str, 'gua_yy': str,
        }
    """
    df = load_zz1000_gua()
    if date is None:
        row = df.iloc[-1]
    else:
        match = df[df['date'] == str(date)]
        if len(match) == 0:
            raise ValueError(f"找不到日期: {date}")
        row = match.iloc[-1]

    g = row['gua']
    return {
        'date': row['date'],
        'close': row['close'],
        'gua': g,
        'gua_name': gua_label(g),
        'gua_yy': gua_yinyang(g),
    }


# ============================================================
# 市场状态查询
# ============================================================

# 各卦的市场状态描述和操作建议
_GUA_STATE = {
    '000': {'状态': '深熊探底',     '特征': '低位+下降+主力撤退, 多杀多阶段',
            '操作': '空仓等待',     '转机': '→艮(主力进场)或→坎(开始反弹)'},
    '001': {'状态': '底部吸筹',     '特征': '低位+下降+主力进场, 机构悄悄买入',
            '操作': '观察等待',     '转机': '→巽(方向转升=反转信号)'},
    '010': {'状态': '反弹乏力',     '特征': '低位+上升+主力撤退, 无主力支撑的反弹',
            '操作': '谨慎观望',     '转机': '→巽(主力加入)或→坤(反弹失败)'},
    '011': {'状态': '底部爆发',     '特征': '低位+上升+主力进场, 真正的底部反转',
            '操作': '积极布局',     '转机': '→乾(进入牛市主升)'},
    '100': {'状态': '高位出货',     '特征': '高位+下降+主力撤退, 主力撤退散户踩踏',
            '操作': '减仓止损',     '转机': '→离(主力护盘)或→坤(跌入低位)'},
    '101': {'状态': '高位护盘',     '特征': '高位+下降+主力进场, 有主力托底的回调',
            '操作': '持仓观察',     '转机': '→乾(调整结束)或→震(主力撤退)'},
    '110': {'状态': '牛末减仓',     '特征': '高位+上升+主力撤退, 缺乏主力的拉升',
            '操作': '逢高减仓',     '转机': '→震(散户狂欢结束)'},
    '111': {'状态': '疯牛主升',     '特征': '高位+上升+主力进场, 三线共振全面看多',
            '操作': '持仓待涨',     '转机': '→兑(主力撤退)或→离(方向转跌)'},
}


def get_market_state(date=None):
    """
    获取指定日期的市场状态描述

    Args:
        date: 日期字符串如 '2026-03-26', 为None则取最新

    Returns:
        dict: {
            'date': str, 'close': float,
            'gua': str, 'gua_name': str, 'gua_yy': str,
            'state': str, 'feature': str, 'advice': str, 'transition': str,
        }
    """
    curr = get_current_gua(date)
    gua_code = _clean_code(curr['gua'])
    state = _GUA_STATE[gua_code]

    return {
        'date': curr['date'],
        'close': curr['close'],
        'gua': gua_code,
        'gua_name': curr['gua_name'],
        'gua_yy': curr['gua_yy'],
        'state': state['状态'],
        'feature': state['特征'],
        'advice': state['操作'],
        'transition': state['转机'],
    }


def print_market_state(date=None):
    """打印格式化的市场状态报告"""
    s = get_market_state(date)

    print(f"\n{'═' * 60}")
    print(f"  市场状态报告  {s['date']}  收盘{s['close']:.2f}")
    print(f"{'═' * 60}")
    print(f"  象卦: {s['gua_name']} ({s['gua_yy']})")
    print(f"  状态: {s['state']} — {s['feature']}")
    print(f"  建议: {s['advice']}")
    print(f"  转机: {s['transition']}")
    print(f"{'═' * 60}")


# ============================================================
# 带连续段标记的每日数据
# ============================================================
def load_zz1000_with_segments(force_reload=False):
    """
    加载每日数据并标记象卦的连续段

    在原始 zz1000_daily 基础上增加:
      - gua_seg: 段ID (连续相同卦象为一段)
      - gua_seg_day: 当前是该段的第几天
      - daily_ret: 每日涨跌幅(%)

    Returns:
        pd.DataFrame
    """
    if 'zz1000_seg' in _cache and not force_reload:
        return _cache['zz1000_seg']

    df = load_zz1000_gua().copy()
    df['daily_ret'] = df['close'].pct_change() * 100
    df['gua_seg'] = (df['gua'] != df['gua'].shift(1)).cumsum()
    df['gua_seg_day'] = df.groupby('gua_seg').cumcount() + 1

    _cache['zz1000_seg'] = df
    return df


def get_gua_segments():
    """
    获取象卦连续段列表

    Returns:
        pd.DataFrame: 每行一个连续段, 包含:
            gua, gua_name, start, end, n_days, start_close, end_close, seg_ret
    """
    df = load_zz1000_with_segments()

    segs = []
    for seg_id, grp in df.groupby('gua_seg'):
        gua = grp['gua'].iloc[0]
        segs.append({
            'gua': gua,
            'gua_name': gua_label(gua),
            'start': grp['date'].iloc[0],
            'end': grp['date'].iloc[-1],
            'n_days': len(grp),
            'start_close': grp['close'].iloc[0],
            'end_close': grp['close'].iloc[-1],
            'seg_ret': round((grp['close'].iloc[-1] / grp['close'].iloc[0] - 1) * 100, 2),
        })

    return pd.DataFrame(segs)


# ============================================================
# 个股买入信号过滤
# ============================================================
def get_buy_filter(date=None):
    """
    获取指定日期的个股买入信号过滤建议

    基于中证象卦状态, 判断当天信号是否值得执行。

    逻辑:
      - 爻1=阴(低位)的卦: 坤/艮/坎/巽 → 可以买入
      - 爻1=阳(高位)的卦: 震/离/兑/乾 → 需要分卦策略

    Args:
        date: 日期字符串, 为None则取最新

    Returns:
        dict: {
            'date': str, 'close': float,
            'gua': str, 'gua_name': str, 'gua_yy': str,
            'can_buy': bool, 'reason': str,
        }
    """
    curr = get_current_gua(date)
    gua_code = _clean_code(curr['gua'])
    yy = gua_yinyang(gua_code)

    # 爻1=阴(低位)→ 可以买入
    can_buy = (gua_code[0] == '0')

    if can_buy:
        reason = f'象卦={gua_label(gua_code)}, 低位阶段, 超跌信号有效'
    else:
        reason = f'象卦={gua_label(gua_code)}, 高位阶段, 需要分卦策略判断'

    return {
        'date': curr['date'],
        'close': curr['close'],
        'gua': gua_code,
        'gua_name': curr['gua_name'],
        'gua_yy': yy,
        'can_buy': can_buy,
        'reason': reason,
    }


def print_buy_filter(date=None):
    """打印格式化的买入过滤报告"""
    f = get_buy_filter(date)

    print(f"\n{'═' * 60}")
    print(f"  个股买入信号过滤  {f['date']}  收盘{f['close']:.2f}")
    print(f"{'═' * 60}")
    print(f"  象卦: {f['gua_name']} ({f['gua_yy']})")
    print(f"{'─' * 60}")
    if f['can_buy']:
        print(f"  可以买入")
    else:
        print(f"  需要分卦策略判断")
    print(f"  理由: {f['reason']}")
    print(f"{'═' * 60}")
