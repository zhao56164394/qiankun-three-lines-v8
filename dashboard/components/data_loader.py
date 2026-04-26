# -*- coding: utf-8 -*-
"""
统一数据加载层 — 消除各页面重复代码, 加缓存

v4.0: 优先加载八卦分治结果, 兼容联合策略结果
"""
import ast
import copy
import os
import sys
import json
import streamlit as st
import pandas as pd

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from data_layer.foundation_data import load_daily_cross_section, load_market_bagua
from data_layer.gua_data import GUA_DISPLAY_NAMES as GUA_NAMES, GUA_ORDER, compat_rename_columns
import experiment_gua as eg
import backtest_8gua as b8
import backtest_baseline as bb

# 关键路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_layer', 'data')
BT_8GUA_FORMAL_PATH = os.path.join(DATA_DIR, 'backtest_8gua_formal_result.json')
BT_8GUA_PATH = os.path.join(DATA_DIR, 'backtest_8gua_result.json')
BT_PATH = os.path.join(DATA_DIR, 'backtest_result.json')
STOCKS_DIR = os.path.join(DATA_DIR, 'stocks')
SNAP_PATH = os.path.join(PROJECT_ROOT, 'live', 'snapshots', 'latest.json')
BAGUA_DEBUG_BASELINE_SNAPSHOT_PATH = os.path.join(DATA_DIR, 'bagua_debug_baseline_snapshot.json')

TIAN_GUA_NAMES = {k: '天' + v for k, v in GUA_NAMES.items()}
REN_GUA_NAMES = {k: '人' + v for k, v in GUA_NAMES.items()}
DI_GUA_NAMES = {k: '地' + v for k, v in GUA_NAMES.items()}
GUA_COLORS = {
    '000': '#22c55e', '001': '#86efac', '010': '#4ade80', '011': '#f59e0b',
    '100': '#ef4444', '101': '#fb923c', '110': '#a78bfa', '111': '#ef4444',
}
GUA_MEANINGS = {
    '000': '至暗时刻', '001': '底部蓄力', '010': '反弹无力', '011': '风起云涌',
    '100': '雷霆坠落', '101': '主力护盘', '110': '散户狂欢', '111': '如日中天',
}



def _load_regime_visual(load_fn, numeric_cols, start_date=None, end_date=None):
    df = load_fn().copy()
    if len(df) == 0:
        return df

    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if start_date is not None:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def _best_bt_path():
    """优先正式固化八卦分治结果，其次当前八卦结果，再次联合策略。"""
    for path, label in [
        (BT_8GUA_FORMAL_PATH, '八卦分治(正式)'),
        (BT_8GUA_PATH, '八卦分治'),
        (BT_PATH, '联合策略'),
    ]:
        if os.path.exists(path):
            return path, label
    return None, None


@st.cache_data(ttl=300)
def load_backtest(source=None):
    """加载回测结果 JSON, 返回 (meta, df_equity, df_trades, yearly)

    source: '8gua' / 'hybrid' / None(自动选最优)
    """
    if source == '8gua':
        path = BT_8GUA_FORMAL_PATH if os.path.exists(BT_8GUA_FORMAL_PATH) else BT_8GUA_PATH
    elif source == 'hybrid':
        path = BT_PATH
    else:
        path, _ = _best_bt_path()

    if path is None or not os.path.exists(path):
        return None, None, None, None

    with open(path, 'r', encoding='utf-8') as f:
        bt = json.load(f)

    meta = bt['meta']

    # 净值DataFrame
    df_eq = pd.DataFrame(bt['daily_equity'])
    df_eq['date'] = pd.to_datetime(df_eq['date'], format='mixed')
    init_cap = meta['init_capital']
    df_eq['nav'] = df_eq['total_equity'] / init_cap
    df_eq['peak'] = df_eq['total_equity'].cummax()
    df_eq['drawdown'] = (df_eq['peak'] - df_eq['total_equity']) / df_eq['peak'] * 100

    # 交易DataFrame
    df_trades = pd.DataFrame(bt['trade_log'])
    if len(df_trades) > 0:
        df_trades['buy_date'] = pd.to_datetime(df_trades['buy_date'], format='mixed')
        df_trades['sell_date'] = pd.to_datetime(df_trades['sell_date'], format='mixed')
        # 兼容: 八卦分治有 gua/sell_method, 联合策略有 mode
        if 'gua' not in df_trades.columns:
            df_trades['gua'] = '-'
        if 'sell_method' not in df_trades.columns:
            df_trades['sell_method'] = '-'

    yearly = bt.get('yearly', {})

    return meta, df_eq, df_trades, yearly


@st.cache_data(ttl=300)
def load_backtest_8gua_extra():
    """加载八卦分治的额外信息: gua_strategy 配置"""
    path = BT_8GUA_FORMAL_PATH if os.path.exists(BT_8GUA_FORMAL_PATH) else BT_8GUA_PATH
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        bt = json.load(f)
    return bt.get('gua_strategy', None)


@st.cache_data(ttl=300)
def load_stock_ohlc(code, start_date=None, end_date=None):
    """加载单只股票的 OHLC + 指标数据"""
    fpath = os.path.join(STOCKS_DIR, f'{code}.csv')
    if not os.path.exists(fpath):
        return None

    df = pd.read_csv(fpath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_zz1000():
    """加载中证1000日线数据"""
    fpath = os.path.join(DATA_DIR, 'zz1000_daily.csv')
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_traded_codes():
    """获取回测中所有交易过的股票代码列表"""
    _, _, df_trades, _ = load_backtest()
    if df_trades is None or len(df_trades) == 0:
        return []
    return sorted(df_trades['code'].unique().tolist())


@st.cache_data(ttl=300)
def _load_all_signals():
    """从回测JSON加载全量原始信号（含被过滤/未买入的）"""
    path, _ = _best_bt_path()
    if path is None or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        bt = json.load(f)
    sigs = bt.get('all_signals')
    if not sigs:
        return None
    df = pd.DataFrame(sigs)
    df['signal_date'] = pd.to_datetime(df['signal_date'], format='mixed')
    df['buy_date'] = pd.to_datetime(df['buy_date'], format='mixed')
    df['sell_date'] = pd.to_datetime(df['sell_date'], format='mixed')
    return df


def load_signals_by_date(date_str):
    """获取指定日期的全部信号

    Returns: dict with keys:
        'signals': DataFrame - 当日全部信号（含被skip过滤的）
        'date': str - 查询日期
        'total': int - 信号总数
        'by_gua': dict - 按中证大象卦分组统计
    """
    df_all = _load_all_signals()
    if df_all is None:
        return {'signals': pd.DataFrame(), 'date': date_str,
                'total': 0, 'by_gua': {}}

    mask = df_all['signal_date'].dt.strftime('%Y-%m-%d') == date_str
    df_day = df_all[mask].copy()

    by_gua = {}
    if len(df_day) > 0:
        for gua, grp in df_day.groupby('zz_year_gua'):
            by_gua[gua] = {
                'total': len(grp),
                'non_skip': int((~grp['is_skip']).sum()),
            }

    return {
        'signals': df_day.sort_values('pool_retail').reset_index(drop=True),
        'date': date_str,
        'total': len(df_day),
        'by_gua': by_gua,
    }


def load_trades_by_date(date_str):
    """获取指定日期实际成交的交易记录"""
    _, _, df_trades, _ = load_backtest()
    if df_trades is None or len(df_trades) == 0:
        return pd.DataFrame()
    mask = df_trades['buy_date'].dt.strftime('%Y-%m-%d') == date_str
    return df_trades[mask].copy()


BAGUA_ORDER = GUA_ORDER
ZONE_COLOR_MAP = {
    'buy': '#7f1d1d',
    'ban': '#14532d',
    'high': '#854d0e',
    'empty': '#374151',
}
ZONE_NAME_MAP = {
    'buy': '正收益买入区',
    'ban': '禁买区',
    'high': '问题区',
    'empty': '无效区',
}


RANK_LABEL_MAP = {
    'tier1': '1等',
    'tier2': '2等',
    'tier3': '3等',
    'empty': '空白',
}
RANK_COLOR_MAP = {
    'tier1': '#b91c1c',
    'tier2': '#ca8a04',
    'tier3': '#15803d',
    'empty': '#374151',
}
PAIR_MEANING_MAP = {
    ('000', '000'): '双底共振', ('000', '001'): '底部蓄势', ('000', '010'): '弱势试弹', ('000', '011'): '逆势抢跑',
    ('000', '111'): '高位背离', ('000', '110'): '虚热失真', ('000', '101'): '逆势护盘', ('000', '100'): '弱上加弱',
    ('001', '000'): '待势补涨', ('001', '001'): '双底蓄力', ('001', '010'): '试探上行', ('001', '011'): '起势在即',
    ('001', '111'): '强股先行', ('001', '110'): '热前蓄势', ('001', '101'): '稳中有强', ('001', '100'): '止跌未稳',
    ('010', '000'): '探底待起', ('010', '001'): '修复筑基', ('010', '010'): '弱反共鸣', ('010', '011'): '转强萌芽',
    ('010', '111'): '高位硬撑', ('010', '110'): '反弹乏力', ('010', '101'): '护盘修复', ('010', '100'): '反抽脆弱',
    ('011', '000'): '低位补涨', ('011', '001'): '顺势起涨', ('011', '010'): '跟涨修复', ('011', '011'): '共振启动',
    ('011', '111'): '强势加速', ('011', '110'): '热中走强', ('011', '101'): '护盘上攻', ('011', '100'): '逆势掉队',
    ('111', '000'): '高潮补涨', ('111', '001'): '高位待发', ('111', '010'): '强市弱股', ('111', '011'): '高位转强',
    ('111', '111'): '双强极盛', ('111', '110'): '高位滞涨', ('111', '101'): '核心抱团', ('111', '100'): '盛极转杀',
    ('110', '000'): '错位补涨', ('110', '001'): '滞涨蓄势', ('110', '010'): '热中弱修', ('110', '011'): '滞涨转强',
    ('110', '111'): '末端强撑', ('110', '110'): '双热将衰', ('110', '101'): '热中护盘', ('110', '100'): '热退急杀',
    ('101', '000'): '托底修复', ('101', '001'): '护盘筑底', ('101', '010'): '护中弱反', ('101', '011'): '护盘转强',
    ('101', '111'): '高位核心', ('101', '110'): '护中滞涨', ('101', '101'): '双护共稳', ('101', '100'): '护而难止',
    ('100', '000'): '超跌待弹', ('100', '001'): '暴跌止稳', ('100', '010'): '杀中反抽', ('100', '011'): '逆杀求生',
    ('100', '111'): '高位补杀', ('100', '110'): '虚热补跌', ('100', '101'): '逆势抗杀', ('100', '100'): '双杀共振',
}


def _pair_meaning(ren_gua, di_gua):
    return PAIR_MEANING_MAP.get((str(ren_gua).zfill(3), str(di_gua).zfill(3)), '卦意待定')


def _fmt_ret(value):
    return '--' if pd.isna(value) else f'{value:+.1f}%'


def _shift_tier(tier, delta):
    tiers = ['tier3', 'tier2', 'tier1']
    if tier not in tiers or delta == 0:
        return tier
    idx = tiers.index(tier)
    idx = min(max(idx + delta, 0), len(tiers) - 1)
    return tiers[idx]


def _semantic_bucket_for_pair(ren_gua, di_gua):
    ren_gua = str(ren_gua).zfill(3)
    di_gua = str(di_gua).zfill(3)
    if (ren_gua, di_gua) in {
        ('000', '000'), ('000', '001'), ('000', '010'),
        ('001', '000'), ('001', '010'), ('110', '110'),
    }:
        return 'tier1', '兑卦优先买：同频低位修复或同频高位兑现'
    if (ren_gua, di_gua) in {('011', '010'), ('011', '011')}:
        return 'tier2', '兑卦观察买：轮动扩散中的顺势补涨'
    if ren_gua == '110' and di_gua == '111':
        return 'tier3', '市场已偏兑现但个股过强，容易变成高位接力'
    if ren_gua in {'000', '001'}:
        return 'tier3', '低位环境下与兑卦主逻辑不同频，先列为C档'
    if ren_gua in {'010', '011', '100', '101', '110', '111'}:
        return 'tier3', '市场与个股节奏错位，不属于兑卦补涨兑现主线'
    return 'tier3', '不在兑卦当前白名单内，归为C档观察'


def _apply_rank_fields(matrix_df):
    matrix_df = matrix_df.copy()

    top_buy_pairs = set(
        matrix_df[
            (matrix_df['buy_count'] > 0) &
            (matrix_df['buy_avg_ret'] > 0)
        ]
        .sort_values(['buy_avg_ret', 'buy_count', 'signal_count'], ascending=[False, False, False])
        .head(5)[['ren_gua', 'di_gua']]
        .itertuples(index=False, name=None)
    )

    def classify(row):
        pair = (row['ren_gua'], row['di_gua'])
        if row['signal_count'] == 0:
            return pd.Series({
                'zone_type': 'empty',
                'zone_reason': 'no signals in this cell',
                'is_top_buy': pair in top_buy_pairs,
                'semantic_bucket': 'empty',
                'semantic_reason': '该格暂无信号，暂不做卦义评级',
                'rank_tier': 'empty',
                'rank_label': RANK_LABEL_MAP['empty'],
                'rank_order': -1,
                'rank_adjustment': 'none',
                'rank_reason': '无信号，保持空白',
                'evidence_text': '无全量信号，无法判断卦义与数据表现',
                'rank_color': RANK_COLOR_MAP['empty'],
            })

        if row['buy_count'] > 0 and row['buy_avg_ret'] > 0:
            zone_type = 'buy'
            zone_reason = 'positive realized buy return'
            if pair in top_buy_pairs:
                zone_reason += '; top 5 buy return'
        elif row['signal_avg_ret'] < -3:
            zone_type = 'ban'
            zone_reason = 'signal average return below -3%'
        elif row['buy_count'] == 0 and row['signal_avg_ret'] > 0:
            zone_type = 'high'
            zone_reason = 'signal average return positive but no actual trade'
        elif row['buy_count'] > 0 and row['signal_avg_ret'] > row['buy_avg_ret']:
            zone_type = 'high'
            zone_reason = 'signal average return exceeds buy average return'
        else:
            zone_type = 'high'
            zone_reason = 'signal exists but does not enter positive-buy or hard-ban bucket'

        semantic_bucket, semantic_reason = _semantic_bucket_for_pair(row['ren_gua'], row['di_gua'])
        rank_tier = semantic_bucket
        rank_adjustment = 'none'
        rank_reason = semantic_reason

        if row['signal_count'] < 3 and rank_tier in {'tier1', 'tier2'}:
            rank_tier = _shift_tier(rank_tier, -1)
            rank_adjustment = 'demote_sample'
            rank_reason = '卦义原档偏强，但样本量不足，先降一档观察'
        if row['buy_count'] > 0 and not pd.isna(row['buy_avg_ret']) and row['buy_avg_ret'] < 0:
            rank_tier = _shift_tier(rank_tier, -1)
            rank_adjustment = 'demote_negative_buy'
            rank_reason = '实际买入转负，说明执行端还需降一档'
        elif row['buy_count'] > 0 and not pd.isna(row['buy_avg_ret']) and row['buy_avg_ret'] > 0 and row['signal_count'] >= 3:
            promoted = _shift_tier(rank_tier, 1)
            if promoted != rank_tier:
                rank_tier = promoted
                rank_adjustment = 'promote_positive_buy'
                rank_reason = '数据结构与实际买入都为正，向上修正一档'

        evidence_text = (
            f"卦义：{semantic_reason}；"
            f"数据：全量{int(row['signal_count'])}笔/{_fmt_ret(row['signal_avg_ret'])}，"
            f"可买{int(row.get('can_buy_count', row['signal_count']))}笔，"
            f"买入{int(row['buy_count'])}笔/{_fmt_ret(row['buy_avg_ret'])}，"
            f"转化率{row['buy_rate_pct']:.1f}%"
        )

        return pd.Series({
            'zone_type': zone_type,
            'zone_reason': zone_reason,
            'is_top_buy': pair in top_buy_pairs,
            'semantic_bucket': semantic_bucket,
            'semantic_reason': semantic_reason,
            'rank_tier': rank_tier,
            'rank_label': RANK_LABEL_MAP[rank_tier],
            'rank_order': {'tier1': 3, 'tier2': 2, 'tier3': 1, 'empty': -1}[rank_tier],
            'rank_adjustment': rank_adjustment,
            'rank_reason': rank_reason,
            'evidence_text': evidence_text,
            'rank_color': RANK_COLOR_MAP[rank_tier],
        })

    classified = matrix_df.apply(classify, axis=1)
    for col in classified.columns:
        matrix_df[col] = classified[col]
    matrix_df['zone_name'] = matrix_df['zone_type'].map(ZONE_NAME_MAP)
    matrix_df['bg_color'] = matrix_df['rank_color']
    matrix_df['pair_meaning'] = matrix_df.apply(lambda row: _pair_meaning(row['ren_gua'], row['di_gua']), axis=1)
    matrix_df['display_line1'] = matrix_df.apply(lambda row: f"{int(row['signal_count'])}/{int(row.get('can_buy_count', row['signal_count']))}/{int(row['buy_count'])}", axis=1)
    matrix_df['display_line2'] = matrix_df.apply(lambda row: f"{_fmt_ret(row['signal_avg_ret'])}/{_fmt_ret(row['buy_avg_ret'])}", axis=1)
    matrix_df['display_line3'] = matrix_df['pair_meaning']
    matrix_df['display_text'] = matrix_df.apply(
        lambda row: f"{row['display_line1']}\n{row['display_line2']}\n{row['display_line3']}",
        axis=1,
    )
    return matrix_df


def get_gua_order():
    return BAGUA_ORDER.copy()


@st.cache_data(ttl=300)
def load_bagua_debug_frames(source='8gua', use_experiment_baseline=False):
    """加载八卦调试所需的 signals / trades / strategy 数据。"""
    if use_experiment_baseline:
        target_gua = str(source).zfill(3) if source else None
        gua_strategy = {g: copy.deepcopy(spec['base_cfg']) for g, spec in eg.GUA_EXPERIMENT_SPECS.items()}
        if target_gua in gua_strategy:
            gua_strategy[target_gua] = copy.deepcopy(eg.get_spec(target_gua)['naked_cfg'])
        return None, pd.DataFrame(), pd.DataFrame(), gua_strategy

    meta, _, df_trades, _ = load_backtest(source=source)
    gua_strategy = load_backtest_8gua_extra() if source == '8gua' else None

    path = (BT_8GUA_FORMAL_PATH if os.path.exists(BT_8GUA_FORMAL_PATH) else BT_8GUA_PATH) if source == '8gua' else BT_PATH
    if path is None or not os.path.exists(path):
        return meta, pd.DataFrame(), pd.DataFrame(), gua_strategy

    with open(path, 'r', encoding='utf-8') as f:
        bt = json.load(f)

    sigs = bt.get('all_signals') or bt.get('signal_detail') or []
    df_signals = pd.DataFrame(sigs)
    if len(df_signals) > 0:
        compat_rename_columns(df_signals)
        if 'gua' not in df_signals.columns and 'tian_gua' in df_signals.columns:
            df_signals['gua'] = df_signals['tian_gua']
        for col in ['signal_date', 'buy_date', 'sell_date']:
            if col in df_signals.columns:
                df_signals[col] = pd.to_datetime(df_signals[col], format='mixed', errors='coerce')
        for col in ['actual_ret', 'hold_days', 'pool_retail']:
            if col in df_signals.columns:
                df_signals[col] = pd.to_numeric(df_signals[col], errors='coerce')
        for col in ['gua', 'ren_gua', 'di_gua']:
            if col in df_signals.columns:
                df_signals[col] = df_signals[col].astype(str).str.zfill(3)
        df_signals = df_signals[df_signals['gua'].isin(BAGUA_ORDER)].copy() if 'gua' in df_signals.columns else pd.DataFrame()

    if df_trades is None:
        df_trades = pd.DataFrame()
    elif len(df_trades) > 0:
        compat_rename_columns(df_trades)
        if 'tian_gua' not in df_trades.columns and 'gua' in df_trades.columns:
            df_trades['tian_gua'] = df_trades['gua']
        for col in ['gua', 'ren_gua', 'di_gua']:
            if col in df_trades.columns:
                df_trades[col] = df_trades[col].astype(str).str.zfill(3)
        for col in ['ret_pct', 'profit', 'hold_days', 'buy_price', 'sell_price', 'cost']:
            if col in df_trades.columns:
                df_trades[col] = pd.to_numeric(df_trades[col], errors='coerce')
        df_trades = df_trades[df_trades['gua'].isin(BAGUA_ORDER)].copy() if 'gua' in df_trades.columns else pd.DataFrame()

    return meta, df_signals, df_trades, gua_strategy



def _parse_gua_set(value):
    if value is None:
        return set()
    if isinstance(value, set):
        return {str(v).zfill(3) for v in value}
    if isinstance(value, (list, tuple)):
        return {str(v).zfill(3) for v in value}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return set()
        if s == 'set()':
            return set()
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = [p.strip() for p in s.split(',') if p.strip()]
        if isinstance(parsed, set):
            return {str(v).zfill(3) for v in parsed}
        if isinstance(parsed, (list, tuple)):
            return {str(v).zfill(3) for v in parsed}
        return {str(parsed).zfill(3)}
    return {str(value).zfill(3)}


@st.cache_data(ttl=300)
def build_bagua_zone_rules(gua_strategy):
    """从 gua_strategy 中提取每个目标卦的市场卦禁买集合。"""
    rules = {}
    for gua in BAGUA_ORDER:
        strat = (gua_strategy or {}).get(gua, {})
        exclude_ren = set()
        allow_stock = set()
        exclude_stock = set()
        for key, value in strat.items():
            if 'exclude_ren_gua' in key:
                exclude_ren |= _parse_gua_set(value)
            elif 'allow_di_gua' in key:
                allow_stock |= _parse_gua_set(value)
            elif 'exclude_di_gua' in key:
                exclude_stock |= _parse_gua_set(value)
        rules[gua] = {
            'exclude_ren_gua': exclude_ren,
            'allow_di_gua': allow_stock,
            'exclude_di_gua': exclude_stock,
            'strategy': strat,
        }
    return rules


@st.cache_data(ttl=None)
def build_bagua_debug_matrix(target_gua, source='8gua', use_experiment_baseline=False, data_version=''):
    """构建单个目标卦的 64 卦联动矩阵。data_version 由调用方传入，数据变更时变化以失效缓存。"""
    target_gua = str(target_gua).zfill(3)

    if use_experiment_baseline:
        spec = eg.get_spec(target_gua)
        payload = eg.build_payload_for_cfg(target_gua, copy.deepcopy(spec['naked_cfg']))
        signal_df = payload['target_sig'].copy()
        trade_df = eg.build_trade_detail(payload['result'], target_gua).copy()
        for _df in (signal_df, trade_df):
            compat_rename_columns(_df)
        signal_df['gua'] = signal_df['tian_gua'].astype(str).str.zfill(3)
        signal_df['ren_gua'] = signal_df['ren_gua'].astype(str).str.zfill(3)
        signal_df['di_gua'] = signal_df['di_gua'].astype(str).str.zfill(3)
        if len(trade_df) > 0:
            trade_df['gua'] = trade_df['gua'].astype(str).str.zfill(3)
            trade_df['ren_gua'] = trade_df['ren_gua'].astype(str).str.zfill(3)
            trade_df['di_gua'] = trade_df['di_gua'].astype(str).str.zfill(3)
        meta = payload['stats']
        gua_strategy = {g: copy.deepcopy(spec_['base_cfg']) for g, spec_ in eg.GUA_EXPERIMENT_SPECS.items()}
        gua_strategy[target_gua] = copy.deepcopy(spec['naked_cfg'])
    else:
        meta, df_signals, df_trades, gua_strategy = load_bagua_debug_frames(source=source)
        signal_df = df_signals[df_signals['gua'] == target_gua].copy() if len(df_signals) else pd.DataFrame()
        trade_df = df_trades[df_trades['gua'] == target_gua].copy() if len(df_trades) else pd.DataFrame()

    rules = build_bagua_zone_rules(gua_strategy)

    if len(signal_df) > 0:
        formal_signal_group = signal_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
            can_buy_count=('code', 'size'),
        ).reset_index()
    else:
        formal_signal_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'can_buy_count'])
    try:
        _baseline = _build_baseline_debug_matrix(target_gua, data_version=data_version)
        naked_signal_group = _baseline['matrix_df'][['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret']].copy()
        naked_detail_signals = _baseline.get('detail_signals', pd.DataFrame()).copy()
    except (KeyError, FileNotFoundError):
        if len(signal_df) > 0:
            naked_signal_group = signal_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
                signal_count=('code', 'size'),
                signal_avg_ret=('actual_ret', 'mean'),
            ).reset_index()
        else:
            naked_signal_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret'])
        naked_detail_signals = signal_df.copy()

    if len(trade_df) > 0:
        trade_group = trade_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
            buy_count=('code', 'size'),
            buy_avg_ret=('ret_pct', 'mean'),
        ).reset_index()
    else:
        trade_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'buy_count', 'buy_avg_ret'])

    base = pd.MultiIndex.from_product([BAGUA_ORDER, BAGUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
    matrix_df = base.merge(naked_signal_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df = matrix_df.merge(formal_signal_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df = matrix_df.merge(trade_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df['signal_count'] = matrix_df['signal_count'].fillna(0).astype(int)
    matrix_df['can_buy_count'] = matrix_df['can_buy_count'].fillna(0).astype(int)
    matrix_df['buy_count'] = matrix_df['buy_count'].fillna(0).astype(int)
    matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
    matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
    matrix_df['ren_name'] = matrix_df['ren_gua'].map(REN_GUA_NAMES)
    matrix_df['di_name'] = matrix_df['di_gua'].map(DI_GUA_NAMES)
    matrix_df['buy_rate_pct'] = (matrix_df['buy_count'] / matrix_df['can_buy_count'].replace(0, pd.NA) * 100).fillna(0)

    rule = rules.get(target_gua, {})
    blocked = rule.get('exclude_ren_gua', set())
    allow_stock = rule.get('allow_di_gua', set())
    exclude_stock = rule.get('exclude_di_gua', set())

    matrix_df = _apply_rank_fields(matrix_df)

    detail_signals = naked_detail_signals if len(naked_detail_signals) > 0 else signal_df.copy()
    detail_can_buy_signals = signal_df.copy()
    detail_trades = trade_df.copy()
    if len(detail_signals) > 0:
        detail_signals['ren_name'] = detail_signals['ren_gua'].map(REN_GUA_NAMES)
        detail_signals['di_name'] = detail_signals['di_gua'].map(DI_GUA_NAMES)
    if len(detail_trades) > 0:
        detail_trades['ren_name'] = detail_trades['ren_gua'].map(REN_GUA_NAMES)
        detail_trades['di_name'] = detail_trades['di_gua'].map(DI_GUA_NAMES)

    return {
        'meta': meta,
        'target_gua': target_gua,
        'target_name': TIAN_GUA_NAMES.get(target_gua, target_gua),
        'matrix_df': matrix_df,
        'detail_signals': detail_signals,
        'detail_can_buy_signals': detail_can_buy_signals,
        'detail_trades': detail_trades,
        'blocked_markets': sorted(blocked),
        'blocked_ren_names': [REN_GUA_NAMES.get(g, g) for g in sorted(blocked)],
        'allowed_stocks': sorted(allow_stock),
        'allowed_di_names': [DI_GUA_NAMES.get(g, g) for g in sorted(allow_stock)],
        'excluded_stocks': sorted(exclude_stock),
        'excluded_di_names': [DI_GUA_NAMES.get(g, g) for g in sorted(exclude_stock)],
        'uses_experiment_baseline': use_experiment_baseline,
    }


@st.cache_data(ttl=None)
def build_all_bagua_debug_payload(source='8gua', data_version=''):
    return {gua: build_bagua_debug_matrix(gua, source=source, data_version=data_version) for gua in BAGUA_ORDER}


DEBUG_DATASET_CONFIG = {
    'formal': {
        'label': '正式策略数据',
        'description': '已固化的八卦分治正式策略回测结果。',
        'source_text': '当前 dashboard 正式回测口径',
    },
    'test': {
        'label': '测试策略数据',
        'description': '逐卦分析过程中使用的专项测试口径。',
        'source_text': '逐卦分析测试口径',
    },
    'baseline': {
        'label': '裸跑基准数据',
        'description': '按每卦独立初始入池阈值运行、只保留最简单买卖的裸跑基准。',
        'source_text': '全市场裸跑基准口径',
    },
}


@st.cache_data(ttl=300)
def get_bagua_debug_dataset_config(dataset_key='formal'):
    dataset_key = dataset_key if dataset_key in DEBUG_DATASET_CONFIG else 'formal'
    return {'key': dataset_key, **DEBUG_DATASET_CONFIG[dataset_key]}


def _is_same_as_formal_base_cfg(target_gua):
    target_gua = str(target_gua).zfill(3)
    if target_gua not in eg.GUA_EXPERIMENT_SPECS:
        return False
    base_cfg = copy.deepcopy(eg.GUA_EXPERIMENT_SPECS[target_gua]['base_cfg'])
    return all(b8.GUA_STRATEGY[target_gua].get(k) == v for k, v in base_cfg.items())


def _apply_buy_case_to_cfg(target_gua, runtime_cfg, buy_case):
    target_gua = str(target_gua).zfill(3)
    spec = eg.get_spec(target_gua)
    if not buy_case:
        return runtime_cfg
    case_updates = dict(spec.get('buy_cases', []))
    if buy_case in case_updates:
        runtime_cfg.update(copy.deepcopy(case_updates[buy_case]))
        return runtime_cfg
    # 动态解析 cross@N 或 double_rise
    fields = spec.get('fields', {})
    buy_mode_field = fields.get('buy_mode')
    cross_field = fields.get('cross')
    if buy_case == 'double_rise':
        if buy_mode_field:
            runtime_cfg[buy_mode_field] = 'double_rise'
    elif buy_case.startswith('cross@'):
        try:
            n = int(buy_case.split('@', 1)[1])
        except (ValueError, IndexError):
            return runtime_cfg
        if buy_mode_field:
            runtime_cfg[buy_mode_field] = 'cross'
        if cross_field:
            runtime_cfg[cross_field] = n
    return runtime_cfg


def _get_test_runtime_cfg(target_gua, buy_case=None, pool_threshold=None, sell=None, pool_days_min=None, pool_days_max=None, pool_depth=None):
    target_gua = str(target_gua).zfill(3)
    spec = eg.get_spec(target_gua)
    # naked_cfg 现在已包含 tiers (derive_naked_cfg 不再 pop), 不需要再单独注入
    runtime_cfg = copy.deepcopy(spec['naked_cfg'])
    runtime_cfg = _apply_buy_case_to_cfg(target_gua, runtime_cfg, buy_case)
    if pool_threshold is not None:
        runtime_cfg['pool_threshold'] = pool_threshold
    if sell is not None:
        runtime_cfg['sell'] = sell
    if pool_days_min is not None:
        runtime_cfg['pool_days_min'] = pool_days_min
    if pool_days_max is not None:
        runtime_cfg['pool_days_max'] = pool_days_max
    # pool_depth=None 沿用 naked_cfg 默认 (None = 不做二次验证)
    # 显式传值时写入，包括传入 None 以明确关闭
    runtime_cfg['pool_depth'] = pool_depth
    return runtime_cfg


def _get_test_dataset_signature(target_gua, buy_case=None, pool_threshold=None, sell=None, tier1=None, tier2=None, pool_days_min=None, pool_days_max=None, pool_depth=None):
    target_gua = str(target_gua).zfill(3)
    if target_gua not in eg.GUA_EXPERIMENT_SPECS:
        return None
    base_key = eg.make_cfg_key(target_gua, _get_test_runtime_cfg(target_gua, buy_case=buy_case, pool_threshold=pool_threshold, sell=sell, pool_days_min=pool_days_min, pool_days_max=pool_days_max, pool_depth=pool_depth))
    return (base_key, ('tier1', tier1), ('tier2', tier2))


@st.cache_data(ttl=None)
def _load_baseline_debug_snapshot(data_version=''):
    if not os.path.exists(BAGUA_DEBUG_BASELINE_SNAPSHOT_PATH):
        raise FileNotFoundError(f'baseline snapshot missing: {BAGUA_DEBUG_BASELINE_SNAPSHOT_PATH}')
    with open(BAGUA_DEBUG_BASELINE_SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data(ttl=None)
def _build_baseline_debug_matrix(target_gua, data_version=''):
    target_gua = str(target_gua).zfill(3)
    snapshot = _load_baseline_debug_snapshot(data_version=data_version)
    raw = (snapshot.get('payloads') or {}).get(target_gua)
    if raw is None:
        raise KeyError(f'baseline snapshot missing target_gua={target_gua}')

    matrix_df = pd.DataFrame(raw.get('matrix_df', []))
    signal_df = pd.DataFrame(raw.get('detail_signals', []))
    trade_df = pd.DataFrame(raw.get('detail_trades', []))

    for df in (matrix_df, signal_df, trade_df):
        compat_rename_columns(df)

    for df, date_cols in ((signal_df, ['signal_date', 'buy_date', 'sell_date']), (trade_df, ['buy_date', 'sell_date'])):
        if len(df) > 0:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

    for df in (signal_df, trade_df, matrix_df):
        if len(df) > 0:
            for col in ['gua', 'ren_gua', 'di_gua']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.zfill(3)

    if len(signal_df) > 0:
        for col in ['actual_ret']:
            if col in signal_df.columns:
                signal_df[col] = pd.to_numeric(signal_df[col], errors='coerce')
        signal_df['ren_name'] = signal_df['ren_gua'].map(REN_GUA_NAMES)
        signal_df['di_name'] = signal_df['di_gua'].map(DI_GUA_NAMES)

    if len(trade_df) > 0:
        for col in ['profit', 'ret_pct', 'hold_days', 'cost', 'buy_price', 'sell_price']:
            if col in trade_df.columns:
                trade_df[col] = pd.to_numeric(trade_df[col], errors='coerce')
        trade_df['ren_name'] = trade_df['ren_gua'].map(REN_GUA_NAMES)
        trade_df['di_name'] = trade_df['di_gua'].map(DI_GUA_NAMES)

    base = pd.MultiIndex.from_product([BAGUA_ORDER, BAGUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
    matrix_df = base.merge(matrix_df, on=['ren_gua', 'di_gua'], how='left')
    matrix_df['signal_count'] = pd.to_numeric(matrix_df['signal_count'], errors='coerce').fillna(0).astype(int)
    matrix_df['can_buy_count'] = matrix_df['signal_count']
    matrix_df['buy_count'] = pd.to_numeric(matrix_df['buy_count'], errors='coerce').fillna(0).astype(int)
    matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
    matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
    matrix_df['ren_name'] = matrix_df['ren_gua'].map(REN_GUA_NAMES)
    matrix_df['di_name'] = matrix_df['di_gua'].map(DI_GUA_NAMES)
    matrix_df['buy_rate_pct'] = (matrix_df['buy_count'] / matrix_df['signal_count'].replace(0, pd.NA) * 100).fillna(0)
    if 'zone_type' not in matrix_df.columns:
        matrix_df['zone_type'] = 'empty'
    matrix_df['zone_type'] = matrix_df['zone_type'].fillna('empty')
    if 'zone_reason' not in matrix_df.columns:
        matrix_df['zone_reason'] = 'no signals in this cell'
    matrix_df['zone_reason'] = matrix_df['zone_reason'].fillna('no signals in this cell')
    if 'is_top_buy' not in matrix_df.columns:
        matrix_df['is_top_buy'] = False
    matrix_df['is_top_buy'] = matrix_df['is_top_buy'].fillna(False).astype(bool)
    matrix_df = _apply_rank_fields(matrix_df)

    return {
        'meta': raw.get('meta', {}),
        'target_gua': raw.get('target_gua', target_gua),
        'target_name': TIAN_GUA_NAMES.get(target_gua, raw.get('target_name', target_gua)),
        'matrix_df': matrix_df,
        'detail_signals': signal_df,
        'detail_can_buy_signals': signal_df,
        'detail_trades': trade_df,
        'blocked_markets': raw.get('blocked_markets', []),
        'blocked_ren_names': raw.get('blocked_ren_names', []),
        'allowed_stocks': raw.get('allowed_stocks', []),
        'allowed_di_names': raw.get('allowed_di_names', []),
        'excluded_stocks': raw.get('excluded_stocks', []),
        'excluded_di_names': raw.get('excluded_di_names', []),
        'uses_experiment_baseline': False,
    }


@st.cache_data(ttl=None)
def _build_test_debug_matrix(target_gua, dataset_signature=None, buy_case=None, pool_threshold=None, sell=None, tier1_threshold=None, tier2_threshold=None, pool_days_min=None, pool_days_max=None, pool_depth=None, start_date=None, end_date=None, data_version=''):
    target_gua = str(target_gua).zfill(3)
    spec = eg.get_spec(target_gua)
    runtime_cfg = _get_test_runtime_cfg(target_gua, buy_case=buy_case, pool_threshold=pool_threshold, sell=sell, pool_days_min=pool_days_min, pool_days_max=pool_days_max, pool_depth=pool_depth)

    # --- 裸跑全量: 直接读快照，与正式/裸跑基准口径一致 ---
    try:
        _baseline = _build_baseline_debug_matrix(target_gua, data_version=data_version)
        naked_signal_group = _baseline['matrix_df'][['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret']].copy()
    except (KeyError, FileNotFoundError):
        naked_signal_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret'])

    # --- 策略信号: 用 test runtime_cfg 计算（可能与裸跑不同） ---
    test_payload = eg.build_payload_for_cfg(target_gua, copy.deepcopy(runtime_cfg))
    signal_df = test_payload['target_sig'].copy()

    compat_rename_columns(signal_df)

    if len(signal_df) > 0:
        signal_df['gua'] = signal_df['tian_gua'].astype(str).str.zfill(3)
        signal_df['ren_gua'] = signal_df['ren_gua'].astype(str).str.zfill(3)
        signal_df['di_gua'] = signal_df['di_gua'].astype(str).str.zfill(3)

    if len(signal_df) > 0 and (start_date or end_date):
        sig_dates = pd.to_datetime(signal_df['signal_date'], errors='coerce')
        if start_date:
            signal_df = signal_df[sig_dates >= pd.Timestamp(start_date)]
        if end_date:
            sig_dates = pd.to_datetime(signal_df['signal_date'], errors='coerce')
            signal_df = signal_df[sig_dates <= pd.Timestamp(end_date)]

    if len(signal_df) > 0:
        if 'is_skip' in signal_df.columns:
            signal_df['is_skip'] = False
        if 'rank_order' in signal_df.columns:
            signal_df['rank_order'] = 0

    # --- 等次过滤: 基于裸跑 avg_ret 分类，应用到策略信号 ---
    if tier1_threshold is not None and tier2_threshold is not None and len(signal_df) > 0:
        tier_map = {}
        for _, row in naked_signal_group.iterrows():
            if int(row.get('signal_count', 0)) <= 0:
                continue
            avg_ret = row['signal_avg_ret']
            if pd.isna(avg_ret):
                continue
            mg = str(row['ren_gua']).zfill(3)
            sg = str(row['di_gua']).zfill(3)
            if avg_ret >= tier1_threshold:
                tier_map[(mg, sg)] = 1
            elif avg_ret >= tier2_threshold:
                tier_map[(mg, sg)] = 2
            else:
                tier_map[(mg, sg)] = 3

        if 'rank_order' not in signal_df.columns:
            signal_df['rank_order'] = 0
        for idx in signal_df.index:
            mg = signal_df.at[idx, 'ren_gua']
            sg = signal_df.at[idx, 'di_gua']
            tier = tier_map.get((mg, sg), 3)
            if tier == 3:
                signal_df.at[idx, 'is_skip'] = True
            elif tier == 1:
                cur = signal_df.at[idx, 'rank_order']
                if pd.isna(cur) or int(cur) < 10:
                    signal_df.at[idx, 'rank_order'] = 10

        re_sim = eg.simulate_case_from_filtered_target(target_gua, test_payload, signal_df)
        trade_df = eg.build_trade_detail(re_sim['result'], target_gua).copy()
        meta = re_sim['stats']
    else:
        trade_df = eg.build_trade_detail(test_payload['result'], target_gua).copy()
        meta = test_payload['stats']

    if len(trade_df) > 0:
        compat_rename_columns(trade_df)
        trade_df['gua'] = trade_df['gua'].astype(str).str.zfill(3)
        trade_df['ren_gua'] = trade_df['ren_gua'].astype(str).str.zfill(3)
        trade_df['di_gua'] = trade_df['di_gua'].astype(str).str.zfill(3)
    can_buy_df = signal_df
    can_buy_group = can_buy_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
        can_buy_count=('code', 'size'),
    ).reset_index() if len(can_buy_df) > 0 else pd.DataFrame(columns=['ren_gua', 'di_gua', 'can_buy_count'])

    trade_group = trade_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
        buy_count=('code', 'size'),
        buy_avg_ret=('ret_pct', 'mean'),
    ).reset_index() if len(trade_df) > 0 else pd.DataFrame(columns=['ren_gua', 'di_gua', 'buy_count', 'buy_avg_ret'])

    base = pd.MultiIndex.from_product([BAGUA_ORDER, BAGUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
    matrix_df = base.merge(naked_signal_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df = matrix_df.merge(can_buy_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df = matrix_df.merge(trade_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df['signal_count'] = matrix_df['signal_count'].fillna(0).astype(int)
    matrix_df['can_buy_count'] = matrix_df['can_buy_count'].fillna(0).astype(int)
    matrix_df['buy_count'] = matrix_df['buy_count'].fillna(0).astype(int)
    matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
    matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
    matrix_df['ren_name'] = matrix_df['ren_gua'].map(REN_GUA_NAMES)
    matrix_df['di_name'] = matrix_df['di_gua'].map(DI_GUA_NAMES)
    matrix_df['buy_rate_pct'] = (matrix_df['buy_count'] / matrix_df['can_buy_count'].replace(0, pd.NA) * 100).fillna(0)

    matrix_df = _apply_rank_fields(matrix_df)

    try:
        _bl = _build_baseline_debug_matrix(target_gua, data_version=data_version)
        naked_detail_signals = _bl.get('detail_signals', pd.DataFrame()).copy()
    except (KeyError, FileNotFoundError):
        naked_detail_signals = pd.DataFrame()
    if len(naked_detail_signals) > 0:
        if 'ren_gua' in naked_detail_signals.columns:
            naked_detail_signals['ren_name'] = naked_detail_signals['ren_gua'].map(REN_GUA_NAMES)
            naked_detail_signals['di_name'] = naked_detail_signals['di_gua'].map(DI_GUA_NAMES)
    if len(signal_df) > 0:
        signal_df['ren_name'] = signal_df['ren_gua'].map(REN_GUA_NAMES)
        signal_df['di_name'] = signal_df['di_gua'].map(DI_GUA_NAMES)
    if len(trade_df) > 0:
        trade_df['ren_name'] = trade_df['ren_gua'].map(REN_GUA_NAMES)
        trade_df['di_name'] = trade_df['di_gua'].map(DI_GUA_NAMES)

    return {
        'meta': meta,
        'target_gua': target_gua,
        'target_name': TIAN_GUA_NAMES.get(target_gua, target_gua),
        'matrix_df': matrix_df,
        'detail_signals': naked_detail_signals,
        'detail_can_buy_signals': can_buy_df,
        'detail_trades': trade_df,
        'blocked_markets': [],
        'blocked_ren_names': [],
        'allowed_stocks': [],
        'allowed_di_names': [],
        'excluded_stocks': [],
        'excluded_di_names': [],
        'filtered_pairs': [],
        'filtered_pair_names': [],
        'uses_experiment_baseline': False,
    }



@st.cache_data(ttl=None)
def build_bagua_debug_matrix_for_dataset(target_gua, dataset_key='formal', test_buy_case=None, test_pool_threshold=None, test_sell=None, tier1_threshold=None, tier2_threshold=None, pool_days_min=None, pool_days_max=None, pool_depth=None, start_date=None, end_date=None, data_version=''):
    dataset_key = dataset_key if dataset_key in DEBUG_DATASET_CONFIG else 'formal'
    fallback_dataset_key = None
    if dataset_key == 'formal':
        payload = build_bagua_debug_matrix(target_gua, source='8gua', use_experiment_baseline=False, data_version=data_version)
    elif dataset_key == 'baseline':
        # v3 综合裸跑语义: 始终读 snapshot (来自 backtest_8gua_naked.py 综合回测分组).
        # 不再支持 UI 参数临时覆盖后走单卦独立跑 — 那是 test 数据集的职责.
        payload = _build_baseline_debug_matrix(target_gua, data_version=data_version)
    else:
        if str(target_gua).zfill(3) in eg.GUA_EXPERIMENT_SPECS:
            sig = _get_test_dataset_signature(target_gua, buy_case=test_buy_case, pool_threshold=test_pool_threshold, sell=test_sell, tier1=tier1_threshold, tier2=tier2_threshold, pool_days_min=pool_days_min, pool_days_max=pool_days_max, pool_depth=pool_depth)
            payload = _build_test_debug_matrix(target_gua, dataset_signature=sig, buy_case=test_buy_case, pool_threshold=test_pool_threshold, sell=test_sell, tier1_threshold=tier1_threshold, tier2_threshold=tier2_threshold, pool_days_min=pool_days_min, pool_days_max=pool_days_max, pool_depth=pool_depth, start_date=start_date, end_date=end_date, data_version=data_version)
        else:
            payload = build_bagua_debug_matrix(target_gua, source='8gua', use_experiment_baseline=False, data_version=data_version)
            fallback_dataset_key = 'formal'

    payload = copy.deepcopy(payload)

    if start_date or end_date:
        for detail_key, date_col in [('detail_signals', 'signal_date'), ('detail_can_buy_signals', 'signal_date'), ('detail_trades', 'buy_date')]:
            df = payload.get(detail_key)
            if df is not None and len(df) > 0 and date_col in df.columns:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                mask = pd.Series(True, index=df.index)
                if start_date:
                    mask &= dates >= pd.Timestamp(start_date)
                if end_date:
                    mask &= dates <= pd.Timestamp(end_date)
                payload[detail_key] = df[mask].reset_index(drop=True)

    payload['dataset'] = get_bagua_debug_dataset_config(dataset_key)
    payload['fallback_dataset_key'] = fallback_dataset_key
    payload['date_range'] = extract_payload_date_range(payload)
    payload['contribution_metrics'] = compute_single_bagua_contribution(payload)
    return payload


@st.cache_data(ttl=None)
def build_all_bagua_debug_payload_for_dataset(dataset_key='formal', test_buy_case=None, test_pool_threshold=None, test_pool_thresholds=None, test_buy_cases=None, test_sell_methods=None, test_tier1_thresholds=None, test_tier2_thresholds=None, test_pool_days_mins=None, test_pool_days_maxs=None, test_pool_depths=None, start_date=None, end_date=None, data_version=''):
    result = {}
    for gua in BAGUA_ORDER:
        pt = None
        if test_pool_thresholds is not None:
            pt = test_pool_thresholds.get(gua, test_pool_threshold)
        elif test_pool_threshold is not None:
            pt = test_pool_threshold
        bc = None
        if test_buy_cases is not None:
            bc = test_buy_cases.get(gua, test_buy_case)
        elif test_buy_case is not None:
            bc = test_buy_case
        sm = None
        if test_sell_methods is not None:
            sm = test_sell_methods.get(gua)
        t1 = test_tier1_thresholds.get(gua) if test_tier1_thresholds else None
        t2 = test_tier2_thresholds.get(gua) if test_tier2_thresholds else None
        pd_min = test_pool_days_mins.get(gua) if test_pool_days_mins else None
        pd_max = test_pool_days_maxs.get(gua) if test_pool_days_maxs else None
        pdp = test_pool_depths.get(gua) if test_pool_depths else None
        result[gua] = build_bagua_debug_matrix_for_dataset(gua, dataset_key=dataset_key, test_buy_case=bc, test_pool_threshold=pt, test_sell=sm, tier1_threshold=t1, tier2_threshold=t2, pool_days_min=pd_min, pool_days_max=pd_max, pool_depth=pdp, start_date=start_date, end_date=end_date, data_version=data_version)
    return result


def extract_payload_date_range(payload):
    dates = []
    for key, col in [('detail_signals', 'signal_date'), ('detail_signals', 'buy_date'), ('detail_trades', 'buy_date'), ('detail_trades', 'sell_date')]:
        df = payload.get(key)
        if df is not None and len(df) > 0 and col in df.columns:
            values = pd.to_datetime(df[col], errors='coerce').dropna()
            if len(values) > 0:
                dates.extend(values.tolist())
    if not dates:
        return {'start': None, 'end': None, 'text': '--'}
    start = min(dates)
    end = max(dates)
    return {'start': start, 'end': end, 'text': f"{start:%Y-%m-%d} → {end:%Y-%m-%d}"}


def compute_single_bagua_contribution(payload):
    matrix_df = payload.get('matrix_df', pd.DataFrame())
    trades = payload.get('detail_trades', pd.DataFrame())
    signals = payload.get('detail_signals', pd.DataFrame())
    can_buy_signals = payload.get('detail_can_buy_signals', signals)
    total_signal = int(matrix_df['signal_count'].sum()) if len(matrix_df) > 0 else 0
    total_buy = int(matrix_df['buy_count'].sum()) if len(matrix_df) > 0 else 0
    if len(matrix_df) > 0 and 'can_buy_count' in matrix_df.columns:
        can_buy_count = int(matrix_df['can_buy_count'].sum())
    elif len(can_buy_signals) > 0 and 'is_skip' in can_buy_signals.columns:
        can_buy_count = int((~can_buy_signals['is_skip'].fillna(False)).sum())
    else:
        can_buy_count = total_signal
    profit = float(pd.to_numeric(trades['profit'], errors='coerce').fillna(0).sum()) if len(trades) > 0 and 'profit' in trades.columns else 0.0
    ret_series = pd.to_numeric(trades['ret_pct'], errors='coerce').dropna() if len(trades) > 0 and 'ret_pct' in trades.columns else pd.Series(dtype=float)
    avg_buy_ret = float(ret_series.mean()) if ret_series.size else 0.0
    win_rate = float((ret_series > 0).mean() * 100) if ret_series.size else 0.0
    return {
        'target_gua': payload.get('target_gua'),
        'target_name': payload.get('target_name'),
        'signal_count': total_signal,
        'can_buy_count': can_buy_count,
        'buy_count': total_buy,
        'profit': profit,
        'avg_buy_ret': avg_buy_ret,
        'win_rate': win_rate,
        'date_range': payload.get('date_range') or extract_payload_date_range(payload),
    }


def compute_bagua_dashboard_summary(all_payloads):
    rows = [compute_single_bagua_contribution(payload) for payload in all_payloads.values()]
    summary_df = pd.DataFrame(rows)
    total_profit = float(summary_df['profit'].sum()) if len(summary_df) > 0 else 0.0
    total_profit_abs = float(summary_df['profit'].abs().sum()) if len(summary_df) > 0 else 0.0
    summary_df['profit_share_pct'] = summary_df['profit'].map(
        lambda x: 0.0 if total_profit_abs == 0 else x / total_profit_abs * 100
    )

    starts = [r['date_range']['start'] for r in rows if r.get('date_range', {}).get('start') is not None]
    ends = [r['date_range']['end'] for r in rows if r.get('date_range', {}).get('end') is not None]
    date_range = {
        'start': min(starts) if starts else None,
        'end': max(ends) if ends else None,
        'text': '--' if not starts or not ends else f"{min(starts):%Y-%m-%d} → {max(ends):%Y-%m-%d}",
    }

    total_signal = int(summary_df['signal_count'].sum()) if len(summary_df) > 0 else 0
    total_buy = int(summary_df['buy_count'].sum()) if len(summary_df) > 0 else 0
    avg_buy_ret = float(summary_df.apply(lambda r: r['avg_buy_ret'] * r['buy_count'], axis=1).sum() / total_buy) if total_buy > 0 else 0.0
    win_rate = float(summary_df.apply(lambda r: r['win_rate'] * r['buy_count'], axis=1).sum() / total_buy) if total_buy > 0 else 0.0

    dataset_key = None
    for payload in all_payloads.values():
        dataset = payload.get('dataset') or {}
        if dataset.get('key'):
            dataset_key = dataset['key']
            break

    meta_total_return = 0.0
    meta_final_capital = 0.0
    meta_trade_count = total_buy
    meta_candidates = [payload.get('meta') for payload in all_payloads.values() if isinstance(payload.get('meta'), dict)]

    if dataset_key == 'baseline':
        # v3 架构: snapshot 每卦 meta 含真实综合回测 final_capital/total_return, 直接读
        unified_meta = next(
            (item for item in meta_candidates if item.get('final_capital') is not None),
            None,
        )
        if unified_meta is not None:
            meta_final_capital = float(unified_meta.get('final_capital', 0.0) or 0.0)
            meta_total_return = float(unified_meta.get('total_return', 0.0) or 0.0)
        else:
            # v2 兼容: per-gua 独立跑 (老 snapshot), 每卦有 init_capital, 累加近似
            baseline_meta = next(
                (
                    item for item in meta_candidates
                    if item.get('init_capital') is not None and item.get('trade_count') is None
                ),
                None,
            )
            if baseline_meta is None:
                baseline_meta = next((item for item in meta_candidates if item.get('init_capital') is not None), None)
            init_capital = float((baseline_meta or {}).get('init_capital', 0.0) or 0.0)
            if init_capital > 0:
                meta_final_capital = init_capital + total_profit
                meta_total_return = (meta_final_capital / init_capital - 1) * 100
    else:
        meta = max(meta_candidates, key=lambda item: float(item.get('final_capital', 0.0)), default=None)
        if meta_candidates:
            init_capital = float(meta.get('init_capital', 0.0) or 0.0)
            meta_final_capital = float(meta.get('final_capital', 0.0) or 0.0)
            if init_capital > 0 and meta_final_capital > 0:
                meta_final_capital = init_capital + total_profit if len(meta_candidates) > 1 else meta_final_capital
                meta_total_return = (meta_final_capital / init_capital - 1) * 100
            else:
                meta_total_return = float(meta.get('total_return', 0.0) or 0.0)

    return {
        'summary_df': summary_df.sort_values(['profit', 'buy_count', 'signal_count'], ascending=[False, False, False]).reset_index(drop=True),
        'date_range': date_range,
        'total_signal_count': total_signal,
        'total_buy_count': total_buy,
        'total_profit': total_profit,
        'avg_buy_ret': avg_buy_ret,
        'win_rate': win_rate,
        'meta_total_return': meta_total_return,
        'meta_final_capital': meta_final_capital,
        'meta_trade_count': meta_trade_count,
    }


@st.cache_data(ttl=300)
def load_market_proxy_index(index_name='中证1000', start_date=None, end_date=None):
    """加载用于市场爻核对的真实指数日线"""
    market = load_market_bagua_visual(start_date=start_date, end_date=end_date)
    if len(market) == 0:
        return pd.DataFrame()

    index_col_map = {
        '中证1000': 'csi1000_close',
        '沪深300': 'hs300_close',
        '中证500': 'csi500_close',
        '全A': 'allA_close',
        '上证': 'sh_close',
        '深证': 'sz_close',
    }
    target_col = index_col_map.get(index_name, 'csi1000_close')

    cross = load_daily_cross_section().copy()
    if len(cross) == 0 or target_col not in cross.columns:
        return pd.DataFrame()

    cols = ['date', target_col]
    for extra in ['sh_close', 'sz_close', 'hs300_close', 'csi500_close', 'csi1000_close', 'allA_close']:
        if extra not in cols and extra in cross.columns:
            cols.append(extra)
    idx = cross[cols].drop_duplicates('date').copy()
    idx['date'] = pd.to_datetime(idx['date'], format='mixed')
    idx = idx.sort_values('date').reset_index(drop=True)
    idx['close'] = pd.to_numeric(idx[target_col], errors='coerce')
    idx = idx.dropna(subset=['close']).copy()
    idx['open'] = idx['close'].shift(1)
    idx['open'] = idx['open'].fillna(idx['close'])
    idx['high'] = idx[['open', 'close']].max(axis=1)
    idx['low'] = idx[['open', 'close']].min(axis=1)
    idx['index_name'] = index_name

    out = idx[['date', 'open', 'high', 'low', 'close', 'index_name']].copy()
    if start_date is not None:
        out = out[out['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        out = out[out['date'] <= pd.to_datetime(end_date)]

    return out.reset_index(drop=True)


@st.cache_data(ttl=300)
def load_market_bagua_visual(start_date=None, end_date=None):
    """加载市场卦可视化数据"""
    numeric_cols = [
        'market_open_proxy', 'market_high_proxy', 'market_low_proxy', 'market_close_proxy',
        'market_trend_55', 'market_trend_anchor_120', 'market_speed_20', 'enhanced_breadth_momo',
        'limit_heat', 'limit_quality', 'seg_id', 'seg_day', 'changed',
        'yao_1', 'yao_2', 'yao_3',
    ]
    return _load_regime_visual(load_market_bagua, numeric_cols, start_date=start_date, end_date=end_date)


MULTI_SCALE_GUA_PATH = os.path.join(DATA_DIR, 'foundation', 'multi_scale_gua_daily.csv')

GUA_MEANINGS_ZH = {
    '111': '乾 疯牛主升',
    '110': '兑 牛末滞涨',
    '101': '离 下跌护盘',
    '100': '震 崩盘加速',
    '011': '巽 底部爆发',
    '010': '坎 反弹乏力',
    '001': '艮 熊底异动',
    '000': '坤 深熊探底',
}


def _load_multi_scale_raw():
    if not os.path.exists(MULTI_SCALE_GUA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(MULTI_SCALE_GUA_PATH, encoding='utf-8-sig',
                     dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    for c in ['d_gua', 'm_gua', 'y_gua']:
        df[c] = df[c].fillna('').astype(str).str.zfill(3).replace('000', '000')
    return df


@st.cache_data(ttl=300)
def load_day_gua_visual(start_date=None, end_date=None):
    """加载日卦 (v10 日线 位/势/变 三爻) 可视化数据"""
    numeric_cols = ['close', 'd_trend', 'd_mf', 'd_pos', 'd_spd', 'd_acc']
    df = _load_regime_visual(_load_multi_scale_raw, numeric_cols,
                             start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return df
    df['gua_code'] = df['d_gua']
    df['gua_name'] = df['d_gua'].map(lambda g: GUA_MEANINGS_ZH.get(g, '')[:1] if g else '')
    df['gua_meaning'] = df['gua_code'].map(lambda g: GUA_MEANINGS_ZH.get(g, ''))
    df['yao_pos'] = df['d_pos']
    df['yao_spd'] = df['d_spd']
    df['yao_acc'] = df['d_acc']
    df['trend'] = df['d_trend']
    df['main_force'] = df['d_mf']
    # 页面兼容别名 (old yao_day/week/month)
    df['yao_day'] = df['d_acc']
    df['yao_week'] = df['d_spd']
    df['yao_month'] = df['d_pos']
    df['changed'] = (df['gua_code'] != df['gua_code'].shift()).astype(int)
    df.loc[df.index[0], 'changed'] = 0
    df['seg_id'] = df['changed'].cumsum() + 1
    df['seg_day'] = df.groupby('seg_id').cumcount() + 1
    df['prev_gua'] = df['gua_code'].shift()
    return df.reset_index(drop=True)


@st.cache_data(ttl=300)
def build_day_gua_segments_summary(start_date=None, end_date=None):
    df = load_day_gua_visual(start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return pd.DataFrame()
    rows = []
    for seg_id, grp in df.groupby('seg_id', sort=True):
        gua = grp['gua_code'].iloc[0]
        name = grp['gua_name'].iloc[0]
        days = len(grp)
        close_start = grp['close'].iloc[0]
        close_end = grp['close'].iloc[-1]
        seg_ret = (close_end / close_start - 1) * 100 if close_start and not pd.isna(close_start) else pd.NA
        rows.append({
            'seg_id': int(seg_id),
            '开始日': grp['date'].iloc[0].strftime('%Y-%m-%d'),
            '结束日': grp['date'].iloc[-1].strftime('%Y-%m-%d'),
            '卦码': gua,
            '卦名': name,
            '卦意': GUA_MEANINGS_ZH.get(gua, ''),
            '持续天数': days,
            '段内涨跌幅%': round(float(seg_ret), 2) if pd.notna(seg_ret) else None,
            'trend均': round(float(grp['trend'].mean()), 1) if grp['trend'].notna().any() else None,
            '主力均': round(float(grp['main_force'].mean()), 1) if grp['main_force'].notna().any() else None,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def build_day_gua_summary(start_date=None, end_date=None):
    df = load_day_gua_visual(start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return pd.DataFrame()
    cnt = df.groupby('gua_code').size().reset_index(name='天数')
    cnt['卦名'] = cnt['gua_code'].map(GUA_MEANINGS_ZH)
    total = cnt['天数'].sum()
    cnt['占比%'] = (cnt['天数'] / total * 100).round(2) if total > 0 else 0
    order = ['111', '110', '101', '100', '011', '010', '001', '000']
    cnt['_order'] = cnt['gua_code'].map({g: i for i, g in enumerate(order)})
    cnt = cnt.sort_values('_order').drop(columns=['_order']).reset_index(drop=True)
    return cnt


@st.cache_data(ttl=300)
def build_market_bagua_segments_summary(start_date=None, end_date=None):
    """汇总市场卦 segment 信息，供页面核对使用"""
    df = load_market_bagua_visual(start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return pd.DataFrame()

    grouped = df.groupby('seg_id', sort=True)
    summary = grouped.agg(
        开始日=('date', 'min'),
        结束日=('date', 'max'),
        卦码=('gua_code', 'last'),
        卦名=('gua_name', 'last'),
        持续天数=('seg_day', 'max'),
        变卦标记数=('changed', 'sum'),
        起始价=('market_close_proxy', 'first'),
        结束价=('market_close_proxy', 'last'),
        速度均值=('market_speed_20', 'mean'),
        广度均值=('enhanced_breadth_momo', 'mean'),
        热度均值=('limit_heat', 'mean'),
        质量均值=('limit_quality', 'mean'),
    ).reset_index()
    summary['段内涨跌幅%'] = (summary['结束价'] / summary['起始价'] - 1.0) * 100.0
    summary['开始日'] = pd.to_datetime(summary['开始日']).dt.strftime('%Y-%m-%d')
    summary['结束日'] = pd.to_datetime(summary['结束日']).dt.strftime('%Y-%m-%d')
    return summary.sort_values('seg_id', ascending=False).reset_index(drop=True)


@st.cache_data(ttl=300)
def build_market_bagua_gua_summary(start_date=None, end_date=None):
    """汇总区间内市场卦分布"""
    df = load_market_bagua_visual(start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return pd.DataFrame()

    out = df.groupby(['gua_code', 'gua_name'], sort=True).size().reset_index(name='天数')
    out['占比%'] = out['天数'] / len(df) * 100.0
    return out.sort_values(['天数', 'gua_code'], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(ttl=300)
def build_market_bagua_change_windows(window=10, start_date=None, end_date=None):
    """提取变卦事件窗口摘要"""
    df = load_market_bagua_visual(start_date=start_date, end_date=end_date)
    if len(df) == 0:
        return pd.DataFrame()

    events = df[df['changed'] == 1].copy()
    if len(events) == 0:
        return pd.DataFrame()

    rows = []
    for idx, row in events.iterrows():
        left = max(0, idx - window)
        right = min(len(df) - 1, idx + window)
        window_df = df.iloc[left:right + 1].copy()
        prev_close = df.iloc[left]['market_close_proxy'] if left < len(df) else pd.NA
        next_close = df.iloc[right]['market_close_proxy'] if right < len(df) else pd.NA
        before_ret = (row['market_close_proxy'] / prev_close - 1.0) * 100.0 if pd.notna(prev_close) and prev_close else pd.NA
        after_ret = (next_close / row['market_close_proxy'] - 1.0) * 100.0 if pd.notna(next_close) and row['market_close_proxy'] else pd.NA
        rows.append({
            '日期': row['date'].strftime('%Y-%m-%d'),
            '前卦': row.get('prev_gua', ''),
            '现卦': row.get('gua_code', ''),
            '卦名': row.get('gua_name', ''),
            '卦意': GUA_MEANINGS.get(str(row.get('gua_code', '')).zfill(3), row.get('gua_name', '')),
            '窗口长度': len(window_df),
            '前看涨跌幅%': before_ret,
            '后看涨跌幅%': after_ret,
            '当日速度': row.get('market_speed_20', pd.NA),
            '当日广度': row.get('enhanced_breadth_momo', pd.NA),
            '当日热度': row.get('limit_heat', pd.NA),
            '当日质量': row.get('limit_quality', pd.NA),
            'segment': row.get('seg_id', pd.NA),
        })
    return pd.DataFrame(rows)


