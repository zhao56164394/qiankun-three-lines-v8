# -*- coding: utf-8 -*-
"""八卦分析法通用实验入口（卦作为变量）"""
import argparse
import contextlib
import copy
import io
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)

import backtest_8gua as b8
import backtest_baseline as bb
import pandas as pd


GUA_LABELS = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}

RUNTIME_CACHE: Optional[Dict[str, Any]] = None
PAYLOAD_CACHE: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, Any]] = {}
BASELINE_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_layer', 'data', 'bagua_debug_baseline_snapshot.json')


GUA_EXPERIMENT_SPECS: Dict[str, Dict[str, Any]] = {
    '110': {
        'name': '兑',
        'base_cfg': {
            'dui_buy': True,
            'dui_buy_mode': 'double_rise',
            'dui_cross_threshold': 20,
            'dui_allow_di_gua': None,
            'dui_exclude_ren_gua': set(),
            'dui_market_stock_whitelist': {
                '000': {'000', '001', '010'},
                '001': {'000', '010'},
                '010': set(),
                '011': {'010', '011'},
                '100': set(),
                '101': set(),
                '110': {'110'},
                '111': set(),
            },
            'pool_threshold': -250,
            'sell': 'dui_bear',
        },
        'naked_cfg': {
            'dui_buy': True,
            'dui_buy_mode': 'double_rise',
            'dui_cross_threshold': 20,
            'dui_allow_di_gua': None,
            'dui_exclude_ren_gua': set(),
            'dui_market_stock_whitelist': None,
            'pool_threshold': -250,
            'sell': 'dui_bear',
        },
        'fields': {
            'market': 'dui_market_stock_whitelist',
            'stock': 'dui_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'dui_buy_mode',
            'cross': 'dui_cross_threshold',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'dui_buy_mode': 'double_rise', 'dui_cross_threshold': 20}),
            ('cross@11', {'dui_buy_mode': 'cross', 'dui_cross_threshold': 11}),
            ('cross@20', {'dui_buy_mode': 'cross', 'dui_cross_threshold': 20}),
            ('cross@30', {'dui_buy_mode': 'cross', 'dui_cross_threshold': 30}),
            ('cross@40', {'dui_buy_mode': 'cross', 'dui_cross_threshold': 40}),
        ],
        'sell_cases': ['dui_bear', 'bear', 'bull', 'trend_break70'],
    },
    '001': {
        'name': '艮',
        'base_cfg': {
            'gen_buy': True,
            'gen_exclude_ren_gua': {'001'},
            'gen_allow_di_gua': {'000', '010'},
            'pool_threshold': -250,
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            'sell': 'bear',
        },
        'naked_cfg': {
            'gen_buy': True,
            'gen_exclude_ren_gua': set(),
            'gen_allow_di_gua': None,
            'pool_threshold': -250,
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            'sell': 'bear',
        },
        'fields': {
            'market': 'gen_exclude_ren_gua',
            'stock': 'gen_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'gen_buy_mode',
            'cross': 'gen_cross_threshold',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'gen_buy_mode': 'double_rise', 'gen_cross_threshold': 20}),
            ('cross@20', {'gen_buy_mode': 'cross', 'gen_cross_threshold': 20}),
            ('cross@30', {'gen_buy_mode': 'cross', 'gen_cross_threshold': 30}),
            ('cross@40', {'gen_buy_mode': 'cross', 'gen_cross_threshold': 40}),
            ('cross@50', {'gen_buy_mode': 'cross', 'gen_cross_threshold': 50}),
        ],
        'sell_cases': ['bear', 'bull', 'trend_break70'],
    },
    '000': {
        'name': '坤',
        'base_cfg': {
            'kun_buy': True,
            'kun_exclude_ren_gua': set(),
            'kun_allow_di_gua': None,
            'pool_threshold': -250,
            'kun_buy_mode': 'double_rise',
            'sell': 'bear',
        },
        'naked_cfg': {
            'kun_buy': True,
            'kun_exclude_ren_gua': set(),
            'kun_allow_di_gua': None,
            'pool_threshold': -250,
            'kun_buy_mode': 'double_rise',
            'sell': 'bear',
        },
        'fields': {
            'market': 'kun_exclude_ren_gua',
            'stock': 'kun_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'kun_buy_mode',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'kun_buy_mode': 'double_rise'}),
        ],
        'sell_cases': ['bear', 'bull', 'trend_break70'],
    },
    '100': {
        'name': '震',
        'base_cfg': {
            'zhen_buy': True,
            'zhen_buy_mode': 'double_rise',
            'zhen_cross_threshold': 20,
            'zhen_exclude_ren_gua': {'001', '011'},
            'zhen_allow_di_gua': None,
            'pool_threshold': -250,
            'sell': 'bull',
        },
        'naked_cfg': {
            'zhen_buy': True,
            'zhen_buy_mode': 'double_rise',
            'zhen_cross_threshold': 20,
            'zhen_exclude_ren_gua': set(),
            'zhen_allow_di_gua': None,
            'pool_threshold': -250,
            'pool_days_min': 1,
            'pool_days_max': 7,
            'sell': 'bear',
        },
        'fields': {
            'market': 'zhen_exclude_ren_gua',
            'stock': 'zhen_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'zhen_buy_mode',
            'cross': 'zhen_cross_threshold',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'zhen_buy_mode': 'double_rise', 'zhen_cross_threshold': 20}),
            ('cross@20', {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 20}),
            ('cross@30', {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 30}),
            ('cross@40', {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 40}),
        ],
        'sell_cases': ['bull', 'bear', 'trend_break70'],
    },
    '101': {
        'name': '离',
        'base_cfg': {
            'li_buy': True,
            'li_buy_mode': 'double_rise',
            'li_cross_threshold': 20,
            'li_exclude_ren_gua': {'001'},
            'li_allow_di_gua': {'000'},
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'naked_cfg': {
            'li_buy': True,
            'li_buy_mode': 'double_rise',
            'li_cross_threshold': 20,
            'li_exclude_ren_gua': set(),
            'li_allow_di_gua': None,
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'fields': {
            'market': 'li_exclude_ren_gua',
            'stock': 'li_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'li_buy_mode',
            'cross': 'li_cross_threshold',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'li_buy_mode': 'double_rise', 'li_cross_threshold': 20}),
            ('cross@20', {'li_buy_mode': 'cross', 'li_cross_threshold': 20}),
            ('cross@30', {'li_buy_mode': 'cross', 'li_cross_threshold': 30}),
            ('cross@40', {'li_buy_mode': 'cross', 'li_cross_threshold': 40}),
        ],
        'sell_cases': ['bear', 'bull', 'trend_break70'],
    },
    '010': {
        'name': '坎',
        'base_cfg': {
            'grades': {'A+', 'A', 'B+', 'B', 'B-'},
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'naked_cfg': {
            'grades': {'A+', 'A', 'B+', 'B', 'B-', 'C', 'D', 'F'},
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'fields': {
            'market': None,
            'stock': None,
            'pool': 'pool_threshold',
            'buy_mode': None,
            'cross': None,
        },
        'stock_mode': None,
        'buy_cases': [
            ('double_rise', {}),
        ],
        'sell_cases': ['bear', 'bull', 'trend_break70'],
    },
    '011': {
        'name': '巽',
        'base_cfg': {
            'xun_buy': 'double_rise',
            'xun_buy_param': 11,
            'xun_allow_di_gua': {'010'},
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'naked_cfg': {
            'xun_buy': 'double_rise',
            'xun_buy_param': 11,
            'xun_allow_di_gua': None,
            'pool_threshold': -250,
            'sell': 'bear',
        },
        'fields': {
            'market': None,
            'stock': 'xun_allow_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': 'xun_buy',
            'cross': 'xun_buy_param',
        },
        'stock_mode': 'allow',
        'buy_cases': [
            ('double_rise', {'xun_buy': 'double_rise', 'xun_buy_param': 11}),
            ('double_rise@20', {'xun_buy': 'double_rise', 'xun_buy_param': 20}),
            ('cross@11', {'xun_buy': 'cross', 'xun_buy_param': 11}),
            ('cross@20', {'xun_buy': 'cross', 'xun_buy_param': 20}),
        ],
        'sell_cases': ['bear', 'bull', 'trend_break70'],
    },
    '111': {
        'name': '乾',
        'base_cfg': {
            'qian_buy': True,
            'qian_cross_threshold': 60,
            'qian_exclude_ren_gua': set(),
            'qian_exclude_di_gua': {'101', '111'},
            'sell': 'qian_bull',
        },
        'naked_cfg': {
            'qian_buy': True,
            'qian_cross_threshold': 60,
            'qian_exclude_ren_gua': set(),
            'qian_exclude_di_gua': set(),
            'pool_threshold': -250,
            'sell': 'qian_bull',
        },
        'fields': {
            'market': 'qian_exclude_ren_gua',
            'stock': 'qian_exclude_di_gua',
            'pool': 'pool_threshold',
            'buy_mode': None,
            'cross': 'qian_cross_threshold',
        },
        'stock_mode': 'exclude',
        'buy_cases': [
            ('cross@40', {'qian_cross_threshold': 40}),
            ('cross@50', {'qian_cross_threshold': 50}),
            ('cross@60', {'qian_cross_threshold': 60}),
            ('cross@70', {'qian_cross_threshold': 70}),
        ],
        'sell_cases': ['qian_bull', 'bull', 'bear'],
    },
}


def freeze_value(value: Any):
    if isinstance(value, set):
        return tuple(sorted(value))
    if isinstance(value, dict):
        return tuple(sorted((k, freeze_value(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(freeze_value(v) for v in value)
    return value


def make_cfg_key(gua: str, cfg: Dict[str, Any]) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    return gua, tuple(sorted((k, freeze_value(v)) for k, v in cfg.items()))


def classify_dui_pair_tier(ren_gua: str, di_gua: str) -> str:
    ren_gua = str(ren_gua).zfill(3)
    di_gua = str(di_gua).zfill(3)
    if (ren_gua, di_gua) in {
        ('000', '000'), ('000', '001'), ('000', '010'),
        ('001', '000'), ('001', '010'), ('110', '110'),
    }:
        return 'tier1'
    if (ren_gua, di_gua) in {
        ('011', '010'), ('011', '011'),
    }:
        return 'tier2'
    return 'tier3'


def apply_dui_rank_fields(sig: pd.DataFrame) -> pd.DataFrame:
    if len(sig) == 0 or 'tian_gua' not in sig.columns:
        return sig
    out = sig.copy()
    mask = out['tian_gua'].astype(str).str.zfill(3) == '110'
    if not mask.any():
        return out

    dui_rows = out.loc[mask].copy()
    pair_avg = dui_rows.groupby(['ren_gua', 'di_gua'], dropna=False)['actual_ret'].mean()

    def classify_rank(row):
        score = pair_avg.get((row['ren_gua'], row['di_gua']), pd.NA)
        if pd.isna(score):
            return 'tier3'
        if score > 5:
            return 'tier1'
        if score >= 0:
            return 'tier2'
        return 'tier3'

    out.loc[mask, 'rank_tier'] = out.loc[mask].apply(classify_rank, axis=1)
    out.loc[mask, 'rank_order'] = out.loc[mask, 'rank_tier'].map({'tier1': 3, 'tier2': 2, 'tier3': 1}).astype(int)
    if 'is_skip' in out.columns:
        out.loc[mask & (out['rank_order'] <= 1), 'is_skip'] = True
    return out


def apply_li_rank_fields(sig: pd.DataFrame) -> pd.DataFrame:
    if len(sig) == 0 or 'tian_gua' not in sig.columns:
        return sig
    out = sig.copy()
    mask = out['tian_gua'].astype(str).str.zfill(3) == '101'
    if not mask.any():
        return out

    li_rows = out.loc[mask].copy()
    pair_avg = li_rows.groupby(['ren_gua', 'di_gua'], dropna=False)['actual_ret'].mean()

    def classify_rank(row):
        score = pair_avg.get((row['ren_gua'], row['di_gua']), pd.NA)
        if pd.isna(score):
            return 'tier3'
        if score > 5:
            return 'tier1'
        if score >= 0:
            return 'tier2'
        return 'tier3'

    out.loc[mask, 'rank_tier'] = out.loc[mask].apply(classify_rank, axis=1)
    out.loc[mask, 'rank_order'] = out.loc[mask, 'rank_tier'].map({'tier1': 3, 'tier2': 2, 'tier3': 1}).astype(int)
    if 'is_skip' in out.columns:
        out.loc[mask & (out['rank_order'] <= 1), 'is_skip'] = True
    return out


def format_gua_set(values: Optional[Iterable[str]]) -> str:
    if values is None:
        return '不限'
    values = list(values)
    if not values:
        return '空集'
    ordered = sorted(values)
    return '/'.join(f"{GUA_LABELS.get(v, v)}({v})" for v in ordered)


def get_spec(gua: str) -> Dict[str, Any]:
    if gua not in GUA_EXPERIMENT_SPECS:
        raise ValueError(f'暂未支持卦 {gua} 的通用分析，请先补充 GUA_EXPERIMENT_SPECS')
    return GUA_EXPERIMENT_SPECS[gua]


def load_runtime_context() -> Dict[str, Any]:
    global RUNTIME_CACHE
    if RUNTIME_CACHE is not None:
        return RUNTIME_CACHE

    with contextlib.redirect_stdout(io.StringIO()):
        zz_df = b8.load_zz1000_full()
        zz1000 = b8.load_zz1000()
        stock_data = b8.load_stocks()
        stk_mf_map = b8._load_stock_main_force()
        big_cycle_context = b8.load_big_cycle_context()
        stock_bagua_map = b8.load_stock_bagua_map()

    # 天卦映射(市场卦 → 分治维度)
    from data_layer.foundation_data import load_market_bagua, load_daily_bagua
    market_bagua_df = load_market_bagua()
    tian_gua_map = {}
    for _, row in market_bagua_df.iterrows():
        tian_gua_map[str(row['date'])] = (b8._clean_gua(row['gua_code']), row.get('gua_name', ''))

    # 人卦映射(主卦/个股横截面排名)
    daily_bagua_df = load_daily_bagua()
    daily_bagua_map = {}
    for _, row in daily_bagua_df.iterrows():
        daily_bagua_map[(str(row['date']), str(row['code']).zfill(6))] = {
            'gua_code': str(row['gua_code']).zfill(3),
            'gua_name': row.get('gua_name', ''),
        }

    RUNTIME_CACHE = {
        'zz_df': zz_df,
        'zz1000': zz1000,
        'stock_data': stock_data,
        'stk_mf_map': stk_mf_map,
        'big_cycle_context': big_cycle_context,
        'stock_bagua_map': stock_bagua_map,
        'tian_gua_map': tian_gua_map,
        'daily_bagua_map': daily_bagua_map,
    }
    return RUNTIME_CACHE


def enrich_signals(sig):
    sig = sig[(sig['signal_date'] >= b8.YEAR_START) &
              (sig['signal_date'] < b8.YEAR_END)].reset_index(drop=True)
    sig = b8.build_512_rolling_pred(sig, min_hist=b8.MIN_512_SAMPLES)
    sig['grade'] = [b8.grade_signal(r['gua_yy'], r['combo_pred'])[0]
                    for _, r in sig.iterrows()]
    if b8.ACTUAL_GRADE_GUAS:
        sig = b8.build_actual_rolling_pred(sig, b8.ACTUAL_GRADE_GUAS, min_hist=3)
    return sig


def clone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cloned = {}
    for key, value in payload.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            cloned[key] = value.copy(deep=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned

def _mark_dui_double_rise(sig: pd.DataFrame, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if len(sig) == 0:
        return sig
    out = sig.copy()
    out['is_dui_double_rise'] = False
    mask = out['tian_gua'].astype(str).str.zfill(3) == '110'
    if not mask.any():
        return out
    for idx, row in out[mask].iterrows():
        df = stock_data.get(row['code'])
        if df is None or len(df) < 2:
            continue
        pos = df.index[df['date'].astype(str) == str(row['signal_date'])]
        if len(pos) == 0:
            continue
        i = int(pos[0])
        if i <= 0:
            continue
        retail = df['retail'].values
        trend = df['trend'].values
        if any(pd.isna(v) for v in [retail[i], retail[i-1], trend[i], trend[i-1]]):
            continue
        out.at[idx, 'is_dui_double_rise'] = bool(retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
    return out


def build_dui_test_baseline_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cache_key = ('110-baseline-test', tuple(sorted((k, freeze_value(v)) for k, v in cfg.items())))
    if cache_key in PAYLOAD_CACHE:
        return clone_payload(PAYLOAD_CACHE[cache_key])

    runtime = load_runtime_context()
    with open(BASELINE_SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
        snapshot = json.load(f)
    raw = (snapshot.get('payloads') or {}).get('110') or {}
    sig = pd.DataFrame(raw.get('detail_signals', [])).copy()
    for col in ['tian_gua', 'gua', 'ren_gua', 'di_gua', 'combo']:
        if col in sig.columns:
            sig[col] = sig[col].astype(str).str.zfill(3)
    if 'di_gua' not in sig.columns and 'stk_gua' in sig.columns:
        sig['di_gua'] = sig['stk_gua']
    if 'stk_gua' not in sig.columns:
        sig['stk_gua'] = sig['di_gua']
    else:
        sig['stk_gua'] = sig['stk_gua'].astype(str).str.zfill(3)
    if 'stk_yy' not in sig.columns and 'di_gua' in sig.columns:
        sig['stk_yy'] = sig['di_gua'].map(b8.to_yinyang)
    sig['tian_gua'] = '110'
    sig['gua'] = '110'
    sig['combo'] = sig['di_gua']
    sig['gua_yy'] = sig['stk_yy']
    sig['sell_method'] = 'bear'
    sig['is_skip'] = False
    sig['ren_gua_name'] = sig['ren_gua'].map(GUA_LABELS).fillna('')
    sig['di_gua_name'] = sig['di_gua'].map(GUA_LABELS).fillna('')
    sig = enrich_signals(sig)
    sig = apply_dui_rank_fields(sig)
    sig = _mark_dui_double_rise(sig, runtime['stock_data'])

    target_sig = sig.copy().reset_index(drop=True)
    non_target_sig = pd.DataFrame(columns=sig.columns)
    filtered_target_sig = target_sig[target_sig['is_dui_double_rise']].copy().reset_index(drop=True)
    merged = filtered_target_sig.sort_values('signal_date').reset_index(drop=True)

    with contextlib.redirect_stdout(io.StringIO()):
        result = bb.simulate_baseline(merged, runtime['zz_df'], max_pos=5, daily_limit=1,
                                      init_capital=b8.INIT_CAPITAL)
    trade_meta = filtered_target_sig[['code', 'buy_date', 'di_gua', 'ren_gua']].copy()
    trade_meta['buy_date'] = trade_meta['buy_date'].astype(str)
    trade_meta['di_gua'] = trade_meta['di_gua'].astype(str).str.zfill(3)
    trade_lookup = {
        (str(row['code']), str(row['buy_date']), str(row['di_gua']).zfill(3)): str(row['ren_gua']).zfill(3)
        for _, row in trade_meta.iterrows()
    }
    for t in result['trade_log']:
        sg = str(t.get('stk_gua', '???')).zfill(3)
        mg = trade_lookup.get((str(t.get('code')), str(t.get('buy_date')), sg), '???')
        t['gua'] = '110'
        t['ren_gua'] = mg
        t['di_gua'] = sg
        t['ren_gua_name'] = GUA_LABELS.get(mg, mg)
        t['di_gua_name'] = GUA_LABELS.get(sg, sg)
        t['sell_method'] = 'bear'

    stats = b8.calc_stats(result, f"{GUA_LABELS.get('110', '110')}实验")
    payload = {
        'cfg': copy.deepcopy(cfg),
        'sig': sig,
        'result': result,
        'stats': stats,
        'target_sig': target_sig,
        'non_target_sig': non_target_sig,
        'filtered_target_sig': filtered_target_sig,
    }
    PAYLOAD_CACHE[cache_key] = clone_payload(payload)
    return clone_payload(payload)


def build_payload_for_cfg(gua: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cache_key = make_cfg_key(gua, cfg)
    if cache_key in PAYLOAD_CACHE:
        return clone_payload(PAYLOAD_CACHE[cache_key])

    runtime = load_runtime_context()
    old_strat = copy.deepcopy(b8.GUA_STRATEGY[gua])
    try:
        b8.GUA_STRATEGY[gua].update(copy.deepcopy(cfg))
        with contextlib.redirect_stdout(io.StringIO()):
            sig = b8.scan_signals_8gua(
                runtime['stock_data'],
                runtime['zz1000'],
                runtime['tian_gua_map'],
                runtime['stk_mf_map'],
                big_cycle_context=runtime['big_cycle_context'],
                stock_bagua_map=runtime['stock_bagua_map'],
                daily_bagua_map=runtime.get('daily_bagua_map'),
            )
            sig = enrich_signals(sig)
            sig = apply_dui_rank_fields(sig)
            sig = apply_li_rank_fields(sig)
            result = b8.simulate_8gua(sig, runtime['zz_df'], max_pos=5, daily_limit=1,
                                      init_capital=b8.INIT_CAPITAL)
            stats = b8.calc_stats(result, f"{GUA_LABELS.get(gua, gua)}实验")
    finally:
        b8.GUA_STRATEGY[gua] = old_strat

    payload = {
        'cfg': copy.deepcopy(cfg),
        'sig': sig,
        'result': result,
        'stats': stats,
        'target_sig': sig[sig['tian_gua'] == gua].copy().reset_index(drop=True),
        'raw_target_sig': sig[sig['tian_gua'] == gua].copy().reset_index(drop=True),
        'non_target_sig': sig[sig['tian_gua'] != gua].copy().reset_index(drop=True),
    }
    PAYLOAD_CACHE[cache_key] = clone_payload(payload)
    return clone_payload(payload)


def simulate_case_from_filtered_target(gua: str, base_payload: Dict[str, Any], filtered_target_sig):
    runtime = load_runtime_context()
    merged = pd.concat([base_payload['non_target_sig'], filtered_target_sig], ignore_index=True)
    merged = merged.sort_values('signal_date').reset_index(drop=True)
    merged = apply_dui_rank_fields(merged)
    merged = apply_li_rank_fields(merged)
    with contextlib.redirect_stdout(io.StringIO()):
        result = b8.simulate_8gua(merged, runtime['zz_df'], max_pos=5, daily_limit=1,
                                  init_capital=b8.INIT_CAPITAL)
        stats = b8.calc_stats(result, f"{GUA_LABELS.get(gua, gua)}卦分析")
    return {
        'sig': merged,
        'result': result,
        'stats': stats,
        'target_sig': filtered_target_sig.copy().reset_index(drop=True),
    }


def summarize_signal_rows(rows) -> Dict[str, float]:
    if len(rows) == 0:
        return {
            'count': 0,
            'win_rate': 0,
            'avg_ret': 0,
            'median_ret': 0,
            'avg_hold': 0,
            'profit': 0,
        }
    rets = rows['actual_ret'].tolist()
    holds = rows['hold_days'].tolist()
    rets_sorted = sorted(rets)
    mid = len(rets_sorted) // 2
    if len(rets_sorted) % 2 == 0:
        median_ret = (rets_sorted[mid - 1] + rets_sorted[mid]) / 2
    else:
        median_ret = rets_sorted[mid]
    wins = sum(1 for r in rets if r > 0)
    profits = [(row['sell_price'] / row['buy_price'] - 1) for _, row in rows.iterrows()]
    return {
        'count': len(rows),
        'win_rate': wins / len(rows) * 100,
        'avg_ret': sum(rets) / len(rets),
        'median_ret': median_ret,
        'avg_hold': sum(holds) / len(holds),
        'profit': sum(profits),
    }


def summarize_target_trades(result: Dict[str, Any], gua: str) -> Dict[str, float]:
    trades = [t for t in result['trade_log'] if t.get('gua') == gua]
    if not trades:
        return {
            'count': 0,
            'win_rate': 0,
            'avg_ret': 0,
            'median_ret': 0,
            'avg_hold': 0,
            'profit': 0,
        }
    rets = [t['ret_pct'] for t in trades]
    profits = [t['profit'] for t in trades]
    holds = [t['hold_days'] for t in trades]
    rets_sorted = sorted(rets)
    mid = len(rets_sorted) // 2
    if len(rets_sorted) % 2 == 0:
        median_ret = (rets_sorted[mid - 1] + rets_sorted[mid]) / 2
    else:
        median_ret = rets_sorted[mid]
    wins = sum(1 for p in profits if p > 0)
    return {
        'count': len(trades),
        'win_rate': wins / len(trades) * 100,
        'avg_ret': sum(rets) / len(rets),
        'median_ret': median_ret,
        'avg_hold': sum(holds) / len(holds),
        'profit': sum(profits),
    }


def build_dual_view_row(label: str, note: str, payload: Dict[str, Any], gua: str) -> Dict[str, Any]:
    signal_stats = summarize_signal_rows(payload['target_sig'])
    trade_stats = summarize_target_trades(payload['result'], gua)
    return {
        'label': label,
        'note': note,
        'signal_count': signal_stats['count'],
        'signal_win_rate': signal_stats['win_rate'],
        'signal_avg_ret': signal_stats['avg_ret'],
        'signal_median_ret': signal_stats['median_ret'],
        'dui_count': trade_stats['count'],
        'dui_win_rate': trade_stats['win_rate'],
        'dui_avg_ret': trade_stats['avg_ret'],
        'dui_median_ret': trade_stats['median_ret'],
        'dui_profit': trade_stats['profit'],
        'total_return': payload['stats']['total_return'],
        'final_capital': payload['stats']['final_capital'],
        'max_dd': payload['stats']['max_dd'],
        'trade_count': payload['stats']['trade_count'],
    }


def build_market_cases() -> List[Dict[str, Any]]:
    cases = [{'label': '市场不过滤', 'excluded': set(), 'note': '只改市场层：空集'}]
    for code, label in GUA_LABELS.items():
        excluded = {code}
        cases.append({
            'label': f'排市场{label}',
            'excluded': excluded,
            'note': f'只改市场层：{format_gua_set(excluded)}',
        })
    return cases


def build_stock_cases(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases = [{'label': '个股不过滤', 'mode': 'all', 'values': None, 'note': '只改个股层：不限'}]
    for code, label in GUA_LABELS.items():
        if spec['stock_mode'] == 'allow':
            values = {code}
            cases.append({
                'label': f'仅{label}',
                'mode': 'allow',
                'values': values,
                'note': f'只改个股层：{format_gua_set(values)}',
            })
        else:
            values = {code}
            cases.append({
                'label': f'排个股{label}',
                'mode': 'exclude',
                'values': values,
                'note': f'只改个股层：{format_gua_set(values)}',
            })
    return cases


def apply_market_filter(target_sig, excluded: Iterable[str]):
    excluded = set(excluded)
    if not excluded:
        return target_sig.copy().reset_index(drop=True)
    return target_sig[~target_sig['ren_gua'].isin(excluded)].copy().reset_index(drop=True)


def apply_stock_filter(target_sig, stock_case: Dict[str, Any]):
    mode = stock_case['mode']
    values = stock_case['values']
    if mode == 'all' or values is None:
        return target_sig.copy().reset_index(drop=True)
    if mode == 'allow':
        return target_sig[target_sig['di_gua'].isin(values)].copy().reset_index(drop=True)
    return target_sig[~target_sig['di_gua'].isin(values)].copy().reset_index(drop=True)


def build_combo_row(gua: str, base_payload: Dict[str, Any], market_case: Dict[str, Any], stock_case: Dict[str, Any]) -> Dict[str, Any]:
    filtered = apply_market_filter(base_payload['target_sig'], market_case['excluded'])
    filtered = apply_stock_filter(filtered, stock_case)
    payload = simulate_case_from_filtered_target(gua, base_payload, filtered)
    row = build_dual_view_row(
        f"{market_case['label']}×{stock_case['label']}",
        f"市场层：{market_case['note'].replace('只改市场层：', '')}；个股层：{stock_case['note'].replace('只改个股层：', '')}",
        payload,
        gua,
    )
    row['market_label'] = market_case['label']
    row['stock_label'] = stock_case['label']
    return row


def summarize_group(df, group_cols, ret_col, profit_col=None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        cols = list(group_cols) + ['count', 'win_rate', 'avg_ret', 'median_ret', 'avg_hold']
        if profit_col:
            cols.append('profit')
        return pd.DataFrame(columns=cols)

    agg = df.groupby(group_cols, dropna=False).agg(
        count=(ret_col, 'size'),
        win_rate=(ret_col, lambda s: (pd.to_numeric(s, errors='coerce') > 0).mean() * 100),
        avg_ret=(ret_col, 'mean'),
        median_ret=(ret_col, 'median'),
        avg_hold=('hold_days', 'mean'),
    ).reset_index()
    if profit_col:
        profit = df.groupby(group_cols, dropna=False)[profit_col].sum().reset_index(name='profit')
        agg = agg.merge(profit, on=group_cols, how='left')
    return agg


def build_trade_detail(result: Dict[str, Any], gua: str) -> pd.DataFrame:
    trades = pd.DataFrame([t for t in result['trade_log'] if t.get('gua') == gua])
    if trades.empty:
        return trades
    for col in ['ren_gua_name', 'di_gua_name']:
        if col not in trades.columns:
            base_col = col.replace('_name', '')
            trades[col] = trades[base_col].map(GUA_LABELS).fillna('')
    return trades


def print_matrix_table(title: str, rows: List[Dict[str, Any]]):
    print('\n' + '=' * 180)
    print(title)
    print('=' * 180)
    print(f"{'联动格':<12} {'全量笔数':>8} {'全量胜率':>10} {'全量均收':>10} {'全量中位':>10} {'买入笔数':>8} {'买入胜率':>10} {'买入均收':>10} {'买入中位':>10} {'买入利润(万)':>13}")
    print('  ' + '-' * 168)
    for r in rows:
        print(
            f"{r['label']:<12} {r['signal_count']:>8} {r['signal_win_rate']:>9.1f}% "
            f"{r['signal_avg_ret']:>+9.2f} {r['signal_median_ret']:>+9.2f} "
            f"{r['trade_count']:>8} {r['trade_win_rate']:>9.1f}% {r['trade_avg_ret']:>+9.2f} "
            f"{r['trade_median_ret']:>+9.2f} {r['trade_profit']/10000:>+12.1f}"
        )
    print('-' * 180)
    for r in rows:
        print(f"- {r['label']}: 市场={r['ren_name']}({r['ren_gua']}), 个股={r['di_name']}({r['di_gua']})")




def print_dual_view_table(title: str, rows: List[Dict[str, Any]]):
    print('\n' + '=' * 180)
    print(title)
    print('=' * 180)
    print(f"{'方案':<24} {'全量笔数':>8} {'全量胜率':>10} {'全量均收':>10} {'全量中位':>10} {'买入笔数':>8} {'买入胜率':>10} {'买入均收':>10} {'买入中位':>10} {'买入利润(万)':>13} {'系统收益%':>10}")
    print('  ' + '-' * 176)
    for r in rows:
        print(
            f"{r['label']:<24} {r['signal_count']:>8} {r['signal_win_rate']:>9.1f}% "
            f"{r['signal_avg_ret']:>+9.2f} {r['signal_median_ret']:>+9.2f} "
            f"{r['dui_count']:>8} {r['dui_win_rate']:>9.1f}% {r['dui_avg_ret']:>+9.2f} "
            f"{r['dui_median_ret']:>+9.2f} {r['dui_profit']/10000:>+12.1f} {r['total_return']:>+9.1f}"
        )
    print('-' * 180)
    for r in rows:
        if r['note']:
            print(f"- {r['label']}: {r['note']}")


def run_naked(gua: str):
    spec = get_spec(gua)
    payload = build_payload_for_cfg(gua, copy.deepcopy(spec['naked_cfg']))
    row = build_dual_view_row('全裸', '按当前裸配置重跑初始入池 + 最简单买卖', payload, gua)
    print_dual_view_table(f"{spec['name']}卦第一步：全裸双视角", [row])
    return [row]


def run_market_stock_matrix(gua: str):
    spec = get_spec(gua)
    base_cfg = copy.deepcopy(spec['naked_cfg'])
    payload = build_payload_for_cfg(gua, base_cfg)

    signal_df = payload['target_sig'].copy()
    signal_df['ren_gua_name'] = signal_df['ren_gua'].map(GUA_LABELS).fillna('')
    signal_df['di_gua_name'] = signal_df['di_gua'].map(GUA_LABELS).fillna('')
    signal_summary = summarize_group(
        signal_df,
        ['ren_gua', 'ren_gua_name', 'di_gua', 'di_gua_name'],
        ret_col='actual_ret',
    )

    trade_df = build_trade_detail(payload['result'], gua)
    trade_summary = summarize_group(
        trade_df,
        ['ren_gua', 'ren_gua_name', 'di_gua', 'di_gua_name'],
        ret_col='ret_pct',
        profit_col='profit',
    )

    matrix = signal_summary.merge(
        trade_summary,
        on=['ren_gua', 'ren_gua_name', 'di_gua', 'di_gua_name'],
        how='outer',
        suffixes=('_signal', '_trade'),
    ).fillna(0)

    rows = []
    for _, row in matrix.iterrows():
        rows.append({
            'label': f"{row['ren_gua_name']}×{row['di_gua_name']}",
            'ren_gua': row['ren_gua'],
            'ren_name': row['ren_gua_name'],
            'di_gua': row['di_gua'],
            'di_name': row['di_gua_name'],
            'signal_count': int(row['count_signal']),
            'signal_win_rate': float(row['win_rate_signal']),
            'signal_avg_ret': float(row['avg_ret_signal']),
            'signal_median_ret': float(row['median_ret_signal']),
            'trade_count': int(row['count_trade']),
            'trade_win_rate': float(row['win_rate_trade']),
            'trade_avg_ret': float(row['avg_ret_trade']),
            'trade_median_ret': float(row['median_ret_trade']),
            'trade_profit': float(row.get('profit', 0) or 0),
        })
    rows.sort(key=lambda r: (r['trade_avg_ret'], r['trade_profit'], r['signal_count']), reverse=True)
    print_matrix_table(f"{spec['name']}卦人卦×地卦 64联动矩阵双视角", rows)
    return rows


def run_market(gua: str):
    spec = get_spec(gua)
    field = spec['fields']['market']
    if not field:
        raise ValueError(f"{spec['name']}卦暂不支持市场层")
    base_cfg = copy.deepcopy(spec['naked_cfg'])
    base_payload = build_payload_for_cfg(gua, base_cfg)
    rows = [build_dual_view_row('市场不过滤', '只改市场层：空集', base_payload, gua)]
    for market_case in build_market_cases()[1:]:
        filtered = apply_market_filter(base_payload['target_sig'], market_case['excluded'])
        payload = simulate_case_from_filtered_target(gua, base_payload, filtered)
        rows.append(build_dual_view_row(market_case['label'], market_case['note'], payload, gua))
    print_dual_view_table(f"{spec['name']}卦第二步：市场层双视角（同一份底表复用）", rows)
    return rows


def run_stock(gua: str):
    spec = get_spec(gua)
    field = spec['fields']['stock']
    if not field:
        raise ValueError(f"{spec['name']}卦暂不支持个股层")
    base_cfg = copy.deepcopy(spec['naked_cfg'])
    base_payload = build_payload_for_cfg(gua, base_cfg)
    rows = []
    for stock_case in build_stock_cases(spec):
        filtered = apply_stock_filter(base_payload['target_sig'], stock_case)
        payload = simulate_case_from_filtered_target(gua, base_payload, filtered)
        rows.append(build_dual_view_row(stock_case['label'], stock_case['note'], payload, gua))
    print_dual_view_table(f"{spec['name']}卦第三步：个股层双视角（同一份底表复用）", rows)
    return rows


def run_market_stock(gua: str):
    spec = get_spec(gua)
    if not spec['fields']['market'] or not spec['fields']['stock']:
        raise ValueError(f"{spec['name']}卦暂不支持市场层+个股层组合分析")
    base_cfg = copy.deepcopy(spec['naked_cfg'])
    base_payload = build_payload_for_cfg(gua, base_cfg)
    market_cases = build_market_cases()
    stock_cases = build_stock_cases(spec)
    rows = []
    for market_case in market_cases:
        for stock_case in stock_cases:
            rows.append(build_combo_row(gua, base_payload, market_case, stock_case))
    rows.sort(key=lambda r: (r['total_return'], r['dui_profit']), reverse=True)
    print_dual_view_table(f"{spec['name']}卦市场层×个股层组合双视角（同一份底表复用）", rows)
    return rows


def run_pool(gua: str):
    spec = get_spec(gua)
    base_threshold = spec['naked_cfg'].get('pool_threshold')
    rows = []
    for threshold in [None, -250, -300, -350, -400, -500]:
        cfg = copy.deepcopy(spec['naked_cfg'])
        if threshold is None:
            cfg.pop('pool_threshold', None)
            label = 'pool=None'
        else:
            cfg['pool_threshold'] = threshold
            label = f'pool={threshold}'
        payload = build_payload_for_cfg(gua, cfg)
        rows.append(build_dual_view_row(label, f'只改初始入池：{label}', payload, gua))
    print_dual_view_table(f"{spec['name']}卦第四步：初始入池阈值双视角（重跑扫描）", rows)
    return rows


def run_buy(gua: str):
    spec = get_spec(gua)
    rows = []
    for label, updates in spec['buy_cases']:
        cfg = copy.deepcopy(spec['naked_cfg'])
        cfg.update(updates)
        payload = build_payload_for_cfg(gua, cfg)
        rows.append(build_dual_view_row(label, f'只改买入层：{label}', payload, gua))
    print_dual_view_table(f"{spec['name']}卦第五步：买入层双视角", rows)
    return rows


def run_sell(gua: str):
    spec = get_spec(gua)
    rows = []
    for sell_method in spec['sell_cases']:
        cfg = copy.deepcopy(spec['naked_cfg'])
        cfg['sell'] = sell_method
        payload = build_payload_for_cfg(gua, cfg)
        rows.append(build_dual_view_row(sell_method, f'只改卖出层：{sell_method}', payload, gua))
    print_dual_view_table(f"{spec['name']}卦第六步：卖出层双视角", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description='八卦分析法通用入口')
    parser.add_argument('--gua', required=True, choices=sorted(GUA_EXPERIMENT_SPECS.keys()), help='目标卦代码')
    parser.add_argument('--layer', choices=['naked', 'market', 'stock', 'market_stock', 'market_stock_matrix', 'pool', 'buy', 'sell'], required=True, help='分析层')
    args = parser.parse_args()

    if args.layer == 'naked':
        run_naked(args.gua)
    elif args.layer == 'market':
        run_market(args.gua)
    elif args.layer == 'stock':
        run_stock(args.gua)
    elif args.layer == 'market_stock':
        run_market_stock(args.gua)
    elif args.layer == 'market_stock_matrix':
        run_market_stock_matrix(args.gua)
    elif args.layer == 'pool':
        run_pool(args.gua)
    elif args.layer == 'buy':
        run_buy(args.gua)
    elif args.layer == 'sell':
        run_sell(args.gua)


if __name__ == '__main__':
    main()
