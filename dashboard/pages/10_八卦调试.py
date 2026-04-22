# -*- coding: utf-8 -*-
"""
页面10: 八卦调试 — 八矩阵总览
"""
import sys
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from dashboard.components.data_loader import (
    BAGUA_ORDER,
    GUA_NAMES,
    TIAN_GUA_NAMES,
    REN_GUA_NAMES,
    DI_GUA_NAMES,
    build_all_bagua_debug_payload_for_dataset,
    compute_bagua_dashboard_summary,
    get_bagua_debug_dataset_config,
)
import experiment_gua as eg
import backtest_8gua as b8

MATRIX_BAGUA_ORDER = ['000', '001', '010', '011', '111', '110', '101', '100']
BAGUA_LAYOUT_ROWS = [['111'], ['110', '011'], ['101', '010'], ['100', '001'], ['000']]
DATASET_KEYS = ['formal', 'test', 'baseline']

DUI_TEST_PRIORITY_BG = '#b91c1c'
DUI_TEST_OBSERVE_BG = '#ca8a04'
DUI_TEST_AVOID_BG = '#15803d'

LI_TEST_PRIORITY_BG = '#b91c1c'
LI_TEST_OBSERVE_BG = '#ca8a04'
LI_TEST_AVOID_BG = '#15803d'

TEST_TIER1_BG = '#b91c1c'
TEST_TIER2_BG = '#ca8a04'
TEST_TIER3_BG = '#15803d'


def _apply_test_grade_overrides(payload: dict, tier1_threshold: float = 5.0, tier2_threshold: float = 0.0) -> pd.DataFrame:
    matrix_df = payload['matrix_df'].copy()
    dataset = (payload.get('dataset') or {}).get('key')
    if dataset != 'test':
        return matrix_df

    def pick_rank(row):
        if int(row.get('signal_count', 0) or 0) <= 0:
            return pd.Series({
                'semantic_bucket': 'empty',
                'rank_tier': 'empty',
                'rank_label': '空白',
                'rank_order': -1,
                'rank_reason': '无信号，保持空白',
                'bg_color': row.get('bg_color'),
            })
        score = row['signal_avg_ret']
        if pd.isna(score):
            score = -999.0
        if score > tier1_threshold:
            return pd.Series({
                'semantic_bucket': 'tier1',
                'rank_tier': 'tier1',
                'rank_label': '1等',
                'rank_order': 3,
                'rank_reason': f'收益分层：大于 {tier1_threshold}% 归为 1等',
                'bg_color': TEST_TIER1_BG,
            })
        if score >= tier2_threshold:
            return pd.Series({
                'semantic_bucket': 'tier2',
                'rank_tier': 'tier2',
                'rank_label': '2等',
                'rank_order': 2,
                'rank_reason': f'收益分层：{tier2_threshold}% 到 {tier1_threshold}% 归为 2等',
                'bg_color': TEST_TIER2_BG,
            })
        return pd.Series({
            'semantic_bucket': 'tier3',
            'rank_tier': 'tier3',
            'rank_label': '3等',
            'rank_order': 1,
            'rank_reason': f'收益分层：小于 {tier2_threshold}% 归为 3等',
            'bg_color': TEST_TIER3_BG,
        })

    ranked = matrix_df.apply(pick_rank, axis=1)
    for col in ranked.columns:
        matrix_df[col] = ranked[col]
    return matrix_df


st.set_page_config(page_title="八卦调试", page_icon="☷", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── 全局基底 ── */
    .stApp { background: linear-gradient(165deg, #07080d 0%, #0d1117 40%, #0f1318 100%); }
    .stApp > header { background: transparent !important; }
    section[data-testid="stSidebar"] { background: #0a0c12 !important; border-right: 1px solid rgba(212,168,83,0.08); }

    /* ── 标题系统 ── */
    h1 { font-family: 'Noto Sans SC', sans-serif !important; font-weight: 700 !important;
         background: linear-gradient(135deg, #d4a853 0%, #f0d48a 50%, #c9973b 100%);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         letter-spacing: 2px; }
    h2, h3 { font-family: 'Noto Sans SC', sans-serif !important; font-weight: 600 !important;
             color: #e8dcc8 !important; letter-spacing: 1px; }
    h4 { font-family: 'Noto Sans SC', sans-serif !important; color: #b8a88a !important; }
    p, span, label, .stMarkdown { font-family: 'Noto Sans SC', sans-serif !important; }

    /* ── 指标卡片 ── */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem; line-height: 1.15; font-weight: 600;
        color: #f0e6d2 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Noto Sans SC', sans-serif !important;
        font-size: 0.82rem; font-weight: 500;
        color: #8b7e6a !important; letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20,22,30,0.9), rgba(15,17,24,0.95));
        border: 1px solid rgba(212,168,83,0.12);
        border-radius: 10px; padding: 14px 12px 10px !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3), inset 0 1px 0 rgba(212,168,83,0.05);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(212,168,83,0.28);
        box-shadow: 0 4px 20px rgba(212,168,83,0.08), inset 0 1px 0 rgba(212,168,83,0.1);
    }

    /* ── 按钮 ── */
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #b8860b, #d4a853) !important;
        color: #0a0c12 !important; font-weight: 600 !important;
        border: none !important; border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(212,168,83,0.25);
        transition: all 0.3s ease;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 4px 20px rgba(212,168,83,0.4);
        transform: translateY(-1px);
    }
    button[data-testid="stBaseButton-secondary"] {
        background: rgba(20,22,30,0.8) !important;
        color: #b8a88a !important;
        border: 1px solid rgba(212,168,83,0.15) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        border-color: rgba(212,168,83,0.35) !important;
        color: #d4a853 !important;
    }

    /* ── 分割线 ── */
    hr { border: none !important; height: 1px !important;
         background: linear-gradient(90deg, transparent, rgba(212,168,83,0.2), transparent) !important;
         margin: 2rem 0 !important; }

    /* ── 表格 ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(212,168,83,0.1); border-radius: 10px;
        overflow: hidden;
    }

    /* ── 表单 ── */
    div[data-testid="stForm"] {
        background: rgba(15,17,24,0.6);
        border: 1px solid rgba(212,168,83,0.08);
        border-radius: 12px; padding: 1.2rem;
    }

    /* ── 输入框 ── */
    input[type="number"], div[data-testid="stSelectbox"] > div {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Caption 样式 ── */
    .stCaption, div[data-testid="stCaptionContainer"] {
        color: #6b5e4a !important; font-size: 0.8rem !important;
    }

    /* ── 章节标题装饰 ── */
    .section-title {
        font-family: 'Noto Sans SC', sans-serif;
        font-size: 1.25rem; font-weight: 600; color: #e8dcc8;
        padding: 10px 0 8px 0; letter-spacing: 1.5px;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #d4a853, transparent) 1;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<h1 style="margin-bottom:2px;">☷ 八卦调试</h1>', unsafe_allow_html=True)
st.caption("八个目标卦同时展示，支持正式策略数据 / 测试策略数据 / 裸跑基准数据三套口径切换。")


def _build_matrix_html(matrix_df: pd.DataFrame, compact: bool = False) -> str:
    prefix = 'mini-' if compact else ''
    ren_label = (lambda g: REN_GUA_NAMES.get(g, g)[:2]) if compact else (lambda g: REN_GUA_NAMES.get(g, g))
    di_label = (lambda g: DI_GUA_NAMES.get(g, g)[:2]) if compact else (lambda g: DI_GUA_NAMES.get(g, g))
    header_cells = ''.join(
        f"<div class='{prefix}hdr {prefix}col'>{di_label(g)}</div>" for g in MATRIX_BAGUA_ORDER
    )
    body_rows = []
    for ren_gua in MATRIX_BAGUA_ORDER:
        row_html = [f"<div class='{prefix}hdr {prefix}row'>{ren_label(ren_gua)}</div>"]
        row_df = matrix_df[matrix_df['ren_gua'] == ren_gua].set_index('di_gua')
        for di_gua in MATRIX_BAGUA_ORDER:
            row = row_df.loc[di_gua]
            parts = str(row['display_text']).split('\n')
            line1 = parts[0] if len(parts) > 0 else ''
            line2 = parts[1] if len(parts) > 1 else ''
            line3 = parts[2] if len(parts) > 2 else ''
            line3_cls = f"{prefix}line3"
            if bool(row.get('is_ranked_pair', False)):
                line3_cls += ' ranked-gua'
            row_html.append(
                "<div class='{prefix}cell {extra_cls}' style='background-color:{bg};'>"
                "<div class='{prefix}line1'>{line1}</div>"
                "<div class='{prefix}line2'>{line2}</div>"
                "<div class='{line3_cls}'>{line3}</div>"
                "</div>".format(
                    prefix=prefix,
                    bg=row['bg_color'],
                    line1=line1,
                    line2=line2,
                    line3=line3,
                    line3_cls=line3_cls,
                    extra_cls='has-buy' if int(row.get('buy_count', 0) or 0) > 0 else 'no-buy'
                )
            )
        body_rows.append(f"<div class='{prefix}matrix-row'>" + ''.join(row_html) + "</div>")

    return f"""
    <style>
    .matrix-wrap {{ overflow-x:auto; margin-top: 8px; }}
    .matrix-grid {{ min-width: 1040px; }}
    .matrix-row {{ display:grid; grid-template-columns: 120px repeat(8, 1fr); gap:6px; margin-bottom:6px; }}
    .hdr {{ border-radius:8px; background: linear-gradient(145deg, #161a24, #111520); color:#c9b896;
            display:flex; align-items:center; justify-content:center; text-align:center;
            padding:6px 6px; font-weight:600; min-height:72px; font-size:14px;
            font-family:'Noto Sans SC', sans-serif; letter-spacing:1px;
            border: 1px solid rgba(212,168,83,0.08);
            text-shadow: 0 1px 3px rgba(0,0,0,0.5); }}
    .hdr.row {{ justify-content:flex-start; padding-left:12px; }}
    .cell {{ border-radius:10px; color:#f0e8d8; min-height:72px; padding:6px 6px 4px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
             border: 1px solid rgba(255,255,255,0.04);
             transition: transform 0.2s ease, box-shadow 0.2s ease; }}
    .cell:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08); }}
    .cell.no-buy {{ opacity: 0.4; filter: saturate(0.6) brightness(0.85); }}
    .cell.has-buy {{ opacity: 1; }}
    .line1 {{ font-family:'JetBrains Mono', monospace; font-size:13px; font-weight:600; line-height:1.1; text-shadow: 0 1px 2px rgba(0,0,0,0.4); }}
    .line2 {{ font-family:'JetBrains Mono', monospace; font-size:12px; margin-top:6px; opacity:0.88; white-space:nowrap; }}
    .line3 {{ font-family:'Noto Sans SC', sans-serif; font-size:11px; margin-top:5px; opacity:0.85; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .line3.ranked-gua {{ display:inline-block; width:fit-content; max-width:100%; padding:1px 0; }}

    .mini-matrix-wrap {{ overflow-x:auto; margin-top: 6px; }}
    .mini-matrix-grid {{ min-width: 420px; }}
    .mini-matrix-row {{ display:grid; grid-template-columns: 54px repeat(8, 1fr); gap:2px; margin-bottom:2px; }}
    .mini-hdr {{ border-radius:6px; background: linear-gradient(145deg, #161a24, #111520); color:#c9b896;
                 display:flex; align-items:center; justify-content:center; text-align:center;
                 padding:2px; font-weight:600; min-height:44px; font-size:13px;
                 font-family:'Noto Sans SC', sans-serif;
                 border: 1px solid rgba(212,168,83,0.06); }}
    .mini-row {{ justify-content:center; padding-left:0; }}
    .mini-cell {{ border-radius:6px; color:#f0e8d8; min-height:48px; padding:3px 2px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
                  display:flex; flex-direction:column; align-items:center; justify-content:flex-start;
                  transition: transform 0.15s ease; }}
    .mini-cell:hover {{ transform: scale(1.05); }}
    .mini-cell.no-buy {{ opacity: 0.4; filter: saturate(0.6) brightness(0.85); }}
    .mini-cell.has-buy {{ opacity: 1; }}
    .mini-line1 {{ font-family:'JetBrains Mono', monospace; font-size:11px; font-weight:600; line-height:1.0; width:100%; text-align:center; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mini-line2 {{ font-family:'JetBrains Mono', monospace; font-size:11px; margin-top:2px; opacity:0.88; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:100%; text-align:center; }}
    .mini-line3 {{ font-family:'Noto Sans SC', sans-serif; font-size:11px; margin-top:2px; opacity:0.88; width:100%; text-align:center; letter-spacing:0.1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .mini-line3.ranked-gua {{ display:inline-block; width:auto; max-width:calc(100% - 6px); padding:1px 0; }}
    </style>
    <div class='{prefix}matrix-wrap'>
      <div class='{prefix}matrix-grid'>
        <div class='{prefix}matrix-row'>
          <div class='{prefix}hdr'></div>
          {header_cells}
        </div>
        {''.join(body_rows)}
      </div>
    </div>
    """


def _format_detail(df: pd.DataFrame, date_cols: list[str], percent_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    if len(df) == 0:
        return df
    out = df.copy()
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors='coerce').dt.strftime('%Y-%m-%d')
    for col in percent_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: '--' if pd.isna(x) else f'{x:+.1f}%')
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: '--' if pd.isna(x) else f'{x:,.0f}')
    return out


def _filter_non_empty(df: pd.DataFrame, only_non_empty: bool, only_with_trade: bool) -> pd.DataFrame:
    out = df.copy()
    if only_non_empty:
        out = out[(out['signal_count'] > 0) | (out['buy_count'] > 0)]
    if only_with_trade:
        out = out[out['buy_count'] > 0]
    return out


def _render_mini_panel(payload: dict):
    metric = payload['contribution_metrics']
    st.markdown(f"### {payload['target_name']}")
    top_metrics = st.columns(7)
    top_metrics[0].metric('裸全', f"{metric['signal_count']}")
    top_metrics[1].metric(f'{ds_label}可', f"{metric.get('can_buy_count', metric['signal_count'])}")
    top_metrics[2].metric(f'{ds_label}买', f"{metric['buy_count']}")
    top_metrics[3].metric('胜率', f"{metric['win_rate']:.1f}%")
    top_metrics[4].metric('均收', f"{metric['avg_buy_ret']:+.1f}%")
    top_metrics[5].metric('利润(万)', f"{metric['profit']/10000:+,.2f}")
    top_metrics[6].metric('占比', f"{metric.get('profit_share_pct', 0.0):+.2f}%")
    _tg = payload.get('target_gua')
    _t1 = st.session_state.get(f'bagua_tier1_{_tg}', 5.0)
    _t2 = st.session_state.get(f'bagua_tier2_{_tg}', 0.0)
    mini_matrix_df = _apply_test_grade_overrides(payload, _t1, _t2)
    st.markdown(_build_matrix_html(mini_matrix_df, compact=True), unsafe_allow_html=True)


if 'bagua_debug_dataset' not in st.session_state:
    st.session_state['bagua_debug_dataset'] = 'formal'

SELL_MODE_LABELS = {
    'bear': 'bear(保守)', 'bull': 'bull(耐心)',
    'kun_bear': 'kun_bear(反转)', 'dui_bear': 'dui_bear(快出)',
    'qian_bull': 'qian_bull(纯牛)', 'trend_break70': 'trend70(跌破)',
}


def _get_spec_default(gua, key, fallback=None):
    spec = eg.GUA_EXPERIMENT_SPECS.get(gua, {})
    naked = spec.get('naked_cfg', {})
    return naked.get(key, fallback)


def _get_default_sell(gua):
    spec = eg.GUA_EXPERIMENT_SPECS.get(gua, {})
    return spec.get('naked_cfg', {}).get('sell', 'bear')


def _get_sell_options(gua):
    spec = eg.GUA_EXPERIMENT_SPECS.get(gua, {})
    cases = spec.get('sell_cases', [])
    if cases:
        return list(cases)
    return ['bear', 'bull', 'trend_break70']


def _get_default_buy_case(gua):
    spec = eg.GUA_EXPERIMENT_SPECS.get(gua, {})
    naked = spec.get('naked_cfg', {})
    fields = spec.get('fields', {})
    buy_mode_field = fields.get('buy_mode')
    cross_field = fields.get('cross')
    if buy_mode_field and naked.get(buy_mode_field) == 'double_rise':
        return 'double_rise'
    if cross_field:
        threshold = naked.get(cross_field, 20)
        return f'cross@{threshold}'
    return 'double_rise'


def _get_buy_case_options(gua):
    spec = eg.GUA_EXPERIMENT_SPECS.get(gua, {})
    cases = spec.get('buy_cases', [])
    if cases:
        return [case[0] for case in cases]
    return ['double_rise']



def _save_strategy_to_code():
    filepath = os.path.join(PROJECT_ROOT, 'backtest_8gua.py')
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_strategy = False
    current_gua = None
    for i, line in enumerate(lines):
        if 'GUA_STRATEGY' in line and '=' in line and '{' in line:
            in_strategy = True
        if not in_strategy:
            continue
        if line.strip() == '}':
            break

        m = re.match(r"\s*'(\d{3})':\s*\{", line)
        if m:
            current_gua = m.group(1)
        if current_gua is None:
            continue

        gua = current_gua
        pool = int(st.session_state.get(f'bagua_pool_{gua}', -250))
        verify = st.session_state.get(f'bagua_pool_verify_{gua}', False)
        verify_val = int(st.session_state.get(f'bagua_verify_val_{gua}', 0))
        sell_method = st.session_state.get(f'bagua_sell_{gua}', 'bear')
        trend_max_str = str(verify_val) if verify else 'None'
        retail_max_str = '500' if verify else 'None'
        pd_on = st.session_state.get(f'bagua_pool_days_{gua}', False)
        pd_min = int(st.session_state.get(f'bagua_pool_days_min_{gua}', 1))
        pd_max = int(st.session_state.get(f'bagua_pool_days_max_{gua}', 7))
        pd_min_str = str(pd_min) if pd_on else 'None'
        pd_max_str = str(pd_max) if pd_on else 'None'

        if "'trend_max':" in line:
            lines[i] = re.sub(r"'trend_max':\s*(?:None|\d+)", f"'trend_max': {trend_max_str}", lines[i])
        if "'retail_max':" in line:
            lines[i] = re.sub(r"'retail_max':\s*(?:None|\d+)", f"'retail_max': {retail_max_str}", lines[i])
        if "'sell':" in line:
            lines[i] = re.sub(r"'sell':\s*'[^']*'", f"'sell': '{sell_method}'", lines[i])
        if "'pool_threshold':" in line:
            lines[i] = re.sub(r"'pool_threshold':\s*-?\d+", f"'pool_threshold': {pool}", lines[i])
        if "'pool_days_min':" in line:
            lines[i] = re.sub(r"'pool_days_min':\s*(?:None|\d+)", f"'pool_days_min': {pd_min_str}", lines[i])
        if "'pool_days_max':" in line:
            lines[i] = re.sub(r"'pool_days_max':\s*(?:None|\d+)", f"'pool_days_max': {pd_max_str}", lines[i])

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)


if 'bagua_pool_global' not in st.session_state:
    st.session_state['bagua_pool_global'] = -250
for _gua in BAGUA_ORDER:
    k_verify = f'bagua_pool_verify_{_gua}'
    if k_verify not in st.session_state:
        st.session_state[k_verify] = False
    k_verify_val = f'bagua_verify_val_{_gua}'
    if k_verify_val not in st.session_state:
        st.session_state[k_verify_val] = 0
    k_t1 = f'bagua_tier1_{_gua}'
    if k_t1 not in st.session_state:
        st.session_state[k_t1] = 5.0
    k_t2 = f'bagua_tier2_{_gua}'
    if k_t2 not in st.session_state:
        st.session_state[k_t2] = 0.0
    k_sell = f'bagua_sell_{_gua}'
    if k_sell not in st.session_state:
        st.session_state[k_sell] = _get_default_sell(_gua)
    k_pd = f'bagua_pool_days_{_gua}'
    if k_pd not in st.session_state:
        st.session_state[k_pd] = (_gua == '100')
    k_pd_min = f'bagua_pool_days_min_{_gua}'
    if k_pd_min not in st.session_state:
        st.session_state[k_pd_min] = 1
    k_pd_max = f'bagua_pool_days_max_{_gua}'
    if k_pd_max not in st.session_state:
        st.session_state[k_pd_max] = 7
    k_bc = f'bagua_buy_case_{_gua}'
    if k_bc not in st.session_state:
        st.session_state[k_bc] = _get_default_buy_case(_gua)

# 裸跑参数初始化（独立于测试参数）
if 'bagua_bl_pool_global' not in st.session_state:
    st.session_state['bagua_bl_pool_global'] = -250
for _gua in BAGUA_ORDER:
    if f'bagua_bl_sell_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_sell_{_gua}'] = _get_default_sell(_gua)
    if f'bagua_bl_buy_case_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_buy_case_{_gua}'] = _get_default_buy_case(_gua)
    if f'bagua_bl_pool_verify_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_pool_verify_{_gua}'] = False
    if f'bagua_bl_verify_val_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_verify_val_{_gua}'] = 0
    if f'bagua_bl_pool_days_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_pool_days_{_gua}'] = (_gua == '100')
    if f'bagua_bl_pool_days_min_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_pool_days_min_{_gua}'] = 1
    if f'bagua_bl_pool_days_max_{_gua}' not in st.session_state:
        st.session_state[f'bagua_bl_pool_days_max_{_gua}'] = 7

DATASET_LABEL_MAP = {'formal': '正式', 'test': '测试', 'baseline': '裸跑'}

button_cols = st.columns(3)
for idx, dataset_key in enumerate(DATASET_KEYS):
    cfg = get_bagua_debug_dataset_config(dataset_key)
    if button_cols[idx].button(cfg['label'], use_container_width=True, type='primary' if st.session_state['bagua_debug_dataset'] == dataset_key else 'secondary'):
        st.session_state['bagua_debug_dataset'] = dataset_key

current_dataset = st.session_state['bagua_debug_dataset']
ds_label = DATASET_LABEL_MAP[current_dataset]

if current_dataset == 'test':
    with st.form('test_strategy_form'):
        st.markdown('#### 策略参数')

        # ── 一、入池条件（裸跑的一部分） ──
        st.markdown('##### 一、入池条件')
        pool_col, _ = st.columns([1, 3])
        with pool_col:
            st.number_input(
                '入池阈值(全局)', value=int(st.session_state['bagua_pool_global']),
                step=50, format='%d', key='in_pool_global',
            )
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                st.checkbox(
                    '二次验证', value=bool(st.session_state[f'bagua_pool_verify_{_gua}']),
                    key=f'in_verify_{_gua}',
                )
                st.number_input(
                    '验证阈值', value=int(st.session_state.get(f'bagua_verify_val_{_gua}', 0)),
                    step=10, format='%d', key=f'in_verify_val_{_gua}',
                )
                st.checkbox(
                    '池内天数', value=bool(st.session_state[f'bagua_pool_days_{_gua}']),
                    key=f'in_pd_{_gua}',
                )
                st.number_input(
                    '最小天数', value=int(st.session_state[f'bagua_pool_days_min_{_gua}']),
                    min_value=0, step=1, format='%d', key=f'in_pd_min_{_gua}',
                )
                st.number_input(
                    '最大天数', value=int(st.session_state[f'bagua_pool_days_max_{_gua}']),
                    min_value=0, step=1, format='%d', key=f'in_pd_max_{_gua}',
                )

        # ── 二、买入过滤（测试专用，裸跑不含） ──
        st.markdown('##### 二、买入过滤')
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                st.number_input(
                    '1等阈值(%)', value=float(st.session_state[f'bagua_tier1_{_gua}']),
                    step=1.0, format='%.1f', key=f'in_t1_{_gua}',
                )
                st.number_input(
                    '2等阈值(%)', value=float(st.session_state[f'bagua_tier2_{_gua}']),
                    step=1.0, format='%.1f', key=f'in_t2_{_gua}',
                )

        # ── 三、买入方式（裸跑的一部分） ──
        st.markdown('##### 三、买入方式')
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                buy_opts = _get_buy_case_options(_gua)
                cur_buy = st.session_state[f'bagua_buy_case_{_gua}']
                if cur_buy not in buy_opts:
                    cur_buy = buy_opts[0]
                st.selectbox(
                    '买入方式', buy_opts,
                    index=buy_opts.index(cur_buy),
                    key=f'in_buy_{_gua}',
                )

        # ── 四、卖出方式（裸跑的一部分） ──
        st.markdown('##### 四、卖出方式')
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                sell_opts = _get_sell_options(_gua)
                cur_sell = st.session_state[f'bagua_sell_{_gua}']
                if cur_sell not in sell_opts:
                    cur_sell = sell_opts[0]
                st.selectbox(
                    '卖出方式', sell_opts,
                    index=sell_opts.index(cur_sell),
                    format_func=lambda x: SELL_MODE_LABELS.get(x, x),
                    key=f'in_sell_{_gua}',
                )

        btn_cols = st.columns(2)
        with btn_cols[0]:
            submitted = st.form_submit_button('运行', use_container_width=True, type='primary')
        with btn_cols[1]:
            saved = st.form_submit_button('保存到代码', use_container_width=True, type='secondary')
    if submitted or saved:
        st.session_state['bagua_pool_global'] = st.session_state['in_pool_global']
        for _gua in MATRIX_BAGUA_ORDER:
            st.session_state[f'bagua_pool_verify_{_gua}'] = st.session_state[f'in_verify_{_gua}']
            st.session_state[f'bagua_verify_val_{_gua}'] = st.session_state[f'in_verify_val_{_gua}']
            st.session_state[f'bagua_tier1_{_gua}'] = st.session_state[f'in_t1_{_gua}']
            st.session_state[f'bagua_tier2_{_gua}'] = st.session_state[f'in_t2_{_gua}']
            st.session_state[f'bagua_sell_{_gua}'] = st.session_state[f'in_sell_{_gua}']
            st.session_state[f'bagua_buy_case_{_gua}'] = st.session_state[f'in_buy_{_gua}']
            st.session_state[f'bagua_pool_days_{_gua}'] = st.session_state[f'in_pd_{_gua}']
            st.session_state[f'bagua_pool_days_min_{_gua}'] = st.session_state[f'in_pd_min_{_gua}']
            st.session_state[f'bagua_pool_days_max_{_gua}'] = st.session_state[f'in_pd_max_{_gua}']
    if saved:
        _save_strategy_to_code()
        st.success('策略参数已保存到 backtest_8gua.py')

elif current_dataset == 'baseline':
    with st.form('baseline_strategy_form'):
        st.markdown('#### 裸跑策略参数')

        # ── 一、入池条件 ──
        st.markdown('##### 一、入池条件')
        pool_col, _ = st.columns([1, 3])
        with pool_col:
            st.number_input(
                '入池阈值(全局)', value=int(st.session_state['bagua_bl_pool_global']),
                step=50, format='%d', key='bl_pool_global',
            )
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                st.checkbox(
                    '二次验证', value=bool(st.session_state[f'bagua_bl_pool_verify_{_gua}']),
                    key=f'bl_verify_{_gua}',
                )
                st.number_input(
                    '验证阈值', value=int(st.session_state.get(f'bagua_bl_verify_val_{_gua}', 0)),
                    step=10, format='%d', key=f'bl_verify_val_{_gua}',
                )
                st.checkbox(
                    '池内天数', value=bool(st.session_state[f'bagua_bl_pool_days_{_gua}']),
                    key=f'bl_pd_{_gua}',
                )
                st.number_input(
                    '最小天数', value=int(st.session_state[f'bagua_bl_pool_days_min_{_gua}']),
                    min_value=0, step=1, format='%d', key=f'bl_pd_min_{_gua}',
                )
                st.number_input(
                    '最大天数', value=int(st.session_state[f'bagua_bl_pool_days_max_{_gua}']),
                    min_value=0, step=1, format='%d', key=f'bl_pd_max_{_gua}',
                )

        # ── 二、买入方式 ──
        st.markdown('##### 二、买入方式')
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                buy_opts = _get_buy_case_options(_gua)
                cur_buy = st.session_state[f'bagua_bl_buy_case_{_gua}']
                if cur_buy not in buy_opts:
                    cur_buy = buy_opts[0]
                st.selectbox(
                    '买入方式', buy_opts,
                    index=buy_opts.index(cur_buy),
                    key=f'bl_buy_{_gua}',
                )

        # ── 三、卖出方式 ──
        st.markdown('##### 三、卖出方式')
        gua_cols = st.columns(8)
        for i, _gua in enumerate(MATRIX_BAGUA_ORDER):
            with gua_cols[i]:
                st.markdown(f"**{TIAN_GUA_NAMES[_gua]}**")
                sell_opts = _get_sell_options(_gua)
                cur_sell = st.session_state[f'bagua_bl_sell_{_gua}']
                if cur_sell not in sell_opts:
                    cur_sell = sell_opts[0]
                st.selectbox(
                    '卖出方式', sell_opts,
                    index=sell_opts.index(cur_sell),
                    format_func=lambda x: SELL_MODE_LABELS.get(x, x),
                    key=f'bl_sell_{_gua}',
                )

        bl_submitted = st.form_submit_button('运行', use_container_width=True, type='primary')
    if bl_submitted:
        st.session_state['bagua_bl_pool_global'] = st.session_state['bl_pool_global']
        for _gua in MATRIX_BAGUA_ORDER:
            st.session_state[f'bagua_bl_pool_verify_{_gua}'] = st.session_state[f'bl_verify_{_gua}']
            st.session_state[f'bagua_bl_verify_val_{_gua}'] = st.session_state[f'bl_verify_val_{_gua}']
            st.session_state[f'bagua_bl_sell_{_gua}'] = st.session_state[f'bl_sell_{_gua}']
            st.session_state[f'bagua_bl_buy_case_{_gua}'] = st.session_state[f'bl_buy_{_gua}']
            st.session_state[f'bagua_bl_pool_days_{_gua}'] = st.session_state[f'bl_pd_{_gua}']
            st.session_state[f'bagua_bl_pool_days_min_{_gua}'] = st.session_state[f'bl_pd_min_{_gua}']
            st.session_state[f'bagua_bl_pool_days_max_{_gua}'] = st.session_state[f'bl_pd_max_{_gua}']


dataset_cfg = get_bagua_debug_dataset_config(current_dataset)
if current_dataset == 'test':
    pool_thresholds = {gua: int(st.session_state['bagua_pool_global']) for gua in BAGUA_ORDER}
    buy_cases = {gua: st.session_state.get(f'bagua_buy_case_{gua}', _get_default_buy_case(gua)) for gua in BAGUA_ORDER}
    sell_methods = {gua: st.session_state[f'bagua_sell_{gua}'] for gua in BAGUA_ORDER}
    tier1_thresholds = {gua: float(st.session_state[f'bagua_tier1_{gua}']) for gua in BAGUA_ORDER}
    tier2_thresholds = {gua: float(st.session_state[f'bagua_tier2_{gua}']) for gua in BAGUA_ORDER}
    pool_days_mins = {}
    pool_days_maxs = {}
    for gua in BAGUA_ORDER:
        if st.session_state.get(f'bagua_pool_days_{gua}', False):
            pool_days_mins[gua] = int(st.session_state[f'bagua_pool_days_min_{gua}'])
            pool_days_maxs[gua] = int(st.session_state[f'bagua_pool_days_max_{gua}'])
        else:
            pool_days_mins[gua] = None
            pool_days_maxs[gua] = None
elif current_dataset == 'baseline':
    pool_thresholds = {gua: int(st.session_state['bagua_bl_pool_global']) for gua in BAGUA_ORDER}
    buy_cases = {gua: st.session_state.get(f'bagua_bl_buy_case_{gua}', _get_default_buy_case(gua)) for gua in BAGUA_ORDER}
    sell_methods = {gua: st.session_state[f'bagua_bl_sell_{gua}'] for gua in BAGUA_ORDER}
    tier1_thresholds = None
    tier2_thresholds = None
    pool_days_mins = {}
    pool_days_maxs = {}
    for gua in BAGUA_ORDER:
        if st.session_state.get(f'bagua_bl_pool_days_{gua}', False):
            pool_days_mins[gua] = int(st.session_state[f'bagua_bl_pool_days_min_{gua}'])
            pool_days_maxs[gua] = int(st.session_state[f'bagua_bl_pool_days_max_{gua}'])
        else:
            pool_days_mins[gua] = None
            pool_days_maxs[gua] = None
else:
    pool_thresholds = None
    buy_cases = None
    sell_methods = None
    tier1_thresholds = None
    tier2_thresholds = None
    pool_days_mins = None
    pool_days_maxs = None
test_buy_case = None  # 保留变量以兼容后续使用
all_payloads = build_all_bagua_debug_payload_for_dataset(
    current_dataset,
    test_buy_case=None,
    test_pool_thresholds=pool_thresholds,
    test_buy_cases=buy_cases,
    test_sell_methods=sell_methods,
    test_tier1_thresholds=tier1_thresholds,
    test_tier2_thresholds=tier2_thresholds,
    test_pool_days_mins=pool_days_mins,
    test_pool_days_maxs=pool_days_maxs,
)
summary = compute_bagua_dashboard_summary(all_payloads)

st.info(f"当前口径：{dataset_cfg['label']}｜{dataset_cfg['description']}")
if current_dataset == 'test':
    fallback_names = [payload['target_name'] for payload in all_payloads.values() if payload.get('fallback_dataset_key')]
    if fallback_names:
        st.caption(f"说明：{', '.join(fallback_names)} 暂无独立测试配置，复用正式策略数据。")
elif current_dataset == 'baseline':
    st.caption('说明：裸跑基准可通过上方参数调整入池条件、买入方式和卖出方式。')

metric_cols = st.columns(6)
metric_cols[0].metric('数据口径', dataset_cfg['label'])
metric_cols[1].metric('数据区间', summary['date_range']['text'])
metric_cols[2].metric('裸全', f"{summary['total_signal_count']}")
metric_cols[3].metric(f'{ds_label}买', f"{summary['meta_trade_count']}")
metric_cols[4].metric('总收益率', f"{summary['meta_total_return']:+,.1f}%")
metric_cols[5].metric('终值', f"{summary['meta_final_capital']/10000:,.1f}万")


st.markdown('---')
st.markdown('<div class="section-title">☰ 八卦位置总览</div>', unsafe_allow_html=True)
for row in BAGUA_LAYOUT_ROWS:
    if len(row) == 1:
        left, center, right = st.columns([1, 2.2, 1])
        with center:
            _render_mini_panel(all_payloads[row[0]])
    else:
        left, mid, right = st.columns([1, 0.08, 1])
        with left:
            _render_mini_panel(all_payloads[row[0]])
        with right:
            _render_mini_panel(all_payloads[row[1]])

st.markdown('---')
st.markdown('<div class="section-title">☲ 各卦贡献</div>', unsafe_allow_html=True)
contrib_df = summary['summary_df'].copy()
contrib_df['目标卦'] = contrib_df['target_name']
contrib_df['裸全'] = contrib_df['signal_count']
contrib_df[f'{ds_label}可'] = contrib_df['can_buy_count']
contrib_df[f'{ds_label}买'] = contrib_df['buy_count']
contrib_df['买入胜率'] = contrib_df['win_rate'].map(lambda x: f'{x:.1f}%')
contrib_df['买入均收'] = contrib_df['avg_buy_ret'].map(lambda x: f'{x:+.1f}%')
contrib_df['利润(万)'] = contrib_df['profit'].map(lambda x: f'{x/10000:+,.2f}')
contrib_df['利润占比'] = contrib_df['profit_share_pct'].map(lambda x: f'{x:+.2f}%')
st.dataframe(
    contrib_df[['目标卦', '裸全', f'{ds_label}可', f'{ds_label}买', '买入胜率', '买入均收', '利润(万)', '利润占比']],
    use_container_width=True,
    hide_index=True,
    height=360,
)

st.markdown('---')
st.markdown('<div class="section-title">☵ 钻取明细</div>', unsafe_allow_html=True)
detail_target = st.radio('钻取目标卦', MATRIX_BAGUA_ORDER, horizontal=True, format_func=lambda g: TIAN_GUA_NAMES.get(g, g))
payload = all_payloads[detail_target]
_dt1 = st.session_state.get(f'bagua_tier1_{detail_target}', 5.0)
_dt2 = st.session_state.get(f'bagua_tier2_{detail_target}', 0.0)
matrix_df = _apply_test_grade_overrides(payload, _dt1, _dt2)

view_mode = st.radio('展示模式', ['矩阵', '表格'], horizontal=True)
col_opt1, col_opt2 = st.columns(2)
with col_opt1:
    only_non_empty = st.checkbox('仅显示非空格子', value=False)
with col_opt2:
    only_with_trade = st.checkbox('仅显示有买入格子', value=False)

filtered_df = _filter_non_empty(matrix_df, only_non_empty, only_with_trade)
sub_metrics = st.columns(6)
sub_metrics[0].metric('目标卦', payload['target_name'])
sub_metrics[1].metric('裸全', f"{int(matrix_df['signal_count'].sum())}")
sub_metrics[2].metric(f'{ds_label}可', f"{int(payload['contribution_metrics'].get('can_buy_count', int(matrix_df.get('can_buy_count', pd.Series(dtype=int)).sum() if 'can_buy_count' in matrix_df.columns else matrix_df['signal_count'].sum())))}")
sub_metrics[3].metric(f'{ds_label}买', f"{int(matrix_df['buy_count'].sum())}")
sub_metrics[4].metric('利润(万)', f"{payload['contribution_metrics'].get('profit', 0.0)/10000:+,.2f}")
sub_metrics[5].metric('胜率', f"{payload['contribution_metrics'].get('win_rate', 0.0):.1f}%")

if view_mode == '矩阵':
    st.markdown('#### 64 卦联动矩阵')
    matrix_for_view = matrix_df if not (only_non_empty or only_with_trade) else matrix_df.copy()
    if only_non_empty or only_with_trade:
        hidden_pairs = set(filtered_df[['ren_gua', 'di_gua']].itertuples(index=False, name=None))
        matrix_for_view = matrix_for_view.copy()
        mask = matrix_for_view.apply(lambda r: (r['ren_gua'], r['di_gua']) not in hidden_pairs, axis=1)
        matrix_for_view.loc[mask, 'display_text'] = '0/0\n--/--\n卦意隐藏'
        matrix_for_view.loc[mask, 'bg_color'] = '#111827'
        matrix_for_view.loc[mask, 'buy_count'] = 0
        if 'is_ranked_pair' in matrix_for_view.columns:
            matrix_for_view.loc[mask, 'is_ranked_pair'] = False
    st.markdown(_build_matrix_html(matrix_for_view), unsafe_allow_html=True)
else:
    st.markdown('#### 64 格展开表')
    table_df = filtered_df.copy()
    table_df['人卦'] = table_df['ren_name']
    table_df['地卦'] = table_df['di_name']
    table_df['区域'] = table_df['zone_name']
    table_df['裸全'] = table_df['signal_count']
    table_df[f'{ds_label}可'] = table_df.get('can_buy_count', table_df['signal_count'])
    table_df[f'{ds_label}买'] = table_df['buy_count']
    table_df['买入转化率'] = table_df['buy_rate_pct'].map(lambda x: f'{x:.1f}%')
    table_df['全量均收'] = table_df['signal_avg_ret'].map(lambda x: '--' if pd.isna(x) else f'{x:+.1f}%')
    table_df['买入均收'] = table_df['buy_avg_ret'].map(lambda x: '--' if pd.isna(x) else f'{x:+.1f}%')
    table_df['评级'] = table_df['rank_label']
    table_df['卦义初判'] = table_df['semantic_bucket'].map({'tier1': '1等', 'tier2': '2等', 'tier3': '3等', 'empty': '空白'}).fillna(table_df['semantic_bucket'])
    table_df['修正动作'] = table_df['rank_adjustment']
    table_df['评级原因'] = table_df['rank_reason']
    st.dataframe(
        table_df[['人卦', '地卦', '区域', '评级', '卦义初判', '修正动作', '裸全', f'{ds_label}可', f'{ds_label}买', '买入转化率', '全量均收', '买入均收', '评级原因', 'zone_reason']]
        .sort_values([f'{ds_label}买', '裸全'], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

st.markdown('#### 单格钻取')
sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    ren_gua = st.selectbox('人卦', MATRIX_BAGUA_ORDER, format_func=lambda g: REN_GUA_NAMES.get(g, g))
with sel_col2:
    di_gua = st.selectbox('地卦', MATRIX_BAGUA_ORDER, format_func=lambda g: DI_GUA_NAMES.get(g, g), index=MATRIX_BAGUA_ORDER.index('111'))

selected_cell = matrix_df[(matrix_df['ren_gua'] == ren_gua) & (matrix_df['di_gua'] == di_gua)].iloc[0]
st.caption(
    f"当前格：{selected_cell['ren_name']} × {selected_cell['di_name']} | "
    f"{selected_cell['zone_name']} | 评级 {selected_cell['rank_label']} | {selected_cell['rank_reason']}"
)
st.caption(f"证据：{selected_cell['evidence_text']}")

signals_detail = payload['detail_signals']
can_buy_detail = payload.get('detail_can_buy_signals', payload['detail_signals'])
if len(signals_detail) > 0:
    signals_detail = signals_detail[
        (signals_detail['ren_gua'] == ren_gua) &
        (signals_detail['di_gua'] == di_gua)
    ].copy()
if len(can_buy_detail) > 0:
    can_buy_detail = can_buy_detail[
        (can_buy_detail['ren_gua'] == ren_gua) &
        (can_buy_detail['di_gua'] == di_gua)
    ].copy()
trades_detail = payload['detail_trades']
if len(trades_detail) > 0:
    trades_detail = trades_detail[
        (trades_detail['ren_gua'] == ren_gua) &
        (trades_detail['di_gua'] == di_gua)
    ].copy()

left, mid, right = st.columns(3)
with left:
    st.markdown(f"#### 裸全明细 ({len(signals_detail)})")
    if len(signals_detail) == 0:
        st.info('该格暂无裸全信号')
    else:
        show_cols = [c for c in ['code', 'signal_date', 'buy_date', 'sell_date', 'actual_ret', 'sell_method', 'grade', 'is_skip'] if c in signals_detail.columns]
        display = _format_detail(signals_detail[show_cols], ['signal_date', 'buy_date', 'sell_date'], ['actual_ret'], [])
        st.dataframe(display, use_container_width=True, hide_index=True, height=360)
with mid:
    st.markdown(f"#### {ds_label}可买明细 ({len(can_buy_detail)})")
    if len(can_buy_detail) == 0:
        st.info(f'该格暂无{ds_label}可买信号')
    else:
        show_cols = [c for c in ['code', 'signal_date', 'buy_date', 'sell_date', 'actual_ret', 'sell_method', 'grade', 'is_skip'] if c in can_buy_detail.columns]
        display = _format_detail(can_buy_detail[show_cols], ['signal_date', 'buy_date', 'sell_date'], ['actual_ret'], [])
        st.dataframe(display, use_container_width=True, hide_index=True, height=360)
with right:
    st.markdown(f"#### {ds_label}买入明细 ({len(trades_detail)})")
    if len(trades_detail) == 0:
        st.info(f'该格暂无{ds_label}买入')
    else:
        show_cols = [c for c in ['code', 'buy_date', 'sell_date', 'ret_pct', 'hold_days', 'sell_method', 'grade', 'profit'] if c in trades_detail.columns]
        display = _format_detail(trades_detail[show_cols], ['buy_date', 'sell_date'], ['ret_pct'], ['profit'])
        st.dataframe(display, use_container_width=True, hide_index=True, height=360)
