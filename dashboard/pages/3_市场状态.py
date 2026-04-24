# -*- coding: utf-8 -*-
"""
页面3: 市场状态 — 八卦罗盘 (6+2模型) + v4.0策略提示

- 默认显示今天的状态 (三个圆盘静态)
- 选择日期范围后可回测播放 (指针动起来)
- 当前卦对应的策略配置实时提示
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dashboard.components.chart_utils import apply_dark_theme
from dashboard.components.gua_display import BAGUA_INFO
from dashboard.components.gua_compass import (
    render_unified_compass, _clean_gua_code, BAGUA_COMPASS,
)
from dashboard.components.data_loader import (
    load_backtest_8gua_extra, GUA_NAMES, GUA_COLORS,
    load_signals_by_date, load_trades_by_date,
    load_market_proxy_index,
    load_day_gua_visual, build_day_gua_segments_summary, build_day_gua_summary,
)
from dashboard.components.lwc_charts import render_market_regime_index_chart

st.set_page_config(page_title="市场状态", page_icon="☰", layout="wide")
st.title("☰ 市场状态")

# 加载数据
try:
    from data_layer.gua_data import (
        get_market_state, get_buy_filter, get_current_gua,
        load_zz1000_with_segments, load_zz1000_gua,
        gua_label,
    )
    data_ok = True
except Exception as e:
    st.error(f"数据加载失败: {e}")
    data_ok = False
    st.stop()

# === 日期 ===
try:
    zz_df = load_zz1000_gua()
    available_dates = sorted(zz_df['date'].values.tolist())
    latest_date = available_dates[-1]
except Exception:
    latest_date = None
    available_dates = []

# ================================================================
# 侧边栏: 模式选择
# ================================================================
mode = st.sidebar.radio("显示模式", ["当天状态", "回测播放"], index=0)

# 数据可用的日期范围
_min_date = pd.to_datetime(available_dates[0]) if available_dates else None
_max_date = pd.to_datetime(latest_date) if latest_date else None

if mode == "当天状态":
    date_input = st.sidebar.date_input(
        "查询日期",
        value=_max_date,
        min_value=_min_date,
        max_value=_max_date,
    )
    date_str = str(date_input)
    if date_str not in available_dates:
        earlier = [d for d in available_dates if d <= date_str]
        date_str = earlier[-1] if earlier else latest_date
    st.sidebar.info(f"数据日期: {date_str}")
else:
    default_start_idx = max(0, len(available_dates) - 120)
    bt_start = st.sidebar.date_input(
        "起始日期",
        value=pd.to_datetime(available_dates[default_start_idx]) if available_dates else None,
        min_value=_min_date,
        max_value=_max_date,
    )
    bt_end = st.sidebar.date_input(
        "结束日期",
        value=_max_date,
        min_value=_min_date,
        max_value=_max_date,
    )
    date_str = str(bt_end) if latest_date else None
    if date_str and date_str not in available_dates:
        earlier = [d for d in available_dates if d <= date_str]
        date_str = earlier[-1] if earlier else latest_date

# ================================================================
# 1. 八卦罗盘 (统一组件)
# ================================================================

try:
    if mode == "当天状态":
        curr = get_current_gua(date_str)
        state = get_market_state(date_str)

        if 'year_gua' in curr and 'month_gua' in curr and 'day_gua' in curr:
            play_data = [{
                'date': curr['date'],
                'year_gua': curr['year_gua'],
                'month_gua': curr['month_gua'],
                'day_gua': curr['day_gua'],
            }]
            compass_html = render_unified_compass(play_data)
            st.components.v1.html(compass_html, height=560, scrolling=False)

            st.markdown("### 三层详情")
            for prefix, label in [('year', '年 · 大周期'), ('month', '月 · 中趋势'), ('day', '日 · 短波动')]:
                with st.expander(f"{label}: {state[f'{prefix}_name']}"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"**状态**: {state[f'{prefix}_state']}")
                        st.markdown(f"**建议**: {state[f'{prefix}_advice']}")
                    with c2:
                        st.markdown(f"**特征**: {state[f'{prefix}_feature']}")
                        st.markdown(f"**转机**: {state[f'{prefix}_transition']}")
            strategy_gua = _clean_gua_code(curr['year_gua'])
        else:
            single_gua = _clean_gua_code(curr['gua'])
            play_data = [{
                'date': curr['date'],
                'year_gua': single_gua,
                'month_gua': single_gua,
                'day_gua': single_gua,
            }]
            compass_html = render_unified_compass(play_data)
            st.components.v1.html(compass_html, height=560, scrolling=False)

            st.markdown("### 单层大象卦详情")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**卦象**: {curr['gua_name']}")
                st.markdown(f"**状态**: {state.get('state', '-')}")
                st.markdown(f"**建议**: {state.get('advice', '-')}")
            with c2:
                st.markdown(f"**特征**: {state.get('feature', '-')}")
                st.markdown(f"**转机**: {state.get('transition', '-')}")
            strategy_gua = single_gua

        # ======== v4.0 策略提示 ========
        st.markdown("### \u2630 当前策略配置 (八卦分治 v4.0)")
        year_gua = strategy_gua
        gua_name = GUA_NAMES.get(year_gua, f'未知({year_gua})')
        gua_color = GUA_COLORS.get(year_gua, '#666')

        STRATEGY_DESC = {
            '000': {'status': '交易', 'buy': '双升+个股年卦=巽+排日乾', 'sell': 'kun_bear(反转)', 'pool': '-250', 'note': '至暗反转期，独立策略'},
            '001': {'status': '交易', 'buy': '双升+中证月卦=巽', 'sell': 'bear(保守)', 'pool': '-250', 'note': '蓄力期，趋势≤20'},
            '010': {'status': '交易', 'buy': '双升, A+/A/B+', 'sell': 'bull(耐心)', 'pool': '-400', 'note': '弱反弹，趋势≤20'},
            '011': {'status': '交易', 'buy': '双升+个股仅坎', 'sell': 'bear(保守)', 'pool': '-300', 'note': '反转期，初始入池直接控制信号量'},
            '100': {'status': '空仓', 'buy': '-', 'sell': '-', 'pool': '-400', 'note': '崩盘加速期，均-6.78%，正式仍建议空仓'},
            '101': {'status': '交易', 'buy': '双升+个股日卦=坤', 'sell': 'bear(保守)', 'pool': '-300', 'note': '护盘期，趋势≤20'},
            '110': {'status': '交易', 'buy': '上穿20+个股仅坤坎兑+排市场兑震', 'sell': 'dui_bear(快出)', 'pool': '-300', 'note': '滞涨期，初始入池直接控制信号量'},
            '111': {'status': '交易', 'buy': '上穿60+排市场离震+排个股离乾', 'sell': 'qian_bull(纯牛)', 'pool': '-250', 'note': '牛顶，独立买入体系'},
        }
        desc = STRATEGY_DESC.get(year_gua, {})

        if desc.get('status') == '空仓':
            st.error(f"🔴 当前中证大象卦 = **{gua_name}** → **空仓**，禁止买入")
            st.caption(desc.get('note', ''))
        else:
            st.success(f"🟢 当前中证大象卦 = **{gua_name}** → **可交易**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("状态", desc.get('status', '?'))
            sc2.metric("买入条件", desc.get('buy', '?'))
            sc3.metric("卖法", desc.get('sell', '?'))
            sc4.metric("入池阈值", desc.get('pool', '?'))
            st.caption(f"说明: {desc.get('note', '')}")

    else:
        bt_start_str = str(bt_start)
        bt_end_str = str(bt_end)

        seg_df = load_zz1000_with_segments()
        mask = (seg_df['date'] >= bt_start_str) & (seg_df['date'] <= bt_end_str)
        bt_data = seg_df[mask].copy()

        if len(bt_data) == 0:
            st.warning("所选日期范围无数据")
        else:
            if {'year_gua', 'month_gua', 'day_gua'}.issubset(bt_data.columns):
                day_ch = max(0, (bt_data['day_gua'] != bt_data['day_gua'].shift(1)).sum() - 1)
                month_ch = max(0, (bt_data['month_gua'] != bt_data['month_gua'].shift(1)).sum() - 1)
                year_ch = max(0, (bt_data['year_gua'] != bt_data['year_gua'].shift(1)).sum() - 1)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("交易日", f"{len(bt_data)} 天")
                c2.metric("年卦变化", f"{year_ch} 次", help="时针 — 转得最慢")
                c3.metric("月卦变化", f"{month_ch} 次", help="分针 — 中等速度")
                c4.metric("日卦变化", f"{day_ch} 次", help="秒针 — 转得最快")

                play_data = []
                for _, row in bt_data.iterrows():
                    play_data.append({
                        'date': row['date'],
                        'year_gua': _clean_gua_code(row['year_gua']),
                        'month_gua': _clean_gua_code(row['month_gua']),
                        'day_gua': _clean_gua_code(row['day_gua']),
                    })
            else:
                gua_col = 'gua'
                gua_ch = max(0, (bt_data[gua_col] != bt_data[gua_col].shift(1)).sum() - 1)
                c1, c2 = st.columns(2)
                c1.metric("交易日", f"{len(bt_data)} 天")
                c2.metric("卦变化", f"{gua_ch} 次")

                play_data = []
                for _, row in bt_data.iterrows():
                    gua_code = _clean_gua_code(row[gua_col])
                    play_data.append({
                        'date': row['date'],
                        'year_gua': gua_code,
                        'month_gua': gua_code,
                        'day_gua': gua_code,
                    })

            compass_html = render_unified_compass(play_data, initial_index=0)
            st.components.v1.html(compass_html, height=600, scrolling=False)

except Exception as e:
    st.error(f"加载市场状态失败: {e}")
    import traceback
    st.code(traceback.format_exc())

# ================================================================
# 1.5 当日选股信号 + 实际买入 (八卦分治)
# ================================================================
if mode == "当天状态":
    st.markdown("---")
    st.markdown("### 📋 当日选股信号")

    try:
        sig_data = load_signals_by_date(date_str)
        df_sig = sig_data['signals']
        trades_today = load_trades_by_date(date_str)

        if len(df_sig) == 0 and len(trades_today) == 0:
            st.info(f"{date_str} 无选股信号")
        else:
            # 汇总指标
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("原始信号", f"{sig_data['total']} 条")
            non_skip_total = sum(v['non_skip'] for v in sig_data['by_gua'].values())
            sc2.metric("非skip信号", f"{non_skip_total} 条")
            sc3.metric("实际买入", f"{len(trades_today)} 笔")
            # 分卦信号统计
            gua_summary = []
            for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
                g_info = sig_data['by_gua'].get(gua)
                if g_info and g_info['total'] > 0:
                    gua_summary.append(f"{GUA_NAMES.get(gua, gua)[:1]}{g_info['total']}")
            sc4.metric("分卦分布", ' '.join(gua_summary) if gua_summary else '-')

            # === 实际买入的交易 ===
            if len(trades_today) > 0:
                st.markdown("#### 🟢 实际买入")
                display_trades = trades_today[['code', 'buy_date', 'sell_date',
                                               'buy_price', 'sell_price', 'ret_pct',
                                               'hold_days', 'profit', 'gua',
                                               'sell_method']].copy()
                display_trades['gua'] = display_trades['gua'].map(
                    lambda x: GUA_NAMES.get(x, x))
                display_trades['buy_date'] = display_trades['buy_date'].dt.strftime('%Y-%m-%d')
                display_trades['sell_date'] = display_trades['sell_date'].dt.strftime('%Y-%m-%d')
                display_trades['ret_pct'] = display_trades['ret_pct'].apply(lambda x: f'{x:+.1f}%')
                display_trades['profit'] = display_trades['profit'].apply(lambda x: f'{x:+,.0f}')
                display_trades['buy_price'] = display_trades['buy_price'].apply(lambda x: f'{x:.2f}')
                display_trades['sell_price'] = display_trades['sell_price'].apply(lambda x: f'{x:.2f}')
                display_trades.columns = ['代码', '买入日', '卖出日', '买价', '卖价',
                                          '收益率', '持仓天', '盈亏', '卦', '卖法']
                st.dataframe(display_trades, use_container_width=True, hide_index=True)

            # === 全部信号列表 ===
            if len(df_sig) > 0:
                with st.expander(f"全部信号明细 ({len(df_sig)} 条)", expanded=False):
                    # 标记哪些信号被实际买入
                    bought_codes = set(trades_today['code'].values) if len(trades_today) > 0 else set()
                    display_sig = df_sig[['code', 'signal_date', 'buy_price',
                                          'actual_ret', 'hold_days', 'pool_retail',
                                          'is_skip', 'zz_year_gua',
                                          'sell_method']].copy()
                    display_sig['bought'] = display_sig['code'].isin(bought_codes).map(
                        {True: '✓', False: ''})
                    display_sig['zz_year_gua'] = display_sig['zz_year_gua'].map(
                        lambda x: GUA_NAMES.get(x, x))
                    display_sig['signal_date'] = display_sig['signal_date'].dt.strftime('%Y-%m-%d')
                    display_sig['actual_ret'] = display_sig['actual_ret'].apply(lambda x: f'{x:+.1f}%')
                    display_sig['buy_price'] = display_sig['buy_price'].apply(lambda x: f'{x:.2f}')
                    display_sig['pool_retail'] = display_sig['pool_retail'].apply(lambda x: f'{x:.0f}')
                    display_sig['is_skip'] = display_sig['is_skip'].map({True: 'skip', False: ''})
                    display_sig.columns = ['代码', '信号日', '买价', '预期收益',
                                           '持仓天', '入池散户', 'skip',
                                           '中证卦', '卖法', '买入']
                    st.dataframe(display_sig, use_container_width=True, hide_index=True,
                                 height=min(400, 35 * len(display_sig) + 38))

    except Exception as e:
        st.warning(f"信号数据加载失败: {e}")

# ================================================================
# 2. 买入过滤
# ================================================================
st.markdown("---")
st.markdown("### 买入信号过滤")

try:
    bf = get_buy_filter(date_str)
    if bf['can_buy']:
        st.success(f"可以买入 | {bf['reason']}")
    else:
        st.error(f"禁止买入 | {bf['reason']}")

    col1.metric("象卦", bf['gua_name'])

except Exception as e:
    st.warning(f"买入过滤数据异常: {e}")

# ================================================================
# 3. 卦象历史时间线
# ================================================================
st.markdown("---")
st.markdown("### 卦象历史 (最近60天)")

try:
    seg_df = load_zz1000_with_segments()
    recent = seg_df[seg_df['date'] <= date_str].tail(60).copy()
    recent['date'] = pd.to_datetime(recent['date'])

    for col, label in [('day_gua', '日卦'), ('month_gua', '月卦')]:
        gua_names = [BAGUA_INFO.get(_clean_gua_code(g), {}).get('name', '?') for g in recent[col]]
        gua_colors = [BAGUA_INFO.get(_clean_gua_code(g), {}).get('color', '#666') for g in recent[col]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['date'],
            y=[BAGUA_INFO.get(_clean_gua_code(g), {}).get('name', '?') for g in recent[col]],
            mode='markers+text',
            text=gua_names,
            textposition='top center',
            marker=dict(size=10, color=gua_colors),
            hovertext=[f"{d}: {n}" for d, n in zip(recent['date'].dt.strftime('%m-%d'), gua_names)],
        ))
        fig.update_layout(
            height=200, title=f'{label}变化',
            yaxis=dict(categoryorder='array',
                       categoryarray=['坤', '艮', '坎', '巽', '震', '离', '兑', '乾']),
        )
        apply_dark_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"卦象历史加载失败: {e}")


# ================================================================
# 日卦可视化核对 (底座卦: 日/月/年 三尺度的日线视角)
# ================================================================
st.markdown("---")
st.markdown("### 📺 日卦可视化核对")
st.caption("日卦 (v10 日线 位/势/变, 滞后带) — 位爻=trend≥50 / 势爻=日差±2滞后+89高位保护 / 变爻=主力±30滞后")

try:
    bb_full = load_day_gua_visual()
    if len(bb_full) == 0:
        st.info("multi_scale_gua_daily.csv 暂无数据；请先运行 data_layer/prepare_multi_scale_gua.py")
    else:
        m_min = bb_full['date'].min().date()
        m_max = bb_full['date'].max().date()

        vc1, vc2, vc3, vc4, vc5, vc6 = st.columns([1.0, 1.0, 1.0, 0.85, 0.95, 1.0])
        with vc1:
            st.markdown("**日卦**")
        with vc2:
            viz_start = st.date_input("核对起始日", value=m_min, min_value=m_min, max_value=m_max, key="bb_viz_start")
        with vc3:
            viz_end = st.date_input("核对结束日", value=m_max, min_value=m_min, max_value=m_max, key="bb_viz_end")
        with vc4:
            changed_only = st.checkbox("只看变卦事件", value=False, key="bb_changed_only")
        with vc5:
            selected_seg = st.number_input("定位 segment", min_value=0, value=0, step=1, help="填 0 表示不过滤", key="bb_sel_seg")
        with vc6:
            index_name = st.selectbox("主图指数", ['中证1000', '沪深300', '中证500', '全A', '上证', '深证'], index=0, key="bb_index_name")

        market_viz = load_day_gua_visual(start_date=str(viz_start), end_date=str(viz_end)).copy()
        market_viz['gua_code'] = market_viz['gua_code'].astype(str).str.zfill(3)
        from dashboard.components.data_loader import GUA_MEANINGS_ZH as _BB_MEAN
        market_viz['gua_meaning'] = market_viz['gua_code'].map(lambda x: _BB_MEAN.get(str(x).zfill(3), ''))
        market_viz['prev_gua_display'] = ''
        if 'prev_gua' in market_viz.columns:
            prev_mask = market_viz['prev_gua'].notna()
            market_viz.loc[prev_mask, 'prev_gua_display'] = market_viz.loc[prev_mask, 'prev_gua'].astype(str).str.zfill(3)
        # render_market_regime_index_chart 需要 yao_1/yao_2/yao_3
        market_viz['yao_1'] = market_viz['yao_pos']
        market_viz['yao_2'] = market_viz['yao_spd']
        market_viz['yao_3'] = market_viz['yao_acc']
        if selected_seg > 0:
            market_viz = market_viz[market_viz['seg_id'] == selected_seg].copy()
        elif changed_only:
            changed_dates = market_viz.loc[market_viz['changed'] == 1, 'date']
            if len(changed_dates) > 0:
                kept = []
                for dt in changed_dates:
                    kept.append(market_viz[(market_viz['date'] >= dt - pd.Timedelta(days=10)) & (market_viz['date'] <= dt + pd.Timedelta(days=10))])
                market_viz = pd.concat(kept, ignore_index=True).drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            else:
                market_viz = market_viz.iloc[0:0].copy()

        if len(market_viz) == 0:
            st.warning("当前筛选条件下无可视化数据")
        else:
            seg_summary = build_day_gua_segments_summary(start_date=str(viz_start), end_date=str(viz_end))
            gua_summary = build_day_gua_summary(start_date=str(viz_start), end_date=str(viz_end))
            index_df = load_market_proxy_index(index_name=index_name, start_date=str(viz_start), end_date=str(viz_end))
            if selected_seg > 0:
                seg_summary = seg_summary[seg_summary['seg_id'] == selected_seg].copy()
                index_df = index_df[index_df['date'].isin(market_viz['date'])].copy() if len(index_df) > 0 else index_df

            avg_seg = seg_summary['持续天数'].mean() if len(seg_summary) > 0 else 0.0
            short_seg_ratio = (seg_summary['持续天数'] <= 5).mean() if len(seg_summary) > 0 else 0.0
            change_count = int(market_viz['changed'].sum())
            current_gua = market_viz.iloc[-1]['gua_name'] if len(market_viz) > 0 else '-'
            last_row = market_viz.iloc[-1]
            regime_label = '日卦'

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("区间交易日", f"{len(market_viz)}")
            mc2.metric("变卦次数", f"{change_count}")
            mc3.metric("平均段长", f"{avg_seg:.1f}")
            mc4.metric("短段占比(≤5天)", f"{short_seg_ratio * 100:.1f}%")
            current_meaning = _BB_MEAN.get(str(last_row['gua_code']).zfill(3), '-')
            prev_gua_text = '-'
            if 'prev_gua_display' in last_row.index and str(last_row['prev_gua_display']).strip() not in ('', 'nan'):
                pcode = str(last_row['prev_gua_display']).zfill(3)
                prev_gua_text = f"{_BB_MEAN.get(pcode, pcode)} ({pcode})"
            st.caption(
                f"当前层级: {regime_label} | 主图采用: {index_name} 日线 | 背景按 segment 显示日卦 | "
                f"当前最后一卦: {current_gua} · {current_meaning}"
            )

            render_market_regime_index_chart(index_df, market_viz, title=f"{index_name}日线 + {regime_label}背景")

            exp1, exp2 = st.columns([1.15, 1])
            with exp1:
                st.markdown('#### 当前状态解释')
                explain_rows = [
                    {'字段': '当前卦', '值': f"{last_row['gua_name']} ({last_row['gua_code']})"},
                    {'字段': '前卦', '值': prev_gua_text},
                    {'字段': '卦意', '值': current_meaning},
                    {'字段': '位爻 (trend≥50)', '值': int(last_row['yao_pos']) if pd.notna(last_row.get('yao_pos')) else '-'},
                    {'字段': '势爻 (日差±2+89保护)', '值': int(last_row['yao_spd']) if pd.notna(last_row.get('yao_spd')) else '-'},
                    {'字段': '变爻 (主力±30)', '值': int(last_row['yao_acc']) if pd.notna(last_row.get('yao_acc')) else '-'},
                ]
                explain_rows.extend([
                    {'字段': 'close', '值': round(float(last_row['close']), 0) if pd.notna(last_row.get('close')) else '-'},
                    {'字段': 'trend', '值': round(float(last_row['trend']), 1) if pd.notna(last_row.get('trend')) else '-'},
                    {'字段': '主力线', '值': round(float(last_row['main_force']), 1) if pd.notna(last_row.get('main_force')) else '-'},
                ])
                explain_rows.extend([
                    {'字段': 'segment', '值': int(last_row['seg_id']) if pd.notna(last_row['seg_id']) else '-'},
                    {'字段': 'segment_day', '值': int(last_row['seg_day']) if pd.notna(last_row['seg_day']) else '-'},
                ])
                explain_df = pd.DataFrame(explain_rows)
                st.dataframe(explain_df, use_container_width=True, hide_index=True, height=420)
            with exp2:
                st.markdown('#### 区间卦分布')
                if len(gua_summary) > 0:
                    fig_gua = go.Figure(go.Bar(
                        x=gua_summary['卦名'].astype(str).str.slice(0, 2),
                        y=gua_summary['天数'],
                        text=gua_summary['占比%'].map(lambda x: f'{x:.1f}%'),
                        textposition='outside',
                        marker_color='#60a5fa',
                    ))
                    fig_gua.update_layout(height=420, title='卦分布天数')
                    apply_dark_theme(fig_gua)
                    st.plotly_chart(fig_gua, use_container_width=True)
                else:
                    st.info('当前区间无卦分布数据')

            # 三爻阶梯图
            fig_yao = go.Figure()
            yao_colors = {'yao_pos': '#ef4444', 'yao_spd': '#f59e0b', 'yao_acc': '#34d399'}
            yao_names = {'yao_pos': '位爻 (trend≥50)', 'yao_spd': '势爻 (日差±2)', 'yao_acc': '变爻 (主力±30)'}
            for idx, col in enumerate(['yao_pos', 'yao_spd', 'yao_acc']):
                fig_yao.add_trace(go.Scatter(
                    x=market_viz['date'],
                    y=market_viz[col] + idx * 1.3,
                    mode='lines',
                    line=dict(color=yao_colors[col], width=2, shape='hv'),
                    name=yao_names[col],
                    customdata=market_viz[['gua_name', 'gua_code', 'seg_day']].values,
                    hovertemplate='%{x|%Y-%m-%d}<br>' + yao_names[col] + ': %{y:.0f}<br>卦: %{customdata[0]} %{customdata[1]}<br>段内第 %{customdata[2]} 天<extra></extra>',
                ))
            fig_yao.update_layout(height=230, yaxis=dict(showticklabels=False), title='三爻阶梯图 (位|势|变)')
            apply_dark_theme(fig_yao)
            st.plotly_chart(fig_yao, use_container_width=True)

            # 驱动因子: 趋势线 + 主力线
            fig_drv = go.Figure()
            fig_drv.add_trace(go.Scatter(x=market_viz['date'], y=market_viz['trend'], mode='lines',
                name='趋势线', line=dict(color='#34d399', width=1.5)))
            fig_drv.add_trace(go.Scatter(x=market_viz['date'], y=market_viz['main_force'], mode='lines',
                name='主力线', line=dict(color='#ef4444', width=1.5), yaxis='y2'))
            fig_drv.add_hline(y=50, line_width=1, line_dash='dot', line_color='#888', annotation_text='位爻 50')
            fig_drv.add_hline(y=89, line_width=1, line_dash='dot', line_color='#f87171', annotation_text='89 高位保护')
            fig_drv.update_layout(
                height=260, title='驱动因子: 趋势线 + 主力线',
                yaxis=dict(title='趋势线'),
                yaxis2=dict(title='主力线', overlaying='y', side='right', zeroline=True, zerolinecolor='#888'),
            )
            apply_dark_theme(fig_drv)
            st.plotly_chart(fig_drv, use_container_width=True)

            st.markdown('#### Segment 摘要')
            if len(seg_summary) > 0:
                show_seg = seg_summary[['开始日', '结束日', '卦名', '卦码', '卦意', '持续天数', '段内涨跌幅%', 'trend均', '主力均']].copy()
                st.dataframe(show_seg, use_container_width=True, hide_index=True, height=320)
            else:
                st.info('当前区间无 segment 摘要')
except Exception as e:
    import traceback
    st.warning(f"日卦可视化加载失败: {e}")
    st.code(traceback.format_exc())
