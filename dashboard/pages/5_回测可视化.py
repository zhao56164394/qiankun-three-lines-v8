# -*- coding: utf-8 -*-
"""
页面5: 回测可视化 — 八卦分治策略 v4.0

核心区块:
  1. 策略总览指标
  2. 净值曲线 (LWC Area)
  3. 分卦贡献饼图 + 分卦统计表
  4. 分卦×年度热力图
  5. 回撤 + 年度收益柱状图
  6. 个股K线 + 交易标注 (按卦筛选)
  7. 盈亏分布 + 持仓天数
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path = [p for p in sys.path if 'QMT' not in p and 'qmt' not in p.lower()] + [PROJECT_ROOT]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.data_loader import (
    load_backtest, load_backtest_8gua_extra, load_stock_ohlc,
    get_traded_codes, GUA_NAMES, GUA_COLORS,
)
from dashboard.components.chart_utils import (
    apply_dark_theme, COLOR_UP, COLOR_DOWN, COLOR_GOLD, COLOR_BLUE, COLOR_NEUTRAL,
)
from dashboard.components.lwc_charts import render_candlestick, render_equity_chart

st.set_page_config(page_title="回测可视化", page_icon="\U0001F4CA", layout="wide")

# ============================================================
# 加载数据
# ============================================================
meta, df_eq, df_trades, yearly = load_backtest()
gua_strategy = load_backtest_8gua_extra()

if meta is None:
    st.warning("回测数据未生成, 请先运行 `python backtest_8gua.py`")
    st.stop()

is_8gua = 'gua' in df_trades.columns and df_trades['gua'].nunique() > 1
label = meta.get('label', '八卦分治')

# ============================================================
# 页面标题 + 概览指标
# ============================================================
st.title(f"\U0001F4CA 回测可视化 — {label}")
st.caption(f"初始资金 {meta['init_capital']:,.0f} | "
           f"终值 {meta['final_capital']:,.0f} | "
           f"区间 {df_eq['date'].min().strftime('%Y-%m')} ~ {df_eq['date'].max().strftime('%Y-%m')}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("总收益率", f"{meta['total_return']:+,.0f}%")
c2.metric("终值", f"{meta['final_capital']/10000:,.0f}万")
c3.metric("最大回撤", f"{meta['max_dd']:.1f}%")
c4.metric("交易笔数", f"{meta['trade_count']}")
c5.metric("胜率", f"{meta['win_rate']:.1f}%")
c6.metric("均收益", f"{meta['avg_ret']:+.1f}%")

st.markdown("---")

# ============================================================
# 区块1: 净值曲线 (LWC Area)
# ============================================================
st.markdown("### \U0001F4C8 策略净值走势")
render_equity_chart(df_eq, height=380, title="策略净值 (基准=1.0)")

# ============================================================
# 区块2: 分卦贡献 (八卦分治专属)
# ============================================================
if is_8gua:
    st.markdown("---")
    st.markdown("### \u2630 分卦贡献")

    col_pie, col_table = st.columns([1, 1.5])

    # 分卦统计
    gua_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    gua_stats = []
    total_profit = df_trades['profit'].sum()

    for gua in gua_list:
        g_trades = df_trades[df_trades['gua'] == gua]
        if len(g_trades) == 0:
            gua_stats.append({
                'gua': gua, 'name': GUA_NAMES.get(gua, gua),
                'count': 0, 'win_rate': 0, 'avg_ret': 0,
                'profit': 0, 'pct': 0,
            })
        else:
            profit = g_trades['profit'].sum()
            wins = (g_trades['profit'] > 0).sum()
            gua_stats.append({
                'gua': gua, 'name': GUA_NAMES.get(gua, gua),
                'count': len(g_trades),
                'win_rate': wins / len(g_trades) * 100,
                'avg_ret': g_trades['ret_pct'].mean(),
                'profit': profit,
                'pct': profit / total_profit * 100 if total_profit != 0 else 0,
            })

    # 饼图
    with col_pie:
        active_stats = [s for s in gua_stats if s['count'] > 0]
        fig_pie = go.Figure(data=[go.Pie(
            labels=[s['name'] for s in active_stats],
            values=[max(0, s['profit']) for s in active_stats],
            marker_colors=[GUA_COLORS.get(s['gua'], '#666') for s in active_stats],
            textinfo='label+percent',
            textposition='inside',
            hole=0.35,
            hovertemplate='%{label}<br>利润: %{value:,.0f}<br>占比: %{percent}<extra></extra>',
        )])
        fig_pie.update_layout(height=350, showlegend=False, title='利润占比')
        apply_dark_theme(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 统计表
    with col_table:
        df_gua = pd.DataFrame(gua_stats)
        df_gua_display = df_gua[df_gua['count'] > 0].copy()
        df_gua_display = df_gua_display.sort_values('profit', ascending=False)
        df_gua_display['profit'] = df_gua_display['profit'].apply(lambda x: f'{x:+,.0f}')
        df_gua_display['avg_ret'] = df_gua_display['avg_ret'].apply(lambda x: f'{x:+.1f}%')
        df_gua_display['win_rate'] = df_gua_display['win_rate'].apply(lambda x: f'{x:.0f}%')
        df_gua_display['pct'] = df_gua_display['pct'].apply(lambda x: f'{x:.1f}%')
        df_gua_display = df_gua_display.rename(columns={
            'name': '卦', 'count': '笔数', 'win_rate': '胜率',
            'avg_ret': '均收益', 'profit': '利润', 'pct': '占比',
        })
        st.dataframe(
            df_gua_display[['卦', '笔数', '胜率', '均收益', '利润', '占比']],
            use_container_width=True, hide_index=True, height=350,
        )

    # 分卦×年度热力图
    st.markdown("#### 分卦×年度利润热力图")
    years = sorted(yearly.keys())
    active_gua = [s['gua'] for s in gua_stats if s['count'] > 0]
    z_data = []
    y_labels = []
    for gua in active_gua:
        row = []
        for y in years:
            g_trades = df_trades[(df_trades['gua'] == gua) &
                                 (df_trades['buy_date'].dt.year == int(y))]
            row.append(g_trades['profit'].sum() if len(g_trades) > 0 else 0)
        z_data.append(row)
        y_labels.append(GUA_NAMES.get(gua, gua))

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_data, x=years, y=y_labels,
        colorscale=[[0, '#ef4444'], [0.5, '#1a1a2e'], [1, '#22c55e']],
        zmid=0,
        text=[[f'{v/10000:+.0f}万' if abs(v) >= 10000 else f'{v:+,.0f}'
               for v in row] for row in z_data],
        texttemplate='%{text}',
        hovertemplate='%{y} %{x}<br>利润: %{z:,.0f}<extra></extra>',
    ))
    fig_heat.update_layout(height=max(250, 40 * len(active_gua) + 100),
                           yaxis=dict(autorange='reversed'))
    apply_dark_theme(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
# 区块3: 回撤 + 年度收益
# ============================================================
st.markdown("---")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 📉 最大回撤")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df_eq['date'], y=-df_eq['drawdown'],
        fill='tozeroy', fillcolor='rgba(239,68,68,0.15)',
        line=dict(color=COLOR_DOWN, width=1),
        name='回撤%',
        hovertemplate='%{x|%Y-%m-%d}<br>回撤: %{y:.1f}%<extra></extra>',
    ))
    fig_dd.update_layout(height=280, yaxis_title='回撤%', showlegend=False)
    apply_dark_theme(fig_dd)
    st.plotly_chart(fig_dd, use_container_width=True)

with col_right:
    st.markdown("### 📊 年度收益")
    if yearly:
        years_sorted = sorted(yearly.keys())
        profits = [yearly[y]['profit'] for y in years_sorted]
        colors = [COLOR_UP if p >= 0 else COLOR_DOWN for p in profits]
        fig_year = go.Figure()
        fig_year.add_trace(go.Bar(
            x=years_sorted, y=profits, marker_color=colors, name='年度盈亏',
            text=[f'{p/10000:+.0f}万' if abs(p) >= 10000 else f'{p:+,.0f}'
                  for p in profits],
            textposition='outside',
            hovertemplate='%{x}<br>盈亏: %{y:,.0f}<extra></extra>',
        ))
        fig_year.update_layout(height=280, yaxis_title='盈亏(元)', showlegend=False)
        apply_dark_theme(fig_year)
        st.plotly_chart(fig_year, use_container_width=True)

st.markdown("---")

# ============================================================
# 区块4: 个股 K 线 + 买卖标注 (可按卦筛选)
# ============================================================
st.markdown("### \U0001F56F 个股K线 + 交易标注")

# 按卦筛选
if is_8gua:
    gua_filter_options = ['全部'] + [GUA_NAMES.get(g, g) for g in gua_list if g in df_trades['gua'].values]
    gua_filter = st.selectbox("按卦筛选交易", gua_filter_options, index=0)
    if gua_filter != '全部':
        filter_code = [k for k, v in GUA_NAMES.items() if v == gua_filter][0]
        df_trades_filtered = df_trades[df_trades['gua'] == filter_code]
    else:
        df_trades_filtered = df_trades
else:
    df_trades_filtered = df_trades

traded_codes = sorted(df_trades_filtered['code'].unique().tolist()) if len(df_trades_filtered) > 0 else []

if traded_codes:
    code_stats = {}
    for _, t in df_trades_filtered.iterrows():
        c = t['code']
        if c not in code_stats:
            code_stats[c] = {'count': 0, 'profit': 0}
        code_stats[c]['count'] += 1
        code_stats[c]['profit'] += t['profit']

    code_options = []
    for c in traded_codes:
        s = code_stats.get(c, {'count': 0, 'profit': 0})
        gua_info = ''
        if is_8gua:
            gua_codes = df_trades_filtered[df_trades_filtered['code'] == c]['gua'].unique()
            gua_info = '/'.join([GUA_NAMES.get(g, g)[:1] for g in gua_codes]) + ' '
        label_str = f"{c} {gua_info}({s['count']}笔, {'+' if s['profit']>=0 else ''}{s['profit']:,.0f})"
        code_options.append((label_str, c))

    code_options.sort(key=lambda x: code_stats.get(x[1], {}).get('profit', 0), reverse=True)

    col_sel1, col_sel2 = st.columns([2, 2])
    with col_sel1:
        selected_label = st.selectbox("选择股票", [x[0] for x in code_options], index=0)
        selected_code = [x[1] for x in code_options if x[0] == selected_label][0]

    stock_trades = df_trades_filtered[df_trades_filtered['code'] == selected_code].copy()
    stock_trades = stock_trades.sort_values('buy_date')

    with col_sel2:
        if len(stock_trades) > 0:
            first_buy = stock_trades['buy_date'].min()
            last_sell = stock_trades['sell_date'].max()
            pad = pd.Timedelta(days=30)
            date_start = (first_buy - pad).strftime('%Y-%m-%d')
            date_end = (last_sell + pad).strftime('%Y-%m-%d')
            # 显示卦信息
            for _, t in stock_trades.iterrows():
                gua_name = GUA_NAMES.get(t.get('gua', ''), '')
                sell_m = t.get('sell_method', '')
                ret_str = f"{t['ret_pct']:+.1f}%"
                st.caption(f"{t['buy_date'].strftime('%Y-%m-%d')} ~ "
                           f"{t['sell_date'].strftime('%Y-%m-%d')} | "
                           f"{gua_name} | {sell_m} | {ret_str}")
        else:
            date_start, date_end = None, None

    df_stock = load_stock_ohlc(selected_code, date_start, date_end)

    if df_stock is not None and len(df_stock) > 0:
        markers = []
        has_prices = 'buy_price' in stock_trades.columns and 'sell_price' in stock_trades.columns
        for _, t in stock_trades.iterrows():
            gua_tag = GUA_NAMES.get(t.get('gua', ''), '')[:1] if is_8gua else ''
            buy_text = f"买 {t['buy_price']:.2f}" if has_prices else "买"
            if gua_tag:
                buy_text = f"{gua_tag} {buy_text}"
            markers.append({
                'time': t['buy_date'].strftime('%Y-%m-%d'),
                'position': 'belowBar',
                'color': GUA_COLORS.get(t.get('gua', ''), '#ef4444'),
                'shape': 'arrowUp',
                'text': buy_text,
            })
            ret_str = f"{t['ret_pct']:+.1f}%"
            sell_text = f"卖 {t['sell_price']:.2f} ({ret_str})" if has_prices else f"卖 ({ret_str})"
            markers.append({
                'time': t['sell_date'].strftime('%Y-%m-%d'),
                'position': 'aboveBar',
                'color': '#22c55e' if t['ret_pct'] < 0 else '#ef4444',
                'shape': 'arrowDown',
                'text': sell_text,
            })

        markers.sort(key=lambda x: x['time'])

        render_candlestick(
            df_stock, markers=markers, height=450,
            title=f"{selected_code} — 日K线 + 交易标注",
            volume=False,
        )

        if 'trend' in df_stock.columns and 'retail' in df_stock.columns:
            fig_ind = make_subplots(rows=1, cols=1)
            fig_ind.add_trace(go.Scatter(
                x=df_stock['date'], y=df_stock['trend'],
                name='趋势线', line=dict(color=COLOR_GOLD, width=1.5),
            ))
            fig_ind.add_trace(go.Scatter(
                x=df_stock['date'], y=df_stock['retail'],
                name='散户线', line=dict(color=COLOR_BLUE, width=1),
                yaxis='y2',
            ))
            fig_ind.add_hline(y=11, line_dash="dot", line_color="#666",
                              annotation_text="11", row=1, col=1)
            fig_ind.add_hline(y=89, line_dash="dot", line_color="#666",
                              annotation_text="89", row=1, col=1)
            fig_ind.update_layout(
                height=200,
                yaxis=dict(title='趋势线'),
                yaxis2=dict(title='散户线', overlaying='y', side='right'),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            apply_dark_theme(fig_ind)
            st.plotly_chart(fig_ind, use_container_width=True)

        st.markdown("#### 该股交易记录")
        if has_prices:
            cols = ['buy_date', 'sell_date', 'buy_price', 'sell_price',
                    'ret_pct', 'hold_days', 'profit']
            col_names = ['买入日', '卖出日', '买入价', '卖出价', '收益率%', '持仓天', '盈亏']
            if is_8gua:
                cols = ['gua', 'sell_method'] + cols
                col_names = ['卦', '卖法'] + col_names
            display_trades = stock_trades[cols].copy()
            display_trades.columns = col_names
        else:
            display_trades = stock_trades[['buy_date', 'sell_date',
                                           'ret_pct', 'hold_days', 'profit']].copy()
            display_trades.columns = ['买入日', '卖出日', '收益率%', '持仓天', '盈亏']

        if '卦' in display_trades.columns:
            display_trades['卦'] = display_trades['卦'].map(
                lambda x: GUA_NAMES.get(x, x))
        display_trades['收益率%'] = display_trades['收益率%'].apply(lambda x: f'{x:+.1f}%')
        display_trades['盈亏'] = display_trades['盈亏'].apply(lambda x: f'{x:+,.0f}')
        display_trades['买入日'] = display_trades['买入日'].dt.strftime('%Y-%m-%d')
        display_trades['卖出日'] = display_trades['卖出日'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_trades, use_container_width=True, hide_index=True)

    else:
        st.warning(f"未找到 {selected_code} 的K线数据")

else:
    st.info("无交易记录" + (" (当前筛选条件下)" if is_8gua else ""))

st.markdown("---")

# ============================================================
# 区块5: 盈亏分布 + 持仓天数
# ============================================================
st.markdown("### 📈 交易统计分布")

col_h1, col_h2 = st.columns(2)

with col_h1:
    rets = df_trades['ret_pct'].values
    fig_ret = go.Figure()
    fig_ret.add_trace(go.Histogram(
        x=rets, nbinsx=40, name='收益率分布',
        marker_color=COLOR_GOLD, opacity=0.8,
        hovertemplate='收益率: %{x:.1f}%<br>笔数: %{y}<extra></extra>',
    ))
    fig_ret.add_vline(x=0, line_dash="dash", line_color="#666")
    median_ret = np.median(rets)
    fig_ret.add_vline(x=median_ret, line_dash="dot", line_color=COLOR_BLUE,
                      annotation_text=f"中位数 {median_ret:.1f}%")
    fig_ret.update_layout(height=300, xaxis_title='收益率%', yaxis_title='笔数',
                          title='单笔收益率分布', showlegend=False)
    apply_dark_theme(fig_ret)
    st.plotly_chart(fig_ret, use_container_width=True)

with col_h2:
    hold_days = df_trades['hold_days'].astype(int).values
    fig_hold = go.Figure()
    fig_hold.add_trace(go.Histogram(
        x=hold_days, nbinsx=30, name='持仓天数分布',
        marker_color=COLOR_BLUE, opacity=0.8,
        hovertemplate='持仓: %{x}天<br>笔数: %{y}<extra></extra>',
    ))
    median_hold = np.median(hold_days)
    fig_hold.add_vline(x=median_hold, line_dash="dot", line_color=COLOR_GOLD,
                       annotation_text=f"中位数 {median_hold:.0f}天")
    fig_hold.update_layout(height=300, xaxis_title='持仓天数', yaxis_title='笔数',
                           title='持仓天数分布', showlegend=False)
    apply_dark_theme(fig_hold)
    st.plotly_chart(fig_hold, use_container_width=True)

# ============================================================
# 底部信息
# ============================================================
st.markdown("---")
st.caption("技术栈: TradingView Lightweight Charts (K线) + Plotly (统计图) + Streamlit (框架)")
