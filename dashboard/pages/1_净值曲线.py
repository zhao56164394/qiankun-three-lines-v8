# -*- coding: utf-8 -*-
"""
页面1: 净值曲线 — 资金走势 + 最大回撤 + 月度收益
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.components.data_loader import load_backtest
from dashboard.components.chart_utils import (
    apply_dark_theme, format_pct, format_money,
    COLOR_UP, COLOR_DOWN, COLOR_NEUTRAL, COLOR_GOLD, COLOR_BLUE,
)

st.set_page_config(page_title="净值曲线", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 净值曲线")

# 加载数据
meta, df_eq, df_trades, yearly = load_backtest()
if meta is None:
    st.warning("回测数据未生成, 请先运行 `python backtest_capital.py`")
    st.stop()

init_cap = meta['init_capital']

# === 净值走势图 ===
st.markdown("### 净值走势")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05)

# 净值线
fig.add_trace(
    go.Scatter(x=df_eq['date'], y=df_eq['nav'],
               name='策略净值', line=dict(color=COLOR_GOLD, width=2),
               fill='tozeroy', fillcolor='rgba(245,158,11,0.1)'),
    row=1, col=1
)

# 基准线 (持有不动 = 1.0)
fig.add_hline(y=1.0, line_dash="dash", line_color="#666",
              annotation_text="基准", row=1, col=1)

# 回撤区域
fig.add_trace(
    go.Scatter(x=df_eq['date'], y=-df_eq['drawdown'],
               name='回撤%', line=dict(color=COLOR_DOWN, width=1),
               fill='tozeroy', fillcolor='rgba(34,197,94,0.15)'),
    row=2, col=1
)

fig.update_yaxes(title_text='净值', row=1, col=1)
fig.update_yaxes(title_text='回撤%', row=2, col=1)
fig.update_layout(height=550, showlegend=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
apply_dark_theme(fig)
st.plotly_chart(fig, use_container_width=True)

# === 关键指标 ===
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("最终净值", f"{df_eq['nav'].iloc[-1]:.4f}")
col2.metric("总收益率", format_pct(meta['total_return']))
col3.metric("最大回撤", f"{meta['max_dd']:.1f}%")
col4.metric("最大回撤日", meta['max_dd_date'])
col5.metric("交易天数", f"{len(df_eq)}")

# === 月度/年度收益柱状图 ===
st.markdown("---")
st.markdown("### 月度收益")

# 计算月度收益
df_eq['year_month'] = df_eq['date'].dt.to_period('M').astype(str)
monthly = df_eq.groupby('year_month').agg(
    start=('total_equity', 'first'),
    end=('total_equity', 'last'),
).reset_index()
monthly['ret'] = (monthly['end'] / monthly['start'] - 1) * 100

colors = [COLOR_UP if r >= 0 else COLOR_DOWN for r in monthly['ret']]

fig_month = go.Figure()
fig_month.add_trace(
    go.Bar(x=monthly['year_month'], y=monthly['ret'],
           marker_color=colors, name='月度收益%')
)
fig_month.update_layout(height=350, xaxis_title='月份', yaxis_title='收益率%')
apply_dark_theme(fig_month)
st.plotly_chart(fig_month, use_container_width=True)

# === 持仓数量分布 ===
st.markdown("---")
st.markdown("### 持仓变化")

fig_pos = go.Figure()
fig_pos.add_trace(
    go.Scatter(x=df_eq['date'], y=df_eq['n_positions'],
               name='持仓数', fill='tozeroy',
               line=dict(color=COLOR_BLUE, width=1),
               fillcolor='rgba(59,130,246,0.2)')
)
fig_pos.update_layout(height=250, yaxis_title='持仓数')
apply_dark_theme(fig_pos)
st.plotly_chart(fig_pos, use_container_width=True)

# 统计信息
st.markdown("---")
with st.expander("详细统计"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **资金统计**
        - 初始: {init_cap:,.0f}
        - 最终: {meta['final_capital']:,.0f}
        - 最高: {df_eq['total_equity'].max():,.0f}
        - 最低: {df_eq['total_equity'].min():,.0f}
        """)
    with c2:
        avg_pos = df_eq['n_positions'].mean()
        zero_days = (df_eq['n_positions'] == 0).sum()
        st.markdown(f"""
        **持仓统计**
        - 平均持仓: {avg_pos:.1f}
        - 空仓天数: {zero_days} ({zero_days/len(df_eq)*100:.1f}%)
        - 满仓天数: {(df_eq['n_positions'] >= 3).sum()}
        """)
