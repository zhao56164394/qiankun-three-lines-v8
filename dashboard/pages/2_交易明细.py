# -*- coding: utf-8 -*-
"""
页面2: 交易明细 — 交易记录表 + 盈亏分布 + 月度统计
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.components.data_loader import load_backtest
from dashboard.components.chart_utils import (
    apply_dark_theme, format_pct, COLOR_UP, COLOR_DOWN, COLOR_BLUE,
)

st.set_page_config(page_title="交易明细", page_icon="\U0001F4CB", layout="wide")
st.title("\U0001F4CB 交易明细")

# 加载数据
meta, _df_eq, df, _yearly = load_backtest()
if meta is None or df is None or len(df) == 0:
    st.warning("回测数据未生成, 请先运行 `python backtest_capital.py`")
    st.stop()

# === 汇总指标 ===
total_trades = len(df)
wins = df[df['profit'] > 0]
losses = df[df['profit'] <= 0]
win_rate = len(wins) / total_trades * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("总交易", f"{total_trades}笔")
col2.metric("胜率", f"{win_rate:.1f}%")
col3.metric("平均收益", format_pct(df['ret_pct'].mean()))
col4.metric("平均持仓", f"{df['hold_days'].mean():.1f}天")
col5.metric("总利润", f"{df['profit'].sum():+,.0f}")

st.markdown("---")

# === 筛选 ===
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    year_filter = st.selectbox("年份", ['全部'] + sorted(df['buy_date'].dt.year.unique().tolist()))
with col_f2:
    result_filter = st.selectbox("结果", ['全部', '盈利', '亏损'])
with col_f3:
    sort_by = st.selectbox("排序", ['买入日期', '收益率', '持仓天数', '利润'])

# 应用筛选
df_show = df.copy()
if year_filter != '全部':
    df_show = df_show[df_show['buy_date'].dt.year == year_filter]
if result_filter == '盈利':
    df_show = df_show[df_show['profit'] > 0]
elif result_filter == '亏损':
    df_show = df_show[df_show['profit'] <= 0]

sort_map = {'买入日期': 'buy_date', '收益率': 'ret_pct', '持仓天数': 'hold_days', '利润': 'profit'}
df_show = df_show.sort_values(sort_map[sort_by], ascending=(sort_by == '买入日期'))

# === 交易记录表 ===
st.markdown("### 交易记录")

has_prices = 'buy_price' in df_show.columns and 'sell_price' in df_show.columns
if has_prices:
    display_df = df_show[['code', 'buy_date', 'sell_date', 'buy_price', 'sell_price',
                           'ret_pct', 'hold_days', 'profit', 'cost']].copy()
    display_df.columns = ['代码', '买入日', '卖出日', '买入价', '卖出价',
                           '收益率%', '持仓天', '利润', '投入']
    display_df['买入价'] = display_df['买入价'].round(2)
    display_df['卖出价'] = display_df['卖出价'].round(2)
else:
    display_df = df_show[['code', 'buy_date', 'sell_date',
                           'ret_pct', 'hold_days', 'profit', 'cost']].copy()
    display_df.columns = ['代码', '买入日', '卖出日',
                           '收益率%', '持仓天', '利润', '投入']
display_df['买入日'] = display_df['买入日'].dt.strftime('%Y-%m-%d')
display_df['卖出日'] = display_df['卖出日'].dt.strftime('%Y-%m-%d')
display_df['收益率%'] = display_df['收益率%'].round(1)
display_df['利润'] = display_df['利润'].round(0)
display_df['投入'] = display_df['投入'].round(0)

st.dataframe(display_df, use_container_width=True, height=400)

st.markdown("---")

# === 盈亏分布 ===
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 收益率分布")
    fig_hist = go.Figure()

    # 分盈利和亏损着色
    win_rets = df_show[df_show['ret_pct'] > 0]['ret_pct']
    loss_rets = df_show[df_show['ret_pct'] <= 0]['ret_pct']

    fig_hist.add_trace(go.Histogram(
        x=win_rets, name='盈利', marker_color=COLOR_UP,
        opacity=0.7, nbinsx=30))
    fig_hist.add_trace(go.Histogram(
        x=loss_rets, name='亏损', marker_color=COLOR_DOWN,
        opacity=0.7, nbinsx=30))

    fig_hist.update_layout(
        barmode='stack', height=350,
        xaxis_title='收益率%', yaxis_title='笔数')
    apply_dark_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)

with c2:
    st.markdown("### 持仓天数分布")
    fig_hold = go.Figure()
    fig_hold.add_trace(go.Histogram(
        x=df_show['hold_days'], name='持仓天数',
        marker_color=COLOR_BLUE, opacity=0.7, nbinsx=25))
    fig_hold.update_layout(height=350,
                           xaxis_title='天数', yaxis_title='笔数')
    apply_dark_theme(fig_hold)
    st.plotly_chart(fig_hold, use_container_width=True)

# === 月度胜率/收益统计 ===
st.markdown("---")
st.markdown("### 月度统计")

df_show['ym'] = df_show['buy_date'].dt.to_period('M').astype(str)
monthly = df_show.groupby('ym').agg(
    count=('ret_pct', 'size'),
    win_count=('profit', lambda x: (x > 0).sum()),
    avg_ret=('ret_pct', 'mean'),
    total_profit=('profit', 'sum'),
).reset_index()
monthly['win_rate'] = monthly['win_count'] / monthly['count'] * 100

fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])

colors_bar = [COLOR_UP if p >= 0 else COLOR_DOWN for p in monthly['total_profit']]
fig_monthly.add_trace(
    go.Bar(x=monthly['ym'], y=monthly['total_profit'],
           name='月度利润', marker_color=colors_bar, opacity=0.7),
    secondary_y=False)

fig_monthly.add_trace(
    go.Scatter(x=monthly['ym'], y=monthly['win_rate'],
               name='胜率%', line=dict(color='#f59e0b', width=2),
               mode='lines+markers', marker=dict(size=4)),
    secondary_y=True)

fig_monthly.update_yaxes(title_text='利润', secondary_y=False)
fig_monthly.update_yaxes(title_text='胜率%', secondary_y=True)
fig_monthly.update_layout(height=350)
apply_dark_theme(fig_monthly)
st.plotly_chart(fig_monthly, use_container_width=True)

# 统计表
with st.expander("月度数据表"):
    monthly_display = monthly.copy()
    monthly_display.columns = ['月份', '笔数', '盈利笔', '均收益%', '总利润', '胜率%']
    monthly_display['均收益%'] = monthly_display['均收益%'].round(1)
    monthly_display['总利润'] = monthly_display['总利润'].round(0)
    monthly_display['胜率%'] = monthly_display['胜率%'].round(1)
    st.dataframe(monthly_display, use_container_width=True)
