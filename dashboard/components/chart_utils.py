# -*- coding: utf-8 -*-
"""
通用图表工具函数
"""
import plotly.graph_objects as go


def apply_dark_theme(fig):
    """统一暗色主题"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Microsoft YaHei', size=12),
        margin=dict(l=40, r=20, t=40, b=30),
    )
    return fig


def format_pct(val, decimals=1):
    """格式化百分比"""
    if val is None:
        return '--'
    return f'{val:+.{decimals}f}%'


def format_money(val):
    """格式化金额"""
    if val is None:
        return '--'
    if abs(val) >= 10000:
        return f'{val/10000:.2f}万'
    return f'{val:,.0f}'


# 颜色常量
COLOR_UP = '#ef4444'      # 涨 - 红色
COLOR_DOWN = '#22c55e'    # 跌 - 绿色
COLOR_NEUTRAL = '#94a3b8' # 中性
COLOR_GOLD = '#f59e0b'    # 金色重点
COLOR_BLUE = '#3b82f6'    # 蓝色
