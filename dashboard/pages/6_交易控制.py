# -*- coding: utf-8 -*-
"""
交易控制 — 实盘/模拟盘启停控制台
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from datetime import datetime
from components.process_manager import (
    get_status, start_strategy, stop_strategy, get_log_tail, is_any_running,
)

st.set_page_config(page_title="交易控制", page_icon="⚡", layout="wide")
st.title("⚡ 交易控制")
st.caption("实盘 / 模拟盘 启停管理")

st.markdown("---")

# 获取两个模式的状态
live_status = get_status('live')
sim_status = get_status('simulate')

any_running, running_mode = is_any_running()

# ============================================================
# 两列布局: 左实盘 右模拟盘
# ============================================================
col_live, col_sim = st.columns(2)

# --- 实盘 ---
with col_live:
    st.subheader("实盘交易")

    if live_status['running']:
        st.success(f"● 运行中 (PID: {live_status.get('pid', '?')})")
        start_time = live_status.get('start_time', '')
        if start_time:
            st.caption(f"启动时间: {start_time}")
        heartbeat = live_status.get('last_heartbeat', '')
        if heartbeat:
            st.caption(f"最近心跳: {heartbeat}")
        phase = live_status.get('phase', '')
        if phase:
            st.caption(f"当前阶段: {phase}")
        # 连接状态
        connected = live_status.get('connected')
        if connected is True:
            st.caption("QMT连接: 已连接")
        elif connected is False:
            st.error("QMT连接: 未连接")
        # 持仓数
        n_pos = live_status.get('positions')
        if n_pos is not None:
            st.caption(f"持仓: {n_pos} 只")

        if st.button("⏹ 停止实盘", key="stop_live", type="primary"):
            ok, msg = stop_strategy('live')
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    else:
        st.info("● 未运行")
        # 显示上次退出原因
        last_phase = live_status.get('last_phase', '')
        if last_phase:
            if 'QMT连接失败' in last_phase:
                st.error(f"上次退出: {last_phase} — 请确认 miniQMT 客户端已启动")
            else:
                st.caption(f"上次状态: {last_phase}")
        # 模拟盘运行中时禁用
        disabled = sim_status['running']
        help_text = "模拟盘运行中，请先停止" if disabled else None

        if st.button("▶ 启动实盘", key="start_live", type="primary",
                      disabled=disabled, help=help_text):
            ok, msg = start_strategy('live')
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

# --- 模拟盘 ---
with col_sim:
    st.subheader("模拟盘")

    if sim_status['running']:
        st.success(f"● 运行中 (PID: {sim_status.get('pid', '?')})")
        start_time = sim_status.get('start_time', '')
        if start_time:
            st.caption(f"启动时间: {start_time}")
        heartbeat = sim_status.get('last_heartbeat', '')
        if heartbeat:
            st.caption(f"最近心跳: {heartbeat}")
        phase = sim_status.get('phase', '')
        if phase:
            st.caption(f"当前阶段: {phase}")
        n_pos = sim_status.get('positions')
        if n_pos is not None:
            st.caption(f"持仓: {n_pos} 只")

        if st.button("⏹ 停止模拟盘", key="stop_sim", type="primary"):
            ok, msg = stop_strategy('simulate')
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()
    else:
        st.info("● 未运行")
        last_phase = sim_status.get('last_phase', '')
        if last_phase:
            st.caption(f"上次状态: {last_phase}")
        disabled = live_status['running']
        help_text = "实盘运行中，请先停止" if disabled else None

        if st.button("▶ 启动模拟盘", key="start_sim", type="primary",
                      disabled=disabled, help=help_text):
            ok, msg = start_strategy('simulate')
            if ok:
                st.success(msg)
            else:
                st.error(msg)
            st.rerun()

# ============================================================
# 运行日志
# ============================================================
st.markdown("---")
st.subheader("运行日志")

# 判断是否有进程在运行（决定是否自动刷新）
_any_running = live_status['running'] or sim_status['running']


@st.fragment(run_every=10 if _any_running else None)
def show_logs():
    """显示日志，运行中时每 10 秒自动刷新"""
    # 重新检测状态
    _live = get_status('live')
    _sim = get_status('simulate')

    # 实盘日志
    live_log = get_log_tail('live', 50)
    sim_log = get_log_tail('simulate', 50)

    if _live['running']:
        st.caption("实盘日志 (每10秒刷新):")
        st.code(live_log or "(等待输出...)", language=None)
    elif _sim['running']:
        st.caption("模拟盘日志 (每10秒刷新):")
        st.code(sim_log or "(等待输出...)", language=None)
    else:
        # 都没运行，显示最近的日志
        if sim_log:
            st.caption("最近模拟盘日志:")
            st.code(sim_log, language=None)
        if live_log:
            st.caption("最近实盘日志:")
            st.code(live_log, language=None)
        if not sim_log and not live_log:
            st.info("暂无运行日志")


show_logs()
