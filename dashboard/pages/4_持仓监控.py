# -*- coding: utf-8 -*-
"""
页面4: 持仓监控 — 实盘持仓 + 今日信号 + 资金使用率
"""
import sys
import os
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from dashboard.components.chart_utils import format_pct, format_money, COLOR_UP, COLOR_DOWN

st.set_page_config(page_title="持仓监控", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA 持仓监控")

# === 加载实盘数据 ===
snap_path = os.path.join(PROJECT_ROOT, 'live', 'snapshots', 'latest.json')
trades_path = os.path.join(PROJECT_ROOT, 'live', 'logs', 'trades.csv')

# 最新持仓快照
snapshot = None
if os.path.exists(snap_path):
    try:
        with open(snap_path, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
    except Exception:
        pass

if snapshot is None:
    st.info("暂无实盘数据。策略运行后会自动生成持仓快照。")
    st.markdown("""
    **启动方式:**
    ```bash
    cd "e:/乾坤三线 v8.0"
    python live/qmt_strategy.py --simulate --once
    ```
    """)

    # 尝试读取交易日志
    if os.path.exists(trades_path):
        st.markdown("---")
        st.markdown("### 交易日志")
        try:
            df_trades = pd.read_csv(trades_path, encoding='utf-8-sig')
            if len(df_trades) > 0:
                st.dataframe(df_trades.tail(20), use_container_width=True)
            else:
                st.caption("交易日志为空")
        except Exception as e:
            st.warning(f"读取交易日志失败: {e}")

    st.stop()

# === 有实盘数据 ===
st.caption(f"更新时间: {snapshot.get('timestamp', '未知')}")

# 汇总指标
col1, col2, col3, col4 = st.columns(4)
col1.metric("总资产", format_money(snapshot.get('total_equity', 0)))
col2.metric("可用现金", format_money(snapshot.get('cash', 0)))
col3.metric("持仓数", f"{snapshot.get('n_positions', 0)}")

# 资金使用率
total = snapshot.get('total_equity', 1)
cash = snapshot.get('cash', 0)
usage = (1 - cash / total) * 100 if total > 0 else 0
col4.metric("资金使用率", f"{usage:.1f}%")

st.markdown("---")

# === 持仓列表 ===
st.markdown("### 当前持仓")
positions = snapshot.get('positions', {})

if positions:
    rows = []
    for code, info in positions.items():
        row = {
            '代码': code,
            '数量': info.get('volume', 0),
            '成本价': info.get('cost_price', info.get('buy_price', 0)),
            '市值': info.get('market_value', 0),
            '卖法': info.get('sell_method', '?'),
            '趋势最高': info.get('running_max', 0),
            '到达89': info.get('reached_89', False),
        }
        rows.append(row)

    df_pos = pd.DataFrame(rows)
    st.dataframe(df_pos, use_container_width=True)
else:
    st.info("当前无持仓")

# === 今日信号 ===
st.markdown("---")
st.markdown("### 今日信号")

# 读取最新的信号文件
signals_dir = os.path.join(PROJECT_ROOT, 'live', 'logs', 'signal_reports')
if os.path.exists(signals_dir):
    signal_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.json')],
                           reverse=True)
    if signal_files:
        latest_signal_file = os.path.join(signals_dir, signal_files[0])
        try:
            with open(latest_signal_file, 'r', encoding='utf-8') as f:
                signal_data = json.load(f)

            st.caption(f"信号日期: {signal_data.get('date', '?')}")
            signals = signal_data.get('signals', [])
            if signals:
                df_sig = pd.DataFrame(signals)
                st.dataframe(df_sig, use_container_width=True)
            else:
                st.info("无买入信号")
        except Exception as e:
            st.warning(f"读取信号文件失败: {e}")
    else:
        st.info("暂无信号记录")
else:
    st.info("信号目录不存在")

# === 交易历史 ===
st.markdown("---")
st.markdown("### 近期交易")

if os.path.exists(trades_path):
    try:
        df_trades = pd.read_csv(trades_path, encoding='utf-8-sig')
        if len(df_trades) > 0:
            st.dataframe(df_trades.tail(20).iloc[::-1], use_container_width=True)
        else:
            st.caption("暂无交易记录")
    except Exception:
        st.caption("读取交易记录失败")

# === 每日快照历史 ===
st.markdown("---")
with st.expander("每日快照历史"):
    snap_dir = os.path.join(PROJECT_ROOT, 'live', 'logs', 'daily_snapshot')
    if os.path.exists(snap_dir):
        snap_files = sorted([f for f in os.listdir(snap_dir) if f.endswith('.json')],
                             reverse=True)
        if snap_files:
            rows = []
            for fname in snap_files[:30]:
                fpath = os.path.join(snap_dir, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                    rows.append({
                        '日期': d.get('date', '?'),
                        '总资产': d.get('total_equity', 0),
                        '现金': d.get('cash', 0),
                        '持仓数': d.get('n_positions', 0),
                    })
                except Exception:
                    pass
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("暂无快照")
    else:
        st.caption("快照目录不存在")
