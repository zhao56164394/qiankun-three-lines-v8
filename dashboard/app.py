# -*- coding: utf-8 -*-
"""
乾坤三线 v8.0 — 八卦分治策略仪表盘

启动: streamlit run dashboard/app.py
"""
import sys
import os

# 确保项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(
    page_title="乾坤三线 v8.0",
    page_icon="\u2630",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("\u2630 乾坤三线 v8.0")
st.caption("量化交易系统 — 八卦分治策略 · 实盘交易 · 回测分析")

# ============================================================
# 系统运行状态
# ============================================================
from components.process_manager import get_status

live_st = get_status('live')
sim_st = get_status('simulate')

sc1, sc2, sc3 = st.columns(3)
with sc1:
    if live_st['running']:
        st.success(f"实盘: 运行中 (PID {live_st.get('pid', '?')})")
    else:
        st.info("实盘: 未运行")
with sc2:
    if sim_st['running']:
        st.success(f"模拟盘: 运行中 (PID {sim_st.get('pid', '?')})")
    else:
        st.info("模拟盘: 未运行")
with sc3:
    # 优先检查八卦分治结果
    bt_8gua_path = os.path.join(PROJECT_ROOT, 'data_layer', 'data', 'backtest_8gua_result.json')
    bt_hybrid_path = os.path.join(PROJECT_ROOT, 'data_layer', 'data', 'backtest_result.json')
    bt_path = bt_8gua_path if os.path.exists(bt_8gua_path) else bt_hybrid_path

    if os.path.exists(bt_path):
        import time as _time
        mtime = os.path.getmtime(bt_path)
        from datetime import datetime as _dt
        bt_time = _dt.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        bt_label = "八卦分治" if bt_path == bt_8gua_path else "联合策略"
        st.info(f"最近回测({bt_label}): {bt_time}")
    else:
        st.warning("回测: 未生成")

st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("导航")
    st.markdown("""
    - **净值曲线** — 资金净值走势 + 回撤
    - **交易明细** — 交易记录 + 盈亏分析
    - **市场状态** — 八卦罗盘 + 策略提示
    - **持仓监控** — 实盘持仓面板
    - **回测可视化** — 分卦贡献 + K线 + 统计
    - **八卦调试** — 64卦联动矩阵 + 可买/禁买/虚高分区
    - **交易控制** — 实盘/模拟盘启停
    - **回测运行** — 可配置回测
    - **数据管理** — 数据更新 + 状态检查
    - **参数调整** — 策略参数可视化编辑
    """)

    st.markdown("---")
    st.markdown("##### 数据状态")

    if os.path.exists(bt_8gua_path):
        st.success("八卦分治回测: 已就绪")
    elif os.path.exists(bt_hybrid_path):
        st.success("联合策略回测: 已就绪")
    else:
        st.warning("回测数据: 未生成")
        st.caption("运行 `python backtest_8gua.py` 生成")

    # 检查实盘数据
    snap_path = os.path.join(PROJECT_ROOT, 'live', 'snapshots', 'latest.json')
    if os.path.exists(snap_path):
        st.success("实盘快照: 已就绪")
    else:
        st.info("实盘快照: 暂无")

# ============================================================
# 主页面: 策略概览
# ============================================================
import json

GUA_NAMES = {
    '000': '坤(至暗)', '001': '艮(蓄力)', '010': '坎(弱反弹)', '011': '巽(反转)',
    '100': '震(暴跌)', '101': '离(护盘)', '110': '兑(滞涨)', '111': '乾(牛顶)',
}

if os.path.exists(bt_path):
    with open(bt_path, 'r', encoding='utf-8') as f:
        bt_data = json.load(f)
    meta = bt_data['meta']

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("总收益率", f"{meta['total_return']:+,.0f}%")
    col2.metric("终值", f"{meta['final_capital']/10000:,.0f}万")
    col3.metric("最大回撤", f"{meta['max_dd']:.1f}%")
    col4.metric("交易笔数", f"{meta['trade_count']}")
    col5.metric("胜率", f"{meta['win_rate']:.1f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 回测概览")
        st.markdown(f"""
        | 指标 | 值 |
        |------|-----|
        | 策略 | {meta.get('label', '八卦分治')} |
        | 初始资金 | {meta['init_capital']:,.0f} |
        | 最终资金 | {meta['final_capital']:,.0f} |
        | 总收益率 | {meta['total_return']:+,.1f}% |
        | 平均收益 | {meta['avg_ret']:+.1f}% |
        | 平均持仓天数 | {meta['avg_hold']:.1f} |
        | 最大回撤 | {meta['max_dd']:.1f}% ({meta['max_dd_date']}) |
        """)

    with c2:
        st.markdown("##### 年度收益")
        yearly = bt_data.get('yearly', {})
        if yearly:
            rows = []
            for y in sorted(yearly.keys()):
                yd = yearly[y]
                wr = yd['wins'] / yd['count'] * 100 if yd['count'] > 0 else 0
                p = yd['profit']
                p_str = f'{p/10000:+.0f}万' if abs(p) >= 10000 else f'{p:+,.0f}'
                rows.append(f"| {y} | {p_str} | {yd['count']} | {wr:.0f}% |")
            st.markdown("| 年份 | 盈亏 | 笔数 | 胜率 |\n|------|------|------|------|\n" +
                         "\n".join(rows))

    # 分卦贡献 (八卦分治专属)
    trade_log = bt_data.get('trade_log', [])
    if trade_log and 'gua' in trade_log[0]:
        st.markdown("---")
        st.markdown("##### 分卦贡献")
        import pandas as pd
        df_t = pd.DataFrame(trade_log)
        total_profit = df_t['profit'].sum()
        rows = []
        for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
            gt = df_t[df_t['gua'] == gua]
            if len(gt) == 0:
                rows.append(f"| {GUA_NAMES.get(gua, gua)} | 空仓 | - | - | - |")
            else:
                p = gt['profit'].sum()
                wr = (gt['profit'] > 0).sum() / len(gt) * 100
                avg_r = gt['ret_pct'].mean()
                pct = p / total_profit * 100 if total_profit != 0 else 0
                p_str = f'{p/10000:+.0f}万' if abs(p) >= 10000 else f'{p:+,.0f}'
                rows.append(f"| {GUA_NAMES.get(gua, gua)} | {len(gt)} | {wr:.0f}% | {avg_r:+.1f}% | {p_str} ({pct:.0f}%) |")
        st.markdown("| 卦 | 笔数 | 胜率 | 均收益 | 利润(占比) |\n"
                     "|------|------|------|--------|------------|\n" +
                     "\n".join(rows))

else:
    st.info("请先运行回测生成数据: `python backtest_8gua.py`")
    st.markdown("""
    ```bash
    cd "e:/乾坤三线 v8.0"
    python backtest_8gua.py
    ```
    """)
