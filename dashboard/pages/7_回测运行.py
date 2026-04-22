# -*- coding: utf-8 -*-
"""
回测运行 — 可配置日期范围的回测控制台
后台进程模式：回测在子进程中运行，切换页面不会中断
"""
import sys
import os
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv311', 'Scripts', 'python.exe')
BACKTEST_SCRIPT = os.path.join(PROJECT_ROOT, 'backtest_capital.py')
RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'live', 'runtime')
BT_PID_FILE = os.path.join(RUNTIME_DIR, 'backtest.pid')
BT_LOG_FILE = os.path.join(RUNTIME_DIR, 'backtest.log')

st.set_page_config(page_title="回测运行", page_icon="🔬", layout="wide")
st.title("🔬 回测运行")
st.caption("配置参数，运行联合策略回测")


# ============================================================
# 工具函数
# ============================================================
def _is_process_alive(pid):
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def _get_backtest_status():
    """检查回测进程是否在运行"""
    if not os.path.exists(BT_PID_FILE):
        return False, None
    try:
        with open(BT_PID_FILE, 'r') as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return False, None
    if _is_process_alive(pid):
        return True, pid
    # 进程已结束，清理 PID 文件
    try:
        os.remove(BT_PID_FILE)
    except OSError:
        pass
    return False, None


def _read_log_tail(n=80):
    if not os.path.exists(BT_LOG_FILE):
        return ''
    try:
        with open(BT_LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except OSError:
        return ''


def _start_backtest(start_str, end_str, capital, filter_trend_val=None, filter_retail_val=None):
    """后台启动回测进程"""
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    cmd = [
        VENV_PYTHON, '-u', BACKTEST_SCRIPT,
        '--start', start_str,
        '--end', end_str,
        '--capital', str(capital),
    ]
    if filter_trend_val is not None:
        cmd.extend(['--filter-trend', str(filter_trend_val)])
    if filter_retail_val is not None:
        cmd.extend(['--filter-retail', str(filter_retail_val)])
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    log_fh = open(BT_LOG_FILE, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    with open(BT_PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    return proc.pid


def _stop_backtest():
    """终止回测进程"""
    running, pid = _get_backtest_status()
    if not running or not pid:
        return
    try:
        subprocess.run(
            ['taskkill', '/F', '/PID', str(pid), '/T'],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        pass
    try:
        os.remove(BT_PID_FILE)
    except OSError:
        pass


# ============================================================
# 参数配置
# ============================================================
st.markdown("---")

# 读取回测配置默认值
import re as _re

def _extract_bt_param(name, default=None):
    """从 backtest_bt/config.py 提取参数"""
    bt_cfg = os.path.join(PROJECT_ROOT, 'backtest_bt', 'config.py')
    try:
        with open(bt_cfg, 'r', encoding='utf-8') as f:
            content = f.read()
        pattern = _re.compile(rf'^{name}\s*=\s*(.+?)(\s*#.*)?$', _re.MULTILINE)
        match = pattern.search(content)
        if match:
            return eval(match.group(1).strip())
    except Exception:
        pass
    return default

col1, col2, col3 = st.columns(3)

with col1:
    start_str = st.text_input("开始日期 (YYYY-MM-DD)",
                              value=str(_extract_bt_param('BACKTEST_START', '2015-01-01')))

with col2:
    end_str = st.text_input("结束日期 (YYYY-MM-DD)",
                            value=str(_extract_bt_param('BACKTEST_END', '2026-04-01')))

with col3:
    init_capital = st.number_input(
        "初始资金 (元)",
        value=int(_extract_bt_param('INIT_CAPITAL', 200000)),
        min_value=10000,
        step=10000,
        format="%d",
    )

# v1.1 过滤参数
fc1, fc2 = st.columns(2)
with fc1:
    filter_trend = st.number_input(
        "买点趋势线上限 (v1.1)",
        value=int(_extract_bt_param('FILTER_TREND_AT_BUY_MAX', 20)),
        min_value=10, max_value=80, step=1,
        help="买点趋势线>此值 → 跳过(已涨太多), 设0关闭",
    )
with fc2:
    filter_retail = st.number_input(
        "散户线回升上限 (v1.1)",
        value=int(_extract_bt_param('FILTER_RETAIL_RECOVERY_MAX', 500)),
        min_value=100, max_value=2000, step=50,
        help="散户线回升幅度>此值 → 跳过(底部已过), 设0关闭",
    )

# 日期校验
from datetime import datetime as _dt
try:
    start_date = _dt.strptime(start_str, '%Y-%m-%d').date()
    end_date = _dt.strptime(end_str, '%Y-%m-%d').date()
except ValueError:
    st.error("日期格式错误，请使用 YYYY-MM-DD 格式")
    st.stop()

if start_date >= end_date:
    st.error("开始日期必须早于结束日期")
    st.stop()

st.markdown("---")

# ============================================================
# 状态检测 + 控制按钮
# ============================================================
running, pid = _get_backtest_status()

if running:
    st.warning(f"回测运行中 (PID: {pid})")
    if st.button("⏹ 停止回测", type="secondary"):
        _stop_backtest()
        st.rerun()
else:
    # 检查上次回测是否刚完成（有日志但没有 PID）
    if os.path.exists(BT_LOG_FILE) and os.path.getsize(BT_LOG_FILE) > 0:
        log_tail = _read_log_tail(5)
        if '回测完成' in log_tail or '保存' in log_tail:
            st.success("上次回测已完成")
            st.cache_data.clear()

    if st.button("▶ 开始回测", type="primary"):
        try:
            new_pid = _start_backtest(start_str, end_str, init_capital,
                                      filter_trend, filter_retail)
            st.success(f"回测已启动 (PID: {new_pid})")
            st.rerun()
        except Exception as e:
            st.error(f"启动失败: {e}")

# ============================================================
# 日志显示（自动刷新）
# ============================================================
@st.fragment(run_every=5 if running else None)
def show_log():
    is_running, _ = _get_backtest_status()
    log = _read_log_tail(80)
    if log:
        st.markdown("##### 回测日志")
        st.code(log, language=None)
    if not is_running and running:
        # 刚从运行变为结束，刷新整个页面
        st.cache_data.clear()
        st.rerun()

show_log()

# ============================================================
# 回测结果展示
# ============================================================
if not running:
    bt_path = os.path.join(PROJECT_ROOT, 'data_layer', 'data', 'backtest_result.json')
    if os.path.exists(bt_path):
        st.markdown("---")
        st.markdown("##### 最近回测结果")
        with open(bt_path, 'r', encoding='utf-8') as f:
            bt_data = json.load(f)
        meta = bt_data['meta']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总收益率", f"{meta['total_return']:+.1f}%")
        col2.metric("最大回撤", f"{meta['max_dd']:.1f}%")
        col3.metric("交易笔数", f"{meta['trade_count']}")
        col4.metric("胜率", f"{meta['win_rate']:.1f}%")

        st.page_link("pages/5_回测可视化.py", label="📊 查看回测可视化 →")
