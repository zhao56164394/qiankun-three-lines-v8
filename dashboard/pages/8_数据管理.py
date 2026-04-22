# -*- coding: utf-8 -*-
"""
数据管理 — 数据状态检查 + 一键更新
通过 xtdata (miniQMT) 更新数据层
"""
import sys
import os
import glob
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv311', 'Scripts', 'python.exe')
UPDATE_SCRIPT = os.path.join(PROJECT_ROOT, 'data_layer', 'update_xtdata.py')
RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'live', 'runtime')
PID_FILE = os.path.join(RUNTIME_DIR, 'update_data.pid')
LOG_FILE = os.path.join(RUNTIME_DIR, 'update_data.log')

DATA_DIR = os.path.join(PROJECT_ROOT, 'data_layer', 'data')
ZZ1000_PATH = os.path.join(DATA_DIR, 'zz1000_daily.csv')
STOCKS_DIR = os.path.join(DATA_DIR, 'stocks')

st.set_page_config(page_title="数据管理", page_icon="💾", layout="wide")
st.title("💾 数据管理")
st.caption("数据状态检查 · 一键更新 (xtdata)")


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


def _get_update_status():
    if not os.path.exists(PID_FILE):
        return False, None
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return False, None
    if _is_process_alive(pid):
        return True, pid
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    return False, None


def _read_log_tail(n=80):
    if not os.path.exists(LOG_FILE):
        return ''
    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except OSError:
        return ''


def _start_update(args=None):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    cmd = [VENV_PYTHON, '-u', UPDATE_SCRIPT]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    log_fh = open(LOG_FILE, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    with open(PID_FILE, 'w') as f:
        f.write(str(proc.pid))
    return proc.pid


def _stop_update():
    running, pid = _get_update_status()
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
        os.remove(PID_FILE)
    except OSError:
        pass


def _get_csv_last_date(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', usecols=['date'])
        return str(df['date'].iloc[-1])
    except Exception:
        return '未知'


# ============================================================
# 区域1: 数据状态
# ============================================================
st.markdown("---")
st.markdown("##### 数据状态")

col1, col2, col3 = st.columns(3)

with col1:
    if os.path.exists(ZZ1000_PATH):
        zz_date = _get_csv_last_date(ZZ1000_PATH)
        st.metric("中证1000", zz_date)
    else:
        st.metric("中证1000", "未生成")

with col2:
    stock_files = glob.glob(os.path.join(STOCKS_DIR, '*.csv'))
    st.metric("个股数量", f"{len(stock_files)} 只")

with col3:
    if stock_files:
        # 抽样检查最新日期
        sample = stock_files[-1]
        stock_date = _get_csv_last_date(sample)
        sample_name = os.path.basename(sample).replace('.csv', '')
        st.metric("个股最新日期", stock_date, delta=f"样本: {sample_name}")
    else:
        st.metric("个股最新日期", "无数据")

# ============================================================
# 区域2: 数据更新
# ============================================================
st.markdown("---")
st.markdown("##### 数据更新")

running, pid = _get_update_status()

if running:
    st.warning(f"数据更新中 (PID: {pid})")
    if st.button("停止更新", type="secondary"):
        _stop_update()
        st.rerun()
else:
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        log_tail = _read_log_tail(5)
        if '更新完成' in log_tail:
            st.success("上次更新已完成")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("更新全部数据", type="primary"):
            try:
                new_pid = _start_update()
                st.success(f"更新已启动 (PID: {new_pid})")
                st.rerun()
            except Exception as e:
                st.error(f"启动失败: {e}")
    with c2:
        if st.button("只更新中证1000"):
            try:
                new_pid = _start_update(['--zz1000-only'])
                st.success(f"更新已启动 (PID: {new_pid})")
                st.rerun()
            except Exception as e:
                st.error(f"启动失败: {e}")
    with c3:
        stock_code = st.text_input("单只股票代码", placeholder="000001")
        if st.button("更新单只") and stock_code:
            try:
                new_pid = _start_update(['--stock', stock_code.strip()])
                st.success(f"更新已启动 (PID: {new_pid})")
                st.rerun()
            except Exception as e:
                st.error(f"启动失败: {e}")


# ============================================================
# 日志显示
# ============================================================
@st.fragment(run_every=3 if running else None)
def show_log():
    is_running, _ = _get_update_status()
    log = _read_log_tail(80)
    if log:
        st.markdown("##### 更新日志")
        st.code(log, language=None)
    if not is_running and running:
        st.cache_data.clear()
        st.rerun()

show_log()
