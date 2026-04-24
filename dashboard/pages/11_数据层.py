# -*- coding: utf-8 -*-
"""
数据层 — 架构总览 · 原始数据检测 · 数据层状态 · 增量更新
"""
import sys
import os
import glob
import subprocess
import zipfile
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd

from data_layer.foundation_config import PATHS, FOUNDATION_DATA_DIR, STOCKS_DATA_DIR, foundation_file

st.set_page_config(page_title="数据层", page_icon="🗂️", layout="wide")
st.title("🗂️ 数据层")
st.caption("架构总览 · 原始数据检测 · 数据层状态 · 增量更新")

VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv311', 'Scripts', 'python.exe')
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable

RUNTIME_DIR = os.path.join(PROJECT_ROOT, 'live', 'runtime')
PID_FILE = os.path.join(RUNTIME_DIR, 'update_datalayer.pid')
LOG_FILE = os.path.join(RUNTIME_DIR, 'update_datalayer.log')

DATA_DIR = os.path.join(PROJECT_ROOT, 'data_layer', 'data')
ZZ1000_PATH = os.path.join(DATA_DIR, 'zz1000_daily.csv')


# ============================================================
# 工具函数
# ============================================================
def _csv_last_date(path, date_col='date'):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', usecols=[date_col])
        return str(df[date_col].iloc[-1])
    except Exception:
        return None


def _csv_row_count(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig', usecols=[0])
        return len(df)
    except Exception:
        return None


def _file_size_mb(path):
    try:
        return os.path.getsize(path) / 1024 / 1024
    except Exception:
        return 0.0


def _zip_last_modified(path):
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')
    except Exception:
        return None


def _dir_latest_csv_date(root):
    """扫描增量CSV目录,返回最新日期(YYYY-MM-DD格式)和文件数"""
    if not os.path.exists(root):
        return None, 0
    latest = ''
    count = 0
    for month_dir in os.listdir(root):
        month_path = os.path.join(root, month_dir)
        if not os.path.isdir(month_path):
            continue
        for fname in os.listdir(month_path):
            if not fname.endswith('.csv'):
                continue
            count += 1
            date_str = fname[:8]
            if date_str.isdigit() and date_str > latest:
                latest = date_str
    if latest:
        return f'{latest[:4]}-{latest[4:6]}-{latest[6:8]}', count
    return None, count


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


def _start_update(script_path, args=None, label=''):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    cmd = [VENV_PYTHON, '-u', script_path]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    log_fh = open(LOG_FILE, 'w', encoding='utf-8')
    if label:
        log_fh.write(f'=== {label} ===\n')
        log_fh.flush()
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


# ============================================================
# Tab 布局
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["架构总览", "原始数据检测", "数据层状态", "增量更新"])


# ============================================================
# Tab1: 架构总览
# ============================================================
with tab1:
    st.markdown("""
### 数据管线架构

```
原始数据源 (百度网盘同步)
  │
  ├── ZIP 汇总包
  │     ├── 个股前复权日线  daily_qfq.zip
  │     └── 指数日线行情   指数_日_kline.zip
  │
  ├── 增量 CSV (每日更新)
  │     ├── 每日指标      → 个股 OHLC / 换手 / 市值 / PE / PB
  │     ├── 指数日线行情   → 中证1000 等 OHLC
  │     ├── 资金流向      → 大小单净流入
  │     └── 筹码数据      → 成本分位 / 胜率
  │
  ├── 榜单数据
  │     ├── 涨停 / 跌停家数统计
  │     ├── 炸板汇总
  │     └── 连板天梯
  │
  └── 板块成分
        ├── 行业板块成分
        └── 概念板块成分
```

```
Layer 1: update_daily.py
  ├── zz1000_daily.csv     中证1000日线 + 趋势线 + 主力线 + 象卦
  └── stocks/*.csv (5000+) 个股日线 + 趋势线 + 散户线 + 主力线 + 象卦
      (数据源: ZIP为主 + 每日CSV补充)
```

```
Layer 2: update_foundation.py
  ├── 1. main_board_universe.csv     主板样本池 (筛选口径)
  ├── 2. daily_cross_section.csv     每日截面宽表 (多源合并)
  ├── 3. daily_5d_scores.csv         5维分数 (势/时/变/重/气)
  ├── 4. daily_forward_returns.csv   前瞻收益 (10/20/60日)
  ├── 5. market_bagua_daily.csv      市场卦 (趋势+动量+广度→三爻)
  └── 6. stock_bagua_daily.csv       个股卦 (个股级三爻编码)
```

```
Layer 3: 回测 / 实盘
  └── 由 backtest_capital.py / live/ 调用 Layer 1 + Layer 2 数据
```
""")

    st.markdown("---")
    st.markdown("#### 原始数据源路径一览")
    source_rows = []
    for key, path in PATHS.items():
        exists = os.path.exists(path)
        source_rows.append({
            '键名': key,
            '路径': path,
            '类型': '目录' if os.path.isdir(path) else ('文件' if os.path.isfile(path) else '-'),
            '状态': '✅' if exists else '❌',
        })
    st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)


# ============================================================
# Tab2: 原始数据检测
# ============================================================
with tab2:
    st.markdown("#### 原始数据源最新日期")

    if st.button("刷新检测", key='refresh_raw'):
        st.cache_data.clear()

    raw_items = []

    # --- ZIP 文件 ---
    zip_sources = [
        ('个股前复权日线 ZIP', 'E:/BaiduSyncdisk/A股数据_zip/daily_qfq.zip'),
        ('指数日线行情 ZIP (legacy)', PATHS.get('index_daily_zip_legacy', '')),
        ('指数日线行情 ZIP', PATHS.get('index_daily_zip', '')),
    ]
    for label, path in zip_sources:
        if os.path.exists(path):
            mod = _zip_last_modified(path)
            sz = _file_size_mb(path)
            raw_items.append({'数据源': label, '最新日期': f'文件修改: {mod}', '大小': f'{sz:.0f} MB', '状态': '✅'})
        else:
            raw_items.append({'数据源': label, '最新日期': '-', '大小': '-', '状态': '❌ 不存在'})

    # --- 增量 CSV 目录 ---
    csv_dirs = [
        ('每日指标 (个股)', PATHS.get('stock_daily_metrics_root', '')),
        ('资金流向 (每日)', PATHS.get('stock_moneyflow_root', '')),
        ('资金流向 (年汇总)', PATHS.get('stock_moneyflow_year_root', '')),
        ('筹码数据', PATHS.get('chip_root', '')),
        ('指数每日指标', PATHS.get('index_daily_root', '')),
        ('指数日线行情 CSV', 'E:/BaiduSyncdisk/指数数据/增量数据/指数日线行情'),
        ('连板天梯', PATHS.get('limit_ladder_root', '')),
        ('涨停榜单 (年汇总)', PATHS.get('limit_up_detail_root', '')),
        ('行业板块成分', PATHS.get('industry_component_root', '')),
        ('概念板块成分', PATHS.get('concept_component_root', '')),
    ]
    for label, root in csv_dirs:
        if os.path.exists(root):
            latest, cnt = _dir_latest_csv_date(root)
            raw_items.append({
                '数据源': label,
                '最新日期': latest or '无CSV',
                '大小': f'{cnt} 个文件',
                '状态': '✅',
            })
        else:
            raw_items.append({'数据源': label, '最新日期': '-', '大小': '-', '状态': '❌ 不存在'})

    # --- 单文件 CSV ---
    csv_files = [
        ('股票列表', PATHS.get('stock_basic', '')),
        ('涨停家数统计', PATHS.get('limit_up_summary', '')),
        ('跌停家数统计', PATHS.get('limit_down_summary', '')),
        ('炸板汇总', PATHS.get('limit_broken_board_summary', '')),
        ('行业板块成分 (全量)', PATHS.get('industry_component_full', '')),
        ('概念板块成分 (全量)', PATHS.get('concept_component_full', '')),
        ('指数基本信息', PATHS.get('index_basic_csi', '')),
    ]
    for label, path in csv_files:
        if os.path.exists(path):
            sz = _file_size_mb(path)
            mod = _zip_last_modified(path)
            raw_items.append({
                '数据源': label,
                '最新日期': f'文件修改: {mod}',
                '大小': f'{sz:.1f} MB',
                '状态': '✅',
            })
        else:
            raw_items.append({'数据源': label, '最新日期': '-', '大小': '-', '状态': '❌ 不存在'})

    df_raw = pd.DataFrame(raw_items)
    st.dataframe(df_raw, use_container_width=True, hide_index=True)


# ============================================================
# Tab3: 数据层状态
# ============================================================
with tab3:
    st.markdown("#### Layer 1: 三线指标层")

    if st.button("刷新状态", key='refresh_layer'):
        st.cache_data.clear()

    l1_col1, l1_col2, l1_col3 = st.columns(3)
    with l1_col1:
        if os.path.exists(ZZ1000_PATH):
            zz_date = _csv_last_date(ZZ1000_PATH)
            zz_rows = _csv_row_count(ZZ1000_PATH)
            st.metric("中证1000", zz_date, delta=f"{zz_rows} 行")
        else:
            st.metric("中证1000", "未生成")

    with l1_col2:
        stock_files = glob.glob(os.path.join(STOCKS_DATA_DIR, '*.csv'))
        st.metric("个股数量", f"{len(stock_files)} 只")

    with l1_col3:
        if stock_files:
            sample = stock_files[-1]
            stock_date = _csv_last_date(sample)
            sample_name = os.path.basename(sample).replace('.csv', '')
            st.metric("个股最新日期", stock_date, delta=f"样本: {sample_name}")
        else:
            st.metric("个股最新日期", "无数据")

    st.markdown("---")
    st.markdown("#### Layer 2: Foundation 底座层")

    foundation_files = [
        ('主板样本池', 'main_board_universe.csv'),
        ('每日截面', 'daily_cross_section.csv'),
        ('5维分数', 'daily_5d_scores.csv'),
        ('前瞻收益', 'daily_forward_returns.csv'),
        ('市场卦', 'market_bagua_daily.csv'),
        ('个股卦', 'stock_bagua_daily.csv'),
    ]

    f_rows = []
    for label, fname in foundation_files:
        path = foundation_file(fname)
        if os.path.exists(path):
            last = _csv_last_date(path)
            sz = _file_size_mb(path)
            f_rows.append({
                '名称': label,
                '文件': fname,
                '最新日期': last or '未知',
                '大小': f'{sz:.1f} MB',
                '状态': '✅',
            })
        else:
            f_rows.append({
                '名称': label,
                '文件': fname,
                '最新日期': '-',
                '大小': '-',
                '状态': '❌ 不存在',
            })

    st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

    # 日期一致性检查
    dates = [r['最新日期'] for r in f_rows if r['最新日期'] not in ('-', '未知', None)]
    if dates:
        if len(set(dates)) == 1:
            st.success(f"所有 Foundation 文件日期一致: {dates[0]}")
        else:
            st.warning(f"日期不一致: {dict(zip([r['名称'] for r in f_rows if r['最新日期'] not in ('-', '未知', None)], dates))}")


# ============================================================
# Tab4: 增量更新
# ============================================================
with tab4:
    st.markdown("#### 增量更新")

    running, pid = _get_update_status()

    if running:
        st.warning(f"更新进行中 (PID: {pid})")
        if st.button("停止更新", type="secondary"):
            _stop_update()
            st.rerun()
    else:
        if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
            log_tail = _read_log_tail(5)
            if '更新完成' in log_tail or '全部完成' in log_tail:
                st.success("上次更新已完成")

        st.markdown("""
> **更新顺序**: Layer 1 (三线指标) → Layer 2 (Foundation 底座)
> - Layer 1: 从 ZIP + 每日CSV 更新中证1000 和全市场个股
> - Layer 2: 从 Layer 1 + 原始数据源更新 7 个 Foundation 文件
""")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("更新全部 (L1+L2)", type="primary"):
                script = os.path.join(PROJECT_ROOT, 'data_layer', '_run_full_update.py')
                _write_full_update_script(script)
                try:
                    new_pid = _start_update(script, label='全量更新 L1+L2')
                    st.success(f"更新已启动 (PID: {new_pid})")
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败: {e}")

        with c2:
            if st.button("只更新 Layer 1"):
                script = os.path.join(PROJECT_ROOT, 'data_layer', 'update_daily.py')
                try:
                    new_pid = _start_update(script, label='Layer 1 更新')
                    st.success(f"更新已启动 (PID: {new_pid})")
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败: {e}")

        with c3:
            if st.button("只更新 Layer 2"):
                script = os.path.join(PROJECT_ROOT, 'data_layer', 'update_foundation.py')
                try:
                    new_pid = _start_update(script, label='Layer 2 (Foundation) 更新')
                    st.success(f"更新已启动 (PID: {new_pid})")
                    st.rerun()
                except Exception as e:
                    st.error(f"启动失败: {e}")

    # 日志显示
    @st.fragment(run_every=3 if running else None)
    def show_log():
        is_running, _ = _get_update_status()
        log = _read_log_tail(100)
        if log:
            st.markdown("##### 更新日志")
            st.code(log, language=None)
        if not is_running and running:
            st.cache_data.clear()
            st.rerun()

    show_log()


def _write_full_update_script(path):
    """生成一个串联 L1+L2 的临时脚本"""
    content = '''# -*- coding: utf-8 -*-
"""全量更新: Layer 1 + Layer 2"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.update_daily import update_all as update_layer1
from data_layer.update_foundation import update_all as update_layer2

print("=" * 60)
print("  全量更新: Layer 1 + Layer 2")
print("=" * 60)

print("\\n>>> Layer 1: 三线指标更新")
update_layer1()

print("\\n>>> Layer 2: Foundation 底座更新")
update_layer2()

print("\\n" + "=" * 60)
print("  全量更新完成")
print("=" * 60)
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
