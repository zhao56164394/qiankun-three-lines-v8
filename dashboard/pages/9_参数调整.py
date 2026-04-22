# -*- coding: utf-8 -*-
"""
参数调整 — 左右对比: 回测(左) vs 实盘(右)

读取 backtest_bt/config.py 和 live/config.py，
左右对比显示，不一致的实盘参数标红。
"""
import sys
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="参数调整", page_icon="⚙", layout="wide")
st.title("⚙ 参数调整")
st.caption("回测(左) vs 实盘(右) — 不一致的实盘参数标红")


LIVE_CONFIG = os.path.join(PROJECT_ROOT, 'live', 'config.py')
BT_CONFIG = os.path.join(PROJECT_ROOT, 'backtest_bt', 'config.py')


# ============================================================
# 读写配置文件
# ============================================================
def _read_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _parse_value(raw):
    raw = raw.strip()
    try:
        return eval(raw)
    except Exception:
        return raw


def _write_param(content, name, new_val):
    if isinstance(new_val, set):
        items = ', '.join(repr(x) for x in sorted(new_val))
        val_str = '{' + items + '}'
    elif isinstance(new_val, dict):
        items = ', '.join(f'{repr(k)}: {repr(v)}' for k, v in new_val.items())
        val_str = '{' + items + '}'
    elif isinstance(new_val, str):
        val_str = repr(new_val)
    else:
        val_str = repr(new_val)

    multi_pattern = re.compile(
        rf'^({name}\s*=\s*)\{{[^}}]*\}}',
        re.MULTILINE | re.DOTALL,
    )
    multi_match = multi_pattern.search(content)
    if multi_match:
        prefix = multi_match.group(1)
        return content[:multi_match.start()] + f'{prefix}{val_str}' + content[multi_match.end():]

    pattern = re.compile(
        rf'^({name}\s*=\s*)(.+?)(\s*#.*)?$',
        re.MULTILINE,
    )
    match = pattern.search(content)
    if match:
        prefix = match.group(1)
        comment = match.group(3) or ''
        return content[:match.start()] + f'{prefix}{val_str}{comment}' + content[match.end():]

    return content


def _extract_param(content, name, default=None):
    multi_pattern = re.compile(
        rf'^{name}\s*=\s*(\{{[^}}]*\}})',
        re.MULTILINE | re.DOTALL,
    )
    multi_match = multi_pattern.search(content)
    if multi_match:
        return _parse_value(multi_match.group(1))

    pattern = re.compile(rf'^{name}\s*=\s*(.+?)(\s*#.*)?$', re.MULTILINE)
    match = pattern.search(content)
    if match:
        return _parse_value(match.group(1))
    return default


def _save_configs(params_live, params_bt):
    live_content = _read_config(LIVE_CONFIG)
    bt_content = _read_config(BT_CONFIG)
    for name, val in params_live.items():
        live_content = _write_param(live_content, name, val)
    for name, val in params_bt.items():
        bt_content = _write_param(bt_content, name, val)
    with open(LIVE_CONFIG, 'w', encoding='utf-8') as f:
        f.write(live_content)
    with open(BT_CONFIG, 'w', encoding='utf-8') as f:
        f.write(bt_content)


# ============================================================
# 辅助: 带差异标红的label
# ============================================================
def _label(text, bt_val, live_val):
    """如果实盘值≠回测值, label加红色标记"""
    if bt_val != live_val:
        return f"🔴 {text}"
    return text


# ============================================================
# 加载当前参数
# ============================================================
try:
    live_src = _read_config(LIVE_CONFIG)
    bt_src = _read_config(BT_CONFIG)
except FileNotFoundError as e:
    st.error(f"配置文件不存在: {e}")
    st.stop()


# ============================================================
# 🔍 选股参数
# ============================================================
st.markdown("---")
st.subheader("🔍 选股参数")

col_bt, col_live = st.columns(2)

# --- 回测端 ---
with col_bt:
    st.markdown("**📊 回测**")
    sc1, sc2 = st.columns(2)
    with sc1:
        bt_pool = st.number_input(
            "散户线入池阈值", key="bt_pool",
            value=int(_extract_param(bt_src, 'POOL_THRESHOLD', -400)),
            min_value=-2000, max_value=0, step=50,
            help="越低越严格，散户线 < 此值才入选股池",
        )
    with sc2:
        bt_trend_trigger = st.number_input(
            "趋势线触发阈值", key="bt_trend_trigger",
            value=int(_extract_param(bt_src, 'TREND_BUY_ABOVE', 11)),
            min_value=0, max_value=50, step=1,
            help="双升时趋势线需 > 此值才生成买入信号",
        )
    # v1.1 过滤参数
    fc1, fc2 = st.columns(2)
    with fc1:
        bt_filter_trend = st.number_input(
            "买点趋势线上限 (v1.1)", key="bt_filter_trend",
            value=int(_extract_param(bt_src, 'FILTER_TREND_AT_BUY_MAX', 20)),
            min_value=10, max_value=80, step=1,
            help="买点趋势线>此值 → 跳过(已涨太多)",
        )
    with fc2:
        bt_filter_retail = st.number_input(
            "散户线回升上限 (v1.1)", key="bt_filter_retail",
            value=int(_extract_param(bt_src, 'FILTER_RETAIL_RECOVERY_MAX', 500)),
            min_value=100, max_value=2000, step=50,
            help="散户线回升幅度>此值 → 跳过(底部已过)",
        )
    bt_min_512 = st.number_input(
        "512分级最小样本数", key="bt_min_512",
        value=int(_extract_param(bt_src, 'MIN_512_SAMPLES', 3)),
        min_value=1, max_value=20, step=1,
    )

# --- 实盘端 ---
live_pool_cur = int(_extract_param(live_src, 'POOL_THRESHOLD', -400))
live_trend_trigger_cur = int(_extract_param(live_src, 'TREND_TRIGGER', 11))
live_filter_trend_cur = int(_extract_param(live_src, 'FILTER_TREND_AT_BUY_MAX', 20))
live_filter_retail_cur = int(_extract_param(live_src, 'FILTER_RETAIL_RECOVERY_MAX', 500))
live_min_512_cur = int(_extract_param(live_src, 'MIN_512_SAMPLES', 3))

with col_live:
    st.markdown("**🎯 实盘**")
    sc1, sc2 = st.columns(2)
    with sc1:
        live_pool = st.number_input(
            _label("散户线入池阈值", bt_pool, live_pool_cur), key="live_pool",
            value=live_pool_cur,
            min_value=-2000, max_value=0, step=50,
        )
    with sc2:
        live_trend_trigger = st.number_input(
            _label("趋势线触发阈值", bt_trend_trigger, live_trend_trigger_cur), key="live_trend_trigger",
            value=live_trend_trigger_cur,
            min_value=0, max_value=50, step=1,
        )
    fc1, fc2 = st.columns(2)
    with fc1:
        live_filter_trend = st.number_input(
            _label("买点趋势线上限 (v1.1)", bt_filter_trend, live_filter_trend_cur), key="live_filter_trend",
            value=live_filter_trend_cur,
            min_value=10, max_value=80, step=1,
        )
    with fc2:
        live_filter_retail = st.number_input(
            _label("散户线回升上限 (v1.1)", bt_filter_retail, live_filter_retail_cur), key="live_filter_retail",
            value=live_filter_retail_cur,
            min_value=100, max_value=2000, step=50,
        )
    live_min_512 = st.number_input(
        _label("512分级最小样本数", bt_min_512, live_min_512_cur), key="live_min_512",
        value=live_min_512_cur,
        min_value=1, max_value=20, step=1,
    )

# 等级过滤
st.markdown("##### 等级过滤 (512分级)")
grade_options = ['A+', 'A', 'B+', 'B', 'B-', 'C', 'D', 'F']

gc_bt, gc_live = st.columns(2)
with gc_bt:
    bt_crazy_cur = _extract_param(bt_src, 'CRAZY_ALLOWED', {'A+', 'A', 'B+', 'B', 'B-', 'D'})
    bt_crazy_grades = st.multiselect(
        "疯狂模式允许等级", key="bt_crazy_grades",
        options=grade_options,
        default=[g for g in grade_options if g in bt_crazy_cur],
    )
    bt_normal_cur = _extract_param(bt_src, 'NORMAL_ALLOWED', {'A+'})
    bt_normal_grades = st.multiselect(
        "常规模式允许等级", key="bt_normal_grades",
        options=grade_options,
        default=[g for g in grade_options if g in bt_normal_cur],
    )

live_crazy_cur = _extract_param(live_src, 'CRAZY_ALLOWED_GRADES', {'A+', 'A', 'B+', 'B', 'B-', 'D'})
live_normal_cur = _extract_param(live_src, 'NORMAL_ALLOWED_GRADES', {'A+'})

with gc_live:
    live_crazy_grades = st.multiselect(
        _label("疯狂模式允许等级", set(bt_crazy_grades), live_crazy_cur), key="live_crazy_grades",
        options=grade_options,
        default=[g for g in grade_options if g in live_crazy_cur],
    )
    live_normal_grades = st.multiselect(
        _label("常规模式允许等级", set(bt_normal_grades), live_normal_cur), key="live_normal_grades",
        options=grade_options,
        default=[g for g in grade_options if g in live_normal_cur],
    )


# ============================================================
# 💰 仓位参数
# ============================================================
st.markdown("---")
st.subheader("💰 仓位参数")

col_bt, col_live = st.columns(2)

bt_max_pos_cur = int(_extract_param(bt_src, 'MAX_POSITIONS', 5))
bt_daily_cur = int(_extract_param(bt_src, 'DAILY_BUY_LIMIT', 1))
bt_pos_mode_cur = _extract_param(bt_src, 'POSITION_MODE', 'equal')
live_max_pos_cur = int(_extract_param(live_src, 'MAX_POSITIONS', 5))
live_daily_cur = int(_extract_param(live_src, 'DAILY_BUY_LIMIT', 1))
live_pos_mode_cur = _extract_param(live_src, 'POSITION_MODE', 'equal')
live_min_buy_cur = int(_extract_param(live_src, 'MIN_BUY_AMOUNT', 1000))

with col_bt:
    st.markdown("**📊 回测**")
    pc1, pc2 = st.columns(2)
    with pc1:
        bt_max_pos = st.number_input(
            "最大持仓数", key="bt_max_pos",
            value=bt_max_pos_cur, min_value=1, max_value=20, step=1,
        )
    with pc2:
        bt_daily = st.number_input(
            "每日买入上限", key="bt_daily",
            value=bt_daily_cur, min_value=1, max_value=10, step=1,
        )
    bt_pos_mode = st.selectbox(
        "仓位模式", key="bt_pos_mode",
        options=['equal', 'available'],
        index=0 if bt_pos_mode_cur == 'equal' else 1,
    )

with col_live:
    st.markdown("**🎯 实盘**")
    pc1, pc2 = st.columns(2)
    with pc1:
        live_max_pos = st.number_input(
            _label("最大持仓数", bt_max_pos, live_max_pos_cur), key="live_max_pos",
            value=live_max_pos_cur, min_value=1, max_value=20, step=1,
        )
    with pc2:
        live_daily = st.number_input(
            _label("每日买入上限", bt_daily, live_daily_cur), key="live_daily",
            value=live_daily_cur, min_value=1, max_value=10, step=1,
        )
    live_pos_mode = st.selectbox(
        _label("仓位模式", bt_pos_mode, live_pos_mode_cur), key="live_pos_mode",
        options=['equal', 'available'],
        index=0 if live_pos_mode_cur == 'equal' else 1,
    )
    live_min_buy = st.number_input(
        "最小买入金额(元)", key="live_min_buy",
        value=live_min_buy_cur, min_value=100, max_value=50000, step=100,
        help="实盘专用: 单笔买入金额低于此值则不买",
    )


# ============================================================
# 📈 买入参数 (疯狂模式触发)
# ============================================================
st.markdown("---")
st.subheader("📈 疯狂模式触发")

col_bt, col_live = st.columns(2)

bt_crazy_trend_cur = int(_extract_param(bt_src, 'CRAZY_TREND_THRESHOLD', 45))
live_crazy_trend_cur = int(_extract_param(live_src, 'CRAZY_TREND_THRESHOLD', 45))
live_crazy_mf_cur = int(_extract_param(live_src, 'CRAZY_MF_THRESHOLD', 0))

with col_bt:
    st.markdown("**📊 回测**")
    bt_crazy_trend = st.number_input(
        "中证趋势线阈值", key="bt_crazy_trend",
        value=bt_crazy_trend_cur, min_value=0, max_value=100, step=5,
        help="中证1000 trend < 此值时进入疯狂模式",
    )

with col_live:
    st.markdown("**🎯 实盘**")
    live_crazy_trend = st.number_input(
        _label("中证趋势线阈值", bt_crazy_trend, live_crazy_trend_cur), key="live_crazy_trend",
        value=live_crazy_trend_cur, min_value=0, max_value=100, step=5,
    )
    live_crazy_mf = st.number_input(
        "中证主力线阈值", key="live_crazy_mf",
        value=live_crazy_mf_cur, min_value=-100, max_value=100, step=5,
        help="实盘专用: main_force > 此值时进入疯狂模式",
    )


# ============================================================
# 📉 卖出参数
# ============================================================
st.markdown("---")
st.subheader("📉 卖出参数")

col_bt, col_live = st.columns(2)

# 回测端变量名不同
bt_no_sell_cur = int(_extract_param(bt_src, 'TREND_MID_ZONE', 50))
bt_cross_cur = int(_extract_param(bt_src, 'TREND_HIGH_ZONE', 89))
bt_wave_end_cur = int(_extract_param(bt_src, 'TREND_FORCE_SELL', 11))
bt_stall_cur = int(_extract_param(bt_src, 'STALL_DAYS', 15))
bt_trail_cur = int(_extract_param(bt_src, 'TRAIL_PCT', 15))
bt_cap_cur = int(_extract_param(bt_src, 'TREND_CAP', 30))

live_no_sell_cur = int(_extract_param(live_src, 'TREND_NO_SELL_BELOW', 50))
live_cross_cur = int(_extract_param(live_src, 'TREND_CROSS_89', 89))
live_wave_end_cur = int(_extract_param(live_src, 'TREND_WAVE_END', 11))
live_stall_cur = int(_extract_param(live_src, 'STALL_DAYS', 15))
live_trail_cur = int(_extract_param(live_src, 'TRAIL_PCT', 15))
live_cap_cur = int(_extract_param(live_src, 'TREND_CAP', 30))

with col_bt:
    st.markdown("**📊 回测**")
    st.markdown("###### 趋势线卖出")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        bt_no_sell = st.number_input("不卖区", key="bt_no_sell", value=bt_no_sell_cur, min_value=0, max_value=100, step=5)
    with tc2:
        bt_cross = st.number_input("穿越线", key="bt_cross", value=bt_cross_cur, min_value=50, max_value=100, step=1)
    with tc3:
        bt_wave_end = st.number_input("波段结束", key="bt_wave_end", value=bt_wave_end_cur, min_value=0, max_value=50, step=1)

    st.markdown("###### 停滞止损 (疯狂)")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        bt_stall = st.number_input("停滞天数", key="bt_stall", value=bt_stall_cur, min_value=3, max_value=30, step=1)
    with sc2:
        bt_trail = st.number_input("回撤%", key="bt_trail", value=bt_trail_cur, min_value=3, max_value=50, step=1)
    with sc3:
        bt_cap = st.number_input("峰值上限", key="bt_cap", value=bt_cap_cur, min_value=10, max_value=80, step=5)

with col_live:
    st.markdown("**🎯 实盘**")
    st.markdown("###### 趋势线卖出")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        live_no_sell = st.number_input(
            _label("不卖区", bt_no_sell, live_no_sell_cur), key="live_no_sell",
            value=live_no_sell_cur, min_value=0, max_value=100, step=5)
    with tc2:
        live_cross = st.number_input(
            _label("穿越线", bt_cross, live_cross_cur), key="live_cross",
            value=live_cross_cur, min_value=50, max_value=100, step=1)
    with tc3:
        live_wave_end = st.number_input(
            _label("波段结束", bt_wave_end, live_wave_end_cur), key="live_wave_end",
            value=live_wave_end_cur, min_value=0, max_value=50, step=1)

    st.markdown("###### 停滞止损 (疯狂)")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        live_stall = st.number_input(
            _label("停滞天数", bt_stall, live_stall_cur), key="live_stall",
            value=live_stall_cur, min_value=3, max_value=30, step=1)
    with sc2:
        live_trail = st.number_input(
            _label("回撤%", bt_trail, live_trail_cur), key="live_trail",
            value=live_trail_cur, min_value=3, max_value=50, step=1)
    with sc3:
        live_cap = st.number_input(
            _label("峰值上限", bt_cap, live_cap_cur), key="live_cap",
            value=live_cap_cur, min_value=10, max_value=80, step=5)

# 牛卖参数 (实盘专用)
live_bull_cur = int(_extract_param(live_src, 'BULL_SELL_CROSS89_COUNT', 2))
with col_live:
    live_bull = st.number_input(
        "牛卖穿89次数", key="live_bull",
        value=live_bull_cur, min_value=1, max_value=5, step=1,
        help="实盘专用: 牛市卖法穿89几次后卖出",
    )


# ============================================================
# 🗂 不常调整的参数 (折叠)
# ============================================================
st.markdown("---")

# --- 八卦过滤 ---
with st.expander("☰ 八卦过滤", expanded=False):
    hex_bt_cur = _extract_param(bt_src, 'SKIP_HEXAGRAMS', set())
    hex_live_cur = _extract_param(live_src, 'SKIP_HEXAGRAMS', set())

    hc1, hc2 = st.columns(2)
    with hc1:
        st.markdown("**📊 回测 — 空仓卦**")
        st.text_area(
            "六爻卦编码", key="bt_hex",
            value=', '.join(sorted(hex_bt_cur)),
            height=68,
        )
    with hc2:
        hex_match = hex_bt_cur == hex_live_cur
        st.markdown(f"**🎯 实盘 — 空仓卦** {'' if hex_match else '🔴'}")
        st.text_area(
            "六爻卦编码", key="live_hex",
            value=', '.join(sorted(hex_live_cur)),
            height=68,
        )

    # 内卦→卖法映射
    st.markdown("##### 内卦→卖法映射")
    sell_bt_cur = _extract_param(bt_src, 'INNER_SELL_METHOD', {})
    sell_live_cur = _extract_param(live_src, 'INNER_SELL_METHOD', {})

    gua_names = {'111': '乾☰', '110': '兑☱', '101': '离☲', '100': '震☳',
                 '011': '巽☴', '010': '坎☵', '001': '艮☶', '000': '坤☷'}

    sm_bt, sm_live = st.columns(2)
    bt_inner_sell = {}
    live_inner_sell = {}

    with sm_bt:
        st.markdown("**📊 回测**")
        cols = st.columns(4)
        for i, (code, name) in enumerate(gua_names.items()):
            with cols[i % 4]:
                cur = sell_bt_cur.get(code, 'bull')
                bt_inner_sell[code] = st.selectbox(name, ['bull', 'bear'],
                    index=0 if cur == 'bull' else 1, key=f'bt_sm_{code}')

    with sm_live:
        st.markdown("**🎯 实盘**")
        cols = st.columns(4)
        for i, (code, name) in enumerate(gua_names.items()):
            with cols[i % 4]:
                bt_m = bt_inner_sell.get(code, 'bull')
                live_m = sell_live_cur.get(code, 'bull')
                label = f"🔴 {name}" if bt_m != live_m else name
                live_inner_sell[code] = st.selectbox(label, ['bull', 'bear'],
                    index=0 if live_m == 'bull' else 1, key=f'live_sm_{code}')

# --- 实盘时间控制 ---
with st.expander("⏰ 实盘时间控制", expanded=False):
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        buy_start = st.text_input("买入开始", value=_extract_param(live_src, 'BUY_START_TIME', '093100'), key="buy_start")
    with tc2:
        buy_end = st.text_input("买入截止", value=_extract_param(live_src, 'BUY_END_TIME', '100000'), key="buy_end")
    with tc3:
        select_time = st.text_input("盘前选股", value=_extract_param(live_src, 'SELECT_TIME', '091500'), key="select_time")

    tc4, tc5 = st.columns(2)
    with tc4:
        sell_rt = st.text_input("实时卖出检查", value=_extract_param(live_src, 'SELL_REALTIME_CHECK', '145500'), key="sell_rt")
    with tc5:
        sell_force = st.text_input("强制卖出时间", value=_extract_param(live_src, 'SELL_FORCE_TIME', '145500'), key="sell_force")

# --- 回测专用参数 ---
with st.expander("🔬 回测专用参数", expanded=False):
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        init_capital = st.number_input(
            "初始资金(元)", key="bt_capital",
            value=int(_extract_param(bt_src, 'INIT_CAPITAL', 200000)),
            min_value=10000, max_value=10000000, step=10000,
        )
    with bc2:
        bt_start = st.text_input("回测开始", value=_extract_param(bt_src, 'BACKTEST_START', '2015-01-01'), key="bt_start")
    with bc3:
        bt_end = st.text_input("回测结束", value=_extract_param(bt_src, 'BACKTEST_END', '2026-04-01'), key="bt_end")

    st.markdown("##### 交易成本")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        commission = st.number_input("佣金费率", key="commission",
            value=float(_extract_param(bt_src, 'COMMISSION_RATE', 0.0003)),
            min_value=0.0, max_value=0.01, step=0.0001, format="%.4f")
    with cc2:
        commission_min = st.number_input("最低佣金(元)", key="commission_min",
            value=float(_extract_param(bt_src, 'COMMISSION_MIN', 5.0)),
            min_value=0.0, max_value=50.0, step=1.0)
    with cc3:
        stamp_tax = st.number_input("印花税率", key="stamp_tax",
            value=float(_extract_param(bt_src, 'STAMP_TAX_RATE', 0.001)),
            min_value=0.0, max_value=0.01, step=0.0001, format="%.4f")


# ============================================================
# 保存按钮
# ============================================================
st.markdown("---")

save_col1, save_col2, save_col3 = st.columns([2, 1, 1])

with save_col1:
    st.info("💡 回测参数保存到 `backtest_bt/config.py`，实盘参数保存到 `live/config.py`，互不干扰。")

with save_col2:
    if st.button("💾 保存参数", type="primary", use_container_width=True):
        try:
            # 解析空仓卦
            def _parse_hex(key):
                text = st.session_state.get(key, '')
                result = set()
                for part in re.split(r'[,\s]+', text):
                    part = part.strip()
                    if re.match(r'^[01]{6}$', part):
                        result.add(part)
                return result

            bt_hex_set = _parse_hex('bt_hex')
            live_hex_set = _parse_hex('live_hex')

            # ---- 实盘参数 ----
            params_live = {
                'POOL_THRESHOLD': live_pool,
                'TREND_TRIGGER': live_trend_trigger,
                'FILTER_TREND_AT_BUY_MAX': live_filter_trend,
                'FILTER_RETAIL_RECOVERY_MAX': live_filter_retail,
                'MIN_512_SAMPLES': live_min_512,
                'CRAZY_ALLOWED_GRADES': set(live_crazy_grades),
                'NORMAL_ALLOWED_GRADES': set(live_normal_grades),
                'MAX_POSITIONS': live_max_pos,
                'DAILY_BUY_LIMIT': live_daily,
                'POSITION_MODE': live_pos_mode,
                'MIN_BUY_AMOUNT': live_min_buy,
                'CRAZY_TREND_THRESHOLD': live_crazy_trend,
                'CRAZY_MF_THRESHOLD': live_crazy_mf,
                'TREND_NO_SELL_BELOW': live_no_sell,
                'TREND_CROSS_89': live_cross,
                'TREND_WAVE_END': live_wave_end,
                'BULL_SELL_CROSS89_COUNT': live_bull,
                'STALL_DAYS': live_stall,
                'TRAIL_PCT': live_trail,
                'TREND_CAP': live_cap,
                'BUY_START_TIME': buy_start,
                'BUY_END_TIME': buy_end,
                'SELECT_TIME': select_time,
                'SELL_REALTIME_CHECK': sell_rt,
                'SELL_FORCE_TIME': sell_force,
                'SKIP_HEXAGRAMS': live_hex_set,
                'INNER_SELL_METHOD': live_inner_sell,
            }

            # ---- 回测参数 ----
            params_bt = {
                'POOL_THRESHOLD': bt_pool,
                'FILTER_TREND_AT_BUY_MAX': bt_filter_trend,
                'FILTER_RETAIL_RECOVERY_MAX': bt_filter_retail,
                'MIN_512_SAMPLES': bt_min_512,
                'CRAZY_ALLOWED': set(bt_crazy_grades),
                'NORMAL_ALLOWED': set(bt_normal_grades),
                'MAX_POSITIONS': bt_max_pos,
                'DAILY_BUY_LIMIT': bt_daily,
                'POSITION_MODE': bt_pos_mode,
                'CRAZY_TREND_THRESHOLD': bt_crazy_trend,
                'STALL_DAYS': bt_stall,
                'TRAIL_PCT': bt_trail,
                'TREND_CAP': bt_cap,
                'TREND_FORCE_SELL': bt_wave_end,
                'TREND_HIGH_ZONE': bt_cross,
                'TREND_MID_ZONE': bt_no_sell,
                'TREND_BUY_ABOVE': bt_trend_trigger,
                'SKIP_HEXAGRAMS': bt_hex_set,
                'INNER_SELL_METHOD': bt_inner_sell,
                'INIT_CAPITAL': init_capital,
                'BACKTEST_START': bt_start,
                'BACKTEST_END': bt_end,
                'COMMISSION_RATE': commission,
                'COMMISSION_MIN': commission_min,
                'STAMP_TAX_RATE': stamp_tax,
            }

            _save_configs(params_live, params_bt)
            st.success("✅ 参数已保存! 回测配置 + 实盘配置已更新。")
            st.balloons()

        except Exception as e:
            st.error(f"保存失败: {e}")
            import traceback
            st.code(traceback.format_exc())

with save_col3:
    if st.button("🔄 重新加载", use_container_width=True):
        st.rerun()

# 快速跳转
st.markdown("---")
lc1, lc2 = st.columns(2)
with lc1:
    st.page_link("pages/7_回测运行.py", label="🔬 去回测验证 →", use_container_width=True)
with lc2:
    st.page_link("pages/6_交易控制.py", label="🎮 去交易控制 →", use_container_width=True)
