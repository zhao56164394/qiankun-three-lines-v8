# -*- coding: utf-8 -*-
"""
八卦分治回测 — 替换 crazy+normal 双模式

核心逻辑:
  每天看中证1000大象卦 → 查表获取该卦的策略参数(选股/买入/卖出)
  不再有 crazy/normal 模式切换

策略参数来源: optimize_8gua.py 寻优结果 + 人工调整(避免小样本过拟合)

调整原则:
  1. 优先选 N≥20 的方案
  2. 独立交易卦(坤/兑/离/艮/乾/震) → 各自按专项配置运行
  3. 艮/离 是盈利核心(占总信号80%+)，用稳健参数
"""
import sys, os, io, json
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)

from backtest_capital import (
    load_zz1000, load_zz1000_full, load_stocks, load_stock_events,
    build_daily_512_snapshot, build_512_rolling_pred, grade_signal,
    to_yinyang, calc_sell_bear, calc_sell_bull, calc_sell_stall,
    calc_sell_trailing, calc_sell_trend_break,
    POOL_THRESHOLD, MIN_512_SAMPLES,
    YEAR_START, YEAR_END, INIT_CAPITAL, DATA_DIR, INNER_SELL_METHOD,
    load_big_cycle_context, build_context_stats, summarize_signal_context,
)
from data_layer.foundation_data import load_stock_bagua_map, load_market_bagua, load_daily_bagua



def _clean_gua(val):
    """清理卦码：去小数点、补零到3位"""
    s = str(val).strip()
    if s in ('nan', 'None', ''):
        return '???'
    if '.' in s:
        s = s.split('.')[0]
    return s.zfill(3) if s else '???'


def _load_stock_main_force():
    """从CSV直接读取每只股票的 main_force 列（load_stocks不含此列）"""
    stock_dir = os.path.join(DATA_DIR, 'stocks')
    mf_map = {}
    for fname in os.listdir(stock_dir):
        if not fname.endswith('.csv'):
            continue
        code = fname.replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(stock_dir, fname), encoding='utf-8-sig',
                             usecols=['date', 'main_force'])
            df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
            mf_map[code] = dict(zip(df['date'], df['main_force']))
        except Exception:
            continue
    return mf_map

GUA_NAMES = {
    '000': '坤(反转)', '001': '艮(蓄力)', '010': '坎(弱反弹)', '011': '巽(反转)',
    '100': '震(崩盘)', '101': '离(护盘)', '110': '兑(滞涨)', '111': '乾(牛顶)',
}

# 需要用实际策略收益做等级判断的卦（旧30日超额无区分力）
ACTUAL_GRADE_GUAS = {'010'}  # 坎卦


def build_gua_context_stats(trades):
    stats = {}
    for t in trades:
        gua = str(t.get('gua', '') or '')
        stats.setdefault(gua, []).append(t)

    out = {}
    for gua, items in sorted(stats.items()):
        profits = [x.get('profit', 0) for x in items]
        rets = [x.get('ret_pct', 0) for x in items]
        wins = sum(1 for p in profits if p > 0)
        out[gua] = {
            'trade_count': len(items),
            'win_rate': wins / len(items) * 100 if items else 0,
            'avg_ret': float(np.mean(rets)) if rets else 0,
            'profit': float(np.sum(profits)) if profits else 0,
        }
    return out


def build_actual_rolling_pred(sig_df, target_guas, min_hist=3):
    """为指定卦的信号用实际策略收益构建滚动预测，覆盖旧combo_pred

    对每个目标卦的信号，按时间排序，统计同个股卦(combo)在之前信号中的平均实际收益。
    严格按signal_date排序，只用过去数据 → 无未来函数。

    非目标卦的信号保持原grade不变。
    """
    # 只处理目标卦的信号
    mask = sig_df['tian_gua'].isin(target_guas)
    target_sig = sig_df[mask].sort_values('signal_date').reset_index(drop=True)

    if len(target_sig) == 0:
        return sig_df

    combo_history = {}
    preds = []

    for _, row in target_sig.iterrows():
        combo = row['combo']
        hist = combo_history.get(combo, [])
        if len(hist) >= min_hist:
            preds.append(np.mean(hist))
        else:
            preds.append(np.nan)
        combo_history.setdefault(combo, []).append(row['actual_ret'])

    target_sig['combo_pred_actual'] = preds

    # 用实际策略收益重新分级
    new_grades = []
    for _, row in target_sig.iterrows():
        pred = row['combo_pred_actual']
        if pd.isna(pred):
            new_grades.append('B')  # 无历史
        elif pred > 15:
            new_grades.append('A+')
        elif pred > 8:
            new_grades.append('A')
        elif pred > 3:
            new_grades.append('B+')
        elif pred > 0:
            new_grades.append('B')
        elif pred > -5:
            new_grades.append('B-')
        else:
            new_grades.append('D')
    target_sig['grade'] = new_grades

    # 写回原DataFrame
    sig_df = sig_df.copy()
    for col in ['grade']:
        sig_df.loc[mask, col] = target_sig[col].values

    return sig_df

# ============================================================
# 八卦策略配置
# ============================================================
# 来源: optimize_8gua.py 输出 + 三轮验证结论
#
# 调整说明:
#   000 坤: 独立策略 — 正式固定双升+kun_bear卖出
#          买入: 散户线<-250入池 → 正式固定双升(t>11)，实验可切换cross
#          过滤: 正式固定排市场坤/兑 + 个股仅兑(110)；二次验证关闭
#          卖出: kun_bear（反转卖法）
#          注: 旧系统个股年卦=巽过滤已移除(新单层象卦无对应概念)
#   001 艮: 独立母版过滤（排市场艮 + 个股仅坤/坎），正式保留双升+bear
#          二次验证关闭（gen_pool_verify=None）
#   010 坎: 深度入池(-400)+双升买+趋势跌破70卖(新等级排D/无趋势回升过滤)
#   011 巽: 共享池(-250)+双升t>11+个股仅坎(010)+池底≤-300+bear
#   100 震: 崩盘加速，独立深池方案（排市场艮/巽 + 池底≤-400 + 双升 + bull）
#   101 离: 主力护盘，趋势≤20，回升≤500
#          注: 旧系统个股日卦=巽过滤已移除(新单层象卦无对应概念)
#   110 兑: 独立策略 — 双升+个股象卦=乾+趋势≤20+dui_bear卖出
#          买入: 散户线<-200入池 → 双升信号 → 个股象卦=乾(111) → 趋势线≤20
#          卖出: dui_bear（快出卖法）
#   111 乾: 独立买入体系(上穿60) + 纯bull卖法
#          买入: 共享池散户线<-250 → 趋势线上穿60
#          过滤: 排除市场卦=离(101)/震(100)；排除个股象卦=离(101)/乾(111)
#          卖出: 纯bull（第2次下穿89卖出，兜底trend<11）

UNIFIED_POOL_THRESHOLD = -250  # 统一入池阈值(fallback)

GUA_STRATEGY = {
    '000': {'grades': set(),  # 坤卦不走等级体系，用独立买入逻辑
            'trend_max': None, 'retail_max': None, 'sell': 'kun_bear', 'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'kun_buy': True,             # 标记: 使用坤卦独立买入逻辑
            'kun_exclude_ren_gua': {'000', '110'},  # 正式固定: 排人卦坤/兑
            'kun_allow_di_gua': {'110'},              # 正式固定: 地卦仅保留兑(110)
            'kun_buy_mode': 'double_rise',             # 正式固定: 双升(t>11)
            'kun_cross_threshold': 20,                 # 上穿阈值(仅cross模式生效)
            },
    '001': {'grades': {'A+', 'A', 'B+', 'B', 'B-', 'C', 'D', 'F'},  # 艮卦=蓄力待发
            'trend_max': None,   'retail_max': None,  'sell': 'bear',   'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'gen_buy': True,                              # 标记: 使用艮卦独立买入逻辑
            'gen_allow_di_gua': {'000', '010'},         # 裸跑结论: 地卦仅保留坤/坎
            'gen_buy_mode': 'double_rise',               # 买点模式: double_rise / cross
            'gen_cross_threshold': 20,                   # 上穿阈值(仅cross模式生效)
            },
    '010': {'grades': {'A+', 'A', 'B+', 'B', 'B-'},             'trend_max': None,  'retail_max': None, 'sell': 'bear',   'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            },   # 坎: 新等级(实际策略收益)排D + 无趋势回升过滤 + 趋势跌破70卖
    '011': {'grades': {'A+', 'A', 'B+', 'B', 'B-', 'C', 'D', 'F'}, 'trend_max': None, 'retail_max': None, 'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'xun_buy': 'double_rise',               # 正式固定: 双升 t>11
            'xun_buy_param': 11,
            'xun_allow_di_gua': {'010'},           # 正式固定: 地卦仅保留坎(010)
            },  # 巽: 初始入池-300+双升t>11+个股仅坎+bear
    '100': {'grades': set(),  # 震卦=崩盘加速，新专项验证后正式启用独立深池长持方案
            'trend_max': None, 'retail_max': None, 'sell': 'bull', 'active': True,
            'pool_threshold': -300, 'pool_days_min': 1, 'pool_days_max': 7,
            'zhen_buy': True,                          # 正式启用: 走震卦独立分支
            'zhen_buy_mode': 'double_rise',           # 正式固定: 双升(t>11)
            'zhen_cross_threshold': 20,               # 上穿阈值(仅cross模式生效)
            'zhen_exclude_ren_gua': {'001', '011'},  # 正式固定: 排人卦艮/巽
            'zhen_allow_di_gua': None,               # 正式固定: 不限地卦
            },
    '101': {'grades': {'A+', 'A', 'B+', 'B', 'B-', 'D'},     # 离卦=主力护盘
            'trend_max': None,   'retail_max': None,  'sell': 'bear',   'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'li_buy': True,                            # 正式启用离卦独立买入逻辑
            'li_buy_mode': 'double_rise',             # 买点模式: double_rise / cross
            'li_cross_threshold': 20,                 # 上穿阈值(仅cross模式生效)
            'li_exclude_ren_gua': {'001'},         # 正式固定: 排人卦艮(001)
            'li_allow_di_gua': {'000'},              # 正式固定: 地卦仅保留坤(000)
            },
    '110': {'grades': set(),  # 兑卦偏高位兑现，不走等级体系，暂保留旧版独立快出逻辑
            'trend_max': None, 'retail_max': None, 'sell': 'dui_bear', 'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'dui_buy': True,             # 标记: 使用兑卦独立买入逻辑
            'dui_buy_mode': 'cross',               # 买点模式: double_rise / cross
            'dui_cross_threshold': 20,                  # 上穿买点阈值(仅cross模式生效)
            'dui_allow_di_gua': {'000', '010', '110'},  # 暂保留当前正式版：地卦仅坤/坎/兑
            'dui_exclude_ren_gua': {'110', '100'},    # 暂保留当前正式版：排人卦兑/震
            },
    '111': {'grades': set(),  # 乾卦不走等级体系，用独立买入逻辑
            'trend_max': None, 'retail_max': None, 'sell': 'qian_bull', 'active': True,
            'pool_threshold': -250, 'pool_days_min': None, 'pool_days_max': None,
            'qian_buy': True,            # 标记: 使用上穿独立买入逻辑
            'qian_cross_threshold': 60,                           # 上穿阈值
            'qian_exclude_ren_gua': set(),                    # 排除人卦(默认关闭)
            'qian_exclude_di_gua': {'101', '111'},               # 排除地卦(离乾)
            },
}


# ============================================================
# 信号扫描 — 按卦分卖法
# ============================================================
def scan_signals_8gua(stock_data, zz1000, tian_gua_map, stk_mf_map=None, big_cycle_context=None, stock_bagua_map=None, daily_bagua_map=None):
    """扫描买入信号，按天卦(市场卦)确定卖法和过滤参数"""

    def _pool_days_ok(strat, idx, start_idx):
        pd_min = strat.get('pool_days_min')
        pd_max = strat.get('pool_days_max')
        if start_idx is None or (pd_min is None and pd_max is None):
            return True
        pd = idx - start_idx
        return (pd_min is None or pd >= pd_min) and (pd_max is None or pd <= pd_max)

    def _calc_pool_days(idx, start_idx):
        return idx - start_idx if start_idx is not None else None

    sell_fns = {
        'bear': lambda sd, i: calc_sell_bear(sd, i),
        'bull': lambda sd, i: calc_sell_bull(sd, i),
        'stall': lambda sd, i: calc_sell_stall(sd, i, stall_days=15, trail_pct=15, trend_cap=30),
        'trail': lambda sd, i: calc_sell_trailing(sd, i, trail_pct=15),
        'trend_break70': lambda sd, i: calc_sell_trend_break(sd, i, trend_floor=70),
    }

    # 策略配置
    zhen_strat = GUA_STRATEGY['100']
    li_strat = GUA_STRATEGY['101']
    qian_strat = GUA_STRATEGY['111']
    dui_strat = GUA_STRATEGY['110']
    gen_strat = GUA_STRATEGY['001']
    kun_strat = GUA_STRATEGY['000']

    all_signals = []
    filter_stats = {'gua_inactive': 0, 'trend_at_buy': 0, 'retail_recovery': 0,
                     'zz_trend': 0, 'di_gua': 0, 'stk_mf': 0,
                     'qian_ren_gua': 0, 'qian_d': 0,
                     'dui_gua': 0, 'dui_trend': 0, 'dui_ren_gua': 0,
                     'gen_gua': 0, 'gen_ren_gua': 0,
                     'xun_gua': 0,
                     'zhen_gua': 0, 'zhen_ren_gua': 0, 'zhen_pool_days': 0, 'pool_days': 0,
                     'li_gua': 0, 'li_ren_gua': 0,
                     'kun_gua': 0, 'kun_ren_gua': 0}

    for code, df in stock_data.items():
        if len(df) < 35:
            continue
        dates = df['date'].values
        trend = df['trend'].values
        retail = df['retail'].values
        closes = df['close'].values
        opens = df['open'].values

        mf_dict = stk_mf_map.get(code, {}) if stk_mf_map else {}

        # === 共享池状态(8卦共用) ===
        pooled = False; pool_retail = 0
        pool_start_idx = None

        for i in range(1, len(df)):
            dt_str = str(dates[i])
            stock_ctx = stock_bagua_map.get((dt_str, str(code).zfill(6)), {}) if stock_bagua_map else {}
            di_gua = _clean_gua(stock_ctx.get('stock_gua', ''))
            if di_gua == '???':
                continue
            di_gua_name = stock_ctx.get('stock_gua_name', '')
            tian_info = tian_gua_map.get(dt_str, ('???', ''))
            tian_gua = tian_info[0] if isinstance(tian_info, tuple) else tian_info
            tian_gua_name = tian_info[1] if isinstance(tian_info, tuple) else ''
            ren_gua_ctx = daily_bagua_map.get((dt_str, str(code).zfill(6)), {}) if daily_bagua_map else {}
            ren_gua = _clean_gua(ren_gua_ctx.get('gua_code', ''))
            ren_gua_name = ren_gua_ctx.get('gua_name', '')
            context = big_cycle_context.get(dt_str, {}) if big_cycle_context else {}
            macro_gua = context.get('macro_gua', '')
            macro_gua_name = context.get('macro_gua_name', '')

            current_pool_threshold = GUA_STRATEGY.get(tian_gua, {}).get('pool_threshold', UNIFIED_POOL_THRESHOLD)

            # ============================================================
            # 共享池: 按卦独立初始入池阈值
            # ============================================================
            if not pooled:
                if not np.isnan(retail[i]) and retail[i] < current_pool_threshold:
                    pooled = True; pool_retail = retail[i]
                    pool_start_idx = i
                if pooled and not np.isnan(retail[i]):
                    pool_retail = min(pool_retail, retail[i])
            else:
                if not np.isnan(retail[i]):
                    pool_retail = min(pool_retail, retail[i])

            # 未入池则跳过信号检测
            if not pooled:
                continue

            # ============================================================
            # 乾卦: 上穿阈值信号 — 满足条件即出池
            # ============================================================
            if tian_gua == '111':
                qian_threshold = qian_strat.get('qian_cross_threshold', 60)
                if (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                        and trend[i-1] < qian_threshold and trend[i] >= qian_threshold):
                    # 上穿阈值条件满足 → 出池(不管过滤结果)
                    qian_market_block = qian_strat.get('qian_exclude_ren_gua', set())
                    if ren_gua in qian_market_block:
                        filter_stats['qian_ren_gua'] += 1
                    elif di_gua in qian_strat.get('qian_exclude_di_gua', set()):
                        filter_stats['qian_d'] += 1
                    elif not _pool_days_ok(qian_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                    else:
                        next_idx = i + 1
                        if next_idx < len(df):
                            buy_price = opens[next_idx]
                            if buy_price > 0 and not np.isnan(buy_price):
                                _, sell_idx = calc_sell_bull(df, next_idx)
                                sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                                sell_price = closes[sell_idx]
                                hold_days = sell_idx - next_idx
                                if hold_days > 0:
                                    all_signals.append({
                                        'code': code, 'signal_date': dt_str,
                                        'buy_date': str(dates[next_idx]),
                                        'sell_date': str(sell_date),
                                        'buy_price': buy_price, 'sell_price': sell_price,
                                        'actual_ret': (sell_price / buy_price - 1) * 100,
                                        'hold_days': hold_days,
                                        'pool_retail': pool_retail,
                                        'pool_days': _calc_pool_days(i, pool_start_idx),
                                        'is_skip': False,
                                        'hex_code': '',
                                        'combo': di_gua,
                                        'di_gua': di_gua,
                                        'di_gua_name': di_gua_name,
                                        'gua_yy': to_yinyang(di_gua),
                                        'tian_gua': tian_gua,
                                        'sell_method': 'qian_bull',
                                        'macro_gua': macro_gua,
                                        'macro_gua_name': macro_gua_name,
                                        'ren_gua': ren_gua,
                                        'ren_gua_name': ren_gua_name,
                                        'tian_gua_name': tian_gua_name,
                                    })
                    # 上穿60触发 → 出池清零
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 兑卦: 支持双升 / 上穿阈值 两种买点模式
            # ============================================================
            if tian_gua == '110':
                dui_mode = dui_strat.get('dui_buy_mode', 'double_rise')
                dui_cross_threshold = dui_strat.get('dui_cross_threshold', 20)
                is_dui_trigger = False
                if dui_mode == 'cross':
                    is_dui_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and trend[i-1] < dui_cross_threshold and trend[i] >= dui_cross_threshold)
                else:
                    is_dui_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                      and retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
                if is_dui_trigger:
                    # ����������� �� ��������ǰ���ŵĸ����ԣ����ų���ǰ���۽��ص��г���
                    dui_pair_whitelist = dui_strat.get('dui_market_stock_whitelist') or {}
                    if dui_pair_whitelist:
                        allowed_stocks = dui_pair_whitelist.get(str(ren_gua).zfill(3), set())
                        if di_gua not in allowed_stocks:
                            filter_stats['dui_ren_gua'] += 1
                            pooled = False; pool_retail = 0; pool_start_idx = None
                            continue
                    else:
                        dui_market_block = dui_strat.get('dui_exclude_ren_gua', set())
                        if ren_gua in dui_market_block:
                            filter_stats['dui_ren_gua'] += 1
                            pooled = False; pool_retail = 0; pool_start_idx = None
                            continue
                        dui_allow = dui_strat.get('dui_allow_di_gua')
                        if dui_allow and di_gua not in dui_allow:
                            filter_stats['dui_gua'] += 1
                            pooled = False; pool_retail = 0; pool_start_idx = None
                            continue
                    if not _pool_days_ok(dui_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    next_idx = i + 1
                    if next_idx < len(df):
                        buy_price = opens[next_idx]
                        if buy_price > 0 and not np.isnan(buy_price):
                            dui_sell_method = dui_strat.get('sell', 'dui_bear')
                            if dui_sell_method == 'dui_bear':
                                _, sell_idx = calc_sell_bear(df, i)
                            else:
                                sell_fn = sell_fns[dui_sell_method]
                                _, sell_idx = sell_fn(df, i)
                            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                            sell_price = closes[sell_idx]
                            hold_days = sell_idx - next_idx
                            if hold_days > 0:
                                all_signals.append({
                                    'code': code, 'signal_date': dt_str,
                                    'buy_date': str(dates[next_idx]),
                                    'sell_date': str(sell_date),
                                    'buy_price': buy_price, 'sell_price': sell_price,
                                    'actual_ret': (sell_price / buy_price - 1) * 100,
                                    'hold_days': hold_days,
                                    'pool_retail': pool_retail,
                                    'pool_days': _calc_pool_days(i, pool_start_idx),
                                    'is_skip': False,
                                    'hex_code': '',
                                    'combo': di_gua,
                                    'di_gua': di_gua,
                                    'di_gua_name': di_gua_name,
                                    'gua_yy': to_yinyang(di_gua),
                                    'tian_gua': tian_gua,
                                    'sell_method': dui_sell_method,
                                    'macro_gua': macro_gua,
                                    'macro_gua_name': macro_gua_name,
                                    'ren_gua': ren_gua,
                                    'ren_gua_name': ren_gua_name,
                                    'tian_gua_name': tian_gua_name,
                                })
                    # 触发买点后 → 出池清零
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 艮卦: 固定母版(排市场艮+仅坤/坎) + 支持双升/上穿 + 二次验证
            # ============================================================
            if tian_gua == '001' and gen_strat.get('gen_buy'):
                gen_mode = gen_strat.get('gen_buy_mode', 'double_rise')
                gen_cross_threshold = gen_strat.get('gen_cross_threshold', 20)
                is_gen_trigger = False
                if gen_mode == 'cross':
                    is_gen_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and trend[i-1] < gen_cross_threshold and trend[i] >= gen_cross_threshold)
                else:
                    is_gen_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                      and retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
                if is_gen_trigger:
                    gen_market_block = gen_strat.get('gen_exclude_ren_gua', set())
                    if ren_gua in gen_market_block:
                        filter_stats['gen_ren_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    gen_allow = gen_strat.get('gen_allow_di_gua')
                    if gen_allow and di_gua not in gen_allow:
                        filter_stats['gen_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    if not _pool_days_ok(gen_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    next_idx = i + 1
                    if next_idx < len(df):
                        buy_price = opens[next_idx]
                        if buy_price > 0 and not np.isnan(buy_price):
                            gen_sell_method = gen_strat.get('sell', 'bear')
                            sell_fn = sell_fns[gen_sell_method]
                            _, sell_idx = sell_fn(df, i)
                            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                            sell_price = closes[sell_idx]
                            hold_days = sell_idx - next_idx
                            if hold_days > 0:
                                all_signals.append({
                                    'code': code, 'signal_date': dt_str,
                                    'buy_date': str(dates[next_idx]),
                                    'sell_date': str(sell_date),
                                    'buy_price': buy_price, 'sell_price': sell_price,
                                    'actual_ret': (sell_price / buy_price - 1) * 100,
                                    'hold_days': hold_days,
                                    'pool_retail': pool_retail,
                                    'pool_days': _calc_pool_days(i, pool_start_idx),
                                    'is_skip': False,
                                    'hex_code': '',
                                    'combo': di_gua,
                                    'di_gua': di_gua,
                                    'di_gua_name': di_gua_name,
                                    'gua_yy': to_yinyang(di_gua),
                                    'tian_gua': tian_gua,
                                    'sell_method': f'gen_{gen_sell_method}',
                                    'macro_gua': macro_gua,
                                    'macro_gua_name': macro_gua_name,
                                    'ren_gua': ren_gua,
                                    'ren_gua_name': ren_gua_name,
                                    'tian_gua_name': tian_gua_name,
                                })
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 坤卦: 支持双升 / 上穿阈值 两种买点模式
            # ============================================================
            if tian_gua == '000':
                kun_mode = kun_strat.get('kun_buy_mode', 'double_rise')
                kun_cross_threshold = kun_strat.get('kun_cross_threshold', 20)
                is_kun_trigger = False
                if kun_mode == 'cross':
                    is_kun_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and trend[i-1] < kun_cross_threshold and trend[i] >= kun_cross_threshold)
                else:
                    is_kun_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                      and not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                      and retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
                if is_kun_trigger:
                    kun_market_block = kun_strat.get('kun_exclude_ren_gua', set())
                    if ren_gua in kun_market_block:
                        filter_stats['kun_ren_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    kun_allow = kun_strat.get('kun_allow_di_gua')
                    if kun_allow and di_gua not in kun_allow:
                        filter_stats['kun_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    if not _pool_days_ok(kun_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    next_idx = i + 1
                    if next_idx < len(df):
                        buy_price = opens[next_idx]
                        if buy_price > 0 and not np.isnan(buy_price):
                            kun_sell_method = kun_strat.get('sell', 'kun_bear')
                            kun_exec_method = kun_sell_method.removeprefix('kun_')
                            if kun_exec_method == 'bear':
                                _, sell_idx = calc_sell_bear(df, i)
                            else:
                                sell_fn = sell_fns[kun_exec_method]
                                _, sell_idx = sell_fn(df, i)
                            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                            sell_price = closes[sell_idx]
                            hold_days = sell_idx - next_idx
                            if hold_days > 0:
                                all_signals.append({
                                    'code': code, 'signal_date': dt_str,
                                    'buy_date': str(dates[next_idx]),
                                    'sell_date': str(sell_date),
                                    'buy_price': buy_price, 'sell_price': sell_price,
                                    'actual_ret': (sell_price / buy_price - 1) * 100,
                                    'hold_days': hold_days,
                                    'pool_retail': pool_retail,
                                    'pool_days': _calc_pool_days(i, pool_start_idx),
                                    'is_skip': False,
                                    'hex_code': '',
                                    'combo': di_gua,
                                    'di_gua': di_gua,
                                    'di_gua_name': di_gua_name,
                                    'gua_yy': to_yinyang(di_gua),
                                    'tian_gua': tian_gua,
                                    'sell_method': f'kun_{kun_exec_method}',
                                    'macro_gua': macro_gua,
                                    'macro_gua_name': macro_gua_name,
                                    'ren_gua': ren_gua,
                                    'ren_gua_name': ren_gua_name,
                                    'tian_gua_name': tian_gua_name,
                                })
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 巽卦: 正式独立分支（个股仅坎 + 池底≤-300 + 双升t>11 + bear）
            # ============================================================
            if tian_gua == '011':
                xun_strat = GUA_STRATEGY['011']
                xun_buy_param = xun_strat.get('xun_buy_param', 11)
                xun_buy_mode = xun_strat.get('xun_buy', 'double_rise')
                xun_allow = xun_strat.get('xun_allow_di_gua')

                if not xun_strat['active']:
                    filter_stats['gua_inactive'] += 1
                    continue

                if np.isnan(trend[i]) or np.isnan(trend[i-1]):
                    continue

                if xun_allow and di_gua not in xun_allow:
                    filter_stats['xun_gua'] += 1
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                # 双升买法: 散户上升 + 趋势上升 + 趋势 > 阈值
                if xun_buy_mode == 'double_rise':
                    is_xun_trigger = (not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                      and retail[i] > retail[i-1]
                                      and trend[i] > trend[i-1] and trend[i] > xun_buy_param)
                else:
                    is_xun_trigger = (trend[i] > trend[i-1] and trend[i] > xun_buy_param)

                if is_xun_trigger:
                    if not _pool_days_ok(xun_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    next_idx = i + 1
                    if next_idx >= len(df):
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    buy_price = opens[next_idx]
                    if buy_price <= 0 or np.isnan(buy_price):
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue

                    _, sell_idx = calc_sell_bear(df, i)
                    sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                    sell_price = closes[sell_idx]
                    hold_days = sell_idx - next_idx

                    if hold_days > 0:
                        all_signals.append({
                            'code': code, 'signal_date': dt_str,
                            'buy_date': str(dates[next_idx]),
                            'sell_date': str(sell_date),
                            'buy_price': buy_price, 'sell_price': sell_price,
                            'actual_ret': (sell_price / buy_price - 1) * 100,
                            'hold_days': hold_days,
                            'pool_retail': pool_retail,
                            'pool_days': _calc_pool_days(i, pool_start_idx),
                            'is_skip': False,
                            'hex_code': '',
                            'combo': di_gua,
                            'di_gua': di_gua,
                            'di_gua_name': di_gua_name,
                            'gua_yy': to_yinyang(di_gua),
                            'tian_gua': tian_gua,
                            'sell_method': 'xun_bear',
                            'macro_gua': macro_gua,
                            'macro_gua_name': macro_gua_name,
                            'ren_gua': ren_gua,
                            'ren_gua_name': ren_gua_name,
                            'tian_gua_name': tian_gua_name,
                        })
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 震卦: 正式独立分支（市场层/个股层/池底验证已按单变量逐层固化）
            # ============================================================
            if tian_gua == '100' and zhen_strat.get('zhen_buy'):
                zhen_mode = zhen_strat.get('zhen_buy_mode', 'double_rise')
                zhen_cross_threshold = zhen_strat.get('zhen_cross_threshold', 20)
                is_zhen_trigger = False
                if zhen_mode == 'cross':
                    is_zhen_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                       and trend[i-1] < zhen_cross_threshold and trend[i] >= zhen_cross_threshold)
                else:
                    is_zhen_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                       and not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                       and retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
                if is_zhen_trigger:
                    zhen_market_block = zhen_strat.get('zhen_exclude_ren_gua', set())
                    if ren_gua in zhen_market_block:
                        filter_stats['zhen_ren_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    zhen_allow = zhen_strat.get('zhen_allow_di_gua')
                    if zhen_allow and di_gua not in zhen_allow:
                        filter_stats['zhen_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    zhen_pd_min = zhen_strat.get('pool_days_min')
                    zhen_pd_max = zhen_strat.get('pool_days_max')
                    if pool_start_idx is not None and (zhen_pd_min is not None or zhen_pd_max is not None):
                        _pool_days = i - pool_start_idx
                        if zhen_pd_min is not None and _pool_days < zhen_pd_min:
                            filter_stats['pool_days'] += 1
                            pooled = False; pool_retail = 0; pool_start_idx = None
                            continue
                        if zhen_pd_max is not None and _pool_days > zhen_pd_max:
                            filter_stats['pool_days'] += 1
                            pooled = False; pool_retail = 0; pool_start_idx = None
                            continue
                    next_idx = i + 1
                    if next_idx < len(df):
                        buy_price = opens[next_idx]
                        if buy_price > 0 and not np.isnan(buy_price):
                            zhen_sell_method = zhen_strat.get('sell', 'bear')
                            sell_fn = sell_fns[zhen_sell_method]
                            _, sell_idx = sell_fn(df, i)
                            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                            sell_price = closes[sell_idx]
                            hold_days = sell_idx - next_idx
                            if hold_days > 0:
                                all_signals.append({
                                    'code': code, 'signal_date': dt_str,
                                    'buy_date': str(dates[next_idx]),
                                    'sell_date': str(sell_date),
                                    'buy_price': buy_price, 'sell_price': sell_price,
                                    'actual_ret': (sell_price / buy_price - 1) * 100,
                                    'hold_days': hold_days,
                                    'pool_retail': pool_retail,
                                    'pool_days': i - pool_start_idx if pool_start_idx is not None else None,
                                    'is_skip': False,
                                    'hex_code': '',
                                    'combo': di_gua,
                                    'di_gua': di_gua,
                                    'di_gua_name': di_gua_name,
                                    'gua_yy': to_yinyang(di_gua),
                                    'tian_gua': tian_gua,
                                    'sell_method': f'zhen_{zhen_sell_method}',
                                    'macro_gua': macro_gua,
                                    'macro_gua_name': macro_gua_name,
                                    'ren_gua': ren_gua,
                                    'ren_gua_name': ren_gua_name,
                                    'tian_gua_name': tian_gua_name,
                                })
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 离卦: 正式独立分支（排市场艮 + 个股仅坤 + 池底二次验证）
            # ============================================================
            if tian_gua == '101' and li_strat.get('li_buy'):
                li_mode = li_strat.get('li_buy_mode', 'double_rise')
                li_cross_threshold = li_strat.get('li_cross_threshold', 20)
                is_li_trigger = False
                if li_mode == 'cross':
                    is_li_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                     and trend[i-1] < li_cross_threshold and trend[i] >= li_cross_threshold)
                else:
                    is_li_trigger = (not np.isnan(trend[i]) and not np.isnan(trend[i-1])
                                     and not np.isnan(retail[i]) and not np.isnan(retail[i-1])
                                     and retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)
                if is_li_trigger:
                    li_market_block = li_strat.get('li_exclude_ren_gua', set())
                    if ren_gua in li_market_block:
                        filter_stats['li_ren_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    li_allow = li_strat.get('li_allow_di_gua')
                    if li_allow and di_gua not in li_allow:
                        filter_stats['li_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    if not _pool_days_ok(li_strat, i, pool_start_idx):
                        filter_stats['pool_days'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue
                    next_idx = i + 1
                    if next_idx < len(df):
                        buy_price = opens[next_idx]
                        if buy_price > 0 and not np.isnan(buy_price):
                            li_sell_method = li_strat.get('sell', 'bear')
                            sell_fn = sell_fns[li_sell_method]
                            _, sell_idx = sell_fn(df, i)
                            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                            sell_price = closes[sell_idx]
                            hold_days = sell_idx - next_idx
                            if hold_days > 0:
                                all_signals.append({
                                    'code': code, 'signal_date': dt_str,
                                    'buy_date': str(dates[next_idx]),
                                    'sell_date': str(sell_date),
                                    'buy_price': buy_price, 'sell_price': sell_price,
                                    'actual_ret': (sell_price / buy_price - 1) * 100,
                                    'hold_days': hold_days,
                                    'pool_retail': pool_retail,
                                    'pool_days': _calc_pool_days(i, pool_start_idx),
                                    'is_skip': False,
                                    'hex_code': '',
                                    'combo': di_gua,
                                    'di_gua': di_gua,
                                    'di_gua_name': di_gua_name,
                                    'gua_yy': to_yinyang(di_gua),
                                    'tian_gua': tian_gua,
                                    'sell_method': f'li_{li_sell_method}',
                                    'macro_gua': macro_gua,
                                    'macro_gua_name': macro_gua_name,
                                    'ren_gua': ren_gua,
                                    'ren_gua_name': ren_gua_name,
                                    'tian_gua_name': tian_gua_name,
                                })
                    pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # ============================================================
            # 常规卦(艮/坎/震/离): 双升信号 — 满足双升即出池
            # ============================================================
            if np.isnan(trend[i]) or np.isnan(trend[i-1]): continue
            if np.isnan(retail[i]) or np.isnan(retail[i-1]): continue

            # 双升信号
            if retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11:
                strat = GUA_STRATEGY.get(tian_gua)
                if strat is None or not strat['active']:
                    filter_stats['gua_inactive'] += 1
                    # 双升触发 → 出池清零
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                # 中证趋势线过滤
                zz_trend_max = strat.get('zz_trend_max')
                if zz_trend_max is not None:
                    zz_info = zz1000.get(dt_str, {})
                    zz_trend_val = zz_info.get('trend', 99)
                    if zz_trend_val > zz_trend_max:
                        filter_stats['zz_trend'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue

                # 买入过滤
                trend_at_buy = trend[i]
                if strat['trend_max'] is not None and trend_at_buy > strat['trend_max']:
                    filter_stats['trend_at_buy'] += 1
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                start = pool_start_idx if pool_start_idx else max(0, i - 60)
                retail_slice = retail[start:i + 1]
                valid_retail = retail_slice[~np.isnan(retail_slice)]
                retail_min_val = float(np.min(valid_retail)) if len(valid_retail) > 0 else retail[i]
                retail_recovery = retail[i] - retail_min_val
                if strat['retail_max'] is not None and retail_recovery > strat['retail_max']:
                    filter_stats['retail_recovery'] += 1
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                # 个股象卦过滤(拒绝已反弹股等)
                di_gua_reject = strat.get('di_gua_reject')
                if di_gua_reject:
                    if di_gua in di_gua_reject:
                        filter_stats['di_gua'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue

                # 个股主力线过滤(要求资金支撑)
                stk_mf_min = strat.get('stk_mf_min')
                if stk_mf_min is not None and stk_mf_map:
                    mf_val = stk_mf_map.get(code, {}).get(dt_str, None)
                    if mf_val is not None and mf_val < stk_mf_min:
                        filter_stats['stk_mf'] += 1
                        pooled = False; pool_retail = 0; pool_start_idx = None
                        continue

                if not _pool_days_ok(strat, i, pool_start_idx):
                    filter_stats['pool_days'] += 1
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                # 卖法(按中证象卦查表)
                sell_method = strat['sell']
                sell_fn = sell_fns[sell_method]
                _, sell_idx = sell_fn(df, i)

                next_idx = i + 1
                if next_idx >= len(df):
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue
                buy_date = dates[next_idx]
                buy_price = opens[next_idx]
                sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
                sell_price = closes[sell_idx]
                hold_days = sell_idx - next_idx

                if buy_price <= 0 or np.isnan(buy_price) or hold_days <= 0:
                    pooled = False; pool_retail = 0; pool_start_idx = None
                    continue

                actual_ret = (sell_price / buy_price - 1) * 100

                all_signals.append({
                    'code': code,
                    'signal_date': dt_str,
                    'buy_date': str(buy_date),
                    'sell_date': str(sell_date),
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'actual_ret': actual_ret,
                    'hold_days': hold_days,
                    'pool_retail': pool_retail,
                    'pool_days': _calc_pool_days(i, pool_start_idx),
                    'is_skip': False,
                    'hex_code': '',
                    'combo': di_gua,
                    'di_gua': di_gua,
                    'di_gua_name': di_gua_name,
                    'gua_yy': to_yinyang(di_gua),
                    'tian_gua': tian_gua,
                    'sell_method': strat['sell'],
                    'macro_gua': macro_gua,
                    'macro_gua_name': macro_gua_name,
                    'ren_gua': ren_gua,
                    'ren_gua_name': ren_gua_name,
                    'tian_gua_name': tian_gua_name,
                })
                # 双升触发 → 出池清零
                pooled = False; pool_retail = 0; pool_start_idx = None

    print(f"  过滤统计: 空仓卦={filter_stats['gua_inactive']}, "
          f"中证趋势={filter_stats['zz_trend']}, "
          f"趋势过高={filter_stats['trend_at_buy']}, "
          f"回升过大={filter_stats['retail_recovery']}, "
          f"象卦拒绝={filter_stats['di_gua']}, "
          f"主力线={filter_stats['stk_mf']}")
    print(f"  坤卦过滤: 人卦黑名单={filter_stats['kun_ren_gua']}, 象卦白名单外={filter_stats['kun_gua']}")
    print(f"  乾卦过滤: 人卦黑名单={filter_stats['qian_ren_gua']}, 象卦离乾={filter_stats['qian_d']}")
    print(f"  兑卦过滤: 人卦黑名单={filter_stats['dui_ren_gua']}, 象卦白名单外={filter_stats['dui_gua']}, 趋势过高={filter_stats['dui_trend']}")
    print(f"  艮卦过滤: 人卦黑名单={filter_stats['gen_ren_gua']}, 象卦白名单外={filter_stats['gen_gua']}")
    print(f"  巽卦过滤: 象卦白名单外={filter_stats['xun_gua']}")
    print(f"  震卦过滤: 人卦黑名单={filter_stats['zhen_ren_gua']}, 象卦白名单外={filter_stats['zhen_gua']}")
    print(f"  离卦过滤: 人卦黑名单={filter_stats['li_ren_gua']}, 象卦白名单外={filter_stats['li_gua']}")
    return pd.DataFrame(all_signals).sort_values('signal_date').reset_index(drop=True)


# ============================================================
# 模拟引擎 — 八卦分治
# ============================================================
def simulate_8gua(sig_df, zz_df, max_pos=5, daily_limit=1, init_capital=None, tian_gua_map_ext=None):
    """八卦分治模拟引擎"""
    capital = init_capital or INIT_CAPITAL
    zz_indexed = zz_df.set_index('date')

    # 天卦映射：优先外部传入，否则从信号中提取
    if tian_gua_map_ext:
        tian_gua_map = {}
        for k, v in tian_gua_map_ext.items():
            tian_gua_map[k] = v[0] if isinstance(v, tuple) else v
    else:
        def _clean_gua(val):
            s = str(val).strip()
            if '.' in s: s = s.split('.')[0]
            return s.zfill(3) if s else '???'
        tian_gua_map = {}
        for _, row in zz_df.iterrows():
            tian_gua_map[row['date']] = _clean_gua(row['gua'])

    sig_by_date = {}
    for _, row in sig_df.iterrows():
        sig_by_date.setdefault(row['signal_date'], []).append(row)

    all_dates = sorted(set(
        sig_df['signal_date'].tolist() + sig_df['sell_date'].tolist()))

    positions = []
    trade_log = []
    daily_equity = []

    for dt in all_dates:
        # 1. 卖出到期持仓
        new_pos = []
        for pos in positions:
            if pos['sell_date'] <= dt:
                profit = (pos['sell_price'] / pos['buy_price'] - 1) * pos['cost']
                capital += pos['cost'] + profit
                trade_log.append({
                    'code': pos['code'], 'buy_date': pos['buy_date'],
                    'sell_date': pos['sell_date'], 'cost': pos['cost'],
                    'profit': profit, 'buy_price': pos['buy_price'],
                    'sell_price': pos['sell_price'],
                    'ret_pct': (pos['sell_price'] / pos['buy_price'] - 1) * 100,
                    'hold_days': pos['hold_days'], 'grade': pos.get('grade', '-'),
                    'gua': pos.get('gua', '???'),
                    'di_gua': pos.get('di_gua', pos.get('combo', '???')),
                    'di_gua_name': pos.get('di_gua_name', ''),
                    'sell_method': pos.get('sell_method', '?'),
                    'macro_gua': pos.get('macro_gua', ''),
                    'macro_gua_name': pos.get('macro_gua_name', ''),
                    'tian_gua': pos.get('tian_gua', ''),
                    'tian_gua_name': pos.get('tian_gua_name', ''),
                    'ren_gua': pos.get('ren_gua', ''),
                    'ren_gua_name': pos.get('ren_gua_name', ''),
                })
            else:
                new_pos.append(pos)
        positions = new_pos

        # 2. 查当天卦策略
        tian_gua = tian_gua_map.get(dt, '???')
        strat = GUA_STRATEGY.get(tian_gua)
        if strat is None or not strat['active']:
            hold_val = sum(p['cost'] for p in positions)
            daily_equity.append({
                'date': dt, 'cash': capital, 'hold_value': hold_val,
                'total_equity': capital + hold_val, 'n_positions': len(positions),
            })
            continue

        # 3. 过滤和买入
        candidates = sig_by_date.get(dt, [])
        if candidates:
            allowed_grades = strat['grades']
            # ���������ź�����ɨ��׶����ȫ�����ˣ����ߵȼ�/skip����
            filtered = []
            for c in candidates:
                sell_method = c.get('sell_method', '')
                rank_order = int(0 if pd.isna(c.get('rank_order')) else c.get('rank_order', 0))
                if rank_order > 0 and rank_order <= 1:
                    continue
                if c.get('is_skip', False):
                    continue
                if (sell_method.startswith('qian_') or sell_method.startswith('dui_')
                        or sell_method.startswith('kun_') or sell_method.startswith('gen_')
                        or sell_method.startswith('xun_') or sell_method.startswith('zhen_')
                        or sell_method.startswith('li_')):
                    filtered.append(c)
                elif c['grade'] in allowed_grades:
                    filtered.append(c)
            filtered.sort(key=lambda x: (-int(0 if pd.isna(x.get('rank_order')) else x.get('rank_order', 0)), x['pool_retail']))
            slots = max_pos - len(positions)
            can_buy = min(slots, daily_limit, len(filtered))
            if can_buy > 0 and capital > 1000:
                total_eq = capital + sum(p['cost'] for p in positions)
                per_slot = total_eq / max_pos
                per_buy = min(per_slot, capital / can_buy)
                for j in range(can_buy):
                    cost = min(per_buy, capital)
                    if cost < 1000:
                        break
                    c = filtered[j]
                    capital -= cost
                    positions.append({
                        'code': c['code'], 'buy_date': c['buy_date'],
                        'sell_date': c['sell_date'], 'buy_price': c['buy_price'],
                        'sell_price': c['sell_price'], 'cost': cost,
                        'hold_days': c['hold_days'], 'grade': c.get('grade', '-'),
                        'gua': tian_gua,
                        'combo': c.get('combo', '???'),
                        'di_gua': c.get('di_gua', c.get('combo', '???')),
                        'di_gua_name': c.get('di_gua_name', ''),
                        'sell_method': c.get('sell_method', '?'),
                        'macro_gua': c.get('macro_gua', ''),
                        'macro_gua_name': c.get('macro_gua_name', ''),
                        'tian_gua': c.get('tian_gua', ''),
                        'tian_gua_name': c.get('tian_gua_name', ''),
                        'ren_gua': c.get('ren_gua', ''),
                        'ren_gua_name': c.get('ren_gua_name', ''),
                    })

        # 4. 记录净值
        hold_val = sum(p['cost'] for p in positions)
        daily_equity.append({
            'date': dt, 'cash': capital, 'hold_value': hold_val,
            'total_equity': capital + hold_val, 'n_positions': len(positions),
        })

    # 清仓
    for pos in positions:
        profit = (pos['sell_price'] / pos['buy_price'] - 1) * pos['cost']
        capital += pos['cost'] + profit
        trade_log.append({
            'code': pos['code'], 'buy_date': pos['buy_date'],
            'sell_date': pos['sell_date'], 'cost': pos['cost'],
            'profit': profit, 'buy_price': pos['buy_price'],
            'sell_price': pos['sell_price'],
            'ret_pct': (pos['sell_price'] / pos['buy_price'] - 1) * 100,
            'hold_days': pos['hold_days'], 'grade': pos.get('grade', '-'),
            'gua': pos.get('gua', '???'),
            'sell_method': pos.get('sell_method', '?'),
            'macro_gua': pos.get('macro_gua', ''),
            'macro_gua_name': pos.get('macro_gua_name', ''),
            'tian_gua': pos.get('tian_gua', ''),
            'tian_gua_name': pos.get('tian_gua_name', ''),
        })

    _init = init_capital or INIT_CAPITAL
    return {
        'final_capital': capital, 'init_capital': _init,
        'total_return': (capital / _init - 1) * 100,
        'trade_log': trade_log, 'daily_equity': daily_equity,
    }


# ============================================================
# 统计
# ============================================================
def calc_stats(result, label=''):
    trades = result['trade_log']
    if not trades:
        return {'label': label, 'final_capital': result['init_capital'],
                'total_return': 0, 'trade_count': 0, 'win_rate': 0,
                'avg_ret': 0, 'max_dd': 0, 'avg_hold': 0}
    rets = [t['ret_pct'] for t in trades]
    wins = [t for t in trades if t['profit'] > 0]
    eq = result['daily_equity']
    peak = result['init_capital']
    max_dd = 0; max_dd_date = ''
    for e in eq:
        if e['total_equity'] > peak:
            peak = e['total_equity']
        dd = (peak - e['total_equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd; max_dd_date = e['date']
    return {
        'label': label,
        'final_capital': result['final_capital'],
        'total_return': result['total_return'],
        'trade_count': len(trades),
        'avg_ret': np.mean(rets),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'avg_hold': np.mean([t['hold_days'] for t in trades]),
        'max_dd': max_dd,
        'max_dd_date': max_dd_date,
    }


# ============================================================
# 主流程
# ============================================================
def run(start_date=None, end_date=None, init_capital=None):
    year_start = start_date or YEAR_START
    year_end = end_date or YEAR_END
    capital = init_capital or INIT_CAPITAL
    big_cycle_context = load_big_cycle_context()
    stock_bagua_map = load_stock_bagua_map()

    print("=" * 100)
    print("  卦分治回测 — 按中证象卦(8卦)配置独立策略")
    print(f"  区间: {year_start} ~ {year_end}  初始资金: {capital:,}")
    print("=" * 100)

    # 输出策略配置
    print("\n  八卦策略配置:")
    print(f"  {'卦':<15} {'状态':<6} {'卖法':<10} {'等级':<25} {'趋势≤':>5} {'回升≤':>5} {'中证≤':>5} {'附加过滤'}")
    print("  " + "-" * 100)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        s = GUA_STRATEGY[gua]
        status = '交易' if s['active'] else '空仓'
        sell_labels = {'bear': 'bear(保守)', 'bull': 'bull(耐心)',
                       'stall': 'stall(停滞)', 'trail': 'trail(止损)',
                       'hybrid': 'hybrid(联合)', 'qian_bull': 'qian_bull(纯牛)',
                       'kun_bear': 'kun_bear(反转)', 'dui_bear': 'dui_bear(快出)',
                       'trend_break70': 'trend70(跌破)'}
        grades_str = ','.join(sorted(s['grades'])) if s['grades'] else '-'
        t_str = str(s['trend_max']) if s['trend_max'] else 'X'
        r_str = str(s['retail_max']) if s['retail_max'] else 'X'
        zz_str = str(s.get('zz_trend_max', '')) if s.get('zz_trend_max') else 'X'
        extra = []
        if gua == '000' and (s.get('kun_exclude_ren_gua')
                             or s.get('kun_buy_mode', 'double_rise') != 'double_rise'
                             or s.get('kun_allow_di_gua') not in (None, {'110'})):
            market_block = s.get('kun_exclude_ren_gua', set())
            market_desc = '/'.join(GUA_NAMES.get(g, g).split('(')[0] for g in sorted(market_block)) if market_block else '无'
            stock_allow = s.get('kun_allow_di_gua')
            stock_desc = '/'.join(GUA_NAMES.get(g, g).split('(')[0] for g in sorted(stock_allow)) if stock_allow else '不限'
            buy_mode = s.get('kun_buy_mode', 'double_rise')
            if buy_mode == 'cross':
                buy_desc = f"cross@{s.get('kun_cross_threshold', 20)}"
            else:
                buy_desc = '双升(t>11)'
            pool_desc = f"初始入池≤{s.get('pool_threshold', UNIFIED_POOL_THRESHOLD)}"
            sell_desc = s.get('sell', 'kun_bear').removeprefix('kun_')
            stock_prefix = '个股仅' if stock_allow else '个股'
            extra.append(f"坤独立:排市场{market_desc} {stock_prefix}{stock_desc} {pool_desc} {buy_desc} {sell_desc}")
            grades_str = '-'
            t_str = 'X'
            r_str = 'X'
        if gua == '100' and s.get('zhen_buy'):
            extra.append(f"震独立:排市场艮/巽 初始入池≤{s.get('pool_threshold', UNIFIED_POOL_THRESHOLD)} 双升(t>11) bull")
            grades_str = '-'
            t_str = 'X'
            r_str = 'X'
        if gua == '101' and s.get('li_buy'):
            extra.append(f"离独立:排市场艮 个股仅坤 初始入池≤{s.get('pool_threshold', UNIFIED_POOL_THRESHOLD)} 双升(t>11)")
            grades_str = '-'
            t_str = 'X'
            r_str = 'X'
        if s.get('stk_year_gua_reject'):
            rej = ','.join(GUA_NAMES.get(g, g) for g in s['stk_year_gua_reject'])
            extra.append(f"拒年卦:{rej}")
        if s.get('stk_mf_min') is not None:
            extra.append(f"主力≥{s['stk_mf_min']}")
        extra_str = ' '.join(extra) if extra else ''
        print(f"  {gua} {GUA_NAMES[gua]:<10} {status:<6} {sell_labels[s['sell']]:<10} "
              f"{grades_str:<25} {t_str:>5} {r_str:>5} {zz_str:>5} {extra_str}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    zz_df = load_zz1000_full()
    zz1000 = load_zz1000()
    stock_data = load_stocks()
    print(f"  个股: {len(stock_data)} 只")

    # 加载个股主力线(用于乾卦主力线过滤)
    stk_mf_map = _load_stock_main_force()
    print(f"  已加载 {len(stk_mf_map)} 只股票的main_force")

    # 构建天卦映射(市场卦 → 分治维度)
    market_bagua_df = load_market_bagua()
    tian_gua_map = {}
    for _, row in market_bagua_df.iterrows():
        tian_gua_map[str(row['date'])] = (_clean_gua(row['gua_code']), row.get('gua_name', ''))
    print(f"  天卦(市场卦)映射: {len(tian_gua_map)} 天")

    # 构建人卦映射(主卦/个股横截面排名)
    daily_bagua_df = load_daily_bagua()
    daily_bagua_map = {}
    for _, row in daily_bagua_df.iterrows():
        daily_bagua_map[(str(row['date']), str(row['code']).zfill(6))] = {
            'gua_code': str(row['gua_code']).zfill(3),
            'gua_name': row.get('gua_name', ''),
        }

    # 2. 扫描信号
    print("\n[2] 扫描八卦分治信号...")
    sig = scan_signals_8gua(stock_data, zz1000, tian_gua_map, stk_mf_map, big_cycle_context=big_cycle_context, stock_bagua_map=stock_bagua_map, daily_bagua_map=daily_bagua_map)
    sig = sig[(sig['signal_date'] >= year_start) &
              (sig['signal_date'] < year_end)].reset_index(drop=True)
    sig = build_512_rolling_pred(sig, min_hist=MIN_512_SAMPLES)
    sig['grade'] = [grade_signal(r['gua_yy'], r['combo_pred'])[0]
                    for _, r in sig.iterrows()]

    # 对指定卦用实际策略收益重新计算等级（旧30日超额无区分力）
    if ACTUAL_GRADE_GUAS:
        sig = build_actual_rolling_pred(sig, ACTUAL_GRADE_GUAS, min_hist=3)
        for g in ACTUAL_GRADE_GUAS:
            n_g = len(sig[sig['tian_gua'] == g])
            print(f"  {g} {GUA_NAMES[g]}: 已切换为实际策略收益等级 ({n_g}笔)")

    print(f"  总信号: {len(sig)}, 非skip: {(~sig['is_skip']).sum()}")
    signal_context = summarize_signal_context(sig)
    print(f"  大周期 macro 分布: {signal_context['macro_gua_counts']}")
    print(f"  天卦 tian 分布: {signal_context.get('tian_gua_counts', {})}")

    # 分卦信号统计
    print(f"\n  分卦信号分布:")
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub = sig[sig['tian_gua'] == gua]
        non_skip = sub[~sub['is_skip']]
        print(f"    {gua} {GUA_NAMES[gua]}: 总{len(sub)}, 非skip {len(non_skip)}")

    # 3. 模拟
    print("\n[3] 八卦分治模拟...")
    result = simulate_8gua(sig, zz_df, max_pos=5, daily_limit=1,
                           init_capital=capital, tian_gua_map_ext=tian_gua_map)
    stats = calc_stats(result, '八卦分治')
    trades = result['trade_log']
    context_stats = build_context_stats(trades)
    gua_context_stats = build_gua_context_stats(trades)

    # ============================================================
    # 输出报告
    # ============================================================
    print("\n" + "=" * 100)
    print("  Part1: 策略总览")
    print("=" * 100)
    print(f"\n  初始资金: {capital:,}")
    print(f"  终值: {stats['final_capital']:,.0f}")
    print(f"  收益: {stats['total_return']:.1f}%")
    print(f"  最大回撤: {stats['max_dd']:.1f}%")
    print(f"  交易笔数: {stats['trade_count']}")
    print(f"  胜率: {stats['win_rate']:.1f}%")
    print(f"  均收益: {stats['avg_ret']:.2f}%")
    print(f"  均持仓: {stats['avg_hold']:.1f}天")

    # Part2: 分卦贡献
    print("\n" + "=" * 100)
    print("  Part2: 分卦贡献")
    print("=" * 100)
    print(f"\n  {'卦':<15} {'笔数':>5} {'胜率%':>6} {'均收益%':>8} {'利润':>14} {'占比%':>6}")
    print("  " + "-" * 60)
    total_profit = sum(t['profit'] for t in trades)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        gua_trades = [t for t in trades if t.get('gua') == gua]
        if not gua_trades:
            print(f"  {gua} {GUA_NAMES[gua]:<10} {'空仓':>5}")
            continue
        n = len(gua_trades)
        wins = sum(1 for t in gua_trades if t['profit'] > 0)
        avg_r = np.mean([t['ret_pct'] for t in gua_trades])
        profit = sum(t['profit'] for t in gua_trades)
        pct = profit / total_profit * 100 if total_profit != 0 else 0
        print(f"  {gua} {GUA_NAMES[gua]:<10} {n:>5} {wins/n*100:>5.1f} {avg_r:>+7.2f} "
              f"{profit:>13,.0f} {pct:>5.1f}")

    # Part3: 年度明细
    print("\n" + "=" * 100)
    print("  Part3: 年度明细")
    print("=" * 100)
    yearly = {}
    for t in trades:
        y = t['buy_date'][:4]
        yearly.setdefault(y, {'profit': 0, 'count': 0, 'wins': 0})
        yearly[y]['profit'] += t['profit']
        yearly[y]['count'] += 1
        if t['profit'] > 0: yearly[y]['wins'] += 1

    print(f"\n  {'年份':<6} {'笔数':>5} {'盈利笔':>5} {'胜率%':>6} {'利润':>14}")
    print("  " + "-" * 40)
    for y in sorted(yearly.keys()):
        v = yearly[y]
        wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
        print(f"  {y:<6} {v['count']:>5} {v['wins']:>5} {wr:>5.1f} {v['profit']:>13,.0f}")

    # Part4: 分卦×年度明细
    print("\n" + "=" * 100)
    print("  Part4: 分卦×年度明细 (利润)")
    print("=" * 100)
    years = sorted(yearly.keys())
    gua_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    header = f"  {'卦':<12}" + "".join(f"{y:>10}" for y in years) + f"{'合计':>12}"
    print(f"\n{header}")
    print("  " + "-" * (12 + 10 * len(years) + 12))
    for gua in gua_list:
        gua_trades = [t for t in trades if t.get('gua') == gua]
        row = f"  {gua} {GUA_NAMES[gua]:<7}"
        total = 0
        for y in years:
            yt = [t for t in gua_trades if t['buy_date'][:4] == y]
            p = sum(t['profit'] for t in yt)
            total += p
            row += f"{p:>+10,.0f}" if yt else f"{'':>10}"
        row += f"{total:>+12,.0f}"
        print(row)

    # Part5: 参数总结
    print("\n" + "=" * 100)
    print("  Part5: 最终参数总结")
    print("=" * 100)
    print(f"""
  八卦分治策略:
    按中证1000象卦分8卦，每卦独立配置选股/买入/卖出参数
    独立卦: 坤(000)排市场坤/兑+个股仅兑+双升+kun_bear
           艮(001)双升+排市场艮+个股坤/坎+bear
           离(101)双升(t>11)+排市场艮+个股仅坤+池底≤-300+bear
           兑(110)上穿20+个股坤/坎/兑+排市场兑/震+池底≤-300+dui_bear
           乾(111)上穿60+排个股离/乾+qian_bull
    震(100): 排市场艮/巽+池底≤-400+双升(t>11)+bull
    仓位: 5仓, 每日限买1笔

  回测结果:
    初始资金: {capital:,}
    终值: {stats['final_capital']:,.0f}
    收益: {stats['total_return']:.1f}%
    最大回撤: {stats['max_dd']:.1f}%
    交易: {stats['trade_count']}笔
    胜率: {stats['win_rate']:.1f}%
""")

    # === 导出 JSON ===
    eq = result['daily_equity']
    def _fmt_date(d):
        s = str(d)[:10]
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
    for e in eq:
        e['date'] = _fmt_date(e['date'])
    for t in result['trade_log']:
        for k in ('buy_date', 'sell_date'):
            if k in t:
                t[k] = _fmt_date(t[k])

    out = {
        'meta': {
            'init_capital': capital,
            'final_capital': round(stats['final_capital'], 2),
            'total_return': round(stats['total_return'], 2),
            'trade_count': stats['trade_count'],
            'label': '八卦分治',
            'win_rate': round(stats['win_rate'], 2),
            'avg_ret': round(stats['avg_ret'], 2),
            'avg_hold': round(stats['avg_hold'], 2),
            'max_dd': round(stats['max_dd'], 2),
            'max_dd_date': stats.get('max_dd_date', ''),
        },
        'daily_equity': eq,
        'trade_log': result['trade_log'],
        'signal_detail': sig.to_dict('records'),
        'yearly': yearly,
        'gua_strategy': {g: {**s, 'grades': list(s['grades'])} for g, s in GUA_STRATEGY.items()},
        'context_stats': context_stats,
        'gua_context_stats': gua_context_stats,
        'signal_context': signal_context,
    }
    out_path = os.path.join(os.path.dirname(__file__), 'data_layer', 'data', 'backtest_8gua_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"  已导出: {out_path}")
    print("=" * 100)

    return result, stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='八卦分治回测')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--capital', type=int, default=None)
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, init_capital=args.capital)
