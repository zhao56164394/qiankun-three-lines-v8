# -*- coding: utf-8 -*-
"""
八卦分治回测 — 日卦分治架构 (v8.3)

核心逻辑:
  分治变量: 市场日卦 d_gua (multi_scale_gua_daily.csv, v10 三爻带滞后带)
           每天按 d_gua 查表获取该卦的策略参数(选股/买入/卖出)
  相比老基线 (天卦 tian_gua 分治): 2014-2026 回测 -3.3% → +346.5%, MDD 66.9% → 52.4%
  固化于 v8.3. 老基线 tian_gua 分支已移除 (见 git v8.2 历史).

策略参数来源: optimize_8gua.py 寻优结果 + 人工调整(避免小样本过拟合)

调整原则:
  1. 优先选 N≥20 的方案
  2. 独立交易卦(坤/兑/离/艮/乾/震) → 各自按专项配置运行
  3. 艮/离 是盈利核心(占总信号80%+)，用稳健参数

备注:
  - 代码里变量名 tian_gua_map / 字段 tian_gua 保留 (历史命名, 内容为 d_gua)
  - 个股层过滤仍用 地卦 (di_gua) + 人卦 (ren_gua), 这部分未改
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
    load_zz1000, load_zz1000_full, load_stocks,
    calc_sell_bear, calc_sell_bull, calc_sell_stall,
    calc_sell_trailing, calc_sell_trend_break,
    YEAR_START, YEAR_END, INIT_CAPITAL, DATA_DIR,
    load_big_cycle_context, summarize_signal_context,
)
from data_layer.foundation_data import load_stock_bagua_map, load_daily_bagua
from data_layer.gua_data import clean_gua as _clean_gua, GUA_DISPLAY_NAMES as GUA_NAMES


def _load_stock_main_force():
    """从 stocks.parquet 读取 main_force 列（load_stocks 不含此列）。
    Parquet 缺失时 fallback 到 5102 个 CSV 循环。"""
    pq_path = os.path.join(DATA_DIR, 'stocks.parquet')
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path, columns=['code', 'date', 'main_force'])
        df['code'] = df['code'].astype(str).str.zfill(6)
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
        mf_map = {code: dict(zip(g['date'].values, g['main_force'].values))
                  for code, g in df.groupby('code', sort=False)}
        return mf_map

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

# 策略 cfg 从 strategy_configs.py 加载, 通过 env STRATEGY_VERSION 切换 (默认 test1)
# 历史: 之前 GUA_STRATEGY 直接定义在此文件, 2026-04-26 抽离便于版本并存
from strategy_configs import get_strategy as _get_strategy
GUA_STRATEGY = _get_strategy()


def _calc_pool_priority(strat, pool_retail_min, pool_days):
    """优先买区 bonus 计算.
    cfg 字段 pool_priority_tiers: list[{depth_max, days_min, days_max, bonus}]
      按列表顺序匹配第一个 (depth+days 都符合) 的 tier, 返回其 bonus.
      depth_max=None → 兜底匹配; 不匹配返回 0.
    用途: 同日多触发时排序键, bonus 大的优先成交.
    """
    if strat is None:
        return 0
    tiers = strat.get('pool_priority_tiers')
    if not tiers:
        return 0
    for tier in tiers:
        depth_max = tier.get('depth_max')
        if depth_max is not None and pool_retail_min > depth_max:
            continue
        days_min = tier.get('days_min')
        days_max = tier.get('days_max')
        if pool_days is not None:
            if days_min is not None and pool_days < days_min:
                continue
            if days_max is not None and pool_days > days_max:
                continue
        return tier.get('bonus', 0)
    return 0


# ============================================================
# 信号扫描 — 按卦分卖法
# ============================================================
def scan_signals_8gua(stock_data, zz1000, tian_gua_map, stk_mf_map=None, big_cycle_context=None, stock_bagua_map=None, daily_bagua_map=None, gate_map=None, stock_gate_map=None):
    """扫描买入信号, 按市场日卦 (d_gua) 确定卖法和过滤参数
    参数 tian_gua_map 是历史命名, 实际传入的是 d_gua 的映射表.
    gate_map: dict[date_str -> (m_gua, y_gua)], 月/年卦激活开关数据源.
              按 GUA_STRATEGY[d_gua] 的 gate_disable_y_gua / gate_disable_m_gua 决定关火.
    """

    def _pool_days_ok(strat, idx, start_idx):
        pd_min = strat.get('pool_days_min')
        pd_max = strat.get('pool_days_max')
        if start_idx is None or (pd_min is None and pd_max is None):
            return True
        pd = idx - start_idx
        return (pd_min is None or pd >= pd_min) and (pd_max is None or pd <= pd_max)

    def _calc_pool_days(idx, start_idx):
        return idx - start_idx if start_idx is not None else None

    def _pool_depth_tier_ok(strat, pool_retail_min, pool_days, current_y_gua=None):
        """按池深分档的池深+池天联合验证。
        tiers = [{'depth_max': -500, 'days_min': None, 'days_max': None, 'days_exclude': None}, ...]
          按从深到浅顺序: 第一个 pool_retail_min <= depth_max 的档位生效；
          生效档位验 days_min/days_max, 再验 days_exclude (可选的禁止区间 [a,b])。
          设 depth_max=None 表示此档位无深度下限 (兜底)。
        返回 (ok: bool, reject_reason: str|None)
          reason 仅 ok=False 时有值: 'pool_depth' 或 'pool_days'
        若 strat 无 tiers, 回退旧逻辑 (pool_depth + pool_days_min/max 独立)。

        条件激活 (pool_depth_tiers_only_y_gua):
          若 strat 有此字段 (set of y_gua), 仅当 current_y_gua 在该 set 中时才执行 tier 检查;
          否则视为 ok=True (等价于裸跑该卦池深规则). 用于 y_gua 条件化 v1 池深规律.
        """
        only_y = strat.get('pool_depth_tiers_only_y_gua')
        if only_y and current_y_gua not in only_y:
            return True, None
        tiers = strat.get('pool_depth_tiers')
        if not tiers:
            # 旧逻辑兼容
            pd_t = strat.get('pool_depth')
            if pd_t is not None and pool_retail_min > pd_t:
                return False, 'pool_depth'
            pd_min = strat.get('pool_days_min')
            pd_max = strat.get('pool_days_max')
            if pool_days is not None:
                if pd_min is not None and pool_days < pd_min:
                    return False, 'pool_days'
                if pd_max is not None and pool_days > pd_max:
                    return False, 'pool_days'
            return True, None
        for tier in tiers:
            depth_max = tier.get('depth_max')
            if depth_max is None or pool_retail_min <= depth_max:
                days_min = tier.get('days_min')
                days_max = tier.get('days_max')
                days_exclude = tier.get('days_exclude')  # [a, b] 禁止池天在 [a,b] 之间
                if pool_days is not None:
                    if days_min is not None and pool_days < days_min:
                        return False, 'pool_days'
                    if days_max is not None and pool_days > days_max:
                        return False, 'pool_days'
                    if days_exclude is not None:
                        ex_min, ex_max = days_exclude
                        if ex_min <= pool_days <= ex_max:
                            return False, 'pool_days'
                return True, None
        return False, 'pool_depth'  # 没匹配任何档位 → 池太浅

    sell_fns = {
        'bear': lambda sd, i: calc_sell_bear(sd, i),
        'bull': lambda sd, i: calc_sell_bull(sd, i),
        'stall': lambda sd, i: calc_sell_stall(sd, i, stall_days=15, trail_pct=15, trend_cap=30),
        'trail': lambda sd, i: calc_sell_trailing(sd, i, trail_pct=15),
        'trend_break70': lambda sd, i: calc_sell_trend_break(sd, i, trend_floor=70),
    }

    # y_gua 主分治: 不再按 d_gua 分支取 strat, 改为运行时按 _today_y_gua 取
    # GUA_STRATEGY 字典 key 是 y_gua, 不是 d_gua

    # 预计算 from_map: 每个日期对应的 last_from y_gua (上次切换前的 y_gua)
    # 用于 change_type_skip (变爻 from→to 路由)
    _from_map = {}
    if gate_map:
        _last_y = None
        _last_from = None
        for _d in sorted(gate_map.keys()):
            _y = gate_map[_d][1]
            if _last_y is not None and _y != _last_y:
                _last_from = _last_y
            _from_map[_d] = _last_from
            _last_y = _y

    all_signals = []
    filter_stats = {'gua_inactive': 0, 'zz_trend': 0, 'di_gua': 0, 'stk_mf': 0,
                     'pool_depth': 0,
                     'qian_ren_gua': 0, 'qian_d': 0,
                     'dui_gua': 0, 'dui_trend': 0, 'dui_ren_gua': 0,
                     'gen_gua': 0, 'gen_ren_gua': 0,
                     'xun_gua': 0,
                     'zhen_gua': 0, 'zhen_ren_gua': 0, 'zhen_pool_days': 0, 'pool_days': 0,
                     'li_gua': 0, 'li_ren_gua': 0,
                     'kun_gua': 0, 'kun_ren_gua': 0,
                     'gate_y_gua': 0, 'gate_m_gua': 0,
                     'change_type': 0}

    for code, df in stock_data.items():
        if len(df) < 35:
            continue
        dates = df['date'].values
        trend = df['trend'].values
        retail = df['retail'].values
        closes = df['close'].values
        opens = df['open'].values

        # 外提: code 在内循环里不变, 预先 zfill (~10M 次 zfill -> 1 次)
        code_str = str(code).zfill(6)
        # 预先把 dates 转 list-of-str (避免每次内循环 str(dates[i]))
        dates_str = dates.astype(str).tolist() if hasattr(dates, 'astype') else [str(d) for d in dates]

        mf_dict = stk_mf_map.get(code, {}) if stk_mf_map else {}

        # === 共享池状态(8卦共用) ===
        pooled = False; pool_retail = 0
        pool_start_idx = None

        for i in range(1, len(df)):
            dt_str = dates_str[i]
            # 当天年卦, 用于 pool_depth_tiers_only_y_gua 条件激活
            _today_y_gua = gate_map.get(dt_str, ('???', '???'))[1] if gate_map else None
            stock_ctx = stock_bagua_map.get((dt_str, code_str), {}) if stock_bagua_map else {}
            di_gua = _clean_gua(stock_ctx.get('di_gua', ''))
            if di_gua == '???':
                continue
            di_gua_name = stock_ctx.get('di_gua_name', '')
            tian_info = tian_gua_map.get(dt_str)
            if tian_info is None:
                tian_gua, tian_gua_name = '???', ''
            elif isinstance(tian_info, tuple):
                tian_gua, tian_gua_name = tian_info[0], tian_info[1]
            else:
                tian_gua, tian_gua_name = tian_info, ''
            ren_gua_ctx = daily_bagua_map.get((dt_str, code_str), {}) if daily_bagua_map else {}
            ren_gua = _clean_gua(ren_gua_ctx.get('gua_code', ''))
            ren_gua_name = ren_gua_ctx.get('gua_name', '')
            context = big_cycle_context.get(dt_str, {}) if big_cycle_context else {}

            # ============================================================
            # 共享池: 入池 → 各卦分支的 pool_depth 做二次验证
            # 入池阈值全局统一 (UNIFIED_POOL_THRESHOLD = -250)
            # ============================================================
            if not pooled:
                if not np.isnan(retail[i]) and retail[i] < UNIFIED_POOL_THRESHOLD:
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
            # Gate: 年/月卦激活开关 (按 d_gua 分支独立配置)
            # 三层粒度 (按从粗到细顺序检查):
            #   gate_disable_y_gua: 关掉整年 (set of y_gua) — 粗粒度, 最简
            #   gate_disable_m_gua: 关掉整月 (set of m_gua) — 中粒度
            #   gate_disable_ym:    关掉 (y_gua, m_gua) 联合 cell — 精细粒度
            # 仅控制信号触发, 不影响入池
            # ============================================================
            if gate_map is not None and tian_gua in GUA_STRATEGY:
                _strat_gate = GUA_STRATEGY[tian_gua]
                _gate_y = _strat_gate.get('gate_disable_y_gua') or ()
                _gate_m = _strat_gate.get('gate_disable_m_gua') or ()
                _gate_ym = _strat_gate.get('gate_disable_ym') or ()
                if _gate_y or _gate_m or _gate_ym:
                    _today_m, _today_y = gate_map.get(dt_str, ('???', '???'))
                    if _gate_y and _today_y in _gate_y:
                        filter_stats['gate_y_gua'] += 1
                        continue
                    if _gate_m and _today_m in _gate_m:
                        filter_stats['gate_m_gua'] += 1
                        continue
                    if _gate_ym and (_today_y, _today_m) in _gate_ym:
                        filter_stats.setdefault('gate_ym', 0)
                        filter_stats['gate_ym'] += 1
                        continue

            # ============================================================
            # Stock Gate (test2): 个股年/月卦激活开关
            # 个股 m_gua/y_gua 由 stock_multi_scale_gua_daily 提供 (逐股校准)
            # 没数据 (NaN) → 默认不买 (skip 该信号)
            # ============================================================
            if stock_gate_map is not None and tian_gua in GUA_STRATEGY:
                _strat_sgate = GUA_STRATEGY[tian_gua]
                _sgate_y = _strat_sgate.get('stock_gate_disable_y_gua') or ()
                _sgate_m = _strat_sgate.get('stock_gate_disable_m_gua') or ()
                _sgate_ym = _strat_sgate.get('stock_gate_disable_ym') or ()
                # 也需要 stock_gate_require_data: 如果设了任何一个 stock gate 字段, 没数据默认 skip
                _has_sgate = bool(_sgate_y or _sgate_m or _sgate_ym
                                  or _strat_sgate.get('stock_gate_require_data', False))
                if _has_sgate:
                    _stk_my = stock_gate_map.get((dt_str, code_str))
                    if _stk_my is None:
                        # 没个股 m/y_gua 数据 → 默认 skip (用户决策: 没数据不买)
                        filter_stats.setdefault('stock_gate_no_data', 0)
                        filter_stats['stock_gate_no_data'] += 1
                        continue
                    _stk_m, _stk_y = _stk_my
                    if _sgate_y and _stk_y in _sgate_y:
                        filter_stats.setdefault('stock_gate_y_gua', 0)
                        filter_stats['stock_gate_y_gua'] += 1
                        continue
                    if _sgate_m and _stk_m in _sgate_m:
                        filter_stats.setdefault('stock_gate_m_gua', 0)
                        filter_stats['stock_gate_m_gua'] += 1
                        continue
                    if _sgate_ym and (_stk_y, _stk_m) in _sgate_ym:
                        filter_stats.setdefault('stock_gate_ym', 0)
                        filter_stats['stock_gate_ym'] += 1
                        continue

            # ============================================================
            # y_gua 主分治: 按当日大盘 y_gua 取 strat → 统一 double_rise 逻辑
            # 起步: 8 桶都用 kun 模板 (double_rise 触发, bear 卖出)
            # ============================================================
            strat = GUA_STRATEGY.get(_today_y_gua) if _today_y_gua else None
            if strat is None or not strat.get('active', True):
                filter_stats['gua_inactive'] += 1
                continue
            if not strat.get('buy', True):
                # 整桶 skip (用于以后某桶整桶不买)
                continue

            # 变爻 skip (test12 yao 框架): 命中 change_type_skip 集合则 skip 该信号
            _skip_ct = strat.get('change_type_skip')
            if _skip_ct:
                _today_from = _from_map.get(dt_str)
                if _today_from is not None:
                    _ct = f'{_today_from}->{_today_y_gua}'
                    if _ct in _skip_ct:
                        filter_stats['change_type'] += 1
                        continue

            # 双升触发 (与 test4 共享分支一致)
            if np.isnan(trend[i]) or np.isnan(trend[i-1]):
                continue
            if np.isnan(retail[i]) or np.isnan(retail[i-1]):
                continue

            buy_mode = strat.get('buy_mode', 'double_rise')
            cross_threshold = strat.get('cross_threshold', 20)
            is_trigger = False
            if buy_mode == 'cross':
                is_trigger = (trend[i-1] < cross_threshold and trend[i] >= cross_threshold)
            else:  # double_rise
                is_trigger = (retail[i] > retail[i-1] and trend[i] > trend[i-1] and trend[i] > 11)

            if not is_trigger:
                continue

            # 过滤: 排市场卦 (ren_gua) / 个股卦白名单 (di_gua)
            exclude_ren = strat.get('exclude_ren_gua') or set()
            if ren_gua in exclude_ren:
                filter_stats.setdefault('y_gua_ren', 0)
                filter_stats['y_gua_ren'] += 1
                pooled = False; pool_retail = 0; pool_start_idx = None
                continue
            allow_di = strat.get('allow_di_gua')
            if allow_di and di_gua not in allow_di:
                filter_stats.setdefault('y_gua_di', 0)
                filter_stats['y_gua_di'] += 1
                pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # 池深 + 池天 (tier-aware)
            _ok, _reason = _pool_depth_tier_ok(strat, pool_retail, _calc_pool_days(i, pool_start_idx), _today_y_gua)
            if not _ok:
                filter_stats[_reason] += 1
                pooled = False; pool_retail = 0; pool_start_idx = None
                continue

            # 卖出 + 记录
            sell_method = strat.get('sell', 'bear')
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
                'tian_gua': tian_gua,
                'sell_method': sell_method,
                'ren_gua': ren_gua,
                'ren_gua_name': ren_gua_name,
                'tian_gua_name': tian_gua_name,
                'y_gua': _today_y_gua,
            })
            pooled = False; pool_retail = 0; pool_start_idx = None


    print(f"  过滤统计: 空仓卦={filter_stats['gua_inactive']}, "
          f"中证趋势={filter_stats['zz_trend']}, "
          f"池底深度={filter_stats['pool_depth']}, "
          f"象卦拒绝={filter_stats['di_gua']}, "
          f"主力线={filter_stats['stk_mf']}")
    print(f"  y_gua 路由过滤: 排市场卦={filter_stats.get('y_gua_ren', 0)}, "
          f"个股卦白名单外={filter_stats.get('y_gua_di', 0)}, "
          f"池天={filter_stats.get('pool_days', 0)}, "
          f"变爻={filter_stats.get('change_type', 0)}")
    return pd.DataFrame(all_signals).sort_values('signal_date').reset_index(drop=True)


# ============================================================
# 模拟引擎 — 八卦分治
# ============================================================
def simulate_8gua(sig_df, zz_df, max_pos=5, daily_limit=1, init_capital=None, tian_gua_map_ext=None):
    """八卦分治模拟引擎"""
    capital = init_capital or INIT_CAPITAL
    zz_indexed = zz_df.set_index('date')

    # 分治卦映射 (市场日卦 d_gua)：优先外部传入，否则从信号中提取
    if tian_gua_map_ext:
        tian_gua_map = {}
        for k, v in tian_gua_map_ext.items():
            tian_gua_map[k] = v[0] if isinstance(v, tuple) else v
    else:
        # 向量化避免 iterrows
        _dates = zz_df['date'].astype(str).values
        _guas = zz_df['gua'].astype(str).map(_clean_gua).values
        tian_gua_map = {_dates[i]: _guas[i] for i in range(len(zz_df))}

    # 把 sig_df 转为 records (dict) 而非 Series, 下游访问比 row['col'] 快 ~10x
    sig_by_date = {}
    sig_records = sig_df.to_dict('records')
    for r in sig_records:
        sig_by_date.setdefault(r['signal_date'], []).append(r)

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
                    'hold_days': pos['hold_days'],
                    'gua': pos.get('gua', '???'),
                    'di_gua': pos.get('di_gua', pos.get('combo', '???')),
                    'di_gua_name': pos.get('di_gua_name', ''),
                    'sell_method': pos.get('sell_method', '?'),
                    'tian_gua': pos.get('tian_gua', ''),
                    'tian_gua_name': pos.get('tian_gua_name', ''),
                    'ren_gua': pos.get('ren_gua', ''),
                    'ren_gua_name': pos.get('ren_gua_name', ''),
                    'y_gua': pos.get('y_gua', '???'),
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
            # 信号已在 scan_signals 阶段按卦策略完整过滤，这里仅排除 rank_order=1 与 skip
            filtered = []
            for c in candidates:
                rank_order = int(0 if pd.isna(c.get('rank_order')) else c.get('rank_order', 0))
                if rank_order > 0 and rank_order <= 1:
                    continue
                if c.get('is_skip', False):
                    continue
                filtered.append(c)
            def _cand_pri(c):
                cgua = str(c.get('tian_gua', '')).zfill(3) if c.get('tian_gua') else ''
                return _calc_pool_priority(GUA_STRATEGY.get(cgua),
                                           c.get('pool_retail', 0), c.get('pool_days'))
            filtered.sort(key=lambda x: (
                -_cand_pri(x),
                -int(0 if pd.isna(x.get('rank_order')) else x.get('rank_order', 0)),
                x['pool_retail']
            ))
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
                        'hold_days': c['hold_days'],
                        'gua': tian_gua,
                        'combo': c.get('combo', '???'),
                        'di_gua': c.get('di_gua', c.get('combo', '???')),
                        'di_gua_name': c.get('di_gua_name', ''),
                        'sell_method': c.get('sell_method', '?'),
                        'tian_gua': c.get('tian_gua', ''),
                        'tian_gua_name': c.get('tian_gua_name', ''),
                        'ren_gua': c.get('ren_gua', ''),
                        'ren_gua_name': c.get('ren_gua_name', ''),
                        'y_gua': c.get('y_gua', '???'),
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
            'hold_days': pos['hold_days'],
            'gua': pos.get('gua', '???'),
            'sell_method': pos.get('sell_method', '?'),
            'tian_gua': pos.get('tian_gua', ''),
            'tian_gua_name': pos.get('tian_gua_name', ''),
            'y_gua': pos.get('y_gua', '???'),
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
    print(f"  {'卦':<15} {'状态':<6} {'卖法':<12} {'入池≤':>6} {'池底≤':>6} {'附加'}")
    print("  " + "-" * 100)
    sell_labels = {'bear': 'bear(保守)', 'bull': 'bull(耐心)',
                   'stall': 'stall(停滞)', 'trail': 'trail(止损)',
                   'hybrid': 'hybrid(联合)', 'qian_bull': 'qian_bull(纯牛)',
                   'kun_bear': 'kun_bear(反转)', 'dui_bear': 'dui_bear(快出)',
                   'trend_break70': 'trend70(跌破)'}
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        s = GUA_STRATEGY[gua]
        status = '交易' if s['active'] else '空仓'
        pool_thr = s.get('pool_threshold', UNIFIED_POOL_THRESHOLD)
        pool_dep = s.get('pool_depth')
        dep_str = str(pool_dep) if pool_dep is not None else 'X'
        extra = []
        if gua == '000' and s.get('kun_buy'):
            stock_allow = s.get('kun_allow_di_gua')
            stock_desc = '/'.join(GUA_NAMES.get(g, g).split('(')[0] for g in sorted(stock_allow)) if stock_allow else '不限'
            extra.append(f"坤独立:个股仅{stock_desc} 双升(t>11)")
        if gua == '100' and s.get('zhen_buy'):
            extra.append("震独立:排人卦艮/巽 双升(t>11) bull")
        if gua == '101' and s.get('li_buy'):
            extra.append("离独立:排人卦艮 个股仅坤 双升(t>11)")
        if gua == '011' and s.get('xun_allow_di_gua'):
            extra.append("巽独立:个股仅坎 双升(t>11)")
        extra_str = ' '.join(extra) if extra else ''
        print(f"  {gua} {GUA_NAMES[gua]:<10} {status:<6} {sell_labels[s['sell']]:<12} "
              f"{pool_thr:>6} {dep_str:>6} {extra_str}")

    # 1. 加载数据
    print("\n[1] 加载数据...")
    zz_df = load_zz1000_full()
    zz1000 = load_zz1000()
    stock_data = load_stocks()
    print(f"  个股: {len(stock_data)} 只")

    # 加载个股主力线(用于乾卦主力线过滤)
    stk_mf_map = _load_stock_main_force()
    print(f"  已加载 {len(stk_mf_map)} 只股票的main_force")

    # 构建分治卦映射: 市场日卦 d_gua (v10 三爻带滞后带)
    # 注: 变量名 tian_gua_map 是历史命名, 实际内容为 d_gua
    # 同时构建 gate_map: 月卦/年卦激活开关用 (按分支独立配置, 高位年/坏月份关火)
    ms_pq = os.path.join(DATA_DIR, 'foundation', 'multi_scale_gua_daily.parquet')
    ms_csv = os.path.join(DATA_DIR, 'foundation', 'multi_scale_gua_daily.csv')
    if os.path.exists(ms_pq):
        ms_df = pd.read_parquet(ms_pq)
    else:
        ms_df = pd.read_csv(ms_csv, encoding='utf-8-sig',
                            dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    ms_df['date'] = pd.to_datetime(ms_df['date']).dt.strftime('%Y-%m-%d')

    # 向量化构造 tian_gua_map / gate_map（替代 iterrows）
    # to_numpy() 强制转 numpy object 数组, 比 .values 在 pyarrow string 列上快 30x
    dates_arr = ms_df['date'].astype(str).to_numpy()
    d_arr = ms_df['d_gua'].astype(str).map(_clean_gua).to_numpy()
    m_arr = ms_df['m_gua'].astype(str).map(_clean_gua).to_numpy()
    y_arr = ms_df['y_gua'].astype(str).map(_clean_gua).to_numpy()
    tian_gua_map = {dates_arr[i]: (d_arr[i], GUA_NAMES.get(d_arr[i], ''))
                    for i in range(len(ms_df)) if d_arr[i] and d_arr[i] != '???'}
    gate_map = {dates_arr[i]: (m_arr[i], y_arr[i]) for i in range(len(ms_df))}
    print(f"  分治卦 (市场日卦 d_gua, v10): {len(tian_gua_map)} 天")
    print(f"  激活开关映射 (m_gua/y_gua): {len(gate_map)} 天")

    # 构建人卦映射(主卦/个股横截面排名) - 向量化构造避免 7.7M 次 iterrows
    # to_numpy() 强制转 numpy object 数组, 比 .values 在 pyarrow string 列上快 30x
    daily_bagua_df = load_daily_bagua()
    if 'gua_name' not in daily_bagua_df.columns:
        daily_bagua_df['gua_name'] = ''
    _dates = daily_bagua_df['date'].astype(str).to_numpy()
    _codes = daily_bagua_df['code'].astype(str).str.zfill(6).to_numpy()
    _guas = daily_bagua_df['gua_code'].astype(str).str.zfill(3).to_numpy()
    _names = daily_bagua_df['gua_name'].fillna('').astype(str).to_numpy()
    daily_bagua_map = {
        (_dates[i], _codes[i]): {'gua_code': _guas[i], 'gua_name': _names[i]}
        for i in range(len(daily_bagua_df))
    }

    # 构建个股 m/y_gua 映射 (test2 用): stock_multi_scale_gua_daily
    # 没数据的票/日 → 个股 gate 时默认 skip
    stock_gate_map = None
    smg_path = os.path.join(DATA_DIR, 'foundation', 'stock_multi_scale_gua_daily.parquet')
    if os.path.exists(smg_path):
        smg = pd.read_parquet(smg_path, columns=['date', 'code', 'm_gua', 'y_gua'])
        smg = smg.dropna(subset=['m_gua', 'y_gua'])
        smg['date'] = smg['date'].astype(str)
        smg['code'] = smg['code'].astype(str).str.zfill(6)
        smg['m_gua'] = smg['m_gua'].astype(str).str.zfill(3)
        smg['y_gua'] = smg['y_gua'].astype(str).str.zfill(3)
        # to_numpy() 强制转 numpy object 数组, 比 .values 在 pyarrow string 列上快 30x
        _d = smg['date'].to_numpy()
        _c = smg['code'].to_numpy()
        _m = smg['m_gua'].to_numpy()
        _y = smg['y_gua'].to_numpy()
        stock_gate_map = {(_d[i], _c[i]): (_m[i], _y[i]) for i in range(len(smg))}
        print(f"  个股 m/y_gua 映射 (test2 stock_gate_map): {len(stock_gate_map):,} 行")
    else:
        print(f"  [WARN] 个股 m/y_gua 数据未生成 ({smg_path}), test2 stock_gate 不可用")

    # 2. 扫描信号
    print("\n[2] 扫描八卦分治信号...")
    sig = scan_signals_8gua(stock_data, zz1000, tian_gua_map, stk_mf_map, big_cycle_context=big_cycle_context, stock_bagua_map=stock_bagua_map, daily_bagua_map=daily_bagua_map, gate_map=gate_map, stock_gate_map=stock_gate_map)
    # signal_date 是 'YYYY-MM-DD' 格式, year_start/end 若是 'YYYYMMDD' 则要转换避免 '-'<'0' 错位
    def _norm(d):
        return f'{d[:4]}-{d[4:6]}-{d[6:]}' if isinstance(d, str) and len(d) == 8 and '-' not in d else d
    sig = sig[(sig['signal_date'] >= _norm(year_start)) &
              (sig['signal_date'] < _norm(year_end))].reset_index(drop=True)

    print(f"  总信号: {len(sig)}, 非skip: {(~sig['is_skip']).sum()}")
    signal_context = summarize_signal_context(sig)
    print(f"  分治卦 (d_gua) 分布: {signal_context.get('tian_gua_counts', {})}")

    # 分卦信号统计
    print(f"\n  分卦信号分布:")
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub = sig[sig['tian_gua'] == gua]
        non_skip = sub[~sub['is_skip']]
        print(f"    {gua} {GUA_NAMES[gua]}: 总{len(sub)}, 非skip {len(non_skip)}")

    # 3. 模拟
    print("\n[3] 八卦分治模拟...")
    # max_pos / daily_limit 从 strategy_configs 按版本读取, env 仍可临时覆盖
    from strategy_configs import get_sim_params, get_version
    _sim = get_sim_params()
    _max_pos = int(os.environ.get('SIM_MAX_POS', _sim['max_pos']))
    _daily_limit = int(os.environ.get('SIM_DAILY_LIMIT', _sim['daily_limit']))
    _ver = get_version()
    print(f"  [strategy={_ver}] max_pos={_max_pos}, daily_limit={_daily_limit}")
    result = simulate_8gua(sig, zz_df, max_pos=_max_pos, daily_limit=_daily_limit,
                           init_capital=capital, tian_gua_map_ext=tian_gua_map)
    stats = calc_stats(result, '八卦分治')
    trades = result['trade_log']
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

    # Part2: 分桶贡献 (按 y_gua 大盘环境分桶)
    print("\n" + "=" * 100)
    print("  Part2: 分桶贡献 (y_gua 大盘环境)")
    print("=" * 100)
    print(f"\n  {'y_gua':<15} {'笔数':>5} {'胜率%':>6} {'均收益%':>8} {'利润':>14} {'占比%':>6}")
    print("  " + "-" * 60)
    total_profit = sum(t['profit'] for t in trades)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        gua_trades = [t for t in trades if t.get('y_gua') == gua]
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

    # Part4: 分桶 (y_gua) × 年度明细
    print("\n" + "=" * 100)
    print("  Part4: 分桶 (y_gua) × 年度明细 (利润)")
    print("=" * 100)
    years = sorted(yearly.keys())
    gua_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    header = f"  {'y_gua':<12}" + "".join(f"{y:>10}" for y in years) + f"{'合计':>12}"
    print(f"\n{header}")
    print("  " + "-" * (12 + 10 * len(years) + 12))
    for gua in gua_list:
        gua_trades = [t for t in trades if t.get('y_gua') == gua]
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
        'gua_strategy': {g: {k: (sorted(v) if isinstance(v, set) else v) for k, v in s.items()} for g, s in GUA_STRATEGY.items()},
        'gua_context_stats': gua_context_stats,
        'signal_context': signal_context,
    }
    out_path = os.path.join(os.path.dirname(__file__), 'data_layer', 'data', 'backtest_y_gua_result.json')
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
