# -*- coding: utf-8 -*-
"""
strategy.py — 乾坤三线 Backtrader 策略 (双模式版)

疯狂模式 + 常规模式联合策略, 完全对齐 backtest_capital.py:
1. 模式切换: 中证1000 trend<45 且 main_force>0 → 疯狂, 否则常规
2. 512分级: 个股年+月+日卦 → 查历史超额 → A+/A/B+/B/B-/C/D/F
3. 疯狂模式: S1全等级(排除C/F), 停滞止损(stall15+trail15%+cap30)
4. 常规模式: 仅A+, 内卦卖法(bear/bull)
5. 买入: 双升 + trend>11 + 非空仓卦 + 等级过滤
"""
import backtrader as bt
import numpy as np
from collections import defaultdict

from .config import (
    POOL_THRESHOLD, SKIP_HEXAGRAMS, INNER_SELL_METHOD,
    MAX_POSITIONS, DAILY_BUY_LIMIT, POSITION_MODE,
    TREND_FORCE_SELL, TREND_HIGH_ZONE, TREND_MID_ZONE, TREND_BUY_ABOVE,
    CRAZY_TREND_THRESHOLD, STALL_DAYS, TRAIL_PCT, TREND_CAP,
    CRAZY_ALLOWED, NORMAL_ALLOWED, YANG_GUAS,
)


def encode_yao(trend_val, speed_val, accel_val):
    """三个指标 → 三爻二进制字符串 (位置-速度-加速度)"""
    if trend_val is None or speed_val is None or accel_val is None:
        return None
    if np.isnan(trend_val) or np.isnan(speed_val) or np.isnan(accel_val):
        return None
    yao1 = 1 if trend_val >= 50 else 0     # 位置
    yao2 = 1 if speed_val > 0 else 0       # 速度
    yao3 = 1 if accel_val > 0 else 0       # 加速度
    return f"{yao1}{yao2}{yao3}"


def grade_signal(year_yy, combo_pred):
    """信号分级: 与 backtest_capital.py 完全一致

    Args:
        year_yy: '阳' 或 '阴'
        combo_pred: 512超额收益预测值 (float 或 nan)

    Returns:
        str: 等级 (A+/A/B+/B/B-/C/D/F)
    """
    if year_yy == '阳':
        if combo_pred is not None and not np.isnan(combo_pred):
            if combo_pred > 3:
                return 'C'
        return 'F'
    # 年阴
    if combo_pred is None or np.isnan(combo_pred):
        return 'B'
    if combo_pred > 3:
        return 'A+'
    elif combo_pred > 1:
        return 'A'
    elif combo_pred > 0:
        return 'B+'
    elif combo_pred > -2:
        return 'B-'
    else:
        return 'D'


class StockState:
    """每只股票的独立状态"""
    __slots__ = ['pooled', 'pool_retail', 'in_wave',
                 'mode', 'sell_method', 'running_max', 'cross_89_count',
                 'stall_count', 'price_peak']

    def __init__(self):
        self.pooled = False
        self.pool_retail = 0.0
        self.in_wave = False
        # 持仓期间的状态
        self.mode = 'normal'        # 'crazy' 或 'normal'
        self.sell_method = 'bear'   # 常规模式卖法
        self.running_max = 0.0      # 持仓期间 trend 最大值
        self.cross_89_count = 0     # 跌破89次数 (牛卖用)
        self.stall_count = 0        # 连续不创新高天数 (疯狂用)
        self.price_peak = 0.0       # 持仓期间最高价格 (疯狂用)

    def reset(self):
        """重置状态 (trend < 11 时调用)"""
        self.pooled = False
        self.pool_retail = 0.0
        self.in_wave = False
        self.mode = 'normal'
        self.sell_method = 'bear'
        self.running_max = 0.0
        self.cross_89_count = 0
        self.stall_count = 0
        self.price_peak = 0.0

    def enter_position(self, mode, sell_method='bear', initial_trend=0.0,
                       initial_price=0.0):
        """进入持仓"""
        self.in_wave = True
        self.mode = mode
        self.sell_method = sell_method
        self.running_max = initial_trend
        self.cross_89_count = 0
        self.stall_count = 0
        self.price_peak = initial_price

    def exit_position(self):
        """退出持仓 (卖出后, 但不重置入池状态 — 等 trend<11 再重置)"""
        self.in_wave = True   # 保持 in_wave=True 防止重复买入
        self.running_max = 0.0
        self.cross_89_count = 0
        self.stall_count = 0
        self.price_peak = 0.0


class QKSXStrategy(bt.Strategy):
    """
    乾坤三线策略 (双模式版)

    datas[0] = 中证1000 (ZZ1000Data, 哨兵, 不交易)
    datas[1:] = 个股 (StockData, 交易标的)
    """
    params = (
        ('pool_threshold', POOL_THRESHOLD),
        ('max_positions', MAX_POSITIONS),
        ('daily_buy_limit', DAILY_BUY_LIMIT),
        ('position_mode', POSITION_MODE),
        ('verbose', False),
        ('snapshots_512', None),    # dict: {date_str: {combo: mean_excess_ret}}
    )

    def __init__(self):
        # 中证1000 哨兵
        self.zz = self.datas[0]
        # 个股列表
        self.stocks = self.datas[1:]

        # 每只股票的状态
        self.states = {}
        for d in self.stocks:
            self.states[d._name] = StockState()

        # 挂单跟踪: order_id → data
        self.pending_orders = {}

        # 买入信息暂存 (用于构建完整交易日志)
        self._buy_info = {}

        # 交易日志 (供外部分析)
        self.trade_log = []

        # 用于计算小象卦 (5日变化 + chg1_chg1加速度)
        self.zz_trend_buf = []
        self.zz_trend_chg1_buf = []

        # 当前持仓的 data 集合
        self.holding_datas = set()

        # 每日统计
        self.daily_stats = []

        # 512快照
        self._snapshots = self.p.snapshots_512 or {}

    def prenext(self):
        self.next()

    def log(self, msg):
        if self.p.verbose:
            try:
                dt = self.zz.datetime.date(0)
                print(f"[{dt}] {msg}")
            except Exception:
                print(f"[?] {msg}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            d = order.data
            code = d._name
            state = self.states.get(code)

            if order.isbuy():
                self.holding_datas.add(d)
                self._buy_info[code] = {
                    'buy_date': str(d.datetime.date(0)),
                    'buy_price': order.executed.price,
                    'size': order.executed.size,
                    'mode': state.mode if state else 'normal',
                }
                self.log(f"买入成交 {code} @ {order.executed.price:.2f}, "
                         f"数量={order.executed.size:.0f}, "
                         f"模式={state.mode if state else '?'}")
            elif order.issell():
                pos = self.getposition(d)
                buy_info = self._buy_info.pop(code, {})
                self.trade_log.append({
                    'code': code,
                    'buy_date': buy_info.get('buy_date', ''),
                    'buy_price': buy_info.get('buy_price', 0),
                    'sell_date': str(d.datetime.date(0)),
                    'sell_price': order.executed.price,
                    'size': buy_info.get('size', 0),
                    'pnl': (order.executed.price - buy_info.get('buy_price', 0))
                           * buy_info.get('size', 0),
                    'mode': buy_info.get('mode', '?'),
                })
                if pos.size == 0:
                    self.holding_datas.discard(d)
                    self.log(f"卖出成交 {code} @ {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            d = order.data
            self.log(f"订单失败 {d._name}: {order.getstatusname()}")

        if order.ref in self.pending_orders:
            del self.pending_orders[order.ref]

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"交易完成 {trade.data._name}: 盈亏={trade.pnlcomm:.0f}")

    def _compute_hexagram(self):
        """计算当天的六爻卦 (大中小象卦体系)

        内卦 = 小象卦: 日线趋势线≥50 + 5日变化>0 + chg1_chg1>0
        外卦 = 大象卦: 直接从zz数据的year_gua列读取 (基于月线预计算)
        """
        if len(self.zz) < 1:
            return None, None, None

        try:
            trend_val = self.zz.trend[0]
        except (IndexError, AttributeError):
            return None, None, None

        if np.isnan(trend_val):
            return None, None, None

        self.zz_trend_buf.append(trend_val)

        if len(self.zz_trend_buf) > 6:
            self.zz_trend_buf = self.zz_trend_buf[-6:]

        # 需要至少6个值: 当前 + 5天前 → chg5, 以及 chg1_chg1 需要再多1天
        if len(self.zz_trend_buf) < 6:
            return None, None, None

        # 速度: 5日变化
        chg5 = self.zz_trend_buf[-1] - self.zz_trend_buf[-6]
        # 加速度: chg1的chg1
        chg1_now = self.zz_trend_buf[-1] - self.zz_trend_buf[-2]
        chg1_prev = self.zz_trend_buf[-2] - self.zz_trend_buf[-3]
        accel = chg1_now - chg1_prev

        # 内卦: 小象卦
        inner = encode_yao(trend_val, chg5, accel)

        # 外卦: 直接读取预计算的大象卦 (基于月线数据)
        try:
            yg_val = int(self.zz.year_gua[0])
            if yg_val < 0:
                outer = None
            else:
                outer = str(yg_val).zfill(3)
                if len(outer) != 3:
                    outer = None
        except (IndexError, AttributeError, ValueError):
            outer = None

        if inner is None or outer is None:
            return None, None, None

        hex_code = inner + outer
        return inner, outer, hex_code

    def _is_crazy_mode(self):
        """判断当前是否疯狂模式: 中证1000 trend<45 且 main_force>0"""
        if len(self.zz) < 1:
            return False
        try:
            zz_trend = self.zz.trend[0]
            zz_mf = self.zz.main_force[0]
        except (IndexError, AttributeError):
            return False
        if np.isnan(zz_trend) or np.isnan(zz_mf):
            return False
        return zz_trend < CRAZY_TREND_THRESHOLD and zz_mf > 0

    def _get_512_grade(self, d):
        """获取个股当天的512分级

        Args:
            d: data feed (个股)

        Returns:
            str: 等级 (A+/A/B+/B/B-/C/D/F)
        """
        try:
            year_gua = str(int(d.year_gua[0])).zfill(3)
            month_gua = str(int(d.month_gua[0])).zfill(3)
            day_gua = str(int(d.day_gua[0])).zfill(3)
        except (IndexError, AttributeError, ValueError):
            return 'B'  # 无数据默认B

        # 年阴阳
        year_yy = '阳' if year_gua in YANG_GUAS else '阴'

        # 查512快照
        try:
            dt_str = str(self.zz.datetime.date(0))
        except Exception:
            dt_str = ''

        snap = self._snapshots.get(dt_str, {})
        combo = f"{year_gua}_{month_gua}_{day_gua}"
        combo_pred = snap.get(combo, np.nan)

        return grade_signal(year_yy, combo_pred)

    def _check_sell_crazy(self, d, state):
        """疯狂模式卖出检查: 停滞止损

        条件:
        1. 从最高价回撤 >= 15% → 卖出
        2. 连续15天趋势线不创新高 + 趋势峰值 < 30 → 卖出
        3. 趋势线 < 11 → 强制卖出
        """
        if len(d) < 2:
            return False

        try:
            trend_now = d.trend[0]
            close_now = d.close[0]
        except (IndexError, AttributeError):
            return False

        if np.isnan(trend_now):
            return False

        # 条件3: trend < 11 强制卖出
        if trend_now < TREND_FORCE_SELL:
            self.log(f"  {d._name}: [疯狂]强制卖出 trend={trend_now:.1f}<{TREND_FORCE_SELL}")
            return True

        # 更新价格最高值
        if not np.isnan(close_now):
            state.price_peak = max(state.price_peak, close_now)

        # 条件1: 从最高价回撤 >= trail_pct
        if state.price_peak > 0 and not np.isnan(close_now):
            dd = (state.price_peak - close_now) / state.price_peak * 100
            if dd >= TRAIL_PCT:
                self.log(f"  {d._name}: [疯狂]回撤止损 dd={dd:.1f}%>={TRAIL_PCT}%")
                return True

        # 条件2: 停滞止损
        if not np.isnan(trend_now):
            if trend_now > state.running_max:
                state.running_max = trend_now
                state.stall_count = 0
            else:
                state.stall_count += 1
                if state.stall_count >= STALL_DAYS and state.running_max < TREND_CAP:
                    self.log(f"  {d._name}: [疯狂]停滞止损 "
                             f"stall={state.stall_count}天, peak={state.running_max:.1f}")
                    return True

        return False

    def _check_sell_normal(self, d, state):
        """常规模式卖出检查: 内卦 bear/bull 卖法"""
        if len(d) < 2:
            return False

        try:
            trend_now = d.trend[0]
            trend_prev = d.trend[-1]
            retail_now = d.retail[0]
            retail_prev = d.retail[-1]
        except (IndexError, AttributeError):
            return False

        if np.isnan(trend_now):
            return False

        # 条件1: trend < 11 强制卖出
        if trend_now < TREND_FORCE_SELL:
            self.log(f"  {d._name}: [常规]强制卖出 trend={trend_now:.1f}<{TREND_FORCE_SELL}")
            return True

        # 更新 running_max
        state.running_max = max(state.running_max, trend_now)

        if np.isnan(trend_prev):
            return False

        if state.sell_method == 'bear':
            # 熊卖: 趋势线 < 50 不卖, 继续持有
            if trend_now < TREND_MID_ZONE:
                return False

            # running_max >= 89 且 trend 跌破 89
            if state.running_max >= TREND_HIGH_ZONE and trend_now < TREND_HIGH_ZONE:
                self.log(f"  {d._name}: [常规]熊卖-跌破89 trend={trend_now:.1f}")
                return True

            # running_max 在 50~89, 双降
            if TREND_MID_ZONE <= state.running_max < TREND_HIGH_ZONE:
                if not np.isnan(retail_now) and not np.isnan(retail_prev):
                    if trend_now < trend_prev and retail_now < retail_prev:
                        self.log(f"  {d._name}: [常规]熊卖-双降 "
                                 f"trend={trend_prev:.1f}->{trend_now:.1f}")
                        return True

        elif state.sell_method == 'bull':
            # 牛卖: running_max >= 89, 第2次跌破89才卖
            if state.running_max >= TREND_HIGH_ZONE:
                if trend_now < TREND_HIGH_ZONE and trend_prev >= TREND_HIGH_ZONE:
                    state.cross_89_count += 1
                    self.log(f"  {d._name}: [常规]牛卖-跌破89 "
                             f"第{state.cross_89_count}次")
                    if state.cross_89_count >= 2:
                        return True

        return False

    def next(self):
        """每个 bar 执行一次"""
        # 1. 计算六爻卦 + 判断模式
        inner, outer, hex_code = self._compute_hexagram()
        is_skip = hex_code in SKIP_HEXAGRAMS if hex_code else False
        sell_method = INNER_SELL_METHOD.get(inner, 'bear') if inner else 'bear'
        is_crazy = self._is_crazy_mode()

        # 当前模式的参数
        allowed_grades = CRAZY_ALLOWED if is_crazy else NORMAL_ALLOWED

        # 2. 检查卖出 (先卖后买, 释放资金和仓位)
        for d in list(self.holding_datas):
            code = d._name
            state = self.states[code]
            pos = self.getposition(d)
            if pos.size <= 0:
                self.holding_datas.discard(d)
                continue

            if len(d) < 2:
                continue

            # 根据持仓时的模式决定卖法 (不是当前模式)
            if state.mode == 'crazy':
                should_sell = self._check_sell_crazy(d, state)
            else:
                should_sell = self._check_sell_normal(d, state)

            if should_sell:
                self.close(data=d)
                state.exit_position()

        # 3. 遍历未持仓股票: 更新状态 + 收集买入候选
        buy_candidates = []

        for d in self.stocks:
            code = d._name
            state = self.states[code]

            if d in self.holding_datas:
                continue

            if len(d) < 2:
                continue

            try:
                trend_now = d.trend[0]
                trend_prev = d.trend[-1]
                retail_now = d.retail[0]
                retail_prev = d.retail[-1]
            except (IndexError, AttributeError):
                continue

            # --- 状态机 ---

            # in_wave 状态: 等待 trend < 11 重置
            if state.in_wave:
                if not np.isnan(trend_now) and trend_now < TREND_FORCE_SELL:
                    state.reset()
                continue

            # 未入池: 检查是否达到入池条件
            if not state.pooled:
                if not np.isnan(retail_now) and retail_now < self.p.pool_threshold:
                    state.pooled = True
                    state.pool_retail = retail_now
                if state.pooled and not np.isnan(retail_now):
                    state.pool_retail = min(state.pool_retail, retail_now)
                continue

            # 已入池: 更新 pool_retail 最低值
            if not np.isnan(retail_now):
                state.pool_retail = min(state.pool_retail, retail_now)

            # 检查买入条件
            if np.isnan(trend_now) or np.isnan(trend_prev):
                continue
            if np.isnan(retail_now) or np.isnan(retail_prev):
                continue

            # 双升 + trend > 11
            if (retail_now > retail_prev and
                trend_now > trend_prev and
                trend_now > TREND_BUY_ABOVE):

                # 空仓卦过滤 — 仍需设置 in_wave=True (与原引擎一致)
                if is_skip:
                    state.in_wave = True
                    continue

                # 512分级
                g = self._get_512_grade(d)

                # 等级过滤
                if g not in allowed_grades:
                    state.in_wave = True
                    continue

                buy_candidates.append({
                    'data': d,
                    'code': code,
                    'pool_retail': state.pool_retail,
                    'sell_method': sell_method,
                    'trend_now': trend_now,
                    'grade': g,
                    'mode': 'crazy' if is_crazy else 'normal',
                })

        # 4. 排序 + 资金分配 + 下单
        if buy_candidates:
            # 所有满足条件的股票设置 in_wave=True (无论是否实际买入)
            for c in buy_candidates:
                self.states[c['code']].in_wave = True

            # 按 pool_retail 从低到高排序
            buy_candidates.sort(key=lambda x: x['pool_retail'])

            current_holding = len(self.holding_datas)
            slots = self.p.max_positions - current_holding
            can_buy = min(slots, self.p.daily_buy_limit, len(buy_candidates))

            if can_buy > 0:
                cash = self.broker.getcash()
                if cash > 1000:
                    if self.p.position_mode == 'equal':
                        total_value = self.broker.getvalue()
                        per_slot = total_value / self.p.max_positions
                        per_buy = min(per_slot, cash / can_buy)
                    else:
                        per_buy = cash / can_buy

                    for i in range(can_buy):
                        c = buy_candidates[i]
                        d = c['data']
                        code = c['code']
                        state = self.states[code]
                        mode = c['mode']

                        cost = min(per_buy, self.broker.getcash())
                        if cost < 1000:
                            break

                        price = d.close[0]
                        if price <= 0 or np.isnan(price):
                            state.enter_position(
                                mode, c['sell_method'], c['trend_now'], price)
                            continue

                        size = int(cost / price / 100) * 100
                        if size <= 0:
                            state.enter_position(
                                mode, c['sell_method'], c['trend_now'], price)
                            continue

                        order = self.buy(data=d, size=size)
                        if order:
                            self.pending_orders[order.ref] = d
                            state.enter_position(
                                mode, c['sell_method'], c['trend_now'], price)
                            self.log(f"买入下单 {code}: "
                                     f"size={size}, 预估价={price:.2f}, "
                                     f"pool_retail={c['pool_retail']:.0f}, "
                                     f"grade={c['grade']}, mode={mode}, "
                                     f"sell_method={c['sell_method']}")

        # 5. 记录每日统计
        if len(self.zz) > 0:
            try:
                dt = self.zz.datetime.date(0)
                self.daily_stats.append({
                    'date': str(dt),
                    'cash': self.broker.getcash(),
                    'value': self.broker.getvalue(),
                    'n_positions': len(self.holding_datas),
                    'hex_code': hex_code,
                    'is_skip': is_skip,
                    'is_crazy': is_crazy,
                })
            except Exception:
                pass

    def stop(self):
        """回测结束时调用"""
        final_value = self.broker.getvalue()
        init_cash = self.broker.startingcash
        ret = (final_value / init_cash - 1) * 100

        # 统计疯狂/常规交易
        crazy_trades = [t for t in self.trade_log if t.get('mode') == 'crazy']
        normal_trades = [t for t in self.trade_log if t.get('mode') == 'normal']

        print(f"\n{'='*60}")
        print(f"回测结束:")
        print(f"  初始资金: {init_cash:,.0f}")
        print(f"  最终资产: {final_value:,.0f}")
        print(f"  总收益率: {ret:+.1f}%")
        print(f"  交易笔数: {len(self.trade_log)} "
              f"(疯狂={len(crazy_trades)}, 常规={len(normal_trades)})")
        print(f"{'='*60}")
