# -*- coding: utf-8 -*-
"""
乾坤三线 v1.0 — miniQMT 主策略文件

日线级别波段策略，操作频率低，每日只需在三个时间点触发:
  09:15 盘前  → 选股 + 八卦过滤
  09:31 开盘  → 下买单
  14:50 盘尾  → 检查卖出条件 (日线级别)
  15:05 收盘  → 更新数据 + 保存日志

通过 xtquant 连接 QMT 客户端运行。

启动方式:
  方式1 (命令行): python qmt_strategy.py
  方式2 (QMT内): 将此文件挂载到 QMT 策略

依赖:
  - xtquant (miniQMT Python SDK)
  - 项目内: signal_engine, risk_manager, trade_logger, config
  - data_layer: gua_data (八卦过滤)
"""
import os
import sys
import json
import time
import traceback
import numpy as np
from datetime import datetime, timedelta

# 确保能导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from live.config import (
    QMT_ACCOUNT, QMT_ACCOUNT_TYPE, QMT_PATH,
    SELECT_TIME, BUY_START_TIME, BUY_END_TIME,
    SELL_CHECK_START, SELL_FORCE_TIME, SELL_REALTIME_CHECK,
    DATA_UPDATE_TIME, ORDER_TIMEOUT,
    MAX_POSITIONS, DAILY_BUY_LIMIT,
)
from live.signal_engine import (
    StockPool, SellTracker, StallSellTracker,
    generate_buy_signals,
    calc_hexagram, determine_sell_method, is_crazy_mode,
    load_stock_events_df, build_512_snapshot, grade_signal, to_yinyang,
    load_all_stock_latest, get_stock_data_by_date,
    load_zz1000_latest,
    calc_realtime_indicators,
)
from live.risk_manager import RiskManager
from live.trade_logger import TradeLogger

# 延迟导入 xtquant (可能未安装)
xttrader = None
xtdata = None
xtconstant = None


def import_xtquant():
    """延迟导入 xtquant，方便在无QMT环境下测试"""
    global xttrader, xtdata, xtconstant
    try:
        from xtquant import xttrader as _xt, xtdata as _xd, xtconstant as _xc
        xttrader = _xt
        xtdata = _xd
        xtconstant = _xc
        # StockAccount 在 xttype 子模块中
        xttrader.StockAccount = _xt.xttype.StockAccount
        return True
    except ImportError:
        print('  !! xtquant 未安装, 仅模拟模式可用')
        return False


class QiankunStrategy:
    """
    乾坤三线实盘策略

    核心流程:
      1. 盘前 → load_data() + scan_signals()
      2. 开盘 → execute_buys()
      3. 盘中 → monitor_sells()
      4. 收盘 → end_of_day()
    """

    def __init__(self, account=None, simulate=False, bypass_filter=False):
        """
        初始化

        Args:
            account: QMT账号, 为None则用config中的
            simulate: True=模拟模式(不真正下单), False=实盘
            bypass_filter: True=跳过年阳禁买(仅测试用)
        """
        self.account = account or QMT_ACCOUNT
        self.simulate = simulate
        self.bypass_filter = bypass_filter

        # 核心模块
        self.stock_pool = StockPool()
        self.risk_manager = RiskManager()
        self.logger = TradeLogger()

        # QMT 连接
        self.trader = None
        self.connected = False

        # 每日状态
        self.today = None               # 今日日期 YYYY-MM-DD
        self.today_signals = []         # 今日买入信号
        self.today_hex_info = None      # 今日六爻卦
        self.today_buy_filter = None    # 今日买入过滤
        self.today_mode = 'normal'      # 今日模式: 'crazy' 或 'normal'
        self.sell_trackers = {}         # 持仓卖出追踪器: {code: SellTracker/StallSellTracker}
        self.positions = {}             # 当前持仓: {code: {volume, buy_price, ...}}
        self.order_times = {}           # 委托时间: {code: timestamp}
        self.stock_events_df = None     # 512事件表 (加载一次)
        self.today_512_snapshot = None  # 今日512快照

        # 状态标志
        self.data_loaded = False
        self.signals_scanned = False
        self.buys_executed = False
        self.sells_checked = False

    # ============================================================
    # 连接 & 初始化
    # ============================================================
    def connect(self):
        """连接 miniQMT"""
        if self.simulate:
            print('  [模拟模式] 跳过QMT连接')
            self.connected = True
            return True

        if not import_xtquant():
            return False

        try:
            path = QMT_PATH
            self.trader = xttrader.XtQuantTrader(path, session=0)
            self.trader.start()

            connect_result = self.trader.connect()
            if connect_result != 0:
                print(f'  !! QMT连接失败, 错误码: {connect_result}')
                return False

            # 注册回调
            self.trader.register_callback(self._on_deal)

            # 订阅账号
            acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
            self.trader.subscribe(acc)

            self.connected = True
            print(f'  QMT连接成功, 账号: {self.account}')
            return True
        except Exception as e:
            print(f'  !! QMT连接异常: {e}')
            traceback.print_exc()
            return False

    def _on_deal(self, deal_info):
        """QMT 成交回调"""
        try:
            code = f'{deal_info.stock_code}'
            direction = '买入' if deal_info.order_type == 23 else '卖出'
            price = deal_info.traded_price
            volume = deal_info.traded_volume

            if direction == '买入':
                self.logger.log_buy(
                    code=code, date=self.today, price=price, volume=volume,
                    sell_method=self.positions.get(code, {}).get('sell_method', 'bear'),
                )
            else:
                buy_price = self.positions.get(code, {}).get('buy_price', 0)
                sell_reason = self.positions.get(code, {}).get('pending_sell_reason', '')
                self.logger.log_sell(
                    code=code, date=self.today, price=price, volume=volume,
                    sell_reason=sell_reason, buy_price=buy_price,
                )
                # 移除持仓
                self.positions.pop(code, None)
                self.sell_trackers.pop(code, None)

            print(f'  [成交回报] {direction} {code} {price:.2f}×{volume}')
        except Exception as e:
            print(f'  !! 成交回报处理异常: {e}')

    # ============================================================
    # 数据加载 (盘前)
    # ============================================================
    def load_data(self):
        """
        加载数据 (盘前09:15调用)

        从预计算的CSV加载全市场数据 + 中证1000卦象数据
        """
        self.today = datetime.now().strftime('%Y-%m-%d')
        print(f'\n{"=" * 60}')
        print(f'  乾坤三线 v1.0 | {self.today} | 盘前数据加载')
        print(f'{"=" * 60}')

        try:
            # 1. 加载全市场个股数据 (最近2天)
            print('  加载个股数据...')
            self.all_stock_data = load_all_stock_latest(n_days=5)
            print(f'  个股: {len(self.all_stock_data)} 只')

            # 2. 加载中证1000卦象数据
            print('  加载中证1000...')
            self.zz1000_data = load_zz1000_latest()
            print(f'  中证1000: {len(self.zz1000_data)} 日')

            # 3. 加载512事件表 (首次加载)
            if self.stock_events_df is None:
                print('  加载512事件表...')
                self.stock_events_df = load_stock_events_df()
                if self.stock_events_df is not None:
                    print(f'  512事件: {len(self.stock_events_df)} 条')
                else:
                    print('  !! 512事件表不存在, 分级将使用默认等级B')

            # 4. 获取今日(或最近日)的市场卦象 + 模式判断
            self._load_market_gua()

            # 4. 从日志恢复持仓
            self._restore_positions()

            # 5. 判断当前模式
            self._detect_mode()

            self.data_loaded = True
            print(f'  数据加载完成')
            return True

        except Exception as e:
            print(f'  !! 数据加载失败: {e}')
            traceback.print_exc()
            return False

    def _load_market_gua(self):
        """加载当日市场卦象"""
        try:
            from data_layer.gua_data import get_buy_filter, get_market_state

            # 找到最近的交易日
            dates_available = sorted(self.zz1000_data.keys())
            if not dates_available:
                print('  !! 无中证1000数据')
                return

            # 用最近的交易日 (可能今日数据还没有)
            latest_date = dates_available[-1]
            zz_entry = self.zz1000_data[latest_date]

            # 计算六爻卦
            self.today_hex_info = calc_hexagram(zz_entry)
            if self.today_hex_info:
                print(f'  六爻卦: {self.today_hex_info["hex_code"]} '
                      f'(内{self.today_hex_info["inner"]} 外{self.today_hex_info["outer"]})')
            else:
                print('  !! 六爻卦计算失败')

            # 买入过滤
            try:
                self.today_buy_filter = get_buy_filter(latest_date)
                can = self.today_buy_filter.get('can_buy', False)
                reason = self.today_buy_filter.get('reason', '')
                grade = self.today_buy_filter.get('grade', '?')
                print(f'  买入过滤: {"可买" if can else "禁买"} | '
                      f'评级{grade} | {reason}')
            except Exception as e:
                print(f'  !! 买入过滤加载失败: {e}')
                self.today_buy_filter = None

            # 市场状态
            try:
                state = get_market_state(latest_date)
                print(f'  市场状态: {state["overall"]}')
                print(f'    年卦: {state["year_name"]} | '
                      f'月卦: {state["month_name"]} | '
                      f'日卦: {state["day_name"]}')
            except Exception:
                pass

        except Exception as e:
            print(f'  !! 市场卦象加载异常: {e}')

    def _detect_mode(self):
        """判断当前模式: 疯狂 or 常规"""
        dates_available = sorted(self.zz1000_data.keys())
        if not dates_available:
            self.today_mode = 'normal'
            return

        latest_date = dates_available[-1]
        zz_entry = self.zz1000_data[latest_date]

        if is_crazy_mode(zz_entry):
            self.today_mode = 'crazy'
            print(f'  当前模式: 疯狂 (trend={zz_entry.get("trend")}, '
                  f'mf={zz_entry.get("main_force")})')
        else:
            self.today_mode = 'normal'
            print(f'  当前模式: 常规 (trend={zz_entry.get("trend")}, '
                  f'mf={zz_entry.get("main_force")})')

    def _restore_positions(self):
        """从日志恢复未平仓持仓"""
        open_pos = self.logger.get_open_positions_from_log()
        if open_pos:
            print(f'  恢复持仓: {len(open_pos)} 只')
            for code, info in open_pos.items():
                self.positions[code] = info
                # 重建卖出追踪器
                method = info.get('sell_method', 'bear')
                if method == 'stall':
                    self.sell_trackers[code] = StallSellTracker(
                        code, info.get('buy_price', 0))
                else:
                    self.sell_trackers[code] = SellTracker(code, method)
                print(f'    {code} | 买入{info["buy_date"]} | '
                      f'{info["buy_price"]:.2f} | {method}卖')
        else:
            print('  无未平仓持仓')

        # 如果在模拟模式且无日志持仓, 也可以从QMT查询
        if not self.simulate and self.connected:
            self._sync_positions_from_qmt()

    def _sync_positions_from_qmt(self):
        """从QMT同步实际持仓 (防止日志与实际不一致)"""
        if not self.trader:
            return
        try:
            acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
            pos_list = self.trader.query_stock_positions(acc)
            for pos in pos_list:
                if pos.volume > 0 and pos.stock_code not in self.positions:
                    print(f'  !! QMT有持仓但日志无记录: {pos.stock_code} '
                          f'{pos.volume}股, 已补充追踪')
                    self.positions[pos.stock_code] = {
                        'buy_date': '未知',
                        'buy_price': pos.open_price,
                        'volume': pos.volume,
                        'sell_method': 'bear',
                    }
                    self.sell_trackers[pos.stock_code] = SellTracker(
                        pos.stock_code, 'bear')
        except Exception as e:
            print(f'  !! QMT持仓同步异常: {e}')

    # ============================================================
    # 选股扫描 (盘前)
    # ============================================================
    def scan_signals(self):
        """
        扫描全市场买入信号 (盘前09:15调用)

        使用预计算的日线数据进行信号判断
        """
        if not self.data_loaded:
            print('  !! 数据未加载, 请先调用 load_data()')
            return []

        print(f'\n  === 信号扫描 ===')

        # 找到最近两个交易日的数据
        all_stock_data = self.all_stock_data
        if not all_stock_data:
            print('  !! 无个股数据')
            return []

        # 获取所有可用日期
        all_dates = set()
        for code, df in all_stock_data.items():
            for d in df['date'].values:
                all_dates.add(str(d))
        all_dates = sorted(all_dates)

        if len(all_dates) < 2:
            print('  !! 交易日不足2天')
            return []

        # 最近两个交易日
        today_date = all_dates[-1]
        yest_date = all_dates[-2]
        print(f'  最近两日: {yest_date} → {today_date}')

        # 提取数据
        data_today = get_stock_data_by_date(all_stock_data, today_date)
        data_yest = get_stock_data_by_date(all_stock_data, yest_date)
        print(f'  今日有数据: {len(data_today)} 只')

        # 生成买入信号
        signals = generate_buy_signals(self.stock_pool, data_today, data_yest)

        # 512分级
        if signals:
            snap = None
            if self.stock_events_df is not None:
                snap = build_512_snapshot(self.stock_events_df, today_date)
                self.today_512_snapshot = snap

            for s in signals:
                year_gua = s.get('year_gua', '')
                month_gua = s.get('month_gua', '')
                day_gua = s.get('day_gua', '')
                year_yy = to_yinyang(year_gua) if year_gua else '阴'
                combo = f'{year_gua}_{month_gua}_{day_gua}'
                combo_pred = snap.get(combo) if snap else None
                grade, grade_desc = grade_signal(year_yy, combo_pred)
                s['grade'] = grade
                s['grade_desc'] = grade_desc
                s['year_yy'] = year_yy
                s['combo'] = combo

            # 按等级过滤
            before_count = len(signals)
            signals = self.risk_manager.filter_by_grade(signals, self.today_mode == 'crazy')
            print(f'  分级过滤: {before_count} → {len(signals)} 只 '
                  f'(模式: {"疯狂" if self.today_mode == "crazy" else "常规"})')

        self.today_signals = signals

        if signals:
            print(f'  买入信号: {len(signals)} 只')
            for i, s in enumerate(signals[:10]):
                print(f'    {i+1}. {s["code"]} | {s.get("grade", "?")}级 '
                      f'| 池底散户线{s["pool_retail"]:.0f} '
                      f'| {s.get("grade_desc", "")}')
        else:
            print('  今日无买入信号')

        # 持仓卖出检查移至14:55实时检查 (_check_realtime_sells)
        # self._check_daily_sells(data_today, data_yest)

        self.signals_scanned = True

        # 候选池摘要
        summary = self.stock_pool.get_pool_summary()
        print(f'  候选池: {summary["pooled_count"]}只入池, '
              f'{summary["in_wave_count"]}只在波段中')

        return signals

    def _check_daily_sells(self, data_today, data_yest):
        """检查持仓的日线级别卖出条件"""
        if not self.positions:
            return

        print(f'\n  === 持仓卖出检查 ({len(self.positions)}只) ===')

        sells_to_execute = []
        for code, pos_info in list(self.positions.items()):
            today = data_today.get(code)
            yest = data_yest.get(code)

            if not today:
                print(f'    {code} | 无今日数据, 跳过')
                continue

            trend = today.get('trend', np.nan)
            retail = today.get('retail', np.nan)
            trend_y = yest.get('trend', np.nan) if yest else np.nan
            retail_y = yest.get('retail', np.nan) if yest else np.nan

            # 获取或创建卖出追踪器
            tracker = self.sell_trackers.get(code)
            if not tracker:
                method = pos_info.get('sell_method', 'bear')
                if method == 'stall':
                    tracker = StallSellTracker(code, pos_info.get('buy_price', 0))
                else:
                    tracker = SellTracker(code, method)
                self.sell_trackers[code] = tracker

            # 检查卖出 (StallSellTracker需要close_price)
            if isinstance(tracker, StallSellTracker):
                close_price = today.get('close', np.nan)
                sell_signal = tracker.check_sell(trend, retail, close_price, trend_y, retail_y)
            else:
                sell_signal = tracker.check_sell(trend, retail, trend_y, retail_y)

            if sell_signal:
                print(f'    {code} | 触发卖出: {sell_signal["reason"]} | '
                      f'趋势{trend:.1f} 散户{retail:.1f}')
                sells_to_execute.append(sell_signal)
            else:
                print(f'    {code} | 继续持有 | '
                      f'趋势{trend:.1f}(max{tracker.running_max:.1f}) '
                      f'散户{retail:.1f}')

        # 记录待卖出
        self.pending_sells = sells_to_execute

    def _check_realtime_sells(self):
        """
        14:55 用实时价格计算今日三线，判断卖出条件

        与 _check_daily_sells 的区别:
          - 不依赖预计算的日线数据，而是用实时价格拼接历史日线重新计算
          - 能反映今天的价格变化，更接近回测中"以收盘价判断"的逻辑
        """
        if not self.positions:
            print(f'\n  === 无持仓, 跳过实时卖出检查 ===')
            return

        print(f'\n  === 实时卖出检查 ({len(self.positions)}只) ===')
        sells = []

        for code, pos_info in list(self.positions.items()):
            # 获取实时价格
            price = self._get_current_price(code)
            if price <= 0:
                print(f'    {code} | 无法获取实时价格, 跳过')
                continue

            # 实时计算今日三线
            indicators = calc_realtime_indicators(code, price)
            if indicators is None:
                print(f'    {code} | 三线计算失败, 跳过')
                continue

            trend = indicators['trend_today']
            retail = indicators['retail_today']
            trend_y = indicators['trend_yest']
            retail_y = indicators['retail_yest']

            # 获取或创建卖出追踪器
            tracker = self.sell_trackers.get(code)
            if not tracker:
                method = pos_info.get('sell_method', 'bear')
                if method == 'stall':
                    tracker = StallSellTracker(code, pos_info.get('buy_price', 0))
                else:
                    tracker = SellTracker(code, method)
                self.sell_trackers[code] = tracker

            # 检查卖出
            if isinstance(tracker, StallSellTracker):
                sell_signal = tracker.check_sell(trend, retail, price, trend_y, retail_y)
            else:
                sell_signal = tracker.check_sell(trend, retail, trend_y, retail_y)

            if sell_signal:
                print(f'    {code} | 触发卖出: {sell_signal["reason"]} | '
                      f'趋势{trend:.1f} 散户{retail:.1f} | 实时价{price:.2f}')
                sells.append(sell_signal)
            else:
                print(f'    {code} | 继续持有 | '
                      f'趋势{trend:.1f}(max{tracker.running_max:.1f}) '
                      f'散户{retail:.1f} | 实时价{price:.2f}')

        self.pending_sells = sells

    # ============================================================
    # 执行买入 (开盘)
    # ============================================================
    def execute_buys(self):
        """
        执行买入 (09:31 开盘后调用)

        根据盘前扫描的信号 + 风控检查 → 下单
        """
        if not self.signals_scanned:
            print('  !! 未扫描信号, 请先调用 scan_signals()')
            return

        print(f'\n  === 执行买入 ===')

        # 风控检查
        n_current = len(self.positions)
        cash, total_equity = self._get_account_info()

        check = self.risk_manager.full_check(
            hex_info=self.today_hex_info,
            buy_filter=self.today_buy_filter,
            current_positions=n_current,
            cash=cash,
            total_equity=total_equity,
            n_signals=len(self.today_signals),
            bypass_market_filter=self.bypass_filter,
        )

        if not check['can_buy']:
            print(f'  风控拦截: {", ".join(check["reasons"])}')
            self.buys_executed = True
            return

        n_to_buy = check['n_to_buy']
        per_amount = check['per_amount']
        print(f'  风控通过: 买{n_to_buy}只, 每只{per_amount:.0f}元')

        # 确定卖法 (基于模式)
        if self.today_mode == 'crazy':
            sell_method = 'stall'
        else:
            inner_code = self.today_hex_info['inner'] if self.today_hex_info else None
            sell_method = determine_sell_method(inner_code) if inner_code else 'bear'

        # 执行买入
        for i in range(n_to_buy):
            signal = self.today_signals[i]
            code = signal['code']
            price = signal.get('open', signal.get('close', 0))

            if price <= 0 or np.isnan(price):
                print(f'  {code} | 价格异常({price}), 跳过')
                continue

            # 计算股数
            volume = self.risk_manager.calc_buy_volume(price, per_amount)
            if volume <= 0:
                print(f'  {code} | 资金不足, 跳过')
                continue

            # 下单
            success = self._place_buy_order(code, price, volume)

            if success:
                # 记录持仓
                self.positions[code] = {
                    'buy_date': self.today,
                    'buy_price': price,
                    'volume': volume,
                    'sell_method': sell_method,
                    'pool_retail': signal['pool_retail'],
                    'grade': signal.get('grade', '?'),
                    'mode': self.today_mode,
                }

                # 创建卖出追踪器 (根据模式)
                if sell_method == 'stall':
                    self.sell_trackers[code] = StallSellTracker(code, price)
                else:
                    self.sell_trackers[code] = SellTracker(code, sell_method)

                # 记录日志
                self.logger.log_buy(
                    code=code, date=self.today, price=price, volume=volume,
                    sell_method=sell_method,
                    pool_retail=signal['pool_retail'],
                    memo=f'信号排名#{i+1}',
                )

                # 更新风控
                self.risk_manager.record_buy()
                self.stock_pool.mark_bought(code, self.today, sell_method)

        self.buys_executed = True

    # ============================================================
    # 执行卖出 (盘尾)
    # ============================================================
    def execute_sells(self):
        """
        执行卖出 (14:50 盘尾调用)

        根据日线级别的卖出信号执行卖出
        """
        if not hasattr(self, 'pending_sells') or not self.pending_sells:
            print(f'\n  === 无待卖出 ===')
            self.sells_checked = True
            return

        print(f'\n  === 执行卖出 ({len(self.pending_sells)}只) ===')

        for sell_signal in self.pending_sells:
            code = sell_signal['code']
            reason = sell_signal['reason']

            pos_info = self.positions.get(code)
            if not pos_info:
                print(f'  {code} | 无持仓信息, 跳过')
                continue

            volume = pos_info.get('volume', 0)
            if volume <= 0:
                continue

            # 标记卖出原因(供成交回报使用)
            pos_info['pending_sell_reason'] = reason

            # 获取当前价格
            current_price = self._get_current_price(code)

            # 下卖单
            success = self._place_sell_order(code, current_price, volume)

            if success:
                buy_price = pos_info.get('buy_price', 0)
                self.logger.log_sell(
                    code=code, date=self.today, price=current_price,
                    volume=volume, sell_reason=reason, buy_price=buy_price,
                )
                # 模拟模式直接移除持仓
                if self.simulate:
                    self.positions.pop(code, None)
                    self.sell_trackers.pop(code, None)

        self.sells_checked = True

    # ============================================================
    # 收盘处理
    # ============================================================
    def end_of_day(self):
        """
        收盘处理 (15:05 调用)

        - 保存每日快照
        - 重置每日状态
        """
        print(f'\n  === 收盘处理 ===')

        # 保存每日快照
        cash, total_equity = self._get_account_info()
        self.logger.save_daily_snapshot(
            date=self.today,
            positions=self.positions,
            cash=cash,
            total_equity=total_equity,
            market_state={
                'hex_info': self.today_hex_info,
                'buy_filter_can': self.today_buy_filter.get('can_buy') if self.today_buy_filter else None,
                'n_signals': len(self.today_signals),
                'mode': self.today_mode,
            },
            signals_today=[
                {'code': s['code'], 'pool_retail': s['pool_retail']}
                for s in self.today_signals
            ],
        )

        # 保存实时快照 (供dashboard)
        self.logger.save_position_snapshot(self.positions, cash, total_equity)

        # 重置每日状态
        self.risk_manager.reset_daily()
        self.today_signals = []
        self.today_mode = 'normal'
        self.today_512_snapshot = None
        self.pending_sells = []
        self.order_times = {}
        self.data_loaded = False
        self.signals_scanned = False
        self.buys_executed = False
        self.sells_checked = False

        print(f'  持仓: {len(self.positions)} 只')
        print(f'  现金: {cash:,.0f} | 总资产: {total_equity:,.0f}')
        print(f'  日终处理完成')

        # 收盘后自动更新数据层 (仅实盘模式)
        if not self.simulate and self.connected:
            try:
                from data_layer.update_xtdata import update_all_xt
                print(f'\n  === 收盘数据更新 (xtdata) ===')
                update_all_xt()
            except Exception as e:
                print(f'  !! 数据更新失败(不影响策略): {e}')

        print()

    # ============================================================
    # QMT 交易接口
    # ============================================================
    def _place_buy_order(self, code, price, volume):
        """下买单"""
        if self.simulate:
            print(f'  [模拟买入] {code} | {price:.2f}×{volume}股 | '
                  f'{price * volume:.0f}元')
            return True

        try:
            acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
            # 使用最新价委托 (非涨停价)
            order_id = self.trader.order_stock(
                acc, code, xtconstant.STOCK_BUY,
                volume, xtconstant.FIX_PRICE, price,
                strategy_name='乾坤三线', order_remark='买入',
            )
            self.order_times[code] = time.time()
            print(f'  [下单成功] 买入 {code} | {price:.2f}×{volume}股 | '
                  f'委托ID: {order_id}')
            return True
        except Exception as e:
            print(f'  !! 买入下单失败 {code}: {e}')
            return False

    def _place_sell_order(self, code, price, volume):
        """下卖单"""
        if self.simulate:
            print(f'  [模拟卖出] {code} | {price:.2f}×{volume}股')
            return True

        try:
            acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
            order_id = self.trader.order_stock(
                acc, code, xtconstant.STOCK_SELL,
                volume, xtconstant.FIX_PRICE, price,
                strategy_name='乾坤三线', order_remark='卖出',
            )
            print(f'  [下单成功] 卖出 {code} | {price:.2f}×{volume}股 | '
                  f'委托ID: {order_id}')
            return True
        except Exception as e:
            print(f'  !! 卖出下单失败 {code}: {e}')
            return False

    def _get_account_info(self):
        """获取账户信息: (可用现金, 总资产)"""
        if self.simulate:
            # 模拟: 假设20万, 减去持仓成本
            held_value = sum(
                p.get('buy_price', 0) * p.get('volume', 0)
                for p in self.positions.values()
            )
            cash = 200000 - held_value
            return max(cash, 0), 200000

        try:
            acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
            acc_info = self.trader.query_stock_asset(acc)
            return acc_info.cash, acc_info.total_asset
        except Exception:
            return 0, 0

    def _get_current_price(self, code):
        """获取当前价格"""
        if self.simulate:
            # 模拟: 从数据中取最新收盘价
            df = self.all_stock_data.get(code)
            if df is not None and len(df) > 0:
                return float(df.iloc[-1]['close'])
            return 0

        try:
            tick = xtdata.get_full_tick([code])
            if tick and code in tick:
                return tick[code]['lastPrice']
        except Exception:
            pass
        return 0

    def _handle_order_timeout(self):
        """检查并撤销超时委托"""
        if self.simulate:
            return

        now = time.time()
        for code, order_time in list(self.order_times.items()):
            if now - order_time > ORDER_TIMEOUT:
                try:
                    # 查询未成交委托并撤单
                    acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
                    orders = self.trader.query_stock_orders(acc)
                    for order in orders:
                        if (order.stock_code == code and
                                order.order_status in [xtconstant.ORDER_JUNK,
                                                        xtconstant.ORDER_UNREPORTED]):
                            self.trader.cancel_order_stock(acc, order.order_id)
                            self.logger.log_cancel(code, self.today)
                            print(f'  [超时撤单] {code}')
                    self.order_times.pop(code, None)
                except Exception as e:
                    print(f'  !! 撤单异常 {code}: {e}')

    # ============================================================
    # 主循环
    # ============================================================
    def run(self):
        """
        主运行循环 — 适用于独立进程模式

        调度逻辑:
          09:15 → load_data() + scan_signals()
          09:31 → execute_buys()
          14:50 → execute_sells()
          15:05 → end_of_day()
          然后等待下一个交易日
        """
        print('=' * 60)
        print('  乾坤三线 v1.0 — miniQMT 实盘策略')
        print(f'  模式: {"模拟" if self.simulate else "实盘"}')
        print(f'  账号: {self.account or "(模拟)"}')
        print('=' * 60)

        self._start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.simulate:
            self.connected = True
            self._write_heartbeat('模拟模式启动')
        else:
            self._write_heartbeat('正在连接QMT...')
            if not self.connect():
                print('  !! 连接失败, 退出')
                self._write_heartbeat('QMT连接失败')
                return
            self._write_heartbeat('QMT已连接')

        while True:
            try:
                now = datetime.now()
                time_str = now.strftime('%H%M%S')

                # 实盘模式: 检测 QMT 连接是否断开
                if not self.simulate and self.trader:
                    try:
                        acc = xttrader.StockAccount(self.account, QMT_ACCOUNT_TYPE)
                        asset = self.trader.query_stock_asset(acc)
                        if asset is None:
                            print('  !! QMT 连接已断开')
                            self._write_heartbeat('QMT连接断开')
                            reconnect = self.trader.connect()
                            if reconnect != 0:
                                print('  !! 重连失败, 等待30秒后重试')
                                self._write_heartbeat('QMT断开, 等待重连...')
                                time.sleep(30)
                                continue
                            else:
                                print('  QMT 重连成功')
                                self._write_heartbeat('QMT已重连')
                    except Exception:
                        self._write_heartbeat('QMT连接异常')
                        time.sleep(30)
                        continue

                # 工作日判断 (周末跳过)
                if now.weekday() >= 5:
                    self._write_heartbeat('等待开盘(周末)')
                    time.sleep(60)
                    continue

                # 盘前: 09:15
                if time_str >= SELECT_TIME and not self.data_loaded:
                    self._write_heartbeat('盘前选股')
                    self.load_data()
                    self.scan_signals()

                # 开盘: 09:31
                if time_str >= BUY_START_TIME and not self.buys_executed:
                    self._write_heartbeat('执行买入')
                    self.execute_buys()

                # 盘中: 超时撤单
                if BUY_START_TIME <= time_str <= BUY_END_TIME:
                    self._handle_order_timeout()

                # 盘尾: 14:55 实时三线卖出检查 + 立即执行
                if time_str >= SELL_REALTIME_CHECK and not self.sells_checked:
                    self._write_heartbeat('实时卖出检查')
                    self._check_realtime_sells()
                    self.execute_sells()

                # 收盘: 15:05
                if time_str >= DATA_UPDATE_TIME and self.sells_checked:
                    self.end_of_day()
                    # 等到次日盘前
                    self._wait_until_next_day()

                # 非交易时段, 低频轮询
                if time_str < SELECT_TIME or time_str > DATA_UPDATE_TIME:
                    self._write_heartbeat('等待开盘')
                    time.sleep(30)
                else:
                    time.sleep(5)

            except KeyboardInterrupt:
                print('\n  用户中断, 退出')
                break
            except Exception as e:
                print(f'  !! 主循环异常: {e}')
                traceback.print_exc()
                time.sleep(30)

    def _write_heartbeat(self, phase=''):
        """写入心跳到 runtime 目录，供 dashboard 读取"""
        runtime_dir = os.path.join(PROJECT_ROOT, 'live', 'runtime')
        os.makedirs(runtime_dir, exist_ok=True)
        mode = 'simulate' if self.simulate else 'live'
        status_path = os.path.join(runtime_dir, f'{mode}.status.json')
        try:
            status = {
                'mode': mode,
                'pid': os.getpid(),
                'start_time': getattr(self, '_start_time', ''),
                'last_heartbeat': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'phase': phase,
                'connected': self.connected,
                'data_loaded': self.data_loaded,
                'positions': len(self.positions),
                'today_mode': getattr(self, 'today_mode', ''),
                'today': getattr(self, 'today', ''),
            }
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass  # 心跳写入失败不影响策略运行

    def _wait_until_next_day(self):
        """等待到次日盘前"""
        self._write_heartbeat('等待次日')
        now = datetime.now()
        tomorrow_9 = (now + timedelta(days=1)).replace(
            hour=9, minute=10, second=0, microsecond=0)
        wait_seconds = (tomorrow_9 - now).total_seconds()
        if wait_seconds > 0:
            print(f'  等待至次日 {tomorrow_9.strftime("%m-%d %H:%M")} '
                  f'({wait_seconds/3600:.1f}小时)')
            time.sleep(wait_seconds)

    # ============================================================
    # 手动执行 (适用于盘前手动调用)
    # ============================================================
    def run_once(self):
        """
        手动执行一次完整流程 (非循环模式)

        适合手动调试或定时任务调用:
          python -c "from live.qmt_strategy import QiankunStrategy; s=QiankunStrategy(simulate=True); s.run_once()"
        """
        self.load_data()
        signals = self.scan_signals()

        print(f'\n  === 运行摘要 ===')
        print(f'  日期: {self.today}')
        print(f'  模式: {"疯狂" if self.today_mode == "crazy" else "常规"}')
        print(f'  买入信号: {len(signals)} 只')
        print(f'  待卖出: {len(getattr(self, "pending_sells", []))} 只')
        print(f'  当前持仓: {len(self.positions)} 只')

        for code, info in self.positions.items():
            tracker = self.sell_trackers.get(code)
            max_trend = tracker.running_max if tracker else 0
            print(f'    {code} | 买入{info.get("buy_date", "?")} | '
                  f'{info.get("buy_price", 0):.2f} | '
                  f'{info.get("sell_method", "?")}卖 | '
                  f'趋势max{max_trend:.1f}')

        # 风控预览 (原始, 不bypass)
        n_current = len(self.positions)
        cash, total_equity = self._get_account_info()
        check = self.risk_manager.full_check(
            self.today_hex_info, self.today_buy_filter,
            n_current, cash, total_equity, len(signals),
            bypass_market_filter=False,
        )
        print(f'  风控结果: {"可买" if check["can_buy"] else "不可买"} | '
              f'{", ".join(check["reasons"])}')

        # === 信号详情 (无论风控是否拦截都显示) ===
        if signals:
            if self.today_mode == 'crazy':
                sell_method = 'stall'
            else:
                inner_code = self.today_hex_info['inner'] if self.today_hex_info else None
                sell_method = determine_sell_method(inner_code) if inner_code else 'bear'
            per_slot = total_equity / MAX_POSITIONS if total_equity > 0 else 0

            print(f'\n  === 信号详情 (假如买入) ===')
            print(f'  模式: {"疯狂" if self.today_mode == "crazy" else "常规"} | 卖法: {sell_method}')
            print(f'  每仓金额: {per_slot:,.0f}元')
            for i, s in enumerate(signals[:MAX_POSITIONS]):
                price = s.get('open', s.get('close', 0))
                volume = int((per_slot * 0.98) // price // 100 * 100) if price > 0 else 0
                print(f'    {i+1}. {s["code"]} | {s.get("grade", "?")}级 '
                      f'| 价格{price:.2f} | {volume}股×{price:.2f}={volume*price:,.0f}元 '
                      f'| 池底散户线{s["pool_retail"]:.0f} '
                      f'| {s.get("grade_desc", "")}')

        # === 如果是bypass模式, 执行模拟买入 ===
        if self.bypass_filter and signals:
            print(f'\n  === [测试模式] 执行模拟买入 ===')
            self.execute_buys()

        # === 保存信号验证文件 ===
        self._save_signal_report(signals, check)

        return signals

    def _save_signal_report(self, signals, check):
        """保存每日信号验证报告 (JSON), 方便逐日对比"""
        import json
        report_dir = os.path.join(PROJECT_ROOT, 'live', 'logs', 'signal_reports')
        os.makedirs(report_dir, exist_ok=True)

        report = {
            'date': self.today,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': self.today_mode,
            'market': {
                'hex_info': self.today_hex_info,
                'buy_filter': {
                    'can_buy': self.today_buy_filter.get('can_buy') if self.today_buy_filter else None,
                    'reason': self.today_buy_filter.get('reason', '') if self.today_buy_filter else '',
                    'grade': self.today_buy_filter.get('grade', '') if self.today_buy_filter else '',
                    'ymd_yy': self.today_buy_filter.get('ymd_yy', '') if self.today_buy_filter else '',
                },
            },
            'risk_check': {
                'can_buy': check['can_buy'],
                'reasons': check['reasons'],
            },
            'signals': [
                {
                    'code': s['code'],
                    'pool_retail': s['pool_retail'],
                    'grade': s.get('grade', '?'),
                    'grade_desc': s.get('grade_desc', ''),
                    'year_yy': s.get('year_yy', ''),
                    'combo': s.get('combo', ''),
                    'trend': s.get('trend', None),
                    'retail': s.get('retail', None),
                    'open': s.get('open', None),
                    'close': s.get('close', None),
                }
                for s in signals
            ],
            'positions': {
                code: {
                    'buy_date': info.get('buy_date', '?'),
                    'buy_price': info.get('buy_price', 0),
                    'sell_method': info.get('sell_method', '?'),
                }
                for code, info in self.positions.items()
            },
            'pending_sells': [
                {'code': s['code'], 'reason': s.get('reason', '')}
                for s in getattr(self, 'pending_sells', [])
            ],
        }

        fname = f'report_{self.today}.json'
        fpath = os.path.join(report_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f'\n  信号报告已保存: {fpath}')


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='乾坤三线 miniQMT 实盘策略')
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='模拟模式(不实际下单)')
    parser.add_argument('--once', action='store_true',
                        help='只运行一次(调试用)')
    parser.add_argument('--bypass-filter', action='store_true',
                        help='跳过年阳禁买(仅测试用, 验证信号是否正确触发买入)')
    parser.add_argument('--account', type=str, default='',
                        help='QMT账号')
    args = parser.parse_args()

    strategy = QiankunStrategy(
        account=args.account or None,
        simulate=args.simulate,
        bypass_filter=args.bypass_filter,
    )

    if args.once:
        strategy.run_once()
    else:
        strategy.run()
