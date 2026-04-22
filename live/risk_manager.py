# -*- coding: utf-8 -*-
"""
乾坤三线 v1.0 — 风控模块

核心职责:
  1. 八卦空仓过滤 (六爻卦 + 年阳禁买)
  2. 仓位控制 (3仓等分)
  3. 买入资金分配
  4. 单日/总持仓限制
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.config import (
    MAX_POSITIONS, DAILY_BUY_LIMIT, POSITION_MODE,
    MIN_BUY_AMOUNT, SKIP_HEXAGRAMS,
    CRAZY_ALLOWED_GRADES, NORMAL_ALLOWED_GRADES,
)


class RiskManager:
    """
    风控管理器

    Usage:
        rm = RiskManager()
        # 每日盘前检查
        can, reason = rm.check_market_filter(hex_info, buy_filter)
        if not can:
            print(f'今日不买: {reason}')
            return

        # 买入时检查仓位
        alloc = rm.calc_buy_allocation(signals, positions, cash, total_equity)
    """

    def __init__(self, max_positions=None, daily_buy_limit=None,
                 position_mode=None, min_buy_amount=None):
        self.max_positions = max_positions or MAX_POSITIONS
        self.daily_buy_limit = daily_buy_limit or DAILY_BUY_LIMIT
        self.position_mode = position_mode or POSITION_MODE
        self.min_buy_amount = min_buy_amount or MIN_BUY_AMOUNT
        self.today_buy_count = 0  # 今日已买入数量

    def reset_daily(self):
        """每日重置"""
        self.today_buy_count = 0

    # ============================================================
    # 市场过滤
    # ============================================================
    def check_market_filter(self, hex_info=None, buy_filter=None):
        """
        检查市场层面是否允许买入 (仅空仓卦过滤)

        Args:
            hex_info: dict {inner, outer, hex_code} — 六爻卦信息
            buy_filter: dict — 保留接口兼容，不再用于年阳禁买

        Returns:
            (bool, str): (是否允许买入, 原因)
        """
        reasons = []

        # 六爻空仓卦过滤
        if hex_info:
            hex_code = hex_info.get('hex_code')
            if hex_code and hex_code in SKIP_HEXAGRAMS:
                reasons.append(f'空仓卦({hex_code})')

        if reasons:
            return False, ' + '.join(reasons)
        return True, '通过'

    # ============================================================
    # 仓位控制
    # ============================================================
    def check_position_limit(self, current_positions):
        """
        检查是否还有空仓位

        Args:
            current_positions: int, 当前持仓数量

        Returns:
            (int, str): (可买入数量, 说明)
        """
        # 剩余仓位
        slots = self.max_positions - current_positions

        # 今日剩余买入额度
        today_remaining = self.daily_buy_limit - self.today_buy_count

        can_buy = min(slots, today_remaining)

        if can_buy <= 0:
            if slots <= 0:
                return 0, f'满仓({current_positions}/{self.max_positions})'
            else:
                return 0, f'今日已买{self.today_buy_count}只(限{self.daily_buy_limit})'

        return can_buy, f'可买{can_buy}只(仓位{current_positions}/{self.max_positions})'

    def calc_buy_allocation(self, n_to_buy, current_positions,
                            cash, total_equity):
        """
        计算每只股票的买入金额

        Args:
            n_to_buy: 本次要买几只
            current_positions: 当前持仓数
            cash: 可用现金
            total_equity: 总资产(现金+持仓市值)

        Returns:
            float: 每只分配的金额
        """
        if n_to_buy <= 0 or cash < self.min_buy_amount:
            return 0

        if self.position_mode == 'equal':
            # 等分: 总资产 / max_positions
            per_slot = total_equity / self.max_positions
            per_buy = min(per_slot, cash / n_to_buy)
        elif self.position_mode == 'available':
            # 可用资金等分
            per_buy = cash / n_to_buy
        else:
            per_buy = cash / n_to_buy

        return max(per_buy, 0)

    def calc_buy_volume(self, price, amount):
        """
        计算买入股数 (整百股)

        Args:
            price: 买入价格
            amount: 分配金额

        Returns:
            int: 买入股数 (整百股), 0表示资金不足
        """
        if price <= 0 or amount < self.min_buy_amount:
            return 0

        # 预留2%作为手续费
        volume = int((amount * 0.98) // price // 100 * 100)
        return volume if volume >= 100 else 0

    def record_buy(self, count=1):
        """记录今日买入"""
        self.today_buy_count += count

    # ============================================================
    # 分级过滤
    # ============================================================
    def filter_by_grade(self, signals, is_crazy):
        """
        按等级过滤信号

        Args:
            signals: list of signal dicts (需含 'grade' 字段)
            is_crazy: bool, 是否疯狂模式

        Returns:
            list: 过滤后的信号
        """
        allowed = CRAZY_ALLOWED_GRADES if is_crazy else NORMAL_ALLOWED_GRADES
        return [s for s in signals if s.get('grade', 'F') in allowed]

    # ============================================================
    # 综合检查
    # ============================================================
    def full_check(self, hex_info, buy_filter, current_positions, cash,
                   total_equity, n_signals, bypass_market_filter=False):
        """
        完整的买入前检查

        Args:
            hex_info: 六爻卦
            buy_filter: 买入过滤
            current_positions: 当前持仓数
            cash: 可用现金
            total_equity: 总资产
            n_signals: 候选信号数
            bypass_market_filter: True=跳过市场过滤(仅测试用)

        Returns:
            dict: {
                'can_buy': bool,
                'n_to_buy': int,        # 实际可买数量
                'per_amount': float,     # 每只分配金额
                'reasons': list[str],    # 所有原因
            }
        """
        result = {
            'can_buy': False,
            'n_to_buy': 0,
            'per_amount': 0,
            'reasons': [],
        }

        # 1. 市场过滤
        can, reason = self.check_market_filter(hex_info, buy_filter)
        if not can:
            if bypass_market_filter:
                result['reasons'].append(f'[已跳过] {reason}')
            else:
                result['reasons'].append(reason)
                return result

        # 2. 仓位检查
        can_buy, pos_reason = self.check_position_limit(current_positions)
        if can_buy <= 0:
            result['reasons'].append(pos_reason)
            return result

        # 3. 实际可买数量
        n_to_buy = min(can_buy, n_signals)
        if n_to_buy <= 0:
            result['reasons'].append('无买入信号')
            return result

        # 4. 资金分配
        per_amount = self.calc_buy_allocation(
            n_to_buy, current_positions, cash, total_equity)
        if per_amount < self.min_buy_amount:
            result['reasons'].append(f'资金不足(每只仅{per_amount:.0f}元)')
            return result

        result['can_buy'] = True
        result['n_to_buy'] = n_to_buy
        result['per_amount'] = per_amount
        result['reasons'].append(f'可买{n_to_buy}只, 每只{per_amount:.0f}元')
        return result
