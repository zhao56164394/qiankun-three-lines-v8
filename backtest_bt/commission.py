# -*- coding: utf-8 -*-
"""
commission.py — A股佣金模型

佣金: max(成交额 × 0.0003, 5.0)，买卖双向
印花税: 成交额 × 0.001，仅卖出
"""
import backtrader as bt
from .config import COMMISSION_RATE, COMMISSION_MIN, STAMP_TAX_RATE


class CNStockCommission(bt.CommInfoBase):
    """A股佣金模型: 佣金(双向) + 印花税(仅卖出)"""

    params = (
        ('commission', COMMISSION_RATE),    # 万三
        ('min_commission', COMMISSION_MIN), # 最低5元
        ('stamp_tax', STAMP_TAX_RATE),      # 千一印花税
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算佣金（买卖双向都会调用）

        size > 0: 买入
        size < 0: 卖出（额外加印花税）
        """
        turnover = abs(size) * price

        # 佣金 (双向)
        comm = max(turnover * self.p.commission, self.p.min_commission)

        # 印花税 (仅卖出)
        if size < 0:
            comm += turnover * self.p.stamp_tax

        return comm


class ZeroCommission(bt.CommInfoBase):
    """零佣金模型 — 用于与原系统对比验证"""

    params = (
        ('commission', 0),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        return 0.0
