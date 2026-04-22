# -*- coding: utf-8 -*-
"""
乾坤三线 v1.0 — 交易日志持久化

记录每笔交易、每日持仓快照、市场状态，供dashboard读取和复盘。

日志文件结构:
  logs/
    trades.csv        交易记录 (追加写)
    daily_snapshot/
      2026-04-05.json  每日持仓快照
    signals/
      2026-04-05.json  每日信号记录
"""
import os
import sys
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.config import LOG_DIR, SNAPSHOT_DIR


class TradeLogger:
    """
    交易日志管理器

    Usage:
        logger = TradeLogger()
        logger.log_buy('000001', '2026-04-05', 10.5, 3000, 'bear', '池底散户线-520')
        logger.log_sell('000001', '2026-04-20', 12.0, '首穿89(熊卖)')
        logger.save_daily_snapshot(date, positions, cash, total_equity, market_state)
    """

    def __init__(self, log_dir=None, snapshot_dir=None):
        self.log_dir = log_dir or LOG_DIR
        self.snapshot_dir = snapshot_dir or SNAPSHOT_DIR

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'daily_snapshot'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'signals'), exist_ok=True)

        self.trades_file = os.path.join(self.log_dir, 'trades.csv')
        self._init_trades_file()

    def _init_trades_file(self):
        """初始化交易记录CSV (如果不存在)"""
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'code', 'direction', 'date', 'time', 'price',
                    'volume', 'amount', 'sell_method', 'sell_reason',
                    'pool_retail', 'memo', 'timestamp',
                ])

    # ============================================================
    # 交易记录
    # ============================================================
    def log_buy(self, code, date, price, volume, sell_method='bear',
                memo='', pool_retail=0):
        """
        记录买入

        Args:
            code: 股票代码
            date: 买入日期 'YYYY-MM-DD'
            price: 买入价格
            volume: 买入股数
            sell_method: 卖法 'bull'/'bear'
            memo: 备注
            pool_retail: 池底散户线
        """
        amount = price * volume
        self._append_trade(
            code=code,
            direction='BUY',
            date=date,
            time=datetime.now().strftime('%H:%M:%S'),
            price=price,
            volume=volume,
            amount=amount,
            sell_method=sell_method,
            sell_reason='',
            pool_retail=pool_retail,
            memo=memo,
        )
        self._print(f'[买入] {code} | {date} | {price:.2f}×{volume}股 | '
                     f'{amount:.0f}元 | {sell_method}卖')

    def log_sell(self, code, date, price, volume, sell_reason='',
                 buy_price=0, memo=''):
        """
        记录卖出

        Args:
            code: 股票代码
            date: 卖出日期
            price: 卖出价格
            volume: 卖出股数
            sell_reason: 卖出原因
            buy_price: 买入价格(计算收益用)
            memo: 备注
        """
        amount = price * volume
        ret_pct = (price / buy_price - 1) * 100 if buy_price > 0 else 0

        self._append_trade(
            code=code,
            direction='SELL',
            date=date,
            time=datetime.now().strftime('%H:%M:%S'),
            price=price,
            volume=volume,
            amount=amount,
            sell_method='',
            sell_reason=sell_reason,
            pool_retail=0,
            memo=f'{memo} 收益{ret_pct:+.1f}%' if buy_price > 0 else memo,
        )
        ret_str = f' | 收益{ret_pct:+.1f}%' if buy_price > 0 else ''
        self._print(f'[卖出] {code} | {date} | {price:.2f}×{volume}股 | '
                     f'{sell_reason}{ret_str}')

    def log_cancel(self, code, date, reason='超时撤单'):
        """记录撤单"""
        self._append_trade(
            code=code,
            direction='CANCEL',
            date=date,
            time=datetime.now().strftime('%H:%M:%S'),
            price=0, volume=0, amount=0,
            sell_method='', sell_reason='',
            pool_retail=0,
            memo=reason,
        )
        self._print(f'[撤单] {code} | {date} | {reason}')

    def _append_trade(self, **kwargs):
        """追加一条交易记录到CSV"""
        kwargs['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.trades_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    kwargs.get('code', ''),
                    kwargs.get('direction', ''),
                    kwargs.get('date', ''),
                    kwargs.get('time', ''),
                    kwargs.get('price', 0),
                    kwargs.get('volume', 0),
                    kwargs.get('amount', 0),
                    kwargs.get('sell_method', ''),
                    kwargs.get('sell_reason', ''),
                    kwargs.get('pool_retail', 0),
                    kwargs.get('memo', ''),
                    kwargs.get('timestamp', ''),
                ])
        except Exception as e:
            print(f'  !! 交易日志写入失败: {e}')

    # ============================================================
    # 每日快照
    # ============================================================
    def save_daily_snapshot(self, date, positions, cash, total_equity,
                            market_state=None, signals_today=None):
        """
        保存每日持仓快照

        Args:
            date: 日期 'YYYY-MM-DD'
            positions: dict {code: {volume, cost_price, current_price, ...}}
            cash: 可用现金
            total_equity: 总资产
            market_state: dict, 市场状态 (可选)
            signals_today: list, 今日信号 (可选)
        """
        snapshot = {
            'date': date,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cash': cash,
            'total_equity': total_equity,
            'n_positions': len(positions),
            'positions': positions,
            'market_state': market_state,
        }

        # 保存快照
        snapshot_path = os.path.join(self.log_dir, 'daily_snapshot', f'{date}.json')
        self._save_json(snapshot_path, snapshot)

        # 保存信号
        if signals_today:
            signal_path = os.path.join(self.log_dir, 'signals', f'{date}.json')
            self._save_json(signal_path, {
                'date': date,
                'signals': signals_today,
            })

    def save_position_snapshot(self, positions, cash, total_equity):
        """
        保存最新持仓快照 (供 dashboard 实时读取)

        Args:
            positions: dict
            cash: float
            total_equity: float
        """
        snapshot = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cash': cash,
            'total_equity': total_equity,
            'n_positions': len(positions),
            'positions': positions,
        }
        path = os.path.join(self.snapshot_dir, 'latest.json')
        self._save_json(path, snapshot)

    # ============================================================
    # 读取历史
    # ============================================================
    def load_trades(self):
        """
        读取全部交易记录

        Returns:
            list[dict]: 交易记录列表
        """
        if not os.path.exists(self.trades_file):
            return []

        trades = []
        try:
            with open(self.trades_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
        except Exception as e:
            print(f'  !! 读取交易记录失败: {e}')
        return trades

    def load_daily_snapshot(self, date):
        """读取指定日期的快照"""
        path = os.path.join(self.log_dir, 'daily_snapshot', f'{date}.json')
        return self._load_json(path)

    def load_latest_snapshot(self):
        """读取最新持仓快照"""
        path = os.path.join(self.snapshot_dir, 'latest.json')
        return self._load_json(path)

    def get_open_positions_from_log(self):
        """
        从交易记录推算当前未平仓持仓

        Returns:
            dict: {code: {buy_date, buy_price, volume, sell_method, pool_retail}}
        """
        trades = self.load_trades()
        positions = {}

        for t in trades:
            code = t.get('code', '')
            direction = t.get('direction', '')

            if direction == 'BUY':
                positions[code] = {
                    'buy_date': t.get('date', ''),
                    'buy_price': float(t.get('price', 0)),
                    'volume': int(t.get('volume', 0)),
                    'sell_method': t.get('sell_method', 'bear'),
                    'pool_retail': float(t.get('pool_retail', 0)),
                }
            elif direction == 'SELL' and code in positions:
                del positions[code]

        return positions

    # ============================================================
    # 工具
    # ============================================================
    def _save_json(self, path, data):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f'  !! JSON保存失败 {path}: {e}')

    def _load_json(self, path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f'  !! JSON读取失败 {path}: {e}')
            return None

    def _print(self, msg):
        """打印带时间戳的日志"""
        ts = datetime.now().strftime('%H:%M:%S')
        print(f'  {ts} {msg}')
