# -*- coding: utf-8 -*-
"""
visualize.py — 可视化封装

1. plot_stock(): 查看指定个股K线 + 买卖标注 + 指标副图
2. plot_portfolio(): 组合净值曲线
"""
import backtrader as bt


def plot_stock(cerebro, strat, code, volume=False):
    """查看指定股票的K线图 + 买卖标注

    Args:
        cerebro: Cerebro 实例 (已运行完成)
        strat: 策略实例
        code: 股票代码 (如 '002460')
        volume: 是否显示成交量
    """
    # 找到目标 data
    target = None
    for d in strat.datas:
        if d._name == code:
            target = d
            break

    if target is None:
        print(f"未找到 {code} 的数据")
        return

    # 设置只画选中的股票 (其他全隐藏)
    for d in strat.datas:
        d.plotinfo.plot = (d._name == code)

    # 绘图 (A股红涨绿跌)
    cerebro.plot(
        style='candlestick',
        barup='red',
        bardown='green',
        volup='red',
        voldown='green',
        volume=volume,
        fmt_x_ticks='%Y-%m',
        fmt_x_data='%Y-%m-%d',
    )


def plot_top(cerebro, strat, n=5, sort_by='pnl'):
    """查看盈利/亏损 Top N 股票的K线图

    Args:
        cerebro: Cerebro 实例
        strat: 策略实例
        n: 显示前几名
        sort_by: 'pnl' (盈亏额) 或 'pnl_pct' (盈亏率)
    """
    # 按盈亏排序交易
    trades_by_code = {}
    for t in strat.trade_log:
        code = t['code']
        if code not in trades_by_code:
            trades_by_code[code] = 0
        trades_by_code[code] += t.get('pnlcomm', t.get('pnl', 0))

    if not trades_by_code:
        print("无交易记录")
        return

    # 排序
    sorted_codes = sorted(trades_by_code.items(),
                          key=lambda x: x[1], reverse=True)

    # 打印排行
    print(f"\n盈亏排行 Top {n}:")
    print(f"{'排名':>4} {'代码':>8} {'盈亏':>12}")
    print("-" * 30)
    for i, (code, pnl) in enumerate(sorted_codes[:n]):
        print(f"{i+1:>4} {code:>8} {pnl:>+12,.0f}")

    # 逐个绘图
    for code, pnl in sorted_codes[:n]:
        print(f"\n绘制 {code} (盈亏={pnl:+,.0f})...")
        plot_stock(cerebro, strat, code)


def plot_worst(cerebro, strat, n=5):
    """查看亏损最多的 Top N"""
    trades_by_code = {}
    for t in strat.trade_log:
        code = t['code']
        if code not in trades_by_code:
            trades_by_code[code] = 0
        trades_by_code[code] += t.get('pnlcomm', t.get('pnl', 0))

    sorted_codes = sorted(trades_by_code.items(),
                          key=lambda x: x[1])

    print(f"\n亏损排行 Top {n}:")
    print(f"{'排名':>4} {'代码':>8} {'盈亏':>12}")
    print("-" * 30)
    for i, (code, pnl) in enumerate(sorted_codes[:n]):
        print(f"{i+1:>4} {code:>8} {pnl:>+12,.0f}")

    for code, pnl in sorted_codes[:n]:
        print(f"\n绘制 {code} (盈亏={pnl:+,.0f})...")
        plot_stock(cerebro, strat, code)


def get_portfolio_stats(strat):
    """提取策略的每日统计数据, 供外部绘图

    Returns:
        list of dict: [{'date', 'cash', 'value', 'n_positions', ...}, ...]
    """
    return strat.daily_stats
