# -*- coding: utf-8 -*-
"""
run.py — 总入口

用法:
    python -m backtest_bt.run                       # 完整回测
    python -m backtest_bt.run --plot 002460          # 回测+查看某股K线
    python -m backtest_bt.run --plot-top 5           # 回测+查看盈利前5
    python -m backtest_bt.run --no-commission        # 不计佣金 (对比验证)
    python -m backtest_bt.run --verbose              # 打印详细日志
    python -m backtest_bt.run --max-pos 5            # 指定最大持仓数
"""
import sys
import os
import argparse
import time

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest_bt.engine import build_cerebro, run_backtest
from backtest_bt.visualize import plot_stock, plot_top, plot_worst


def main():
    parser = argparse.ArgumentParser(description='乾坤三线 Backtrader 回测')
    parser.add_argument('--plot', type=str, default=None,
                        help='回测后查看指定股票K线 (如 002460)')
    parser.add_argument('--plot-top', type=int, default=0,
                        help='回测后查看盈利前N的K线')
    parser.add_argument('--plot-worst', type=int, default=0,
                        help='回测后查看亏损前N的K线')
    parser.add_argument('--no-commission', action='store_true',
                        help='不计佣金 (用于与原系统对比)')
    parser.add_argument('--verbose', action='store_true',
                        help='打印详细日志')
    parser.add_argument('--max-pos', type=int, default=None,
                        help='最大持仓数')
    parser.add_argument('--daily-limit', type=int, default=None,
                        help='每日买入上限')
    parser.add_argument('--pos-mode', type=str, default=None,
                        choices=['equal', 'available'],
                        help='仓位模式')
    parser.add_argument('--start', type=str, default=None,
                        help='回测起始日期 (如 2020-01-01)')
    parser.add_argument('--end', type=str, default=None,
                        help='回测结束日期 (如 2025-12-31)')
    parser.add_argument('--no-cache', action='store_true',
                        help='不使用扫描缓存')

    args = parser.parse_args()

    print("=" * 60)
    print("乾坤三线 Backtrader 回测引擎")
    print("=" * 60)

    t_total = time.time()

    # 构建引擎
    cerebro = build_cerebro(
        start_date=args.start,
        end_date=args.end,
        commission=not args.no_commission,
        verbose=args.verbose,
        max_positions=args.max_pos,
        daily_buy_limit=args.daily_limit,
        position_mode=args.pos_mode,
    )

    # 运行回测
    strat = run_backtest(cerebro)

    print(f"\n总耗时: {time.time() - t_total:.1f}s")

    # 可视化
    if args.plot:
        print(f"\n绘制 {args.plot} K线图...")
        plot_stock(cerebro, strat, args.plot)

    if args.plot_top > 0:
        plot_top(cerebro, strat, args.plot_top)

    if args.plot_worst > 0:
        plot_worst(cerebro, strat, args.plot_worst)


if __name__ == '__main__':
    main()
