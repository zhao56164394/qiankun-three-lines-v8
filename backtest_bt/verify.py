# -*- coding: utf-8 -*-
"""
verify.py — 与原引擎 backtest_capital.py 对比验证

对比方式:
1. 运行原引擎获取交易记录
2. 运行 Backtrader 引擎获取交易记录
3. 逐笔对比买卖日期和价格
4. 对比总收益率、回撤等统计指标

注意: 由于卖出价差异 (原=信号日收盘, BT=次日开盘),
      交易记录不会完全一致, 但信号日和买入价应该高度吻合。
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_original_results():
    """加载原引擎的回测结果 JSON"""
    bt_path = os.path.join(PROJECT_ROOT, 'data_layer', 'data',
                           'backtest_result.json')
    if not os.path.exists(bt_path):
        print(f"原引擎结果不存在: {bt_path}")
        print("请先运行: python backtest_capital.py")
        return None

    with open(bt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def run_bt_backtest(no_commission=True, verbose=False):
    """运行 Backtrader 回测"""
    from backtest_bt.engine import build_cerebro, run_backtest

    cerebro = build_cerebro(
        commission=not no_commission,
        verbose=verbose,
    )
    strat = run_backtest(cerebro)
    return cerebro, strat


def compare_trades(original_trades, bt_trades):
    """对比交易记录

    原引擎格式: {code, buy_date, sell_date, buy_price, sell_price, ...}
    BT引擎格式: 从 strat.trade_log 提取
    """
    print(f"\n{'='*60}")
    print("交易记录对比")
    print(f"{'='*60}")
    print(f"原引擎交易数: {len(original_trades)}")
    print(f"BT引擎交易数: {len(bt_trades)}")

    if len(original_trades) == 0 or len(bt_trades) == 0:
        print("  数据不足, 无法对比")
        return

    # 按 (code, buy_date) 分组原引擎交易
    orig_map = {}
    for t in original_trades:
        key = (t['code'], t['buy_date'][:10])
        orig_map[key] = t

    # 统计匹配情况
    matched = 0
    unmatched_bt = []

    for t in bt_trades:
        code = t['code']
        # BT 的日期格式可能不同
        buy_date = str(t.get('buy_date', ''))[:10]
        key = (code, buy_date)

        if key in orig_map:
            matched += 1
            orig = orig_map[key]
            # 对比买入价
            orig_bp = orig['buy_price']
            bt_bp = t.get('buy_price', 0)
            if abs(orig_bp - bt_bp) > 0.01:
                print(f"  买入价差异 {code}@{buy_date}: "
                      f"原={orig_bp:.2f}, BT={bt_bp:.2f}")
        else:
            unmatched_bt.append(t)

    print(f"\n匹配: {matched}/{len(bt_trades)} ({matched/len(bt_trades)*100:.1f}%)")
    if unmatched_bt:
        print(f"BT独有: {len(unmatched_bt)} 笔")
        for t in unmatched_bt[:10]:
            print(f"  {t['code']} @ {t.get('buy_date', '?')}")


def compare_stats(original_meta, bt_strat):
    """对比统计指标"""
    print(f"\n{'='*60}")
    print("统计指标对比")
    print(f"{'='*60}")

    # 原引擎指标
    orig_return = original_meta['total_return']
    orig_dd = original_meta.get('max_dd', 0)
    orig_trades = original_meta.get('trade_count', 0)
    orig_winrate = original_meta.get('win_rate', 0)

    # BT引擎指标
    bt_value = bt_strat.broker.getvalue()
    bt_init = bt_strat.broker.startingcash
    bt_return = (bt_value / bt_init - 1) * 100
    bt_trades = len(bt_strat.trade_log)

    try:
        dd = bt_strat.analyzers.drawdown.get_analysis()
        bt_dd = dd.max.drawdown
    except Exception:
        bt_dd = 0

    try:
        ta = bt_strat.analyzers.trades.get_analysis()
        won = ta.won.total if hasattr(ta.won, 'total') else 0
        bt_winrate = won / bt_trades * 100 if bt_trades > 0 else 0
    except Exception:
        bt_winrate = 0

    fmt = "{:>16s} {:>12s} {:>12s} {:>12s}"
    print(fmt.format("指标", "原引擎", "BT引擎", "差异"))
    print("-" * 56)

    print(fmt.format(
        "总收益率",
        f"{orig_return:+.1f}%",
        f"{bt_return:+.1f}%",
        f"{bt_return - orig_return:+.1f}pp"
    ))
    print(fmt.format(
        "最大回撤",
        f"{orig_dd:.1f}%",
        f"{bt_dd:.1f}%",
        f"{bt_dd - orig_dd:+.1f}pp"
    ))
    print(fmt.format(
        "交易笔数",
        f"{orig_trades}",
        f"{bt_trades}",
        f"{bt_trades - orig_trades:+d}"
    ))
    print(fmt.format(
        "胜率",
        f"{orig_winrate:.1f}%",
        f"{bt_winrate:.1f}%",
        f"{bt_winrate - orig_winrate:+.1f}pp"
    ))

    # 差异评估
    print(f"\n差异评估:")
    ret_diff = abs(bt_return - orig_return)
    if ret_diff < 5:
        print(f"  总收益率偏差 {ret_diff:.1f}pp — 注意: 卖出价差异(原=收盘, BT=次日开盘)会导致系统性偏差")
    else:
        print(f"  ⚠ 总收益率偏差 {ret_diff:.1f}pp 较大, 需检查逻辑")


def main():
    """运行对比验证"""
    print("=" * 60)
    print("对比验证: 原引擎 vs Backtrader")
    print("=" * 60)

    # 1. 加载原引擎结果
    print("\n[Step 1] 加载原引擎结果...")
    orig = load_original_results()
    if orig is None:
        return

    orig_meta = orig['meta']
    orig_trades = orig.get('trade_log', [])
    print(f"  原引擎: 收益={orig_meta['total_return']:+.1f}%, "
          f"交易={orig_meta['trade_count']}笔")

    # 2. 运行 Backtrader
    print("\n[Step 2] 运行 Backtrader 回测 (零佣金模式)...")
    cerebro, strat = run_bt_backtest(no_commission=True, verbose=False)

    # 3. 对比统计
    compare_stats(orig_meta, strat)

    # 4. 对比交易记录 (如果BT有详细交易日志)
    # 注意: BT 的 trade_log 格式与原引擎不同
    # 后续可以增强策略的 notify_trade 来记录更多信息
    print(f"\nBT引擎完成: {len(strat.trade_log)} 笔交易")


if __name__ == '__main__':
    main()
