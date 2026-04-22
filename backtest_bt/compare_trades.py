# -*- coding: utf-8 -*-
"""
compare_trades.py — 逐笔对比原引擎 vs BT引擎的交易记录

找出差异的根本原因:
1. BT有但原引擎没有的交易 (多买)
2. 原引擎有但BT没有的交易 (漏买)
3. 同一笔交易的卖出日期/价格差异
"""
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_original_trades():
    """加载原引擎 3仓_等分 的交易记录"""
    # 先运行原引擎获取 3仓_等分 的结果
    from backtest_capital import (
        load_zz1000_dual, load_stocks,
        scan_signals_with_sell, simulate_capital
    )

    print("加载原引擎数据...")
    zz1000 = load_zz1000_dual()
    stock_data = load_stocks()
    print(f"  个股: {len(stock_data)} 只")

    print("扫描信号...")
    all_dates, daily_candidates = scan_signals_with_sell(
        stock_data, zz1000, '2015-01-01', '2026-04-01')

    print("模拟 3仓_等分...")
    cfg = {
        'init_capital': 200000,
        'max_positions': 3,
        'daily_buy_limit': 3,
        'position_mode': 'equal',
    }
    result = simulate_capital(all_dates, daily_candidates, cfg)
    print(f"  原引擎: 收益={result['total_return']:+.1f}%, "
          f"交易={result['trade_count']}笔")
    return result['trade_log']


def load_bt_trades():
    """运行BT引擎获取交易记录"""
    from backtest_bt.engine import build_cerebro, run_backtest

    print("\n运行BT引擎 (零佣金)...")
    cerebro = build_cerebro(commission=False, verbose=False)
    strat = run_backtest(cerebro)

    trades = strat.trade_log
    print(f"  BT引擎: 交易={len(trades)}笔")
    return trades


def compare(orig_trades, bt_trades):
    """逐笔对比"""
    print(f"\n{'='*80}")
    print("逐笔对比")
    print(f"{'='*80}")
    print(f"原引擎: {len(orig_trades)} 笔")
    print(f"BT引擎: {len(bt_trades)} 笔")

    # 按 (code, buy_date) 建索引
    orig_map = {}
    for t in orig_trades:
        key = (t['code'], t['buy_date'][:10])
        orig_map[key] = t

    bt_map = {}
    for t in bt_trades:
        key = (t['code'], t['buy_date'][:10])
        bt_map[key] = t

    # 匹配
    matched = []
    bt_only = []
    orig_only = []

    for key, bt in bt_map.items():
        if key in orig_map:
            matched.append((orig_map[key], bt))
        else:
            bt_only.append(bt)

    for key, orig in orig_map.items():
        if key not in bt_map:
            orig_only.append(orig)

    print(f"\n匹配: {len(matched)} 笔")
    print(f"BT独有: {len(bt_only)} 笔")
    print(f"原引擎独有: {len(orig_only)} 笔")

    # 分析匹配的交易
    if matched:
        print(f"\n--- 匹配交易的差异 ---")
        buy_price_diffs = []
        sell_price_diffs = []
        sell_date_diffs = 0
        for orig, bt in matched:
            bp_diff = bt['buy_price'] - orig['buy_price']
            buy_price_diffs.append(bp_diff)

            orig_sell = orig['sell_date'][:10]
            bt_sell = bt.get('sell_date', '')[:10]
            if orig_sell != bt_sell:
                sell_date_diffs += 1

            sp_diff = bt.get('sell_price', 0) - orig['sell_price']
            sell_price_diffs.append(sp_diff)

        import numpy as np
        bp_arr = np.array(buy_price_diffs)
        sp_arr = np.array(sell_price_diffs)
        print(f"  买入价差异: 均值={np.mean(bp_arr):.4f}, "
              f"最大={np.max(np.abs(bp_arr)):.4f}, "
              f"完全一致={np.sum(np.abs(bp_arr) < 0.01)}/{len(bp_arr)}")
        print(f"  卖出日期不同: {sell_date_diffs}/{len(matched)}")
        print(f"  卖出价差异: 均值={np.mean(sp_arr):.2f}, "
              f"最大={np.max(np.abs(sp_arr)):.2f}")

        # 打印卖出日期不同的前10笔
        if sell_date_diffs > 0:
            print(f"\n  卖出日期不同的交易 (前20笔):")
            count = 0
            for orig, bt in matched:
                orig_sell = orig['sell_date'][:10]
                bt_sell = bt.get('sell_date', '')[:10]
                if orig_sell != bt_sell:
                    orig_ret = orig.get('ret_pct', 0)
                    bt_pnl = bt.get('pnl', 0)
                    print(f"    {orig['code']} 买入={orig['buy_date'][:10]}: "
                          f"原卖={orig_sell} BT卖={bt_sell} "
                          f"原收益={orig_ret:+.1f}% BT盈亏={bt_pnl:+.0f}")
                    count += 1
                    if count >= 20:
                        break

    # 打印BT独有的交易
    if bt_only:
        print(f"\n--- BT独有交易 (前20笔) ---")
        for t in bt_only[:20]:
            print(f"  {t['code']} 买入={t['buy_date']} "
                  f"卖出={t.get('sell_date', '?')} "
                  f"盈亏={t.get('pnl', 0):+.0f}")

    # 打印原引擎独有的交易
    if orig_only:
        print(f"\n--- 原引擎独有交易 (前20笔) ---")
        for t in orig_only[:20]:
            print(f"  {t['code']} 买入={t['buy_date'][:10]} "
                  f"卖出={t['sell_date'][:10]} "
                  f"收益={t.get('ret_pct', 0):+.1f}%")

    # 汇总盈亏
    orig_total_pnl = sum(t.get('profit', 0) for t in orig_trades)
    bt_total_pnl = sum(t.get('pnl', 0) for t in bt_trades)
    print(f"\n--- 盈亏汇总 ---")
    print(f"  原引擎总盈亏: {orig_total_pnl:+,.0f}")
    print(f"  BT引擎总盈亏: {bt_total_pnl:+,.0f}")


if __name__ == '__main__':
    orig_trades = load_original_trades()
    bt_trades = load_bt_trades()
    compare(orig_trades, bt_trades)
