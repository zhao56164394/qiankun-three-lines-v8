# -*- coding: utf-8 -*-
"""
基线回测 — 回归最简策略

核心逻辑:
  1. 统一入池: 散户线 < -250
  2. 买入: 趋势线上穿11 (前日<11, 今日>=11)
  3. 卖出: 统一bear卖法
  4. 无等级过滤 — 先跑裸信号看各卦表现
  5. 分卦统计: 信号数、交易数、平均每日持仓
"""
import sys, os, io, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)

from backtest_capital import (
    load_zz1000, load_zz1000_full, load_stocks,
    calc_sell_bear,
    YEAR_START, YEAR_END, INIT_CAPITAL, DATA_DIR,
)
from data_layer.gua_data import clean_gua as _clean_gua, GUA_NAMES

POOL_THRESHOLD = -250
POOL_THRESHOLD_BY_GUA = {
    '000': -250,
    '001': -250,
    '010': -250,
    '011': -250,
    '100': -250,
    '101': -250,
    '110': -250,
    '111': -250,
}


# ============================================================
# 信号扫描 — 趋势线上穿11 + 统一bear卖
# ============================================================
def scan_signals_baseline(stock_data, ren_gua_map):
    """
    入池: 按中证目标卦读取独立初始入池阈值
    买入: 趋势线上穿11 (trend[i-1] < 11 and trend[i] >= 11)
    卖出: bear卖法(从买入点开始算)
    上穿触发 → 出池清零
    """
    all_signals = []

    for code, df in stock_data.items():
        if len(df) < 35:
            continue
        dates = df['date'].values
        trend = df['trend'].values
        retail = df['retail'].values
        closes = df['close'].values
        opens = df['open'].values

        pooled = False
        pool_retail = 0

        for i in range(1, len(df)):
            dt_str = str(dates[i])
            ren_gua = ren_gua_map.get(dt_str, '???')

            current_pool_threshold = POOL_THRESHOLD_BY_GUA.get(ren_gua, POOL_THRESHOLD)

            # --- 入池逻辑 ---
            if not pooled:
                if not np.isnan(retail[i]) and retail[i] < current_pool_threshold:
                    pooled = True
                    pool_retail = retail[i]
                if pooled and not np.isnan(retail[i]):
                    pool_retail = min(pool_retail, retail[i])
            else:
                if not np.isnan(retail[i]):
                    pool_retail = min(pool_retail, retail[i])

            if not pooled:
                continue

            # --- 趋势线上穿11 ---
            if np.isnan(trend[i]) or np.isnan(trend[i-1]):
                continue
            if not (trend[i-1] < 11 and trend[i] >= 11):
                continue

            # 上穿11触发 → 出池
            next_idx = i + 1
            if next_idx >= len(df):
                pooled = False
                pool_retail = 0
                continue

            buy_price = opens[next_idx]
            if buy_price <= 0 or np.isnan(buy_price):
                pooled = False
                pool_retail = 0
                continue

            # bear卖法(从信号日i开始)
            _, sell_idx = calc_sell_bear(df, i)
            sell_date = dates[sell_idx] if sell_idx < len(dates) else dates[-1]
            sell_price = closes[sell_idx]
            hold_days = sell_idx - next_idx

            if hold_days <= 0:
                pooled = False
                pool_retail = 0
                continue

            di_gua = _clean_gua(df['gua'].iloc[i])

            all_signals.append({
                'code': code,
                'signal_date': dt_str,
                'buy_date': str(dates[next_idx]),
                'sell_date': str(sell_date),
                'buy_price': buy_price,
                'sell_price': sell_price,
                'actual_ret': (sell_price / buy_price - 1) * 100,
                'hold_days': hold_days,
                'pool_retail': pool_retail,
                'ren_gua': ren_gua,
                'di_gua': di_gua,
            })

            # 上穿触发 → 出池清零
            pooled = False
            pool_retail = 0

    return pd.DataFrame(all_signals).sort_values('signal_date').reset_index(drop=True)


# ============================================================
# 模拟引擎 — 统一策略, 记录分卦持仓
# ============================================================
def simulate_baseline(sig_df, zz_df, max_pos=5, daily_limit=1, init_capital=None):
    capital = init_capital or INIT_CAPITAL

    # 构建中证象卦映射
    ren_gua_map = {}
    for _, row in zz_df.iterrows():
        ren_gua_map[row['date']] = _clean_gua(row['gua'])

    sig_by_date = {}
    for _, row in sig_df.iterrows():
        sig_by_date.setdefault(row['signal_date'], []).append(row)

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
                    'ren_gua': pos.get('ren_gua', '???'),
                    'di_gua': pos.get('di_gua', '???'),
                })
            else:
                new_pos.append(pos)
        positions = new_pos

        # 2. 买入 — 无等级过滤, 按pool_retail排序
        candidates = sig_by_date.get(dt, [])
        if candidates:
            candidates = sorted(candidates, key=lambda x: (-int(0 if pd.isna(x.get('rank_order')) else x.get('rank_order', 0)), x['pool_retail']))
            slots = max_pos - len(positions)
            can_buy = min(slots, daily_limit, len(candidates))
            if can_buy > 0 and capital > 1000:
                total_eq = capital + sum(p['cost'] for p in positions)
                per_slot = total_eq / max_pos
                per_buy = min(per_slot, capital / can_buy)
                for j in range(can_buy):
                    cost = min(per_buy, capital)
                    if cost < 1000:
                        break
                    c = candidates[j]
                    capital -= cost
                    positions.append({
                        'code': c['code'], 'buy_date': c['buy_date'],
                        'sell_date': c['sell_date'], 'buy_price': c['buy_price'],
                        'sell_price': c['sell_price'], 'cost': cost,
                        'hold_days': c['hold_days'],
                        'ren_gua': c.get('ren_gua', '???'),
                        'di_gua': c.get('di_gua', '???'),
                    })

        # 3. 记录净值 + 当日持仓卦分布
        hold_val = sum(p['cost'] for p in positions)
        daily_equity.append({
            'date': dt, 'cash': capital, 'hold_value': hold_val,
            'total_equity': capital + hold_val, 'n_positions': len(positions),
            'ren_gua': ren_gua_map.get(dt, '???'),
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
            'ren_gua': pos.get('ren_gua', '???'),
            'di_gua': pos.get('di_gua', '???'),
        })

    _init = init_capital or INIT_CAPITAL
    return {
        'final_capital': capital, 'init_capital': _init,
        'total_return': (capital / _init - 1) * 100,
        'trade_log': trade_log, 'daily_equity': daily_equity,
    }


# ============================================================
# 主流程
# ============================================================
def run(start_date=None, end_date=None, init_capital=None):
    year_start = start_date or YEAR_START
    year_end = end_date or YEAR_END
    capital = init_capital or INIT_CAPITAL

    print("=" * 100)
    print("  基线回测 — 每卦独立初始入池阈值 + 上穿11买 + bear卖 + 无过滤")
    print(f"  区间: {year_start} ~ {year_end}  初始资金: {capital:,}")
    print("=" * 100)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    zz_df = load_zz1000_full()
    stock_data = load_stocks()
    print(f"  个股: {len(stock_data)} 只")

    # 构建中证象卦映射
    ren_gua_map = {}
    for _, row in zz_df.iterrows():
        ren_gua_map[row['date']] = _clean_gua(row['gua'])

    # 2. 扫描信号
    print("\n[2] 扫描基线信号 (上穿11 + bear卖)...")
    sig = scan_signals_baseline(stock_data, ren_gua_map)
    sig = sig[(sig['signal_date'] >= year_start) &
              (sig['signal_date'] < year_end)].reset_index(drop=True)
    print(f"  总信号: {len(sig)}")

    # 分卦信号统计
    print(f"\n  === 分卦信号分布(信号日中证象卦) ===")
    print(f"  {'卦':<12} {'信号数':>6} {'均收益%':>8} {'中位收益%':>9} {'胜率%':>6} {'均持仓天':>8}")
    print("  " + "-" * 56)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub = sig[sig['ren_gua'] == gua]
        if len(sub) == 0:
            print(f"  {gua} {GUA_NAMES[gua]:<6} {0:>6}")
            continue
        avg_r = sub['actual_ret'].mean()
        med_r = sub['actual_ret'].median()
        wr = (sub['actual_ret'] > 0).mean() * 100
        avg_h = sub['hold_days'].mean()
        print(f"  {gua} {GUA_NAMES[gua]:<6} {len(sub):>6} {avg_r:>+7.2f} {med_r:>+8.2f} {wr:>5.1f} {avg_h:>7.1f}")

    # 分个股象卦信号统计
    print(f"\n  === 分卦信号分布(个股象卦) ===")
    print(f"  {'卦':<12} {'信号数':>6} {'均收益%':>8} {'中位收益%':>9} {'胜率%':>6}")
    print("  " + "-" * 48)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub = sig[sig['di_gua'] == gua]
        if len(sub) == 0:
            print(f"  {gua} {GUA_NAMES[gua]:<6} {0:>6}")
            continue
        avg_r = sub['actual_ret'].mean()
        med_r = sub['actual_ret'].median()
        wr = (sub['actual_ret'] > 0).mean() * 100
        print(f"  {gua} {GUA_NAMES[gua]:<6} {len(sub):>6} {avg_r:>+7.2f} {med_r:>+8.2f} {wr:>5.1f}")

    # 中证卦 x 个股卦 交叉统计
    print(f"\n  === 中证卦 x 个股卦 交叉统计 (均收益%) ===")
    gua_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    col_label = 'zz\\stk'
    header = f"  {col_label:<8}" + "".join(f"  {GUA_NAMES[g]:>5}" for g in gua_list) + f"  {'合计':>6}"
    print(header)
    print("  " + "-" * (8 + 7 * 8 + 8))
    for zg in gua_list:
        row = f"  {GUA_NAMES[zg]:<8}"
        zz_sub = sig[sig['ren_gua'] == zg]
        for sg in gua_list:
            cross = zz_sub[zz_sub['di_gua'] == sg]
            if len(cross) >= 3:
                row += f"  {cross['actual_ret'].mean():>+5.1f}"
            elif len(cross) > 0:
                row += f"  ({cross['actual_ret'].mean():>+4.0f})"
            else:
                row += f"  {'--':>5}"
        if len(zz_sub) > 0:
            row += f"  {zz_sub['actual_ret'].mean():>+5.1f}"
        else:
            row += f"  {'--':>6}"
        print(row)

    # 3. 模拟(无过滤, 全进全出)
    print(f"\n[3] 模拟 (5仓, 每日限买1笔, 无过滤)...")
    result = simulate_baseline(sig, zz_df, max_pos=5, daily_limit=1,
                               init_capital=capital)
    trades = result['trade_log']
    eq = result['daily_equity']

    # 基础统计
    if trades:
        rets = [t['ret_pct'] for t in trades]
        wins = [t for t in trades if t['profit'] > 0]
        peak = capital
        max_dd = 0
        max_dd_date = ''
        for e in eq:
            if e['total_equity'] > peak:
                peak = e['total_equity']
            dd = (peak - e['total_equity']) / peak * 100
            if dd > max_dd:
                max_dd = dd
                max_dd_date = e['date']
        avg_pos = np.mean([e['n_positions'] for e in eq]) if eq else 0
    else:
        rets = []
        wins = []
        max_dd = 0
        max_dd_date = ''
        avg_pos = 0

    print(f"\n{'=' * 100}")
    print(f"  Part1: 策略总览")
    print(f"{'=' * 100}")
    print(f"\n  初始资金: {capital:,}")
    print(f"  终值: {result['final_capital']:,.0f}")
    print(f"  收益: {result['total_return']:.1f}%")
    print(f"  最大回撤: {max_dd:.1f}% ({max_dd_date})")
    print(f"  交易笔数: {len(trades)}")
    print(f"  胜率: {len(wins)/len(trades)*100:.1f}%" if trades else "  胜率: N/A")
    print(f"  均收益: {np.mean(rets):.2f}%" if rets else "  均收益: N/A")
    print(f"  中位收益: {np.median(rets):.2f}%" if rets else "  中位收益: N/A")
    print(f"  均持仓天: {np.mean([t['hold_days'] for t in trades]):.1f}" if trades else "")
    print(f"  平均每日持仓: {avg_pos:.2f}")

    # Part2: 分卦交易统计(中证象卦)
    print(f"\n{'=' * 100}")
    print(f"  Part2: 分卦交易统计(中证象卦 = 买入当天中证1000所处卦)")
    print(f"{'=' * 100}")
    total_profit = sum(t['profit'] for t in trades)
    print(f"\n  {'卦':<12} {'信号数':>6} {'交易数':>6} {'胜率%':>6} {'均收益%':>8} {'中位%':>7} "
          f"{'利润':>14} {'占比%':>6} {'均持仓':>6} {'日均仓':>6}")
    print("  " + "-" * 90)
    for gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sig_count = len(sig[sig['ren_gua'] == gua])
        gua_trades = [t for t in trades if t.get('ren_gua') == gua]
        if not gua_trades:
            print(f"  {gua} {GUA_NAMES[gua]:<6} {sig_count:>6} {0:>6}")
            continue
        n = len(gua_trades)
        w = sum(1 for t in gua_trades if t['profit'] > 0)
        avg_r = np.mean([t['ret_pct'] for t in gua_trades])
        med_r = np.median([t['ret_pct'] for t in gua_trades])
        profit = sum(t['profit'] for t in gua_trades)
        pct = profit / total_profit * 100 if total_profit != 0 else 0
        avg_h = np.mean([t['hold_days'] for t in gua_trades])
        # 该卦的日均持仓: 该卦交易总持仓天数 / 总交易天数
        total_hold_days = sum(t['hold_days'] for t in gua_trades)
        total_trading_days = len(eq) if eq else 1
        daily_pos = total_hold_days / total_trading_days
        print(f"  {gua} {GUA_NAMES[gua]:<6} {sig_count:>6} {n:>6} {w/n*100:>5.1f} {avg_r:>+7.2f} "
              f"{med_r:>+6.2f} {profit:>13,.0f} {pct:>5.1f} {avg_h:>5.1f} {daily_pos:>5.2f}")
    # 汇总行
    if trades:
        total_hold = sum(t['hold_days'] for t in trades)
        total_trading_days = len(eq) if eq else 1
        print(f"  {'合计':<10} {len(sig):>6} {len(trades):>6} "
              f"{len(wins)/len(trades)*100:>5.1f} {np.mean(rets):>+7.2f} "
              f"{np.median(rets):>+6.2f} {total_profit:>13,.0f} {'100.0':>5} "
              f"{np.mean([t['hold_days'] for t in trades]):>5.1f} "
              f"{total_hold/total_trading_days:>5.2f}")

    # Part3: 年度明细
    print(f"\n{'=' * 100}")
    print(f"  Part3: 年度明细")
    print(f"{'=' * 100}")
    yearly = {}
    for t in trades:
        y = t['buy_date'][:4]
        yearly.setdefault(y, {'profit': 0, 'count': 0, 'wins': 0})
        yearly[y]['profit'] += t['profit']
        yearly[y]['count'] += 1
        if t['profit'] > 0:
            yearly[y]['wins'] += 1

    print(f"\n  {'年份':<6} {'笔数':>5} {'盈利笔':>5} {'胜率%':>6} {'利润':>14}")
    print("  " + "-" * 40)
    for y in sorted(yearly.keys()):
        v = yearly[y]
        wr = v['wins'] / v['count'] * 100 if v['count'] > 0 else 0
        print(f"  {y:<6} {v['count']:>5} {v['wins']:>5} {wr:>5.1f} {v['profit']:>13,.0f}")

    # Part4: 分卦 x 年度明细(利润)
    print(f"\n{'=' * 100}")
    print(f"  Part4: 分卦x年度 (利润)")
    print(f"{'=' * 100}")
    years = sorted(yearly.keys())
    gua_list = ['000', '001', '010', '011', '100', '101', '110', '111']
    header = f"  {'卦':<10}" + "".join(f"{y:>10}" for y in years) + f"{'合计':>12}"
    print(f"\n{header}")
    print("  " + "-" * (10 + 10 * len(years) + 12))
    for gua in gua_list:
        gua_trades = [t for t in trades if t.get('ren_gua') == gua]
        row = f"  {gua} {GUA_NAMES[gua]:<5}"
        total = 0
        for y in years:
            yt = [t for t in gua_trades if t['buy_date'][:4] == y]
            p = sum(t['profit'] for t in yt)
            total += p
            row += f"{p:>+10,.0f}" if yt else f"{'':>10}"
        row += f"{total:>+12,.0f}"
        print(row)

    # Part5: 分卦 x 年度 (笔数)
    print(f"\n  分卦x年度 (笔数)")
    header2 = f"  {'卦':<10}" + "".join(f"{y:>6}" for y in years) + f"{'合计':>8}"
    print(header2)
    print("  " + "-" * (10 + 6 * len(years) + 8))
    for gua in gua_list:
        gua_trades = [t for t in trades if t.get('ren_gua') == gua]
        row = f"  {gua} {GUA_NAMES[gua]:<5}"
        total_n = 0
        for y in years:
            yt = [t for t in gua_trades if t['buy_date'][:4] == y]
            row += f"{len(yt):>6}" if yt else f"{'':>6}"
            total_n += len(yt)
        row += f"{total_n:>8}"
        print(row)

    print(f"\n{'=' * 100}")
    print(f"  基线策略: 每卦独立初始入池阈值 + 上穿11 + bear卖 + 无过滤")
    print(f"  {capital:,} → {result['final_capital']:,.0f} ({result['total_return']:.1f}%)")
    print(f"  {len(trades)}笔交易, 胜率{len(wins)/len(trades)*100:.1f}%, "
          f"均收益{np.mean(rets):.2f}%, 最大回撤{max_dd:.1f}%" if trades else "")
    print(f"  平均每日持仓: {avg_pos:.2f}")
    print(f"{'=' * 100}")

    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='基线回测')
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--capital', type=int, default=None)
    args = parser.parse_args()
    run(start_date=args.start, end_date=args.end, init_capital=args.capital)
