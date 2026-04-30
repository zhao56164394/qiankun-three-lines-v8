# -*- coding: utf-8 -*-
"""九安医疗 002432 D6+U1 24 腿完整轨迹 + 每腿盈亏分析

仓位守恒: 假设 100K 资金, 每次卖出后保留全部资金, 下次买入用全部资金
"""
import os, sys, io
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'))
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p['date'] = p['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = g['date'].astype(str)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    df = p.merge(g, on=['code', 'date'], how='left')

    code = '002432'
    buy_date = '2021-11-03'
    sub = df[df['code'] == code].sort_values('date').reset_index(drop=True)
    bidx = sub[sub['date'] == buy_date].index[0]
    sub = sub.iloc[bidx:bidx+200].reset_index(drop=True)

    bp_first = sub['close'].iloc[0]

    # 模拟 D6+U1, 仓位守恒 100K
    capital = 100_000
    holding = True
    cur_buy_price = bp_first
    cur_shares = capital / bp_first  # 不取整, 看百分比
    cum_pnl_pct = 0  # 累计 PnL (%)
    cash = 0
    legs_log = []
    legs_log.append({
        'leg': 1, 'date': sub['date'].iloc[0], 'action': 'buy',
        'price': bp_first, 'cur_buy': bp_first, 'cum_ret%': 0.0,
        'mf': sub['main_force'].iloc[0], 'retail': sub['retail'].iloc[0], 'trend': sub['d_trend'].iloc[0]
    })

    cum_mult = 1.0
    legs = 1
    end_idx = None
    end_reason = None

    for k in range(1, len(sub)):
        r = sub.iloc[k]
        td = r['d_trend']
        td_prev = sub['d_trend'].iloc[k-1]
        mf_c = r['main_force'] - sub['main_force'].iloc[k-1]
        ret_c = r['retail'] - sub['retail'].iloc[k-1]
        td_c = td - td_prev if not pd.isna(td_prev) else 0

        # trend<11 终结
        if not pd.isna(td) and td < 11:
            if holding:
                cum_mult *= r['close'] / cur_buy_price
                legs += 1
                legs_log.append({
                    'leg': legs, 'date': r['date'], 'action': 'sell-td<11',
                    'price': r['close'], 'cur_buy': cur_buy_price,
                    'cum_ret%': (cum_mult-1)*100,
                    'mf': r['main_force'], 'retail': r['retail'], 'trend': td
                })
            end_idx = k
            end_reason = 'td<11'
            break

        # D6: 三线齐降
        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= r['close'] / cur_buy_price
                legs += 1
                holding = False
                legs_log.append({
                    'leg': legs, 'date': r['date'], 'action': 'sell-D6',
                    'price': r['close'], 'cur_buy': cur_buy_price,
                    'cum_ret%': (cum_mult-1)*100,
                    'mf': r['main_force'], 'retail': r['retail'], 'trend': td
                })
        else:
            # U1: mf 上升
            if mf_c > 0:
                cur_buy_price = r['close']
                holding = True
                legs += 1
                legs_log.append({
                    'leg': legs, 'date': r['date'], 'action': 'buy-U1',
                    'price': r['close'], 'cur_buy': cur_buy_price,
                    'cum_ret%': (cum_mult-1)*100,
                    'mf': r['main_force'], 'retail': r['retail'], 'trend': td
                })

    # 输出每腿
    print(f'=== 九安医疗 002432 D6+U1 完整 {legs} 腿轨迹 ===\n')
    print(f'入场: {sub["date"].iloc[0]}, 入场价 {bp_first:.2f}\n')
    print(f'{"腿":>3} {"日期":<12} {"动作":<14} {"价格":>7} {"上次买":>7} {"本腿ret%":>9} '
          f'{"累计ret%":>10} {"mf":>5} {"retail":>6} {"trend":>6}')
    for i, l in enumerate(legs_log):
        # 本腿 ret
        if l['action'].startswith('sell'):
            this_ret = (l['price'] / l['cur_buy'] - 1) * 100
        else:
            this_ret = 0
        print(f'{l["leg"]:>3} {l["date"]:<12} {l["action"]:<14} {l["price"]:>7.2f} '
              f'{l["cur_buy"]:>7.2f} {this_ret:>+8.2f}% {l["cum_ret%"]:>+9.2f}% '
              f'{l["mf"]:>+5.0f} {l["retail"]:>+5.0f} {l["trend"]:>+5.1f}')

    print(f'\n=== 段终结: {sub["date"].iloc[end_idx] if end_idx else "未终结"} ({end_reason}) ===')
    print(f'最终累计 ret = {(cum_mult-1)*100:+.2f}%')

    # 配对分析: 每对"卖→买"的价差
    print(f'\n=== 卖→买 配对分析 ===\n')
    print(f'{"卖":<14} {"卖价":>7} {"买":<14} {"买价":>7} {"价差":>7} {"价差%":>7}')
    pairs = []
    for i in range(1, len(legs_log)):
        if legs_log[i]['action'].startswith('sell') and i+1 < len(legs_log) and legs_log[i+1]['action'] == 'buy-U1':
            sell_price = legs_log[i]['price']
            buy_price = legs_log[i+1]['price']
            diff = buy_price - sell_price
            diff_pct = (buy_price/sell_price - 1) * 100
            pairs.append((legs_log[i]['date'], sell_price, legs_log[i+1]['date'], buy_price, diff, diff_pct))
            print(f'{legs_log[i]["date"]:<14} {sell_price:>6.2f}  {legs_log[i+1]["date"]:<14} {buy_price:>6.2f}  '
                  f'{diff:>+6.2f}  {diff_pct:>+6.2f}%')

    print(f'\n=== 配对统计 ===')
    if pairs:
        diffs_pct = [p[5] for p in pairs]
        print(f'  共 {len(pairs)} 对')
        print(f'  买回比卖价高 (亏价差): {sum(1 for d in diffs_pct if d>0)} 对')
        print(f'  买回比卖价低 (赚价差): {sum(1 for d in diffs_pct if d<0)} 对')
        print(f'  平均价差%: {sum(diffs_pct)/len(diffs_pct):+.2f}%')
        print(f'  累乘价差比 (理论): 用 (1 + diff%) 乘起来')
        cum = 1.0
        for d in diffs_pct:
            cum *= (1 + d/100)
        print(f'  累乘 (1+diff%) = {(cum-1)*100:+.2f}% (这是"切换损失")')
        print()
        print(f'  解释: 这个累乘表示"如果你能预知, 不卖不买, 直接持有"会比 D6+U1 多赚 X%')
        print(f'        baseline 是直接持有到 1-04 才卖, D6+U1 是反复切换累计损失这部分')


if __name__ == '__main__':
    main()
