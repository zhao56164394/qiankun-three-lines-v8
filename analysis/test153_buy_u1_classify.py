# -*- coding: utf-8 -*-
"""九安医疗 24 腿: 分析每个 buy-U1 是"聪明"还是"傻"
找出区分依据 (mf 强弱? trend 高低? retail 状态? 价格相对位置?)
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

    # 模拟 D6+U1, 记录每次 buy-U1 时的状态 + 上次卖价
    holding = True
    cur_buy_price = bp_first
    legs_info = []  # (action, date, price, mf, retail, trend, last_sell_price, last_sell_mf, last_sell_retail, last_sell_trend, days_since_sell)
    last_sell_idx = -1
    last_sell_info = None

    legs_info.append(('buy', sub['date'].iloc[0], bp_first,
                       sub['main_force'].iloc[0], sub['retail'].iloc[0], sub['d_trend'].iloc[0],
                       None, None, None, None, None, None))

    for k in range(1, len(sub)):
        r = sub.iloc[k]
        td = r['d_trend']
        td_prev = sub['d_trend'].iloc[k-1]
        mf_c = r['main_force'] - sub['main_force'].iloc[k-1]
        ret_c = r['retail'] - sub['retail'].iloc[k-1]
        td_c = td - td_prev if not pd.isna(td_prev) else 0

        if not pd.isna(td) and td < 11:
            if holding:
                legs_info.append(('sell-td<11', r['date'], r['close'],
                                    r['main_force'], r['retail'], td,
                                    cur_buy_price, None, None, None, None, None))
            break

        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                legs_info.append(('sell-D6', r['date'], r['close'],
                                    r['main_force'], r['retail'], td,
                                    cur_buy_price, None, None, None, None, None))
                holding = False
                last_sell_idx = k
                last_sell_info = (r['date'], r['close'], r['main_force'], r['retail'], td)
        else:
            if mf_c > 0:
                # 记录 buy-U1 状态
                days_since = k - last_sell_idx
                ls_date, ls_price, ls_mf, ls_ret, ls_td = last_sell_info
                price_rel_to_sell = (r['close'] / ls_price - 1) * 100
                legs_info.append(('buy-U1', r['date'], r['close'],
                                    r['main_force'], r['retail'], td,
                                    ls_price, ls_mf, ls_ret, ls_td,
                                    days_since, price_rel_to_sell))
                cur_buy_price = r['close']
                holding = True

    # 配对每个 buy-U1 与下一次 sell, 计算本腿 ret
    pairs = []
    for i in range(1, len(legs_info)):
        if legs_info[i][0] == 'buy-U1' and i+1 < len(legs_info) and legs_info[i+1][0].startswith('sell'):
            buy = legs_info[i]
            sell = legs_info[i+1]
            this_ret = (sell[2] / buy[2] - 1) * 100
            pairs.append({
                'leg_no': i + 1,
                'buy_date': buy[1],
                'buy_price': buy[2],
                'buy_mf': buy[3],
                'buy_retail': buy[4],
                'buy_trend': buy[5],
                'last_sell_date': buy[1],  # placeholder
                'last_sell_price': buy[6],
                'last_sell_mf': buy[7],
                'last_sell_retail': buy[8],
                'last_sell_trend': buy[9],
                'days_since_sell': buy[10],
                'price_rel_to_sell%': buy[11],
                'next_sell_date': sell[1],
                'next_sell_price': sell[2],
                'this_leg_ret%': this_ret,
                'smart': '⭐' if this_ret > 5 else ('⚠' if this_ret < 0 else '一般'),
            })

    df_p = pd.DataFrame(pairs)
    print(f'=== 11 个 buy-U1 详细 ===\n')
    print(f'{"评":<3} {"日期":<12} {"买价":>6} {"卖到买涨%":>9} {"距离":>4} {"mf":>5} '
          f'{"retail":>7} {"trend":>6} {"上次卖mf":>8} {"上次卖retail":>11} {"上次卖trend":>10} {"本腿ret%":>9}')
    for _, r in df_p.iterrows():
        print(f'{r["smart"]:<3} {r["buy_date"]:<12} {r["buy_price"]:>5.2f} '
              f'{r["price_rel_to_sell%"]:>+8.2f}% {r["days_since_sell"]:>3}d '
              f'{r["buy_mf"]:>+5.0f} {r["buy_retail"]:>+6.0f} {r["buy_trend"]:>+5.1f} '
              f'{r["last_sell_mf"]:>+8.0f} {r["last_sell_retail"]:>+10.0f} {r["last_sell_trend"]:>+9.1f} '
              f'{r["this_leg_ret%"]:>+8.2f}%')

    # 找区分聪明/傻的特征
    print(f'\n=== 聪明 (本腿>+5%) vs 傻 (本腿<0%) 的特征对比 ===\n')
    smart = df_p[df_p['this_leg_ret%'] > 5]
    dumb = df_p[df_p['this_leg_ret%'] < 0]
    print(f'  聪明 ({len(smart)} 笔), 傻 ({len(dumb)} 笔)\n')
    print(f'  特征             聪明 avg     傻 avg      差')
    for f in ['buy_mf', 'buy_retail', 'buy_trend', 'last_sell_mf',
                'last_sell_retail', 'last_sell_trend', 'days_since_sell',
                'price_rel_to_sell%']:
        s_avg = smart[f].mean()
        d_avg = dumb[f].mean()
        print(f'  {f:<22} {s_avg:>+8.1f}    {d_avg:>+8.1f}   {s_avg-d_avg:>+8.1f}')

    # 检验候选条件
    print(f'\n=== 候选过滤条件检验 (在该 case 内) ===')
    print(f'  目标: 用条件 X 把 6 个傻笔过滤掉, 保留 5 个聪明笔\n')

    conditions = [
        ('mf > 50',                 lambda r: r['buy_mf'] > 50),
        ('mf > 100',                lambda r: r['buy_mf'] > 100),
        ('mf > 200',                lambda r: r['buy_mf'] > 200),
        ('retail > 0',              lambda r: r['buy_retail'] > 0),
        ('trend > 50',              lambda r: r['buy_trend'] > 50),
        ('trend > 70',              lambda r: r['buy_trend'] > 70),
        ('trend > 89',              lambda r: r['buy_trend'] > 89),
        ('mf > 0 AND trend > 50',   lambda r: r['buy_mf'] > 0 and r['buy_trend'] > 50),
        ('mf > 50 AND trend > 70',  lambda r: r['buy_mf'] > 50 and r['buy_trend'] > 70),
        ('上次卖 mf > 0',            lambda r: r['last_sell_mf'] > 0),
        ('mf > 上次卖 mf',           lambda r: r['buy_mf'] > r['last_sell_mf']),
        ('mf > 上次卖 mf 且 mf > 0', lambda r: r['buy_mf'] > r['last_sell_mf'] and r['buy_mf'] > 0),
        ('retail > 上次卖 retail',    lambda r: r['buy_retail'] > r['last_sell_retail']),
        ('买价 < 上次卖价',           lambda r: r['price_rel_to_sell%'] < 0),
    ]

    print(f'  {"条件":<28} {"通过聪明":>9} {"通过傻":>9} {"过滤后 ret":>12}')
    for label, fn in conditions:
        pass_smart = sum(1 for _, r in smart.iterrows() if fn(r))
        pass_dumb = sum(1 for _, r in dumb.iterrows() if fn(r))
        # 过滤后保留这些笔, 算累计 ret
        kept = df_p[df_p.apply(fn, axis=1)]
        if len(kept) == 0:
            kept_ret = 0
        else:
            cum = 1.0
            for r_ in kept['this_leg_ret%']:
                cum *= (1 + r_/100)
            kept_ret = (cum - 1) * 100
        print(f'  {label:<28} {pass_smart:>5}/5    {pass_dumb:>5}/6     {kept_ret:>+8.2f}%')


if __name__ == '__main__':
    main()
