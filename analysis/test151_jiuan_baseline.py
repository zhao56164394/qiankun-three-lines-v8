# -*- coding: utf-8 -*-
"""九安医疗 002432 baseline 卖出日详细
+ D6+U1 卖出日详细对比
"""
import os, sys, io
import numpy as np
import pandas as pd

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
    sub = sub.iloc[bidx:bidx+200].copy()  # 看 200 天
    bp = sub['close'].iloc[0]
    sub['ret_pct'] = (sub['close'] / bp - 1) * 100
    sub['mf_chg'] = sub['main_force'].diff()
    sub['retail_chg'] = sub['retail'].diff()
    sub['td_chg'] = sub['d_trend'].diff()

    # baseline 卖出模拟
    cross_count = 0
    running_max = sub['d_trend'].iloc[0]
    base_sell = None
    base_reason = None
    for k in range(1, len(sub)):
        td = sub['d_trend'].iloc[k]
        td_prev = sub['d_trend'].iloc[k-1]
        if not pd.isna(td):
            running_max = max(running_max, td)
        if not pd.isna(td) and td < 11:
            base_sell = k; base_reason = 'td<11'; break
        if running_max >= 89 and td < 89 and td_prev >= 89:
            cross_count += 1
            if cross_count >= 2:
                base_sell = k; base_reason = 'bull_2nd'; break
        if k >= 20:
            seg = sub['d_trend'].iloc[:k+1].dropna()
            if len(seg) and seg.max() < 89:
                base_sell = k; base_reason = 'ts20'; break

    print(f'=== 九安医疗 002432 入场 {buy_date} 价格 {bp:.2f} ===\n')
    print(f'{"日期":<12} {"close":>7} {"ret%":>9} {"trend":>6} {"mf":>6} {"mf_chg":>7} '
          f'{"retail":>7} {"ret_chg":>8} {"td_chg":>7} {"gua":<4} {"标记":<10}')

    # D6 卖+ U1 买模拟
    holding = True
    cur_buy = sub['close'].iloc[0]
    cum_mult = 1.0
    d6u1_legs = [(buy_date, bp, 'buy')]
    d6u1_end = None

    for k in range(len(sub)):
        r = sub.iloc[k]
        marks = []
        if k == base_sell:
            marks.append(f'[BASE-{base_reason}]')

        # D6+U1 模拟
        if k > 0:
            td = r['d_trend']
            if not pd.isna(td) and td < 11 and d6u1_end is None:
                if holding:
                    cum_mult *= r['close'] / cur_buy
                    d6u1_legs.append((r['date'], r['close'], 'sell-td<11'))
                d6u1_end = k
                marks.append(f'[D6U1-end td<11]')
            elif d6u1_end is None:
                mf_c = r['mf_chg']
                ret_c = r['retail_chg']
                td_c = r['td_chg']
                if not pd.isna(mf_c) and not pd.isna(ret_c) and not pd.isna(td_c):
                    if holding and mf_c < 0 and ret_c < 0 and td_c < 0:
                        cum_mult *= r['close'] / cur_buy
                        d6u1_legs.append((r['date'], r['close'], 'sell-D6'))
                        holding = False
                        marks.append('[sell-D6]')
                    elif (not holding) and mf_c > 0:
                        cur_buy = r['close']
                        d6u1_legs.append((r['date'], r['close'], 'buy-U1'))
                        holding = True
                        marks.append('[buy-U1]')

        marks_str = ' '.join(marks) if marks else ''
        td_chg_str = f'{r["td_chg"]:>+6.1f}' if not pd.isna(r["td_chg"]) else '   nan'
        mf_chg_str = f'{r["mf_chg"]:>+6.0f}' if not pd.isna(r["mf_chg"]) else '   nan'
        ret_chg_str = f'{r["retail_chg"]:>+7.0f}' if not pd.isna(r["retail_chg"]) else '    nan'
        print(f'{r["date"]:<12} {r["close"]:>7.2f} {r["ret_pct"]:>+8.1f}% '
              f'{r["d_trend"]:>6.1f} {r["main_force"]:>+6.0f} {mf_chg_str} '
              f'{r["retail"]:>+6.0f} {ret_chg_str} {td_chg_str} {r["d_gua"]:<4} {marks_str}')

        if k > base_sell + 5 and d6u1_end is not None and k > d6u1_end + 5:
            print(f'  ... (后续略)')
            break

    print(f'\n=== baseline 卖出: {sub["date"].iloc[base_sell]} 价格 {sub["close"].iloc[base_sell]:.2f} '
          f'ret {sub["ret_pct"].iloc[base_sell]:+.1f}% ({base_reason}) ===')

    print(f'\n=== D6+U1 段终结: {sub["date"].iloc[d6u1_end] if d6u1_end else "未触发"} ===')
    if d6u1_end:
        print(f'  最终 ret = {(cum_mult-1)*100:+.1f}%')
        print(f'  腿数: {len(d6u1_legs)}')
        for l in d6u1_legs[:30]:
            print(f'    {l[2]:<14} {l[0]} {l[1]:.2f}')


if __name__ == '__main__':
    main()
