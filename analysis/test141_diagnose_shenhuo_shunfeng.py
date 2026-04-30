# -*- coding: utf-8 -*-
"""神火股份 2016-02-17 那笔失败案例的详细诊断
+ 顺丰 2016-01-19 那笔成功案例的详细对比

为什么 bull_2nd 在神火上崩, 在顺丰上飞?
为什么 mf 拐点法在顺丰上反复砍腿?
真正的卖点应该看什么?
"""
import os, sys, io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print('=== 神火 vs 顺丰 持仓期日详细数据 ===\n')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'))
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    p['code'] = p['code'].astype(str).str.zfill(6)
    p['date'] = p['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    g['date'] = g['date'].astype(str)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)

    df = p.merge(g, on=['code', 'date'], how='left')

    cases = [
        ('神火股份失败', '000933', '2016-02-17', '2016-05-16'),
        ('顺丰成功',     '002352', '2016-01-19', '2016-04-19'),
    ]

    for tag, code, buy_date, end_date in cases:
        print(f'\n========== {tag} {code} {buy_date} → {end_date} ==========')
        sub = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        sub = sub[(sub['date'] >= buy_date) & (sub['date'] <= end_date)].copy()
        sub['mf_3d'] = sub['main_force'].rolling(3, min_periods=1).mean()
        sub['mf_5d'] = sub['main_force'].rolling(5, min_periods=1).mean()
        sub['mf_10d'] = sub['main_force'].rolling(10, min_periods=1).mean()

        buy_price = sub['close'].iloc[0]
        sub['ret_pct'] = (sub['close'] / buy_price - 1) * 100
        sub['mf_5d_chg'] = sub['mf_5d'].diff()

        # 关键: 用 trend 看 89 上下穿
        sub['above_89'] = sub['d_trend'] >= 89

        print(f'\n  {"日期":<12} {"close":>6} {"ret%":>7} {"trend":>6} {"a89":>3} '
              f'{"mf":>6} {"mf3":>6} {"mf5":>6} {"mf5_chg":>7} {"gua":<4}')
        for _, r in sub.iterrows():
            print(f'  {r["date"]:<12} {r["close"]:>6.2f} {r["ret_pct"]:>+6.1f}% '
                  f'{r["d_trend"]:>6.1f} {"Y" if r["above_89"] else " ":>3} '
                  f'{r["main_force"]:>+6.0f} {r["mf_3d"]:>+6.0f} {r["mf_5d"]:>+6.0f} '
                  f'{r["mf_5d_chg"]:>+6.0f} {r["d_gua"]:<4}')

        # 关键时刻分析
        print(f'\n  关键: 持仓期最高浮盈 / 浮亏 / 最终')
        print(f'    最高浮盈: {sub["ret_pct"].max():+.2f}% (在 {sub.loc[sub["ret_pct"].idxmax(), "date"]})')
        print(f'    最低浮亏: {sub["ret_pct"].min():+.2f}% (在 {sub.loc[sub["ret_pct"].idxmin(), "date"]})')
        print(f'    最终:     {sub["ret_pct"].iloc[-1]:+.2f}%')

        # trend 89 穿越次数
        crosses = 0
        for i in range(1, len(sub)):
            if sub['d_trend'].iloc[i-1] >= 89 and sub['d_trend'].iloc[i] < 89:
                crosses += 1
        print(f'    trend 下穿 89 次数: {crosses}')

        # mf_5d 方向变化次数
        sign_changes = 0
        prev_sign = 0
        for i in range(1, len(sub)):
            cur = sub['mf_5d_chg'].iloc[i]
            if pd.isna(cur): continue
            cur_sign = 1 if cur > 0 else (-1 if cur < 0 else 0)
            if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
                sign_changes += 1
            if cur_sign != 0:
                prev_sign = cur_sign
        print(f'    mf_5d 方向变化次数: {sign_changes}')


if __name__ == '__main__':
    main()
