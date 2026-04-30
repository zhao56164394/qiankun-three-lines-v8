# -*- coding: utf-8 -*-
"""顺丰 002352 4-19 之后到 6-21 的逐日 — 看主力线/散户线状态
判断: 60 天后停止买卖的应该是什么条件?

并看几个典型大牛股 (顺丰/002432/000957/600096) 的完整持仓期
搞清楚: 真正的"段终结"条件应该是什么
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

    cases = [
        ('顺丰 002352', '002352', '2016-01-19', '2016-09-30'),
        ('暴涨股 002432', '002432', '2021-11-03', '2022-09-30'),
    ]

    for tag, code, buy_date, end_date in cases:
        print(f'\n========== {tag} {buy_date} → {end_date} ==========\n')
        sub = df[df['code'] == code].sort_values('date').reset_index(drop=True)
        sub = sub[(sub['date'] >= buy_date) & (sub['date'] <= end_date)].copy()
        if len(sub) == 0:
            print('无数据')
            continue
        bp = sub['close'].iloc[0]
        sub['ret_pct'] = (sub['close'] / bp - 1) * 100
        sub['mf_chg'] = sub['main_force'].diff()
        sub['retail_chg'] = sub['retail'].diff()
        sub['mf_5d'] = sub['main_force'].rolling(5, min_periods=1).mean()
        sub['mf_ago_5d_min'] = sub['main_force'].rolling(5, min_periods=1).min()
        sub['mf_ago_5d_max'] = sub['main_force'].rolling(5, min_periods=1).max()

        # 标记三线方向
        def label(r):
            if pd.isna(r['mf_chg']) or pd.isna(r['retail_chg']): return ''
            mu = r['mf_chg'] > 0; ru = r['retail_chg'] > 0
            md = r['mf_chg'] < 0; rd = r['retail_chg'] < 0
            if mu and ru: return '双升'
            if md and rd: return '双降'
            if mu and rd: return 'mf↑ ret↓'
            if md and ru: return 'mf↓ ret↑'
            return '平'
        sub['signal'] = sub.apply(label, axis=1)

        # 显示 (每 5 行间断打印一次, 把信号天密集打)
        print(f'{"日期":<12} {"close":>7} {"ret%":>7} {"trend":>6} {"gua":<4} '
              f'{"mf":>6} {"mf_chg":>7} {"retail":>7} {"ret_chg":>8} {"mf_5d":>7} {"signal":<10}')
        for _, r in sub.iterrows():
            mark = ''
            if r['ret_pct'] >= 100: mark = ' ★ 翻倍'
            elif r['ret_pct'] >= 200: mark = ' ★★ 翻 2 倍'
            print(f'{r["date"]:<12} {r["close"]:>7.2f} {r["ret_pct"]:>+6.1f}% {r["d_trend"]:>6.1f} {r["d_gua"]:<4} '
                  f'{r["main_force"]:>+6.0f} {r["mf_chg"]:>+6.0f} '
                  f'{r["retail"]:>+6.0f} {r["retail_chg"]:>+7.0f} {r["mf_5d"]:>+6.0f} {r["signal"]:<10}{mark}')

        # 关键: 60 天后还涨, 主力在不在?
        print(f'\n  60d 后 mf/retail 状态:')
        if len(sub) > 60:
            after_60 = sub.iloc[60:].copy()
            print(f'    60-end avg mf = {after_60["main_force"].mean():+.0f}')
            print(f'    60-end avg retail = {after_60["retail"].mean():+.0f}')
            print(f'    mf>0 的天数比例: {(after_60["main_force"]>0).mean()*100:.1f}%')
            print(f'    mf>50 的天数比例: {(after_60["main_force"]>50).mean()*100:.1f}%')

        # 终止条件候选: 看哪天起 主力线持续转弱
        print(f'\n  各种"终止信号"出现日:')
        # E1: 5 日 mf 都 < 0
        for i in range(5, len(sub)):
            mf5 = sub['main_force'].iloc[i-4:i+1]
            if (mf5 < 0).all():
                print(f'    5 日 mf 都<0 首次: {sub["date"].iloc[i]} (close={sub["close"].iloc[i]:.2f}, ret={sub["ret_pct"].iloc[i]:+.1f}%)')
                break
        else:
            print(f'    5 日 mf 都<0: 从未触发')

        # E2: 5 日 mf 都 < 50
        for i in range(5, len(sub)):
            mf5 = sub['main_force'].iloc[i-4:i+1]
            if (mf5 < 50).all():
                print(f'    5 日 mf 都<50 首次: {sub["date"].iloc[i]} (close={sub["close"].iloc[i]:.2f}, ret={sub["ret_pct"].iloc[i]:+.1f}%)')
                break

        # E3: trend<11
        for i in range(len(sub)):
            if not pd.isna(sub['d_trend'].iloc[i]) and sub['d_trend'].iloc[i] < 11:
                print(f'    trend<11 首次: {sub["date"].iloc[i]} (close={sub["close"].iloc[i]:.2f}, ret={sub["ret_pct"].iloc[i]:+.1f}%)')
                break
        else:
            print(f'    trend<11: 从未触发')


if __name__ == '__main__':
    main()
