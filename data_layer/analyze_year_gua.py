# -*- coding: utf-8 -*-
"""年卦 (月 K 尺度) 变卦事件分析 — 月末采样 + 牛熊周期对应

年卦 (day-view) 每天用当月未收 K 更新, 同月内会随价格波动翻转.
这里只取每个自然月最后一个交易日的 年卦, 作为该月定格值.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NAME_ZH = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
           '100': '震', '101': '离', '110': '兑', '111': '乾'}
MEAN_ZH = {'000': '深熊探底', '001': '熊底异动', '010': '反弹乏力', '011': '底部爆发',
           '100': '崩盘加速', '101': '下跌护盘', '110': '牛末滞涨', '111': '疯牛主升'}


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')
    df = pd.read_csv(src, encoding='utf-8-sig', dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['y_gua'].notna() & (df['y_gua'] != '') & (df['y_gua'] != 'nan')].copy()
    df['y_gua'] = df['y_gua'].str.zfill(3)
    df = df.sort_values('date').reset_index(drop=True)

    df['ym'] = df['date'].dt.to_period('M')
    month_end = df.groupby('ym').last().reset_index().sort_values('date').reset_index(drop=True)

    month_end['prev_gua'] = month_end['y_gua'].shift()
    month_end['changed'] = (month_end['y_gua'] != month_end['prev_gua']) & month_end['prev_gua'].notna()

    events = month_end[month_end['changed']].reset_index(drop=True)
    events['event_idx_in_month_end'] = month_end[month_end['changed']].index.tolist()
    events['seg_end_idx'] = list(events['event_idx_in_month_end'].tolist()[1:]) + [len(month_end) - 1]

    print(f'年卦有效期: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')
    print(f'自然月数: {len(month_end)}, 月末变卦次数: {len(events)}')
    print()

    hdr = f'{"#":<3} {"月末":<12} {"从":<6} {"到":<10} {"收":>7} {"段月数":>5} {"段收益%":>8} {"段内高/低%":>12}'
    print(hdr)
    print('-' * 85)

    for i, r in events.iterrows():
        si = r['event_idx_in_month_end']
        ei = int(r['seg_end_idx'])
        seg = month_end.iloc[si:ei + 1]

        months = len(seg)
        c0 = float(seg['close'].iloc[0])
        c1 = float(seg['close'].iloc[-1])
        ret = (c1 / c0 - 1) * 100

        d0 = seg['date'].iloc[0]
        d1 = seg['date'].iloc[-1]
        daily_seg = df[(df['date'] >= d0) & (df['date'] <= d1)]
        hi = daily_seg['close'].max()
        lo = daily_seg['close'].min()
        hi_pct = (hi / c0 - 1) * 100
        lo_pct = (lo / c0 - 1) * 100

        prev = str(r['prev_gua']).zfill(3)
        cur = str(r['y_gua']).zfill(3)
        prev_s = f'{prev}{NAME_ZH.get(prev, "?")}'
        cur_s = f'{cur}{NAME_ZH.get(cur, "?")}{MEAN_ZH.get(cur, "")}'

        print(f'{i+1:<3} {r["date"].strftime("%Y-%m-%d"):<12} {prev_s:<6} {cur_s:<12} '
              f'{c0:>7.0f} {months:>5} {ret:>+7.1f}% {f"+{hi_pct:.0f}/{lo_pct:.0f}":>12}')


if __name__ == '__main__':
    main()
