# -*- coding: utf-8 -*-
"""月卦 (周 K 尺度) 按牛熊周期分组 + 大段详列"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NAME_ZH = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
           '100': '震', '101': '离', '110': '兑', '111': '乾'}
MEAN_ZH = {'000': '深熊探底', '001': '熊底异动', '010': '反弹乏力', '011': '底部爆发',
           '100': '崩盘加速', '101': '下跌护盘', '110': '牛末滞涨', '111': '疯牛主升'}

# 8 个牛熊周期
CYCLES = [
    ('2005-06-01', '2007-10-31', '2005-07-2007-10 超级大牛'),
    ('2007-11-01', '2008-11-30', '2007-11-2008-11 金融危机熊'),
    ('2008-12-01', '2014-05-31', '2008-12-2014-05 长阴跌横盘'),
    ('2014-06-01', '2015-06-30', '2014-06-2015-06 杠杆牛'),
    ('2015-07-01', '2016-01-31', '2015-07-2016-01 股灾+熔断'),
    ('2016-02-01', '2018-12-31', '2016-02-2018-12 慢牛转慢熊'),
    ('2019-01-01', '2021-02-28', '2019-01-2021-02 结构牛'),
    ('2021-03-01', '2024-02-28', '2021-03-2024-02 深熊'),
    ('2024-03-01', '2026-12-31', '2024-03- 政策牛+震荡'),
]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')
    df = pd.read_csv(src, encoding='utf-8-sig', dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['m_gua'].notna() & (df['m_gua'] != '') & (df['m_gua'] != 'nan')].copy()
    df['m_gua'] = df['m_gua'].str.zfill(3)
    df = df.sort_values('date').reset_index(drop=True)

    # 每周 K 收盘 (周五) 采样 — 月卦天然采样点
    df['yw'] = df['date'].dt.to_period('W-FRI')
    sample = df.groupby('yw').last().reset_index().sort_values('date').reset_index(drop=True)

    sample['prev'] = sample['m_gua'].shift()
    sample['changed'] = (sample['m_gua'] != sample['prev']) & sample['prev'].notna()
    events = sample[sample['changed']].reset_index()
    events.rename(columns={'index': 'e_idx'}, inplace=True)
    events['seg_end_idx'] = list(events['e_idx'].tolist()[1:]) + [len(sample) - 1]

    def build_segs():
        rows = []
        for _, r in events.iterrows():
            si, ei = int(r['e_idx']), int(r['seg_end_idx'])
            seg = sample.iloc[si:ei + 1]
            c0 = float(seg['close'].iloc[0])
            c1 = float(seg['close'].iloc[-1])
            d0 = seg['date'].iloc[0]
            d1 = seg['date'].iloc[-1]
            daily_seg = df[(df['date'] >= d0) & (df['date'] <= d1)]
            hi_pct = (daily_seg['close'].max() / c0 - 1) * 100
            lo_pct = (daily_seg['close'].min() / c0 - 1) * 100
            prev = str(r['prev']).zfill(3)
            cur = str(r['m_gua']).zfill(3)
            rows.append({
                'date': d0, 'end_date': d1, 'prev': prev, 'cur': cur,
                'weeks': len(seg), 'close0': c0, 'close1': c1,
                'ret': (c1 / c0 - 1) * 100, 'hi_pct': hi_pct, 'lo_pct': lo_pct,
            })
        return pd.DataFrame(rows)

    segs = build_segs()

    print(f'月卦有效期: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')
    print(f'周 K 总数 {len(sample)}   周末变卦总数 {len(segs)}')
    print()

    print('=== 周期汇总 (阳卦 = 乾兑离震 位=1; 阴卦 = 巽坎艮坤 位=0) ===')
    print(f'{"周期":<40} {"天数":>6} {"阳卦%":>7} {"阴卦%":>7} {"乾%":>5} {"坤%":>5} {"主导":>6}')
    for s, e, name in CYCLES:
        sub = df[(df['date'] >= s) & (df['date'] <= e)]
        if len(sub) == 0:
            continue
        pos = sub['m_gua'].str[0].astype(int)
        up = (pos == 1).sum() / len(sub) * 100
        dn = (pos == 0).sum() / len(sub) * 100
        qian = (sub['m_gua'] == '111').sum() / len(sub) * 100
        kun = (sub['m_gua'] == '000').sum() / len(sub) * 100
        domin = '乾' if qian >= kun else '坤'
        print(f'{name:<40} {len(sub):>6} {up:>6.1f}% {dn:>6.1f}% {qian:>4.1f}% {kun:>4.1f}%  {domin:>5}')

    print('\n=== 大段 (≥8 周 ≈2 月) ===')
    big = segs[segs['weeks'] >= 8].reset_index(drop=True)
    print(f'共 {len(big)} 段 (占 {len(big)/len(segs)*100:.1f}%)')
    print(f'{"#":<3} {"起":<12} {"止":<12} {"从→到":<14} {"周":>4} {"收益%":>7} {"高/低%":>10}')
    print('-' * 80)
    for i, r in big.iterrows():
        cur_s = f'{r["prev"]}{NAME_ZH[r["prev"]]}→{r["cur"]}{NAME_ZH[r["cur"]]}'
        print(f'{i+1:<3} {r["date"].strftime("%Y-%m-%d"):<12} {r["end_date"].strftime("%Y-%m-%d"):<12} '
              f'{cur_s:<14} {r["weeks"]:>4} {r["ret"]:>+6.1f}% '
              f'{f"+{r.hi_pct:.0f}/{r.lo_pct:.0f}":>10}')

    print('\n=== 大段 方向准确率 (卦意 vs 段收益) ===')

    def is_yang_gua(g):
        return g[0] == '1'

    hit = sum(1 for _, r in big.iterrows() if is_yang_gua(r['cur']) == (r['ret'] > 0))
    print(f'  ≥8 周段命中: {hit}/{len(big)} = {hit/max(len(big),1)*100:.1f}%')
    huge = segs[segs['weeks'] >= 13].reset_index(drop=True)
    hit2 = sum(1 for _, r in huge.iterrows() if is_yang_gua(r['cur']) == (r['ret'] > 0))
    print(f'  ≥13 周段命中: {hit2}/{len(huge)} = {hit2/max(len(huge),1)*100:.1f}%')


if __name__ == '__main__':
    main()
