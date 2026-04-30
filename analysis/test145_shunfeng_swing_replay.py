# -*- coding: utf-8 -*-
"""顺丰 002352 60 天逐日 mf+retail 三线模拟波段
+ 多个双升/双降阈值扫描

阶段 1: 把顺丰每天的 mf/retail 双线方向标出来
        (升/降/平) 看双升/双降信号在哪些天发生

阶段 2: 用顺丰的真实数据回放波段:
        建仓 → 双升持有 → 双降卖 → 单升买回 → 双降卖 → ...
        看一段下来累积收益是多少

阶段 3: 多个阈值参数扫描全样本
"""
import os, sys, io, time
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

    # ============ 1. 顺丰逐日轨迹 ============
    sf = df[df['code'] == '002352'].sort_values('date').reset_index(drop=True)
    sf = sf[sf['date'].between('2016-01-19', '2016-04-19')].reset_index(drop=True)
    sf['ret_pct'] = (sf['close'] / sf['close'].iloc[0] - 1) * 100
    sf['mf_chg'] = sf['main_force'].diff()
    sf['retail_chg'] = sf['retail'].diff()

    # 双升/双降标记
    def label(row):
        if pd.isna(row['mf_chg']) or pd.isna(row['retail_chg']): return ''
        mu = row['mf_chg'] > 0
        ru = row['retail_chg'] > 0
        md = row['mf_chg'] < 0
        rd = row['retail_chg'] < 0
        if mu and ru: return '双升'
        if md and rd: return '双降'
        if mu and rd: return 'mf↑ ret↓'
        if md and ru: return 'mf↓ ret↑'
        return '平'
    sf['signal'] = sf.apply(label, axis=1)

    print('=== 顺丰 002352 2016-01-19 → 04-19 逐日 ===\n')
    print(f'{"日期":<12} {"close":>6} {"ret%":>7} {"trend":>6} {"mf":>6} {"mf_chg":>7} '
          f'{"retail":>7} {"ret_chg":>8} {"signal":<10} {"gua":<4}')
    for _, r in sf.iterrows():
        print(f'{r["date"]:<12} {r["close"]:>6.2f} {r["ret_pct"]:>+6.1f}% {r["d_trend"]:>6.1f} '
              f'{r["main_force"]:>+6.0f} {r["mf_chg"]:>+6.0f} '
              f'{r["retail"]:>+6.0f} {r["retail_chg"]:>+7.0f} {r["signal"]:<10} {r["d_gua"]:<4}')

    # 数一下信号
    print('\n=== 信号统计 ===')
    print(sf['signal'].value_counts())

    # ============ 2. 双升/双降阈值扫描 ============
    print('\n\n=== 用真实波段回放: 多个阈值 ===\n')

    def replay(sf, mf_thr, ret_thr):
        """模拟回放
        建仓: 入场日 (第 1 行) 100% 满仓
        卖出: 双降且 mf<-mf_thr OR retail<-ret_thr 的强双降 (微降不算)
            实际: mf_chg<=-mf_thr AND retail_chg<=-ret_thr
        再买: mf 上升 (mf_chg>0) AND mf>50 (强位) AND trend>11
        清仓: 60 日末或 trend<11

        返回: 累积收益, 交易腿数, 每次交易记录
        """
        cum_mult = 1.0
        holding = True
        cur_buy_price = sf['close'].iloc[0]
        legs = []
        legs.append(('buy', sf['date'].iloc[0], sf['close'].iloc[0]))

        for i in range(1, len(sf)):
            row = sf.iloc[i]
            cls = row['close']
            td = row['d_trend']
            mf_c = row['mf_chg']
            ret_c = row['retail_chg']
            mf = row['main_force']

            # 强卖: trend<11
            if not pd.isna(td) and td < 11:
                if holding:
                    cum_mult *= cls / cur_buy_price
                    legs.append(('sell-td<11', row['date'], cls))
                    holding = False
                break

            # 双降卖
            if holding and not pd.isna(mf_c) and not pd.isna(ret_c):
                if mf_c <= -mf_thr and ret_c <= -ret_thr:
                    cum_mult *= cls / cur_buy_price
                    legs.append(('sell-双降', row['date'], cls))
                    holding = False
                    continue

            # 再买
            if not holding and not pd.isna(mf_c):
                if mf_c > 0 and mf > 50 and (pd.isna(td) or td > 11):
                    cur_buy_price = cls
                    legs.append(('buy', row['date'], cls))
                    holding = True
                    continue

        # 末尾强平
        if holding:
            cls = sf['close'].iloc[-1]
            cum_mult *= cls / cur_buy_price
            legs.append(('sell-末尾', sf['date'].iloc[-1], cls))

        return (cum_mult - 1) * 100, legs

    test_pairs = [
        (50, 50),
        (100, 50),
        (100, 100),
        (200, 100),
        (300, 150),
        (500, 200),
        (1, 1),  # 任意降都算
    ]

    print(f'{"mf_thr":>8} {"ret_thr":>8} {"final_ret":>10} {"legs":>5}')
    for mf_t, ret_t in test_pairs:
        ret, legs = replay(sf, mf_t, ret_t)
        n_legs = sum(1 for l in legs if l[0] == 'buy') + sum(1 for l in legs if l[0].startswith('sell'))
        print(f'{mf_t:>8} {ret_t:>8} {ret:>+9.2f}% {n_legs:>5}')
        print(f'    交易: ', ' → '.join(f'{l[0][:4]}@{l[1][-5:]}/{l[2]:.2f}' for l in legs))

    # ============ 3. 神火 000933 同样回放 ============
    print('\n\n=== 神火 000933 同步回放 ===\n')
    sh = df[df['code'] == '000933'].sort_values('date').reset_index(drop=True)
    sh = sh[sh['date'].between('2016-02-17', '2016-05-16')].reset_index(drop=True)
    sh['ret_pct'] = (sh['close'] / sh['close'].iloc[0] - 1) * 100
    sh['mf_chg'] = sh['main_force'].diff()
    sh['retail_chg'] = sh['retail'].diff()

    print(f'{"mf_thr":>8} {"ret_thr":>8} {"final_ret":>10} {"legs":>5}')
    for mf_t, ret_t in test_pairs:
        ret, legs = replay(sh, mf_t, ret_t)
        n_legs = sum(1 for l in legs if l[0] == 'buy') + sum(1 for l in legs if l[0].startswith('sell'))
        print(f'{mf_t:>8} {ret_t:>8} {ret:>+9.2f}% {n_legs:>5}')
        print(f'    交易: ', ' → '.join(f'{l[0][:4]}@{l[1][-5:]}/{l[2]:.2f}' for l in legs))

    # ============ 4. 看顺丰每个 5 日窗口的 mf/retail 同步性 ============
    print('\n\n=== 顺丰每 5 日窗口 mf/retail 同步性 ===\n')
    for i in range(0, len(sf) - 4, 5):
        window = sf.iloc[i:i+5]
        mf_total = window['mf_chg'].sum()
        ret_total = window['retail_chg'].sum()
        print(f'  {window["date"].iloc[0]:<12} → {window["date"].iloc[-1]:<12} '
              f'  5d Σmf_chg={mf_total:>+6.0f}, Σretail_chg={ret_total:>+6.0f}, '
              f'  ret={(window["close"].iloc[-1]/window["close"].iloc[0]-1)*100:>+5.1f}%')


if __name__ == '__main__':
    main()
