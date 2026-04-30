# -*- coding: utf-8 -*-
"""出池机制 + 入池后巽日数量 vs 主升浪 分析

问题 1: 散户回升出池 vs 巽日触发出池
  - 当前: retail>=0 出池
  - 候选: 巽日触发后出池 (即"用一次买信号", 之后池子作废)
  - 候选: 不主动出池, 直到下次入池重置

  比较: 同一只股票的"池期"内可能有多次巽日, 第 1 次质量好还是后面好?

问题 2: 入池后巽日数量与主升浪关系
  - 假设主升浪 = 30 日内 d_gua=111 ≥ 10 日 (强 trend)
  - 看: 入池后第 N 次巽日的主升浪率 / ret_30 / win
  - 如果"第 1 次最强, 后面递减" → 用 1 次后出池
  - 如果"前几次都强" → 不主动出池
  - 如果"持续都好" → 当前 retail>=0 出池可能限制太严

数据建模:
  对每只股票, 找到所有"入池-退池"循环 (用 retail<-250 入, retail>=0 出)
  在每个池期内, 找所有巽日, 编号 1, 2, 3, ...
  对每个巽日: 评估 30 日 ret, 是否主升浪
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
TRIGGER_GUA = '011'
REGIME_Y = '000'
POOL_THR = -250
POOL_EXIT_RETAIL = 0


def main():
    t0 = time.time()
    print('=== 出池机制 + 池内巽日数量分析 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫池期内的所有巽日 (按池期编号 + 主升浪标记) ...')
    events = []
    pool_periods = 0  # 池期数

    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf
        sun_seq = 0  # 池期内巽日序号 (0 = 没入池, 1 = 第 1 次, ...)

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i

            # 入池
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail_arr[gi]
                sun_seq = 0
                pool_periods += 1
            elif in_pool and retail_arr[gi] < pool_min_retail:
                pool_min_retail = retail_arr[gi]

            # 出池: retail>=0
            if in_pool and retail_arr[gi] >= POOL_EXIT_RETAIL:
                in_pool = False
                continue

            # 巽日触发: 仅在 mkt_y=000 + 个股d=011
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                sun_seq += 1

                # 30 日 ret
                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100

                # 主升浪标记: 后续 30 日内 stk_d=='111' 出现 ≥10 次
                gua_window = stk_d_arr[gi+1:gi+EVAL_WIN+1]
                qian_count = (gua_window == '111').sum()
                is_zsl = qian_count >= 10  # 主升浪

                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'pool_period_id': pool_periods,
                    'sun_seq': sun_seq,
                    'pool_days_from_enter': i - pool_enter_i,
                    'cur_retail': retail_arr[gi],
                    'pool_min_retail': pool_min_retail,
                    'ret_30': ret_30,
                    'is_zsl': is_zsl,
                    'qian_count_30d': int(qian_count),
                })

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,} (池期数: {pool_periods})')
    print(f'  全集: ret_30={df_e["ret_30"].mean():+.2f}%, win={(df_e["ret_30"]>0).mean()*100:.1f}%, '
          f'主升率={df_e["is_zsl"].mean()*100:.1f}%')

    # ============ 1. sun_seq (池内巽日序号) 分析 ============
    print('\n=== 1. 池内巽日序号 vs ret_30 / 主升率 ===\n')
    print(f'  {"序号":<4} {"n":>6} {"占比":>5} {"avg_ret":>9} {"win%":>7} {"主升率":>7} {"qian/30d":>8}')
    seq_groups = df_e.groupby('sun_seq')
    for seq in sorted(df_e['sun_seq'].unique()):
        if seq > 10: continue
        sub = seq_groups.get_group(seq)
        if len(sub) < 30: continue
        avg = sub['ret_30'].mean()
        win = (sub['ret_30']>0).mean()*100
        zsl = sub['is_zsl'].mean()*100
        qc = sub['qian_count_30d'].mean()
        print(f'  {seq:<4} {len(sub):>6,} {len(sub)/len(df_e)*100:>4.1f}% '
              f'{avg:>+8.2f}% {win:>6.1f}% {zsl:>6.1f}% {qc:>7.1f}')

    # 序号分箱
    print('\n  按序号分组:')
    bins = [0, 1, 2, 3, 5, 100]
    labels = ['第1次', '第2次', '第3次', '第4-5次', '第6+次']
    df_e['sb'] = pd.cut(df_e['sun_seq'], bins=bins, labels=labels, right=True)
    print(f'  {"组":<10} {"n":>6} {"avg_ret":>9} {"win%":>7} {"主升率":>7}')
    for lab in labels:
        sub = df_e[df_e['sb'] == lab]
        if len(sub) < 30: continue
        print(f'  {lab:<10} {len(sub):>6,} {sub["ret_30"].mean():>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["is_zsl"].mean()*100:>6.1f}%')

    # ============ 2. 池期内总巽日数 vs 主升浪率 ============
    print('\n=== 2. 池期内巽日总数 vs 池期质量 ===')
    print('  (一个池期内若巽日多 = 主力反复给信号, 是否主升浪率更高?)\n')
    period_summary = df_e.groupby('pool_period_id').agg(
        n_sun=('sun_seq', 'max'),
        avg_ret=('ret_30', 'mean'),
        max_zsl=('is_zsl', 'max'),
        max_ret=('ret_30', 'max'),
    ).reset_index()
    print(f'  {"巽日总数":<8} {"池期数":>6} {"占比":>5} {"avg ret":>9} {"含主升池率":>10} {"max ret":>9}')
    for n_sun in [1, 2, 3, 4, 5]:
        sub = period_summary[period_summary['n_sun'] == n_sun]
        if len(sub) < 10: continue
        print(f'  ={n_sun:<7} {len(sub):>6,} {len(sub)/len(period_summary)*100:>4.1f}% '
              f'{sub["avg_ret"].mean():>+8.2f}% {sub["max_zsl"].mean()*100:>9.1f}% '
              f'{sub["max_ret"].mean():>+8.2f}%')
    sub = period_summary[period_summary['n_sun'] >= 6]
    if len(sub) >= 5:
        print(f'  >=6     {len(sub):>6,} {len(sub)/len(period_summary)*100:>4.1f}% '
              f'{sub["avg_ret"].mean():>+8.2f}% {sub["max_zsl"].mean()*100:>9.1f}% '
              f'{sub["max_ret"].mean():>+8.2f}%')

    # ============ 3. 池期内"哪一次巽日最强"分布 ============
    print('\n=== 3. 池期内每一次巽日 ret 排序: "最强次"是第几次 ===')
    # 对每个池期, 找 ret_30 最高的那次的 sun_seq
    idx_best = df_e.groupby('pool_period_id')['ret_30'].idxmax()
    best_seq = df_e.loc[idx_best, ['pool_period_id', 'sun_seq', 'ret_30']].copy()
    period_n = df_e.groupby('pool_period_id')['sun_seq'].max().rename('n_sun')
    best_seq = best_seq.merge(period_n.reset_index(), on='pool_period_id')
    best_seq2 = best_seq[best_seq['n_sun'] >= 2]
    print(f'  在巽日数 >= 2 的池期 ({len(best_seq2)} 个) 中, 最强次的位置:')
    print(f'  {"位置":<8} {"占比":>5} (期望: 若第 1 次最强, 此处占比应该 > 50%)')
    pos_counts = best_seq2['sun_seq'].value_counts().sort_index()
    total = len(best_seq2)
    for seq, cnt in pos_counts.head(10).items():
        print(f'  第 {seq:<3} 次  {cnt/total*100:>4.1f}% (n={cnt})')

    # ============ 4. 池期内累积 ret (买第 1 次拿 30 日) ============
    print('\n=== 4. 在池期内, 第 1 次买 vs 等"最强次"买 vs 全部买 ===')
    # 第 1 次买
    first_only = df_e[df_e['sun_seq'] == 1]
    print(f'  策略 A (第 1 次): n={len(first_only)} ret={first_only["ret_30"].mean():+.2f}% '
          f'win={(first_only["ret_30"]>0).mean()*100:.1f}% 主升={first_only["is_zsl"].mean()*100:.1f}%')

    # 全部买
    all_buy = df_e
    print(f'  策略 B (全部巽日): n={len(all_buy)} ret={all_buy["ret_30"].mean():+.2f}% '
          f'win={(all_buy["ret_30"]>0).mean()*100:.1f}% 主升={all_buy["is_zsl"].mean()*100:.1f}%')

    # 第 1 次和第 2 次都买
    first_two = df_e[df_e['sun_seq'] <= 2]
    print(f'  策略 C (前 2 次): n={len(first_two)} ret={first_two["ret_30"].mean():+.2f}% '
          f'win={(first_two["ret_30"]>0).mean()*100:.1f}% 主升={first_two["is_zsl"].mean()*100:.1f}%')

    # 池期内有主升浪的, 主升浪在第几次出现
    print('\n=== 5. 主升浪池期: 第 1 次巽日就是主升浪吗 ===')
    zsl_periods = df_e[df_e['is_zsl']]['pool_period_id'].unique()
    print(f'  含主升浪的池期: {len(zsl_periods)} ({len(zsl_periods)/pool_periods*100:.1f}%)')
    # 在主升浪池期中, 主升浪第一次出现在第几次巽日
    zsl_first_seq = []
    for pid in zsl_periods:
        pool_evts = df_e[df_e['pool_period_id'] == pid].sort_values('sun_seq')
        zsl_evts = pool_evts[pool_evts['is_zsl']]
        if len(zsl_evts):
            zsl_first_seq.append(zsl_evts['sun_seq'].iloc[0])
    if zsl_first_seq:
        zsl_first_seq = pd.Series(zsl_first_seq)
        print(f'  主升浪首次出现位置:')
        for seq, cnt in zsl_first_seq.value_counts().sort_index().head(8).items():
            print(f'    第 {seq:<3} 次  {cnt/len(zsl_first_seq)*100:>5.1f}% (n={cnt})')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
