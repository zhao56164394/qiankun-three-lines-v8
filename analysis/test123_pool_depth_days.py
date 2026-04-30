# -*- coding: utf-8 -*-
"""坤 + sanhu<-250 入池 — 池深/池天 信号质量分析

每个买入事件都是 "在某个时刻入池 → 等待 N 天 → 触发巽日 + score>=2 才买".
两个维度:
  - 池深 (depth) = 入池期间 retail (或 sanhu_5d) 最低值
  - 池天 (days) = 入池到触发买入的交易日数

假设:
  池深越深 → 主力吸筹越坚决 → ret 越高
  池天太短 (<3 天) = 刚跌就反弹, 主力还没充分吸筹
  池天太长 (>60 天) = 池信号过期, 已经反弹完了

输出:
  1. 池深 5 档 × n / ret / 胜率
  2. 池天 5 档 × n / ret / 胜率
  3. 池深 × 池天 交叉表
  4. score=2 / score=3 分别看池深池天分布
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
REGIME_Y = '000'
TRIGGER_GUA = '011'
POOL_THR = -250


def main():
    t0 = time.time()
    print('=== 坤 + sanhu<-250 入池: 池深/池天 信号质量 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件 (带池深/池天)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf
        pool_min_sanhu5 = np.inf
        pool_enter_retail = 0.0  # 入池时 retail (单点)
        pool_enter_sanhu5 = 0.0

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i

            # 入池: 第一次 retail < -250
            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail_arr[gi]
                pool_min_sanhu5 = sh5_arr[gi] if not np.isnan(sh5_arr[gi]) else 0.0
                pool_enter_retail = retail_arr[gi]
                pool_enter_sanhu5 = sh5_arr[gi] if not np.isnan(sh5_arr[gi]) else 0.0

            # 池中持续更新最低值
            if in_pool:
                if retail_arr[gi] < pool_min_retail:
                    pool_min_retail = retail_arr[gi]
                if not np.isnan(sh5_arr[gi]) and sh5_arr[gi] < pool_min_sanhu5:
                    pool_min_sanhu5 = sh5_arr[gi]

            # 触发: regime + 巽日 + 强避雷 + score>=2
            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False; continue

                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                if score < 2:
                    in_pool = False; continue

                pool_days = i - pool_enter_i
                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100

                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'score': score, 'ret_30': ret_30,
                    'pool_days': pool_days,
                    'pool_min_retail': pool_min_retail,
                    'pool_min_sanhu5': pool_min_sanhu5,
                    'pool_enter_retail': pool_enter_retail,
                    'pool_enter_sanhu5': pool_enter_sanhu5,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')
    print(f'  池天: min={df_e["pool_days"].min()} max={df_e["pool_days"].max()} '
          f'avg={df_e["pool_days"].mean():.1f} 中位={df_e["pool_days"].median():.0f}')
    print(f'  池深 (retail 最低): min={df_e["pool_min_retail"].min():.0f} '
          f'avg={df_e["pool_min_retail"].mean():.0f} 中位={df_e["pool_min_retail"].median():.0f}')
    print(f'  池深 (sanhu_5d 最低): min={df_e["pool_min_sanhu5"].min():.0f} '
          f'avg={df_e["pool_min_sanhu5"].mean():.0f} 中位={df_e["pool_min_sanhu5"].median():.0f}')

    # ============ 1. 池深分箱 (retail 最低) ============
    print('\n=== 池深分箱: pool_min_retail (入池期间 retail 最低值) ===')
    bins_retail = [-np.inf, -1000, -700, -500, -400, -300, POOL_THR + 0.01]
    labels_retail = ['<-1000', '[-1000,-700)', '[-700,-500)', '[-500,-400)', '[-400,-300)', '[-300,-250)']
    df_e['depth_bin'] = pd.cut(df_e['pool_min_retail'], bins=bins_retail, labels=labels_retail, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"中位":>9}')
    for lab in labels_retail:
        sub = df_e[df_e['depth_bin'] == lab]
        if len(sub) == 0: continue
        print(f'  {lab:<14} {len(sub):>6} {sub["ret_30"].mean():>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["ret_30"].median():>+7.2f}%')

    # ============ 2. 池深分箱 (sanhu_5d 最低) ============
    print('\n=== 池深分箱: pool_min_sanhu5 (入池期间 sanhu_5d 最低值) ===')
    bins_sh5 = [-np.inf, -800, -500, -350, -250, -150, np.inf]
    labels_sh5 = ['<-800', '[-800,-500)', '[-500,-350)', '[-350,-250)', '[-250,-150)', '>=-150']
    df_e['depth_sh5_bin'] = pd.cut(df_e['pool_min_sanhu5'], bins=bins_sh5, labels=labels_sh5, right=False)
    print(f'  {"档":<14} {"n":>6} {"avg_ret":>9} {"win%":>7} {"中位":>9}')
    for lab in labels_sh5:
        sub = df_e[df_e['depth_sh5_bin'] == lab]
        if len(sub) == 0: continue
        print(f'  {lab:<14} {len(sub):>6} {sub["ret_30"].mean():>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["ret_30"].median():>+7.2f}%')

    # ============ 3. 池天分箱 ============
    print('\n=== 池天分箱: pool_days (入池→触发交易日数) ===')
    bins_days = [0, 3, 7, 15, 30, 60, np.inf]
    labels_days = ['1-2', '3-6', '7-14', '15-29', '30-59', '60+']
    df_e['days_bin'] = pd.cut(df_e['pool_days'], bins=bins_days, labels=labels_days, right=False)
    print(f'  {"档":<10} {"n":>6} {"avg_ret":>9} {"win%":>7} {"中位":>9}')
    for lab in labels_days:
        sub = df_e[df_e['days_bin'] == lab]
        if len(sub) == 0: continue
        print(f'  {lab:<10} {len(sub):>6} {sub["ret_30"].mean():>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {sub["ret_30"].median():>+7.2f}%')

    # ============ 4. 池深 × 池天 交叉 ============
    print('\n=== 池深 × 池天 交叉 (avg_ret%) ===')
    pivot = df_e.pivot_table(values='ret_30', index='depth_bin', columns='days_bin',
                                  aggfunc='mean', observed=True)
    print(pivot.round(2).to_string())

    print('\n=== 池深 × 池天 交叉 (n) ===')
    pivot_n = df_e.pivot_table(values='ret_30', index='depth_bin', columns='days_bin',
                                  aggfunc='count', observed=True)
    print(pivot_n.to_string())

    # ============ 5. 按 score 看池深/池天 ============
    print('\n=== score=3 池深分布 ===')
    sub3 = df_e[df_e['score'] == 3]
    print(f'  n={len(sub3)}, 全 ret={sub3["ret_30"].mean():+.2f}%')
    if len(sub3) > 30:
        for lab in labels_retail:
            ssub = sub3[sub3['depth_bin'] == lab]
            if len(ssub) >= 5:
                print(f'  {lab:<14} n={len(ssub):>3} ret={ssub["ret_30"].mean():>+6.2f}% '
                      f'win {(ssub["ret_30"]>0).mean()*100:>5.1f}%')

    print('\n=== score=2 池深分布 ===')
    sub2 = df_e[df_e['score'] == 2]
    print(f'  n={len(sub2)}, 全 ret={sub2["ret_30"].mean():+.2f}%')
    if len(sub2) > 30:
        for lab in labels_retail:
            ssub = sub2[sub2['depth_bin'] == lab]
            if len(ssub) >= 10:
                print(f'  {lab:<14} n={len(ssub):>4} ret={ssub["ret_30"].mean():>+6.2f}% '
                      f'win {(ssub["ret_30"]>0).mean()*100:>5.1f}%')

    # ============ 6. score=2 用池深升级 (能否找到 score=3 等价质量) ============
    print('\n=== 探索: score=2 + 深池 vs score=3 ===')
    print('  目标: 看 score=2 子集中, 池深够深的能否赶上 score=3')
    s3_avg = sub3['ret_30'].mean() if len(sub3) else 0
    print(f'  score=3 全集 ret = {s3_avg:+.2f}% (n={len(sub3)})')
    for thr in [-300, -400, -500, -700, -1000]:
        s2_deep = sub2[sub2['pool_min_retail'] < thr]
        if len(s2_deep) >= 10:
            print(f'  score=2 + retail<{thr}: n={len(s2_deep)} ret={s2_deep["ret_30"].mean():+.2f}% '
                  f'win {(s2_deep["ret_30"]>0).mean()*100:.1f}%')

    # ============ 7. 反向: 池深浅或池天异常的 score=3 是否真值得买 ============
    print('\n=== 反向: score=3 中, 池浅/池天极端 表现 ===')
    if len(sub3) >= 10:
        s3_shallow = sub3[sub3['pool_min_retail'] > -350]
        s3_deep = sub3[sub3['pool_min_retail'] < -500]
        s3_short_days = sub3[sub3['pool_days'] <= 3]
        s3_long_days = sub3[sub3['pool_days'] >= 30]
        for lab, ssub in [('池浅 (>-350)', s3_shallow), ('池深 (<-500)', s3_deep),
                           ('池天<=3', s3_short_days), ('池天>=30', s3_long_days)]:
            if len(ssub):
                print(f'  {lab:<14} n={len(ssub):>3} ret={ssub["ret_30"].mean():>+6.2f}% '
                      f'win {(ssub["ret_30"]>0).mean()*100:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
