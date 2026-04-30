# -*- coding: utf-8 -*-
"""坤+入池 — 信号数本身就是分类器"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
REGIME_Y = '000'
TRIGGER_GUA = '011'


def main():
    t0 = time.time()
    print('=== 坤+入池: 信号数 vs ret 关系 (含全部 score, 不限 score>=2) ===\n')

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
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    df['t_prev'] = df.groupby('code', sort=False)['d_trend'].shift(1)
    df['cross_below_11'] = (df['t_prev'] >= 11) & (df['d_trend'] < 11)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    cross_arr = df['cross_below_11'].to_numpy()
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy()
    stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy()
    mkt_m_arr = df['mkt_m'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件 (在池+坤+巽+避雷, 不限 score)...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        n = e - s
        in_pool = False

        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if cross_arr[gi]:
                in_pool = True

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                if stk_y_arr[gi] == '011' or stk_m_arr[gi] == '111':
                    in_pool = False
                    continue

                # score
                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'score': score, 'ret_30': ret_30,
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    daily_count = df_e.groupby('date').size()
    df_e['n_today'] = df_e['date'].map(daily_count)
    print(f'  事件: {len(df_e):,}, 涉及 {df_e["date"].nunique()} 天')

    # ============ 信号数细分 ============
    print('\n=== 信号数细分 (含全部 score) ===')
    bins = [
        (1, 1, '=1 只'),
        (2, 2, '=2 只'),
        (3, 5, '3-5 只'),
        (6, 10, '6-10 只'),
        (11, 20, '11-20 只'),
        (21, 50, '21-50 只'),
        (51, 100, '51-100 只'),
        (101, 9999, '100+ 只'),
    ]
    print(f'  {"信号数":<14} {"日数":>5} {"事件":>7} {"avg ret":>9} {"win%":>7} {"score=2":>9} {"score=3":>9} {"score=4":>9}')
    for lo, hi, label in bins:
        sub = df_e[(df_e['n_today']>=lo)&(df_e['n_today']<=hi)]
        days_in = ((daily_count>=lo)&(daily_count<=hi)).sum()
        if len(sub) == 0: continue
        s2 = (sub['score']==2).sum() / len(sub) * 100
        s3 = (sub['score']==3).sum() / len(sub) * 100
        s4 = (sub['score']>=4).sum() / len(sub) * 100
        print(f'  {label:<14} {days_in:>5} {len(sub):>7,} {sub["ret_30"].mean():>+8.2f}% '
              f'{(sub["ret_30"]>0).mean()*100:>6.1f}% {s2:>8.1f}% {s3:>8.1f}% {s4:>8.1f}%')

    # ============ 分 score 看信号数细分 ============
    print('\n=== 按 score 分别看信号数 ===')
    for sc in [2, 3, 4]:
        sub_score = df_e[df_e['score']==sc]
        if len(sub_score) == 0: continue
        print(f'\n  score={sc} (总 n={len(sub_score):,}):')
        for lo, hi, label in bins:
            sub = sub_score[(sub_score['n_today']>=lo)&(sub_score['n_today']<=hi)]
            if len(sub) == 0: continue
            print(f'    {label:<14} n={len(sub):>5,} avg {sub["ret_30"].mean():>+6.2f}% win {(sub["ret_30"]>0).mean()*100:>5.1f}%')

    # ============ 试: 最大化"少信号日 + 高 score" ============
    print('\n=== 最佳组合: score 高 + 信号数少 ===')
    print(f'  {"组合":<30} {"事件":>6} {"avg ret":>8} {"win%":>7}')
    combos = [
        ('score=4 + n_today<=10', (df_e['score']==4) & (df_e['n_today']<=10)),
        ('score=4 + n_today<=20', (df_e['score']==4) & (df_e['n_today']<=20)),
        ('score=4 + n_today<=50', (df_e['score']==4) & (df_e['n_today']<=50)),
        ('score=4 任意', df_e['score']==4),
        ('score=3 + n_today<=10', (df_e['score']==3) & (df_e['n_today']<=10)),
        ('score=3 + n_today<=20', (df_e['score']==3) & (df_e['n_today']<=20)),
        ('score=3 + n_today<=50', (df_e['score']==3) & (df_e['n_today']<=50)),
        ('score=3 任意', df_e['score']==3),
        ('score>=2 + n_today<=10', (df_e['score']>=2) & (df_e['n_today']<=10)),
        ('score>=2 + n_today<=20', (df_e['score']>=2) & (df_e['n_today']<=20)),
        ('score>=2 + n_today<=50', (df_e['score']>=2) & (df_e['n_today']<=50)),
        ('score>=2 任意', df_e['score']>=2),
    ]
    for label, mask in combos:
        sub = df_e[mask]
        if len(sub):
            print(f'  {label:<30} {len(sub):>6,} {sub["ret_30"].mean():>+7.2f}% {(sub["ret_30"]>0).mean()*100:>6.1f}%')

    # 信号数 1-3 的日子, 这些是真"独家信号"
    print('\n=== 极致 — 信号数 1 / 2 / 3 的日子 ===')
    for n in [1, 2, 3]:
        sub = df_e[df_e['n_today']==n]
        days = (daily_count==n).sum()
        if len(sub):
            print(f'  n={n}: {days} 天, {len(sub):,} 事件, '
                  f'avg ret {sub["ret_30"].mean():>+6.2f}%, win {(sub["ret_30"]>0).mean()*100:>5.1f}%, '
                  f'score 分布: 2={int((sub["score"]==2).sum())}, 3={int((sub["score"]==3).sum())}, 4={int((sub["score"]>=4).sum())}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
