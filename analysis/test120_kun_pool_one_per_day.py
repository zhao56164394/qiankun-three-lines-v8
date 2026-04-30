# -*- coding: utf-8 -*-
"""坤+入池 — 每日 1 只 实测策略

基于 test119 发现:
  - 信号越多, ret 越高 (大盘反转日)
  - 少信号日 (1-2 只) 是噪音, ret -1.96%
  - score=3 在多信号日 +12.85% / 78% 胜率
  - score=2 在多信号日 +6.78% / 62% 胜率

策略: 每日只买 1 只
  方案 A: 任意 score>=2, 每日 Top 1 (按 score+code)
  方案 B: 每日只在有 score=3 时买 (优先 score=3)
  方案 C: 每日只在 n_today >= 50 时买 (大反转日才入场)
  方案 D: 每日只在 n_today >= 50 + score>=3 时买 (双过滤)
  方案 E: 每日 1 只 + 同 score 内按低价排
  方案 F: 每日 1 只 + 同 score 内按散户线最低排 (sanhu_5d 升)

输出: 各方案的 触发日数 / avg ret / 胜率 / 累乘期末 (假设 30 日复利)
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


def main():
    t0 = time.time()
    print('=== 坤+入池 每日 1 只实测 ===\n')

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
    df['mf_5d'] = df.groupby('code', sort=False)['main_force'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    df['sanhu_5d'] = df.groupby('code', sort=False)['retail'].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
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
    mf5_arr = df['mf_5d'].to_numpy().astype(np.float64)
    sh5_arr = df['sanhu_5d'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫入场事件...')
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

                score = 0
                if mkt_m_arr[gi] == '100': score += 1
                if mkt_d_arr[gi] == '011': score += 1
                if mkt_m_arr[gi] == '010': score += 1
                if stk_m_arr[gi] == '010': score += 1

                ret_30 = (close_arr[gi+EVAL_WIN] / close_arr[gi] - 1) * 100
                events.append({
                    'date': date_arr[gi], 'code': code_arr[gi],
                    'score': score, 'ret_30': ret_30,
                    'close': close_arr[gi],
                    'mf_5d': mf5_arr[gi],
                    'sanhu_5d': sh5_arr[gi],
                })
                in_pool = False

    df_e = pd.DataFrame(events)
    daily_count = df_e.groupby('date').size()
    df_e['n_today'] = df_e['date'].map(daily_count)

    print(f'  事件: {len(df_e):,}, 涉及 {df_e["date"].nunique()} 天')

    # ============ 各方案 ============
    schemes = []

    # A: 任意 score>=2, 每日 Top 1 (按 score 降, code 升)
    df_a = df_e[df_e['score']>=2].sort_values(['date','score','code'],
                                                  ascending=[True,False,True])
    df_a = df_a.drop_duplicates('date', keep='first')
    schemes.append(('A: score>=2 每日 1 只 (按 score+code)', df_a))

    # B: 每日只在 score=3 信号有时买
    df_b = df_e[df_e['score']>=3].sort_values(['date','score','code'],
                                                  ascending=[True,False,True])
    df_b = df_b.drop_duplicates('date', keep='first')
    schemes.append(('B: 仅 score>=3 日 (每日 1 只)', df_b))

    # C: n_today >= 50 + score>=2
    df_c = df_e[(df_e['n_today']>=50) & (df_e['score']>=2)].sort_values(['date','score','code'],
                                                                            ascending=[True,False,True])
    df_c = df_c.drop_duplicates('date', keep='first')
    schemes.append(('C: n_today>=50 日, score>=2 每日 1 只', df_c))

    # D: n_today >= 50 + score>=3
    df_d = df_e[(df_e['n_today']>=50) & (df_e['score']>=3)].sort_values(['date','score','code'],
                                                                            ascending=[True,False,True])
    df_d = df_d.drop_duplicates('date', keep='first')
    schemes.append(('D: n_today>=50 日 + score>=3 (每日 1 只)', df_d))

    # E: score>=2 + 同 score 内按 close 升
    df_x = df_e[df_e['score']>=2].sort_values(['date','score','close','code'],
                                                  ascending=[True,False,True,True])
    df_e2 = df_x.drop_duplicates('date', keep='first')
    schemes.append(('E: score>=2 + 低价优先 (close↑)', df_e2))

    # F: score>=2 + 同 score 内按 sanhu_5d 升
    df_x = df_e[df_e['score']>=2].sort_values(['date','score','sanhu_5d','code'],
                                                  ascending=[True,False,True,True])
    df_f = df_x.drop_duplicates('date', keep='first')
    schemes.append(('F: score>=2 + 散户低优先 (sanhu_5d↑)', df_f))

    # G: score>=2 + 同 score 内按 mf_5d 降 (主力强)
    df_x = df_e[df_e['score']>=2].sort_values(['date','score','mf_5d','code'],
                                                  ascending=[True,False,False,True])
    df_g = df_x.drop_duplicates('date', keep='first')
    schemes.append(('G: score>=2 + 主力强优先 (mf_5d↓)', df_g))

    # H: 所有信号, 不限 score
    df_x = df_e.sort_values(['date','score','code'], ascending=[True,False,True])
    df_h = df_x.drop_duplicates('date', keep='first')
    schemes.append(('H: 任意信号 (score>=0) 每日 1 只', df_h))

    # ============ 对比 ============
    print('\n=== 各方案 (每日 1 只) 对比 ===\n')
    print(f'  {"方案":<42} {"触发日":>5} {"avg ret":>9} {"win%":>7} {"中位 ret":>9} {"max":>7} {"min":>7}')
    for label, df_x in schemes:
        if len(df_x) == 0: continue
        avg = df_x['ret_30'].mean()
        win = (df_x['ret_30']>0).mean() * 100
        med = df_x['ret_30'].median()
        mx = df_x['ret_30'].max()
        mn = df_x['ret_30'].min()
        print(f'  {label:<42} {len(df_x):>5} {avg:>+8.2f}% {win:>6.1f}% '
              f'{med:>+7.2f}% {mx:>+6.1f}% {mn:>+6.1f}%')

    # 模拟资金回测 (单只全仓, 30 日复利)
    print('\n=== 模拟单只全仓 30 日复利 (200K 起) ===')
    print('  注意: 这是单只满仓简化, 实际持仓周期可能重叠, 仅估算')
    print(f'  {"方案":<42} {"触发":>5} {"复利期末":>10} {"年化":>9}')
    for label, df_x in schemes:
        if len(df_x) == 0: continue
        # 累乘 (1 + ret/100), 但 30 日内不能再交易, 这里粗略用每个事件 1+ret 累乘
        # 因为事件之间可能有重叠, 这是"上限"估算
        df_x_sorted = df_x.sort_values('date')
        nav = 200_000
        # 顺序处理, 假设每次满仓
        for ret in df_x_sorted['ret_30'].values:
            nav *= (1 + ret / 100)
        # 时间跨度
        days = (pd.to_datetime(df_x_sorted['date'].iloc[-1]) - pd.to_datetime(df_x_sorted['date'].iloc[0])).days
        years = days / 365 if days > 0 else 1
        annual = ((nav/200_000)**(1/years) - 1)*100 if years > 0 else 0
        print(f'  {label:<42} {len(df_x):>5} ¥{nav/1000:>8.0f}K {annual:>+7.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
