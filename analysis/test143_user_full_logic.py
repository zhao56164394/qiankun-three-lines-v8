# -*- coding: utf-8 -*-
"""验证用户完整逻辑链 — 建仓 + 清仓

入场组件:
  E1: retail<-250 池中
  E2: mf 第 1 次上穿 50
  E3: retail 上升 (today > yesterday)

清仓组件 (单独测每一种, 不组合):
  X1: 个股 d_gua = 震 (100)
  X2: 连续 N 天涨停 (close 涨幅 >= 9.7%)
  X3: mf 5 日下降斜率 (mf 5 日内跌 > 200)
  X4: trend < 11
  X5: 30 日浮盈 timeout
  对比组: bull_2nd / bull_1st

每种 vs 现有 S0 (bull_2nd)
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
LOOKBACK = 30


def find_signals_user_logic(arrays):
    """E1+E2+E3: retail<-250 池中, mf 第1次上穿 50, retail 上升"""
    code_starts = arrays['starts']; code_ends = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']

    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            # E1: 入池 (上沿穿透)
            if not in_pool and cur_below and not prev_below:
                in_pool = True

            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)  # E2
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)  # E3

            # 同日满足三条件 → 建仓
            if in_pool and mf_cross_up and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                    'cur_retail': retail[gi],
                    'cur_mf': mf[gi],
                    'mf_chg': mf[gi] - last_mf if not np.isnan(last_mf) else 0,
                    'retail_chg': retail[gi] - last_retail if not np.isnan(last_retail) else 0,
                })
                in_pool = False  # 触发后出池

            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below

    return pd.DataFrame(events)


def sell_X1(buy_idx, td, close, gua, max_end):
    """个股 d_gua = 震 (100) 清仓"""
    bp = close[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        if gua[k] == '100':
            return k, 'zhen', (close[k]/bp-1)*100
        if k - buy_idx >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_X2(buy_idx, td, close, gua, max_end, n_consec=3):
    """连续 N 天涨停 (>=9.7%) 后清仓"""
    bp = close[buy_idx]
    consec = 0
    for k in range(buy_idx + 1, max_end + 1):
        if k > 0 and not np.isnan(close[k-1]) and close[k-1] > 0:
            chg = (close[k]/close[k-1] - 1) * 100
            if chg >= 9.7:
                consec += 1
                if consec >= n_consec:
                    return k, f'{n_consec}_limit_up', (close[k]/bp-1)*100
            else:
                consec = 0
        if k - buy_idx >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_X3(buy_idx, td, close, gua, mf_arr, max_end, mf_drop=200):
    """5 日 mf 累计下降 > 200 清仓"""
    bp = close[buy_idx]
    for k in range(buy_idx + 5, max_end + 1):
        seg = mf_arr[k-4:k+1]
        if not np.any(np.isnan(seg)) and (seg.max() - seg[-1]) > mf_drop:
            return k, f'mf_drop_{mf_drop}', (close[k]/bp-1)*100
        if k - buy_idx >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_X4(buy_idx, td, close, gua, max_end):
    """trend<11 清仓"""
    bp = close[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            return k, 'td_below_11', (close[k]/bp-1)*100
        if k - buy_idx >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_X5(buy_idx, td, close, gua, max_end):
    """30 日固定 timeout"""
    bp = close[buy_idx]
    k = min(max_end, buy_idx + 30)
    return k, 'fix_30d', (close[k]/bp-1)*100


def sell_bull2(buy_idx, td, close, gua, max_end):
    """对比: bull_2nd / TS20 / 60d"""
    bp = close[buy_idx]
    cross_count = 0
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx
        if not np.isnan(td[k]):
            running_max = max(running_max, td[k])
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count >= 2:
                return k, 'bull_2nd', (close[k]/bp-1)*100
        if days >= 20:
            seg = td[buy_idx:k+1]
            valid = seg[~np.isnan(seg)]
            if len(valid) > 0 and valid.max() < 89:
                return k, 'ts20', (close[k]/bp-1)*100
        if days >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def sell_bull1(buy_idx, td, close, gua, max_end):
    """bull_1st 对比"""
    bp = close[buy_idx]
    crossed = False
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] >= 89:
            crossed = True
        if crossed and not np.isnan(td[k]) and td[k] < 89 and td[k-1] >= 89:
            return k, 'bull_1st', (close[k]/bp-1)*100
        if k - buy_idx >= MAX_HOLD:
            return k, 'timeout', (close[k]/bp-1)*100
    return max_end, 'fc', (close[max_end]/bp-1)*100


def main():
    t0 = time.time()
    print('=== 用户完整逻辑: 建仓 + 多种清仓 ===\n')

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

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'mf': df['main_force'].to_numpy().astype(np.float64),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    gua_arr = df['stk_d'].to_numpy()
    mf_arr = arrays['mf']

    df_e = find_signals_user_logic(arrays)
    print(f'  E1+E2+E3 信号: {len(df_e):,}')

    # 测各种清仓
    rows = {label: [] for label in ['X1', 'X2_3', 'X2_2', 'X3_200', 'X3_300', 'X4', 'X5', 'bull2', 'bull1']}

    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)

        for label, fn in [
            ('X1', lambda *a: sell_X1(gi, trend_arr, close_arr, gua_arr, max_end)),
            ('X2_3', lambda *a: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 3)),
            ('X2_2', lambda *a: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 2)),
            ('X3_200', lambda *a: sell_X3(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, 200)),
            ('X3_300', lambda *a: sell_X3(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, 300)),
            ('X4', lambda *a: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end)),
            ('X5', lambda *a: sell_X5(gi, trend_arr, close_arr, gua_arr, max_end)),
            ('bull2', lambda *a: sell_bull2(gi, trend_arr, close_arr, gua_arr, max_end)),
            ('bull1', lambda *a: sell_bull1(gi, trend_arr, close_arr, gua_arr, max_end)),
        ]:
            si, r, ret = fn()
            rows[label].append({'date': ev['date'], 'code': ev['code'],
                                  'days': si - gi, 'reason': r, 'ret_pct': ret})

    print('\n=== 各清仓模式单事件级 ===\n')
    print(f'  {"模式":<28} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>5} {"max":>7} {"min":>7}')
    desc = {
        'X1': 'X1 震卦 (100) 清仓',
        'X2_3': 'X2 连 3 涨停 清仓',
        'X2_2': 'X2 连 2 涨停 清仓',
        'X3_200': 'X3 5日 mf 跌 200 清仓',
        'X3_300': 'X3 5日 mf 跌 300 清仓',
        'X4': 'X4 trend<11 清仓',
        'X5': 'X5 固定 30 日 timeout',
        'bull2': 'bull_2nd (现有 baseline)',
        'bull1': 'bull_1st 简单',
    }
    summary = {}
    for label in ['bull2', 'X1', 'X2_3', 'X2_2', 'X3_200', 'X3_300', 'X4', 'X5', 'bull1']:
        df_x = pd.DataFrame(rows[label])
        avg = df_x['ret_pct'].mean()
        win = (df_x['ret_pct']>0).mean()*100
        med = df_x['ret_pct'].median()
        days = df_x['days'].mean()
        mx = df_x['ret_pct'].max()
        mn = df_x['ret_pct'].min()
        summary[label] = avg
        print(f'  {desc[label]:<28} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}% {days:>4.1f}d '
              f'{mx:>+6.1f}% {mn:>+6.1f}%')

    # 神火 / 顺丰 各清仓
    print('\n=== 神火 vs 顺丰 各清仓表现 ===\n')
    for code, dt in [('000933', '2016-02-17'), ('002352', '2016-01-19')]:
        print(f'  {code} {dt}:')
        for label in ['bull2', 'X1', 'X2_2', 'X2_3', 'X3_200', 'X4', 'bull1']:
            df_x = pd.DataFrame(rows[label])
            sub = df_x[(df_x['code'] == code) & (df_x['date'] == dt)]
            if len(sub):
                r = sub.iloc[0]
                print(f'    {label}: {r["ret_pct"]:>+7.2f}% / {r["days"]:>3}d / {r["reason"]}')
            else:
                print(f'    {label}: 无 (该日无信号)')

    # 组合策略: 多卖点取最早触发者
    print('\n=== 组合卖点 (取最早触发) ===\n')

    def sell_combo(buy_idx, td, close, gua, mf, max_end, parts):
        """parts: 卖点函数列表, 返回每个的 sell_idx, 取最早"""
        results = []
        for fn in parts:
            si, r, ret = fn()
            results.append((si, r, ret))
        # 取最早
        results.sort(key=lambda x: x[0])
        return results[0]

    rows_combo = {label: [] for label in ['C1', 'C2', 'C3', 'C4', 'C5']}
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)

        # C1: 震 + trend<11 + 60d
        si, r, ret = sell_combo(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, [
            lambda: sell_X1(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end),
        ])
        rows_combo['C1'].append({'date':ev['date'],'code':ev['code'],'days':si-gi,'reason':r,'ret_pct':ret})

        # C2: 震 + trend<11 + 连 2 涨停
        si, r, ret = sell_combo(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, [
            lambda: sell_X1(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 2),
        ])
        rows_combo['C2'].append({'date':ev['date'],'code':ev['code'],'days':si-gi,'reason':r,'ret_pct':ret})

        # C3: 震 + trend<11 + 连 2 涨停 + mf 跌 200
        si, r, ret = sell_combo(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, [
            lambda: sell_X1(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 2),
            lambda: sell_X3(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, 200),
        ])
        rows_combo['C3'].append({'date':ev['date'],'code':ev['code'],'days':si-gi,'reason':r,'ret_pct':ret})

        # C4: bull_2nd + trend<11 + 连 2 涨停 (现有 + 你的清仓)
        si, r, ret = sell_combo(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, [
            lambda: sell_bull2(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 2),
        ])
        rows_combo['C4'].append({'date':ev['date'],'code':ev['code'],'days':si-gi,'reason':r,'ret_pct':ret})

        # C5: bull_2nd + trend<11 + 连 2 涨停 + mf 跌 200
        si, r, ret = sell_combo(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, [
            lambda: sell_bull2(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X4(gi, trend_arr, close_arr, gua_arr, max_end),
            lambda: sell_X2(gi, trend_arr, close_arr, gua_arr, max_end, 2),
            lambda: sell_X3(gi, trend_arr, close_arr, gua_arr, mf_arr, max_end, 200),
        ])
        rows_combo['C5'].append({'date':ev['date'],'code':ev['code'],'days':si-gi,'reason':r,'ret_pct':ret})

    print(f'  {"组合":<32} {"avg_ret":>9} {"win%":>7} {"中位":>9} {"持仓":>5} {"max":>7} {"min":>7}')
    for label, name in [
        ('C1', 'C1: 震+trend<11'),
        ('C2', 'C2: 震+trend<11+连2涨停'),
        ('C3', 'C3: C2+mf 跌200'),
        ('C4', 'C4: bull2+trend<11+连2涨停'),
        ('C5', 'C5: C4+mf 跌200'),
    ]:
        df_x = pd.DataFrame(rows_combo[label])
        avg = df_x['ret_pct'].mean()
        win = (df_x['ret_pct']>0).mean()*100
        med = df_x['ret_pct'].median()
        days = df_x['days'].mean()
        mx = df_x['ret_pct'].max()
        mn = df_x['ret_pct'].min()
        print(f'  {name:<32} {avg:>+8.2f}% {win:>6.1f}% {med:>+7.2f}% {days:>4.1f}d '
              f'{mx:>+6.1f}% {mn:>+6.1f}%')

    # 神火/顺丰 在组合下
    print('\n=== 神火 vs 顺丰 组合表现 ===\n')
    for code, dt in [('000933', '2016-02-17'), ('002352', '2016-01-19')]:
        print(f'  {code} {dt}:')
        for label in ['C1', 'C2', 'C3', 'C4', 'C5']:
            df_x = pd.DataFrame(rows_combo[label])
            sub = df_x[(df_x['code'] == code) & (df_x['date'] == dt)]
            if len(sub):
                r = sub.iloc[0]
                print(f'    {label}: {r["ret_pct"]:>+7.2f}% / {r["days"]:>3}d / {r["reason"]}')

    # 组合策略 reason 分布
    print('\n=== C5 reason 分布 ===')
    df_c5 = pd.DataFrame(rows_combo['C5'])
    for r, cnt in df_c5['reason'].value_counts().items():
        sub = df_c5[df_c5['reason'] == r]
        print(f'  {r:<18} n={cnt:>4} ret={sub["ret_pct"].mean():>+5.2f}% '
              f'win={(sub["ret_pct"]>0).mean()*100:>5.1f}% hold={sub["days"].mean():.1f}d')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
