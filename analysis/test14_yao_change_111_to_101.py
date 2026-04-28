# -*- coding: utf-8 -*-
"""验证: 日卦 乾(111)→离(101) 是否为卖点

两条流:
  A. 大盘日卦 d_gua: 111→101 当日 → 看大盘 N 天收益 (验证大盘卖出时机)
  B. 个股日卦 d_gua: 111→101 当日 → 看个股 N 天收益 (验证个股卖出时机)

期望: 如果是真卖点, N 天后收益应该显著为负.

输出:
  事件次数, 事件后 1/3/5/10/20 日收益的均值/中位/胜率
  跟基线 (随机日) 对比, 看是否显著
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


HOLD_DAYS = [1, 3, 5, 10, 20]


def load_data():
    """加载大盘 + 个股 12 窗口卦数据"""
    print('=== 加载数据 ===')
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'close', 'd_gua'])
    market['date'] = market['date'].astype(str)
    market['d_gua'] = market['d_gua'].astype(str).str.zfill(3)
    market = market.sort_values('date').reset_index(drop=True)

    stocks = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                              columns=['date', 'code', 'd_gua'])
    stocks['date'] = stocks['date'].astype(str)
    stocks['code'] = stocks['code'].astype(str).str.zfill(6)
    stocks['d_gua'] = stocks['d_gua'].astype(str).str.zfill(3)

    # 加载 stocks daily close (从 stocks.parquet)
    print('  读取 stocks.parquet (close 列)...')
    stk_close = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                                 columns=['date', 'code', 'close'])
    stk_close['date'] = stk_close['date'].astype(str)
    stk_close['code'] = stk_close['code'].astype(str).str.zfill(6)
    return market, stocks, stk_close


def find_change_events(df, gua_col, from_g, to_g):
    """找出所有 from_g → to_g 切换事件的日期 (按 date 顺序, 去重相邻同向)"""
    df = df.sort_values('date').reset_index(drop=True)
    df['prev'] = df[gua_col].shift(1)
    mask = (df['prev'] == from_g) & (df[gua_col] == to_g)
    return df.loc[mask, 'date'].tolist()


def compute_forward_returns(df, event_dates, hold_days, close_col='close'):
    """对每个事件日期, 计算事件次日开始 hold_days 后的收益率"""
    df = df.sort_values('date').reset_index(drop=True)
    df['date_idx'] = range(len(df))
    date2idx = dict(zip(df['date'], df['date_idx']))
    closes = df[close_col].values

    rets = []
    for d in event_dates:
        i = date2idx.get(d)
        if i is None or i + hold_days >= len(closes):
            continue
        # 事件日 i, 收盘买入 (或日卦确认日开盘 i+1 买)
        # 简化: 用事件日 close 作买入价, i+hold_days close 作卖出价
        # 因为目的是验证"卖点", 我们想"如果继续持有", 看跌多少
        c_now = closes[i]
        c_then = closes[i + hold_days]
        if c_now > 0:
            rets.append((c_then / c_now - 1) * 100)
    return rets


def baseline_random_returns(df, hold_days, n_samples=5000, seed=42, close_col='close'):
    """基线: 随机日的 N 天后收益, 作为对照 alpha 计算"""
    df = df.sort_values('date').reset_index(drop=True)
    closes = df[close_col].values
    rng = np.random.RandomState(seed)
    n = len(closes) - hold_days
    if n <= 0:
        return []
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    rets = []
    for i in idx:
        c0 = closes[i]
        c1 = closes[i + hold_days]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return rets


def main():
    market, stocks, stk_close = load_data()

    # === A. 大盘日卦 111→101 ===
    print('\n' + '=' * 90)
    print('# A. 大盘日卦 111→101 (乾变离) 验证')
    print('=' * 90)
    events_m = find_change_events(market, 'd_gua', '111', '101')
    print(f'事件次数: {len(events_m)}')
    if len(events_m) > 0:
        print(f'前 5 个事件日: {events_m[:5]}')
        print(f'后 5 个事件日: {events_m[-5:]}')

    print(f'\n  {"持有天":<8} {"事件 N":>7} {"事件均收%":>10} {"事件中位%":>10} {"胜率%":>7} '
          f'{"基线均%":>9} {"alpha%":>8} {"alpha CI95%":>16}')
    print('  ' + '-' * 90)
    for h in HOLD_DAYS:
        rets = compute_forward_returns(market, events_m, h)
        if not rets:
            continue
        base = baseline_random_returns(market, h)
        alpha = np.mean(rets) - np.mean(base)
        # bootstrap CI for alpha (event mean - base mean)
        rng = np.random.RandomState(42)
        boots = []
        for _ in range(1000):
            r1 = rng.choice(rets, len(rets), replace=True).mean()
            r2 = rng.choice(base, len(base), replace=True).mean()
            boots.append(r1 - r2)
        ci = np.percentile(boots, [2.5, 97.5])
        win = (np.array(rets) > 0).mean() * 100
        print(f'  {h:<8} {len(rets):>7} {np.mean(rets):>+9.2f} {np.median(rets):>+9.2f} {win:>6.1f} '
              f'{np.mean(base):>+8.2f} {alpha:>+7.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]')

    # === B. 个股日卦 111→101 ===
    print('\n' + '=' * 90)
    print('# B. 个股日卦 111→101 (乾变离) 验证')
    print('=' * 90)
    print('  按 code 分组找事件...')

    # 找所有个股的 111→101 事件
    stocks_sorted = stocks.sort_values(['code', 'date']).reset_index(drop=True)
    stocks_sorted['prev_d'] = stocks_sorted.groupby('code')['d_gua'].shift(1)
    events_s = stocks_sorted[
        (stocks_sorted['prev_d'] == '111') & (stocks_sorted['d_gua'] == '101')
    ][['date', 'code']].copy()
    print(f'  事件次数 (个股): {len(events_s)}')

    # 加载 close
    print('  merge close ...')
    stk_close_idx = stk_close.set_index(['code', 'date'])['close']
    events_s['c_now'] = events_s.set_index(['code', 'date']).index.map(stk_close_idx)

    # 按 (code, date) 算 N 天后 close
    print('  计算 forward returns (按 code 分组)...')
    code_dates = stk_close.sort_values(['code', 'date']).reset_index(drop=True)

    # 给每只票排索引, 然后按 (code, date) 找 i+h 的 close
    code_dates['idx_in_code'] = code_dates.groupby('code').cumcount()
    pos_lookup = {(row['code'], row['date']): row['idx_in_code']
                   for _, row in code_dates.iterrows()}

    # 按 code 取 close 数组
    closes_by_code = {code: g['close'].values for code, g in code_dates.groupby('code')}

    print(f'  {"持有天":<8} {"事件 N":>7} {"事件均收%":>10} {"事件中位%":>10} {"胜率%":>7} '
          f'{"基线均%":>9} {"alpha%":>8} {"alpha CI95%":>16}')
    print('  ' + '-' * 90)

    for h in HOLD_DAYS:
        rets = []
        for _, row in events_s.iterrows():
            code = row['code']; d = row['date']
            i = pos_lookup.get((code, d))
            if i is None: continue
            arr = closes_by_code.get(code)
            if arr is None or i + h >= len(arr): continue
            c0 = arr[i]; c1 = arr[i + h]
            if c0 > 0:
                rets.append((c1 / c0 - 1) * 100)
        if not rets:
            continue
        # 基线: 个股随机日 (sample 5000 次)
        rng = np.random.RandomState(42)
        base = []
        all_codes = list(closes_by_code.keys())
        for _ in range(5000):
            code = rng.choice(all_codes)
            arr = closes_by_code[code]
            if len(arr) <= h: continue
            i = rng.randint(0, len(arr) - h)
            c0 = arr[i]; c1 = arr[i + h]
            if c0 > 0:
                base.append((c1 / c0 - 1) * 100)
        alpha = np.mean(rets) - np.mean(base)
        rng2 = np.random.RandomState(42)
        boots = []
        for _ in range(1000):
            r1 = rng2.choice(rets, len(rets), replace=True).mean()
            r2 = rng2.choice(base, min(len(base), 1000), replace=True).mean()
            boots.append(r1 - r2)
        ci = np.percentile(boots, [2.5, 97.5])
        win = (np.array(rets) > 0).mean() * 100
        print(f'  {h:<8} {len(rets):>7} {np.mean(rets):>+9.2f} {np.median(rets):>+9.2f} {win:>6.1f} '
              f'{np.mean(base):>+8.2f} {alpha:>+7.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]')


if __name__ == '__main__':
    main()
