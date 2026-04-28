# -*- coding: utf-8 -*-
"""Step 30 — bear/bull/M3+止损/M1+止损 在 v2 买入事件上验证

机制:
  B1 bear (项目现成): 50~89 双降 (trend↓ + retail↓) OR 首穿89
  B2 bull (项目现成): 第二次穿 89
  M3+SL (下穿89 + 止损 -8%): 任一触发
  M1+SL (乾→其他 + 止损 -8%): 任一触发

通用兜底: 持仓 60 日强卖
评估窗口: 60 日
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
QIAN_RUN = 10
HARD_TIMEOUT = 60
STOP_LOSS = -8.0  # 止损阈值 %
REGIME_Y = '000'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
]

AVOID = [
    ('mkt_d', '000'), ('mkt_d', '001'), ('mkt_d', '100'), ('mkt_d', '101'),
    ('stk_y', '001'), ('stk_y', '011'),
    ('stk_m', '101'), ('stk_m', '110'), ('stk_m', '111'),
]


def sell_bear(buy_idx, td, retail, cl, end_idx):
    """50~89 双降 OR 首穿 89"""
    n = len(td)
    end = min(end_idx, n - 1)
    running_max = td[buy_idx]
    for k in range(buy_idx + 1, end + 1):
        if np.isnan(td[k]) or np.isnan(retail[k]): continue
        if k == 0: continue
        running_max = max(running_max, td[k])
        if np.isnan(td[k-1]) or np.isnan(retail[k-1]): continue
        # 50-89 双降
        if running_max >= 50 and td[k] < 89:
            if td[k] < td[k-1] and retail[k] < retail[k-1]:
                return k, 'bear_double'
        # 首穿 89
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            return k, 'bear_cross89'
    return end, 'timeout'


def sell_bull(buy_idx, td, retail, cl, end_idx):
    """第二次穿 89"""
    n = len(td)
    end = min(end_idx, n - 1)
    running_max = td[buy_idx]
    cross_count = 0
    for k in range(buy_idx + 1, end + 1):
        if np.isnan(td[k]) or np.isnan(retail[k]): continue
        if k == 0: continue
        running_max = max(running_max, td[k])
        if np.isnan(td[k-1]): continue
        if running_max >= 89 and td[k] < 89 and td[k-1] >= 89:
            cross_count += 1
            if cross_count == 2:
                return k, 'bull_cross89_2nd'
    return end, 'timeout'


def sell_m3_sl(buy_idx, td, cl, end_idx, buy_close):
    """下穿 89 OR 浮亏 -8%"""
    n = len(td)
    end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        # 止损
        ret = (cl[k] / buy_close - 1) * 100
        if ret <= STOP_LOSS:
            return k, 'stop_loss'
        # 下穿 89
        if k > 0 and td[k-1] > 89 and td[k] <= 89:
            return k, 'cross89'
    return end, 'timeout'


def sell_m1_sl(buy_idx, gua, cl, end_idx, buy_close):
    """乾→其他 OR 浮亏 -8%"""
    n = len(gua)
    end = min(end_idx, n - 1)
    for k in range(buy_idx + 1, end + 1):
        ret = (cl[k] / buy_close - 1) * 100
        if ret <= STOP_LOSS:
            return k, 'stop_loss'
        if gua[k-1] == '111' and gua[k] != '111':
            return k, 'qian_change'
    return end, 'timeout'


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
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
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_d']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    retail_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_d_arr = df['mkt_d'].to_numpy(); mkt_m_arr = df['mkt_m'].to_numpy(); mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 扫 v2 买入
    print(f'\n=== 扫 v2 买入事件 ===')
    avoid_arr_map = {'mkt_d': mkt_d_arr, 'mkt_m': mkt_m_arr,
                     'stk_y': stk_y_arr, 'stk_m': stk_m_arr}
    buy_events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + HARD_TIMEOUT + 5: continue
        gua = stk_d_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - HARD_TIMEOUT):
            gi = s + i
            if mkt_y_arr[gi] != REGIME_Y: continue
            if gua[i] != '011': continue
            avoid = False
            for col, val in AVOID:
                if avoid_arr_map[col][gi] == val:
                    avoid = True; break
            if avoid: continue
            score = 0
            if mkt_m_arr[gi] == '100': score += 1
            if mkt_d_arr[gi] == '011': score += 1
            if mf_arr[gi] > 100: score += 1
            if stk_m_arr[gi] == '010': score += 1
            if score < 2: continue
            buy_events.append((gi, ci, score))
    print(f'  v2 买入: {len(buy_events):,}')

    # 模拟 4 种
    print(f'\n=== 模拟 4 种卖点 ===')
    t1 = time.time()
    results = {'B1_bear': [], 'B2_bull': [], 'M3_SL': [], 'M1_SL': []}
    for gi, ci, score in buy_events:
        s = code_starts[ci]; e = code_ends[ci]
        local_buy = gi - s
        cl_seg = close_arr[s:e]
        gua_seg = stk_d_arr[s:e]
        td_seg = trend_arr[s:e]
        retail_seg = retail_arr[s:e]
        n_local = len(gua_seg)
        max_end = min(local_buy + HARD_TIMEOUT, n_local - 1)
        buy_close = cl_seg[local_buy]
        buy_date = date_arr[gi]
        # 60 日内乾天数
        n_qian_60 = (gua_seg[local_buy:max_end+1] == '111').sum()
        is_zsl = n_qian_60 >= QIAN_RUN

        common = {'date': buy_date, 'score': score, 'is_zsl': is_zsl, 'n_qian_60': int(n_qian_60)}

        # B1 bear
        sl, exit_t = sell_bear(local_buy, td_seg, retail_seg, cl_seg, max_end)
        results['B1_bear'].append({**common, 'hold': sl - local_buy,
                                     'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

        # B2 bull
        sl, exit_t = sell_bull(local_buy, td_seg, retail_seg, cl_seg, max_end)
        results['B2_bull'].append({**common, 'hold': sl - local_buy,
                                     'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

        # M3 + SL
        sl, exit_t = sell_m3_sl(local_buy, td_seg, cl_seg, max_end, buy_close)
        results['M3_SL'].append({**common, 'hold': sl - local_buy,
                                  'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

        # M1 + SL
        sl, exit_t = sell_m1_sl(local_buy, gua_seg, cl_seg, max_end, buy_close)
        results['M1_SL'].append({**common, 'hold': sl - local_buy,
                                  'ret': (cl_seg[sl] / buy_close - 1) * 100, 'exit': exit_t})

    print(f'  完成 {time.time()-t1:.1f}s')

    # === 全样本对比 ===
    print(f'\n## 4 机制对比 ({len(buy_events):,} v2 买入)')
    print(f'  {"机制":<10} {"期望%":>7} {"中位%":>7} {"胜率":>6} {"均持仓":>6} {"最大%":>7} {"最小%":>7}')
    print('  ' + '-' * 65)
    for m in ['B1_bear', 'B2_bull', 'M3_SL', 'M1_SL']:
        d = pd.DataFrame(results[m])
        ret_m = d['ret'].mean(); ret_med = d['ret'].median()
        win = (d['ret'] > 0).mean() * 100
        hold = d['hold'].mean()
        print(f'  {m:<10} {ret_m:>+6.2f}% {ret_med:>+6.2f}% {win:>5.1f}% {hold:>5.1f} '
              f'{d["ret"].max():>+6.1f}% {d["ret"].min():>+6.1f}%')

    # 主升 vs 假突破
    print(f'\n## 主升浪 vs 假突破 拆解')
    print(f'  {"机制":<10} {"主升期望":>9} {"主升持仓":>9} {"主升胜率":>9} {"假期望":>8} {"假持仓":>8}')
    print('  ' + '-' * 70)
    for m in ['B1_bear', 'B2_bull', 'M3_SL', 'M1_SL']:
        d = pd.DataFrame(results[m])
        zsl = d[d['is_zsl']]; fake = d[~d['is_zsl']]
        zwin = (zsl['ret'] > 0).mean() * 100
        print(f'  {m:<10} {zsl["ret"].mean():>+7.2f}% {zsl["hold"].mean():>7.1f} {zwin:>7.1f}% '
              f'{fake["ret"].mean():>+7.2f}% {fake["hold"].mean():>7.1f}')

    # 退出类型
    print(f'\n## 退出类型分布')
    for m in ['B1_bear', 'B2_bull', 'M3_SL', 'M1_SL']:
        d = pd.DataFrame(results[m])
        exit_dist = d['exit'].value_counts(normalize=True) * 100
        print(f'  {m}:  ', end='')
        for k, v in exit_dist.items():
            print(f'{k}={v:.0f}%  ', end='')
        print()

    # walk-forward
    print(f'\n## walk-forward 各段期望%')
    print(f'  {"段":<14} {"baseline":>9}', end='')
    for m in ['B1_bear', 'B2_bull', 'M3_SL', 'M1_SL']:
        print(f' {m:>10}', end='')
    print()
    print('  ' + '-' * 80)
    # 用 M6=fixed30 作为 baseline 参考
    for w in WINDOWS:
        print(f'  {w[0]:<14}', end='')
        # baseline: 取所有 v2 事件该段的 ret_30 (用 B1_bear 数据中的 ret_30 = c[+30]/c[buy] 不太一样)
        # 这里用各机制的段均
        b1 = pd.DataFrame(results['B1_bear'])
        seg_b = b1[(b1['date'] >= w[1]) & (b1['date'] < w[2])]
        if len(seg_b) < 30:
            print(f'  样本不足 ({len(seg_b)})')
            continue
        # 没 baseline 列, 用 B1 自己作为参照
        print(f' {"":>9}', end='')
        for m in ['B1_bear', 'B2_bull', 'M3_SL', 'M1_SL']:
            d = pd.DataFrame(results[m])
            seg = d[(d['date'] >= w[1]) & (d['date'] < w[2])]
            if len(seg) < 30:
                print(f' {"--":>9}', end='')
            else:
                print(f' {seg["ret"].mean():>+8.2f}%', end='')
        print()


if __name__ == '__main__':
    main()
