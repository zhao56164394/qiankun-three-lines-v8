# -*- coding: utf-8 -*-
"""个股变卦买卖点 walk-forward 验证 (事件级简化模拟)

不重写 backtest, 直接事件层模拟:
  买入信号 = 个股变卦 X→Y (买点候选)
  卖出 = 持有 N 天后强制卖出 (固定持有期, 简化)

输出 7 段 walk-forward:
  - baseline (随机日买): 期望接近 0 alpha
  - 各候选信号: 段内累加 alpha vs baseline

判定: 5+/2- 真规律, 4+ 不稳定, 否则非规律
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

# 7 段 walk-forward 窗口
WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01', '2018 大熊'),
    ('w2_2019',    '2019-01-01', '2020-01-01', '2019 反弹'),
    ('w3_2020',    '2020-01-01', '2021-01-01', '2020 抱团'),
    ('w4_2021',    '2021-01-01', '2022-01-01', '2021 延续'),
    ('w5_2022',    '2022-01-01', '2023-01-01', '2022 杀跌'),
    ('w6_2023_24', '2023-01-01', '2025-01-01', '2023-24 震荡'),
    ('w7_2025_26', '2025-01-01', '2026-04-21', '2025-26 慢牛'),
]

# Top 候选信号 (来自 test18 跨持有期 ≥4/5 一致)
# (流, 类型, from, to, 期望)
BUY_CANDIDATES = [
    ('y_arr', '个股年卦', '111', '000', '+2.11'),
    ('y_arr', '个股年卦', '101', '000', '+1.05'),
    ('y_arr', '个股年卦', '111', '100', '+0.80'),
    ('y_arr', '个股年卦', '000', '010', '+0.71'),
    ('m_arr', '个股月卦', '010', '011', '+0.62'),
]

SELL_CANDIDATES = [
    ('y_arr', '个股年卦', '110', '101', '-2.58'),
    ('y_arr', '个股年卦', '111', '101', '-0.85'),
    ('y_arr', '个股年卦', '111', '001', '-0.81'),
    ('d_arr', '个股日卦', '011', '001', '-0.81'),
    ('d_arr', '个股日卦', '111', '001', '-0.67'),
]

HOLD_DAYS = 10  # 简化: 固定持有 10 日


def load_data():
    print('=== 加载数据 ===')
    t0 = time.time()
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                         columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str)
    g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                         columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str)
    p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  merged: {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].values
    date_arr = df['date'].values
    close_arr = df['close'].values.astype(np.float32)
    d_arr = df['d_gua'].values
    m_arr = df['m_gua'].values
    y_arr = df['y_gua'].values

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    return {
        'code_arr': code_arr, 'date_arr': date_arr, 'close_arr': close_arr,
        'd_arr': d_arr, 'm_arr': m_arr, 'y_arr': y_arr,
        'code_starts': code_starts, 'code_ends': code_ends,
    }


def find_changes(data, gua_arr_name, from_g, to_g):
    """找该变卦的全局 row_idx"""
    g = data[gua_arr_name]
    code_arr = data['code_arr']
    same_code = np.r_[False, code_arr[1:] == code_arr[:-1]]
    valid_g_prev = np.r_[False, g[:-1] == from_g]
    valid_g_now = (g == to_g)
    is_event = same_code & valid_g_prev & valid_g_now
    return np.where(is_event)[0]


def fwd_returns(data, event_idx, hold):
    close_arr = data['close_arr']
    code_starts = data['code_starts']
    code_ends = data['code_ends']
    code_seg = np.searchsorted(code_starts, event_idx, side='right') - 1
    end_of_code = code_ends[code_seg]
    valid_mask = (event_idx + hold) < end_of_code
    valid_idx = event_idx[valid_mask]
    if len(valid_idx) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype='U10')
    c0 = close_arr[valid_idx]
    c1 = close_arr[valid_idx + hold]
    rets_mask = c0 > 0
    valid_idx_final = valid_idx[rets_mask]
    rets = ((c1[rets_mask] / c0[rets_mask] - 1.0) * 100.0).astype(np.float32)
    dates_final = data['date_arr'][valid_idx_final]
    return rets, dates_final


def baseline_random(data, hold, n_in_window, dates_filter, seed=42):
    """随机 (code, idx), 限制日期在 dates_filter 内"""
    rng = np.random.RandomState(seed)
    code_starts = data['code_starts']; code_ends = data['code_ends']
    close_arr = data['close_arr']; date_arr = data['date_arr']
    n_codes = len(code_starts)
    rets = []
    attempts = 0
    while len(rets) < n_in_window and attempts < n_in_window * 100:
        attempts += 1
        c = rng.randint(0, n_codes)
        s = code_starts[c]; e = code_ends[c]
        if e - s <= hold + 1: continue
        i = rng.randint(s, e - hold - 1)
        if date_arr[i] not in dates_filter: continue
        c0 = close_arr[i]; c1 = close_arr[i + hold]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return np.array(rets, dtype=np.float32)


def main():
    t_all = time.time()
    data = load_data()

    # 全期日期集合 (按窗口切)
    all_dates = data['date_arr']

    print('\n' + '=' * 100)
    print('# Walk-forward: 个股变卦买卖点 7 段验证')
    print('=' * 100)

    # 对每个候选信号, 7 段验证
    all_results = []
    for cands, label, expected_sign in [(BUY_CANDIDATES, '★ 买点', '+'),
                                         (SELL_CANDIDATES, '✗ 卖点', '-')]:
        print(f'\n## {label}')
        print(f'  {"信号":<22} {"期望α":>7}  {"w1":>7} {"w2":>7} {"w3":>7} {"w4":>7} {"w5":>7} {"w6":>7} {"w7":>7}  {"判定":>10}')
        print('  ' + '-' * 110)

        for arr_name, stream, f, t, expected in cands:
            event_idx = find_changes(data, arr_name, f, t)
            if len(event_idx) == 0:
                continue
            rets_all, dates_all = fwd_returns(data, event_idx, HOLD_DAYS)

            row = [f'  {stream} {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]}']
            n_pos = 0; n_neg = 0; n_zero = 0
            cell_alphas = []
            for w_label, ws, we, _ in WINDOWS:
                # 段内事件
                mask = (dates_all >= ws) & (dates_all < we)
                if mask.sum() < 30:
                    row.append(f'{"-":>7}')
                    cell_alphas.append(None)
                    continue
                seg_rets = rets_all[mask]
                # 段内基线
                seg_dates_set = set(d for d in all_dates if ws <= d < we)
                base = baseline_random(data, HOLD_DAYS, 5000, seg_dates_set)
                if len(base) < 100:
                    row.append(f'{"-":>7}')
                    cell_alphas.append(None)
                    continue
                alpha = float(seg_rets.mean() - base.mean())
                cell_alphas.append(alpha)
                if expected_sign == '+':
                    if alpha > 0.1: n_pos += 1; marker = '✅'
                    elif alpha < -0.1: n_neg += 1; marker = '❌'
                    else: n_zero += 1; marker = '○ '
                else:
                    if alpha < -0.1: n_pos += 1; marker = '✅'
                    elif alpha > 0.1: n_neg += 1; marker = '❌'
                    else: n_zero += 1; marker = '○ '
                row.append(f'{alpha:>+5.2f}{marker}')

            n_valid = sum(1 for a in cell_alphas if a is not None)
            if n_valid >= 5:
                if n_pos >= 5 and n_neg <= 1: verdict = '★真规律'
                elif n_pos >= 4: verdict = '○不稳定'
                else: verdict = '✗非规律'
            else:
                verdict = '— 段不足'
            row.insert(1, f'{expected:>7}')  # 在第一列后插入期望α
            row.append(f'{verdict:>10}')
            print('  '.join(row))
            all_results.append({
                'stream': stream, 'from': f, 'to': t, 'sign': expected_sign,
                'n_pos': n_pos, 'n_neg': n_neg, 'n_zero': n_zero,
                'verdict': verdict, 'cell_alphas': cell_alphas,
            })

    print(f'\n=== 总耗时: {time.time()-t_all:.1f}s ===')

    # 汇总
    print('\n' + '=' * 80)
    print('# 反过拟合判定汇总')
    print('=' * 80)
    n_real = sum(1 for r in all_results if r['verdict'] == '★真规律')
    n_unst = sum(1 for r in all_results if r['verdict'] == '○不稳定')
    n_fake = sum(1 for r in all_results if r['verdict'] == '✗非规律')
    print(f'  ★ 真规律: {n_real}/{len(all_results)}')
    print(f'  ○ 不稳定: {n_unst}/{len(all_results)}')
    print(f'  ✗ 非规律: {n_fake}/{len(all_results)}')


if __name__ == '__main__':
    main()
