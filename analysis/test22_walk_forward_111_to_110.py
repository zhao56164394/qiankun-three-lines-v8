# -*- coding: utf-8 -*-
"""验证 个股日卦 111乾→110兑 是否真买点 (walk-forward 7 段)"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOWS = [
    ('w1_2018', '2018-01-01', '2019-01-01', '2018 大熊'),
    ('w2_2019', '2019-01-01', '2020-01-01', '2019 反弹'),
    ('w3_2020', '2020-01-01', '2021-01-01', '2020 抱团'),
    ('w4_2021', '2021-01-01', '2022-01-01', '2021 延续'),
    ('w5_2022', '2022-01-01', '2023-01-01', '2022 杀跌'),
    ('w6_2023_24', '2023-01-01', '2025-01-01', '2023-24 震荡'),
    ('w7_2025_26', '2025-01-01', '2026-04-21', '2025-26 慢牛'),
]

HOLD_DAYS = 10


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                         columns=['date', 'code', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                         columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].values; date_arr = df['date'].values
    close_arr = df['close'].values.astype(np.float32)
    d_arr = df['d_gua'].values

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]
    same_code = np.r_[False, code_arr[1:] == code_arr[:-1]]

    # 测试候选: 111乾→110兑 (单变地位, 唯一 4/5 ★ 日卦买点)
    f, t = '111', '110'
    valid_prev = np.r_[False, d_arr[:-1] == f]
    valid_now = (d_arr == t)
    is_event = same_code & valid_prev & valid_now
    event_idx = np.where(is_event)[0]
    print(f'\n个股日卦 111→110 事件: {len(event_idx):,}')

    # forward returns
    code_seg = np.searchsorted(code_starts, event_idx, side='right') - 1
    end_of_code = code_ends[code_seg]
    valid_mask = (event_idx + HOLD_DAYS) < end_of_code
    valid_idx = event_idx[valid_mask]
    c0 = close_arr[valid_idx]; c1 = close_arr[valid_idx + HOLD_DAYS]
    rets_mask = c0 > 0
    valid_idx_final = valid_idx[rets_mask]
    rets = ((c1[rets_mask] / c0[rets_mask] - 1.0) * 100.0).astype(np.float32)
    dates_final = date_arr[valid_idx_final]
    print(f'有效收益样本: {len(rets):,}, 全期均α (vs 0.86 baseline): {rets.mean()-0.86:+.2f}%')

    # 按 7 段 walk-forward
    print(f'\n{"窗口":<14} {"事件 N":>7} {"事件均":>9} {"基线均":>9} {"alpha":>8}  {"判定":>6}')
    print('-' * 60)
    n_pos = 0; n_neg = 0; n_zero = 0
    rng = np.random.RandomState(42)
    for w_label, ws, we, _ in WINDOWS:
        mask = (dates_final >= ws) & (dates_final < we)
        if mask.sum() < 30:
            print(f'{w_label:<14} {mask.sum():>7}  样本不足'); continue
        seg_rets = rets[mask]
        # 段内随机基线 (从该段日期取)
        seg_dates = set(d for d in date_arr if ws <= d < we)
        n_codes = len(code_starts)
        base = []
        for _ in range(5000):
            code = rng.randint(0, n_codes)
            s = code_starts[code]; e = code_ends[code]
            if e - s <= HOLD_DAYS + 1: continue
            i = rng.randint(s, e - HOLD_DAYS - 1)
            if date_arr[i] not in seg_dates: continue
            c0_ = close_arr[i]; c1_ = close_arr[i + HOLD_DAYS]
            if c0_ > 0:
                base.append((c1_ / c0_ - 1) * 100)
            if len(base) >= 1000: break
        base_mean = np.mean(base) if base else 0
        alpha = float(seg_rets.mean() - base_mean)
        marker = '✅' if alpha > 0.1 else ('❌' if alpha < -0.1 else '○ ')
        if alpha > 0.1: n_pos += 1
        elif alpha < -0.1: n_neg += 1
        else: n_zero += 1
        print(f'{w_label:<14} {mask.sum():>7} {seg_rets.mean():>+8.2f}% {base_mean:>+8.2f}% {alpha:>+7.2f}% {marker}')

    if n_pos >= 5 and n_neg <= 1: verdict = '★真规律'
    elif n_pos >= 4: verdict = '○不稳定'
    else: verdict = '✗非规律'
    print(f'\n汇总: +{n_pos} / -{n_neg} / ○{n_zero}  {verdict}')


if __name__ == '__main__':
    main()
