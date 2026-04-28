# -*- coding: utf-8 -*-
"""5 条真信号 × 大盘 y_gua regime 上下文分析

判断是否需要按大盘年卦做分治:
  对每条信号, 拆分 trigger 当日 y_gua = 000 / 111 / 其他 三种上下文
  看 alpha 是否显著不同, 如果是 → 分治值得; 如果否 → 不分治
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

SIGNALS = [
    ('m_gua', '个股月卦', '010', '011', '+', '买点'),
    ('y_gua', '个股年卦', '111', '101', '-', '卖点'),
    ('y_gua', '个股年卦', '111', '001', '-', '卖点'),
    ('d_gua', '个股日卦', '011', '001', '-', '卖点'),
    ('d_gua', '个股日卦', '111', '001', '-', '卖点'),
]

HOLD_DAYS = 10


def main():
    t0 = time.time()
    print('=== 加载数据 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                         columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                         columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    df = g.merge(p, on=['date', 'code'], how='inner').sort_values(['code', 'date']).reset_index(drop=True)
    # market y_gua 加进来作上下文
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y_gua'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y_gua']]
    df = df.merge(market, on='date', how='left')
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

    code_arr = df['code'].values; date_arr = df['date'].values
    close_arr = df['close'].values.astype(np.float32)
    d_arr = df['d_gua'].values; m_arr = df['m_gua'].values; y_arr = df['y_gua'].values
    mkt_y_arr = df['mkt_y_gua'].values

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arr_map = {'d_gua': d_arr, 'm_gua': m_arr, 'y_gua': y_arr}
    same_code = np.r_[False, code_arr[1:] == code_arr[:-1]]

    print(f'\n## 5 条真信号 × 大盘 y_gua 上下文 alpha 拆分 (HOLD={HOLD_DAYS}日)')
    print(f'  {"信号":<28} {"上下文":<14} {"事件 N":>8} {"alpha%":>8}  {"vs 全期":>8}')
    print('  ' + '-' * 75)

    for arr_name, stream_name, f, t, sign, kind in SIGNALS:
        gua_a = arr_map[arr_name]
        # 找事件 row_idx
        valid_prev = np.r_[False, gua_a[:-1] == f]
        valid_now = (gua_a == t)
        is_event = same_code & valid_prev & valid_now
        event_idx = np.where(is_event)[0]

        # forward returns + valid
        code_seg = np.searchsorted(code_starts, event_idx, side='right') - 1
        end_of_code = code_ends[code_seg]
        valid_mask = (event_idx + HOLD_DAYS) < end_of_code
        valid_idx = event_idx[valid_mask]
        c0 = close_arr[valid_idx]; c1 = close_arr[valid_idx + HOLD_DAYS]
        rets_mask = c0 > 0
        valid_idx_final = valid_idx[rets_mask]
        rets = ((c1[rets_mask] / c0[rets_mask] - 1.0) * 100.0).astype(np.float32)

        # 事件当日的大盘 y_gua
        ctx_y = mkt_y_arr[valid_idx_final]

        # 全期 alpha (vs 0 baseline 简化)
        # baseline_random alpha 应是 hold=10d 时 +0.86%, 用这个
        BASELINE = 0.86
        full_alpha = float(rets.mean() - BASELINE)
        sig_label = f'{stream_name} {f}{GUA_NAMES[f]}→{t}{GUA_NAMES[t]} ({kind})'
        print(f'  {sig_label:<28} {"全期":<14} {len(rets):>8} {full_alpha:>+7.2f}  {"--":>8}')

        # 按 ctx_y 拆分
        for ctx in ['000', '111', '其他']:
            if ctx == '其他':
                mask_ctx = ~np.isin(ctx_y, ['000', '111'])
            else:
                mask_ctx = (ctx_y == ctx)
            if mask_ctx.sum() < 50:
                print(f'  {"":<28} {f"y_gua={ctx} ({GUA_NAMES.get(ctx, ctx)})":<14} {mask_ctx.sum():>8}  样本不足')
                continue
            ctx_rets = rets[mask_ctx]
            ctx_alpha = float(ctx_rets.mean() - BASELINE)
            diff = ctx_alpha - full_alpha
            ctx_label = f'y_gua={ctx} ({GUA_NAMES.get(ctx, ctx)})'
            print(f'  {"":<28} {ctx_label:<14} {len(ctx_rets):>8} {ctx_alpha:>+7.2f}  {diff:>+7.2f}')
        print()


if __name__ == '__main__':
    main()
