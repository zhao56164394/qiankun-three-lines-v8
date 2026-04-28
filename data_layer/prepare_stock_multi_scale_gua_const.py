# -*- coding: utf-8 -*-
"""个股多尺度卦 — 固定常量校准版 (对照组)
SPD_HYST = 3.5 (常量), ACC_HYST = 93 (常量)
对应通达信公式 test2_stock_gua_main.tdx 的参数

输出: data_layer/data/foundation/stock_multi_scale_gua_daily_const.parquet
对比 stock_multi_scale_gua_daily.parquet (滑动 STD 版)
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.prepare_multi_scale_gua import compute_scale_per_day  # noqa: E402

POS_THR = 50
SPD_HIGH_PROTECT = 89
SPD_HYST = 3.5         # 固定常量 (通达信版)
ACC_HYST = 93.0        # 固定常量 (通达信版)
MIN_HISTORY = 60       # 通达信无 252 滑动, 只需有少量历史就能起算

GUA_NAMES = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def apply_v10_rules_const(trend, mf):
    """大盘版 v10 规则但用 SPD_HYST=3.5 / ACC_HYST=93 (固定常量)"""
    n = len(trend)
    yao_pos = np.where(np.isnan(trend), np.nan, (trend >= POS_THR).astype(float))
    trend_prev = np.concatenate([[np.nan], trend[:-1]])
    delta = trend - trend_prev

    yao_spd = np.full(n, np.nan)
    last = 0
    for i in range(n):
        t = trend[i]
        if np.isnan(t):
            yao_spd[i] = np.nan
            continue
        d = delta[i]
        if not np.isnan(d):
            if d > SPD_HYST:
                last = 1
            elif d < -SPD_HYST:
                last = 0
        if t >= SPD_HIGH_PROTECT:
            last = 1
        yao_spd[i] = last

    yao_acc = np.full(n, np.nan)
    last = 0
    for i in range(n):
        v = mf[i]
        if np.isnan(v):
            yao_acc[i] = np.nan
            continue
        if v > ACC_HYST:
            last = 1
        elif v < -ACC_HYST:
            last = 0
        yao_acc[i] = last

    gua = []
    for i in range(n):
        if any(np.isnan([yao_pos[i], yao_spd[i], yao_acc[i]])):
            gua.append('')
        else:
            gua.append(f'{int(yao_pos[i])}{int(yao_spd[i])}{int(yao_acc[i])}')
    return yao_pos, yao_spd, yao_acc, gua


def process_one_stock(code, df):
    df = df.sort_values('date').reset_index(drop=True)
    d_trend = df['trend'].astype(float).values
    d_mf = df['main_force'].astype(float).values

    d_pos, d_spd, d_acc, d_gua = apply_v10_rules_const(d_trend, d_mf)
    m_trend, m_mf = compute_scale_per_day(df, 'W-FRI')
    m_pos, m_spd, m_acc, m_gua = apply_v10_rules_const(m_trend, m_mf)
    y_trend, y_mf = compute_scale_per_day(df, 'M')
    y_pos, y_spd, y_acc, y_gua = apply_v10_rules_const(y_trend, y_mf)

    return pd.DataFrame({
        'date': df['date'].values,
        'code': code,
        'd_trend': d_trend, 'd_mf': d_mf,
        'd_pos': d_pos, 'd_spd': d_spd, 'd_acc': d_acc, 'd_gua': d_gua,
        'm_trend': m_trend, 'm_mf': m_mf,
        'm_pos': m_pos, 'm_spd': m_spd, 'm_acc': m_acc, 'm_gua': m_gua,
        'y_trend': y_trend, 'y_mf': y_mf,
        'y_pos': y_pos, 'y_spd': y_spd, 'y_acc': y_acc, 'y_gua': y_gua,
    })


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'stocks.parquet')
    dst = os.path.join(root, 'data_layer', 'data', 'foundation', 'stock_multi_scale_gua_daily_const.parquet')

    print(f'读取 {src}')
    t0 = time.time()
    df_all = pd.read_parquet(src)
    print(f'  {len(df_all):,} 行 ({time.time()-t0:.0f}s)')

    df_all = df_all.dropna(subset=['trend', 'main_force']).copy()
    df_all['code'] = df_all['code'].astype(str).str.zfill(6)
    df_all['date'] = df_all['date'].astype(str)

    codes = df_all['code'].unique()
    print(f'  股票数: {len(codes)}, 用固定常量 SPD={SPD_HYST}, ACC={ACC_HYST}')

    chunks = []
    skipped = 0
    t0 = time.time()
    for i, code in enumerate(codes):
        df = df_all[df_all['code'] == code]
        if len(df) < MIN_HISTORY + 5:
            skipped += 1
            continue
        out = process_one_stock(code, df)
        chunks.append(out)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(codes)} ({time.time()-t0:.0f}s, skipped {skipped})')

    print(f'\n合并 {len(chunks)} 只股, 跳过 {skipped}')
    result = pd.concat(chunks, ignore_index=True)
    for col in ['d_gua', 'm_gua', 'y_gua']:
        result[col] = result[col].replace('', pd.NA)

    print(f'  总行数: {len(result):,}')

    # 对比两份数据
    print(f'\n=== 对比 滑动std 版 vs 固定常量版 ===')
    sg_path = os.path.join(root, 'data_layer', 'data', 'foundation', 'stock_multi_scale_gua_daily.parquet')
    if os.path.exists(sg_path):
        sg = pd.read_parquet(sg_path, columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua'])
        sg['code'] = sg['code'].astype(str).str.zfill(6)
        sg['date'] = sg['date'].astype(str)
        merged = result[['date', 'code', 'd_gua', 'm_gua', 'y_gua']].merge(
            sg, on=['date', 'code'], suffixes=('_const', '_std'), how='inner'
        )
        merged_d = merged.dropna(subset=['d_gua_const', 'd_gua_std'])
        merged_m = merged.dropna(subset=['m_gua_const', 'm_gua_std'])
        merged_y = merged.dropna(subset=['y_gua_const', 'y_gua_std'])
        print(f'  {"层":<6} {"对照行数":>10} {"卦码相同%":>10}')
        for label, mg, c, s in [
            ('d_gua', merged_d, 'd_gua_const', 'd_gua_std'),
            ('m_gua', merged_m, 'm_gua_const', 'm_gua_std'),
            ('y_gua', merged_y, 'y_gua_const', 'y_gua_std'),
        ]:
            same = (mg[c] == mg[s]).mean() * 100
            print(f'  {label:<6} {len(mg):>10,} {same:>9.2f}%')

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    result.to_parquet(dst, compression='snappy', index=False)
    print(f'\n保存 {dst}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
