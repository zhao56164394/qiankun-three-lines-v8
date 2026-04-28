# -*- coding: utf-8 -*-
"""个股多尺度卦 (test2 版) — 个股 d/m/y 三尺度

核心差异 (vs 现有 stock_daily_gua):
  - SPD_HYST 和 ACC_HYST 由 252 天滑动校准给每只股 (大盘统一 2/30 → 个股自适应)
  - POS_THR=50, SPD_HIGH_PROTECT=89 不变 (位爻语义保持一致)
  - 三爻定义与大盘对称: (位 trend>=50, 势 trend变化, 变 mf变化)
  - 不足 252 天的样本: 无卦 (NaN)

新增字段 vs stock_daily_gua:
  - 月卦 m_gua / 年卦 y_gua (W-FRI / M 压缩)
  - 校准后的 SPD_HYST_i / ACC_HYST_i 也存出, 方便诊断

输出: data_layer/data/foundation/stock_multi_scale_gua_daily.parquet
列: date, code,
    d_trend, d_mf, d_pos, d_spd, d_acc, d_gua, d_spd_hyst, d_acc_hyst,
    m_trend, m_mf, m_pos, m_spd, m_acc, m_gua,
    y_trend, y_mf, y_pos, y_spd, y_acc, y_gua,
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.indicator import _tdx_sma, _tdx_ema  # noqa: E402
from data_layer.prepare_multi_scale_gua import compute_scale_per_day  # noqa: E402


# 不变常量 (大盘共用)
POS_THR = 50
SPD_HIGH_PROTECT = 89

# 滞后带 — 跟大盘统一 (用户决策 2026-04-28: 不再逐股校准, 直接用大盘默认)
SPD_HYST = 2.0
ACC_HYST = 30.0

# 滑动校准窗口 (保留以满足 MIN_HISTORY 的历史下限要求, 不再用于校准)
CALIB_WINDOW = 252
MIN_HISTORY = 252      # 不足 252 天 → 不生成卦 (NaN)

GUA_NAMES = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}


def calibrate_per_stock(trend, mf):
    """跟大盘统一: 不再逐股校准, 全部用 SPD_HYST=2.0 / ACC_HYST=30.0.
    保留函数签名是为了兼容下游 (返回长度同 trend 的常数数组).
    MIN_HISTORY 之前的位置返回 NaN, 触发 d_gua=NaN (跟旧行为一致).
    """
    n = len(trend)
    spd = np.full(n, SPD_HYST, dtype=float)
    acc = np.full(n, ACC_HYST, dtype=float)
    # MIN_HISTORY 前没足够历史: NaN
    spd[:MIN_HISTORY] = np.nan
    acc[:MIN_HISTORY] = np.nan
    return spd, acc


def apply_v10_rules_calibrated(trend, mf, spd_hyst_arr, acc_hyst_arr):
    """带逐日校准滞后带的 v10 规则.
    SPD_HYST 和 ACC_HYST 都是按位置变化的数组.
    """
    n = len(trend)
    yao_pos = np.where(np.isnan(trend), np.nan, (trend >= POS_THR).astype(float))

    trend_prev = np.concatenate([[np.nan], trend[:-1]])
    delta = trend - trend_prev

    yao_spd = np.full(n, np.nan)
    last = 0
    for i in range(n):
        t = trend[i]
        sh = spd_hyst_arr[i]
        if np.isnan(t) or np.isnan(sh):
            yao_spd[i] = np.nan
            continue
        d = delta[i]
        if not np.isnan(d):
            if d > sh:
                last = 1
            elif d < -sh:
                last = 0
        if t >= SPD_HIGH_PROTECT:
            last = 1
        yao_spd[i] = last

    yao_acc = np.full(n, np.nan)
    last = 0
    for i in range(n):
        v = mf[i]
        ah = acc_hyst_arr[i]
        if np.isnan(v) or np.isnan(ah):
            yao_acc[i] = np.nan
            continue
        if v > ah:
            last = 1
        elif v < -ah:
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
    """一只股: 生成 d/m/y 三尺度卦 (含校准).
    df 需含 date, open, close, high, low, trend, main_force.
    """
    df = df.sort_values('date').reset_index(drop=True)

    # --- 日卦 (12 日窗口, 与大盘对齐) ---
    # 旧: 用 df['trend'] (默认 calc_trend_line period=55)
    # 新: compute_scale_per_day freq='D' window=12, 与大盘 m/d/y 全部 12 窗口对齐
    d_trend, d_mf = compute_scale_per_day(df, 'D', window=12)

    # --- 校准 SPD/ACC 滞后带 (基于日卦 trend/mf) ---
    spd_hyst_arr, acc_hyst_arr = calibrate_per_stock(d_trend, d_mf)

    # --- 日卦 ---
    d_pos, d_spd, d_acc, d_gua = apply_v10_rules_calibrated(d_trend, d_mf, spd_hyst_arr, acc_hyst_arr)

    # --- 月卦 (周尺度, 12 周窗口) ---
    m_trend, m_mf = compute_scale_per_day(df, 'W-FRI', window=12)
    # 月卦自己也校准: 用周级 trend.diff / mf 的滚动 std
    m_spd_hyst, m_acc_hyst = calibrate_per_stock(m_trend, m_mf)
    m_pos, m_spd, m_acc, m_gua = apply_v10_rules_calibrated(m_trend, m_mf, m_spd_hyst, m_acc_hyst)

    # --- 年卦 (月尺度, 12 月窗口) ---
    y_trend, y_mf = compute_scale_per_day(df, 'M', window=12)
    y_spd_hyst, y_acc_hyst = calibrate_per_stock(y_trend, y_mf)
    y_pos, y_spd, y_acc, y_gua = apply_v10_rules_calibrated(y_trend, y_mf, y_spd_hyst, y_acc_hyst)

    return pd.DataFrame({
        'date': df['date'].values,
        'code': code,
        'd_trend': d_trend, 'd_mf': d_mf,
        'd_pos': d_pos, 'd_spd': d_spd, 'd_acc': d_acc, 'd_gua': d_gua,
        'd_spd_hyst': spd_hyst_arr, 'd_acc_hyst': acc_hyst_arr,
        'm_trend': m_trend, 'm_mf': m_mf,
        'm_pos': m_pos, 'm_spd': m_spd, 'm_acc': m_acc, 'm_gua': m_gua,
        'y_trend': y_trend, 'y_mf': y_mf,
        'y_pos': y_pos, 'y_spd': y_spd, 'y_acc': y_acc, 'y_gua': y_gua,
    })


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'stocks.parquet')
    dst = os.path.join(root, 'data_layer', 'data', 'foundation', 'stock_multi_scale_gua_daily.parquet')

    print(f'读取 {src}')
    t0 = time.time()
    df_all = pd.read_parquet(src)
    print(f'  {len(df_all):,} 行 ({time.time()-t0:.0f}s)')

    df_all = df_all.dropna(subset=['trend', 'main_force']).copy()
    df_all['code'] = df_all['code'].astype(str).str.zfill(6)
    df_all['date'] = df_all['date'].astype(str)

    codes = df_all['code'].unique()
    print(f'  股票数: {len(codes)}')

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

    # 卦码空字符串改 NaN
    for col in ['d_gua', 'm_gua', 'y_gua']:
        result[col] = result[col].replace('', pd.NA)

    print(f'  总行数: {len(result):,}')
    print(f'  日期范围: {result["date"].min()} ~ {result["date"].max()}')

    # 分布对比 - 看是否更合理
    print(f'\n=== 8 态分布对比 (test2 校准 vs test1 大盘参数) ===')
    test2_dist = result['d_gua'].dropna().value_counts(normalize=True).sort_index() * 100

    # 加载 test1 (现有 stock_daily_gua) 做对比
    sd_path = os.path.join(root, 'data_layer', 'data', 'foundation', 'stock_daily_gua.parquet')
    if os.path.exists(sd_path):
        sd = pd.read_parquet(sd_path)
        test1_dist = sd['gua_code'].value_counts(normalize=True).sort_index() * 100
        print(f'  {"卦":<8} {"test2":>8} {"test1":>8} {"差异":>8}')
        for g in ['000', '001', '010', '011', '100', '101', '110', '111']:
            t2 = test2_dist.get(g, 0)
            t1 = test1_dist.get(g, 0)
            print(f'  {g} {GUA_NAMES[g]:<6} {t2:>7.2f}% {t1:>7.2f}% {t2-t1:>+7.2f}')

    print(f'\n=== 校准参数分布 (按个股全期均值) ===')
    per_stock = result.groupby('code').agg(
        spd_h=('d_spd_hyst', 'mean'),
        acc_h=('d_acc_hyst', 'mean'),
    ).dropna()
    print(f'  SPD_HYST_i: 均{per_stock["spd_h"].mean():.2f}  中位{per_stock["spd_h"].median():.2f}  P10={per_stock["spd_h"].quantile(0.1):.2f}  P90={per_stock["spd_h"].quantile(0.9):.2f}')
    print(f'  ACC_HYST_i: 均{per_stock["acc_h"].mean():.2f}  中位{per_stock["acc_h"].median():.2f}  P10={per_stock["acc_h"].quantile(0.1):.2f}  P90={per_stock["acc_h"].quantile(0.9):.2f}')
    print(f'  (大盘默认 SPD_HYST=2.0, ACC_HYST=30.0)')

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    result.to_parquet(dst, compression='snappy', index=False)
    print(f'\n保存 {dst}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
