# -*- coding: utf-8 -*-
"""
呼吸战法 — 指标计算模块

三大指标（日线级别）：
  趋势线: X=(C-LLV(L,55))/(HHV(H,55)-LLV(L,55))*100
          V11=3*SMA(X,5,1)-2*SMA(SMA(X,5,1),3,1)  趋势线=EMA(V11,3)
  主力线: EMA((C-MA(C,7))/MA(C,7)*480, 2)*5
  散户线: EMA((C-MA(C,11))/MA(C,11)*480, 7)*5

分钟级趋势线（做T用，与日线公式相同但用分钟OHLC）
"""

import numpy as np
import pandas as pd


# ═══════════════════════════ 基础函数 ═══════════════════════════

def _tdx_sma(arr, N, M):
    """通达信 SMA(X, N, M) = (M*X + (N-M)*prev) / N"""
    result = np.full(len(arr), np.nan)
    first_valid = -1
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            first_valid = i
            break
    if first_valid < 0:
        return result
    result[first_valid] = arr[first_valid]
    for i in range(first_valid + 1, len(arr)):
        if np.isnan(arr[i]):
            continue
        result[i] = (M * arr[i] + (N - M) * result[i - 1]) / N
    return result


def _tdx_ema(arr, N):
    """通达信 EMA(X, N) = prev*(1-k) + X*k, k=2/(N+1)"""
    k = 2.0 / (N + 1)
    result = np.full(len(arr), np.nan)
    prev = np.nan
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            continue
        if np.isnan(prev):
            prev = v
        else:
            prev = prev * (1 - k) + v * k
        result[i] = prev
    return result


# ═══════════════════════════ 日线指标 ═══════════════════════════

def calc_trend_line(closes, highs, lows, period=55):
    """
    趋势线 (0~100)
    输入: numpy数组, 至少 period 个有效值
    period: LLV/HHV窗口长度, 默认55
    输出: 同长度numpy数组
    """
    C = np.array(closes, dtype=float)
    H = np.array(highs, dtype=float)
    L = np.array(lows, dtype=float)

    llv = pd.Series(L).rolling(period, min_periods=period).min().values
    hhv = pd.Series(H).rolling(period, min_periods=period).max().values
    denom = hhv - llv
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.where(denom == 0, 100.0, (C - llv) / denom * 100)
    X[np.isnan(llv)] = np.nan

    sma1 = _tdx_sma(X, 5, 1)
    sma2 = _tdx_sma(sma1, 3, 1)
    V11 = 3 * sma1 - 2 * sma2
    trend = _tdx_ema(V11, 3)
    return trend


def calc_main_force_line(closes):
    """
    主力线: EMA((C-MA(C,7))/MA(C,7)*480, 2)*5
    输入: numpy数组
    输出: 同长度numpy数组
    """
    C = np.array(closes, dtype=float)
    ma7 = pd.Series(C).rolling(7, min_periods=7).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = (C - ma7) / ma7 * 480
    raw[np.isnan(ma7)] = np.nan
    result = _tdx_ema(raw, 2) * 5
    return result


def calc_retail_line(closes):
    """
    散户线: EMA((C-MA(C,11))/MA(C,11)*480, 7)*5
    输入: numpy数组
    输出: 同长度numpy数组
    """
    C = np.array(closes, dtype=float)
    ma11 = pd.Series(C).rolling(11, min_periods=11).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        raw = (C - ma11) / ma11 * 480
    raw[np.isnan(ma11)] = np.nan
    result = _tdx_ema(raw, 7) * 5
    return result
