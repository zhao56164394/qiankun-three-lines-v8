# -*- coding: utf-8 -*-
"""N 日线中周期卦对比 · 找最合适的"牛熊状态卦"

背景: 年卦 (月线合成) 延迟过大, 且 y_gua=101 信号 98% 集中 2015, 不稳定.
      周卦 (m_gua, 按 W-FRI period 合成) 延迟小, 但 period-based 会在周一跳变.
      希望找一个 rolling N 日窗口的卦, 又能描述牛熊, 又能尽早翻转.

方法:
  1. 对每个 N ∈ {3,5,7,10,14,20}, 每天 D 取过去 N 天合成一个 rolling K:
       hi = max(high[D-N+1..D])
       lo = min(low[D-N+1..D])
       close = close[D]
     按 v10 规则 (55 N日线窗口) 算 trend + main_force + 三爻 + 卦码
  2. 再拉进 multi_scale 的 d_gua (1日 period)、m_gua (周 period)、y_gua (月 period) 作对比.
  3. 用历史牛熊区间做命中率评估 + 回测区间全量 signal 的 actual_ret 做 spread 评估.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strategy.indicator import _tdx_sma, _tdx_ema  # noqa: E402

ZZ1000 = os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv')
MULTI  = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

POS_THR = 50
SPD_HYST = 2.0
SPD_HIGH_PROTECT = 89
ACC_HYST = 30.0

GUA_NAMES = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}

BULL_PERIODS = [
    ('2014-07-01', '2015-06-12', '14-15杠杆牛'),
    ('2019-01-04', '2019-04-19', '19Q1反弹'),
    ('2020-04-01', '2021-02-18', '20疫后小牛'),
    ('2024-09-24', '2025-10-15', '24政策牛'),
]
BEAR_PERIODS = [
    ('2015-06-15', '2016-01-28', '15股灾+熔断'),
    ('2018-02-01', '2018-12-28', '18贸战熊'),
    ('2021-02-22', '2024-02-05', '21-24深熊'),
]


def compute_n_day_gua(df: pd.DataFrame, N: int):
    """Rolling N 日卦: 每天取过去 N 天合成一个 bar, 按 v10 规则生成卦.

    序列的每个元素 = 截止到 D 的 N 日 rolling bar (hi, lo, close).
    因为每天产生一个 bar, 55 日线窗口 = 55 个 N 日 bar = ~55 天历史 (而非 55*N).
    但 N 日 bar 的 trend 在 trend-series 的每一点, 已经把过去 N 天的波动"吸进来"了.
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)

    hi_n = np.full(n, np.nan)
    lo_n = np.full(n, np.nan)
    cl_n = close.copy()  # close 就用当天

    for i in range(n):
        start = max(0, i - N + 1)
        hi_n[i] = np.max(high[start:i + 1])
        lo_n[i] = np.min(low[start:i + 1])

    # 55 根 N 日 bar 窗口做 RSV
    hhv = pd.Series(hi_n).rolling(55, min_periods=1).max().values
    llv = pd.Series(lo_n).rolling(55, min_periods=1).min().values
    denom = hhv - llv
    rsv = np.where(denom > 0, (cl_n - llv) / denom * 100, 50.0)
    sma1 = _tdx_sma(rsv, 5, 1)
    sma2 = _tdx_sma(sma1, 3, 1)
    v11 = 3 * sma1 - 2 * sma2
    trend = _tdx_ema(v11, 3)

    # main_force
    ma7 = pd.Series(cl_n).rolling(7, min_periods=1).mean().values
    raw_mf = np.where(ma7 > 0, (cl_n - ma7) / ma7 * 480, 0.0)
    mf_inner = _tdx_ema(raw_mf, 2)
    mf = mf_inner * 5

    return trend, mf


def apply_v10(trend, mf):
    n = len(trend)
    yao_pos = np.where(np.isnan(trend), np.nan, (trend >= POS_THR).astype(float))

    trend_prev = np.concatenate([[np.nan], trend[:-1]])
    delta = trend - trend_prev
    yao_spd = np.full(n, np.nan)
    last = 0
    for i in range(n):
        t = trend[i]
        if np.isnan(t):
            yao_spd[i] = np.nan; continue
        d = delta[i]
        if not np.isnan(d):
            if d > SPD_HYST: last = 1
            elif d < -SPD_HYST: last = 0
        if t >= SPD_HIGH_PROTECT: last = 1
        yao_spd[i] = last

    yao_acc = np.full(n, np.nan)
    last = 0
    for i in range(n):
        v = mf[i]
        if np.isnan(v):
            yao_acc[i] = np.nan; continue
        if v > ACC_HYST: last = 1
        elif v < -ACC_HYST: last = 0
        yao_acc[i] = last

    gua = []
    for i in range(n):
        if any(np.isnan([yao_pos[i], yao_spd[i], yao_acc[i]])):
            gua.append('')
        else:
            gua.append(f"{int(yao_pos[i])}{int(yao_spd[i])}{int(yao_acc[i])}")
    return yao_pos, gua


def period_hit_rate(dates, gua_arr, periods, is_bull: bool):
    """某区间内, 位爫=1 (或 =0) 占比
    is_bull=True: 算 位=1 占比 (牛市命中率)
    is_bull=False: 算 位=0 占比 (熊市命中率)
    """
    results = []
    for s, e, name in periods:
        mask = (dates >= s) & (dates <= e)
        if not mask.any():
            continue
        sub_gua = gua_arr[mask]
        valid = sub_gua != ''
        if not valid.any():
            continue
        positions = np.array([int(g[0]) for g in sub_gua[valid]])
        if is_bull:
            hit_rate = (positions == 1).mean() * 100
        else:
            hit_rate = (positions == 0).mean() * 100
        results.append((name, mask.sum(), hit_rate))
    return results


def main():
    print('\n  N 日线中周期卦对比')
    print('  数据源:', ZZ1000, '+', MULTI)

    df = pd.read_csv(ZZ1000, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    df = df.sort_values('date').reset_index(drop=True)
    print(f'  {len(df)} 条 ({df["date"].iloc[0]} ~ {df["date"].iloc[-1]})')

    # 现有的 d/m/y_gua
    ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')

    results = {}  # key = label, value = (gua_arr, desc)

    # 现有: d_gua (日卦=1日 period), m_gua (周卦=W-FRI period), y_gua (年卦=月 period)
    ms_sub = ms[ms['date'].isin(df['date'])].sort_values('date').reset_index(drop=True)
    results['d_gua (1日 period)'] = ms_sub['d_gua'].fillna('').values
    results['m_gua (周 period)'] = ms_sub['m_gua'].fillna('').values
    results['y_gua (月 period)'] = ms_sub['y_gua'].fillna('').values

    # N 日 rolling 卦: N ∈ {3,5,7,10,14,20}
    for N in [3, 5, 7, 10, 14, 20]:
        tr, mf = compute_n_day_gua(df, N)
        _, gua = apply_v10(tr, mf)
        results[f'N={N} rolling'] = np.array(gua)

    dates = np.array(df['date'].values)

    # 指标 1: 翻转频率 (回测区间内)
    mask_bt = (dates >= '2014-08-01') & (dates <= '2026-04-21')
    print('\n' + '=' * 90)
    print('  指标 1 · 翻转频率 (回测区间 2014-08-01 ~ 2026-04-21)')
    print('=' * 90)
    print(f'  {"卦":<22} {"有效天数":>8} {"翻转次数":>8} {"次/年":>8} {"天/次":>8}')
    print('  ' + '-' * 60)
    for label, gua in results.items():
        sub = gua[mask_bt]
        valid = sub != ''
        sv = sub[valid]
        if len(sv) == 0:
            continue
        chg = np.concatenate([[0], (sv[1:] != sv[:-1]).astype(int)])
        n_chg = int(chg.sum())
        yrs = len(sv) / 252
        print(f'  {label:<22} {len(sv):>8} {n_chg:>8} {n_chg/yrs:>8.1f} {len(sv)/max(n_chg,1):>8.1f}')

    # 指标 2: 牛市命中率 (位爫=1)
    print('\n' + '=' * 120)
    print('  指标 2 · 牛市命中率 (阳卦 位=1 占比, 越高越准)')
    print('=' * 120)
    hdr = f'  {"区间":<36} {"日数":>5}'
    for label in results.keys():
        hdr += f' {label[:16]:>16}'
    print(hdr)
    print('  ' + '-' * 120)
    totals = {lab: [0.0, 0] for lab in results.keys()}
    for s, e, name in BULL_PERIODS:
        mask = (dates >= s) & (dates <= e)
        ndays = int(mask.sum())
        if ndays == 0: continue
        row = f'  {s[:10]}~{e[:10]} {name:<16} {ndays:>5}'
        for label, gua in results.items():
            sub = gua[mask]
            valid = sub != ''
            if not valid.any():
                row += f' {"-":>16}'; continue
            pos = np.array([int(g[0]) for g in sub[valid]])
            hit = (pos == 1).mean() * 100
            totals[label][0] += hit / 100 * valid.sum()
            totals[label][1] += valid.sum()
            row += f' {hit:>15.1f}%'
        print(row)
    # 加权平均
    row = f'  {"加权命中率":<36} {"":>5}'
    for label in results.keys():
        t = totals[label]
        pct = t[0] / max(t[1], 1) * 100 if t[1] else 0
        row += f' {pct:>15.1f}%'
    print(row)

    # 指标 3: 熊市命中率 (位爫=0)
    print('\n' + '=' * 120)
    print('  指标 3 · 熊市命中率 (阴卦 位=0 占比, 越高越准)')
    print('=' * 120)
    print(hdr)
    print('  ' + '-' * 120)
    totals = {lab: [0.0, 0] for lab in results.keys()}
    for s, e, name in BEAR_PERIODS:
        mask = (dates >= s) & (dates <= e)
        ndays = int(mask.sum())
        if ndays == 0: continue
        row = f'  {s[:10]}~{e[:10]} {name:<16} {ndays:>5}'
        for label, gua in results.items():
            sub = gua[mask]
            valid = sub != ''
            if not valid.any():
                row += f' {"-":>16}'; continue
            pos = np.array([int(g[0]) for g in sub[valid]])
            hit = (pos == 0).mean() * 100
            totals[label][0] += hit / 100 * valid.sum()
            totals[label][1] += valid.sum()
            row += f' {hit:>15.1f}%'
        print(row)
    row = f'  {"加权命中率":<36} {"":>5}'
    for label in results.keys():
        t = totals[label]
        pct = t[0] / max(t[1], 1) * 100 if t[1] else 0
        row += f' {pct:>15.1f}%'
    print(row)

    # 指标 4: 牛顶/熊底 反应延迟
    print('\n' + '=' * 120)
    print('  指标 4 · 牛顶/熊底翻转反应延迟 (关键拐点后首次位爫改向的距离, 天)')
    print('=' * 120)
    critical = [
        ('2015-06-12', 'bull_top', '15牛顶'),
        ('2016-01-28', 'bear_end', '16熔断底'),
        ('2018-12-28', 'bear_end', '18熊底'),
        ('2020-03-23', 'bear_end', '20疫情底'),
        ('2021-02-18', 'bull_top', '21小盘顶'),
        ('2022-04-26', 'bear_end', '22熊底'),
        ('2024-02-05', 'bear_end', '24雪球底'),
        ('2024-09-24', 'bull_start', '24政策牛启动'),
    ]
    hdr = f'  {"拐点":<32}'
    for label in results.keys():
        hdr += f' {label[:16]:>16}'
    print(hdr)
    print('  ' + '-' * 100)
    for dt, typ, name in critical:
        # bull_start / bull_top: 期望 "位=1" 开始 / 结束
        # bear_end: 期望 "位=1" 出现
        target = 1 if typ in ('bull_start', 'bear_end') else 0
        row = f'  {dt} {typ:<10} {name:<14}'
        dt_idx = np.where(dates >= dt)[0]
        if len(dt_idx) == 0:
            row += ' ' * 20 * len(results); print(row); continue
        i0 = dt_idx[0]
        for label, gua in results.items():
            # 从 i0 起找第一次 位=target 的天
            found = None
            for j in range(i0, min(i0 + 200, len(gua))):
                g = gua[j]
                if g != '' and int(g[0]) == target:
                    found = j - i0
                    break
            if found is None:
                row += f' {">200":>16}'
            else:
                row += f' {found:>15}天'
        print(row)

    print('\n[说明]')
    print('  翻转频率: 越接近预期 "中周期波动" 越好 (周卦 ~15/年 ~17 天/次 是一个参考)')
    print('  命中率: 牛市位爫=1 / 熊市位爫=0 占比, 越高说明该卦"判对"牛熊越稳')
    print('  反应延迟: 拐点后首次位爫翻转的距离, 越小越灵敏')


if __name__ == '__main__':
    main()
