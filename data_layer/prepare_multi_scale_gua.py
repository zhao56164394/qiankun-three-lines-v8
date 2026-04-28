# -*- coding: utf-8 -*-
"""日/月/年 三尺度底座卦 (v10 规则, 无未来函数)

对每个交易日 D:
  日卦: 日线 trend + main_force → v10 三爻 (55 日窗口, 短波)
  月卦: 周K + 本周未收K → trend+mf (55 周窗口 ≈ 1 年, 中波)
  年卦: 月K + 本月未收K → trend+mf (55 月窗口 ≈ 4.5 年, 长波)

v10 三爻 (共用):
  位: trend >= 50
  势: trend 日/周/月差 ±2 滞后带 + trend >= 89 高位保护
  变: main_force ±30 滞后带

输出列: d_trend, d_mf, d_pos, d_spd, d_acc, d_gua, d_name  (日卦)
        m_trend, m_mf, m_pos, m_spd, m_acc, m_gua, m_name  (月卦 = 周尺度)
        y_trend, y_mf, y_pos, y_spd, y_acc, y_gua, y_name  (年卦 = 月尺度)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.indicator import _tdx_sma, _tdx_ema  # noqa: E402


POS_THR = 50
SPD_HYST = 2.0
SPD_HIGH_PROTECT = 89
ACC_HYST = 30.0

GUA_NAMES = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}
GUA_MEANINGS = {
    '000': '深熊探底', '001': '熊底异动', '010': '反弹乏力', '011': '底部爆发',
    '100': '崩盘加速', '101': '下跌护盘', '110': '牛末滞涨', '111': '疯牛主升',
}


def compute_scale_per_day(df_daily: pd.DataFrame, freq: str, window: int = 55):
    """对每个交易日, 在 freq 级别 K 线序列(完整历史 + 当日未收) 上计算 trend 和 main_force

    freq: 'W-FRI' (周) 或 'M' (月)
    window: HHV/LLV 窗口长度 (周数 / 月数), 默认 55 (旧版本); 短窗口 12 命中率更高
    返回: (trend_arr, mf_arr), 长度 = len(df_daily)
    """
    df = df_daily.copy().sort_values('date').reset_index(drop=True)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['period'] = df['date_dt'].dt.to_period(freq).astype(str)

    # 期内滚动: high cummax, low cummin (至当日)
    df['hi_run'] = df.groupby('period')['high'].cummax()
    df['lo_run'] = df.groupby('period')['low'].cummin()

    # 期序
    periods_ordered = df['period'].drop_duplicates().tolist()
    p2idx = {p: i for i, p in enumerate(periods_ordered)}
    df['p_idx'] = df['period'].map(p2idx)

    # 完整周/月 K 的 hi/lo/close (每期最后一天的 hi_run/lo_run/close)
    p_stats = df.groupby('p_idx', sort=True).agg(
        hi=('high', 'max'),
        lo=('low', 'min'),
        close=('close', 'last'),
    )
    all_hi = p_stats['hi'].values
    all_lo = p_stats['lo'].values
    all_close = p_stats['close'].values
    n_p = len(p_stats)

    # 完整期 trend 的中间状态 (SMA1, SMA2, trend value at each period end)
    # min_periods=1 配通达信 LLV 行为: 不足 window 根时用现有所有 bar
    hhv = pd.Series(all_hi).rolling(window, min_periods=1).max().values
    llv = pd.Series(all_lo).rolling(window, min_periods=1).min().values
    denom = hhv - llv
    rsv_w = np.where(denom > 0, (all_close - llv) / denom * 100, 50.0)
    sma1_w = _tdx_sma(rsv_w, 5, 1)
    sma2_w = _tdx_sma(sma1_w, 3, 1)
    v11_w = 3 * sma1_w - 2 * sma2_w
    trend_w = _tdx_ema(v11_w, 3)

    # main_force 中间状态 (MA7 用 min_periods=1 配通达信)
    ma7 = pd.Series(all_close).rolling(7, min_periods=1).mean().values
    raw_mf = np.where(ma7 > 0, (all_close - ma7) / ma7 * 480, 0.0)
    mf_inner_w = _tdx_ema(raw_mf, 2)  # EMA α=2/3

    # 每日: 基于期 w-1 的状态 + 当日未收 K, 递推一步得到当日 trend 和 mf
    n = len(df)
    trend_arr = np.full(n, np.nan)
    mf_arr = np.full(n, np.nan)

    p_idx_arr = df['p_idx'].values
    hi_run_arr = df['hi_run'].values
    lo_run_arr = df['lo_run'].values
    close_arr = df['close'].values

    for i in range(n):
        w = int(p_idx_arr[i])
        p_hi = hi_run_arr[i]
        p_lo = lo_run_arr[i]
        p_cl = close_arr[i]

        # RSV: 窗口 = 完整期[max(0,w-54)..w-1] + 当日未收 (w=0 时只有 partial)
        start = max(0, w - (window - 1))
        completed_lo = all_lo[start:w] if w > start else np.array([])
        completed_hi = all_hi[start:w] if w > start else np.array([])
        llv_val = min(completed_lo.min() if len(completed_lo) > 0 else p_lo, p_lo)
        hhv_val = max(completed_hi.max() if len(completed_hi) > 0 else p_hi, p_hi)
        rsv_val = ((p_cl - llv_val) / (hhv_val - llv_val) * 100) if hhv_val > llv_val else 50.0

        # 递推 SMA1 / SMA2 / V11 / trend
        if w == 0:
            # 本期是第 1 期, 用 partial 作初值 (配 _tdx_sma/ema 初始化)
            s1 = rsv_val
            s2 = s1
            v11 = 3 * s1 - 2 * s2
            tr_val = v11
        else:
            s1_prev = sma1_w[w - 1]
            s2_prev = sma2_w[w - 1]
            tr_prev = trend_w[w - 1]
            s1 = (4 * s1_prev + rsv_val) / 5
            s2 = (2 * s2_prev + s1) / 3
            v11 = 3 * s1 - 2 * s2
            tr_val = 0.5 * v11 + 0.5 * tr_prev
        trend_arr[i] = tr_val

        # main_force: MA7 = 最近 min(7, w+1) 个完整期 close + partial close
        take = min(6, w)
        ma7_sum = all_close[w - take:w].sum() + p_cl
        ma7_val = ma7_sum / (take + 1)
        raw_mf_val = ((p_cl - ma7_val) / ma7_val * 480) if ma7_val > 0 else 0.0
        if w == 0:
            mi = raw_mf_val
        else:
            mi_prev = mf_inner_w[w - 1]
            mi = (1.0 / 3) * mi_prev + (2.0 / 3) * raw_mf_val
        mf_arr[i] = mi * 5

    return trend_arr, mf_arr


def apply_v10_rules(trend, mf):
    """v10 三爻 + 卦码 (带滞后带)"""
    n = len(trend)
    yao_pos = np.where(np.isnan(trend), np.nan, (trend >= POS_THR).astype(float))

    # 势爻
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

    # 变爻
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
            gua.append(f"{int(yao_pos[i])}{int(yao_spd[i])}{int(yao_acc[i])}")
    return yao_pos, yao_spd, yao_acc, gua


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'zz1000_daily.csv')
    dst = os.path.join(root, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')

    print(f'读取 {src}')
    df = pd.read_csv(src, encoding='utf-8-sig')
    print(f'  {len(df)} 条 ({df["date"].iloc[0]} ~ {df["date"].iloc[-1]})')

    # --- 日卦 (日 K 尺度, 12 日窗口 — 命中率 88% / 旧 55 日仅 61%) ---
    # 旧: 用 df['trend'] 列 (默认 55 日窗口)
    # 新: 自己用 compute_scale_per_day 算 freq='D' window=12, 与 m/y 卦一致
    # 注: 日级别没有期内未收, 直接每天 = 一期, partial=close
    print('计算 日卦 (日 K 尺度, 12 日窗口) ...')
    d_trend, d_mf = compute_scale_per_day(df, 'D', window=12)
    d_pos, d_spd, d_acc, d_gua = apply_v10_rules(d_trend, d_mf)

    # --- 月卦 (周尺度, 12 周窗口 — 命中率 100% / 旧 55 周仅 70.8%) ---
    print('计算 月卦 (周 K 尺度, 12 周窗口) ...')
    m_trend, m_mf = compute_scale_per_day(df, 'W-FRI', window=12)
    m_pos, m_spd, m_acc, m_gua = apply_v10_rules(m_trend, m_mf)

    # --- 年卦 (月尺度, 12 月窗口 — 已校准, 命中率 72%) ---
    print('计算 年卦 (月 K 尺度, 12 月窗口) ...')
    y_trend, y_mf = compute_scale_per_day(df, 'M', window=12)
    y_pos, y_spd, y_acc, y_gua = apply_v10_rules(y_trend, y_mf)

    out = pd.DataFrame({
        'date': df['date'].values,
        'close': df['close'].values,
        'd_trend': d_trend, 'd_mf': d_mf, 'd_pos': d_pos, 'd_spd': d_spd, 'd_acc': d_acc, 'd_gua': d_gua,
        'm_trend': m_trend, 'm_mf': m_mf, 'm_pos': m_pos, 'm_spd': m_spd, 'm_acc': m_acc, 'm_gua': m_gua,
        'y_trend': y_trend, 'y_mf': y_mf, 'y_pos': y_pos, 'y_spd': y_spd, 'y_acc': y_acc, 'y_gua': y_gua,
    })
    out['d_name'] = out['d_gua'].map(GUA_NAMES).fillna('')
    out['m_name'] = out['m_gua'].map(GUA_NAMES).fillna('')
    out['y_name'] = out['y_gua'].map(GUA_NAMES).fillna('')

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    out.to_csv(dst, index=False, encoding='utf-8-sig')
    print(f'保存 {dst}')

    # --- 翻转统计 ---
    print('\n=== 三尺度翻转频率 (剔除未定义) ===')
    for label, col in [('日卦', 'd_gua'), ('月卦', 'm_gua'), ('年卦', 'y_gua')]:
        s = out[out[col] != ''].reset_index(drop=True)
        if len(s) == 0:
            continue
        chg = (s[col] != s[col].shift()).astype(int)
        chg.iloc[0] = 0
        n_chg = int(chg.sum())
        yrs = len(s) / 252
        print(f'  {label}: 有效 {len(s)} 天, 翻转 {n_chg} ({n_chg/yrs:.1f}/年), 平均 {len(s)/max(n_chg,1):.1f} 天/次')

    # --- 分布统计 ---
    print('\n=== 三尺度卦分布占比 % ===')
    print(f'{"卦":<8} {"日":>6} {"月":>6} {"年":>6}')
    for g in ['111', '110', '101', '100', '011', '010', '001', '000']:
        name = f'{g} {GUA_NAMES[g]}({GUA_MEANINGS[g]})'
        dpct = (out['d_gua'] == g).sum() / max(len(out), 1) * 100
        mpct = (out['m_gua'] == g).sum() / max(len(out), 1) * 100
        ypct = (out['y_gua'] == g).sum() / max(len(out), 1) * 100
        print(f'{name:<24} {dpct:>5.1f} {mpct:>5.1f} {ypct:>5.1f}')

    # --- 关键牛熊节点对比 ---
    print('\n=== 关键节点 日/月/年 卦 ===')
    nodes = [
        ('2007-10-16', '超级牛顶 (6124)'),
        ('2008-10-28', '金融危机底'),
        ('2009-08-04', '反弹顶'),
        ('2013-06-25', '钱荒底'),
        ('2014-12-01', '牛启动'),
        ('2015-06-12', '牛顶 15000'),
        ('2015-07-10', '股灾中'),
        ('2015-08-26', '暴跌底'),
        ('2016-01-28', '熔断底'),
        ('2018-01-29', '小牛顶'),
        ('2018-12-28', '熊底'),
        ('2019-04-08', '反弹确认'),
        ('2020-03-23', '疫情底'),
        ('2021-02-18', '小盘顶'),
        ('2022-04-26', '熊底'),
        ('2024-02-05', '雪球底'),
        ('2024-09-24', '政策启动'),
        ('2024-10-08', '政策牛顶'),
        ('2025-09-01', '新一轮'),
    ]
    print(f'{"日期":<12} {"事件":<16} {"收盘":>6} '
          f'{"日卦":>8} {"月卦":>8} {"年卦":>8}')
    for d, ev in nodes:
        row = out[out['date'] == d]
        if len(row) == 0:
            row = out[out['date'] <= d].tail(1)
        if len(row) == 0:
            continue
        r = row.iloc[0]

        def fmt(gua_col, name_col):
            g = r[gua_col]
            if g == '' or pd.isna(g):
                return '-'
            return f'{g}{r[name_col]}'
        print(f'{r["date"]:<12} {ev:<16} {r["close"]:>6.0f} '
              f'{fmt("d_gua","d_name"):>8} {fmt("m_gua","m_name"):>8} {fmt("y_gua","y_name"):>8}')

    evaluate_bull_bear_accuracy(out)


def evaluate_bull_bear_accuracy(out: pd.DataFrame):
    """人工标定牛熊区间, 统计各尺度在区间内 阳卦(位=1) / 阴卦(位=0) 占比

    阳卦比例 在 牛市 区间 = 命中率; 阴卦比例 在 熊市 区间 = 命中率
    """
    # 中证1000 (近似大盘) 公认牛熊区间 (保守)
    bull_periods = [
        ('2008-11-01', '2009-08-04', '09反弹'),
        ('2014-07-01', '2015-06-12', '14-15杠杆牛'),
        ('2019-01-04', '2019-04-19', '19Q1反弹'),
        ('2020-04-01', '2021-02-18', '20疫后小牛'),
        ('2024-09-24', '2025-12-31', '24-25政策牛'),
    ]
    bear_periods = [
        ('2007-11-01', '2008-10-28', '08金融危机'),
        ('2015-06-15', '2016-01-28', '15股灾+熔断'),
        ('2018-02-01', '2018-12-28', '18贸战熊'),
        ('2021-02-22', '2024-02-05', '21-24深熊'),
    ]

    def pct_in(df_slice, col, is_bull=True):
        s = df_slice[df_slice[col] != '']
        if len(s) == 0:
            return np.nan, 0
        # 位爻 = 卦码第 1 位
        pos = s[col].str[0].astype(int)
        cnt_up = (pos == 1).sum()  # 阳卦
        cnt_dn = (pos == 0).sum()  # 阴卦
        if is_bull:
            return cnt_up / len(s) * 100, len(s)
        else:
            return cnt_dn / len(s) * 100, len(s)

    print('\n=== 牛市命中率 (阳卦占比 越高越准) ===')
    print(f'{"区间":<26} {"日数":>6} {"日卦":>6} {"月卦":>6} {"年卦":>6}')
    totals = {'d': [0, 0], 'm': [0, 0], 'y': [0, 0]}  # [阳卦天数, 总天数]
    for s, e, name in bull_periods:
        slc = out[(out['date'] >= s) & (out['date'] <= e)]
        if len(slc) == 0:
            continue
        dp, dn = pct_in(slc, 'd_gua', True)
        mp, mn = pct_in(slc, 'm_gua', True)
        yp, yn = pct_in(slc, 'y_gua', True)
        lab = f'{s[:10]}~{e[:10]} {name}'
        print(f'{lab:<40} {len(slc):>6} {_fmt(dp):>6} {_fmt(mp):>6} {_fmt(yp):>6}')
        for key, pct, total in [('d', dp, dn), ('m', mp, mn), ('y', yp, yn)]:
            if not np.isnan(pct):
                totals[key][0] += pct / 100 * total
                totals[key][1] += total
    print(f'{"牛市加权平均":<40} {"":>6} '
          f'{_fmt(totals["d"][0]/max(totals["d"][1],1)*100):>6} '
          f'{_fmt(totals["m"][0]/max(totals["m"][1],1)*100):>6} '
          f'{_fmt(totals["y"][0]/max(totals["y"][1],1)*100):>6}')

    print('\n=== 熊市命中率 (阴卦占比 越高越准) ===')
    print(f'{"区间":<26} {"日数":>6} {"日卦":>6} {"月卦":>6} {"年卦":>6}')
    totals = {'d': [0, 0], 'm': [0, 0], 'y': [0, 0]}
    for s, e, name in bear_periods:
        slc = out[(out['date'] >= s) & (out['date'] <= e)]
        if len(slc) == 0:
            continue
        dp, dn = pct_in(slc, 'd_gua', False)
        mp, mn = pct_in(slc, 'm_gua', False)
        yp, yn = pct_in(slc, 'y_gua', False)
        lab = f'{s[:10]}~{e[:10]} {name}'
        print(f'{lab:<40} {len(slc):>6} {_fmt(dp):>6} {_fmt(mp):>6} {_fmt(yp):>6}')
        for key, pct, total in [('d', dp, dn), ('m', mp, mn), ('y', yp, yn)]:
            if not np.isnan(pct):
                totals[key][0] += pct / 100 * total
                totals[key][1] += total
    print(f'{"熊市加权平均":<40} {"":>6} '
          f'{_fmt(totals["d"][0]/max(totals["d"][1],1)*100):>6} '
          f'{_fmt(totals["m"][0]/max(totals["m"][1],1)*100):>6} '
          f'{_fmt(totals["y"][0]/max(totals["y"][1],1)*100):>6}')


def _fmt(v):
    if np.isnan(v):
        return '-'
    return f'{v:.1f}'


if __name__ == '__main__':
    main()
