# -*- coding: utf-8 -*-
"""日/月/年 三尺度共振信号 v1

信号定义:
  pos_score = d_pos + m_pos + y_pos  (0-3, 位爻共振)
  spd_score = d_spd + m_spd + y_spd  (0-3, 势爻共振)
  acc_score = d_acc + m_acc + y_acc  (0-3, 变爻共振)
  total = pos_score + spd_score + acc_score  (0-9, 综合)

  pos_score = 3: 三尺度高位共振 = 强牛
  pos_score = 2: 两尺度高位 = 牛偏多
  pos_score = 1: 一尺度高位 = 熊偏弱
  pos_score = 0: 三尺度低位 = 深熊

输出:
  1. 每种分数的历史天数分布
  2. 每种分数的 前瞻 20/60 日 平均收益
  3. 简单策略回测: 不同阈值下的 满仓/空仓 净值 vs buy&hold
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')
    dst = os.path.join(root, 'data_layer', 'data', 'foundation', 'resonance_signal_daily.csv')

    df = pd.read_csv(src, encoding='utf-8-sig', dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 确保三尺度都有效 (全部有值)
    mask = (
        df['d_pos'].notna() & df['m_pos'].notna() & df['y_pos'].notna()
        & df['d_spd'].notna() & df['m_spd'].notna() & df['y_spd'].notna()
        & df['d_acc'].notna() & df['m_acc'].notna() & df['y_acc'].notna()
    )
    df = df[mask].reset_index(drop=True)

    for c in ['d_pos', 'm_pos', 'y_pos', 'd_spd', 'm_spd', 'y_spd', 'd_acc', 'm_acc', 'y_acc']:
        df[c] = df[c].astype(int)

    df['pos_score'] = df['d_pos'] + df['m_pos'] + df['y_pos']
    df['spd_score'] = df['d_spd'] + df['m_spd'] + df['y_spd']
    df['acc_score'] = df['d_acc'] + df['m_acc'] + df['y_acc']
    df['total'] = df['pos_score'] + df['spd_score'] + df['acc_score']

    # 保存
    keep = ['date', 'close', 'd_gua', 'm_gua', 'y_gua',
            'pos_score', 'spd_score', 'acc_score', 'total']
    df[keep].to_csv(dst, index=False, encoding='utf-8-sig')
    print(f'保存 {dst}')

    print(f'\n样本范围: {df["date"].iloc[0].date()} ~ {df["date"].iloc[-1].date()}')
    print(f'天数: {len(df)}')

    # === 1. 位爻共振分布 + 前瞻收益 ===
    df['fwd_20'] = df['close'].shift(-20) / df['close'] - 1
    df['fwd_60'] = df['close'].shift(-60) / df['close'] - 1
    df['fwd_120'] = df['close'].shift(-120) / df['close'] - 1

    print('\n=== 位爻共振分数分布 ===')
    print(f'{"分数":>4} {"天数":>6} {"占比":>7} {"后20天%":>9} {"后60天%":>9} {"后120天%":>10}')
    for s in [3, 2, 1, 0]:
        sub = df[df['pos_score'] == s]
        if len(sub) == 0:
            continue
        pct = len(sub) / len(df) * 100
        f20 = sub['fwd_20'].mean() * 100 if sub['fwd_20'].notna().any() else np.nan
        f60 = sub['fwd_60'].mean() * 100 if sub['fwd_60'].notna().any() else np.nan
        f120 = sub['fwd_120'].mean() * 100 if sub['fwd_120'].notna().any() else np.nan
        print(f'{s:>4} {len(sub):>6} {pct:>6.1f}% {f20:>+8.2f}% {f60:>+8.2f}% {f120:>+9.2f}%')

    print('\n=== 势爻共振分数分布 ===')
    print(f'{"分数":>4} {"天数":>6} {"占比":>7} {"后20天%":>9} {"后60天%":>9}')
    for s in [3, 2, 1, 0]:
        sub = df[df['spd_score'] == s]
        if len(sub) == 0:
            continue
        pct = len(sub) / len(df) * 100
        f20 = sub['fwd_20'].mean() * 100 if sub['fwd_20'].notna().any() else np.nan
        f60 = sub['fwd_60'].mean() * 100 if sub['fwd_60'].notna().any() else np.nan
        print(f'{s:>4} {len(sub):>6} {pct:>6.1f}% {f20:>+8.2f}% {f60:>+8.2f}%')

    print('\n=== 变爻共振分数分布 ===')
    print(f'{"分数":>4} {"天数":>6} {"占比":>7} {"后20天%":>9} {"后60天%":>9}')
    for s in [3, 2, 1, 0]:
        sub = df[df['acc_score'] == s]
        if len(sub) == 0:
            continue
        pct = len(sub) / len(df) * 100
        f20 = sub['fwd_20'].mean() * 100 if sub['fwd_20'].notna().any() else np.nan
        f60 = sub['fwd_60'].mean() * 100 if sub['fwd_60'].notna().any() else np.nan
        print(f'{s:>4} {len(sub):>6} {pct:>6.1f}% {f20:>+8.2f}% {f60:>+8.2f}%')

    print('\n=== 综合分 0-9 分布 ===')
    print(f'{"分":>3} {"天数":>6} {"占比":>7} {"后20天%":>9} {"后60天%":>9} {"后120天%":>10}')
    for s in range(9, -1, -1):
        sub = df[df['total'] == s]
        if len(sub) == 0:
            continue
        pct = len(sub) / len(df) * 100
        f20 = sub['fwd_20'].mean() * 100 if sub['fwd_20'].notna().any() else np.nan
        f60 = sub['fwd_60'].mean() * 100 if sub['fwd_60'].notna().any() else np.nan
        f120 = sub['fwd_120'].mean() * 100 if sub['fwd_120'].notna().any() else np.nan
        print(f'{s:>3} {len(sub):>6} {pct:>6.1f}% {f20:>+8.2f}% {f60:>+8.2f}% {f120:>+9.2f}%')

    # === 2. 回测: 不同入场阈值 vs buy&hold ===
    def backtest(signal_col, thresholds):
        df_bt = df.copy()
        ret_daily = df_bt['close'].pct_change().fillna(0).values
        n = len(df_bt)
        results = []
        # buy&hold
        bh_eq = (1 + pd.Series(ret_daily)).cumprod().values
        years = (df_bt['date'].iloc[-1] - df_bt['date'].iloc[0]).days / 365.25
        bh_cagr = bh_eq[-1] ** (1 / years) - 1
        # max drawdown
        bh_peak = pd.Series(bh_eq).cummax()
        bh_dd = ((bh_eq - bh_peak) / bh_peak).min() * 100
        results.append(('Buy&Hold', 1.0, bh_eq[-1], bh_cagr * 100, bh_dd, 100.0))

        for thr in thresholds:
            pos = (df_bt[signal_col] >= thr).astype(float).values
            # T+1: 使用前一天信号决定今天持仓
            pos = np.concatenate([[0], pos[:-1]])
            strat_ret = pos * ret_daily
            eq = (1 + pd.Series(strat_ret)).cumprod().values
            cagr = eq[-1] ** (1 / years) - 1
            peak = pd.Series(eq).cummax()
            dd = ((eq - peak) / peak).min() * 100
            coverage = pos.mean() * 100
            results.append((f'{signal_col}>={thr}', thr, eq[-1], cagr * 100, dd, coverage))
        return results

    print('\n=== 回测: pos_score >= 阈值 则持有 (T+1), 否则空仓 ===')
    print(f'{"策略":<18} {"末值":>7} {"CAGR%":>7} {"MaxDD%":>8} {"持仓占比%":>9}')
    for name, thr, eq, cagr, dd, cov in backtest('pos_score', [3, 2, 1]):
        print(f'{name:<18} {eq:>7.2f} {cagr:>+7.2f}% {dd:>+7.1f}% {cov:>8.1f}%')

    print('\n=== 回测: total >= 阈值 则持有 ===')
    print(f'{"策略":<18} {"末值":>7} {"CAGR%":>7} {"MaxDD%":>8} {"持仓占比%":>9}')
    for name, thr, eq, cagr, dd, cov in backtest('total', [9, 7, 5, 3]):
        print(f'{name:<18} {eq:>7.2f} {cagr:>+7.2f}% {dd:>+7.1f}% {cov:>8.1f}%')


if __name__ == '__main__':
    main()
