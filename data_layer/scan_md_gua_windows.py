# -*- coding: utf-8 -*-
"""扫描月卦/日卦窗口 — 找命中率最优窗口

类似 y_gua 12 月窗口的方法:
  对月卦 (周 K 尺度), 扫窗口 12 / 18 / 26 / 35 / 55 周
  对日卦 (日 K 尺度), 扫窗口 12 / 21 / 35 / 55 / 89 日

命中率定义: ≥N 段 ([0]位 = 1) == (段收益 > 0)
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from strategy.indicator import _tdx_sma, _tdx_ema
from data_layer.prepare_multi_scale_gua import apply_v10_rules


def compute_per_day(df_daily, freq, window):
    """复制 prepare_multi_scale_gua.compute_scale_per_day 但参数化窗口"""
    df = df_daily.copy().sort_values('date').reset_index(drop=True)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['period'] = df['date_dt'].dt.to_period(freq).astype(str)
    df['hi_run'] = df.groupby('period')['high'].cummax()
    df['lo_run'] = df.groupby('period')['low'].cummin()
    periods_ordered = df['period'].drop_duplicates().tolist()
    p2idx = {p: i for i, p in enumerate(periods_ordered)}
    df['p_idx'] = df['period'].map(p2idx)
    p_stats = df.groupby('p_idx', sort=True).agg(
        hi=('high', 'max'), lo=('low', 'min'), close=('close', 'last'))
    all_hi = p_stats['hi'].values
    all_lo = p_stats['lo'].values
    all_close = p_stats['close'].values

    hhv = pd.Series(all_hi).rolling(window, min_periods=1).max().values
    llv = pd.Series(all_lo).rolling(window, min_periods=1).min().values
    denom = hhv - llv
    rsv_w = np.where(denom > 0, (all_close - llv) / denom * 100, 50.0)
    sma1_w = _tdx_sma(rsv_w, 5, 1)
    sma2_w = _tdx_sma(sma1_w, 3, 1)
    v11_w = 3 * sma1_w - 2 * sma2_w
    trend_w = _tdx_ema(v11_w, 3)
    ma7 = pd.Series(all_close).rolling(7, min_periods=1).mean().values
    raw_mf = np.where(ma7 > 0, (all_close - ma7) / ma7 * 480, 0.0)
    mf_inner_w = _tdx_ema(raw_mf, 2)

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
        start = max(0, w - (window - 1))
        completed_lo = all_lo[start:w] if w > start else np.array([])
        completed_hi = all_hi[start:w] if w > start else np.array([])
        llv_val = min(completed_lo.min() if len(completed_lo) > 0 else p_lo, p_lo)
        hhv_val = max(completed_hi.max() if len(completed_hi) > 0 else p_hi, p_hi)
        rsv_val = ((p_cl - llv_val) / (hhv_val - llv_val) * 100) if hhv_val > llv_val else 50.0
        if w == 0:
            s1 = rsv_val; s2 = s1; v11 = 3 * s1 - 2 * s2; tr_val = v11
        else:
            s1_prev = sma1_w[w - 1]; s2_prev = sma2_w[w - 1]; tr_prev = trend_w[w - 1]
            s1 = (4 * s1_prev + rsv_val) / 5
            s2 = (2 * s2_prev + s1) / 3
            v11 = 3 * s1 - 2 * s2
            tr_val = 0.5 * v11 + 0.5 * tr_prev
        trend_arr[i] = tr_val
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


def evaluate_hitrate(df_daily, gua_arr, freq, min_seg_len):
    """按段评估: ≥min_seg_len 段, 阳卦 (位=1) == (段收益>0) 命中率
    freq = 'W-FRI' 或 'M' or 'D'
    """
    df = df_daily.copy().reset_index(drop=True)
    df['gua'] = gua_arr
    df['date_dt'] = pd.to_datetime(df['date'])
    if freq == 'W-FRI':
        df['period'] = df['date_dt'].dt.to_period('W-FRI').astype(str)
    elif freq == 'M':
        df['period'] = df['date_dt'].dt.to_period('M').astype(str)
    else:
        df['period'] = df['date'].astype(str)
    sample = df.groupby('period').last().reset_index().sort_values('date_dt').reset_index(drop=True)
    sample = sample[sample['gua'].notna() & (sample['gua'] != '')].reset_index(drop=True)
    sample['prev'] = sample['gua'].shift()
    sample['changed'] = (sample['gua'] != sample['prev']) & sample['prev'].notna()
    events = sample[sample['changed']].reset_index().rename(columns={'index': 'e_idx'})
    events['seg_end_idx'] = list(events['e_idx'].tolist()[1:]) + [len(sample) - 1]

    rows = []
    for _, r in events.iterrows():
        si, ei = int(r['e_idx']), int(r['seg_end_idx'])
        seg = sample.iloc[si:ei + 1]
        c0 = float(seg['close'].iloc[0])
        c1 = float(seg['close'].iloc[-1])
        rows.append({'cur': str(r['gua']).zfill(3),
                     'len': len(seg),
                     'ret': (c1 / c0 - 1) * 100})
    segs = pd.DataFrame(rows)
    if len(segs) == 0:
        return None

    big = segs[segs['len'] >= min_seg_len]
    n_big = len(big)
    if n_big == 0:
        return {'n_segs': len(segs), 'n_big': 0, 'hit': 0, 'hit_rate': 0,
                'avg_seg_len': segs['len'].mean()}

    hit = sum(1 for _, r in big.iterrows() if (r['cur'][0] == '1') == (r['ret'] > 0))
    return {
        'n_segs': len(segs),
        'n_big': n_big,
        'hit': hit,
        'hit_rate': hit / n_big * 100,
        'avg_seg_len': segs['len'].mean(),
    }


def main():
    src = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    if not os.path.exists(src):
        src = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.csv')
        df = pd.read_csv(src, encoding='utf-8-sig',
                         dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    else:
        df = pd.read_parquet(src)

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df = df.sort_values('date').reset_index(drop=True)

    # 加载原始日 K 用于 月/日卦 重算
    src_idx = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'index_kline.parquet')
    if os.path.exists(src_idx):
        idx = pd.read_parquet(src_idx)
        # 用沪深300/中证全指作为大盘代理 (跟现有月卦一致)
        # 注: 检查 idx 列
        print(f'index_kline 列: {idx.columns.tolist()[:10]}')
    else:
        # fallback: 读 zz1000
        from backtest_capital import load_zz1000_full
        idx = load_zz1000_full()
        print(f'zz1000_full 列: {idx.columns.tolist()[:10]}, len={len(idx)}')

    if 'date' not in idx.columns:
        idx = idx.reset_index()
    idx['date'] = pd.to_datetime(idx['date']).dt.strftime('%Y-%m-%d')
    idx = idx.sort_values('date').reset_index(drop=True)
    # 只保留 multi_scale_gua_daily 同期间
    idx = idx[(idx['date'] >= df['date'].min()) & (idx['date'] <= df['date'].max())].reset_index(drop=True)
    print(f'index 数据: {len(idx)} 天 ({idx["date"].iloc[0]} ~ {idx["date"].iloc[-1]})')

    # === 当前月卦 (55 周) baseline ===
    print('\n## 当前 m_gua (55 周窗口)')
    cur_m_gua = df['m_gua'].astype(str).str.zfill(3).values
    res = evaluate_hitrate(df, cur_m_gua, 'W-FRI', min_seg_len=8)
    print(f'  ≥8 周段: {res["hit"]}/{res["n_big"]} = {res["hit_rate"]:.1f}%, 总段 {res["n_segs"]}, 平均段长 {res["avg_seg_len"]:.1f} 周')
    res = evaluate_hitrate(df, cur_m_gua, 'W-FRI', min_seg_len=13)
    print(f'  ≥13 周段: {res["hit"]}/{res["n_big"]} = {res["hit_rate"]:.1f}%')

    # === 当前日卦 (55 日) baseline ===
    print('\n## 当前 d_gua (55 日窗口)')
    cur_d_gua = df['d_gua'].astype(str).str.zfill(3).values
    res = evaluate_hitrate(df, cur_d_gua, 'D', min_seg_len=10)
    print(f'  ≥10 日段: {res["hit"]}/{res["n_big"]} = {res["hit_rate"]:.1f}%, 总段 {res["n_segs"]}, 平均段长 {res["avg_seg_len"]:.1f} 日')
    res = evaluate_hitrate(df, cur_d_gua, 'D', min_seg_len=20)
    print(f'  ≥20 日段: {res["hit"]}/{res["n_big"]} = {res["hit_rate"]:.1f}%')

    # === 扫描月卦窗口候选 ===
    print('\n## 月卦窗口扫描 (周 K 尺度)')
    print(f'{"窗口(周)":<10} {"≥8周命中":>10} {"≥13周命中":>10} {"总段":>5} {"平均段长":>8}')
    for w in [12, 18, 26, 35, 55]:
        trend, mf = compute_per_day(idx, 'W-FRI', w)
        _, _, _, gua = apply_v10_rules(trend, mf)
        gua_arr = np.array([str(g).zfill(3) if g else '' for g in gua])
        # 截到 df 同期长度
        gua_df = pd.DataFrame({'date': idx['date'].values, 'close': idx['close'].values, 'gua': gua_arr})
        gua_df = gua_df[gua_df['date'].isin(df['date'].values)].reset_index(drop=True)
        # 用 idx 自己评估
        eval_df = idx.rename(columns={'close': 'close'})  # already has close
        res8 = evaluate_hitrate(eval_df, gua_arr, 'W-FRI', min_seg_len=8)
        res13 = evaluate_hitrate(eval_df, gua_arr, 'W-FRI', min_seg_len=13)
        print(f'{w:<10} {res8["hit"]}/{res8["n_big"]} ({res8["hit_rate"]:.1f}%)  '
              f'{res13["hit"]}/{res13["n_big"]} ({res13["hit_rate"]:.1f}%)  '
              f'{res8["n_segs"]:>5} {res8["avg_seg_len"]:>7.1f}周')

    # === 扫描日卦窗口候选 ===
    print('\n## 日卦窗口扫描 (日 K 尺度)')
    print(f'{"窗口(日)":<10} {"≥10日命中":>10} {"≥20日命中":>10} {"总段":>5} {"平均段长":>8}')
    for w in [12, 21, 35, 55, 89]:
        trend, mf = compute_per_day(idx, 'D', w)
        _, _, _, gua = apply_v10_rules(trend, mf)
        gua_arr = np.array([str(g).zfill(3) if g else '' for g in gua])
        res10 = evaluate_hitrate(idx, gua_arr, 'D', min_seg_len=10)
        res20 = evaluate_hitrate(idx, gua_arr, 'D', min_seg_len=20)
        print(f'{w:<10} {res10["hit"]}/{res10["n_big"]} ({res10["hit_rate"]:.1f}%)  '
              f'{res20["hit"]}/{res20["n_big"]} ({res20["hit_rate"]:.1f}%)  '
              f'{res10["n_segs"]:>5} {res10["avg_seg_len"]:>7.1f}日')


if __name__ == '__main__':
    main()
