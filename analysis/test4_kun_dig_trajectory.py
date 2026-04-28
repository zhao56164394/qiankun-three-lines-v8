# -*- coding: utf-8 -*-
"""Phase 3 坤桶研究 Step 4 — 上穿 11 之前 30 天的轨迹对比挖掘.

入池视角: 找出"成功组"和"失败组"在 [-30, 0] 天里, 哪些指标的轨迹/形态差异最大.

挖掘内容:
  1. 多指标的中位数轨迹叠加 (retail / main_force / trend / close-base)
  2. 关键转折点的时间分布 (主力转正第几天 / 散户最低点第几天 / trend最低点第几天)
  3. 形态特征 (上穿前的"反弹幅度" / "波动率" / "持续下跌天数")

输出: 控制台轨迹表 + 关键时点对比 + 形态摘要
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = 30  # 回看 30 天


def load_signals():
    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test4',
                     'kun_naked_t11_t89.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    df = pd.DataFrame(d['results'])
    df['entry_date'] = df['entry_date'].astype(str)
    df['code'] = df['code'].astype(str).str.zfill(6)
    return df


def load_main_codes():
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                                       'main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    return set(uni[uni['board'] == '主板']['code'].unique())


def load_stocks():
    main_codes = load_main_codes()
    df = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet'),
                         columns=['code', 'date', 'close', 'trend', 'retail', 'main_force'])
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = df['date'].astype(str).str[:10]
    df = df[df['code'].isin(main_codes)].copy()
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df['idx'] = df.groupby('code').cumcount()
    return df


def build_trajectories(sigs, stocks):
    """对每个信号, 抽取 [-30, 0] 共 31 天的轨迹, 锚定 t=entry_date 为 0."""
    print(f'[轨迹] 构造 {len(sigs)} 信号 × {WIN+1} 天 = {len(sigs)*(WIN+1)} 行')
    # 给信号查找在 stocks 里的 idx
    sig2 = sigs.merge(stocks[['code', 'date', 'idx']],
                      left_on=['code', 'entry_date'], right_on=['code', 'date'],
                      how='inner').drop(columns='date')

    rows = []
    for offset in range(-WIN, 1):
        tmp = sig2.copy()
        tmp['t'] = offset
        tmp['target_idx'] = tmp['idx'] + offset
        rows.append(tmp[['code', 'target_idx', 't', 'success', 'entry_date']])
    expanded = pd.concat(rows, ignore_index=True)

    # join stocks
    res = expanded.merge(stocks[['code', 'idx', 'close', 'trend', 'retail', 'main_force']],
                         left_on=['code', 'target_idx'], right_on=['code', 'idx'],
                         how='left').drop(columns='idx')
    res = res.dropna(subset=['close', 'trend', 'retail', 'main_force'])
    print(f'[轨迹] 有效行: {len(res)}')
    return res


def median_trajectories(traj):
    """两组在每个 t 的中位数对比"""
    print('\n' + '=' * 100)
    print('轨迹中位数: 成功组 vs 失败组, 上穿 11 之前 30 天')
    print('=' * 100)
    for col in ['retail', 'main_force', 'trend']:
        agg = traj.groupby(['t', 'success'])[col].median().unstack('success')
        agg.columns = ['fail', 'success']
        agg['diff (succ-fail)'] = agg['success'] - agg['fail']
        print(f'\n  [{col}] 各天 (t=信号当日为0)')
        print(f'  {"t":>4} {"成功":>10} {"失败":>10} {"差值 succ-fail":>16}')
        # 仅展示几个关键时间点
        key_ts = [-30, -25, -20, -15, -10, -7, -5, -3, -1, 0]
        for t in key_ts:
            if t in agg.index:
                row = agg.loc[t]
                print(f'  {t:>4d} {row["success"]:>+10.2f} {row["fail"]:>+10.2f} '
                      f'{row["diff (succ-fail)"]:>+15.2f}')

    # close 单独处理: 转换成相对 t=0 的累计变化%
    print(f'\n  [close 价格变化%, 相对 t=0] 各天')
    print(f'  {"t":>4} {"成功":>10} {"失败":>10} {"差值":>10}')
    # 用 entry_date 当日 close 为 base
    base = traj[traj['t'] == 0].set_index(['code', 'entry_date'])['close'].rename('base_close')
    traj2 = traj.merge(base, left_on=['code', 'entry_date'], right_index=True, how='left')
    traj2['close_pct'] = (traj2['close'] / traj2['base_close'] - 1) * 100
    agg_c = traj2.groupby(['t', 'success'])['close_pct'].median().unstack('success')
    agg_c.columns = ['fail', 'success']
    for t in [-30, -25, -20, -15, -10, -7, -5, -3, -1, 0]:
        if t in agg_c.index:
            row = agg_c.loc[t]
            print(f'  {t:>4d} {row["success"]:>+9.2f}% {row["fail"]:>+9.2f}% '
                  f'{row["success"] - row["fail"]:>+9.2f}%')


def turning_points(traj):
    """关键转折点: 主力转正/散户最低/trend最低 在第几天"""
    print('\n' + '=' * 100)
    print('关键转折点: 在 [-30, 0] 30 天内, 各转折事件出现的时间分布')
    print('=' * 100)

    # 1. main_force 由负转正: 第一次 main_force 由 ≤0 变 >0 的 t
    def first_mf_turn_pos(g):
        g = g.sort_values('t')
        mf = g['main_force'].values
        for i in range(1, len(mf)):
            if mf[i-1] <= 0 and mf[i] > 0:
                return g['t'].values[i]
        return np.nan

    # 2. retail 最低点 t
    def retail_min_t(g):
        i = g['retail'].idxmin()
        return g.loc[i, 't'] if pd.notna(i) else np.nan

    # 3. main_force 最高点 t
    def mf_max_t(g):
        i = g['main_force'].idxmax()
        return g.loc[i, 't'] if pd.notna(i) else np.nan

    # 4. trend 最低点 t
    def trend_min_t(g):
        i = g['trend'].idxmin()
        return g.loc[i, 't'] if pd.notna(i) else np.nan

    print('\n  [处理...] 这一步可能需要 20-60 秒')
    grouped = traj.groupby(['code', 'entry_date', 'success'])
    rec = grouped.agg(
        mf_turn_pos_t=('t', lambda x: np.nan),  # 占位, 用下面替换
    )
    # 直接用 numpy 加速
    sigs_keys = traj[['code', 'entry_date', 'success']].drop_duplicates().reset_index(drop=True)
    out = {'mf_turn': [], 'retail_min': [], 'mf_max': [], 'trend_min': [], 'success': []}
    for _, row in sigs_keys.iterrows():
        sub = traj[(traj['code'] == row['code']) &
                   (traj['entry_date'] == row['entry_date'])].sort_values('t')
        if len(sub) < WIN:
            continue
        t_arr = sub['t'].values
        mf = sub['main_force'].values
        retail = sub['retail'].values
        trend = sub['trend'].values

        # mf 转正: prev<=0 and curr>0
        mf_turn = np.nan
        for i in range(1, len(mf)):
            if mf[i-1] <= 0 and mf[i] > 0:
                mf_turn = t_arr[i]
                break

        out['mf_turn'].append(mf_turn)
        out['retail_min'].append(t_arr[np.argmin(retail)])
        out['mf_max'].append(t_arr[np.argmax(mf)])
        out['trend_min'].append(t_arr[np.argmin(trend)])
        out['success'].append(row['success'])

    df = pd.DataFrame(out)
    print(f'  [完成] 信号数: {len(df)}')

    for col in ['mf_turn', 'retail_min', 'mf_max', 'trend_min']:
        print(f'\n  [{col}] 出现时间分布 (t=信号当日为0)')
        for label, mask in [('成功组', df['success']), ('失败组', ~df['success'].astype(bool))]:
            sub = df.loc[mask, col].dropna()
            if len(sub) == 0:
                print(f'    {label}: 无')
                continue
            print(f'    {label}: n={len(sub)}, '
                  f'mean={sub.mean():.1f}, '
                  f'median={sub.median():.0f}, '
                  f'p25={sub.quantile(.25):.0f}, p75={sub.quantile(.75):.0f}')

        # 没出现 mf_turn 的比例 (整 30 天主力都没转正)
        if col == 'mf_turn':
            for label, mask in [('成功组', df['success']), ('失败组', ~df['success'].astype(bool))]:
                sub = df.loc[mask, col]
                no_turn = sub.isna().sum()
                tot = len(sub)
                print(f'    [{label}] 主力始终未转正 (整 30 天): {no_turn}/{tot} = {no_turn/tot*100:.1f}%')

    return df


def shape_features(traj):
    """形态特征: 单调性 / 波动率 / 反弹幅度等"""
    print('\n' + '=' * 100)
    print('形态特征: 上穿 11 之前 30 天的统计特征对比 (中位数)')
    print('=' * 100)

    feats_list = []
    sigs_keys = traj[['code', 'entry_date', 'success']].drop_duplicates().reset_index(drop=True)
    for _, row in sigs_keys.iterrows():
        sub = traj[(traj['code'] == row['code']) &
                   (traj['entry_date'] == row['entry_date'])].sort_values('t')
        if len(sub) < WIN:
            continue
        retail = sub['retail'].values
        mf = sub['main_force'].values
        trend = sub['trend'].values
        close = sub['close'].values

        feats_list.append({
            'success': row['success'],
            # 主力线: 30 天内的累计上涨幅度 (max-min) / 距 0 的次数
            'mf_range': float(np.max(mf) - np.min(mf)),
            'mf_above0_n': int((mf > 0).sum()),
            'mf_neg2pos_jumps': int(((mf[1:] > 0) & (mf[:-1] <= 0)).sum()),
            # 散户线: 30 天内最低值 / 累计下降幅度
            'retail_min': float(np.min(retail)),
            'retail_range': float(np.max(retail) - np.min(retail)),
            # trend 上穿前的"反弹空间": min trend
            'trend_min': float(np.min(trend)),
            'trend_below11_n': int((trend < 11).sum()),
            # 价格回调幅度: max close 到 entry close 的回调
            'close_max_drawdown': float((close.min() / close.max() - 1) * 100),
            'close_30d_chg': float((close[-1] / close[0] - 1) * 100),
        })

    fdf = pd.DataFrame(feats_list)
    print(f'  样本数: {len(fdf)}')

    print(f'\n  {"特征":<30} {"成功组中位":>12} {"失败组中位":>12} {"差值":>10}')
    print('  ' + '-' * 65)
    cols = ['mf_range', 'mf_above0_n', 'mf_neg2pos_jumps',
            'retail_min', 'retail_range',
            'trend_min', 'trend_below11_n',
            'close_max_drawdown', 'close_30d_chg']
    for c in cols:
        s = fdf.loc[fdf['success'], c].median()
        f = fdf.loc[~fdf['success'].astype(bool), c].median()
        print(f'  {c:<30} {s:>+12.2f} {f:>+12.2f} {s - f:>+10.2f}')


def main():
    print('=== 加载 ===')
    sigs = load_signals()
    BASE = sigs['success'].mean() * 100
    print(f'信号 {len(sigs)}, 成功率 {BASE:.2f}%')
    stocks = load_stocks()
    print(f'stocks 行 {len(stocks)}, 股票 {stocks["code"].nunique()}')

    traj = build_trajectories(sigs, stocks)

    median_trajectories(traj)
    turning_points(traj)
    shape_features(traj)


if __name__ == '__main__':
    main()
