# -*- coding: utf-8 -*-
"""6 个底座卦 "八卦分治" 适用性比较

对每个卦计算 5 维指标，衡量"用它的 8 个状态分治策略"是否可行：
  1. 样本覆盖: 起止日 + 总观察数
  2. 状态分布: 8 态占比 + 熵比 (熵/log2(8), 越高越均匀)
  3. 段长: 平均连续保持同一状态的天数 (段太短 = 择时摩擦大)
  4. 前瞻收益分层: 8 态的 ret_fwd_60d 均值极差 (max-min) 和 std
  5. 单调性: gua_code 视作 0-7 整数, 与 ret_fwd_60d 的 Spearman rank 相关

市场级卦 (日/月/年/天): 前瞻收益 = 中证1000 指数 fwd_60
个股级卦 (地/人): 前瞻收益 = 个股 fwd_60 (每股自己的 regime 对应自己的前瞻)
"""
import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FND = os.path.join(ROOT, 'data_layer', 'data', 'foundation')
GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAME = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
            '100': '震', '101': '离', '110': '兑', '111': '乾'}


def load_index_fwd():
    """中证1000 指数前瞻收益"""
    idx = pd.read_csv(os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv'),
                      encoding='utf-8-sig', usecols=['date', 'close'])
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.sort_values('date').reset_index(drop=True)
    idx['fwd_20'] = idx['close'].shift(-20) / idx['close'] - 1
    idx['fwd_60'] = idx['close'].shift(-60) / idx['close'] - 1
    return idx[['date', 'fwd_20', 'fwd_60']]


def load_stock_fwd():
    """个股前瞻收益 (60 日)"""
    fp = os.path.join(FND, 'daily_forward_returns.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'code', 'ret_fwd_20d', 'ret_fwd_60d'],
                     dtype={'code': str})
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'ret_fwd_20d': 'fwd_20', 'ret_fwd_60d': 'fwd_60'})
    # 源数据是百分比 (-11.94 表示 -11.94%), 统一成小数
    df['fwd_20'] = df['fwd_20'] / 100.0
    df['fwd_60'] = df['fwd_60'] / 100.0
    return df


def normalize_gua_str(x):
    """把 gua_code 规范为 '000'-'111' 3 位字符串. 支持 str / int 输入."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in ('', 'nan'):
        return None
    return s.zfill(3)


def compute_metrics(df_gua, gua_col, fwd_col, stock_col=None):
    """核心统计
    df_gua: 已合并前瞻收益的数据
    gua_col: 卦列名 (值已规范为 '000'-'111')
    fwd_col: 前瞻收益列
    stock_col: 如果是个股级, 段长按 (stock, seg_run) 统计
    """
    g = df_gua[gua_col]
    mask = g.isin(GUA_ORDER) & df_gua[fwd_col].notna()
    sub = df_gua[mask].copy()
    if len(sub) == 0:
        return None

    n_obs = len(sub)
    start = sub['date'].min().date()
    end = sub['date'].max().date()

    # 分布 + 熵
    dist = sub[gua_col].value_counts(normalize=True).reindex(GUA_ORDER, fill_value=0.0)
    ent = -sum(p * math.log2(p) for p in dist.values if p > 0)
    ent_ratio = ent / 3.0  # 均匀时 log2(8)=3

    # 段长: 按 stock(若有)+date 排序, 检测 gua 变化, 统计段长
    srt = sub.sort_values([stock_col, 'date'] if stock_col else ['date'])
    if stock_col:
        grp = srt.groupby(stock_col, sort=False)
        chg = (srt[gua_col] != grp[gua_col].shift()) | grp[gua_col].shift().isna()
        seg_id_global = chg.cumsum()
        seg_lens = seg_id_global.value_counts()
    else:
        chg = srt[gua_col] != srt[gua_col].shift()
        seg_id = chg.cumsum()
        seg_lens = seg_id.value_counts()
    mean_seg = float(seg_lens.mean())
    n_segs = int(len(seg_lens))

    # 前瞻收益 8 态均值
    fwd_by_state = sub.groupby(gua_col)[fwd_col].agg(['mean', 'count']).reindex(GUA_ORDER)
    fwd_means = fwd_by_state['mean']
    fwd_counts = fwd_by_state['count'].fillna(0).astype(int)

    spread = (fwd_means.max() - fwd_means.min()) * 100 if fwd_means.notna().any() else np.nan
    std = fwd_means.std() * 100 if fwd_means.notna().any() else np.nan

    # Spearman rank corr: 手写 (rank 后 Pearson), 避开 scipy 依赖
    gua_int = sub[gua_col].map({g: i for i, g in enumerate(GUA_ORDER)})
    r1 = gua_int.rank()
    r2 = sub[fwd_col].rank()
    rank_corr = r1.corr(r2)

    return {
        'n_obs': n_obs,
        'start': str(start),
        'end': str(end),
        'dist': dist,
        'ent_ratio': ent_ratio,
        'min_state_pct': float(dist.min()) * 100,
        'max_state_pct': float(dist.max()) * 100,
        'mean_seg': mean_seg,
        'n_segs': n_segs,
        'fwd_means': fwd_means,
        'fwd_counts': fwd_counts,
        'fwd_spread': spread,
        'fwd_std': std,
        'rank_corr': rank_corr,
    }


def print_metrics(name, m):
    if m is None:
        print(f'\n[{name}] 无可用数据')
        return
    print(f'\n=== {name} ===')
    print(f'覆盖: {m["start"]} ~ {m["end"]}   观察数: {m["n_obs"]:,}   段数: {m["n_segs"]:,}   平均段长: {m["mean_seg"]:.1f}')
    print(f'8 态分布 (%):')
    hdr = '  ' + ' '.join(f'{g}{GUA_NAME[g]:>2}' for g in GUA_ORDER)
    print(hdr)
    print('  ' + ' '.join(f'{m["dist"][g]*100:>4.1f}' for g in GUA_ORDER))
    print(f'  熵比 {m["ent_ratio"]:.3f}   最小态占比 {m["min_state_pct"]:.2f}%   最大态占比 {m["max_state_pct"]:.2f}%')

    print(f'8 态 fwd_60 均值 (%):')
    print(hdr)
    vals = []
    for g in GUA_ORDER:
        v = m['fwd_means'][g]
        vals.append(f'{v*100:+5.2f}' if pd.notna(v) else '   -')
    print('  ' + ' '.join(vals))
    print(f'  极差 {m["fwd_spread"]:.2f}%   标准差 {m["fwd_std"]:.2f}%   rank_corr {m["rank_corr"]:+.3f}')


def analyze_multi_scale():
    """日/月/年 三尺度卦"""
    fp = os.path.join(FND, 'multi_scale_gua_daily.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    df['date'] = pd.to_datetime(df['date'])
    for c in ['d_gua', 'm_gua', 'y_gua']:
        df[c] = df[c].apply(normalize_gua_str)
    idx = load_index_fwd()
    df = df.merge(idx, on='date', how='left')

    out = {}
    for col, name in [('d_gua', '日卦'), ('m_gua', '月卦'), ('y_gua', '年卦')]:
        out[name] = compute_metrics(df, col, 'fwd_60')
    return out


def analyze_tian():
    """天卦 (market_bagua_daily.csv)"""
    fp = os.path.join(FND, 'market_bagua_daily.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'gua_code'],
                     dtype={'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['tian_gua'] = df['gua_code'].apply(normalize_gua_str)
    idx = load_index_fwd()
    df = df.merge(idx, on='date', how='left')
    return {'天卦': compute_metrics(df, 'tian_gua', 'fwd_60')}


def analyze_di():
    """地卦 (stock_bagua_daily.csv)"""
    fp = os.path.join(FND, 'stock_bagua_daily.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'code', 'gua_code'],
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['di_gua'] = df['gua_code'].apply(normalize_gua_str)
    fwd = load_stock_fwd()
    df = df.merge(fwd[['date', 'code', 'fwd_60']], on=['date', 'code'], how='left')
    return {'地卦': compute_metrics(df, 'di_gua', 'fwd_60', stock_col='code')}


def analyze_ren():
    """人卦 (daily_bagua_sequence.csv)"""
    fp = os.path.join(FND, 'daily_bagua_sequence.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig',
                     usecols=['date', 'code', 'gua_code'],
                     dtype={'code': str, 'gua_code': str})
    df['date'] = pd.to_datetime(df['date'])
    df['ren_gua'] = df['gua_code'].apply(normalize_gua_str)
    fwd = load_stock_fwd()
    df = df.merge(fwd[['date', 'code', 'fwd_60']], on=['date', 'code'], how='left')
    return {'人卦': compute_metrics(df, 'ren_gua', 'fwd_60', stock_col='code')}


def rank_and_print(all_metrics):
    print('\n\n========== 六卦综合排名 ==========')
    hdr = f'{"卦":<6} {"样本":>12} {"熵比":>5} {"最小态%":>7} {"段长":>6} {"极差%":>6} {"标准差%":>7} {"rank_corr":>10}'
    print(hdr)
    print('-' * len(hdr))

    rows = []
    for name, m in all_metrics.items():
        if m is None:
            continue
        rows.append({
            'name': name,
            'n_obs': m['n_obs'],
            'ent': m['ent_ratio'],
            'min_pct': m['min_state_pct'],
            'seg': m['mean_seg'],
            'spread': m['fwd_spread'],
            'std': m['fwd_std'],
            'corr': m['rank_corr'],
        })

    for r in rows:
        print(f'{r["name"]:<6} {r["n_obs"]:>12,} {r["ent"]:>5.3f} {r["min_pct"]:>6.2f}% '
              f'{r["seg"]:>6.1f} {r["spread"]:>5.2f}% {r["std"]:>6.2f}% {r["corr"]:>+10.3f}')

    print('\n打分规则 (各维度排名 1=最好, 6=最差; 综合分越低越好):')
    print('  - 熵比 越高越好 (状态均匀)')
    print('  - 最小态占比 越高越好 (无死角)')
    print('  - 段长 越长越好 (抗抖动)')
    print('  - fwd 极差 越大越好 (分层明显)')
    print('  - fwd 标准差 越大越好 (分层明显)')
    print('  - |rank_corr| 越大越好 (单调性)')

    metrics_hi = ['ent', 'min_pct', 'seg', 'spread', 'std']
    # 按 |corr| 排
    n = len(rows)
    ranks = {r['name']: 0 for r in rows}
    for key in metrics_hi:
        srt = sorted(rows, key=lambda x: -x[key])
        for i, r in enumerate(srt):
            ranks[r['name']] += i + 1
    srt = sorted(rows, key=lambda x: -abs(x['corr']))
    for i, r in enumerate(srt):
        ranks[r['name']] += i + 1

    print(f'\n{"卦":<6} {"综合分":>7}  (6 维排名之和)')
    for name, sc in sorted(ranks.items(), key=lambda x: x[1]):
        print(f'{name:<6} {sc:>7}')


def main():
    all_metrics = {}

    print('> 加载 日/月/年 三尺度卦...')
    all_metrics.update(analyze_multi_scale())

    print('> 加载 天卦 (market_bagua_daily)...')
    all_metrics.update(analyze_tian())

    print('> 加载 地卦 (stock_bagua_daily, ~7.4M 行)...')
    all_metrics.update(analyze_di())

    print('> 加载 人卦 (daily_bagua_sequence, ~7.7M 行)...')
    all_metrics.update(analyze_ren())

    for name, m in all_metrics.items():
        print_metrics(name, m)

    rank_and_print(all_metrics)


if __name__ == '__main__':
    main()
