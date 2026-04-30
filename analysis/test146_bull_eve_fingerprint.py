# -*- coding: utf-8 -*-
"""test146 — 牛股前夜指纹对比

牛股 = 个股 d_gua='111' 连续 ≥10 日的段
前夜 = 段起点 day0 的前一天 (day0-1)

输出: 主升 vs 对照, 算各特征的占比差 / 中位数差, 按显著性排序.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(),'wb',closefd=False),
                              encoding='utf-8', line_buffering=True)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, 'data_layer', 'data', 'foundation')

MIN_RUN = 10            # 主升浪连续乾卦门槛
SAMPLE_RATIO = 5        # 对照样本相对主升的倍数
RNG_SEED = 42

GUA_NAME = {'111':'乾','110':'巽','101':'离','100':'艮',
            '011':'兑','010':'坎','001':'震','000':'坤'}


def load():
    print('[load]')
    t0 = time.time()
    stk = pd.read_parquet(os.path.join(F, 'stock_multi_scale_gua_daily.parquet'),
        columns=['date','code','d_gua','m_gua','y_gua','d_trend','d_mf'])
    stk = stk.sort_values(['code','date']).reset_index(drop=True)
    print(f'  stock shape={stk.shape}')

    mkt = pd.read_parquet(os.path.join(F, 'multi_scale_gua_daily.parquet'),
        columns=['date','d_gua','m_gua','y_gua','d_trend','d_mf'])
    mkt = mkt.rename(columns={
        'd_gua':'mkt_d_gua', 'm_gua':'mkt_m_gua', 'y_gua':'mkt_y_gua',
        'd_trend':'mkt_d_trend', 'd_mf':'mkt_d_mf'})
    mkt['date'] = mkt['date'].astype(str)
    print(f'  market shape={mkt.shape}')
    print(f'  耗时 {time.time()-t0:.1f}s')
    return stk, mkt


def find_events(stk):
    """找每只票 d_gua='111' 连续 >= MIN_RUN 日的段起点 day0 索引和 day0-1 信息."""
    rows = []
    n_codes = 0
    n_runs = 0
    t0 = time.time()
    for code, sub in stk.groupby('code', sort=False):
        n_codes += 1
        n = len(sub)
        if n < MIN_RUN + 5:
            continue
        g = sub['d_gua'].to_numpy()
        d = sub['date'].to_numpy()

        is_q = (g == '111')
        # 段起点
        prev_q = np.zeros(n, dtype=bool); prev_q[1:] = is_q[:-1]
        seg_start = is_q & ~prev_q
        # 段结束 (下一天非乾)
        next_q = np.zeros(n, dtype=bool); next_q[:-1] = is_q[1:]
        seg_end = is_q & ~next_q

        starts = np.where(seg_start)[0]
        ends = np.where(seg_end)[0]
        # starts 和 ends 长度可能不一致(末段未结束) — 取 min
        L = min(len(starts), len(ends))
        for i in range(L):
            s, e = starts[i], ends[i]
            run_len = e - s + 1
            if run_len < MIN_RUN:
                continue
            n_runs += 1
            # 前夜 = day0 - 1
            if s == 0:
                continue  # 没有前一天
            rows.append({
                'code': code,
                'day0_idx': s,           # 这只票内部 idx
                'eve_idx': s - 1,
                'eve_date': d[s - 1],
                'day0_date': d[s],
                'run_len': run_len,
            })
    print(f'  扫码 {n_codes}, 主升浪段 {n_runs}, 有前夜 {len(rows)}, 耗时 {time.time()-t0:.1f}s')
    return pd.DataFrame(rows)


def sample_control(stk, ev_df, ratio=SAMPLE_RATIO):
    """对照: 同一只票, 排除主升浪段及其前夜, 任取普通日."""
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    # 先按 code 收集要排除的 idx
    excl = {}
    for code, sub in ev_df.groupby('code'):
        ex = set()
        for _, row in sub.iterrows():
            for k in range(int(row['eve_idx']), int(row['day0_idx']) + MIN_RUN):
                ex.add(k)
        excl[code] = ex

    target_n = len(ev_df) * ratio
    # 全市场总日数 ≈ 10M, 主升 ≈ 几万, 对照可在主升池所在的 code 内随机
    codes_with_events = set(ev_df['code'].unique())

    for code, sub in stk.groupby('code', sort=False):
        if code not in codes_with_events:
            continue
        n = len(sub)
        if n < 5: continue
        d = sub['date'].to_numpy()
        bad = excl.get(code, set())
        cand = [i for i in range(1, n) if i not in bad]
        if not cand: continue
        # 每只票配额: 让总样本 ~= target_n
        per_code = max(1, int(np.ceil(target_n / max(1, len(codes_with_events)))))
        chosen = rng.choice(cand, size=min(per_code, len(cand)), replace=False)
        for k in chosen:
            rows.append({
                'code': code,
                'eve_idx': int(k),
                'eve_date': d[int(k)],
            })
    df = pd.DataFrame(rows)
    print(f'  对照样本 {len(df)} (目标 {target_n})')
    return df


def attach_features(stk, mkt, df):
    """根据 (code, eve_idx) 取该日的个股 + 大盘特征."""
    stk_arr = {col: stk[col].to_numpy() for col in
               ['d_gua','m_gua','y_gua','d_trend','d_mf']}
    # 全局 idx 不能直接用 eve_idx (那是 sub 内 idx). 转: 用 (code, eve_date) join
    # 简化: 把 stk 按 (code,date) 索引一次
    stk_idx = stk.set_index(['code','date'])

    # 准备 join key
    df = df.copy()
    df['eve_date'] = df['eve_date'].astype(str)

    # 个股特征 join
    feat = stk_idx.loc[
        list(zip(df['code'].astype(str), df['eve_date'])),
        ['d_gua','m_gua','y_gua','d_trend','d_mf']
    ].reset_index(drop=True)
    out = pd.concat([df.reset_index(drop=True), feat], axis=1)

    # 大盘特征 join (按 eve_date)
    out = out.merge(mkt, left_on='eve_date', right_on='date', how='left')
    return out


def fingerprint_categorical(bull, ctrl, col, gua_map=None):
    """卦象类: 按值算主升 / 对照 占比, 算 lift = bull% - ctrl%."""
    bv = bull[col].astype(str).value_counts(normalize=True).rename('bull_pct')
    cv = ctrl[col].astype(str).value_counts(normalize=True).rename('ctrl_pct')
    df = pd.concat([bv, cv], axis=1).fillna(0.0)
    df['lift_pp'] = (df['bull_pct'] - df['ctrl_pct']) * 100
    df = df.sort_values('lift_pp', ascending=False)
    return df


def fingerprint_numerical(bull, ctrl, col):
    bx = bull[col].dropna()
    cx = ctrl[col].dropna()
    if len(bx) == 0 or len(cx) == 0: return None
    return {
        'bull_med': float(bx.median()),
        'ctrl_med': float(cx.median()),
        'med_diff': float(bx.median() - cx.median()),
        'bull_mean': float(bx.mean()),
        'ctrl_mean': float(cx.mean()),
        'mean_diff': float(bx.mean() - cx.mean()),
    }


def main():
    stk, mkt = load()

    print('\n[1] 提取主升浪事件 (d_gua=111 连续 >=10 日)')
    ev = find_events(stk)
    if len(ev) == 0:
        print('无事件, 退出'); return

    # run_len 分布
    print('  连续乾卦天数分布:')
    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f'    Q{int(q*100)}: {ev["run_len"].quantile(q):.0f} 日')
    print(f'    最长: {ev["run_len"].max()} 日')

    print('\n[2] 采样对照样本 (排除主升浪段及前夜)')
    ctrl = sample_control(stk, ev, ratio=SAMPLE_RATIO)

    print('\n[3] join 前夜特征 (个股 + 大盘)')
    bull = attach_features(stk, mkt, ev)
    ctrl = attach_features(stk, mkt, ctrl)
    print(f'  bull N={len(bull)}, ctrl N={len(ctrl)}')

    # 卦象指纹 (6 个)
    print('\n========== 卦象指纹 (主升 vs 对照, lift_pp = 占比差 in 百分点) ==========')
    cols_cat = [
        ('d_gua', '个股日卦 (前夜)'),
        ('m_gua', '个股月卦'),
        ('y_gua', '个股年卦'),
        ('mkt_d_gua', '大盘日卦'),
        ('mkt_m_gua', '大盘月卦'),
        ('mkt_y_gua', '大盘年卦'),
    ]
    for col, label in cols_cat:
        print(f'\n--- {label} ({col}) ---')
        fp = fingerprint_categorical(bull, ctrl, col)
        for code, row in fp.iterrows():
            name = GUA_NAME.get(str(code), str(code))
            tag = '⭐' if row['lift_pp'] >= 5 else ('☆' if row['lift_pp'] >= 2 else '')
            print(f'  {code} {name}  bull={row["bull_pct"]*100:5.1f}%  '
                  f'ctrl={row["ctrl_pct"]*100:5.1f}%  lift={row["lift_pp"]:+6.2f}pp  {tag}')

    # 数值指纹
    print('\n========== 数值指纹 (主升 vs 对照, 中位数差) ==========')
    num_cols = ['d_trend', 'd_mf', 'mkt_d_trend', 'mkt_d_mf']
    for col in num_cols:
        r = fingerprint_numerical(bull, ctrl, col)
        if r is None:
            print(f'  {col}: (空)'); continue
        print(f'  {col:<12}  bull_med={r["bull_med"]:7.2f}  ctrl_med={r["ctrl_med"]:7.2f}  '
              f'diff={r["med_diff"]:+7.2f}  | mean_diff={r["mean_diff"]:+7.2f}')

    # 输出落地
    out_dir = os.path.join(ROOT, 'data_layer', 'data', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    bull.to_parquet(os.path.join(out_dir, 'test146_bull_eve.parquet'),
                    engine='pyarrow', compression='snappy')
    ctrl.to_parquet(os.path.join(out_dir, 'test146_ctrl_eve.parquet'),
                    engine='pyarrow', compression='snappy')
    print(f'\n落地: data_layer/data/analysis/test146_bull_eve.parquet (N={len(bull)})')


if __name__ == '__main__':
    main()
