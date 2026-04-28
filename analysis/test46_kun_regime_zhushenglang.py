# -*- coding: utf-8 -*-
"""Step 23 — 大盘 y_gua = 000 坤 regime 内 主升浪指纹

只看大盘 y_gua=000 期间:
  - 主升浪事件 (个股 d_gua=111 连续≥10 日, 起点 day0)
  - 提取 day0-1 前夜特征
  - 与该 regime 内随机对照 (任意巽日) 比较
  - 找出在坤 regime 独有的强分离特征
  - walk-forward (该 regime 内分段, 跨段稳定)
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QIAN_RUN = 10
LOOKBACK = 30

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

REGIME_Y = '000'
REGIME_NAME = '坤'


def main():
    t0 = time.time()
    print(f'=== 加载 (focus regime: 大盘 y={REGIME_Y}{REGIME_NAME}) ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend', 'd_gua', 'm_gua', 'y_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_d'] = market['d_gua'].astype(str).str.zfill(3)
    market['mkt_m'] = market['m_gua'].astype(str).str.zfill(3)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_d', 'mkt_m', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['d_trend', 'close', 'd_gua', 'mkt_d']).reset_index(drop=True)
    print(f'  全数据: {len(df):,} 行, {time.time()-t0:.1f}s')

    # 计算 regime mask: 该数据点处于 大盘 y_gua=000 期间
    in_regime = (df['mkt_y'] == REGIME_Y).to_numpy()
    n_regime = in_regime.sum()
    print(f'  在坤 regime 内: {n_regime:,} 行 ({n_regime/len(df)*100:.1f}%)')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    mf_arr = df['main_force'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d = df['d_gua'].to_numpy(); stk_m = df['m_gua'].to_numpy(); stk_y = df['y_gua'].to_numpy()
    mkt_d = df['mkt_d'].to_numpy(); mkt_m = df['mkt_m'].to_numpy(); mkt_y = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # === 找坤 regime 内主升浪事件 ===
    print(f'\n=== 找坤 regime 内主升浪事件 (d_gua=111 连续 ≥{QIAN_RUN} 日) ===')
    qian_starts = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + QIAN_RUN: continue
        gua = stk_d[s:e]
        is_qian = (gua == '111')
        i = 0; n = len(gua)
        while i < n:
            if is_qian[i]:
                j = i
                while j < n and is_qian[j]:
                    j += 1
                if j - i >= QIAN_RUN and i >= LOOKBACK:
                    # 必须 day0 处于坤 regime
                    if in_regime[s + i]:
                        qian_starts.append(s + i)
                i = j
            else:
                i += 1
    qian_starts = np.array(qian_starts)
    print(f'  坤 regime 主升浪事件: {len(qian_starts):,}')

    def extract_features(idx):
        prev = idx - 1
        if prev < 0: return None
        code_seg = np.searchsorted(code_starts, idx, side='right') - 1
        s_code = code_starts[code_seg]
        if prev < s_code + LOOKBACK: return None
        # day0-1 必须也在坤 regime (避免 regime 边界)
        if not in_regime[prev]: return None
        win_lo = prev - LOOKBACK + 1
        feat = {
            'date': date_arr[idx],
            'stk_d': stk_d[prev], 'stk_m': stk_m[prev], 'stk_y': stk_y[prev],
            'mkt_d': mkt_d[prev], 'mkt_m': mkt_m[prev],
            'trend': trend_arr[prev], 'mf': mf_arr[prev], 'sanhu': sanhu_arr[prev],
            'trend_5d': trend_arr[prev] - trend_arr[max(prev-4, win_lo)],
            'trend_30d': trend_arr[prev] - trend_arr[win_lo],
            'trend_min': trend_arr[win_lo:prev+1].min(),
            'trend_max': trend_arr[win_lo:prev+1].max(),
            'mf_30d_mean': mf_arr[win_lo:prev+1].mean(),
            'mf_30d_min': mf_arr[win_lo:prev+1].min(),
            'mf_30d_max': mf_arr[win_lo:prev+1].max(),
            'mf_5d_mean': mf_arr[max(prev-4, win_lo):prev+1].mean(),
            'sanhu_30d_mean': sanhu_arr[win_lo:prev+1].mean(),
            'sanhu_30d_min': sanhu_arr[win_lo:prev+1].min(),
            'sanhu_5d_mean': sanhu_arr[max(prev-4, win_lo):prev+1].mean(),
        }
        for g_v in GUAS:
            feat[f'pct_d_{g_v}'] = (stk_d[win_lo:prev+1] == g_v).sum() / LOOKBACK
        return feat

    print(f'\n=== 提取主升浪前夜特征 ===')
    qian_features = [extract_features(idx) for idx in qian_starts]
    qian_features = [f for f in qian_features if f is not None]
    df_q = pd.DataFrame(qian_features)
    print(f'  有效: {len(df_q):,}')

    # === 对照: 坤 regime 内 任意巽日 ===
    print(f'\n=== 对照组: 坤 regime 内 任意巽日 ===')
    rng = np.random.RandomState(42)
    qian_set = set(qian_starts.tolist())

    # 找所有坤 regime 内巽日
    xun_in_regime = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        gua = stk_d[s:e]
        for i in range(LOOKBACK, len(gua)):
            global_idx = s + i
            if not in_regime[global_idx]: continue
            if gua[i] != '011': continue
            if global_idx in qian_set: continue  # 排除主升浪起点
            xun_in_regime.append(global_idx)
    xun_in_regime = np.array(xun_in_regime)
    print(f'  坤 regime 内非主升浪 巽日: {len(xun_in_regime):,}')

    n_sample = min(len(df_q) * 5, len(xun_in_regime))
    sample_idx = rng.choice(xun_in_regime, size=n_sample, replace=False)
    ctrl_features = [extract_features(idx) for idx in sample_idx]
    ctrl_features = [f for f in ctrl_features if f is not None]
    df_ctrl = pd.DataFrame(ctrl_features)
    print(f'  对照组样本: {len(df_ctrl):,}')

    # === 1. 卦象差异 ===
    print(f'\n## 1. 当下 (day0-1) 卦象 主升浪 vs 对照')
    for col, label in [('stk_d', '个股 d_gua'), ('stk_m', '个股 m_gua'), ('stk_y', '个股 y_gua'),
                        ('mkt_d', '大盘 d_gua'), ('mkt_m', '大盘 m_gua')]:
        print(f'\n  {label} (按 |差| 排序):')
        diffs = []
        for g_v in GUAS:
            r = (df_q[col] == g_v).mean() * 100
            c = (df_ctrl[col] == g_v).mean() * 100
            diffs.append((g_v, r, c, r - c))
        diffs.sort(key=lambda x: -abs(x[3]))
        for g_v, r, c, d in diffs[:5]:
            mark = '★' if d >= 5 else ('⚠' if d <= -5 else '')
            print(f'    {g_v}{GUA_NAMES[g_v]}  主升 {r:>5.1f}%  对照 {c:>5.1f}%  差 {d:>+5.1f}  {mark}')

    # === 2. 数值特征 ===
    print(f'\n## 2. 数值特征 主升浪 vs 对照')
    print(f'  {"特征":<22} {"主升中位":>9} {"对照中位":>9} {"差(主-对)":>10}')
    print('  ' + '-' * 60)
    num_cols = ['trend', 'mf', 'sanhu',
                'trend_5d', 'trend_30d', 'trend_min', 'trend_max',
                'mf_30d_mean', 'mf_30d_min', 'mf_30d_max', 'mf_5d_mean',
                'sanhu_30d_mean', 'sanhu_30d_min', 'sanhu_5d_mean']
    diffs_n = []
    for c in num_cols:
        rm = df_q[c].median(); cm = df_ctrl[c].median()
        diffs_n.append((c, rm, cm, rm - cm))
    diffs_n.sort(key=lambda x: -abs(x[3]))
    for c, rm, cm, d in diffs_n:
        print(f'  {c:<22} {rm:>+8.2f} {cm:>+8.2f} {d:>+8.2f}')

    # === 3. 前 30 日卦象频率 ===
    print(f'\n## 3. 前 30 日 个股 d_gua 频率 主升浪 vs 对照')
    diffs_pct = []
    for g_v in GUAS:
        c = f'pct_d_{g_v}'
        r = df_q[c].mean() * 100; cm = df_ctrl[c].mean() * 100
        diffs_pct.append((g_v, r, cm, r - cm))
    diffs_pct.sort(key=lambda x: -abs(x[3]))
    for g_v, r, cm, d in diffs_pct[:5]:
        mark = '★' if d >= 2 else ('⚠' if d <= -2 else '')
        print(f'  {g_v}{GUA_NAMES[g_v]}  主均 {r:>5.1f}%  对均 {cm:>5.1f}%  差 {d:>+5.1f}  {mark}')

    # === 4. Top 单项指纹 lift (在坤 regime 巽日上看 lift) ===
    print(f'\n## 4. 各候选指纹 在坤 regime 巽日上的真实 lift (vs regime baseline)')
    # 重新 scan 该 regime 内所有巽日并算 ret_30
    print(f'  扫描坤 regime 巽日 (用于真实 lift 验证)...')
    EVAL_WIN = 30
    events_kun = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = stk_d[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(gua)
        for i in range(LOOKBACK, n - EVAL_WIN):
            gi = s + i
            if not in_regime[gi]: continue
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            wlo = i - LOOKBACK + 1
            events_kun.append({
                'date': date_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_d': mkt_d[gi], 'mkt_m': mkt_m[gi],
                'stk_m': stk_m[gi], 'stk_y': stk_y[gi],
                'trend': td[i], 'mf': mf[i], 'sanhu': sanhu[i],
                'trend_5d': td[i] - td[max(i-4, wlo)],
                'mf_30d_mean': mf[wlo:i+1].mean(),
                'mf_30d_min': mf[wlo:i+1].min(),
                'sanhu_5d_mean': sanhu[max(i-4, wlo):i+1].mean(),
                'sanhu_30d_min': sanhu[wlo:i+1].min(),
            })
    df_kun = pd.DataFrame(events_kun)
    base_ret = df_kun['ret_30'].mean()
    base_zsl = (df_kun['n_qian'] >= QIAN_RUN).mean() * 100
    print(f'  坤 regime 巽日: {len(df_kun):,}, baseline 期望 {base_ret:+.2f}%, 主升率 {base_zsl:.1f}%')

    # 候选条件: 加正向 / 排负向
    print(f'\n  ## 加正向条件 (剩 % + 期望 + 主升率)')
    pos_conds = [
        ('+ mkt_m=010坎', df_kun['mkt_m'] == '010'),
        ('+ stk_m=010坎', df_kun['stk_m'] == '010'),
        ('+ stk_y=000坤', df_kun['stk_y'] == '000'),
        ('+ stk_y=010坎', df_kun['stk_y'] == '010'),
        ('+ mkt_d=011巽', df_kun['mkt_d'] == '011'),
        ('+ trend_5d > 5', df_kun['trend_5d'] > 5),
        ('+ trend_5d > 10', df_kun['trend_5d'] > 10),
        ('+ mf_30d_mean < -10', df_kun['mf_30d_mean'] < -10),
        ('+ mf_30d_min < -200', df_kun['mf_30d_min'] < -200),
        ('+ sanhu_30d_min < -100', df_kun['sanhu_30d_min'] < -100),
        ('+ sanhu_5d < -30', df_kun['sanhu_5d_mean'] < -30),
        ('+ mf > 50 (主力当日已涌入)', df_kun['mf'] > 50),
    ]
    print(f'  {"条件":<28} {"剩 n":>7} {"剩%":>5} {"期望":>7} {"lift":>6} {"主升率":>7} {"率lift":>7}')
    print('  ' + '-' * 75)
    for label, mask in pos_conds:
        sub = df_kun[mask]
        if len(sub) < 200: continue
        r = sub['ret_30'].mean()
        zsl = (sub['n_qian'] >= QIAN_RUN).mean() * 100
        mark = '✅' if r - base_ret >= 1 else ('❌' if r - base_ret <= -0.5 else '○')
        print(f'  {label:<28} {len(sub):>7,} {len(sub)/len(df_kun)*100:>4.0f}% '
              f'{r:>+6.2f}% {r-base_ret:>+5.2f} {zsl:>5.1f}% {zsl-base_zsl:>+5.1f}  {mark}')


if __name__ == '__main__':
    main()
