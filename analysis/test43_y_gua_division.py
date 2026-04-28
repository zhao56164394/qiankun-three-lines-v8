# -*- coding: utf-8 -*-
"""Step 20 — 按大盘 y_gua 分治 巽日策略

对每个大盘 y_gua 8 态, 各自看:
  - 巽日 baseline 期望 / 主升率
  - 各避雷条件 lift
  - 大盘 m=坎/个股 m=坎 的 lift
  - 找出每个 regime 下的最佳子策略

输出: 8 个 regime × 多个子策略, 看哪个 regime 适合什么打法
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN = 30
ZSL_THRESH = 10

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
GUAS = ['000', '001', '010', '011', '100', '101', '110', '111']

WINDOWS = [
    ('w1_2018',    '2018-01-01', '2019-01-01'),
    ('w2_2019',    '2019-01-01', '2020-01-01'),
    ('w3_2020',    '2020-01-01', '2021-01-01'),
    ('w4_2021',    '2021-01-01', '2022-01-01'),
    ('w5_2022',    '2022-01-01', '2023-01-01'),
    ('w6_2023_24', '2023-01-01', '2025-01-01'),
    ('w7_2025_26', '2025-01-01', '2026-04-21'),
]


def main():
    t0 = time.time()
    print('=== 加载 ===')
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
    print(f'  {len(df):,} 行, {time.time()-t0:.1f}s')

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

    # 扫所有巽日
    print(f'\n=== 扫描所有巽日 ===')
    t1 = time.time()
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + EVAL_WIN + 5: continue
        td = trend_arr[s:e]; cl = close_arr[s:e]; gua = stk_d[s:e]
        mf = mf_arr[s:e]; sanhu = sanhu_arr[s:e]
        n = len(td)
        for i in range(LOOKBACK, n - EVAL_WIN):
            if gua[i] != '011': continue
            seg_gua = gua[i:i+EVAL_WIN]
            n_qian = (seg_gua == '111').sum()
            ret_30 = (cl[i+EVAL_WIN] / cl[i] - 1) * 100
            wlo = i - LOOKBACK + 1
            mf30 = mf[wlo:i+1].mean()
            t5 = td[i] - td[max(i-4, wlo)]
            gi = s + i
            events.append({
                'date': date_arr[gi],
                'n_qian': int(n_qian), 'ret_30': ret_30,
                'mkt_y': mkt_y[gi], 'mkt_m': mkt_m[gi], 'mkt_d': mkt_d[gi],
                'stk_m': stk_m[gi], 'stk_y': stk_y[gi],
                'trend_5d': t5, 'mf_30d_mean': mf30,
                'mf': mf[i], 'sanhu': sanhu[i],
            })
    df_e = pd.DataFrame(events)
    print(f'  巽日: {len(df_e):,}, {time.time()-t1:.1f}s')

    # === 按大盘 y_gua 分治 ===
    print(f'\n## 按大盘 y_gua 分治 (8 regime, 各自看子策略表现)')
    print()
    print(f'{"y":<5} {"事件 N":>7} {"baseline":>10} {"+B避雷":>10} {"+B+大盘m=坎":>12} {"+B+双坎":>10}')
    print('-' * 70)

    # 总览各 regime 表现
    regime_summary = []
    for y_gua in GUAS:
        sub = df_e[df_e['mkt_y'] == y_gua]
        if len(sub) < 100:
            continue
        n_total = len(sub)
        ret_base = sub['ret_30'].mean()
        zsl_base = (sub['n_qian'] >= ZSL_THRESH).mean() * 100

        # 子策略 B 避雷 (排除 mkt_y=011 不适用因为已是当前 y, 排除其他 4 项)
        # 注意: mkt_y=011 自避雷只对其他 regime 有意义, 在 mkt_y=011 自身的 regime 内不能再排除全部
        if y_gua == '011':
            # mkt_y = 011 自身的 regime 不能再 self-exclude, 跳 mkt_y 排除
            mask_b = (
                (sub['mkt_d'] != '110') &
                (sub['stk_m'] != '101') &
                (sub['trend_5d'] >= 0) &
                (sub['mf_30d_mean'] <= 0)
            )
        else:
            mask_b = (
                (sub['mkt_d'] != '110') &
                (sub['stk_m'] != '101') &
                (sub['trend_5d'] >= 0) &
                (sub['mf_30d_mean'] <= 0)
            )
        sub_b = sub[mask_b]
        if len(sub_b) < 30:
            ret_b = float('nan'); zsl_b = float('nan')
        else:
            ret_b = sub_b['ret_30'].mean(); zsl_b = (sub_b['n_qian'] >= ZSL_THRESH).mean() * 100

        # B + 大盘 m = 010 坎
        sub_e = sub_b[sub_b['mkt_m'] == '010']
        if len(sub_e) < 30:
            ret_e = float('nan')
        else:
            ret_e = sub_e['ret_30'].mean()

        # B + 双坎
        sub_f = sub_b[(sub_b['mkt_m'] == '010') & (sub_b['stk_m'] == '010')]
        if len(sub_f) < 30:
            ret_f = float('nan')
        else:
            ret_f = sub_f['ret_30'].mean()

        print(f'{y_gua}{GUA_NAMES[y_gua]:<3} {n_total:>7,} {ret_base:>+8.2f}% [{zsl_base:>4.0f}%]'
              f' {ret_b:>+7.2f}% ({len(sub_b):>5})  '
              f'{ret_e:>+7.2f}% ({len(sub_e):>5})  '
              f'{ret_f:>+7.2f}% ({len(sub_f):>4})')
        regime_summary.append({
            'y_gua': y_gua, 'n': n_total,
            'ret_base': ret_base, 'ret_b': ret_b, 'ret_e': ret_e, 'ret_f': ret_f,
            'n_b': len(sub_b), 'n_e': len(sub_e), 'n_f': len(sub_f),
        })

    # === 详细看每个 regime 内 大盘 m_gua 8 态的细分 ===
    print(f'\n\n## 各 regime 内 大盘 m_gua 8 态细分 期望 ret')
    print(f'\n  {"regime y":<8} | {"":>6}', end='')
    for m_v in GUAS:
        print(f' {m_v}{GUA_NAMES[m_v]:<2}', end='')
    print()
    print('  ' + '-' * 80)
    for y_gua in GUAS:
        sub_y = df_e[df_e['mkt_y'] == y_gua]
        if len(sub_y) < 100: continue
        print(f'  {y_gua}{GUA_NAMES[y_gua]:<5} | n={len(sub_y):<5}', end='')
        for m_v in GUAS:
            sub_ym = sub_y[sub_y['mkt_m'] == m_v]
            if len(sub_ym) < 30:
                print(f' {"--":>5}', end='')
            else:
                print(f' {sub_ym["ret_30"].mean():>+4.1f}', end='')
        print()

    # === 每 regime 找最佳大盘 m_gua + 个股 m_gua 组合 ===
    print(f'\n\n## 每 regime 最佳 (大盘y, 大盘m, 个股m) 组合 (n≥200)')
    rows = []
    for y_v in GUAS:
        sub_y = df_e[df_e['mkt_y'] == y_v]
        if len(sub_y) < 200: continue
        best = None
        for mm in GUAS:
            for sm in GUAS:
                sub = sub_y[(sub_y['mkt_m'] == mm) & (sub_y['stk_m'] == sm)]
                if len(sub) < 200: continue
                ret = sub['ret_30'].mean()
                zsl = (sub['n_qian'] >= ZSL_THRESH).mean() * 100
                if best is None or ret > best[0]:
                    best = (ret, mm, sm, len(sub), zsl)
        if best:
            ret, mm, sm, n, zsl = best
            base_y = sub_y['ret_30'].mean()
            print(f'  {y_v}{GUA_NAMES[y_v]:<3} regime: 最佳 大盘m={mm}{GUA_NAMES[mm]} 个股m={sm}{GUA_NAMES[sm]}  '
                  f'n={n:>5,}  期望 {ret:>+6.2f}% (vs regime base {base_y:>+5.2f}, lift {ret-base_y:>+5.2f})  '
                  f'主升率 {zsl:>4.1f}%')


if __name__ == '__main__':
    main()
