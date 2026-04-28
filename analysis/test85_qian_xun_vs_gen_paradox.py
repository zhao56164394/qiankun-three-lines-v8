# -*- coding: utf-8 -*-
"""验证悖论: 巽日占主升浪起点前一日 60.6%, 但艮日 baseline +2.31% 比巽日 +1.70% 高
这是为什么?

拆解:
  对每个触发卦 (巽/艮/兑/坎):
    - 主升浪事件数 (n_qian≥10) + 期望
    - 假突破事件数 (n_qian<10) + 期望
    - 同一卦内, 真主升浪贡献多少 / 假突破拖累多少
    - 看不同评估窗口 (15/20/30/45 日) 期望对比
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
EVAL_WIN_LIST = [15, 20, 30, 45, 60]
QIAN_RUN = 10
REGIME_Y = '111'

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}


def main():
    t0 = time.time()
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g['stk_d'] = g['d_gua'].astype(str).str.zfill(3)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'mkt_y']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    # 重点 4 个触发卦
    TRIGGERS = ['011', '001', '110', '010']  # 巽/艮/兑/坎

    print(f'\n=== 4 个触发卦在乾 regime 拆解 ===')
    print(f'\n## 5 个评估窗口下 各触发卦平均期望')
    print(f'  {"卦":<10} {"n":>8} {"主升率%":>8} ', end='')
    for w in EVAL_WIN_LIST:
        print(f'{w}d% '.rjust(7), end='')
    print()

    rows = []
    for state in TRIGGERS:
        events = []
        for ci in range(len(code_starts)):
            s = code_starts[ci]; e = code_ends[ci]
            if e - s < LOOKBACK + max(EVAL_WIN_LIST) + 5: continue
            cl = close_arr[s:e]; gua = stk_d_arr[s:e]
            n = len(gua)
            for i in range(LOOKBACK, n - max(EVAL_WIN_LIST)):
                gi = s + i
                if mkt_y_arr[gi] != REGIME_Y: continue
                if stk_d_arr[gi] != state: continue
                seg_gua = gua[i:i+30]
                n_qian = (seg_gua == '111').sum()
                rec = {'is_zsl': n_qian >= QIAN_RUN, 'n_qian': int(n_qian), 'date': date_arr[gi]}
                for w in EVAL_WIN_LIST:
                    rec[f'r{w}'] = (cl[i+w] / cl[i] - 1) * 100
                events.append(rec)
        df_e = pd.DataFrame(events)

        n = len(df_e)
        zsl = df_e['is_zsl'].mean() * 100
        label = f'{state}{GUA_NAMES[state]}'
        line = f'  {label:<10} {n:>8,} {zsl:>7.1f} '
        for w in EVAL_WIN_LIST:
            r = df_e[f'r{w}'].mean()
            line += f'{r:>+5.2f} '.rjust(7)
        print(line)
        rows.append((state, df_e))

    # 主升浪 vs 假突破 拆解 (用 30 日)
    print(f'\n## 主升浪 vs 假突破 拆解 (30 日窗口)')
    print(f'  {"卦":<10} {"主升n":>7} {"主升期望":>9} {"假n":>7} {"假期望":>8} {"加权平均":>9}')
    for state, df_e in rows:
        zsl = df_e[df_e['is_zsl']]
        fake = df_e[~df_e['is_zsl']]
        zsl_ret = zsl['r30'].mean()
        fake_ret = fake['r30'].mean()
        avg = df_e['r30'].mean()
        label = f'{state}{GUA_NAMES[state]}'
        print(f'  {label:<10} {len(zsl):>7,} {zsl_ret:>+8.2f} {len(fake):>7,} {fake_ret:>+7.2f} {avg:>+8.2f}')

    # 验证: 巽日主升浪起点前一日 60.6%, 是不是因为巽日"基数大"
    print(f'\n## 巽日 vs 艮日 主升浪起点贡献分析')
    print(f'  巽日总 n=201,538, 主升浪起点 6,846 → 主升起点率 6846/201538 = {6846/201538*100:.2f}%')
    print(f'  艮日总 n=61,265,  主升浪起点 325   → 主升起点率 325/61265 = {325/61265*100:.2f}%')
    print(f'  → 巽日的"主升浪起点产出率"({6846/201538*100:.2f}%) 比艮日({325/61265*100:.2f}%)高 6×')
    print(f'  → 但巽日基数大 3.3×, 所以总主升浪数巽 6846 vs 艮 325 = 21×')

    # 业务结论
    print(f'\n## 业务结论')
    print(f'  巽日 baseline +1.70% (主升 +12% / 假 -7% 加权)')
    print(f'  艮日 baseline +2.31% (主升 +X / 假 +Y 加权)')


if __name__ == '__main__':
    main()
