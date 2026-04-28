# -*- coding: utf-8 -*-
"""为什么 sig 在 y_gua 桶下不均衡? 是状态天数本身少, 还是 cfg 过滤造成?

输出:
  1. 12 月窗口版 y_gua 各状态在 2014-2022 IS 段的天数分布
  2. sig 在该 y_gua 状态下的"触发率" = sig_n / state_days
  3. 对比新旧 (12 vs 55) 月版的天数分布 (确认窗口改动没让某状态消失)
"""
import os
import sys
import io
import json
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUA_NAME = {
    '000': '坤(深熊)', '001': '艮(吸筹)', '010': '坎(乏力)', '011': '巽(底爆)',
    '100': '震(出货)', '101': '离(护盘)', '110': '兑(末减)', '111': '乾(疯牛)',
}

IS_START, IS_END = '20140101', '20221231'


def load_y_gua(parquet_name):
    path = os.path.join(ROOT, 'data_layer', 'data', 'foundation', parquet_name)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=['date', 'y_gua'])
    df['date'] = df['date'].astype(str)
    df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
    return df


def main():
    # 1. 当前版 (12 月窗口)
    df_new = load_y_gua('multi_scale_gua_daily.parquet')
    df_old = load_y_gua('multi_scale_gua_daily.window55_backup.parquet')

    print(f'\n=== 12 月窗口版 总天数: {len(df_new)} ===')
    if df_old is not None:
        print(f'=== 55 月窗口版 (备份) 总天数: {len(df_old)} ===')
    else:
        print('(55 月备份不存在, 跳过对比)')

    # 2. IS 段过滤
    df_new_is = df_new[(df_new['date'] >= IS_START) & (df_new['date'] <= IS_END)]
    print(f'\nIS 段 (2014-2022): {len(df_new_is)} 天')

    # 3. 各 y_gua 状态天数 (12 月 IS)
    print('\n' + '=' * 80)
    print('y_gua 状态天数分布 — 12 月 IS (2014-2022)')
    print('=' * 80)
    cnt_new = df_new_is['y_gua'].value_counts().sort_index()
    total_new = cnt_new.sum()
    print(f'  {"y_gua":<6} {"卦名":<10} {"天数":>6} {"占比":>7}')
    print('  ' + '-' * 40)
    for y in sorted(cnt_new.index):
        n = cnt_new[y]
        pct = n / total_new * 100
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} {n:>6d} {pct:>6.1f}%')

    # 4. 对比新旧 (如有)
    if df_old is not None:
        df_old_is = df_old[(df_old['date'] >= IS_START) & (df_old['date'] <= IS_END)]
        cnt_old = df_old_is['y_gua'].value_counts().sort_index()
        total_old = cnt_old.sum()

        print('\n' + '=' * 80)
        print('对比: 12 月 vs 55 月 在同一 IS 段的天数分布')
        print('=' * 80)
        print(f'  {"y_gua":<6} {"卦名":<10} {"12月天数":>9} {"12月%":>7} '
              f'{"55月天数":>9} {"55月%":>7}')
        print('  ' + '-' * 60)
        all_y = sorted(set(cnt_new.index) | set(cnt_old.index))
        for y in all_y:
            n_new = int(cnt_new.get(y, 0))
            n_old = int(cnt_old.get(y, 0))
            p_new = n_new / total_new * 100
            p_old = n_old / total_old * 100
            print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} '
                  f'{n_new:>9d} {p_new:>6.1f}% {n_old:>9d} {p_old:>6.1f}%')

    # 5. sig 触发率 = IS_baseline 的 sig_n 在该 y_gua 下 / 该 y_gua 状态天数
    print('\n' + '=' * 80)
    print('sig 触发率: sig_n / 状态天数 (信号"触发密度")')
    print('=' * 80)
    is_path = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3', 'IS_baseline.json')
    with open(is_path, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    sigs['buy_date'] = sigs['buy_date'].astype(str)
    y_map = dict(zip(df_new['date'], df_new['y_gua']))
    sigs['y_gua'] = sigs['buy_date'].map(y_map)

    sig_cnt = sigs['y_gua'].value_counts().sort_index()
    print(f'  {"y_gua":<6} {"卦名":<10} {"天数":>6} {"sig_n":>6} '
          f'{"sig/天":>8} {"sig %":>7}')
    print('  ' + '-' * 55)
    for y in sorted(cnt_new.index):
        n_day = int(cnt_new[y])
        n_sig = int(sig_cnt.get(y, 0))
        rate = n_sig / n_day if n_day else 0
        sig_pct = n_sig / sig_cnt.sum() * 100
        print(f'  {y:<6} {GUA_NAME.get(y, "?"):<10} {n_day:>6d} {n_sig:>6d} '
              f'{rate:>8.2f} {sig_pct:>6.1f}%')

    print('\n说明:')
    print('  - sig/天 高 = 该 y_gua 状态下信号触发频繁 (熊市低位多)')
    print('  - sig/天 低 = 该 y_gua 状态下 cfg 过滤偏严或本就少触发')
    print('  - 天数 vs sig% 严重错位 → 反映是 cfg 偏好不是市场客观')


if __name__ == '__main__':
    main()
