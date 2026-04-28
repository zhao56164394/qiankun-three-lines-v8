# -*- coding: utf-8 -*-
"""变爻探索: y_gua 在 IS 期 (2014-2022) 的变化频率, 方向, 持续时间

输出:
  1. y_gua 状态序列 (按时间)
  2. 每次状态切换 (from→to + 日期 + 哪位变)
  3. 每个 y_gua 状态的持续天数分布
  4. sig 按"距上次变爻 d 天" 切片的 alpha
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

POS_NAMES = ['天位(长期)', '人位(中期)', '地位(短期)']  # 假设位顺序为 天/人/地, 高位到低位


def load_y_gua_series():
    """加载 IS 期 y_gua 时间序列"""
    p = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    df = pd.read_parquet(p, columns=['date', 'y_gua'])
    df['date'] = df['date'].astype(str)
    df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
    df = df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    return df


def find_changes(df):
    """找出所有 y_gua 状态切换点"""
    changes = []
    df = df.copy()
    df['prev_y'] = df['y_gua'].shift(1)
    for _, row in df.iterrows():
        if pd.isna(row['prev_y']):
            continue
        if row['y_gua'] != row['prev_y']:
            from_y = row['prev_y']
            to_y = row['y_gua']
            # 哪几位变
            diff_pos = []
            for i, (a, b) in enumerate(zip(from_y, to_y)):
                if a != b:
                    diff_pos.append((i, POS_NAMES[i], a, b))
            n_diff = len(diff_pos)
            changes.append({
                'date': row['date'],
                'from_y': from_y,
                'to_y': to_y,
                'n_diff_bits': n_diff,
                'diff_pos_str': ' '.join([f'{name}:{a}->{b}' for _, name, a, b in diff_pos]),
            })
    return changes


def compute_state_durations(df):
    """每段稳定 y_gua 状态的持续天数"""
    runs = []
    cur_y, cur_start = df.iloc[0]['y_gua'], df.iloc[0]['date']
    for _, row in df.iloc[1:].iterrows():
        if row['y_gua'] != cur_y:
            runs.append({'y_gua': cur_y, 'start': cur_start, 'end': row['date']})
            cur_y = row['y_gua']
            cur_start = row['date']
    runs.append({'y_gua': cur_y, 'start': cur_start, 'end': df.iloc[-1]['date']})
    for r in runs:
        r['duration_days'] = (pd.to_datetime(r['end']) - pd.to_datetime(r['start'])).days
    return runs


def annotate_sigs_with_change(sigs_path, df_y):
    """为 sig 加上 距上次 y_gua 变爻的天数 + 上次变爻 from→to"""
    with open(sigs_path, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    sigs['buy_date'] = sigs['buy_date'].astype(str)

    # 构建 y_gua 序列 + 每天对应"距上次变爻 d 天" + "上次变爻 from→to"
    df = df_y.copy()
    df['prev_y'] = df['y_gua'].shift(1)
    df['is_change'] = (df['y_gua'] != df['prev_y']) & df['prev_y'].notna()
    df['last_change_date'] = df['date'].where(df['is_change']).ffill()
    df['last_from'] = df['prev_y'].where(df['is_change']).ffill()
    df['days_since_change'] = pd.to_datetime(df['date']).sub(
        pd.to_datetime(df['last_change_date'])).dt.days
    df_lookup = df.set_index('date')[['y_gua', 'days_since_change', 'last_from']]

    sigs['days_since_change'] = sigs['buy_date'].map(df_lookup['days_since_change'])
    sigs['last_from'] = sigs['buy_date'].map(df_lookup['last_from'])
    sigs['y_gua'] = sigs['buy_date'].map(df_lookup['y_gua'])
    return sigs


def main():
    df_y = load_y_gua_series()
    print(f'y_gua 序列: {len(df_y)} 天 ({df_y["date"].min()} ~ {df_y["date"].max()})')

    # 1. 各 y_gua 状态总天数
    print('\n=== 各 y_gua 状态总天数 ===')
    counts = df_y['y_gua'].value_counts().sort_index()
    for y, n in counts.items():
        print(f'  {y} {GUA_NAMES.get(y, "?")}: {n} 天 ({n/len(df_y)*100:.1f}%)')

    # 2. 状态切换列表
    changes = find_changes(df_y)
    print(f'\n=== y_gua 状态切换次数: {len(changes)} 次 ===')
    print(f'  变 1 位: {sum(1 for c in changes if c["n_diff_bits"]==1)}')
    print(f'  变 2 位: {sum(1 for c in changes if c["n_diff_bits"]==2)}')
    print(f'  变 3 位: {sum(1 for c in changes if c["n_diff_bits"]==3)}')

    # 3. 列出所有切换 (前 30 + 后 10)
    print('\n=== 所有 y_gua 切换事件 ===')
    print(f'  {"日期":<10} {"from→to":<12} {"位差":>4} {"变爻":<25}')
    print('  ' + '-' * 60)
    for c in changes:
        from_name = GUA_NAMES.get(c['from_y'], '?')
        to_name = GUA_NAMES.get(c['to_y'], '?')
        print(f'  {c["date"]:<10} {c["from_y"]}{from_name}→{c["to_y"]}{to_name}  '
              f'{c["n_diff_bits"]:>4}  {c["diff_pos_str"]}')

    # 4. 每段持续期
    runs = compute_state_durations(df_y)
    print(f'\n=== 每段 y_gua 状态持续期 (共 {len(runs)} 段) ===')
    print(f'  {"y_gua":<8} {"start":<10} {"end":<10} {"持续天":>7}')
    print('  ' + '-' * 50)
    for r in runs:
        print(f'  {r["y_gua"]:<3} {GUA_NAMES.get(r["y_gua"], "?"):<4} '
              f'{r["start"]:<10} {r["end"]:<10} {r["duration_days"]:>7}')
    durations = [r['duration_days'] for r in runs]
    print(f'\n  持续天数: 中位 {np.median(durations):.0f} 天, '
          f'min {min(durations)}, max {max(durations)}, mean {np.mean(durations):.0f}')

    # 5. sig 按"距上次变爻 d 天" 切片
    sig_path = os.path.join(ROOT, 'data_layer', 'data', 'ablation',
                            'test6_pool_depth', 'baseline_IS.json')
    sigs = annotate_sigs_with_change(sig_path, df_y)
    sigs = sigs[sigs['days_since_change'].notna()].copy()
    sigs['d_bucket'] = pd.cut(sigs['days_since_change'],
                               bins=[-1, 30, 90, 180, 365, 100000],
                               labels=['0-30天', '31-90天', '91-180天',
                                       '181-365天', '>365天'])
    print('\n=== sig 按 距上次变爻 d 天 分桶 (全量, 不分卦) ===')
    print(f'  {"d 桶":<14} {"n":>6} {"mean%":>7} {"win%":>6}')
    print('  ' + '-' * 40)
    for d in ['0-30天', '31-90天', '91-180天', '181-365天', '>365天']:
        sub = sigs[sigs['d_bucket'] == d]
        if len(sub) == 0:
            continue
        win = (sub['actual_ret'] > 0).mean() * 100
        print(f'  {d:<14} {len(sub):>6} {sub["actual_ret"].mean():>+7.2f} {win:>6.1f}')

    # 6. sig 按"上次变爻 from→to" 切片 (限于占比>2% 的变爻类型)
    print('\n=== sig 按 上次变爻 from→to 切片 (sig >100 才显示) ===')
    sigs['change_type'] = sigs['last_from'].astype(str) + '→' + sigs['y_gua'].astype(str)
    g = sigs.groupby('change_type', observed=True)['actual_ret'].agg(['count', 'mean'])
    g = g[g['count'] >= 100].sort_values('mean', ascending=False)
    print(f'  {"from→to":<14} {"n":>6} {"mean%":>7}')
    print('  ' + '-' * 40)
    for ct, row in g.iterrows():
        print(f'  {ct:<14} {row["count"]:>6} {row["mean"]:>+7.2f}')


if __name__ == '__main__':
    main()
