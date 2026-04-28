# -*- coding: utf-8 -*-
"""实验 1: 年卦 SPD×ACC 滞后带扫描

对每个 (SPD_HYST, ACC_HYST) 组合, 重算 y_gua, 统计:
  - 牛市阳卦命中率
  - 熊市阴卦命中率
  - y_gua 切换次数 / 频率
  - 主导 y_gua 是否对应业务标签 (深熊/主升 等)

不跑回测, 纯统计 (30 秒内出结果).

输出: top 候选清单
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'data_layer'))

from prepare_multi_scale_gua import compute_scale_per_day, apply_v10_rules
import prepare_multi_scale_gua as pmg

# 读取原始数据
src = os.path.join(ROOT, 'data_layer', 'data', 'zz1000_daily.csv')
df = pd.read_csv(src, encoding='utf-8-sig')
df['date'] = df['date'].astype(str)
print(f'读取 {len(df)} 条 ({df["date"].iloc[0]} ~ {df["date"].iloc[-1]})')

# 月尺度 trend / mf 只算一次 (与 SPD/ACC 无关, 只与窗口/聚合周期有关)
print('计算月尺度 trend / mf ...')
y_trend, y_mf = compute_scale_per_day(df, 'M')

# 牛熊区间 (沿用脚本里的)
bull_periods = [
    ('2014-07-01', '2015-06-12', '14-15杠杆牛'),
    ('2019-01-04', '2019-04-19', '19Q1反弹'),
    ('2020-04-01', '2021-02-18', '20疫后小牛'),
    ('2024-09-24', '2025-12-31', '24-25政策牛'),
]
bear_periods = [
    ('2015-06-15', '2016-01-28', '15股灾+熔断'),
    ('2018-02-01', '2018-12-28', '18贸战熊'),
    ('2021-02-22', '2024-02-05', '21-24深熊'),
]


def eval_config(spd_hyst, acc_hyst):
    """重算 y_gua 用给定 SPD/ACC, 返回评估指标"""
    # 临时改全局参数
    orig_spd = pmg.SPD_HYST
    orig_acc = pmg.ACC_HYST
    pmg.SPD_HYST = spd_hyst
    pmg.ACC_HYST = acc_hyst
    y_pos, y_spd, y_acc, y_gua = apply_v10_rules(y_trend, y_mf)
    pmg.SPD_HYST = orig_spd
    pmg.ACC_HYST = orig_acc

    g = pd.Series(y_gua, index=df['date'].values)
    g_valid = g[g != '']

    # 切换次数
    chg = (g_valid != g_valid.shift()).astype(int)
    if len(chg) > 0:
        chg.iloc[0] = 0
    n_switch = int(chg.sum())
    n_segs = n_switch + 1
    avg_seg_days = len(g_valid) / max(n_segs, 1)

    # 牛市阳卦占比
    bull_yang_pct = []
    bull_total = 0
    bull_yang = 0
    for s, e, _ in bull_periods:
        sub = g_valid[(g_valid.index >= s) & (g_valid.index <= e)]
        if len(sub) == 0: continue
        pos = sub.str[0].astype(int)
        bull_yang += (pos==1).sum()
        bull_total += len(sub)
    bull_hit = bull_yang / max(bull_total, 1) * 100

    # 熊市阴卦占比
    bear_total = 0
    bear_yin = 0
    for s, e, _ in bear_periods:
        sub = g_valid[(g_valid.index >= s) & (g_valid.index <= e)]
        if len(sub) == 0: continue
        pos = sub.str[0].astype(int)
        bear_yin += (pos==0).sum()
        bear_total += len(sub)
    bear_hit = bear_yin / max(bear_total, 1) * 100

    # 综合命中率
    combined_hit = (bull_yang + bear_yin) / max(bull_total + bear_total, 1) * 100

    # y_gua 分布
    counts = g_valid.value_counts()
    top = counts.head(3).to_dict()

    return {
        'spd': spd_hyst, 'acc': acc_hyst,
        'switches': n_switch,
        'avg_seg_days': avg_seg_days,
        'bull_hit%': bull_hit,
        'bear_hit%': bear_hit,
        'combined_hit%': combined_hit,
        'top3': top,
    }


# 16 组扫描
SPD_VALS = [1.0, 2.0, 3.0, 5.0]
ACC_VALS = [15.0, 20.0, 30.0, 50.0]

print(f'\n=== 实验 1: 16 组 (SPD × ACC) 配置评估 ===\n')
results = []
for spd in SPD_VALS:
    for acc in ACC_VALS:
        r = eval_config(spd, acc)
        results.append(r)

# 排序: 综合命中率高 + 切换不太频繁
df_res = pd.DataFrame(results)
df_res = df_res.sort_values('combined_hit%', ascending=False)

print(f'  {"SPD":>4} {"ACC":>4} {"切换":>5} {"段长(天)":>8} {"牛命中%":>8} {"熊命中%":>8} {"综合%":>7}  top3 y_gua')
for _, r in df_res.iterrows():
    is_baseline = (r['spd']==2 and r['acc']==30)
    mark = ' ←现行' if is_baseline else ''
    top_str = ', '.join(f'{k}({v})' for k, v in list(r['top3'].items())[:3])
    print(f'  {r["spd"]:>4.1f} {r["acc"]:>4.0f} {r["switches"]:>5} {r["avg_seg_days"]:>8.0f} '
          f'{r["bull_hit%"]:>7.1f}% {r["bear_hit%"]:>7.1f}% {r["combined_hit%"]:>6.1f}%  {top_str}{mark}')

# Top 3 候选
print(f'\n=== Top 3 候选 (按综合命中率) ===')
for i, (_, r) in enumerate(df_res.head(3).iterrows(), 1):
    print(f'  {i}. SPD={r["spd"]} ACC={r["acc"]}: 综合 {r["combined_hit%"]:.1f}%, 切换 {r["switches"]}, 主导 {list(r["top3"].keys())[0]}')

# 落地
import json
with open(os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test3', 'phase2_y_gua_param_scan.json'), 'w', encoding='utf-8') as f:
    json.dump([{k:v for k,v in r.items() if k!='top3'} | {'top3_keys': list(r['top3'].keys())} for _, r in df_res.iterrows()],
              f, ensure_ascii=False, indent=2, default=str)
print(f'\n  落地: phase2_y_gua_param_scan.json')
