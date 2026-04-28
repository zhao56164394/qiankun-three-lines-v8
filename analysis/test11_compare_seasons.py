# -*- coding: utf-8 -*-
"""4 种 8 卦→季节合并方案对比, sig 视角

判定标准:
  季间分离: 各季 mean 差距大 (季间方差大)
  季内一致: 同季内 y_gua mean 接近 (季内方差小)
  ANOVA F = 季间方差 / 季内方差 越大越好
  bootstrap 各季 mean CI 不重叠 = 真分治
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sigs():
    p = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth', 'baseline_IS.json')
    with open(p, encoding='utf-8') as f:
        d = json.load(f)
    sigs = pd.DataFrame(d['signal_detail'])
    p = os.path.join(ROOT, 'data_layer', 'data', 'foundation', 'multi_scale_gua_daily.parquet')
    yg = pd.read_parquet(p, columns=['date', 'y_gua'])
    yg['date'] = yg['date'].astype(str)
    yg['y_gua'] = yg['y_gua'].astype(str).str.zfill(3)
    sigs['y_gua'] = sigs['buy_date'].astype(str).map(dict(zip(yg['date'], yg['y_gua'])))
    return sigs


def boot_ci(arr, n_boot=1000, seed=42):
    if len(arr) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


SCHEMES = {
    'A_business': {  # 业务直觉 1+3+1+3 (我用的)
        '熊_探底': ['000'],
        '转牛': ['001', '010', '011'],
        '牛_主升': ['111'],
        '转熊': ['100', '101', '110'],
    },
    'B_bitcount': {  # 按位爻向上数 1+3+3+1
        '0位_全低': ['000'],
        '1位': ['001', '010', '100'],
        '2位': ['011', '101', '110'],
        '3位_全高': ['111'],
    },
    'C_half': {  # 二分高低 4+4 (按首位 = 天位)
        '低_首位0': ['000', '001', '010', '011'],
        '高_首位1': ['100', '101', '110', '111'],
    },
    'D_meancluster': {  # IS sig mean 自然分组 (有过拟合风险, 仅参考)
        '强正_mean>5': ['000', '010'],
        '弱中_mean[-2,5]': ['001', '011', '100', '110', '111'],
        '大坑_mean<-5': ['101'],
    },
    'E_2bit_no_renpos': {  # 用户提: 人位忽略, 2爻 (天+地) 4类
        '00_低低': ['000', '010'],
        '01_低高': ['001', '011'],
        '10_高低': ['100', '110'],
        '11_高高': ['101', '111'],
    },
}


def evaluate_scheme(name, mapping, sigs):
    """评估一种合并方案"""
    print(f'\n{"=" * 90}')
    print(f'# 方案 {name}')
    print('=' * 90)

    # 反向映射
    yg2season = {}
    for season, ygs in mapping.items():
        for yg in ygs:
            yg2season[yg] = season
    sigs = sigs.copy()
    sigs['season'] = sigs['y_gua'].map(yg2season)

    # 各季 + 季内分裂统计
    print(f'\n{"季":<22} {"含卦":<24} {"n":>6} {"mean%":>7} {"95%CI":>16}')
    print('-' * 80)
    season_means = {}
    for season, ygs in mapping.items():
        sub = sigs[sigs['season'] == season]
        n = len(sub)
        m = sub['actual_ret'].mean()
        ci_lo, ci_hi = boot_ci(sub['actual_ret'].values)
        ci_str = f'[{ci_lo:+.2f},{ci_hi:+.2f}]' if ci_lo else 'n/a'
        print(f'{season:<22} {",".join(ygs):<24} {n:>6} {m:>+7.2f} {ci_str:>16}')
        season_means[season] = (n, m, ci_lo, ci_hi)

        # 季内分裂 (各 y_gua 单独看)
        if len(ygs) > 1:
            for yg in ygs:
                yg_sub = sigs[sigs['y_gua'] == yg]
                if len(yg_sub) < 10:
                    continue
                ym = yg_sub['actual_ret'].mean()
                yci = boot_ci(yg_sub['actual_ret'].values)
                yci_str = f'[{yci[0]:+.2f},{yci[1]:+.2f}]' if yci[0] else 'n/a'
                print(f'  └ {yg:<19} {"":<24} {len(yg_sub):>6} {ym:>+7.2f} {yci_str:>16}')

    # ANOVA: 季间 vs 季内方差
    grand_mean = sigs['actual_ret'].mean()
    ss_between = 0  # 季间平方和
    ss_within = 0   # 季内平方和
    n_groups = 0
    n_total = 0
    for season, ygs in mapping.items():
        sub = sigs[sigs['season'] == season]
        if len(sub) < 30:
            continue
        n_groups += 1
        n_total += len(sub)
        ss_between += len(sub) * (sub['actual_ret'].mean() - grand_mean) ** 2
        ss_within += ((sub['actual_ret'] - sub['actual_ret'].mean()) ** 2).sum()
    if n_groups > 1:
        df_between = n_groups - 1
        df_within = n_total - n_groups
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within if df_within > 0 else 0
        f_stat = ms_between / ms_within if ms_within > 0 else float('inf')
        # 季间分离度 (各季 mean 的 std)
        means = [m for _, m, _, _ in season_means.values()]
        between_std = float(np.std(means))
        # 各季 CI 是否不重叠 (定性)
        seasons_sorted = sorted(season_means.items(), key=lambda x: x[1][1])
        ci_overlap_count = 0
        for i in range(len(seasons_sorted) - 1):
            _, (n1, m1, lo1, hi1) = seasons_sorted[i]
            _, (n2, m2, lo2, hi2) = seasons_sorted[i+1]
            if hi1 is not None and lo2 is not None and hi1 > lo2:
                ci_overlap_count += 1
        print(f'\n  ANOVA F={f_stat:.1f} (季间方差/季内方差, 越大越好)')
        print(f'  季间 mean std = {between_std:.2f}% (越大越好)')
        print(f'  CI 重叠对: {ci_overlap_count}/{n_groups - 1} (越少越好, 0 = 完美分离)')


def main():
    sigs = load_sigs()
    print(f'baseline IS sig: {len(sigs)}, mean {sigs["actual_ret"].mean():+.2f}%')

    # 8 卦原始数据先列出
    print('\n## 原始 8 卦 sig 视角 (参考)')
    print(f'{"y_gua":<6} {"n":>6} {"mean%":>7} {"95%CI":>16}')
    print('-' * 50)
    for y in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub = sigs[sigs['y_gua'] == y]
        if len(sub) == 0: continue
        m = sub['actual_ret'].mean()
        ci = boot_ci(sub['actual_ret'].values)
        ci_str = f'[{ci[0]:+.2f},{ci[1]:+.2f}]' if ci[0] else 'n/a'
        print(f'{y:<6} {len(sub):>6} {m:>+7.2f} {ci_str:>16}')

    for name, mapping in SCHEMES.items():
        evaluate_scheme(name, mapping, sigs)

    # 跨方案汇总
    print(f'\n\n{"=" * 90}')
    print('# 跨方案 ANOVA F 汇总 (高 = 分治好)')
    print('=' * 90)


if __name__ == '__main__':
    main()
