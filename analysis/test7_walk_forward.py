# -*- coding: utf-8 -*-
"""test7 v1/v2/v3 + baseline (test6) 的 7 段 walk-forward 验证.

每段窗口: (label, start, end, 描述)
对每段跑 4 个 cfg (baseline / v1 / v2 / v3), 共 7 × 4 = 28 跑.
看 v1/v2/v3 的 alpha (vs baseline) 在 7 个 OOS 段的稳定性.

按 Phase 2 反过拟合纪律: 真规律 = 大部分段都 +, 0 反向.
"""
import os
import sys
import json
import time
import functools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from test6_pool_depth_ablation import setup_data, run_case
from strategy_configs import get_strategy

print = functools.partial(print, flush=True)

OUT_DIR = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test7_walk_forward')
os.makedirs(OUT_DIR, exist_ok=True)

# 7 段窗口 (与 Phase 2 一致)
WINDOWS = [
    ('w1_2018',    '20180101', '20190101', '2018 大熊'),
    ('w2_2019',    '20190101', '20200101', '2019 反弹'),
    ('w3_2020',    '20200101', '20210101', '2020 抱团'),
    ('w4_2021',    '20210101', '20220101', '2021 延续'),
    ('w5_2022',    '20220101', '20230101', '2022 杀跌'),
    ('w6_2023_24', '20230101', '20250101', '2023-24 震荡'),
    ('w7_2025_26', '20250101', '20260417', '2025-26 慢牛'),
]

VERSIONS = ['test6', 'test7v1', 'test7v2', 'test7v3']


def main():
    print('=== 加载数据 ===')
    data = setup_data()

    summary = []
    total_runs = len(WINDOWS) * len(VERSIONS)
    run_idx = 0
    t_start = time.time()
    for label, start, end, desc in WINDOWS:
        print(f'\n{"=" * 80}')
        print(f'窗口 {label} ({start} ~ {end}, {desc})')
        print('=' * 80)
        for ver in VERSIONS:
            os.environ['STRATEGY_VERSION'] = ver
            cfg = get_strategy()
            run_idx += 1
            t0 = time.time()
            sig, trd, stat = run_case(data, cfg, start, end, init_capital=200000)
            elapsed = time.time() - t0
            cap = stat['final_capital'] / 10000
            print(f'  [{run_idx}/{total_runs}] {ver:<10}: 终值 {cap:>5.1f}万, '
                  f'sig {len(sig):>4}, trd {len(trd):>3}, {elapsed:.1f}s')
            summary.append({
                'window': label, 'desc': desc,
                'start': start, 'end': end,
                'version': ver,
                'final_cap_wan': cap,
                'sig_n': len(sig),
                'trd_n': len(trd),
            })

    elapsed = time.time() - t_start
    print(f'\n=== 完成 {total_runs} 跑, 总耗时 {elapsed/60:.1f} 分钟 ===\n')

    # 落地
    import pandas as pd
    sdf = pd.DataFrame(summary)
    out_csv = os.path.join(OUT_DIR, 'walk_forward_summary.csv')
    sdf.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'落地: {out_csv}')

    # 总览表
    print('\n' + '=' * 100)
    print('# 7 段 walk-forward 终值对比 (单位: 万)')
    print('=' * 100)
    print(f'  {"窗口":<14} {"描述":<14} ' + ' '.join(f'{v:>10}' for v in VERSIONS))
    print('  ' + '-' * 80)
    for label, _, _, desc in WINDOWS:
        row = [f'  {label:<14} {desc:<14}']
        for ver in VERSIONS:
            r = sdf[(sdf['window']==label) & (sdf['version']==ver)]
            if len(r) == 0:
                row.append(f'{"-":>10}')
            else:
                row.append(f'{r.iloc[0]["final_cap_wan"]:>10.1f}')
        print(' '.join(row))

    # 各版本 7 段累加 + alpha 计算
    print('\n' + '=' * 100)
    print('# 各 cfg 版本 7 段 alpha 分布 (alpha = ver_终值 - baseline_终值, 单位: 万)')
    print('=' * 100)
    print(f'  {"窗口":<14} {"baseline":>9} ' + ' '.join(f'{v+"_α":>11}' for v in VERSIONS[1:]))
    print('  ' + '-' * 70)

    total_alpha = {ver: 0 for ver in VERSIONS[1:]}
    sign_count = {ver: {'+': 0, '-': 0, '0': 0} for ver in VERSIONS[1:]}
    for label, _, _, desc in WINDOWS:
        baseline = sdf[(sdf['window']==label) & (sdf['version']=='test6')].iloc[0]['final_cap_wan']
        row = [f'  {label:<14} {baseline:>+8.1f}']
        for ver in VERSIONS[1:]:
            r = sdf[(sdf['window']==label) & (sdf['version']==ver)]
            cap = r.iloc[0]['final_cap_wan']
            alpha = cap - baseline
            total_alpha[ver] += alpha
            if alpha > 0.5:
                sign_count[ver]['+'] += 1
                marker = '✅'
            elif alpha < -0.5:
                sign_count[ver]['-'] += 1
                marker = '❌'
            else:
                sign_count[ver]['0'] += 1
                marker = '○ '
            row.append(f'{alpha:>+8.1f} {marker}')
        print(' '.join(row))

    print(f'\n  {"7 段汇总":<14} {"":>9} ' +
          ' '.join(f'{total_alpha[v]:>+8.1f}   ' for v in VERSIONS[1:]))

    print('\n[反过拟合判定]')
    for ver in VERSIONS[1:]:
        sc = sign_count[ver]
        verdict = '★ 真规律' if sc['+'] >= 5 and sc['-'] <= 1 else \
                  ('○ 不稳定' if sc['+'] >= 4 else '✗ 非规律')
        print(f'  {ver}: + {sc["+"]} / - {sc["-"]} / ○ {sc["0"]}, '
              f'7 段累加 alpha {total_alpha[ver]:+.1f}万 {verdict}')


if __name__ == '__main__':
    main()
