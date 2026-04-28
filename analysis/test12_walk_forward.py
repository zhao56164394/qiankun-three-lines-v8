# -*- coding: utf-8 -*-
"""test12 yao_min walk-forward 验证

v_yao_min vs test6 baseline, 7 段 OOS:
  仅 skip 双向 ★ 2 条变爻 (111->101 + 001->000), 4827 sig (35%)
  其他不动. 这是 Phase 1 最小验证.

判定:
  5+/0- 真规律 ★, 4+ 不稳定 ○, 否则 ✗
  通过 → Phase 2 (月日卦校准)
  失败 → 重新思考变爻框架
"""
import os
import sys
import time
import functools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from test6_pool_depth_ablation import setup_data, run_case
from strategy_configs import get_strategy

print = functools.partial(print, flush=True)

OUT_DIR = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test12_walk_forward')
os.makedirs(OUT_DIR, exist_ok=True)

WINDOWS = [
    ('w1_2018',    '20180101', '20190101', '2018 大熊'),
    ('w2_2019',    '20190101', '20200101', '2019 反弹'),
    ('w3_2020',    '20200101', '20210101', '2020 抱团'),
    ('w4_2021',    '20210101', '20220101', '2021 延续'),
    ('w5_2022',    '20220101', '20230101', '2022 杀跌'),
    ('w6_2023_24', '20230101', '20250101', '2023-24 震荡'),
    ('w7_2025_26', '20250101', '20260417', '2025-26 慢牛'),
]

VERSIONS = ['test6', 'test12yaomin']


def main():
    print('=== 加载数据 ===')
    data = setup_data()

    summary = []
    total = len(WINDOWS) * len(VERSIONS)
    idx = 0
    t0_all = time.time()
    for label, start, end, desc in WINDOWS:
        print(f'\n{"=" * 80}')
        print(f'窗口 {label} ({start} ~ {end}, {desc})')
        print('=' * 80)
        for ver in VERSIONS:
            os.environ['STRATEGY_VERSION'] = ver
            cfg = get_strategy()
            idx += 1
            t0 = time.time()
            sig, trd, stat = run_case(data, cfg, start, end, init_capital=200000)
            elapsed = time.time() - t0
            cap = stat['final_capital'] / 10000
            print(f'  [{idx}/{total}] {ver:<14}: 终值 {cap:>5.1f}万, '
                  f'sig {len(sig):>4}, trd {len(trd):>3}, {elapsed:.1f}s')
            summary.append({
                'window': label, 'desc': desc, 'start': start, 'end': end,
                'version': ver, 'final_cap_wan': cap,
                'sig_n': len(sig), 'trd_n': len(trd),
            })

    elapsed = time.time() - t0_all
    print(f'\n=== 完成 {total} 跑, 总耗时 {elapsed/60:.1f} 分钟 ===\n')

    import pandas as pd
    sdf = pd.DataFrame(summary)
    out_csv = os.path.join(OUT_DIR, 'walk_forward_summary.csv')
    sdf.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'落地: {out_csv}')

    # 总览
    print('\n' + '=' * 90)
    print('# 7 段 walk-forward 终值对比 (单位: 万)')
    print('=' * 90)
    print(f'  {"窗口":<14} {"描述":<14} ' + ' '.join(f'{v:>14}' for v in VERSIONS))
    print('  ' + '-' * 80)
    for label, _, _, desc in WINDOWS:
        row = [f'  {label:<14} {desc:<14}']
        for ver in VERSIONS:
            r = sdf[(sdf['window'] == label) & (sdf['version'] == ver)]
            if len(r) == 0:
                row.append(f'{"-":>14}')
            else:
                row.append(f'{r.iloc[0]["final_cap_wan"]:>14.1f}')
        print(' '.join(row))

    # alpha
    print('\n' + '=' * 90)
    print('# yao_min alpha (vs baseline, 单位: 万)')
    print('=' * 90)
    print(f'  {"窗口":<14} {"baseline":>10} {"yaomin":>10} {"alpha":>10} {"sig 减":>8}')
    print('  ' + '-' * 65)
    total_alpha = 0
    sign = {'+': 0, '-': 0, '0': 0}
    for label, _, _, desc in WINDOWS:
        b = sdf[(sdf['window'] == label) & (sdf['version'] == 'test6')].iloc[0]
        v = sdf[(sdf['window'] == label) & (sdf['version'] == 'test12yaomin')].iloc[0]
        alpha = v['final_cap_wan'] - b['final_cap_wan']
        sig_drop = b['sig_n'] - v['sig_n']
        total_alpha += alpha
        if alpha > 0.5:
            sign['+'] += 1; marker = '✅'
        elif alpha < -0.5:
            sign['-'] += 1; marker = '❌'
        else:
            sign['0'] += 1; marker = '○ '
        print(f'  {label:<14} {b["final_cap_wan"]:>+9.1f} {v["final_cap_wan"]:>+9.1f}  '
              f'{alpha:>+8.1f} {marker}  -{sig_drop:>5}')

    print(f'\n  汇总 alpha: {total_alpha:+.1f}万,  +{sign["+"]} / -{sign["-"]} / ○{sign["0"]}')
    verdict = '★ 真规律' if sign['+'] >= 5 and sign['-'] <= 1 else \
              ('○ 不稳定' if sign['+'] >= 4 else '✗ 非规律')
    print(f'\n[反过拟合判定] {verdict}')
    print(f'  → {"Phase 2 启动 (月日卦校准)" if verdict.startswith("★") else "重新思考变爻框架"}')


if __name__ == '__main__':
    main()
