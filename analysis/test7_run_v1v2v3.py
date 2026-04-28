# -*- coding: utf-8 -*-
"""跑 test7 v1/v2/v3 三套 cfg 的 IS + OOS, 与 test6 baseline 对比"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 不预设环境变量, 在循环里切换
sys.path.insert(0, os.path.dirname(__file__))
from test6_pool_depth_ablation import setup_data, run_case
from strategy_configs import get_strategy

import functools
print = functools.partial(print, flush=True)

OUT_DIR = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test7_pool_depth_landed')
os.makedirs(OUT_DIR, exist_ok=True)

PHASES = [('IS', '20140101', '20221231'),
          ('OOS', '20230101', '20260417')]

VERSIONS = ['test6', 'test7v1', 'test7v2', 'test7v3']


def main():
    print('=== 加载数据 ===')
    data = setup_data()

    summary = []
    for ver in VERSIONS:
        os.environ['STRATEGY_VERSION'] = ver
        cfg = get_strategy()
        for phase, ys, ye in PHASES:
            t0 = time.time()
            sig, trd, stat = run_case(data, cfg, ys, ye, init_capital=200000)
            elapsed = time.time() - t0
            print(f'\n[{ver}] {phase} ({ys}~{ye}): '
                  f'终值 {stat["final_capital"]/10000:.1f}万, '
                  f'sig {len(sig)}, trd {len(trd)}, {elapsed:.1f}s')

            # 按 y_gua 分组统计
            sig_with_y = sig.copy()
            sig_with_y['y_gua'] = sig_with_y['signal_date'].astype(str).map(
                lambda d: data['gate_map'].get(d, ('???','???'))[1]
            )
            for tg in ['000','001','010','011','100','101','110','111']:
                sig_y = sig_with_y[sig_with_y['y_gua']==tg]
                trd_y = [t for t in trd if t.get('y_gua')==tg]
                summary.append({
                    'version': ver, 'phase': phase, 'y_gua': tg,
                    'sig_n': len(sig_y),
                    'sig_mean%': sig_y['actual_ret'].mean() if len(sig_y) else None,
                    'trd_n': len(trd_y),
                    'trd_利万': sum(t['profit'] for t in trd_y) / 10000,
                    'final_cap_wan': stat['final_capital']/10000,
                })

            # 落地 sig+trd
            out_path = os.path.join(OUT_DIR, f'{ver}_{phase}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'meta': {'version': ver, 'phase': phase, 'period': f'{ys}-{ye}',
                             'final_capital': stat['final_capital']},
                    'signal_detail': sig.to_dict('records'),
                    'trade_log': trd,
                }, f, ensure_ascii=False, default=str)
            print(f'  落地: {out_path}')

    # 写 summary csv
    import pandas as pd
    sdf = pd.DataFrame(summary)
    out_csv = os.path.join(OUT_DIR, 'summary.csv')
    sdf.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'\n落地 summary: {out_csv}')

    # 横向对比
    NAMES = {'000':'坤','001':'艮','010':'坎','011':'巽','100':'震','101':'离','110':'兑','111':'乾'}
    print('\n' + '=' * 110)
    print('# IS / OOS 终值对比 (test6=baseline, v1/v2/v3 三个 cfg)')
    print('=' * 110)
    for phase in ['IS', 'OOS']:
        print(f'\n[{phase}]')
        for ver in VERSIONS:
            row = sdf[(sdf['version']==ver) & (sdf['phase']==phase)].iloc[0]
            print(f'  {ver:<10}: 终值 {row["final_cap_wan"]:>5.1f}万  '
                  f'(总 sig 由各桶汇总, trd 由 simulate)')

    print('\n' + '=' * 110)
    print('# 各桶 trd 利润对比 (单位: 万)')
    print('=' * 110)
    for phase in ['IS', 'OOS']:
        print(f'\n[{phase}]')
        print(f'  {"y_gua":<10} ' + ' '.join(f'{v:>9}' for v in VERSIONS))
        print('  ' + '-' * 50)
        for tg in ['000','001','010','011','100','101','110','111']:
            row = [f'  {tg} {NAMES[tg]:<5}']
            for ver in VERSIONS:
                t = sdf[(sdf['version']==ver) & (sdf['phase']==phase) & (sdf['y_gua']==tg)]
                if len(t)==0:
                    row.append(f'{"-":>9}')
                else:
                    row.append(f'{t.iloc[0]["trd_利万"]:>+8.1f}')
            print(' '.join(row))

    print('\n' + '=' * 110)
    print('# 各桶 sig_n 对比 (反映过滤强度)')
    print('=' * 110)
    for phase in ['IS', 'OOS']:
        print(f'\n[{phase}]')
        print(f'  {"y_gua":<10} ' + ' '.join(f'{v:>9}' for v in VERSIONS))
        print('  ' + '-' * 50)
        for tg in ['000','001','010','011','100','101','110','111']:
            row = [f'  {tg} {NAMES[tg]:<5}']
            for ver in VERSIONS:
                t = sdf[(sdf['version']==ver) & (sdf['phase']==phase) & (sdf['y_gua']==tg)]
                if len(t)==0:
                    row.append(f'{"-":>9}')
                else:
                    row.append(f'{int(t.iloc[0]["sig_n"]):>9d}')
            print(' '.join(row))


if __name__ == '__main__':
    main()
