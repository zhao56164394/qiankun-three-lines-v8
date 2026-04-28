# -*- coding: utf-8 -*-
"""快速验证 test12yaomin change_type skip 是否生效 (跑 w3 2020 单段)"""
import os
import sys
import functools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from test6_pool_depth_ablation import setup_data, run_case
import strategy_configs

print = functools.partial(print, flush=True)


def main():
    print('=== 加载数据 ===')
    data = setup_data()

    # baseline + yaomin 都跑 w3 2020
    for ver in ['test6', 'test12yaomin']:
        os.environ['STRATEGY_VERSION'] = ver
        cfg = strategy_configs.get_strategy()
        if ver == 'test12yaomin':
            print(f'\n[cfg] 000 change_type_skip = {cfg["000"].get("change_type_skip")}')
        print(f'\n=== {ver} w3_2020 ===')
        sig, trd, stat = run_case(data, cfg, '20200101', '20210101', 200000)
        print(f'>>> {ver}: sig={len(sig)}, trd={len(trd)}, '
              f'终值={stat["final_capital"]/10000:.1f}万')


if __name__ == '__main__':
    main()
