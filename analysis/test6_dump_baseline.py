# -*- coding: utf-8 -*-
"""单独跑 test6 baseline IS, 落地 sig + trade_log 给后续 y_gua 桶分析用"""
import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ['STRATEGY_VERSION'] = 'test6'

# 先 import (它会做自己的 stdout 包装), 再做 print flush
sys.path.insert(0, os.path.dirname(__file__))
from test6_pool_depth_ablation import setup_data, run_case
from strategy_configs import get_strategy

import functools
print = functools.partial(print, flush=True)

OUT_DIR = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print('=== 加载数据 ===')
    data = setup_data()

    cfg = get_strategy()
    print('\n=== 跑 baseline IS ===')
    t0 = time.time()
    sig, trd, stat = run_case(data, cfg, '20140101', '20221231', init_capital=200000)
    print(f'  IS: 终值 {stat["final_capital"]/10000:.1f}万, sig {len(sig)}, trd {len(trd)}, '
          f'耗时 {time.time()-t0:.1f}s')

    out_path = os.path.join(OUT_DIR, 'baseline_IS.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {'strategy_version': 'test6', 'period': '20140101-20221231',
                     'final_capital': stat['final_capital']},
            'signal_detail': sig.to_dict('records'),
            'trade_log': trd,
        }, f, ensure_ascii=False, default=str)
    print(f'  落地: {out_path}')

    print('\n=== 跑 baseline OOS ===')
    t0 = time.time()
    sig, trd, stat = run_case(data, get_strategy(), '20230101', '20260417', init_capital=200000)
    print(f'  OOS: 终值 {stat["final_capital"]/10000:.1f}万, sig {len(sig)}, trd {len(trd)}, '
          f'耗时 {time.time()-t0:.1f}s')

    out_path = os.path.join(OUT_DIR, 'baseline_OOS.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {'strategy_version': 'test6', 'period': '20230101-20260417',
                     'final_capital': stat['final_capital']},
            'signal_detail': sig.to_dict('records'),
            'trade_log': trd,
        }, f, ensure_ascii=False, default=str)
    print(f'  落地: {out_path}')


if __name__ == '__main__':
    main()
