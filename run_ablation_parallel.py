# -*- coding: utf-8 -*-
"""
run_ablation_parallel.py — 八卦消融实验并行执行器

按卦并行（每个 worker 跑一整个卦的全部 cfg）：
  - 同卦内部仍串行（实验之间共享 PAYLOAD_CACHE 收益大）
  - 8 卦切给 N 个 worker，理论提速 N×（受限于 worker 数）

用法:
  python run_ablation_parallel.py --all-gua --layer pool          # 8 卦的 pool 实验并行
  python run_ablation_parallel.py --all-gua --layer naked,pool,buy,sell  # 多 layer 串行 × 8 卦并行
  python run_ablation_parallel.py --gua 000,001,010 --layer pool  # 指定卦集
  python run_ablation_parallel.py --workers 2 ...                 # 自定义进程数

注意:
  - 子进程启动用 spawn (Windows 必须), 每个 worker 独立加载数据 (~50s 冷启动)
  - 磁盘缓存 (data_layer/data/_payload_disk_cache/) 跨 worker 共享, 第二次跑同 cfg 秒级
  - 内存峰值 ≈ workers × 5GB, 4 worker 大约吃 20GB
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


VALID_LAYERS = ['naked', 'market', 'stock', 'market_stock', 'market_stock_matrix',
                'pool', 'buy', 'sell']
ALL_GUA = ['000', '001', '010', '011', '100', '101', '110', '111']


def _worker_run_layers(gua: str, layers: List[str]) -> Tuple[str, str, float, str]:
    """子进程入口：跑一个卦的指定 layer 列表。

    返回: (gua, status, elapsed, error_msg)
    """
    t0 = time.time()
    try:
        # 每个 worker 独立 import (spawn 方式)
        import experiment_gua as eg

        for layer in layers:
            fn_name = f'run_{layer}'
            fn = getattr(eg, fn_name, None)
            if fn is None:
                return (gua, 'ERROR', time.time() - t0, f'experiment_gua.{fn_name} not found')
            fn(gua)

        return (gua, 'OK', time.time() - t0, '')
    except Exception as e:
        tb = traceback.format_exc()
        return (gua, 'ERROR', time.time() - t0, f'{type(e).__name__}: {e}\n{tb}')


def run_parallel(guas: List[str], layers: List[str], workers: int) -> None:
    print('=' * 60)
    print(f'  消融实验并行执行: {len(guas)} 卦 × {len(layers)} layer  ({workers} workers)')
    print(f'  卦: {guas}')
    print(f'  layer: {layers}')
    print('=' * 60)

    t0 = time.time()
    tasks = [(gua, layers) for gua in guas]

    # spawn 上下文: Windows 必须, Linux 也建议 (避免 fork 复制大对象 + 锁状态)
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers) as pool:
        results = []
        # imap_unordered: 完成一个就返回, 让 print 进度更实时
        for r in pool.imap_unordered(_worker_run_layers_unpack, tasks):
            gua, status, elapsed, err = r
            results.append(r)
            if status == 'OK':
                print(f'  [DONE] {gua}: {elapsed:.0f}s')
            else:
                print(f'  [FAIL] {gua}: {elapsed:.0f}s — {err}')

    total = time.time() - t0
    n_ok = sum(1 for r in results if r[1] == 'OK')
    n_fail = len(results) - n_ok

    print()
    print('=' * 60)
    print(f'  完成: {n_ok}/{len(results)} 卦成功, {n_fail} 失败')
    print(f'  总耗时: {total:.0f}s ({total/60:.1f} min)')
    if results:
        avg_per = sum(r[2] for r in results) / len(results)
        # 串行估算 = sum(elapsed); 并行实际 = total
        ser = sum(r[2] for r in results)
        if total > 0:
            print(f'  串行预估: {ser:.0f}s ({ser/60:.1f} min)  →  并行实际: {total:.0f}s  →  提速 {ser/total:.1f}x')
    print('=' * 60)


def _worker_run_layers_unpack(args):
    return _worker_run_layers(*args)


def main():
    parser = argparse.ArgumentParser(
        description='八卦消融实验并行执行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--all-gua', action='store_true',
                        help='跑全部 8 卦')
    parser.add_argument('--gua', type=str, default='',
                        help='指定卦集合 (逗号分隔, 如: 000,001,010)')
    parser.add_argument('--layer', type=str, required=True,
                        help=f'实验 layer (逗号分隔). 可选: {",".join(VALID_LAYERS)}')
    parser.add_argument('--workers', type=int, default=4,
                        help='并行 worker 数 (默认 4)')
    args = parser.parse_args()

    # 解析卦集合
    if args.all_gua:
        guas = list(ALL_GUA)
    elif args.gua:
        guas = [g.strip() for g in args.gua.split(',') if g.strip()]
        invalid = [g for g in guas if g not in ALL_GUA]
        if invalid:
            parser.error(f'未知卦码: {invalid}. 合法值: {ALL_GUA}')
    else:
        parser.error('必须指定 --all-gua 或 --gua')

    # 解析 layer
    layers = [l.strip() for l in args.layer.split(',') if l.strip()]
    invalid = [l for l in layers if l not in VALID_LAYERS]
    if invalid:
        parser.error(f'未知 layer: {invalid}. 合法值: {VALID_LAYERS}')

    workers = max(1, min(args.workers, len(guas)))

    run_parallel(guas, layers, workers)


if __name__ == '__main__':
    main()
