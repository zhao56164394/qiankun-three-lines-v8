# -*- coding: utf-8 -*-
"""八卦分治 · 综合裸跑回测

现有两套裸跑机制:
  1. experiment_gua.derive_naked_cfg()  — 单卦裸跑配置 (过滤全关 + sell=bear + pool_depth=None)
  2. rebuild_baseline_snapshot.py       — 每卦独立跑 payload (无综合净值, 无资金约束)

本脚本的定位: 用 (1) 的 naked_cfg 驱动 backtest_8gua.run() 全量综合回测
 → 产出 2014-2026 综合净值曲线 + 5 仓资金约束下的实际交易流水
 → 结果与 formal 版 (+346.5%) 直接对比 "分治 + 触发" 本身的 alpha

输出: data_layer/data/backtest_8gua_naked_result.json
     (formal 的 backtest_8gua_result.json 保持不动)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_8gua as b8
import experiment_gua as eg


RESULT_DIR = os.path.join(os.path.dirname(__file__), 'data_layer', 'data')
FORMAL_RESULT_PATH = os.path.join(RESULT_DIR, 'backtest_8gua_result.json')
NAKED_RESULT_PATH = os.path.join(RESULT_DIR, 'backtest_8gua_naked_result.json')
FORMAL_BACKUP_PATH = FORMAL_RESULT_PATH + '.formal_backup'


def patch_strategy_to_naked():
    """把 b8.GUA_STRATEGY 8 卦全部替换成 naked_cfg"""
    print('\n[PATCH] GUA_STRATEGY 切换为裸跑配置')
    for gua in sorted(b8.GUA_STRATEGY.keys()):
        b8.GUA_STRATEGY[gua] = eg.derive_naked_cfg(gua)
    print('  过滤全关 (di_gua 白名单 / ren_gua 黑名单 → 空)')
    print('  pool_depth → None (无二次验证)')
    print('  sell → bear (所有卦统一保守卖法)')
    print('  保留: 入池阈值 (-250), 触发模式 (双升/上穿 N), 分治架构')


def patch_pool_threshold():
    """支持 env var POOL_THRESHOLD 放宽 / 收紧全局入池阈值 (默认 -250)"""
    override = os.environ.get('POOL_THRESHOLD')
    if override is not None:
        new_thr = int(override)
        old_thr = b8.UNIFIED_POOL_THRESHOLD
        b8.UNIFIED_POOL_THRESHOLD = new_thr
        print(f'[PATCH pool] UNIFIED_POOL_THRESHOLD: {old_thr} → {new_thr}')


def patch_disable_gate():
    """env DISABLE_GATE=1 → 清空所有 gate_disable_* 字段, 用于无 gate baseline 标定"""
    if os.environ.get('DISABLE_GATE') in ('1', 'true', 'True'):
        cleared = 0
        for gua, cfg in b8.GUA_STRATEGY.items():
            for k in ('gate_disable_y_gua', 'gate_disable_m_gua', 'gate_disable_ym'):
                if k in cfg and cfg[k]:
                    cfg[k] = set() if 'ym' not in k else set()
                    cleared += 1
        print(f'[PATCH gate] DISABLE_GATE=1 → 清空 {cleared} 个 gate 字段 (用于 baseline 重标定)')


def patch_apply_ablation():
    """env ABLATION_PATCH_PATH=xxx.json → 在 GUA_STRATEGY 上应用 patch
    patch 文件格式: {gua: {field: value | {'__set__':True, 'items':[...]}, ...}, ...}
    set 中元素若是 list (长度2) 自动转 tuple, 用于 gate_disable_ym 这种 set of tuples
    """
    import json as _json
    path = os.environ.get('ABLATION_PATCH_PATH')
    if not path or not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        patches = _json.load(f)
    print(f'[PATCH ablation] 应用 {path}')
    for gua, fields in patches.items():
        if gua not in b8.GUA_STRATEGY:
            print(f'  WARN: gua {gua} 不存在')
            continue
        for k, v in fields.items():
            if isinstance(v, dict) and v.get('__set__'):
                items = v['items']
                # tuple 化 list-of-list
                converted = set()
                for it in items:
                    if isinstance(it, list) and len(it) == 2 and all(isinstance(x, str) for x in it):
                        converted.add(tuple(it))
                    elif isinstance(it, list):
                        converted.add(tuple(it))
                    else:
                        converted.add(it)
                b8.GUA_STRATEGY[gua][k] = converted
                print(f'  {gua}.{k} = {converted}')
            else:
                b8.GUA_STRATEGY[gua][k] = v
                print(f'  {gua}.{k} = {v}')


def patch_disable_filters():
    """env DISABLE_FILTERS=1 → 清空所有过滤白/黑名单 (di/ren), 用于"无过滤" 测试.
    用于阶段 3 (di/ren 过滤) 的 baseline."""
    if os.environ.get('DISABLE_FILTERS') in ('1', 'true', 'True'):
        cleared = 0
        for gua, cfg in b8.GUA_STRATEGY.items():
            for k in list(cfg.keys()):
                if (k.endswith('_exclude_ren_gua') or k.endswith('_allow_di_gua')
                        or k.endswith('_exclude_di_gua') or k.endswith('_market_stock_whitelist')):
                    if cfg[k]:
                        cfg[k] = set() if 'allow' not in k or k.endswith('_market_stock_whitelist') else None
                        # allow_di_gua 设 None 表示不限
                        if k.endswith('_allow_di_gua') or k.endswith('_market_stock_whitelist'):
                            cfg[k] = None
                        else:
                            cfg[k] = set()
                        cleared += 1
        print(f'[PATCH filters] DISABLE_FILTERS=1 → 清空 {cleared} 个 di/ren 过滤字段')


def main():
    # 1. 备份当前 result.json (formal 内容), 避免被裸跑覆盖丢失
    if os.path.exists(FORMAL_RESULT_PATH):
        os.replace(FORMAL_RESULT_PATH, FORMAL_BACKUP_PATH)
        print(f'[备份] formal result → {os.path.basename(FORMAL_BACKUP_PATH)}')

    # 自定义结果输出路径 (用于消融实验)
    out_path = os.environ.get('ABLATION_RESULT_PATH') or NAKED_RESULT_PATH

    try:
        # 2. 裸跑
        patch_strategy_to_naked()
        patch_pool_threshold()
        patch_disable_gate()
        patch_disable_filters()
        patch_apply_ablation()  # 最后应用 ablation patch, 覆盖前面的清空
        # 支持 env 自定义时间区间 (用于 OOS / IS 切片)
        ys = os.environ.get('BACKTEST_START')
        ye = os.environ.get('BACKTEST_END')
        if ys or ye:
            print(f'[PATCH 时间区间] {ys or "默认"} ~ {ye or "默认"}')
        result, stats = b8.run(start_date=ys, end_date=ye)

        # 3. 刚生成的 result.json 是裸跑内容, 改名为 naked_result.json (或 ablation 自定义路径)
        if os.path.exists(FORMAL_RESULT_PATH):
            os.replace(FORMAL_RESULT_PATH, out_path)
            print(f'\n[SAVE] 裸跑结果 → {out_path}')
    finally:
        # 4. 恢复 formal 备份 (无论成败)
        if os.path.exists(FORMAL_BACKUP_PATH):
            os.replace(FORMAL_BACKUP_PATH, FORMAL_RESULT_PATH)
            print(f'[还原] formal result ← {os.path.basename(FORMAL_BACKUP_PATH)}')


if __name__ == '__main__':
    main()
