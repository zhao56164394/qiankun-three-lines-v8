# -*- coding: utf-8 -*-
"""重新生成 bagua_debug_baseline_snapshot.json（仅更新有 EXPERIMENT_SPECS 的卦）。

可作脚本直接运行，也可被 data_layer 更新流水线导入调用 main()。
"""
import copy, io, json, os, sys
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_layer.gua_data import GUA_ORDER, GUA_NAMES, compat_rename_columns


def main(verbose: bool = True) -> str:
    """重建 baseline 快照，返回快照文件路径。"""
    import experiment_gua as eg
    eg.load_runtime_context()

    snapshot_path = eg.BASELINE_SNAPSHOT_PATH

    if os.path.exists(snapshot_path):
        with open(snapshot_path, 'r', encoding='utf-8') as f:
            old_snapshot = json.load(f)
        old_payloads = old_snapshot.get('payloads', {})
    else:
        old_payloads = {}

    new_payloads = {}
    for gua in GUA_ORDER:
        if gua in eg.GUA_EXPERIMENT_SPECS:
            if verbose:
                print(f'Rebuilding {gua}({GUA_NAMES[gua]}) from naked_cfg...')
            spec = eg.get_spec(gua)
            naked_cfg = copy.deepcopy(spec['naked_cfg'])
            payload = eg.build_payload_for_cfg(gua, naked_cfg)
            signal_df = payload['target_sig'].copy()
            trade_df = eg.build_trade_detail(payload['result'], gua).copy()

            if len(signal_df) > 0:
                signal_df['gua'] = signal_df['tian_gua'].astype(str).str.zfill(3)
                signal_df['ren_gua'] = signal_df['ren_gua'].astype(str).str.zfill(3)
                signal_df['di_gua'] = signal_df['di_gua'].astype(str).str.zfill(3)
            if len(trade_df) > 0:
                trade_df['gua'] = trade_df['gua'].astype(str).str.zfill(3)
                compat_rename_columns(trade_df)
                trade_df['ren_gua'] = trade_df['ren_gua'].astype(str).str.zfill(3)
                trade_df['di_gua'] = trade_df['di_gua'].astype(str).str.zfill(3)

            signal_group = signal_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
                signal_count=('code', 'size'),
                signal_avg_ret=('actual_ret', 'mean'),
            ).reset_index() if len(signal_df) > 0 else pd.DataFrame(columns=['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret'])

            trade_group = trade_df.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
                buy_count=('code', 'size'),
                buy_avg_ret=('ret_pct', 'mean'),
            ).reset_index() if len(trade_df) > 0 else pd.DataFrame(columns=['ren_gua', 'di_gua', 'buy_count', 'buy_avg_ret'])

            base = pd.MultiIndex.from_product([GUA_ORDER, GUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
            matrix_df = base.merge(signal_group, on=['ren_gua', 'di_gua'], how='left')
            matrix_df = matrix_df.merge(trade_group, on=['ren_gua', 'di_gua'], how='left')
            matrix_df['signal_count'] = matrix_df['signal_count'].fillna(0).astype(int)
            matrix_df['buy_count'] = matrix_df['buy_count'].fillna(0).astype(int)
            matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
            matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
            matrix_df['ren_name'] = matrix_df['ren_gua'].map(GUA_NAMES)
            matrix_df['di_name'] = matrix_df['di_gua'].map(GUA_NAMES)

            new_payloads[gua] = {
                'target_gua': gua,
                'target_name': GUA_NAMES.get(gua, gua),
                'matrix_df': matrix_df.to_dict(orient='records'),
                'detail_signals': signal_df.to_dict(orient='records') if len(signal_df) > 0 else [],
                'detail_trades': trade_df.to_dict(orient='records') if len(trade_df) > 0 else [],
            }
            if verbose:
                print(f'  -> signals={len(signal_df)}, trades={len(trade_df)}')
        else:
            if verbose:
                print(f'Keeping old snapshot for {gua}({GUA_NAMES[gua]})')
            new_payloads[gua] = old_payloads.get(gua, {})

    snapshot = {"dataset": "baseline", "version": 2, "payloads": new_payloads}
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1, default=str)

    if verbose:
        print(f'\nSnapshot saved to {snapshot_path}')

    # 预热 dashboard 裸跑页"默认二次验证"的 test 模式缓存：
    # 若 GUA_STRATEGY 里某卦 pool_depth 非 None，UI 会默认勾选二次验证，走 test 模式。
    # 这里提前算好写 pkl。当前所有 pool_depth 都是 None，循环会跳过（no-op）。
    try:
        import backtest_8gua as b8
        for gua in GUA_ORDER:
            strat = b8.GUA_STRATEGY.get(gua, {})
            pool_depth = strat.get('pool_depth')
            if pool_depth is None:
                continue
            spec = eg.get_spec(gua)
            warm_cfg = copy.deepcopy(spec['naked_cfg'])
            warm_cfg['pool_depth'] = pool_depth
            if verbose:
                print(f'Warming test-mode cache for {gua}({GUA_NAMES[gua]}) pool_depth={pool_depth}...')
            eg.build_payload_for_cfg(gua, warm_cfg)
            if verbose:
                print(f'  -> OK')
    except Exception as e:
        if verbose:
            print(f'  !! test-mode 预热失败（不影响 snapshot）: {e}')

    # 清理旧 data_version 的 payload pkl (数据变更后老版本永不命中，纯浪费磁盘)
    try:
        import glob
        curr_ver = eg.data_version_stamp()
        cache_dir = eg.PAYLOAD_DISK_CACHE_DIR
        removed = 0
        for p in glob.glob(os.path.join(cache_dir, '*.pkl')):
            ver = os.path.basename(p).replace('.pkl', '').split('_')[-1]
            if ver != curr_ver:
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
        if verbose and removed > 0:
            print(f'清理陈旧 payload pkl: {removed} 个')
    except Exception as e:
        if verbose:
            print(f'  !! 清理陈旧 pkl 失败: {e}')

    return snapshot_path


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', closefd=False),
        encoding='utf-8', line_buffering=True)
    main(verbose=True)
