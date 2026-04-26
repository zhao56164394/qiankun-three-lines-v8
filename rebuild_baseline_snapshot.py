# -*- coding: utf-8 -*-
"""重建 bagua_debug_baseline_snapshot.json (dashboard 裸跑数据集).

v3 架构 (2026-04): 改为"综合裸跑结果分组"模式
  旧架构 v2: 每卦独立调 build_payload_for_cfg(naked_cfg), 无资金约束
  新架构 v3: 先 backtest_8gua_naked.py 跑一次综合回测 (8 卦共享 5 仓资金),
            再按 tian_gua 把 signal_detail / trade_log 分组生成 per-gua payload

数据源: data_layer/data/backtest_8gua_naked_result.json
       (来自 python backtest_8gua_naked.py)

可作脚本直接运行，也可被 data_layer 更新流水线导入调用 main()。
"""
import io, json, os, sys
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_layer.gua_data import GUA_ORDER, GUA_NAMES, compat_rename_columns


NAKED_RESULT_PATH = os.path.join(_ROOT, 'data_layer', 'data', 'backtest_8gua_naked_result.json')


def _empty_matrix():
    base = pd.MultiIndex.from_product([GUA_ORDER, GUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
    for col in ['signal_count', 'buy_count']:
        base[col] = 0
    for col in ['signal_avg_ret', 'buy_avg_ret']:
        base[col] = pd.NA
    base['ren_name'] = base['ren_gua'].map(GUA_NAMES)
    base['di_name'] = base['di_gua'].map(GUA_NAMES)
    return base


def _build_matrix(sig_gua: pd.DataFrame, trade_gua: pd.DataFrame) -> pd.DataFrame:
    """构建 ren_gua × di_gua 的 8×8 矩阵"""
    if len(sig_gua) > 0:
        signal_group = sig_gua.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
            signal_count=('code', 'size'),
            signal_avg_ret=('actual_ret', 'mean'),
        ).reset_index()
    else:
        signal_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'signal_count', 'signal_avg_ret'])

    if len(trade_gua) > 0:
        trade_group = trade_gua.groupby(['ren_gua', 'di_gua'], dropna=False).agg(
            buy_count=('code', 'size'),
            buy_avg_ret=('ret_pct', 'mean'),
        ).reset_index()
    else:
        trade_group = pd.DataFrame(columns=['ren_gua', 'di_gua', 'buy_count', 'buy_avg_ret'])

    base = pd.MultiIndex.from_product([GUA_ORDER, GUA_ORDER], names=['ren_gua', 'di_gua']).to_frame(index=False)
    matrix_df = base.merge(signal_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df = matrix_df.merge(trade_group, on=['ren_gua', 'di_gua'], how='left')
    matrix_df['signal_count'] = matrix_df['signal_count'].fillna(0).astype(int)
    matrix_df['buy_count'] = matrix_df['buy_count'].fillna(0).astype(int)
    matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
    matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
    matrix_df['ren_name'] = matrix_df['ren_gua'].map(GUA_NAMES)
    matrix_df['di_name'] = matrix_df['di_gua'].map(GUA_NAMES)
    return matrix_df


def _normalize_gua_cols(df: pd.DataFrame):
    for col in ['tian_gua', 'ren_gua', 'di_gua', 'gua']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(3)


def main(verbose: bool = True) -> str:
    """重建 baseline 快照，返回快照文件路径。"""
    import experiment_gua as eg
    snapshot_path = eg.BASELINE_SNAPSHOT_PATH

    if not os.path.exists(NAKED_RESULT_PATH):
        raise FileNotFoundError(
            f'裸跑综合回测结果不存在: {NAKED_RESULT_PATH}\n'
            f'请先运行: python backtest_8gua_naked.py'
        )

    if verbose:
        print(f'读取综合裸跑结果: {NAKED_RESULT_PATH}')
    with open(NAKED_RESULT_PATH, 'r', encoding='utf-8') as f:
        naked_result = json.load(f)

    meta = naked_result.get('meta', {})
    sig_all = pd.DataFrame(naked_result.get('signal_detail', []))
    trade_all = pd.DataFrame(naked_result.get('trade_log', []))
    compat_rename_columns(sig_all)
    compat_rename_columns(trade_all)
    _normalize_gua_cols(sig_all)
    _normalize_gua_cols(trade_all)

    if verbose:
        print(f'  信号 {len(sig_all):>5,} 条, 交易 {len(trade_all):>4,} 笔')
        print(f'  综合指标: 终值 {meta.get("final_capital", "?"):,} '
              f'收益 {meta.get("total_return", "?")}% '
              f'MDD {meta.get("max_dd", "?")}%')

    new_payloads = {}
    for gua in GUA_ORDER:
        sig_gua = sig_all[sig_all['tian_gua'] == gua].copy() if len(sig_all) else sig_all.copy()
        trade_gua = trade_all[trade_all.get('gua', pd.Series([], dtype=str)) == gua].copy() if len(trade_all) else trade_all.copy()

        matrix_df = _build_matrix(sig_gua, trade_gua)
        # 每卦 payload['meta'] 写入综合回测指标, dashboard summary 据此显示综合净值
        # (dashboard 逻辑: baseline 分支若读到 final_capital 直接用, 否则走累加近似)
        new_payloads[gua] = {
            'target_gua': gua,
            'target_name': GUA_NAMES.get(gua, gua),
            'meta': dict(meta) if meta else {},
            'matrix_df': matrix_df.to_dict(orient='records'),
            'detail_signals': sig_gua.to_dict(orient='records') if len(sig_gua) else [],
            'detail_trades': trade_gua.to_dict(orient='records') if len(trade_gua) else [],
        }
        if verbose:
            print(f'  {gua} {GUA_NAMES[gua]:<12} signals={len(sig_gua):>5,}  trades={len(trade_gua):>4,}')

    snapshot = {
        'dataset': 'baseline',
        'version': 3,
        'source': 'unified_backtest',
        'source_file': os.path.basename(NAKED_RESULT_PATH),
        'unified_meta': meta,
        'payloads': new_payloads,
    }
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1, default=str)

    if verbose:
        print(f'\nSnapshot saved to {snapshot_path}')

    return snapshot_path


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(
        open(sys.stdout.fileno(), 'wb', closefd=False),
        encoding='utf-8', line_buffering=True)
    main(verbose=True)
