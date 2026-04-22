# -*- coding: utf-8 -*-
"""重新生成 bagua_debug_baseline_snapshot.json（仅更新有 EXPERIMENT_SPECS 的卦）"""
import copy, io, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(
    open(sys.stdout.fileno(), 'wb', closefd=False),
    encoding='utf-8', line_buffering=True)

import experiment_gua as eg
eg.load_runtime_context()

SNAPSHOT_PATH = eg.BASELINE_SNAPSHOT_PATH
BAGUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']
GUA_NAMES = {
    '000': '坤', '001': '艮', '010': '坎', '011': '巽',
    '100': '震', '101': '离', '110': '兑', '111': '乾',
}

with open(SNAPSHOT_PATH, 'r', encoding='utf-8') as f:
    old_snapshot = json.load(f)
old_payloads = old_snapshot.get('payloads', {})

new_payloads = {}
for gua in BAGUA_ORDER:
    if gua in eg.GUA_EXPERIMENT_SPECS:
        print(f'Rebuilding {gua}({GUA_NAMES[gua]}) from naked_cfg...')
        spec = eg.get_spec(gua)
        naked_cfg = copy.deepcopy(spec['naked_cfg'])
        payload = eg.build_payload_for_cfg(gua, naked_cfg)
        signal_df = payload['target_sig'].copy()
        trade_df = eg.build_trade_detail(payload['result'], gua).copy()

        if len(signal_df) > 0:
            signal_df['gua'] = signal_df['zz_gua'].astype(str).str.zfill(3)
            signal_df['market_gua'] = signal_df['market_gua'].astype(str).str.zfill(3)
            signal_df['stock_gua'] = signal_df['stock_gua'].astype(str).str.zfill(3)
        if len(trade_df) > 0:
            trade_df['gua'] = trade_df['gua'].astype(str).str.zfill(3)
            trade_df['market_gua'] = trade_df['market_gua'].astype(str).str.zfill(3)
            trade_df['stock_gua'] = trade_df['stock_gua'].astype(str).str.zfill(3)

        signal_group = signal_df.groupby(['market_gua', 'stock_gua'], dropna=False).agg(
            signal_count=('code', 'size'),
            signal_avg_ret=('actual_ret', 'mean'),
        ).reset_index() if len(signal_df) > 0 else pd.DataFrame(columns=['market_gua', 'stock_gua', 'signal_count', 'signal_avg_ret'])

        trade_group = trade_df.groupby(['market_gua', 'stock_gua'], dropna=False).agg(
            buy_count=('code', 'size'),
            buy_avg_ret=('ret_pct', 'mean'),
        ).reset_index() if len(trade_df) > 0 else pd.DataFrame(columns=['market_gua', 'stock_gua', 'buy_count', 'buy_avg_ret'])

        base = pd.MultiIndex.from_product([BAGUA_ORDER, BAGUA_ORDER], names=['market_gua', 'stock_gua']).to_frame(index=False)
        matrix_df = base.merge(signal_group, on=['market_gua', 'stock_gua'], how='left')
        matrix_df = matrix_df.merge(trade_group, on=['market_gua', 'stock_gua'], how='left')
        matrix_df['signal_count'] = matrix_df['signal_count'].fillna(0).astype(int)
        matrix_df['buy_count'] = matrix_df['buy_count'].fillna(0).astype(int)
        matrix_df['signal_avg_ret'] = pd.to_numeric(matrix_df['signal_avg_ret'], errors='coerce')
        matrix_df['buy_avg_ret'] = pd.to_numeric(matrix_df['buy_avg_ret'], errors='coerce')
        matrix_df['market_name'] = matrix_df['market_gua'].map(GUA_NAMES)
        matrix_df['stock_name'] = matrix_df['stock_gua'].map(GUA_NAMES)

        new_payloads[gua] = {
            'target_gua': gua,
            'target_name': GUA_NAMES.get(gua, gua),
            'matrix_df': matrix_df.to_dict(orient='records'),
            'detail_signals': signal_df.to_dict(orient='records') if len(signal_df) > 0 else [],
            'detail_trades': trade_df.to_dict(orient='records') if len(trade_df) > 0 else [],
        }
        sig_count = len(signal_df)
        buy_count = len(trade_df)
        print(f'  -> signals={sig_count}, trades={buy_count}')
    else:
        print(f'Keeping old snapshot for {gua}({GUA_NAMES[gua]})')
        new_payloads[gua] = old_payloads.get(gua, {})

snapshot = {"dataset": "baseline", "version": 1, "payloads": new_payloads}
with open(SNAPSHOT_PATH, 'w', encoding='utf-8') as f:
    json.dump(snapshot, f, ensure_ascii=False, indent=1, default=str)

print(f'\nSnapshot saved to {SNAPSHOT_PATH}')
