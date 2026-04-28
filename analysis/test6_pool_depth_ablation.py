# -*- coding: utf-8 -*-
"""Phase 3 池深 ablation runner (test6 真裸基线之上)

按 strategy-ablation skill 范式:
  - 全量 sig: scan_signals 输出 (不受 max_pos 限制)
  - 实买 trd: simulate 后实际成交
  - 综合判定: 见 SKILL 矩阵

Ablation 设计:
  对每个 y_gua 桶 (000-111), 单独改其 pool_depth_tiers, 其他 7 桶保持 baseline (无 tiers).
  候选档位 (3 种, baseline 不重复):
    - A. 排极深 ≤-400 (depth_max=-400, days_min=99999)
    - B. 排深   ≤-300 (depth_max=-300, days_min=99999)
    - C. 排中浅 ≤-200 (depth_max=-200, days_min=99999)

跑 8 桶 × 3 候选 = 24 case (× IS+OOS = 48 跑).

输出:
  data_layer/data/ablation/test6_pool_depth/
    case_summary.csv: 每个 case 的 (桶, 改动, sig_n, sig_mean, ci, trd_n, trd_利万, verdict, IS/OOS)
    {case_id}.json:   每个 case 的 sig_log + trade_log
"""
import os
import sys
import json
import time
import copy
import functools
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 强制 STRATEGY_VERSION = test6
os.environ['STRATEGY_VERSION'] = 'test6'

# 不要直接 import backtest_y_gua, 它会在加载时执行 _get_strategy() 一次
# 我们需要在每个 case 之前修改 GUA_STRATEGY
import backtest_y_gua as bt
from backtest_capital import (load_zz1000, load_zz1000_full, load_stocks)
from strategy_configs import get_strategy as _get_strategy, get_sim_params

# 强制每次 print 刷新, 否则 pipe 模式下看不到进度
print = functools.partial(print, flush=True)

OUT_DIR = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test6_pool_depth')
os.makedirs(OUT_DIR, exist_ok=True)

GUA_NAMES_SHORT = {'000':'坤','001':'艮','010':'坎','011':'巽',
                   '100':'震','101':'离','110':'兑','111':'乾'}

# Ablation 候选档位定义
CANDIDATES = [
    ('A_d400', '排极深 ≤-400', [{'depth_max': -400, 'days_min': 99999, 'days_max': None},
                                {'depth_max': None, 'days_min': 0, 'days_max': None}]),
    ('B_d300', '排深 ≤-300',   [{'depth_max': -300, 'days_min': 99999, 'days_max': None},
                                {'depth_max': None, 'days_min': 0, 'days_max': None}]),
    ('C_d200', '排中浅 ≤-200', [{'depth_max': -200, 'days_min': 99999, 'days_max': None},
                                {'depth_max': None, 'days_min': 0, 'days_max': None}]),
]

PHASES = [('IS', '20140101', '20221231'), ('OOS', '20230101', '20260417')]


def bootstrap_ci(arr, n_boot=1000, ci=95):
    if len(arr) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    n = len(arr)
    boots = np.empty(n_boot)
    a = np.asarray(arr)
    for i in range(n_boot):
        boots[i] = a[rng.integers(0, n, n)].mean()
    return float(np.percentile(boots, (100-ci)/2)), float(np.percentile(boots, 100-(100-ci)/2))


def verdict(sig_n, ci_lo, ci_hi, trd_n, trd_li_wan):
    if sig_n < 10:
        return '— 不下结论'
    if sig_n < 20:
        return '○ 灰(样本<20)'
    if not np.isnan(ci_hi) and ci_hi < 0:
        if trd_n >= 5 and trd_li_wan < 0:
            return '✗ 真有害'
        return '○ sig负但trd未兑'
    if not np.isnan(ci_lo) and ci_lo > 0:
        return '★ 真有益'
    if trd_n >= 8 and trd_li_wan > 10:
        return '★ 实战有益'
    if trd_n < 5:
        return '○ 灰(trd样本少)'
    return '○ 中性'


def setup_data():
    """一次性加载所有数据 (跨 case 复用)"""
    print('=' * 90)
    print('[INIT] 加载数据 (一次性)')
    print('=' * 90)
    t0 = time.time()
    big_cycle_context = bt.load_big_cycle_context()
    stock_bagua_map = bt.load_stock_bagua_map()
    zz_df = load_zz1000_full()
    zz1000 = load_zz1000()
    stock_data = load_stocks()
    stk_mf_map = bt._load_stock_main_force()

    # tian_gua_map / gate_map / daily_bagua_map / stock_gate_map
    DATA_DIR = bt.DATA_DIR
    ms_pq = os.path.join(DATA_DIR, 'foundation', 'multi_scale_gua_daily.parquet')
    ms_df = pd.read_parquet(ms_pq) if os.path.exists(ms_pq) else \
            pd.read_csv(os.path.join(DATA_DIR, 'foundation', 'multi_scale_gua_daily.csv'),
                        dtype={'d_gua': str, 'm_gua': str, 'y_gua': str})
    ms_df['date'] = ms_df['date'].astype(str)
    _clean = bt._clean_gua
    d_arr = ms_df['d_gua'].astype(str).map(_clean).to_numpy()
    m_arr = ms_df['m_gua'].astype(str).map(_clean).to_numpy()
    y_arr = ms_df['y_gua'].astype(str).map(_clean).to_numpy()
    dates_arr = ms_df['date'].to_numpy()
    tian_gua_map = {dates_arr[i]: (d_arr[i], '') for i in range(len(ms_df))}
    gate_map = {dates_arr[i]: (m_arr[i], y_arr[i]) for i in range(len(ms_df))}

    daily_bagua_map = None  # 此 ablation 不需要 daily_bagua / stock_gate
    stock_gate_map = None
    daily_bagua_map_path = os.path.join(DATA_DIR, 'foundation', 'daily_bagua_features.parquet')
    if os.path.exists(daily_bagua_map_path):
        try:
            db = pd.read_parquet(daily_bagua_map_path, columns=['date', 'd_gua', 'm_gua'])
            db['date'] = db['date'].astype(str)
            daily_bagua_map = {r['date']: (str(r['d_gua']), str(r['m_gua'])) for _, r in db.iterrows()}
        except Exception:
            pass

    smg_path = os.path.join(DATA_DIR, 'foundation', 'stock_multi_scale_gua_daily.parquet')
    if os.path.exists(smg_path):
        smg = pd.read_parquet(smg_path, columns=['date', 'code', 'm_gua', 'y_gua'])
        smg['date'] = smg['date'].astype(str)
        smg['code'] = smg['code'].astype(str).str.zfill(6)
        smg['m_gua'] = smg['m_gua'].astype(str).map(_clean)
        smg['y_gua'] = smg['y_gua'].astype(str).map(_clean)
        stock_gate_map = {(r['date'], r['code']): (r['m_gua'], r['y_gua']) for _, r in smg.iterrows()}

    print(f'  耗时: {time.time()-t0:.1f}s')
    return dict(
        zz_df=zz_df, zz1000=zz1000, stock_data=stock_data,
        stk_mf_map=stk_mf_map, big_cycle_context=big_cycle_context,
        stock_bagua_map=stock_bagua_map, daily_bagua_map=daily_bagua_map,
        gate_map=gate_map, stock_gate_map=stock_gate_map,
        tian_gua_map=tian_gua_map,
    )


def _norm_date(d):
    """'YYYYMMDD' → 'YYYY-MM-DD'. signal_date 是 'YYYY-MM-DD' 字符串,
    跟 'YYYYMMDD' 字符串直接 < 比较会因 '-' (45) < '0' (48) 错位 1 年."""
    if isinstance(d, str) and len(d) == 8 and '-' not in d:
        return f'{d[:4]}-{d[4:6]}-{d[6:]}'
    return d


def run_case(data, cfg, year_start, year_end, init_capital):
    """跑单个 case, 返回 (sig_df, trade_log, stats)"""
    # 注入 cfg 到 backtest_y_gua 模块
    bt.GUA_STRATEGY = cfg
    sim_params = get_sim_params()

    sig = bt.scan_signals_8gua(
        data['stock_data'], data['zz1000'], data['tian_gua_map'],
        stk_mf_map=data['stk_mf_map'],
        big_cycle_context=data['big_cycle_context'],
        stock_bagua_map=data['stock_bagua_map'],
        daily_bagua_map=data['daily_bagua_map'],
        gate_map=data['gate_map'],
        stock_gate_map=data['stock_gate_map'],
    )
    ys = _norm_date(year_start)
    ye = _norm_date(year_end)
    sig = sig[(sig['signal_date'] >= ys) &
              (sig['signal_date'] < ye)].reset_index(drop=True)

    result = bt.simulate_8gua(
        sig, data['zz_df'],
        max_pos=sim_params['max_pos'],
        daily_limit=sim_params['daily_limit'],
        init_capital=init_capital,
        tian_gua_map_ext=data['tian_gua_map'],
    )
    return sig, result['trade_log'], {
        'final_capital': result['final_capital'],
        'init_capital': result['init_capital'],
        'n_sig': len(sig),
        'n_trd': len(result['trade_log']),
    }


def collect_case_metrics(sig, trade_log, gate_map):
    """按 y_gua 桶分组算 sig/trd 指标"""
    rows = []
    sig = sig.copy()
    sig['y_gua'] = sig['signal_date'].map(
        lambda d: gate_map.get(d, ('???', '???'))[1]
    )
    for y in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sig_y = sig[sig['y_gua'] == y]
        trd_y = [t for t in trade_log if t.get('y_gua') == y]
        sig_n = len(sig_y)
        if sig_n == 0:
            rows.append({'y_gua': y, 'sig_n': 0, 'sig_mean%': np.nan,
                        'ci_lo': np.nan, 'ci_hi': np.nan,
                        'trd_n': len(trd_y),
                        'trd_利万': sum(t['profit'] for t in trd_y) / 10000,
                        'trd_mean%': np.nan, 'verdict': '— 无信号'})
            continue
        sig_ret = sig_y['actual_ret'].values
        ci_lo, ci_hi = bootstrap_ci(sig_ret)
        trd_li_wan = sum(t['profit'] for t in trd_y) / 10000
        rows.append({
            'y_gua': y,
            'sig_n': sig_n,
            'sig_mean%': float(np.mean(sig_ret)),
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'trd_n': len(trd_y),
            'trd_利万': trd_li_wan,
            'trd_mean%': float(np.mean([t['ret_pct'] for t in trd_y])) if trd_y else np.nan,
            'verdict': verdict(sig_n, ci_lo, ci_hi, len(trd_y), trd_li_wan),
        })
    return rows


if __name__ == '__main__':
    print('=' * 90)
    print('Phase 3 池深 ablation runner (test6 裸基线之上)')
    print('=' * 90)

    # 1. 加载数据
    data = setup_data()

    # 2. baseline (无 tiers)
    base_cfg = _get_strategy()  # test6 当前

    all_results = []  # (case_id, phase, y_gua, ...)

    # 跑 baseline 一次
    for phase, ys, ye in PHASES:
        print(f'\n[BASELINE] {phase} ({ys}~{ye})')
        t0 = time.time()
        sig, trd, stat = run_case(data, copy.deepcopy(base_cfg), ys, ye,
                                  init_capital=200000 if phase=='IS' else 200000)
        print(f'  终值 {stat["final_capital"]/10000:.1f}万, sig {stat["n_sig"]}, trd {stat["n_trd"]}, '
              f'耗时 {time.time()-t0:.1f}s')
        rows = collect_case_metrics(sig, trd, data['gate_map'])
        for r in rows:
            r['case_id'] = 'baseline'
            r['target_y_gua'] = '-'
            r['cand'] = '-'
            r['cand_desc'] = 'no tiers'
            r['phase'] = phase
            r['final_capital_wan'] = stat['final_capital'] / 10000
            r['n_sig_total'] = stat['n_sig']
            r['n_trd_total'] = stat['n_trd']
            all_results.append(r)

    # 3. 8 桶 × 3 候选 = 24 case
    case_idx = 0
    total_cases = 8 * len(CANDIDATES) * len(PHASES)
    print(f'\n[ABLATION] {total_cases} 个 case 即将开始')

    for target_y_gua in ['000', '001', '010', '011', '100', '101', '110', '111']:
        for cand_id, cand_desc, tiers in CANDIDATES:
            for phase, ys, ye in PHASES:
                case_idx += 1
                cfg = copy.deepcopy(base_cfg)
                cfg[target_y_gua]['pool_depth_tiers'] = tiers
                # 不加 only_y_gua, 这样 tier 永远生效

                t0 = time.time()
                sig, trd, stat = run_case(data, cfg, ys, ye, init_capital=200000)
                elapsed = time.time() - t0
                print(f'[{case_idx}/{total_cases}] y_gua={target_y_gua} {GUA_NAMES_SHORT[target_y_gua]} '
                      f'{cand_id} ({cand_desc}) {phase}: '
                      f'终值 {stat["final_capital"]/10000:.1f}万, sig {stat["n_sig"]}, '
                      f'trd {stat["n_trd"]}, {elapsed:.1f}s')

                rows = collect_case_metrics(sig, trd, data['gate_map'])
                for r in rows:
                    r['case_id'] = f'{target_y_gua}_{cand_id}'
                    r['target_y_gua'] = target_y_gua
                    r['cand'] = cand_id
                    r['cand_desc'] = cand_desc
                    r['phase'] = phase
                    r['final_capital_wan'] = stat['final_capital'] / 10000
                    r['n_sig_total'] = stat['n_sig']
                    r['n_trd_total'] = stat['n_trd']
                    all_results.append(r)

    # 4. 落地
    df = pd.DataFrame(all_results)
    out_csv = os.path.join(OUT_DIR, 'case_summary.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig', float_format='%.2f')
    print(f'\n落地: {out_csv} ({len(df)} 行)')

    # 5. 汇总: 仅看"target_y_gua=被改桶"那行 + 终值变化
    print('\n' + '=' * 100)
    print('Ablation 汇总 (仅看被改桶在自己 cfg 改动下的反应 + 终值)')
    print('=' * 100)
    for phase in ['IS', 'OOS']:
        print(f'\n[{phase}]')
        sub = df[df['phase'] == phase]
        baseline_cap = sub[sub['case_id'] == 'baseline']['final_capital_wan'].iloc[0]
        print(f'  baseline 终值: {baseline_cap:.1f} 万')
        # 对每个被改桶, 列其 3 个候选下: 终值变化 + 该桶 sig/trd 指标
        for tg in ['000', '001', '010', '011', '100', '101', '110', '111']:
            for cand_id, cand_desc, _ in CANDIDATES:
                cid = f'{tg}_{cand_id}'
                cs = sub[(sub['case_id'] == cid) & (sub['y_gua'] == tg)]
                if len(cs) == 0:
                    continue
                row = cs.iloc[0]
                cap = row['final_capital_wan']
                d_cap = cap - baseline_cap
                print(f'  {tg} {GUA_NAMES_SHORT[tg]:<2} {cand_id:<8} ({cand_desc:<12}): '
                      f'终值 {cap:.1f}万 ({d_cap:+.1f}万), '
                      f'桶内 sig={int(row["sig_n"]):>5}, '
                      f'trd={int(row["trd_n"]):>3}, '
                      f'trd_利={row["trd_利万"]:+6.1f}万, {row["verdict"]}')
    print()
