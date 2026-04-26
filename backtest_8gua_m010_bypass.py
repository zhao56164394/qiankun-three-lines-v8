# -*- coding: utf-8 -*-
"""m_gua=010 日 跳过地/人卦过滤 实验

假设: 周卦 m_gua=010 坎 跨年稳定 +14.31% (7 年 6 正), 且对个股地/人卦近乎免疫
      (8 桶内均收都是 +12~16%). 在这种中周期状态下, 地/人卦过滤只会"误伤",
      反而拒掉本来该买的优信号.

实验: 在 formal 配置基础上, 只在 m_gua=010 日 (约 3345 个候选信号) 放开
      地/人卦过滤 (allow_di_gua → None, exclude_*_gua → set()), 其余 7 种
      m_gua 状态保持 formal 不变.

方法 (不改 backtest_8gua.py):
  1. 跑 formal → 捕获 sig_formal (含所有 formal 过滤)
  2. 临时 patch GUA_STRATEGY: 清空 allow/exclude, 保留 pool_depth_tiers/sell/trigger
     跑 nofilter → 捕获 sig_nofilter (只通过 pool 和 tier, 但地/人卦过滤被跳过)
  3. 建 m_gua 映射 {date: m_gua}
  4. 合并: m=010 当天取 nofilter, 其余天取 formal
  5. 喂给 simulate_8gua → 得到新的 final_capital
  6. 与 formal 基线对比
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_8gua as b8


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT, 'data_layer', 'data')
FORMAL_RESULT = os.path.join(RESULT_DIR, 'backtest_8gua_result.json')
FORMAL_BACKUP = FORMAL_RESULT + '.m010_backup'
MULTI = os.path.join(RESULT_DIR, 'foundation', 'multi_scale_gua_daily.csv')

# 在 formal 基础上"仅清空地/人卦过滤, 其余保留"的 patch
_NOFILTER_PATCH = {
    '000': {'kun_exclude_ren_gua': set(), 'kun_allow_di_gua': None},
    '001': {'gen_allow_di_gua': None},
    '010': {},
    '011': {'xun_allow_di_gua': None},
    '100': {'zhen_exclude_ren_gua': set(), 'zhen_allow_di_gua': None},
    '101': {'li_exclude_ren_gua': set(), 'li_allow_di_gua': None},
    '110': {'dui_exclude_ren_gua': set(), 'dui_allow_di_gua': None,
            'dui_market_stock_whitelist': None},
    '111': {'qian_exclude_ren_gua': set(), 'qian_exclude_di_gua': set()},
}


def load_m_gua_map() -> dict:
    df = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'m_gua': str})
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return dict(zip(df['date'], df['m_gua'].fillna('')))


def apply_patch(patch):
    """对 b8.GUA_STRATEGY 施加 patch, 返回原值快照用于还原."""
    snapshot = {g: copy.deepcopy(s) for g, s in b8.GUA_STRATEGY.items()}
    for g, kv in patch.items():
        for k, v in kv.items():
            b8.GUA_STRATEGY[g][k] = copy.deepcopy(v) if isinstance(v, (set, dict, list)) else v
    return snapshot


def restore_patch(snapshot):
    for g, s in snapshot.items():
        b8.GUA_STRATEGY[g] = s


def run_capture(label):
    """运行 b8.run(), 抑制 stdout, 返回 (result, stats, sig_df)
    sig_df 从 b8.run() 写入的 JSON 文件里读回."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result, stats = b8.run()
    with open(FORMAL_RESULT, encoding='utf-8') as f:
        d = json.load(f)
    sig_df = pd.DataFrame(d.get('signal_detail', []))
    return result, stats, sig_df


def merge_signals(sig_formal: pd.DataFrame, sig_nofilter: pd.DataFrame, m_gua_map: dict) -> pd.DataFrame:
    """m_gua=010 当天信号用 nofilter, 其余用 formal."""
    sf = sig_formal.copy()
    sn = sig_nofilter.copy()
    sf['_m_gua'] = sf['signal_date'].map(m_gua_map).fillna('')
    sn['_m_gua'] = sn['signal_date'].map(m_gua_map).fillna('')

    part_formal = sf[sf['_m_gua'] != '010']
    part_nofilter = sn[sn['_m_gua'] == '010']

    merged = pd.concat([part_formal, part_nofilter], ignore_index=True)
    merged = merged.drop_duplicates(subset=['code', 'signal_date'], keep='first')
    return merged.drop(columns=['_m_gua'])


def summarize_stats(trade_log, init_capital, final_capital):
    if not trade_log:
        return {
            'final_capital': final_capital,
            'total_return': (final_capital / init_capital - 1) * 100,
            'trade_count': 0,
            'win_rate': 0,
        }
    wins = sum(1 for t in trade_log if (t.get('profit') or 0) > 0)
    return {
        'final_capital': final_capital,
        'total_return': (final_capital / init_capital - 1) * 100,
        'trade_count': len(trade_log),
        'win_rate': wins / len(trade_log) * 100,
    }


def main():
    # 备份 formal result
    if os.path.exists(FORMAL_RESULT):
        shutil.copyfile(FORMAL_RESULT, FORMAL_BACKUP)
        print(f'[备份] formal result → {os.path.basename(FORMAL_BACKUP)}')

    try:
        # 1. formal 基线
        print('\n===== 1. formal 基线 =====')
        r_formal, s_formal, sig_formal = run_capture('formal')
        print(f'  formal final: {s_formal["final_capital"]:>15,.0f}  '
              f'total_return {s_formal["total_return"]:+.1f}%  '
              f'trades {s_formal["trade_count"]}  win_rate {s_formal["win_rate"]:.1f}%')
        print(f'  formal signals: {len(sig_formal):,}')

        # 2. nofilter 跑 (只清地/人卦过滤, 保留 pool/tier/sell/trigger)
        print('\n===== 2. nofilter 跑 (地/人卦过滤清空) =====')
        snapshot = apply_patch(_NOFILTER_PATCH)
        try:
            r_nofilter, s_nofilter, sig_nofilter = run_capture('nofilter')
        finally:
            restore_patch(snapshot)
        print(f'  nofilter final: {s_nofilter["final_capital"]:>15,.0f}  '
              f'total_return {s_nofilter["total_return"]:+.1f}%  '
              f'trades {s_nofilter["trade_count"]}')
        print(f'  nofilter signals: {len(sig_nofilter):,}')

        # 3. 合并信号: m=010 天用 nofilter, 其余用 formal
        print('\n===== 3. 合并 (m=010 天 nofilter, 其余 formal) =====')
        m_gua_map = load_m_gua_map()
        sig_merged = merge_signals(sig_formal, sig_nofilter, m_gua_map)
        n_m010_days = sum(1 for v in m_gua_map.values() if v == '010')
        n_m010_sigs = (sig_merged['signal_date'].map(m_gua_map) == '010').sum()
        n_nonm010_sigs = len(sig_merged) - n_m010_sigs
        print(f'  m_gua=010 的日历天数: {n_m010_days} (约 15% 回测天)')
        print(f'  合并后信号: {len(sig_merged):,} 条')
        print(f'    其中 m=010 天 (来自 nofilter): {n_m010_sigs:,}')
        print(f'    其余天   (来自 formal):   {n_nonm010_sigs:,}')

        # 4. simulate_8gua 回放合并信号
        print('\n===== 4. 用合并信号重跑资金曲线 =====')
        # 需要 zz_df (DataFrame) 和 d_gua_map
        zz_df = b8.load_zz1000_full()
        ms = pd.read_csv(MULTI, encoding='utf-8-sig', dtype={'d_gua': str})
        ms['date'] = pd.to_datetime(ms['date']).dt.strftime('%Y-%m-%d')
        d_gua_map = dict(zip(ms['date'], ms['d_gua'].fillna('')))

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sim = b8.simulate_8gua(sig_merged, zz_df, tian_gua_map_ext=d_gua_map)

        init_cap = sim['init_capital']
        final_cap = sim['final_capital']
        ss = summarize_stats(sim['trade_log'], init_cap, final_cap)
        print(f'  merged final: {ss["final_capital"]:>15,.0f}  '
              f'total_return {ss["total_return"]:+.1f}%  '
              f'trades {ss["trade_count"]}  win_rate {ss["win_rate"]:.1f}%')

        # 5. 对比
        print('\n' + '=' * 80)
        print('  结果对比')
        print('=' * 80)
        print(f'  {"方案":<24} {"final_capital":>15} {"total_return":>12} {"trades":>7} {"win_rate":>8}')
        print('  ' + '-' * 70)
        print(f'  {"formal (基线)":<24} {s_formal["final_capital"]:>15,.0f} '
              f'{s_formal["total_return"]:>+11.1f}% {s_formal["trade_count"]:>7} {s_formal["win_rate"]:>7.1f}%')
        print(f'  {"nofilter (全无 di/ren)":<24} {s_nofilter["final_capital"]:>15,.0f} '
              f'{s_nofilter["total_return"]:>+11.1f}% {s_nofilter["trade_count"]:>7} {s_nofilter["win_rate"]:>7.1f}%')
        print(f'  {"m=010 bypass":<24} {ss["final_capital"]:>15,.0f} '
              f'{ss["total_return"]:>+11.1f}% {ss["trade_count"]:>7} {ss["win_rate"]:>7.1f}%')
        print()
        delta_bypass = ss["final_capital"] - s_formal["final_capital"]
        delta_bypass_pp = ss["total_return"] - s_formal["total_return"]
        delta_nofilter = s_nofilter["final_capital"] - s_formal["final_capital"]
        delta_nofilter_pp = s_nofilter["total_return"] - s_formal["total_return"]
        print(f'  nofilter vs formal: {delta_nofilter:+,.0f} ({delta_nofilter_pp:+.1f} pp)')
        print(f'  m=010 bypass vs formal: {delta_bypass:+,.0f} ({delta_bypass_pp:+.1f} pp)')
    finally:
        if os.path.exists(FORMAL_BACKUP):
            shutil.copyfile(FORMAL_BACKUP, FORMAL_RESULT)
            os.remove(FORMAL_BACKUP)
            print(f'\n[还原] formal result')


if __name__ == '__main__':
    main()
