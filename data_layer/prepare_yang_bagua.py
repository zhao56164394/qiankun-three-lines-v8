# -*- coding: utf-8 -*-
"""
数据层 - 阳系统八卦级别全量训练

用全量历史数据计算各层(年/月/日)八卦统计 + 512组合统计
只用段首采样(每段第一天), 避免自相关

输出:
  data_layer/data/yang_bagua_layer.csv   — 3层×8卦=24行, 各卦统计+排名
  data_layer/data/yang_512_bagua.csv     — 512种年月日组合, 八卦级别统计

新数据到来后重新运行本脚本即可更新
"""
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from bagua_engine import BAGUA_TABLE

GUA_ORDER = ['000', '001', '010', '011', '100', '101', '110', '111']

# 八卦圆理论顺序 (市场周期: 底→顶→底)
BAGUA_CIRCLE = ['000', '001', '010', '011', '111', '110', '101', '100']
CIRCLE_STAGE = {
    '000': 0, '001': 1, '010': 2, '011': 3,
    '111': 4, '110': 5, '101': 6, '100': 7,
}

MIN_SAMPLES = 3  # 最少段首样本数


def prepare_yang_bagua():
    """全量训练阳系统八卦统计"""
    print("=" * 80)
    print("阳系统八卦级别 — 全量训练")
    print("=" * 80)

    # ── 加载数据 ──
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'zz1000_daily.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['date'] = df['date'].astype(str)
    for col in ['year_gua', 'month_gua', 'day_gua']:
        df[col] = df[col].astype(str).str.zfill(3)

    # 未来30日收益
    df['ret_30'] = (df['close'].shift(-30) / df['close'] - 1) * 100
    valid = df[df['ret_30'].notna()].copy()

    # 各层段首标记
    for col in ['year_gua', 'month_gua', 'day_gua']:
        valid[f'{col}_seg_start'] = valid[col] != valid[col].shift(1)

    # 512组合段首标记
    valid['ymd_gua'] = valid['year_gua'] + '_' + valid['month_gua'] + '_' + valid['day_gua']
    valid['ymd_seg_start'] = valid['ymd_gua'] != valid['ymd_gua'].shift(1)

    print(f"数据范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({len(df)}个交易日)")
    print(f"有效样本(有未来30日收益): {len(valid)}个交易日")

    # ════════════════════════════════════════════════════════════
    #  第一部分: 各层八卦统计 → yang_bagua_layer.csv
    # ════════════════════════════════════════════════════════════
    print("\n计算各层八卦统计...")
    layer_rows = []

    for layer_name, col in [('year', 'year_gua'), ('month', 'month_gua'), ('day', 'day_gua')]:
        seg_col = f'{col}_seg_start'
        means = {}

        for code in GUA_ORDER:
            mask = (valid[col] == code) & valid[seg_col]
            rets = valid.loc[mask, 'ret_30']
            n_total = (valid[col] == code).sum()
            n_seg = len(rets)

            row = {
                'layer': layer_name,
                'gua_code': code,
                'gua_name': BAGUA_TABLE[code][0],
                'gua_symbol': BAGUA_TABLE[code][1],
                'gua_meaning': BAGUA_TABLE[code][2],
                'gua_yy': BAGUA_TABLE[code][3],
                'circle_pos': CIRCLE_STAGE[code],
                'n_total': n_total,
                'n_seg': n_seg,
            }

            if n_seg >= MIN_SAMPLES:
                row['mean_ret'] = round(rets.mean(), 4)
                row['median_ret'] = round(rets.median(), 4)
                row['std_ret'] = round(rets.std(), 4)
                row['win_rate'] = round((rets > 0).mean() * 100, 2)
                row['max_ret'] = round(rets.max(), 4)
                row['min_ret'] = round(rets.min(), 4)
                means[code] = rets.mean()
            else:
                for k in ['mean_ret', 'median_ret', 'std_ret', 'win_rate', 'max_ret', 'min_ret']:
                    row[k] = np.nan

            layer_rows.append(row)

        # 计算层内排名 (均值从高到低)
        sorted_codes = sorted(means.keys(), key=lambda c: means[c], reverse=True)
        rank_map = {c: i + 1 for i, c in enumerate(sorted_codes)}
        for row in layer_rows:
            if row['layer'] == layer_name:
                row['rank'] = rank_map.get(row['gua_code'], np.nan)

    layer_df = pd.DataFrame(layer_rows)

    # 保存
    layer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'yang_bagua_layer.csv')
    layer_df.to_csv(layer_path, index=False, encoding='utf-8-sig')
    print(f"保存: {layer_path} ({len(layer_df)}行)")

    # 打印各层统计
    for layer_name in ['year', 'month', 'day']:
        layer_label = {'year': '年卦', 'month': '月卦', 'day': '日卦'}[layer_name]
        sub = layer_df[layer_df['layer'] == layer_name].sort_values('rank')
        print(f"\n  ── {layer_label} (按均值排名) ──")
        print(f"  {'排名':>4} {'卦象':>6} {'阴阳':>4} {'段数':>5} {'均值':>8} {'中位':>8} {'胜率':>6} {'圆位':>4}")
        print(f"  {'─' * 55}")
        for _, r in sub.iterrows():
            if not np.isnan(r['mean_ret']):
                print(f"  {r['rank']:>4.0f} {r['gua_name']}{r['gua_symbol']:>4} {r['gua_yy']:>4}"
                      f" {r['n_seg']:>5} {r['mean_ret']:>+7.2f}% {r['median_ret']:>+7.2f}%"
                      f" {r['win_rate']:>5.1f}% {r['circle_pos']:>4.0f}")
            else:
                print(f"  {'---':>4} {r['gua_name']}{r['gua_symbol']:>4} {r['gua_yy']:>4}"
                      f" {r['n_seg']:>5} {'---':>8} {'---':>8} {'---':>6} {r['circle_pos']:>4.0f}")

    # ════════════════════════════════════════════════════════════
    #  第二部分: 512种组合统计 → yang_512_bagua.csv
    # ════════════════════════════════════════════════════════════
    print(f"\n\n计算512种年×月×日组合统计...")
    combo_rows = []

    for yg in GUA_ORDER:
        for mg in GUA_ORDER:
            for dg in GUA_ORDER:
                combo = f'{yg}_{mg}_{dg}'
                mask = (valid['ymd_gua'] == combo) & valid['ymd_seg_start']
                rets = valid.loc[mask, 'ret_30']
                n_total = (valid['ymd_gua'] == combo).sum()
                n_seg = len(rets)

                row = {
                    'year_gua': yg, 'month_gua': mg, 'day_gua': dg,
                    'year_name': BAGUA_TABLE[yg][0],
                    'month_name': BAGUA_TABLE[mg][0],
                    'day_name': BAGUA_TABLE[dg][0],
                    'year_yy': BAGUA_TABLE[yg][3],
                    'month_yy': BAGUA_TABLE[mg][3],
                    'day_yy': BAGUA_TABLE[dg][3],
                    'n_total': n_total,
                    'n_seg': n_seg,
                }

                if n_seg >= MIN_SAMPLES:
                    row['mean_ret'] = round(rets.mean(), 4)
                    row['median_ret'] = round(rets.median(), 4)
                    row['win_rate'] = round((rets > 0).mean() * 100, 2)
                else:
                    row['mean_ret'] = np.nan
                    row['median_ret'] = np.nan
                    row['win_rate'] = np.nan

                combo_rows.append(row)

    combo_df = pd.DataFrame(combo_rows)

    # 全局排名
    valid_mask = combo_df['n_seg'] >= MIN_SAMPLES
    sorted_idx = combo_df.loc[valid_mask, 'mean_ret'].sort_values(ascending=False).index
    rank_map = {i: r + 1 for r, i in enumerate(sorted_idx)}
    combo_df['rank'] = combo_df.index.map(lambda i: rank_map.get(i, np.nan))
    ranked = combo_df.loc[valid_mask].sort_values('rank')

    # 保存
    combo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'data', 'yang_512_bagua.csv')
    combo_df.to_csv(combo_path, index=False, encoding='utf-8-sig')

    n_valid = (combo_df['n_seg'] >= MIN_SAMPLES).sum()
    print(f"保存: {combo_path} ({len(combo_df)}行, 有效{n_valid}种)")

    # 打印TOP10和BOTTOM10
    if n_valid >= 10:
        top10 = ranked.head(10)
        bot10 = ranked.tail(10).iloc[::-1]

        print(f"\n  ── TOP 10 最强组合 ──")
        print(f"  {'排名':>4} {'年':>3} {'月':>3} {'日':>3} {'阴阳':>6} {'段数':>5} {'均值':>8} {'胜率':>6}")
        print(f"  {'─' * 50}")
        for _, r in top10.iterrows():
            yy = r['year_yy'][0] + r['month_yy'][0] + r['day_yy'][0]
            print(f"  {r['rank']:>4.0f} {r['year_name']:>3} {r['month_name']:>3} {r['day_name']:>3}"
                  f" {yy:>6} {r['n_seg']:>5} {r['mean_ret']:>+7.2f}% {r['win_rate']:>5.1f}%")

        print(f"\n  ── BOTTOM 10 最弱组合 ──")
        print(f"  {'排名':>4} {'年':>3} {'月':>3} {'日':>3} {'阴阳':>6} {'段数':>5} {'均值':>8} {'胜率':>6}")
        print(f"  {'─' * 50}")
        for _, r in bot10.iterrows():
            yy = r['year_yy'][0] + r['month_yy'][0] + r['day_yy'][0]
            print(f"  {r['rank']:>4.0f} {r['year_name']:>3} {r['month_name']:>3} {r['day_name']:>3}"
                  f" {yy:>6} {r['n_seg']:>5} {r['mean_ret']:>+7.2f}% {r['win_rate']:>5.1f}%")

    # ════════════════════════════════════════════════════════════
    #  第三部分: 与阴系统对比
    # ════════════════════════════════════════════════════════════
    yin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'gua_512_advice.csv')
    if os.path.exists(yin_path):
        print(f"\n\n{'=' * 80}")
        print("  阴阳系统512组合对比")
        print(f"{'=' * 80}")

        yin_df = pd.read_csv(yin_path, encoding='utf-8-sig')
        for col in ['year_gua', 'month_gua', 'day_gua']:
            yin_df[col] = yin_df[col].astype(str).str.zfill(3)

        # 合并对比
        merged = combo_df.merge(
            yin_df[['year_gua', 'month_gua', 'day_gua', 'mean_seg', 'win_rate']],
            on=['year_gua', 'month_gua', 'day_gua'],
            suffixes=('_yang', '_yin'),
            how='inner',
        )

        both_valid = merged[
            (merged['n_seg'] >= MIN_SAMPLES) & merged['mean_seg'].notna()
        ].copy()

        if len(both_valid) > 0:
            # 方向一致性
            dir_hits = ((both_valid['mean_ret'] > 0) & (both_valid['mean_seg'] > 0)) | \
                       ((both_valid['mean_ret'] <= 0) & (both_valid['mean_seg'] <= 0))
            dir_rate = dir_hits.sum() / len(both_valid) * 100

            # Spearman
            try:
                from scipy import stats
                rho, p = stats.spearmanr(both_valid['mean_ret'], both_valid['mean_seg'])
                rho_str = f"Spearman ρ = {rho:+.3f} (p={p:.4f})"
            except ImportError:
                rho_str = "scipy未安装, 跳过Spearman"

            print(f"  阴阳都有效的组合: {len(both_valid)}种")
            print(f"  方向一致性: {dir_hits.sum()}/{len(both_valid)} = {dir_rate:.1f}%")
            print(f"  排序一致性: {rho_str}")

    print(f"\n{'=' * 80}")
    print("全量训练完成")
    print(f"{'=' * 80}")

    return layer_df, combo_df


if __name__ == '__main__':
    prepare_yang_bagua()
