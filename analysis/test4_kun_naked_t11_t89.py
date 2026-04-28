# -*- coding: utf-8 -*-
"""Phase 3 坤桶研究 Step 1 — 裸 11→89 配对基线统计.

目的:
  在坤期 (y_gua=000) 内, 个股趋势线"上穿 11"信号, 60 天内能走到"下穿 89"的成功率/收益.
  这是基线 — 后续加主力线/散户线过滤的提升量, 都以此为参照.

定义:
  - 信号 (entry): close 当日 trend >= 11 且前一日 trend < 11, 且大盘 y_gua = '000'
  - 成功 (exit success): 入场后 1..60 天内, 首个 close 下穿 89 (前一日 >=89, 当日 <89)
  - 失败 (exit forced): 入场后 60 天内未触发下穿 89, 强制平仓在第 60 天

输出:
  - 控制台报表 (信号数 / 成功率 / 收益分布 / 持仓天数)
  - data_layer/data/ablation/test4/kun_naked_t11_t89.json (供下一步联合主力/散户用)

数据源: 主板 3112 只 (universe 过滤后)
"""
import os
import sys
import io
import json
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLD_MAX = 60


def load_main_codes():
    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                                       'main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    return set(uni[uni['board'] == '主板']['code'].unique())


def load_y_gua_map():
    df = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'foundation',
                                      'multi_scale_gua_daily.parquet'),
                         columns=['date', 'y_gua'])
    df['date'] = df['date'].astype(str)
    df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
    return dict(zip(df['date'], df['y_gua']))


def main():
    print('=== 加载数据 ===')
    main_codes = load_main_codes()
    print(f'主板代码: {len(main_codes)}')

    df = pd.read_parquet(os.path.join(ROOT, 'data_layer', 'data', 'stocks.parquet'),
                         columns=['code', 'date', 'close', 'trend'])
    df['code'] = df['code'].astype(str).str.zfill(6)
    df['date'] = df['date'].astype(str).str[:10]
    df = df[df['code'].isin(main_codes)].copy()
    print(f'主板行数: {len(df)}, 唯一股: {df["code"].nunique()}')

    y_map = load_y_gua_map()
    print(f'y_gua 映射: {len(y_map)} 天')

    # 排序 + 按 code 分组扫描信号
    df = df.sort_values(['code', 'date']).reset_index(drop=True)

    print('\n=== 扫描信号 ===')
    results = []
    skip_short = 0
    n_codes = 0
    for code, g in df.groupby('code', sort=False):
        n_codes += 1
        g = g.reset_index(drop=True)
        n = len(g)
        if n < HOLD_MAX + 2:
            continue

        trends = g['trend'].to_numpy(dtype=float)
        closes = g['close'].to_numpy(dtype=float)
        dates = g['date'].to_numpy()

        prev_trends = np.concatenate([[np.nan], trends[:-1]])
        cross_up_11 = (prev_trends < 11) & (trends >= 11)
        cross_down_89 = (prev_trends >= 89) & (trends < 89)

        # 该股每天的 y_gua
        y_arr = np.array([y_map.get(d, '???') for d in dates])
        sig_idxs = np.where(cross_up_11 & (y_arr == '000'))[0]

        for i in sig_idxs:
            # 数据右截断: 至少要有 HOLD_MAX 天未来
            if i + HOLD_MAX >= n:
                skip_short += 1
                continue

            end = i + 1 + HOLD_MAX
            window_trend = trends[i+1:end]
            window_close = closes[i+1:end]
            future_cd = np.where(cross_down_89[i+1:end])[0]

            entry_close = closes[i]
            entry_date = dates[i]
            max_trend_future = float(np.nanmax(window_trend)) if len(window_trend) else np.nan
            max_close_future = float(np.nanmax(window_close)) if len(window_close) else np.nan
            max_pct_future = (max_close_future / entry_close - 1) * 100 if entry_close > 0 else np.nan

            if len(future_cd):
                j = i + 1 + int(future_cd[0])
                exit_close = closes[j]
                results.append({
                    'code': code, 'entry_date': entry_date, 'exit_date': dates[j],
                    'hold_days': int(j - i),
                    'success': True,
                    'ret_pct': (exit_close / entry_close - 1) * 100,
                    'max_pct': max_pct_future,
                    'max_trend': max_trend_future,
                })
            else:
                # 强制 60 天平仓
                j = i + HOLD_MAX
                exit_close = closes[j]
                results.append({
                    'code': code, 'entry_date': entry_date, 'exit_date': dates[j],
                    'hold_days': HOLD_MAX,
                    'success': False,
                    'ret_pct': (exit_close / entry_close - 1) * 100,
                    'max_pct': max_pct_future,
                    'max_trend': max_trend_future,
                })

    print(f'扫描股票数: {n_codes}, 信号数: {len(results)}, 因数据右截断跳过: {skip_short}')

    if not results:
        print('无信号, 退出')
        return

    rs = pd.DataFrame(results)
    n_sig = len(rs)
    n_succ = int(rs['success'].sum())
    succ_rate = n_succ / n_sig * 100

    # === 总体统计 ===
    print('\n' + '=' * 80)
    print('总体: 坤期裸 11→89 配对')
    print('=' * 80)
    print(f'  信号数: {n_sig}')
    print(f'  成功 (走到下穿 89): {n_succ} ({succ_rate:.1f}%)')
    print(f'  失败 (60 天未到 89): {n_sig - n_succ} ({100-succ_rate:.1f}%)')

    # 成功信号
    succ = rs[rs['success']]
    if len(succ):
        print(f'\n  [成功信号] 持仓天数: mean={succ["hold_days"].mean():.1f}, '
              f'median={succ["hold_days"].median():.0f}, '
              f'p25={succ["hold_days"].quantile(.25):.0f}, p75={succ["hold_days"].quantile(.75):.0f}')
        print(f'  [成功信号] 收益% : mean={succ["ret_pct"].mean():+.2f}%, '
              f'median={succ["ret_pct"].median():+.2f}%, '
              f'p25={succ["ret_pct"].quantile(.25):+.2f}%, p75={succ["ret_pct"].quantile(.75):+.2f}%')
        print(f'  [成功信号] 期间最高% : mean={succ["max_pct"].mean():+.2f}%, '
              f'median={succ["max_pct"].median():+.2f}%')

    # 失败信号
    fail = rs[~rs['success']]
    if len(fail):
        print(f'\n  [失败信号] 强制平仓收益% : mean={fail["ret_pct"].mean():+.2f}%, '
              f'median={fail["ret_pct"].median():+.2f}%')
        print(f'  [失败信号] 期间最高% : mean={fail["max_pct"].mean():+.2f}%, '
              f'median={fail["max_pct"].median():+.2f}%')
        print(f'  [失败信号] 期间最高 trend: mean={fail["max_trend"].mean():.1f}, '
              f'median={fail["max_trend"].median():.0f}')
        # 失败的信号里, 最高 trend 到了哪些档位?
        print(f'\n  [失败信号 期间最高 trend 分布]:')
        for label, lo, hi in [('0-30', 0, 30), ('30-50', 30, 50), ('50-70', 50, 70),
                              ('70-89', 70, 89)]:
            n = ((fail['max_trend'] >= lo) & (fail['max_trend'] < hi)).sum()
            print(f'    {label}: {n} ({n/len(fail)*100:.1f}%)')

    # === 总收益: 按 sig 平均 ===
    total_pct_mean = rs['ret_pct'].mean()
    total_pct_median = rs['ret_pct'].median()
    print(f'\n  [全信号合并] 实际收益% mean={total_pct_mean:+.2f}%, median={total_pct_median:+.2f}%')

    # === 按年/段 分布 ===
    print('\n' + '=' * 80)
    print('按年统计 (信号数 / 成功率 / 平均收益%)')
    print('=' * 80)
    rs['year'] = rs['entry_date'].str[:4]
    print(f'  {"年":<6} {"信号":>6} {"成功":>6} {"成功率":>8} {"平均收益%":>10}')
    print('  ' + '-' * 50)
    for yr in sorted(rs['year'].unique()):
        sub = rs[rs['year'] == yr]
        sr = sub['success'].mean() * 100
        rt = sub['ret_pct'].mean()
        print(f'  {yr:<6} {len(sub):>6d} {int(sub["success"].sum()):>6d} '
              f'{sr:>7.1f}% {rt:>+9.2f}%')

    # 落地
    out_dir = os.path.join(ROOT, 'data_layer', 'data', 'ablation', 'test4')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kun_naked_t11_t89.json')
    out = {
        'meta': {
            'n_signals': int(n_sig), 'n_success': int(n_succ),
            'success_rate_pct': float(succ_rate),
            'mean_ret_pct': float(total_pct_mean), 'median_ret_pct': float(total_pct_median),
            'hold_max': HOLD_MAX,
            'universe': 'main_board (incl ST), 3112 stocks',
        },
        'results': rs.to_dict('records'),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n落地: {out_path}')


if __name__ == '__main__':
    main()
