# -*- coding: utf-8 -*-
"""导出 M2 出池规则下的实际买入案例供通达信复盘

输出每笔交易:
  入池日, 入池 retail, 入池 mf
  池期内 retail 最低值 / mf 最高值 / 池天
  买入日, 买入 retail, 买入 mf, 买入价
  卖出日, 卖出价, ret%, 卖出原因
  期间是否主升浪
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
TRIGGER_GUA = '011'
REGIME_Y = '000'
POOL_THR = -250
MF_EXIT = 50
LOOKBACK = 30


def main():
    t0 = time.time()
    print('=== 导出 M2 案例供通达信复盘 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board', 'name'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())
    code2name = dict(zip(uni['code'], uni['name']))

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend', 'mkt_y']).reset_index(drop=True)
    print(f'  {len(df):,} 行')

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    print('扫池期 + 巽日触发 (M2 出池: mf<=50) ...')
    events = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        pool_enter_i = -1
        pool_min_retail = np.inf
        pool_max_mf = -np.inf

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i

            if not in_pool and retail_arr[gi] < POOL_THR:
                in_pool = True
                pool_enter_i = i
                pool_min_retail = retail_arr[gi]
                pool_max_mf = mf_arr[gi]
            if in_pool:
                if retail_arr[gi] < pool_min_retail:
                    pool_min_retail = retail_arr[gi]
                if mf_arr[gi] > pool_max_mf:
                    pool_max_mf = mf_arr[gi]

            if in_pool and mf_arr[gi] <= MF_EXIT:
                in_pool = False
                continue

            if in_pool and mkt_y_arr[gi] == REGIME_Y and stk_d_arr[gi] == TRIGGER_GUA:
                # 模拟卖出
                cross_count = 0
                running_max = trend_arr[gi]
                sell_idx = None
                sell_reason = None
                max_end = min(s + n - 1, gi + MAX_HOLD)
                for k in range(gi+1, max_end+1):
                    days_h = k - gi
                    if not np.isnan(trend_arr[k]):
                        running_max = max(running_max, trend_arr[k])
                    if running_max >= 89 and trend_arr[k] < 89 and trend_arr[k-1] >= 89:
                        cross_count += 1
                        if cross_count >= 2:
                            sell_idx = k; sell_reason = 'bull_2nd'; break
                    if days_h >= 20:
                        seg = trend_arr[gi:k+1]
                        valid = seg[~np.isnan(seg)]
                        if len(valid) > 0 and valid.max() < 89:
                            sell_idx = k; sell_reason = 'ts20'; break
                    if days_h >= MAX_HOLD:
                        sell_idx = k; sell_reason = 'timeout'; break
                if sell_idx is None:
                    sell_idx = max_end; sell_reason = 'force_close'

                # 主升浪标记
                gua_window = stk_d_arr[gi+1:gi+31]
                qian_count = (gua_window == '111').sum()

                events.append({
                    'code': code_arr[gi],
                    'name': code2name.get(code_arr[gi], '?'),
                    'pool_enter_date': date_arr[s+pool_enter_i],
                    'pool_enter_retail': retail_arr[s+pool_enter_i],
                    'pool_enter_mf': mf_arr[s+pool_enter_i],
                    'pool_min_retail': pool_min_retail,
                    'pool_max_mf': pool_max_mf,
                    'pool_days': i - pool_enter_i,
                    'buy_date': date_arr[gi],
                    'buy_retail': retail_arr[gi],
                    'buy_mf': mf_arr[gi],
                    'buy_price': close_arr[gi],
                    'sell_date': date_arr[sell_idx],
                    'sell_price': close_arr[sell_idx],
                    'days_held': sell_idx - gi,
                    'ret_pct': (close_arr[sell_idx]/close_arr[gi]-1)*100,
                    'reason': sell_reason,
                    'qian_30d': int(qian_count),
                    'is_zsl': qian_count >= 10,
                })
                in_pool = False  # 简化: 触发后出池 (跟资金回测一致, 同股不重复买)

    df_e = pd.DataFrame(events)
    print(f'  事件: {len(df_e):,}')
    print(f'  全集: ret={df_e["ret_pct"].mean():+.2f}% win={(df_e["ret_pct"]>0).mean()*100:.1f}% '
          f'主升率={df_e["is_zsl"].mean()*100:.1f}%')

    # 按 pool_min_retail 排序 (深池优先, 与策略一致)
    df_e = df_e.sort_values('buy_date')

    # 选 case: 不同年份各挑 2-3 个最有代表性的 (主升浪 + 高 ret)
    print('\n=== 推荐复盘案例 (按年份分组) ===\n')
    df_e['year'] = pd.to_datetime(df_e['buy_date']).dt.year

    samples = []
    for y in sorted(df_e['year'].unique()):
        sub = df_e[df_e['year'] == y].copy()
        if len(sub) < 3: continue
        # 主升浪 + 高 ret 各 1 个 + 失败案例 1 个
        zsl = sub[sub['is_zsl']].sort_values('ret_pct', ascending=False)
        fail = sub[(~sub['is_zsl']) & (sub['ret_pct'] < -3)].sort_values('ret_pct')
        if len(zsl) >= 1:
            samples.append(('成功-主升浪', zsl.iloc[0]))
        if len(zsl) >= 2:
            samples.append(('成功-中等', zsl.iloc[len(zsl)//2]))
        if len(fail) >= 1:
            samples.append(('失败案例', fail.iloc[0]))

    print(f'{"类型":<12} {"代码":<8} {"名称":<10} {"入池日":<12} {"买入日":<12} {"卖出日":<12} '
          f'{"池天":>4} {"入 retail":>8} {"入 mf":>6} {"池底 retail":>10} {"池顶 mf":>7} '
          f'{"买 retail":>8} {"买 mf":>6} {"ret%":>7} {"持仓":>4} {"reason":<10} {"乾日":>4}')
    for tag, row in samples[:30]:
        print(f'{tag:<12} {row["code"]:<8} {str(row["name"])[:8]:<10} '
              f'{row["pool_enter_date"]:<12} {row["buy_date"]:<12} {row["sell_date"]:<12} '
              f'{row["pool_days"]:>4} {row["pool_enter_retail"]:>+8.0f} {row["pool_enter_mf"]:>+6.0f} '
              f'{row["pool_min_retail"]:>+10.0f} {row["pool_max_mf"]:>+7.0f} '
              f'{row["buy_retail"]:>+8.0f} {row["buy_mf"]:>+6.0f} '
              f'{row["ret_pct"]:>+6.2f}% {row["days_held"]:>4} {row["reason"]:<10} {row["qian_30d"]:>4}')

    # 写出全部 trades
    out_dir = os.path.join(ROOT, 'data_layer/data/results')
    os.makedirs(out_dir, exist_ok=True)
    df_e.to_csv(os.path.join(out_dir, 'kun_m2_cases.csv'), index=False, encoding='utf-8-sig')
    print(f'\n  全部案例写到 kun_m2_cases.csv')
    print(f'  排序建议: 按 buy_date 升序, 看不同时期的 case')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
