# -*- coding: utf-8 -*-
"""D × U 笛卡尔积扫描 — 找出最优"双降卖 + 单升买"波段组合

D (双降卖) 候选:
  D1: mf_chg<0 AND retail_chg<0
  D2: D1 AND mf>0 AND retail>0
  D3: mf_chg<-50 AND retail_chg<-50
  D4: mf_chg<-100 AND retail_chg<-50
  D5: 5d Σmf<-100 AND 5d Σretail<-50
  D6: D1 AND trend 下降
  D7: D1 AND mf<上次买入mf

U (单升买) 候选:
  U1: mf_chg>0
  U2: mf_chg>0 AND mf>50
  U3: mf_chg>50
  U4: mf 连续 2 日上升
  U5: U1 AND trend>11
  U6: U1 AND mf>上次卖出mf

最大段终结:
  trend<11 (强卖) OR 60d timeout
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INIT_CAPITAL = 200_000
MAX_HOLD = 60
LOOKBACK = 30


def find_signals(arrays):
    """E1+E2+E3 入场: retail<-250 池中 + mf 上穿 50 + retail 上升"""
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']
    date = arrays['date']; code = arrays['code']

    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        n = e - s
        in_pool = False
        prev_below = False
        last_mf = -np.inf
        last_retail = np.nan

        for i in range(LOOKBACK, n - MAX_HOLD - 1):
            gi = s + i
            cur_below = retail[gi] < -250

            if not in_pool and cur_below and not prev_below:
                in_pool = True

            mf_cross_up = (last_mf <= 50) and (mf[gi] > 50)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)

            if in_pool and mf_cross_up and retail_rising:
                events.append({
                    'date': date[gi], 'code': code[gi],
                    'buy_idx_global': gi,
                })
                in_pool = False

            last_mf = mf[gi]
            last_retail = retail[gi]
            prev_below = cur_below

    return pd.DataFrame(events)


def is_d_signal(d, k, mf, retail, td, last_buy_mf):
    """检查双降卖信号 d 在 k 处是否触发"""
    if k < 1: return False
    mf_c = mf[k] - mf[k-1]
    ret_c = retail[k] - retail[k-1]
    if np.isnan(mf_c) or np.isnan(ret_c): return False

    if d == 'D1':
        return mf_c < 0 and ret_c < 0
    elif d == 'D2':
        return mf_c < 0 and ret_c < 0 and mf[k] > 0 and retail[k] > 0
    elif d == 'D3':
        return mf_c < -50 and ret_c < -50
    elif d == 'D4':
        return mf_c < -100 and ret_c < -50
    elif d == 'D5':
        if k < 5: return False
        mf_5 = mf[k] - mf[k-5]
        ret_5 = retail[k] - retail[k-5]
        return mf_5 < -100 and ret_5 < -50
    elif d == 'D6':
        if k < 1: return False
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0
        return mf_c < 0 and ret_c < 0 and td_c < 0
    elif d == 'D7':
        return mf_c < 0 and ret_c < 0 and mf[k] < last_buy_mf
    return False


def is_u_signal(u, k, mf, td, last_sell_mf):
    """检查单升买信号 u 在 k 处是否触发"""
    if k < 1: return False
    mf_c = mf[k] - mf[k-1]
    if np.isnan(mf_c): return False

    if u == 'U1':
        return mf_c > 0
    elif u == 'U2':
        return mf_c > 0 and mf[k] > 50
    elif u == 'U3':
        return mf_c > 50
    elif u == 'U4':
        if k < 2: return False
        mf_c2 = mf[k-1] - mf[k-2] if not np.isnan(mf[k-2]) else 0
        return mf_c > 0 and mf_c2 > 0
    elif u == 'U5':
        return mf_c > 0 and (np.isnan(td[k]) or td[k] > 11)
    elif u == 'U6':
        return mf_c > 0 and mf[k] > last_sell_mf
    return False


def simulate_swing(buy_idx, td, close, mf, retail, max_end, d_mode, u_mode):
    """波段模拟"""
    bp_first = close[buy_idx]
    cum_mult = 1.0
    holding = True
    cur_buy_price = bp_first
    last_buy_mf = mf[buy_idx]  # 上次买入时 mf
    last_sell_mf = mf[buy_idx]  # 上次卖出时 mf (初始化)
    legs = 0

    for k in range(buy_idx + 1, max_end + 1):
        days = k - buy_idx

        # trend<11 强卖
        if not np.isnan(td[k]) and td[k] < 11:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'td<11', (cum_mult-1)*100, legs

        # 60d timeout
        if days >= MAX_HOLD:
            if holding:
                cum_mult *= close[k] / cur_buy_price
                legs += 1
            return k, 'timeout', (cum_mult-1)*100, legs

        if holding:
            # 检查 D 信号
            if is_d_signal(d_mode, k, mf, retail, td, last_buy_mf):
                cum_mult *= close[k] / cur_buy_price
                legs += 1
                holding = False
                last_sell_mf = mf[k]
                continue
        else:
            # 检查 U 信号
            if is_u_signal(u_mode, k, mf, td, last_sell_mf):
                cur_buy_price = close[k]
                last_buy_mf = mf[k]
                holding = True
                continue

    # 末尾
    if holding:
        cum_mult *= close[max_end] / cur_buy_price
        legs += 1
    return max_end, 'fc', (cum_mult-1)*100, legs


def main():
    t0 = time.time()
    print('=== D × U 笛卡尔积扫描 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d'}, inplace=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date', 'code'], how='inner')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {
        'code': code_arr,
        'date': df['date'].to_numpy(),
        'retail': df['retail'].to_numpy().astype(np.float64),
        'mf': df['main_force'].to_numpy().astype(np.float64),
        'starts': code_starts, 'ends': code_ends,
    }
    close_arr = df['close'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    mf_arr = arrays['mf']
    retail_arr = arrays['retail']

    df_e = find_signals(arrays)
    print(f'  入场信号: {len(df_e):,}')

    D_modes = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    U_modes = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']

    print(f'\n=== 全样本 D × U 网格 (avg ret%) ===\n')
    print(f'  {" ":<6}', end='')
    for u in U_modes:
        print(f' {u:>8}', end='')
    print()

    grid = {}
    for d in D_modes:
        print(f'  {d:<6}', end='')
        for u in U_modes:
            rets = []
            for _, ev in df_e.iterrows():
                gi = int(ev['buy_idx_global'])
                ci = np.searchsorted(code_starts, gi, side='right') - 1
                e = code_ends[ci]
                max_end = min(e - 1, gi + MAX_HOLD)
                _, _, ret, _ = simulate_swing(gi, trend_arr, close_arr, mf_arr, retail_arr,
                                                  max_end, d, u)
                rets.append(ret)
            avg = np.mean(rets)
            grid[(d, u)] = avg
            print(f' {avg:>+7.2f}%', end='')
        print()

    # 找最优 5 组
    sorted_grid = sorted(grid.items(), key=lambda x: -x[1])
    print(f'\n=== 全样本 Top 5 组合 ===\n')
    for (d, u), avg in sorted_grid[:5]:
        # 详细看
        rets = []; legs_list = []; reasons = []
        for _, ev in df_e.iterrows():
            gi = int(ev['buy_idx_global'])
            ci = np.searchsorted(code_starts, gi, side='right') - 1
            e = code_ends[ci]
            max_end = min(e - 1, gi + MAX_HOLD)
            _, r, ret, legs = simulate_swing(gi, trend_arr, close_arr, mf_arr, retail_arr,
                                                  max_end, d, u)
            rets.append(ret); legs_list.append(legs); reasons.append(r)
        df_x = pd.DataFrame({'ret': rets, 'legs': legs_list, 'reason': reasons})
        win = (df_x['ret']>0).mean()*100
        med = df_x['ret'].median()
        print(f'  {d}+{u}: avg={avg:+.2f}%, win={win:.1f}%, 中位={med:+.2f}%, '
              f'avg_legs={df_x["legs"].mean():.1f}')

    # 神火 / 顺丰 在 D × U 网格的表现
    print(f'\n=== 顺丰 002352 (60 天最终 +20.54%) D × U 网格 ===\n')
    sf = df[df['code'] == '002352'].sort_values('date').reset_index(drop=True)
    sf_idx = sf[sf['date'] == '2016-01-19'].index[0]
    sf_close = sf['close'].to_numpy().astype(np.float64)
    sf_mf = sf['main_force'].to_numpy().astype(np.float64)
    sf_ret = sf['retail'].to_numpy().astype(np.float64)
    sf_td = sf['d_trend'].to_numpy().astype(np.float64)
    sf_max = min(len(sf) - 1, sf_idx + 60)

    print(f'  {" ":<6}', end='')
    for u in U_modes:
        print(f' {u:>8}', end='')
    print()
    sf_grid = {}
    for d in D_modes:
        print(f'  {d:<6}', end='')
        for u in U_modes:
            _, _, ret, _ = simulate_swing(sf_idx, sf_td, sf_close, sf_mf, sf_ret,
                                                sf_max, d, u)
            sf_grid[(d, u)] = ret
            print(f' {ret:>+7.2f}%', end='')
        print()

    print(f'\n=== 神火 000933 (60 天最终 -59%) D × U 网格 ===\n')
    sh = df[df['code'] == '000933'].sort_values('date').reset_index(drop=True)
    sh_idx = sh[sh['date'] == '2016-02-17'].index[0]
    sh_close = sh['close'].to_numpy().astype(np.float64)
    sh_mf = sh['main_force'].to_numpy().astype(np.float64)
    sh_ret = sh['retail'].to_numpy().astype(np.float64)
    sh_td = sh['d_trend'].to_numpy().astype(np.float64)
    sh_max = min(len(sh) - 1, sh_idx + 60)

    print(f'  {" ":<6}', end='')
    for u in U_modes:
        print(f' {u:>8}', end='')
    print()
    sh_grid = {}
    for d in D_modes:
        print(f'  {d:<6}', end='')
        for u in U_modes:
            _, _, ret, _ = simulate_swing(sh_idx, sh_td, sh_close, sh_mf, sh_ret,
                                                sh_max, d, u)
            sh_grid[(d, u)] = ret
            print(f' {ret:>+7.2f}%', end='')
        print()

    # 找出"两边都正"的组合
    print(f'\n=== 顺丰 + 神火 都 >0 的 D × U 组合 ===\n')
    win_both = []
    for (d, u), sf_v in sf_grid.items():
        sh_v = sh_grid[(d, u)]
        if sf_v > 0 and sh_v > 0:
            win_both.append((d, u, sf_v, sh_v, grid[(d, u)]))
    win_both.sort(key=lambda x: -(x[2] + x[3]))
    for d, u, sf_v, sh_v, all_v in win_both[:10]:
        print(f'  {d}+{u}: 顺丰{sf_v:>+6.2f}% / 神火{sh_v:>+6.2f}% / 全样本{all_v:>+5.2f}%')

    # baseline 对比
    print(f'\n=== baseline (bull_2nd 单次) 对比 ===')
    base_rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_HOLD)
        bp = close_arr[gi]
        cross_count = 0
        running_max = trend_arr[gi]
        ret_pct = 0; reason = 'fc'
        for k in range(gi + 1, max_end + 1):
            days = k - gi
            if not np.isnan(trend_arr[k]):
                running_max = max(running_max, trend_arr[k])
            if running_max >= 89 and trend_arr[k] < 89 and trend_arr[k-1] >= 89:
                cross_count += 1
                if cross_count >= 2:
                    ret_pct = (close_arr[k]/bp-1)*100
                    break
            if days >= 20:
                seg = trend_arr[gi:k+1]
                valid = seg[~np.isnan(seg)]
                if len(valid) > 0 and valid.max() < 89:
                    ret_pct = (close_arr[k]/bp-1)*100
                    break
            if days >= MAX_HOLD:
                ret_pct = (close_arr[k]/bp-1)*100
                break
        else:
            ret_pct = (close_arr[max_end]/bp-1)*100
        base_rets.append(ret_pct)
    print(f'  全样本 avg = {np.mean(base_rets):+.2f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
