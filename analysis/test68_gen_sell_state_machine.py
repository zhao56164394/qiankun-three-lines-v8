# -*- coding: utf-8 -*-
"""艮 regime 卖点 v3 — 状态机 (主升进入态 → 退出信号)

策略: 不依赖固定窗口, 用"主升状态确认 + 状态退出"两阶段

进入态判定 (满足任一进入主升状态):
  S1: d_trend 到过 ≥80
  S2: 已出现过 d_gua=111 乾
  S3: trend 5 日斜率 ≥ +20

进入态后, 退出信号 (任一触发即出):
  E1: trend 从最高点回撤 ≥ X 点
  E2: 个股月卦切到 110 兑 (中期顶)
  E3: 个股月卦切到 111 乾 (月线全强 = 末段)
  E4: d_gua 第 N 次切回 000 坤 (跌势重启)
  E5: 跌破入场价 -10%

兜底: 不存在! 取 MAX_HOLD=180 测真实是否会触发, 不触发就标 timeout

进入态前, 只用 -10% 价格止损 (避免假突破死扛)
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOOKBACK = 30
MAX_HOLD = 180
QIAN_RUN = 10


def sell_state_machine(td_seg, gua_seg, stk_m_seg, cl_seg,
                        enter_trigger='td80',  # td80 / qian1 / slope20
                        exit_trigger='td_drop15',  # td_drop_X / m_dui / m_qian / kun_back / no_progress_30
                        stop_pct=-10):
    """状态机卖点

    返回: (sell_idx, sell_type)
    """
    n = len(td_seg)
    p0 = cl_seg[0]
    in_state = False
    state_max_td = 0
    state_max_p = p0
    kun_count = 0
    state_enter_idx = -1

    for k in range(1, n):
        # 价格止损 (任何阶段)
        if (cl_seg[k] / p0 - 1) * 100 <= stop_pct:
            return k, 'stop_pct'

        # 进入态判定
        if not in_state:
            entered = False
            if enter_trigger == 'td80' and not np.isnan(td_seg[k]) and td_seg[k] >= 80:
                entered = True
            elif enter_trigger == 'qian1' and gua_seg[k] == '111':
                entered = True
            elif enter_trigger == 'slope20' and k >= 5:
                if not np.isnan(td_seg[k]) and not np.isnan(td_seg[k-5]):
                    if td_seg[k] - td_seg[k-5] >= 20:
                        entered = True
            if entered:
                in_state = True
                state_enter_idx = k
                state_max_td = td_seg[k] if not np.isnan(td_seg[k]) else 0
                state_max_p = cl_seg[k]
            continue

        # 进入态后
        if not np.isnan(td_seg[k]):
            state_max_td = max(state_max_td, td_seg[k])
        if cl_seg[k] > state_max_p:
            state_max_p = cl_seg[k]

        # 退出信号
        if exit_trigger.startswith('td_drop'):
            x = int(exit_trigger.split('_')[-1])
            if not np.isnan(td_seg[k]) and state_max_td - td_seg[k] >= x:
                return k, exit_trigger
        elif exit_trigger == 'm_dui':
            if stk_m_seg[k] == '110':
                return k, 'm_dui'
        elif exit_trigger == 'm_qian':
            if stk_m_seg[k] == '111':
                return k, 'm_qian'
        elif exit_trigger == 'm_strong':
            if stk_m_seg[k] in {'110', '111'}:
                return k, 'm_strong'
        elif exit_trigger == 'kun_back':
            if gua_seg[k] == '000':
                kun_count += 1
                if kun_count >= 1:
                    return k, 'kun_back'
        elif exit_trigger == 'p_drop_max10':
            # 从状态内最高价回撤 10%
            if (cl_seg[k] / state_max_p - 1) * 100 <= -10:
                return k, 'p_drop10'
        elif exit_trigger == 'p_drop_max15':
            if (cl_seg[k] / state_max_p - 1) * 100 <= -15:
                return k, 'p_drop15'
        elif exit_trigger == 'p_drop_max7':
            if (cl_seg[k] / state_max_p - 1) * 100 <= -7:
                return k, 'p_drop7'
        elif exit_trigger == 'no_progress_30':
            # 进入态后 ≥30 日 trend 没创新高
            if k - state_enter_idx >= 30 and state_max_td <= td_seg[state_enter_idx] + 2:
                return k, 'no_progress_30'

    return n - 1, 'timeout'


def main():
    print('=== 加载 ===')
    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'm_gua', 'y_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        g[c] = g[c].astype(str).str.zfill(3)
    g.rename(columns={'d_gua': 'stk_d', 'm_gua': 'stk_m', 'y_gua': 'stk_y'}, inplace=True)

    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'y_gua'])
    market['date'] = market['date'].astype(str)
    market['mkt_y'] = market['y_gua'].astype(str).str.zfill(3)
    market = market[['date', 'mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'main_force', 'retail'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)

    df = g.merge(p, on=['date', 'code'], how='inner').merge(market, on='date', how='left')
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df = df.dropna(subset=['close', 'stk_d', 'd_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy(); date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    stk_d_arr = df['stk_d'].to_numpy()
    stk_m_arr = df['stk_m'].to_numpy(); stk_y_arr = df['stk_y'].to_numpy()
    mkt_y_arr = df['mkt_y'].to_numpy()
    trend_arr = df['d_trend'].to_numpy().astype(np.float32)
    sanhu_arr = df['retail'].to_numpy().astype(np.float32)

    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    rows = []
    for ci in range(len(code_starts)):
        s = code_starts[ci]; e = code_ends[ci]
        if e - s < LOOKBACK + MAX_HOLD + 5: continue
        for i in range(LOOKBACK, e - s - MAX_HOLD - 1):
            gi = s + i
            if mkt_y_arr[gi] != '001': continue
            if stk_d_arr[gi] != '011': continue
            if stk_m_arr[gi] == '111': continue
            score = 0
            if stk_y_arr[gi] == '101': score += 1
            if gi - 5 >= s:
                sanhu_5d = float(np.nanmean(sanhu_arr[gi-5:gi+1]))
                if sanhu_5d < -50: score += 1
                elif sanhu_5d < -30: score += 1
            if score < 2: continue

            buy = i; end = i + MAX_HOLD + 1
            if s + end > e: continue
            gua_seg = stk_d_arr[s+buy:s+end]
            stk_m_seg = stk_m_arr[s+buy:s+end]
            cl_seg = close_arr[s+buy:s+end]
            td_seg = trend_arr[s+buy:s+end]
            n_qian = int((gua_seg[:31] == '111').sum())

            row = {'date': date_arr[gi], 'is_zsl': n_qian >= QIAN_RUN}

            # 测多种状态机组合
            combos = [
                ('td80_pdrop10', 'td80', 'p_drop_max10'),
                ('td80_pdrop7',  'td80', 'p_drop_max7'),
                ('td80_pdrop15', 'td80', 'p_drop_max15'),
                ('td80_tddrop15','td80', 'td_drop_15'),
                ('td80_tddrop20','td80', 'td_drop_20'),
                ('td80_mdui',    'td80', 'm_dui'),
                ('td80_mqian',   'td80', 'm_qian'),
                ('td80_mstrong', 'td80', 'm_strong'),
                ('td80_kunback', 'td80', 'kun_back'),
                ('td80_noprog30','td80', 'no_progress_30'),
                ('qian1_pdrop10','qian1','p_drop_max10'),
                ('qian1_mdui',   'qian1','m_dui'),
                ('qian1_mqian',  'qian1','m_qian'),
                ('qian1_mstrong','qian1','m_strong'),
                ('qian1_tddrop15','qian1','td_drop_15'),
                ('qian1_kunback','qian1','kun_back'),
            ]

            for name, ent, ext in combos:
                k, st = sell_state_machine(td_seg, gua_seg, stk_m_seg, cl_seg,
                                            enter_trigger=ent, exit_trigger=ext)
                if k < len(cl_seg):
                    row[f'r_{name}'] = (cl_seg[k] / cl_seg[0] - 1) * 100
                    row[f'h_{name}'] = k
                    row[f't_{name}'] = st
            rows.append(row)

    df_h = pd.DataFrame(rows)
    df_h['seg'] = ''
    df_h.loc[(df_h['date'] >= '2019-01-01') & (df_h['date'] < '2020-01-01'), 'seg'] = 'w2_2019'
    df_h.loc[(df_h['date'] >= '2021-01-01') & (df_h['date'] < '2022-01-01'), 'seg'] = 'w4_2021'
    print(f'\n入场: {len(df_h)}')

    print(f'\n## 状态机卖点对比 (进入态 + 退出信号)')
    print(f'  {"机制":<22} {"全期望":>7} {"胜率":>5} {"持仓":>5} {"主升期":>8} {"假期":>7} {"w2":>7} {"w4":>7} {"timeout%":>9} {"日bps":>7}')
    combos = ['td80_pdrop10', 'td80_pdrop7', 'td80_pdrop15',
               'td80_tddrop15', 'td80_tddrop20',
               'td80_mdui', 'td80_mqian', 'td80_mstrong', 'td80_kunback',
               'td80_noprog30',
               'qian1_pdrop10', 'qian1_mdui', 'qian1_mqian', 'qian1_mstrong',
               'qian1_tddrop15', 'qian1_kunback']
    for name in combos:
        rcol = f'r_{name}'; hcol = f'h_{name}'; tcol = f't_{name}'
        sub = df_h.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean()
        win = (sub[rcol] > 0).mean() * 100
        hold = sub[hcol].mean()
        zsl = sub[sub['is_zsl']][rcol].mean() if sub['is_zsl'].sum() > 0 else float('nan')
        fake = sub[~sub['is_zsl']][rcol].mean() if (~sub['is_zsl']).sum() > 0 else float('nan')
        ret_w2 = sub[sub['seg'] == 'w2_2019'][rcol].mean() if (sub['seg'] == 'w2_2019').sum() > 0 else float('nan')
        ret_w4 = sub[sub['seg'] == 'w4_2021'][rcol].mean() if (sub['seg'] == 'w4_2021').sum() > 0 else float('nan')
        to_pct = (sub[tcol] == 'timeout').mean() * 100
        bps = ret / hold * 100 if hold > 0 else 0
        print(f'  {name:<22} {ret:>+6.2f} {win:>4.1f} {hold:>5.1f} {zsl:>+7.2f} {fake:>+6.2f} {ret_w2:>+6.2f} {ret_w4:>+6.2f} {to_pct:>8.1f} {bps:>+6.1f}')

    # 看排名 top 3
    print(f'\n## 排名 (按全期望)')
    perf = []
    for name in combos:
        rcol = f'r_{name}'; hcol = f'h_{name}'; tcol = f't_{name}'
        sub = df_h.dropna(subset=[rcol])
        if len(sub) == 0: continue
        ret = sub[rcol].mean(); hold = sub[hcol].mean()
        to_pct = (sub[tcol] == 'timeout').mean() * 100
        perf.append((name, ret, hold, to_pct))
    perf.sort(key=lambda x: x[1], reverse=True)
    for name, ret, hold, to_pct in perf[:8]:
        print(f'  {name:<22} 期望 {ret:+.2f}% 持仓 {hold:.1f} timeout {to_pct:.1f}%')


if __name__ == '__main__':
    main()
