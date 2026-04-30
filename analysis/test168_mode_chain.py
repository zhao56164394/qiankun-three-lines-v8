# -*- coding: utf-8 -*-
"""三模式是否是同一暴涨股的时间序列?

每个 ≥+100% 暴涨股, 看它是不是经历了:
  深抛 (cur_retail<-150) → 中性 (-50~+50) → 高位 (>+100)

方法:
1. 取所有 ≥+100% 暴涨段的入场日 (NoP 触发)
2. 对每只股, 在这只股全部历史里找:
   - 模式 1 触发日 (cur_retail<-150 + NoP 条件)
   - 模式 2 触发日 (-50~+50)
   - 模式 3 触发日 (>+100)
3. 看暴涨股入场日是这只股第一次出现的模式, 还是后续模式
4. 看暴涨股之前 60d 是否有过模式 1, 那么模式 2/3 就是模式 1 的后续

输出:
  暴涨股按"入场模式"分类
  - 仅 1: 入场日属模式 1, 之前 60d 无模式 1/2/3 (新启动)
  - 1 后续 2: 入场日属模式 2, 之前 60d 内有模式 1 触发 (1 的后续)
  - 1 后续 3: 入场日属模式 3, 之前 60d 内有模式 1 触发
  - 独立 2: 入场日属模式 2, 之前 60d 内无任何 NoP 触发
  - 独立 3: 入场日属模式 3, 之前 60d 内无任何 NoP 触发
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_nopool(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        last_mf = np.nan; last_retail = np.nan
        last_trigger = -999
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if mf_rising and retail_rising and trend_ok and (i - last_trigger) >= 30:
                events.append({
                    'date':date[gi],'code':code[gi],
                    'buy_idx_global':gi,'i_in_code':i,
                    'cur_retail':retail[gi],
                })
                last_trigger = i
            last_mf = mf[gi]; last_retail = retail[gi]
    return pd.DataFrame(events)


def simulate_t0(buy_idx, td, close, mf, retail, max_end):
    bp = close[buy_idx]; cum_mult = 1.0; holding = True
    cur_buy_price = bp
    for k in range(buy_idx + 1, max_end + 1):
        if not np.isnan(td[k]) and td[k] < 11:
            if holding: cum_mult *= close[k] / cur_buy_price
            return (cum_mult-1)*100
        if k < 1: continue
        mf_c = mf[k] - mf[k-1] if not np.isnan(mf[k-1]) else 0
        ret_c = retail[k] - retail[k-1] if not np.isnan(retail[k-1]) else 0
        td_c = td[k] - td[k-1] if not np.isnan(td[k-1]) else 0
        if holding:
            if mf_c < 0 and ret_c < 0 and td_c < 0:
                cum_mult *= close[k] / cur_buy_price
                holding = False
        else:
            if mf_c > 0:
                cur_buy_price = close[k]; holding = True
    if holding: cum_mult *= close[max_end] / cur_buy_price
    return (cum_mult-1)*100


def classify_mode(cur_retail):
    if cur_retail < -150:
        return 'M1'
    elif cur_retail < 50:
        return 'M2'
    else:
        return 'M3'


def main():
    t0 = time.time()
    print('=== test168: 三模式时序关系 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','d_trend']).reset_index(drop=True)

    code_arr = df['code'].to_numpy()
    date_arr = df['date'].to_numpy()
    close_arr = df['close'].to_numpy().astype(np.float64)
    retail_arr = df['retail'].to_numpy().astype(np.float64)
    mf_arr = df['main_force'].to_numpy().astype(np.float64)
    trend_arr = df['d_trend'].to_numpy().astype(np.float64)
    code_change = np.r_[True, code_arr[1:] != code_arr[:-1]]
    code_starts = np.where(code_change)[0]
    code_ends = np.r_[code_starts[1:], len(code_arr)]

    arrays = {'code':code_arr,'date':date_arr,'retail':retail_arr,'mf':mf_arr,'td':trend_arr,
              'starts':code_starts,'ends':code_ends}

    print('  生成 NoP 触发...')
    df_e = find_signals_nopool(arrays)
    print(f'    {len(df_e):,} 事件')

    print('  计算 ret...')
    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets
    df_e['mode'] = df_e['cur_retail'].apply(classify_mode)
    df_e['gi'] = df_e['buy_idx_global'].astype(int)

    # ===== 取 ≥+100% 暴涨股 =====
    baggers = df_e[df_e['ret_pct']>=100].copy()
    print(f'\n  ≥+100% 暴涨股: {len(baggers)}')
    print(f'  按入场模式分布:')
    for m, g in baggers.groupby('mode'):
        print(f'    {m}: {len(g)} ({len(g)/len(baggers)*100:.1f}%)')

    # ===== 关键: 对每只暴涨股, 看入场前 60d 是否有过其他 NoP 触发 =====
    print(f'\n{"="*82}')
    print(f'  暴涨股入场前 60d / 30d 内是否有更早的 NoP 触发')
    print(f'{"="*82}')

    # 把所有 NoP 事件按 (code, gi) 索引
    df_e_sorted = df_e.sort_values(['code', 'gi']).reset_index(drop=True)
    by_code = {c: g.reset_index(drop=True) for c, g in df_e_sorted.groupby('code')}

    classify_results = []
    for _, b in baggers.iterrows():
        code = b['code']; gi = b['gi']; b_mode = b['mode']
        same_code = by_code.get(code)
        if same_code is None:
            classify_results.append({'mode': b_mode, 'class': '独立', 'prior_60d': None})
            continue
        # 找 60 日内 (gi - 60 ≤ x_gi < gi) 的前置触发
        prior = same_code[(same_code['gi'] >= gi - 60) & (same_code['gi'] < gi)]
        if len(prior) == 0:
            classify_results.append({'mode': b_mode, 'class': '独立', 'prior_60d': '无'})
        else:
            # 看最近的前置模式
            last_prior = prior.iloc[-1]
            prior_modes = '|'.join(prior['mode'].unique())
            classify_results.append({
                'mode': b_mode, 'class': '后续',
                'prior_60d': prior_modes,
                'prior_n': len(prior),
                'last_prior_mode': last_prior['mode'],
                'days_since_last': gi - int(last_prior['gi']),
            })
    df_c = pd.DataFrame(classify_results)
    baggers = baggers.reset_index(drop=True)
    baggers['class'] = df_c['class']
    baggers['prior_60d'] = df_c['prior_60d']
    baggers['last_prior_mode'] = df_c.get('last_prior_mode')

    print(f'\n  --- 暴涨股: 独立启动 vs 后续触发 ---')
    print(f'  {"入场模式":<10} {"独立 (60d 内无前置)":>20} {"后续 (60d 内有前置)":>22} {"总":>6}')
    for m in ['M1', 'M2', 'M3']:
        sub = baggers[baggers['mode']==m]
        n_indep = (sub['class']=='独立').sum()
        n_follow = (sub['class']=='后续').sum()
        total = len(sub)
        print(f'  {m:<10} {n_indep:>10} ({n_indep/total*100:>4.1f}%)    '
              f'{n_follow:>10} ({n_follow/total*100:>4.1f}%)    {total:>6}')

    # ===== 后续模式的"前置"是什么模式? =====
    print(f'\n{"="*82}')
    print(f'  对于"后续"暴涨股, 60d 内最近的前置触发是什么模式?')
    print(f'{"="*82}')
    print(f'\n  {"入场模式":<10} {"前置 M1":>10} {"前置 M2":>10} {"前置 M3":>10} {"总后续":>8}')
    for m in ['M1', 'M2', 'M3']:
        sub = baggers[(baggers['mode']==m) & (baggers['class']=='后续')]
        if len(sub) == 0: continue
        n_pm1 = (sub['last_prior_mode']=='M1').sum()
        n_pm2 = (sub['last_prior_mode']=='M2').sum()
        n_pm3 = (sub['last_prior_mode']=='M3').sum()
        total = len(sub)
        print(f'  {m:<10} {n_pm1:>5} ({n_pm1/total*100:>4.1f}%) '
              f'{n_pm2:>5} ({n_pm2/total*100:>4.1f}%) '
              f'{n_pm3:>5} ({n_pm3/total*100:>4.1f}%) {total:>8}')

    # ===== 关键问题: M2/M3 暴涨股, 60d 内是否有过 M1 触发? =====
    print(f'\n{"="*82}')
    print(f'  M2/M3 暴涨股, 60d 内是否有过 M1 (深抛) 触发?')
    print(f'{"="*82}')
    print(f'\n  {"入场模式":<10} {"60d 内有过 M1":>18} {"60d 内无 M1":>14} {"总":>6}')
    for m in ['M2', 'M3']:
        sub = baggers[baggers['mode']==m]
        if len(sub) == 0: continue
        # 60d 前置中包含 M1?
        had_m1 = sub['prior_60d'].fillna('').str.contains('M1', regex=False)
        n_had = had_m1.sum()
        n_not = (~had_m1).sum()
        total = len(sub)
        print(f'  {m:<10} {n_had:>10} ({n_had/total*100:>4.1f}%)    '
              f'{n_not:>10} ({n_not/total*100:>4.1f}%)    {total:>6}')

    # ===== 拓宽到 120d 看 =====
    print(f'\n{"="*82}')
    print(f'  M2/M3 暴涨股, 拓宽到 120d 内是否有过 M1?')
    print(f'{"="*82}')
    classify_120 = []
    for _, b in baggers.iterrows():
        code = b['code']; gi = b['gi']
        same_code = by_code.get(code)
        if same_code is None:
            classify_120.append({'had_m1_120d': False}); continue
        prior = same_code[(same_code['gi'] >= gi - 120) & (same_code['gi'] < gi)]
        had_m1 = (prior['mode']=='M1').any() if len(prior) > 0 else False
        classify_120.append({'had_m1_120d': had_m1})
    df_c120 = pd.DataFrame(classify_120)
    baggers['had_m1_120d'] = df_c120['had_m1_120d']

    print(f'\n  {"入场模式":<10} {"120d 内有 M1":>16} {"120d 内无 M1":>14} {"总":>6}')
    for m in ['M2', 'M3']:
        sub = baggers[baggers['mode']==m]
        n_had = sub['had_m1_120d'].sum()
        n_not = (~sub['had_m1_120d']).sum()
        total = len(sub)
        print(f'  {m:<10} {n_had:>10} ({n_had/total*100:>4.1f}%)    '
              f'{n_not:>10} ({n_not/total*100:>4.1f}%)    {total:>6}')

    # ===== 拓宽到 240d 看 =====
    print(f'\n{"="*82}')
    print(f'  M2/M3 暴涨股, 拓宽到 240d 内是否有过 M1?')
    print(f'{"="*82}')
    classify_240 = []
    for _, b in baggers.iterrows():
        code = b['code']; gi = b['gi']
        same_code = by_code.get(code)
        if same_code is None:
            classify_240.append({'had_m1_240d': False}); continue
        prior = same_code[(same_code['gi'] >= gi - 240) & (same_code['gi'] < gi)]
        had_m1 = (prior['mode']=='M1').any() if len(prior) > 0 else False
        classify_240.append({'had_m1_240d': had_m1})
    df_c240 = pd.DataFrame(classify_240)
    baggers['had_m1_240d'] = df_c240['had_m1_240d']
    print(f'\n  {"入场模式":<10} {"240d 内有 M1":>16} {"240d 内无 M1":>14} {"总":>6}')
    for m in ['M2', 'M3']:
        sub = baggers[baggers['mode']==m]
        n_had = sub['had_m1_240d'].sum()
        n_not = (~sub['had_m1_240d']).sum()
        total = len(sub)
        print(f'  {m:<10} {n_had:>10} ({n_had/total*100:>4.1f}%)    '
              f'{n_not:>10} ({n_not/total*100:>4.1f}%)    {total:>6}')

    # ===== 看 retail 整段轨迹: M2/M3 暴涨股入场前的 retail 走势 =====
    print(f'\n{"="*82}')
    print(f'  M2/M3 暴涨股 入场前 60/120/240d retail 最低值分布')
    print(f'{"="*82}')

    print(f'\n  {"入场模式":<10} {"窗口":<8} {"retail最低":<14} {"<-250%":>8} {"<-150%":>8} {"<-50%":>8}')
    for m in ['M2', 'M3']:
        sub = baggers[baggers['mode']==m]
        for window in [60, 120, 240]:
            min_retails = []
            for _, b in sub.iterrows():
                gi = int(b['gi'])
                # 找 code 边界
                ci = np.searchsorted(code_starts, gi, side='right') - 1
                start_i = max(code_starts[ci], gi - window)
                seg = retail_arr[start_i:gi+1]
                seg_v = seg[~np.isnan(seg)]
                if len(seg_v) > 0:
                    min_retails.append(np.min(seg_v))
            if len(min_retails) == 0: continue
            mr = np.array(min_retails)
            p_under_250 = (mr < -250).mean() * 100
            p_under_150 = (mr < -150).mean() * 100
            p_under_50 = (mr < -50).mean() * 100
            med = np.median(mr)
            print(f'  {m:<10} {window:>3}d   med={med:>+5.0f}      {p_under_250:>5.1f}%   '
                  f'{p_under_150:>5.1f}%   {p_under_50:>5.1f}%')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
