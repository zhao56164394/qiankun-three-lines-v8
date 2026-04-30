# -*- coding: utf-8 -*-
"""v5 + 坤 regime — K=1~2 暴涨股识别能力评估

评判标准换: 不再看 avg ret, 看
  - 命中 +50% (≥1.5x)
  - 命中 +100% (≥2x)
  - 命中 +200% (≥3x)
  - 命中率 (这些笔占比)
  - 期望值 (avg ret 对比)

每个因子 (池深 / cur_mf / cur_retail / ret_5d / cur_trend) 排序后取
  - 横向 top-1 / top-2 (每天)
  - 阈值排雷
看暴涨股命中能力.
"""
import os, sys, io, time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_TRACK = 365
LOOKBACK = 30


def find_signals_v5(arrays):
    cs = arrays['starts']; ce = arrays['ends']
    retail = arrays['retail']; mf = arrays['mf']; td = arrays['td']
    date = arrays['date']; code = arrays['code']
    events = []
    for ci in range(len(cs)):
        s = cs[ci]; e = ce[ci]
        if e - s < LOOKBACK + MAX_TRACK + 5: continue
        n = e - s
        in_pool = False; prev_below = False
        last_mf = np.nan; last_retail = np.nan
        pool_min_retail = np.inf
        for i in range(LOOKBACK, n - MAX_TRACK - 1):
            gi = s + i
            cur_below = retail[gi] < -250
            if not in_pool and cur_below and not prev_below:
                in_pool = True; pool_min_retail = retail[gi]
            if in_pool and retail[gi] < pool_min_retail:
                pool_min_retail = retail[gi]
            mf_rising = (not np.isnan(last_mf)) and (mf[gi] > last_mf)
            retail_rising = (not np.isnan(last_retail)) and (retail[gi] > last_retail)
            trend_ok = (not np.isnan(td[gi])) and (td[gi] > 11)
            if in_pool and mf_rising and retail_rising and trend_ok:
                ci_s = arrays['starts'][ci]
                i5 = max(gi - 5, ci_s)
                i30 = max(gi - 30, ci_s)
                ret_5d_v = retail[gi] - retail[i5] if not np.isnan(retail[i5]) else np.nan
                mf_5d_v = mf[gi] - mf[i5] if not np.isnan(mf[i5]) else np.nan
                events.append({'date':date[gi],'code':code[gi],
                               'buy_idx_global':gi,
                               'pool_min_retail':pool_min_retail,
                               'cur_mf':mf[gi],
                               'cur_retail':retail[gi],
                               'cur_trend':td[gi],
                               'ret_5d':ret_5d_v,
                               'mf_5d':mf_5d_v})
                in_pool = False
            last_mf = mf[gi]; last_retail = retail[gi]
            prev_below = cur_below
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


def stat_picks(df_picks, label):
    """返回选股列的暴涨股命中统计"""
    n = len(df_picks)
    avg = df_picks['ret_pct'].mean()
    win = (df_picks['ret_pct']>0).mean()*100
    h50 = (df_picks['ret_pct']>=50).sum()
    h100 = (df_picks['ret_pct']>=100).sum()
    h200 = (df_picks['ret_pct']>=200).sum()
    h500 = (df_picks['ret_pct']>=500).sum()
    return {
        'label': label, 'n': n, 'avg': avg, 'win': win,
        'h50': h50, 'h100': h100, 'h200': h200, 'h500': h500,
        'r50': h50/n*100 if n else 0,
        'r100': h100/n*100 if n else 0,
        'r200': h200/n*100 if n else 0,
        'r500': h500/n*100 if n else 0,
    }


def main():
    t0 = time.time()
    print('=== test164: K=1~2 暴涨股识别能力 ===\n')

    uni = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/main_board_universe.parquet'),
                          columns=['code', 'board'])
    uni['code'] = uni['code'].astype(str).str.zfill(6)
    main_codes = set(uni[uni['board'] == '主板']['code'].unique())

    g = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/stock_multi_scale_gua_daily.parquet'),
                        columns=['date', 'code', 'd_gua', 'd_trend'])
    g['date'] = g['date'].astype(str); g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[g['code'].isin(main_codes)].reset_index(drop=True)
    g['d_gua'] = g['d_gua'].astype(str).str.zfill(3).replace({'nan':''})
    g.rename(columns={'d_gua':'stk_d'}, inplace=True)

    mkt = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                          columns=['date', 'y_gua'])
    mkt['date'] = mkt['date'].astype(str)
    mkt['mkt_y'] = mkt['y_gua'].astype(str).str.zfill(3).replace({'nan':''})
    mkt = mkt[['date','mkt_y']].drop_duplicates('date')

    p = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                        columns=['date', 'code', 'close', 'retail', 'main_force'])
    p['date'] = p['date'].astype(str); p['code'] = p['code'].astype(str).str.zfill(6)
    p = p[p['code'].isin(main_codes)].reset_index(drop=True)

    df = g.merge(p, on=['date','code'], how='inner').merge(mkt, on='date', how='left')
    df = df.sort_values(['code','date']).reset_index(drop=True)
    df = df.dropna(subset=['close','stk_d','d_trend','mkt_y']).reset_index(drop=True)
    df = df[df['mkt_y'] == '000'].reset_index(drop=True)

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
    df_e = find_signals_v5(arrays)
    print(f'  v5 入场事件: {len(df_e):,}')

    rets = []
    for _, ev in df_e.iterrows():
        gi = int(ev['buy_idx_global'])
        ci = np.searchsorted(code_starts, gi, side='right') - 1
        e = code_ends[ci]
        max_end = min(e - 1, gi + MAX_TRACK)
        rets.append(simulate_t0(gi, trend_arr, close_arr, mf_arr, retail_arr, max_end))
    df_e['ret_pct'] = rets
    df_e['year'] = df_e['date'].str[:4]

    # ===== Baseline 全样本暴涨股分布 =====
    print(f'\n{"="*82}')
    print(f'  Baseline (全部 v5 入场) 暴涨股分布')
    print(f'{"="*82}')

    s_base = stat_picks(df_e, 'baseline')
    print(f'\n  n={s_base["n"]:,}  avg={s_base["avg"]:+.2f}%  win={s_base["win"]:.1f}%')
    print(f'  暴涨股: ≥+50%={s_base["h50"]} ({s_base["r50"]:.1f}%)  '
          f'≥+100%={s_base["h100"]} ({s_base["r100"]:.2f}%)  '
          f'≥+200%={s_base["h200"]} ({s_base["r200"]:.2f}%)  '
          f'≥+500%={s_base["h500"]} ({s_base["r500"]:.2f}%)')

    # 按年看暴涨股集中度
    print(f'\n  按年暴涨股分布 (≥+100%):')
    for y, sub in df_e.groupby('year'):
        h100 = (sub['ret_pct']>=100).sum()
        h200 = (sub['ret_pct']>=200).sum()
        print(f'    {y}: n={len(sub):>4}, ≥+100%={h100} ({h100/len(sub)*100:.2f}%), '
              f'≥+200%={h200} ({h200/len(sub)*100:.2f}%)')

    # ===== 各因子 横向 top-K (每天 1 只 / 2 只) =====
    print(f'\n{"="*82}')
    print(f'  各因子 横向 top-K 暴涨股命中')
    print(f'{"="*82}')

    factors = [
        ('pool_min_retail', True,  '池深 ↑ (深→浅)'),
        ('pool_min_retail', False, '池深 ↓ (浅→深)'),
        ('cur_mf',          True,  'cur_mf ↑'),
        ('cur_mf',          False, 'cur_mf ↓'),
        ('cur_retail',      True,  'cur_retail ↑'),
        ('cur_retail',      False, 'cur_retail ↓'),
        ('ret_5d',          True,  'ret_5d ↑'),
        ('ret_5d',          False, 'ret_5d ↓'),
        ('mf_5d',           True,  'mf_5d ↑'),
        ('mf_5d',           False, 'mf_5d ↓'),
        ('cur_trend',       True,  'cur_trend ↑'),
        ('cur_trend',       False, 'cur_trend ↓'),
    ]

    for K in [1, 2]:
        print(f'\n  --- 每日选 top-{K} ---')
        print(f'\n  {"factor":<22} {"n":>5} {"avg":>8} {"win":>6} '
              f'{"≥50":>5} {"≥100":>5} {"≥200":>5} {"≥500":>4} {"r100":>7} {"r200":>7}')
        for col, asc, label in factors:
            df_picks = df_e.sort_values(['date', col, 'code'], ascending=[True, asc, True]).groupby('date').head(K)
            s = stat_picks(df_picks, label)
            print(f'  {label:<22} {s["n"]:>5} {s["avg"]:>+7.2f}% {s["win"]:>5.1f}% '
                  f'{s["h50"]:>5} {s["h100"]:>5} {s["h200"]:>5} {s["h500"]:>4} '
                  f'{s["r100"]:>+6.2f}% {s["r200"]:>+6.2f}%')

    # ===== 阈值排雷 + top-K =====
    print(f'\n{"="*82}')
    print(f'  最强组合: cur_mf <= 阈值 (排雷) + 池深 ↑ top-1 / top-2')
    print(f'{"="*82}')

    print(f'\n  {"阈值":<28} {"n":>5} {"avg":>8} {"win":>6} '
          f'{"≥50":>5} {"≥100":>5} {"≥200":>5} {"≥500":>4} {"r100":>7} {"r200":>7}')
    for K in [1, 2]:
        print(f'  --- top-{K} ---')
        for thr in [-200, -100, -50, 0, 50]:
            df_filt = df_e[df_e['cur_mf'] <= thr]
            if len(df_filt) < 50: continue
            df_picks = df_filt.sort_values(['date', 'pool_min_retail', 'code'],
                                             ascending=[True, True, True]).groupby('date').head(K)
            s = stat_picks(df_picks, f'cur_mf<={thr}, 池深↑ top-{K}')
            print(f'  {s["label"]:<28} {s["n"]:>5} {s["avg"]:>+7.2f}% {s["win"]:>5.1f}% '
                  f'{s["h50"]:>5} {s["h100"]:>5} {s["h200"]:>5} {s["h500"]:>4} '
                  f'{s["r100"]:>+6.2f}% {s["r200"]:>+6.2f}%')

    # ===== 显示具体暴涨股 =====
    print(f'\n{"="*82}')
    print(f'  ≥+200% 暴涨股清单 (按 ret 倒序)')
    print(f'{"="*82}')
    bigwins = df_e[df_e['ret_pct'] >= 200].sort_values('ret_pct', ascending=False)
    print(f'\n  共 {len(bigwins)} 只 ≥+200%')
    print(f'  {"日期":<12} {"代码":<8} {"ret":>10} {"池深":>8} {"cur_mf":>8} {"cur_retail":>10} {"ret_5d":>8} {"cur_trend":>9}')
    for _, r in bigwins.head(20).iterrows():
        print(f'  {r["date"]:<12} {r["code"]:<8} {r["ret_pct"]:>+9.1f}% '
              f'{r["pool_min_retail"]:>+7.0f} {r["cur_mf"]:>+7.0f} {r["cur_retail"]:>+9.0f} '
              f'{r["ret_5d"]:>+7.0f} {r["cur_trend"]:>+8.1f}')

    print(f'\n=== 完成, {time.time()-t0:.1f}s ===')


if __name__ == '__main__':
    main()
