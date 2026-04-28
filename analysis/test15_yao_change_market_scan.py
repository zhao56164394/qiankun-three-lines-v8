# -*- coding: utf-8 -*-
"""6 卦流变卦事件挖掘 (Phase A: 仅大盘 3 流)

每个变卦事件 (X→Y), 大盘 y/m/d 三流, 看事件当日所有持仓个股 N 天后收益的均值.

输出:
  168 个候选 (3 流 × 56 真变卦)
  每个候选: 事件 N 个数, alpha (vs 基线), 95% CI
  按 CI 显著性筛选 → 真买点 (CI > 0) / 真卖点 (CI < 0) / 灰区 (跨 0)
"""
import os
import sys
import io
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}
HOLD_DAYS = [1, 3, 5, 10, 20]


def load_data():
    print('=== 加载数据 ===')
    t0 = time.time()
    market = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/foundation/multi_scale_gua_daily.parquet'),
                              columns=['date', 'd_gua', 'm_gua', 'y_gua'])
    market['date'] = market['date'].astype(str)
    for c in ['d_gua', 'm_gua', 'y_gua']:
        market[c] = market[c].astype(str).str.zfill(3)
    market = market.sort_values('date').reset_index(drop=True)
    print(f'  market: {len(market)} 天, {time.time()-t0:.1f}s')

    t0 = time.time()
    stk = pd.read_parquet(os.path.join(ROOT, 'data_layer/data/stocks.parquet'),
                          columns=['date', 'code', 'close'])
    stk['date'] = stk['date'].astype(str)
    stk['code'] = stk['code'].astype(str).str.zfill(6)
    stk = stk.sort_values(['code', 'date']).reset_index(drop=True)
    print(f'  stocks: {len(stk):,} 行, {time.time()-t0:.1f}s')

    # 按 code 分组取 close 数组 + date 数组 (一次转换, 后续直接 numpy 算)
    print('  building per-code close+date arrays...')
    t0 = time.time()
    code_data = {}
    for code, g in stk.groupby('code'):
        code_data[code] = (g['date'].values, g['close'].values)
    print(f'  {len(code_data)} 只票, {time.time()-t0:.1f}s')
    return market, code_data


def find_event_dates(market, gua_col, from_g, to_g):
    """找 X→Y 切换事件的日期列表"""
    market = market.sort_values('date').reset_index(drop=True)
    market['prev'] = market[gua_col].shift(1)
    mask = (market['prev'] == from_g) & (market[gua_col] == to_g)
    return market.loc[mask, 'date'].tolist()


def compute_event_returns(event_dates, code_data, hold_days, max_per_event=200):
    """
    对每个事件日期, 取所有可买入个股 (该日有 close 的票), 算 N 天后收益.
    max_per_event: 限制每事件最大票数, 避免维度爆炸 (随机抽样)
    返回: list of returns (单只票单事件的 N 天收益%)
    """
    rng = np.random.RandomState(42)
    rets = []
    event_set = set(event_dates)

    # 每只票: 找出在事件日 i, 看 i 到 i+h 的收益
    for code, (dates, closes) in code_data.items():
        # 找出该票哪些日子是事件日
        event_idx = [i for i, d in enumerate(dates) if d in event_set]
        if not event_idx:
            continue
        for i in event_idx:
            if i + hold_days >= len(closes):
                continue
            c0 = closes[i]
            c1 = closes[i + hold_days]
            if c0 > 0:
                rets.append((c1 / c0 - 1) * 100)

    # 限制总样本 (避免每个候选 100k+ 算 bootstrap 太慢)
    if len(rets) > 50000:
        rets = rng.choice(rets, 50000, replace=False).tolist()
    return rets


def baseline_random_returns(code_data, hold_days, n_samples=10000, seed=42):
    """随机日的 N 天后收益基线 (个股池, 抽 n_samples 次 (code, day) 随机抽样)"""
    rng = np.random.RandomState(seed)
    codes = list(code_data.keys())
    rets = []
    while len(rets) < n_samples:
        code = rng.choice(codes)
        dates, closes = code_data[code]
        if len(closes) <= hold_days + 1:
            continue
        i = rng.randint(0, len(closes) - hold_days - 1)
        c0 = closes[i]; c1 = closes[i + hold_days]
        if c0 > 0:
            rets.append((c1 / c0 - 1) * 100)
    return rets


def boot_alpha_ci(event_rets, base_rets, n_boot=1000, seed=42):
    if len(event_rets) < 30:
        return None, None
    rng = np.random.RandomState(seed)
    n_event = len(event_rets); n_base = len(base_rets)
    boots = np.empty(n_boot)
    e = np.asarray(event_rets); b = np.asarray(base_rets)
    for i in range(n_boot):
        r1 = e[rng.randint(0, n_event, n_event)].mean()
        r2 = b[rng.randint(0, n_base, n_base)].mean()
        boots[i] = r1 - r2
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    market, code_data = load_data()

    # 预计算各持有期基线
    print('\n=== 计算基线 ===')
    base_by_h = {}
    for h in HOLD_DAYS:
        base_by_h[h] = baseline_random_returns(code_data, h)
        print(f'  hold={h}d: 基线均值 {np.mean(base_by_h[h]):+.2f}%, n={len(base_by_h[h])}')

    # 6 卦流 (这版只跑 3 大盘流, 个股流另外)
    streams = [
        ('y_gua', '大盘年卦'),
        ('m_gua', '大盘月卦'),
        ('d_gua', '大盘日卦'),
    ]
    rows = []

    for gua_col, stream_name in streams:
        print(f'\n=== 扫描 {stream_name} ({gua_col}) ===')
        for from_g in '000 001 010 011 100 101 110 111'.split():
            for to_g in '000 001 010 011 100 101 110 111'.split():
                if from_g == to_g:
                    continue
                events = find_event_dates(market, gua_col, from_g, to_g)
                if len(events) < 5:
                    continue
                for h in HOLD_DAYS:
                    event_rets = compute_event_returns(events, code_data, h)
                    if len(event_rets) < 30:
                        continue
                    base = base_by_h[h]
                    alpha = np.mean(event_rets) - np.mean(base)
                    ci_lo, ci_hi = boot_alpha_ci(event_rets, base)
                    if ci_lo is None:
                        continue
                    win = (np.array(event_rets) > 0).mean() * 100

                    if ci_lo > 0:
                        verdict = '★买点'
                    elif ci_hi < 0:
                        verdict = '✗卖点'
                    else:
                        verdict = '○灰区'

                    rows.append({
                        'stream': stream_name, 'gua_col': gua_col,
                        'from': from_g, 'to': to_g, 'hold': h,
                        'n_events': len(events), 'n_rets': len(event_rets),
                        'event_mean': np.mean(event_rets),
                        'alpha': alpha, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                        'win_rate': win, 'verdict': verdict,
                    })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(ROOT, 'data_layer/data/ablation/test15_yao_change_scan')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'market_scan.csv'), index=False, encoding='utf-8-sig',
              float_format='%.3f')
    print(f'\n落地: {out_dir}/market_scan.csv ({len(df)} 行)')

    # 输出 ★ 买点 / ✗ 卖点
    print('\n' + '=' * 110)
    print('# 显著真买点 (alpha CI 下限 > 0)')
    print('=' * 110)
    buy = df[df['verdict'] == '★买点'].sort_values('alpha', ascending=False)
    print(f'  {"流":<10} {"变卦":<10} {"hold":>5} {"events":>7} {"n":>7} '
          f'{"alpha%":>8} {"95%CI":>16} {"胜率%":>6}')
    print('  ' + '-' * 90)
    for _, r in buy.iterrows():
        ct = f'{r["from"]}{GUA_NAMES[r["from"]]}→{r["to"]}{GUA_NAMES[r["to"]]}'
        ci = f'[{r["ci_lo"]:+.2f},{r["ci_hi"]:+.2f}]'
        print(f"  {r['stream']:<10} {ct:<12} {r['hold']:>5}d {r['n_events']:>7} {r['n_rets']:>7} "
              f"{r['alpha']:>+7.2f} {ci:>16} {r['win_rate']:>5.1f}")

    print('\n' + '=' * 110)
    print('# 显著真卖点 (alpha CI 上限 < 0)')
    print('=' * 110)
    sell = df[df['verdict'] == '✗卖点'].sort_values('alpha')
    print(f'  {"流":<10} {"变卦":<10} {"hold":>5} {"events":>7} {"n":>7} '
          f'{"alpha%":>8} {"95%CI":>16} {"胜率%":>6}')
    print('  ' + '-' * 90)
    for _, r in sell.iterrows():
        ct = f'{r["from"]}{GUA_NAMES[r["from"]]}→{r["to"]}{GUA_NAMES[r["to"]]}'
        ci = f'[{r["ci_lo"]:+.2f},{r["ci_hi"]:+.2f}]'
        print(f"  {r['stream']:<10} {ct:<12} {r['hold']:>5}d {r['n_events']:>7} {r['n_rets']:>7} "
              f"{r['alpha']:>+7.2f} {ci:>16} {r['win_rate']:>5.1f}")

    # 汇总
    print('\n' + '=' * 110)
    print('# 汇总')
    print('=' * 110)
    print(f'  总候选: {len(df)}')
    print(f'  ★买点: {(df["verdict"]=="★买点").sum()}')
    print(f'  ✗卖点: {(df["verdict"]=="✗卖点").sum()}')
    print(f'  ○灰区: {(df["verdict"]=="○灰区").sum()}')


if __name__ == '__main__':
    main()
