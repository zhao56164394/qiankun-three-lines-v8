# -*- coding: utf-8 -*-
"""个股日卦变爻细分 — 按 哪位变 分析

读 test18 已扫的 full_scan.csv, 重新按"变爻位置"聚合:
  - 仅天位变 (位[0]): 大趋势转折 (例: 011→111 = 天位由 0→1, 牛/熊切换)
  - 仅人位变 (位[1]): 中波转折
  - 仅地位变 (位[2]): 短反弹/回调
  - 多位同变: 系统性

按类型分组, 看哪类变爻是真买点 / 真卖点
"""
import os
import sys
import io
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUA_NAMES = {'000': '坤', '001': '艮', '010': '坎', '011': '巽',
             '100': '震', '101': '离', '110': '兑', '111': '乾'}

# 三爻命名: index 0 = 天 (长期), 1 = 人 (中期), 2 = 地 (短期)
POS_NAMES = {0: '天位(长期)', 1: '人位(中期)', 2: '地位(短期)'}


def classify_change(f, t):
    """X→Y 变爻分类. 返回 (changed_positions list, change_type str)"""
    diffs = [i for i in range(3) if f[i] != t[i]]
    if len(diffs) == 1:
        return diffs, f'单变-{POS_NAMES[diffs[0]]}'
    elif len(diffs) == 2:
        names = '+'.join([POS_NAMES[i].split('(')[0] for i in diffs])
        return diffs, f'双变-{names}'
    else:  # 3 位都变
        return diffs, '三爻齐变'


def main():
    # 读 test18 已扫数据
    src = os.path.join(ROOT, 'data_layer/data/ablation/test18_yao_change_fast/full_scan.csv')
    df = pd.read_csv(src, encoding='utf-8-sig')

    # 只看 个股日卦 (买卖点核心)
    df_d = df[df['stream'] == '个股日卦'].copy()
    print(f'个股日卦记录: {len(df_d)} 行 ({df_d["from"].nunique()} from × ... × 5 hold)')

    # zfill 保证 from/to 是 3 位
    df_d['from'] = df_d['from'].astype(str).str.zfill(3)
    df_d['to'] = df_d['to'].astype(str).str.zfill(3)

    # 分类
    df_d['change_type'] = df_d.apply(lambda r: classify_change(r['from'], r['to'])[1], axis=1)
    df_d['n_diff'] = df_d.apply(lambda r: len(classify_change(r['from'], r['to'])[0]), axis=1)

    # === 聚合: 按变爻类型分桶 ===
    print('\n## 个股日卦变爻类型分布')
    # 按 from-to 唯一组合 (去重 hold)
    unique_changes = df_d[['from', 'to', 'change_type', 'n_diff']].drop_duplicates()
    print(unique_changes.groupby('change_type').size().to_string())

    # === 单爻变各位的 alpha 模式 ===
    for n_diff_filter in [1, 2, 3]:
        print(f'\n{"=" * 100}')
        if n_diff_filter == 1:
            print(f'# 单爻变 (天位/人位/地位 三类各 8 候选)')
        elif n_diff_filter == 2:
            print(f'# 双爻变 (天人/天地/人地 三类)')
        else:
            print(f'# 三爻齐变 (1 类)')
        print(f'{"=" * 100}')

        sub = df_d[df_d['n_diff'] == n_diff_filter].copy()

        # 跨期一致性: 同一 (from, to) 至少 4/5 同向
        groups = sub.groupby(['from', 'to', 'change_type'])
        rows = []
        for (f, t, ct), g in groups:
            if len(g) < 4: continue
            n_buy = (g['verdict'] == '★买点').sum()
            n_sell = (g['verdict'] == '✗卖点').sum()
            avg_alpha = g['alpha'].mean()
            n_events = g.iloc[0]['n_events']
            rows.append({
                'from': f, 'to': t, 'change_type': ct,
                'n_buy_holds': n_buy, 'n_sell_holds': n_sell,
                'avg_alpha': avg_alpha, 'n_events': n_events,
                'side': '★买' if n_buy >= 4 else ('✗卖' if n_sell >= 4 else '○混乱'),
            })
        rdf = pd.DataFrame(rows).sort_values('avg_alpha', ascending=False)

        # 显示
        print(f'\n  {"变卦":<10} {"类型":<22} {"events":>8} {"4/5买":>5} {"4/5卖":>5} {"均α%":>7}  {"判定":>6}')
        print('  ' + '-' * 80)
        for _, r in rdf.iterrows():
            ct_short = r['change_type'].split('-', 1)[1] if '-' in r['change_type'] else r['change_type']
            arrow = f'{r["from"]}{GUA_NAMES[r["from"]]}→{r["to"]}{GUA_NAMES[r["to"]]}'
            print(f'  {arrow:<12} {ct_short:<22} {int(r["n_events"]):>8} '
                  f'{int(r["n_buy_holds"]):>4}/5 {int(r["n_sell_holds"]):>4}/5 '
                  f'{r["avg_alpha"]:>+6.2f}%  {r["side"]:>4}')

    # === 各变爻位置 总体 alpha 倾向 ===
    print(f'\n\n{"=" * 100}')
    print('# 各变爻"位置"的总体 alpha 倾向 (按位置类型聚合所有变卦, 看共性)')
    print('=' * 100)
    print(f'  {"变爻位置":<26} {"候选数":>6} {"事件总":>10} {"均α":>7} {"★买":>4} {"✗卖":>4} {"○混":>4}')
    print('  ' + '-' * 75)
    for ct in df_d['change_type'].unique():
        sub = df_d[df_d['change_type'] == ct]
        # 按变卦聚合
        groups = sub.groupby(['from', 'to'])
        n_buy = 0; n_sell = 0; n_mix = 0
        events_total = 0
        alphas = []
        for (f, t), g in groups:
            if len(g) < 4: continue
            n_b = (g['verdict'] == '★买点').sum()
            n_s = (g['verdict'] == '✗卖点').sum()
            avg_a = g['alpha'].mean()
            alphas.append(avg_a)
            events_total += g.iloc[0]['n_events']
            if n_b >= 4: n_buy += 1
            elif n_s >= 4: n_sell += 1
            else: n_mix += 1
        if not alphas: continue
        print(f'  {ct:<26} {len(alphas):>6} {events_total:>10,} '
              f'{np.mean(alphas):>+6.2f}% {n_buy:>4} {n_sell:>4} {n_mix:>4}')


if __name__ == '__main__':
    main()
