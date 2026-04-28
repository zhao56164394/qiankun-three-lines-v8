# -*- coding: utf-8 -*-
"""池深 ablation 结果重新输出 (按 strategy-ablation skill 标准三视角表)"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'data_layer/data/ablation/test6_pool_depth/case_summary.csv'))

# y_gua 必须保持 3 位字符串
df['y_gua'] = df['y_gua'].astype(str).str.zfill(3)
df['target_y_gua'] = df['target_y_gua'].astype(str).str.zfill(3) if df['target_y_gua'].dtype != 'O' else df['target_y_gua']

GUA_NAMES = {'000':'坤','001':'艮','010':'坎','011':'巽',
             '100':'震','101':'离','110':'兑','111':'乾'}

# 定位 baseline 的桶内基准 (每桶 baseline IS / OOS sig_n, trd_利)
def baseline_row(phase, y):
    return df[(df['case_id']=='baseline') & (df['phase']==phase) & (df['y_gua']==y)].iloc[0]

print('=' * 130)
print('Phase 3 池深 ablation 三视角分析 (按 strategy-ablation skill 标准)')
print('=' * 130)
print('视角:')
print('  视角 1 (sig 全量): 通过 cfg 过滤的潜在信号. sig_n / sig_mean% / 95% CI')
print('  视角 2 (trd 实买): 经 max_pos=5 + 排序后实买. trd_n / trd_利万 / trd_mean%')
print('  视角 3 (综合 verdict): 按 SKILL.md 矩阵给判定')
print()
print('judgement matrix:')
print('  sig_n>=20 + CI 全负 + trd_n>=5 + trd_利<0  → ✗ 真有害')
print('  sig_n>=20 + CI 全正                       → ★ 真有益')
print('  sig_n>=20 + CI 跨0 + trd_n>=8 + trd_利>+10 → ★ 实战有益')

for phase in ['IS', 'OOS']:
    print('\n' + '#' * 130)
    print(f'# {phase} 段')
    print('#' * 130)

    sub_phase = df[df['phase']==phase]
    base_total = sub_phase[sub_phase['case_id']=='baseline'].iloc[0]['final_capital_wan']
    print(f'\n[baseline] {phase} 总终值: {base_total:.1f} 万 (无 pool_depth_tiers, 全 8 桶 sig 总和)')
    print()

    # 8 桶 baseline 行
    print('=' * 130)
    print(f'baseline 各 y_gua 桶基准 (test6 真裸基线, {phase} 段)')
    print('=' * 130)
    print(f'  {"y_gua":<10} {"sig_n":>6} {"sig_mean%":>10} {"CI 95%":>16} '
          f'{"trd_n":>6} {"trd_利万":>9} {"trd_mean%":>10} {"verdict":<14}')
    print('  ' + '-' * 95)
    for y in ['000','001','010','011','100','101','110','111']:
        r = baseline_row(phase, y)
        ci = f'[{r["ci_lo"]:>+5.1f},{r["ci_hi"]:>+5.1f}]' if pd.notna(r['ci_lo']) else '   -    '
        print(f'  {y} {GUA_NAMES[y]:<5} {int(r["sig_n"]):>6d} {r["sig_mean%"]:>+9.2f}% '
              f'{ci:>16} {int(r["trd_n"]):>6d} {r["trd_利万"]:>+9.2f} '
              f'{r["trd_mean%"]:>+9.2f}% {r["verdict"]:<14}')

    # 每个目标桶的三个候选
    print('\n' + '=' * 130)
    print(f'各目标桶 × 候选档位的 ablation 结果 ({phase} 段)')
    print('=' * 130)

    for tg in ['000','001','010','011','100','101','110','111']:
        # 该桶 baseline
        b = baseline_row(phase, tg)
        b_sig_n = int(b['sig_n'])
        b_trd_li = b['trd_利万']

        print(f'\n  [{tg} {GUA_NAMES[tg]}] baseline 桶内: sig_n={b_sig_n}, '
              f'sig_mean={b["sig_mean%"]:+.2f}%, '
              f'trd_n={int(b["trd_n"])}, trd_利={b_trd_li:+.1f}万 ({b["verdict"]})')
        print(f'  {"档位":<22} {"sig_n":>6} {"sig_mean%":>10} {"CI 95%":>16} '
              f'{"trd_n":>5} {"trd_利万":>8} {"trd_利变":>9} {"verdict":<14} {"总终值":>7} {"终值变":>8}')
        print('  ' + '-' * 120)
        for cand in ['A_d400', 'B_d300', 'C_d200']:
            cid = f'{tg}_{cand}'
            cs = sub_phase[(sub_phase['case_id']==cid) & (sub_phase['y_gua']==tg)]
            if len(cs) == 0:
                continue
            r = cs.iloc[0]
            cap = r['final_capital_wan']
            d_cap = cap - base_total
            d_li = r['trd_利万'] - b_trd_li
            ci = f'[{r["ci_lo"]:>+5.1f},{r["ci_hi"]:>+5.1f}]' if pd.notna(r['ci_lo']) else '   -    '
            print(f'  {r["cand_desc"]:<22} {int(r["sig_n"]):>6d} '
                  f'{r["sig_mean%"]:>+9.2f}% {ci:>16} '
                  f'{int(r["trd_n"]):>5d} {r["trd_利万"]:>+8.2f} {d_li:>+8.2f} '
                  f'{r["verdict"]:<14} {cap:>6.1f}万 {d_cap:>+7.1f}万')
print()
