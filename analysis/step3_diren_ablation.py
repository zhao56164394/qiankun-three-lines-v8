# -*- coding: utf-8 -*-
"""Step 3: 地人卦过滤 消融实验

baseline = naked cfg (di/ren 全部清空).
候选 patch = 给某分支加上 di/ren 黑白名单 (= 当前正式 cfg 的值).

LOO_i = 应用所有候选除 i (= 给 i 不加过滤, 其他都加).
判定:
  LOO_i > full → i 的过滤独立有害 (不加反而好)
  LOO_i < full → i 的过滤独立有益 (该加)

注意: di/ren 黑白名单本来在前面 stage 没接 gate 时是基于全量样本标定的,
gate 接好后剩下样本变了, 这些黑白名单可能已不再最优 → 这正是消融要回答的.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import run_ablation


candidates = [
    {'name': 'kun_exclude_ren_000_110','gua': '000', 'patch': {'kun_exclude_ren_gua': {'000','110'}}},
    {'name': 'kun_allow_di_110',      'gua': '000', 'patch': {'kun_allow_di_gua': {'110'}}},
    {'name': 'gen_allow_di_000_010',  'gua': '001', 'patch': {'gen_allow_di_gua': {'000','010'}}},
    {'name': 'xun_allow_di_010',      'gua': '011', 'patch': {'xun_allow_di_gua': {'010'}}},
    {'name': 'zhen_exclude_ren_001_011','gua': '100','patch': {'zhen_exclude_ren_gua': {'001','011'}}},
    {'name': 'dui_exclude_ren_100_110','gua': '110','patch': {'dui_exclude_ren_gua': {'100','110'}}},
    {'name': 'dui_allow_di_000_010_110','gua': '110','patch': {'dui_allow_di_gua': {'000','010','110'}}},
    {'name': 'qian_exclude_di_101_111','gua': '111','patch': {'qian_exclude_di_gua': {'101','111'}}},
]


if __name__ == '__main__':
    run_ablation(candidates, tag='step3_diren')
