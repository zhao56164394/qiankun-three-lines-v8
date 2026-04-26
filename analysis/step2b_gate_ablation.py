# -*- coding: utf-8 -*-
"""Step 2b: 年月卦 gate 消融实验

候选 = 当前所有 gate 字段 (5 个 y_gate + 2 个 ym_gate = 7 个).
每个候选 patch 设为"敲掉这条 gate" (设 set()).

LOO_i = 只保留 gate_i, 敲掉其他.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import run_ablation


candidates = [
    # 5 个 y_gua gate
    {'name': 'y_kun_101_110',  'gua': '000', 'patch': {'gate_disable_y_gua': set()}},
    {'name': 'y_gen_011_101',  'gua': '001', 'patch': {'gate_disable_y_gua': set()}},
    {'name': 'y_kan_101',      'gua': '010', 'patch': {'gate_disable_y_gua': set()}},
    {'name': 'y_xun_101',      'gua': '011', 'patch': {'gate_disable_y_gua': set()}},
    {'name': 'y_li_101_110_111','gua': '101', 'patch': {'gate_disable_y_gua': set()}},
    # 2 个 ym gate
    {'name': 'ym_dui_111x',    'gua': '110', 'patch': {'gate_disable_ym': set()}},
    {'name': 'ym_qian_2cells', 'gua': '111', 'patch': {'gate_disable_ym': set()}},
]


if __name__ == '__main__':
    run_ablation(candidates, tag='step2b_gate')
