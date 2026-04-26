# -*- coding: utf-8 -*-
"""Step 4: 买点择优 消融实验 (模式切换)

baseline = 当前 cfg (各分支自己的买点模式).
候选: 把每个分支的买点模式换成 "另一种" — 看是否更好.

第一轮: 6 个分支独立的模式切换 (qian threshold 单独做 add-one 因为是连续参数)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import run_ablation


candidates = [
    {'name': 'kun_to_cross',    'gua': '000', 'patch': {'kun_buy_mode': 'cross', 'kun_cross_threshold': 20}},
    {'name': 'gen_to_cross',    'gua': '001', 'patch': {'gen_buy_mode': 'cross', 'gen_cross_threshold': 20}},
    {'name': 'xun_to_cross',    'gua': '011', 'patch': {'xun_buy': 'cross', 'xun_buy_param': 20}},
    {'name': 'zhen_to_cross',   'gua': '100', 'patch': {'zhen_buy_mode': 'cross', 'zhen_cross_threshold': 20}},
    {'name': 'li_to_cross',     'gua': '101', 'patch': {'li_buy_mode': 'cross', 'li_cross_threshold': 20}},
    {'name': 'dui_to_double',   'gua': '110', 'patch': {'dui_buy_mode': 'double_rise'}},
]


if __name__ == '__main__':
    run_ablation(candidates, tag='step4_buy_mode')
