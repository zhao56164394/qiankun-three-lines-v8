# -*- coding: utf-8 -*-
"""Step 5: 卖点择优 消融实验

baseline = naked cfg (全 bear 卖法).
候选 patch = 给某分支配上专属卖法.
LOO_i = 应用所有候选除 i (只让 i 维持 bear).

判定语义:
  full = 全部分支用专属卖法 (= 当前正式 cfg 的卖法)
  LOO_i = 除 i 之外都用专属卖法
  → 如果 LOO_i > full, 说明 i 的专属卖法独立有害 (改回 bear 反而好)
  → 如果 LOO_i < full, 说明 i 的专属卖法独立有益 (该保留)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import run_ablation


candidates = [
    {'name': 'kun_uses_kun_bear', 'gua': '000', 'patch': {'sell': 'kun_bear'}},
    {'name': 'zhen_uses_bull',    'gua': '100', 'patch': {'sell': 'bull'}},
    {'name': 'dui_uses_dui_bear', 'gua': '110', 'patch': {'sell': 'dui_bear'}},
    {'name': 'qian_uses_qian_bull','gua': '111','patch': {'sell': 'qian_bull'}},
]


if __name__ == '__main__':
    run_ablation(candidates, tag='step5_sell')
