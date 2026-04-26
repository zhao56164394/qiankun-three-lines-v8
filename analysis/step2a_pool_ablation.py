# -*- coding: utf-8 -*-
"""Step 2a: 池深+池天 消融实验

验证现有 pool_depth_tiers / pool_days_min/max 配置是否每条都独立有效.

候选 (敲掉这条配置, 看总收益怎么变):
  A: 000 坤的 days_exclude=[4,10] 死区   → 替换为 tiers=[{depth_max:None, days_min:0, days_max:None}]
  B: 100 震的 pool_days_min=1, pool_days_max=7 → 设 None
  C: 101 离的 tier1 (深池 ≤-350 拒绝) → 改成接受
  D: 101 离的 tier2 days_max=15 → 设 None (只保留深池拒绝)

判定: full=当前配置, leave-one-out=敲掉一个候选
  如果 LOO_A < full → A 该保留 (敲掉变差)
  如果 LOO_A > full → A 该去掉 (敲掉变好)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.ablation import run_ablation


candidates = [
    {
        'name': 'kun_days_exclude_4_10',
        'gua': '000',
        'patch': {
            # 敲掉死区: 改为无 exclude 的 tier
            'pool_depth_tiers': [
                {'depth_max': None, 'days_min': 0, 'days_max': None}
            ],
        },
    },
    {
        'name': 'zhen_pool_days_1_7',
        'gua': '100',
        'patch': {
            'pool_days_min': None,
            'pool_days_max': None,
        },
    },
    {
        'name': 'li_tier1_deep_reject',
        'gua': '101',
        'patch': {
            # 敲掉深池拒绝, 保留 tier2 浅池 0-15 窗口
            'pool_depth_tiers': [
                # 改 tier1 为接受任何池天
                {'depth_max': -350, 'days_min': 0, 'days_max': None},
                {'depth_max': -250, 'days_min': 0, 'days_max': 15},
            ],
        },
    },
    {
        'name': 'li_tier2_days_max_15',
        'gua': '101',
        'patch': {
            # 敲掉 tier2 days_max, 保留 tier1 深池拒绝
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 99999, 'days_max': None},
                {'depth_max': -250, 'days_min': 0, 'days_max': None},
            ],
        },
    },
]


if __name__ == '__main__':
    run_ablation(candidates, tag='step2a_pool')
