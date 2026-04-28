# -*- coding: utf-8 -*-
"""Phase 2 v3 综合视角 cfg patches (最保守).

不买区 (pool_depth_tiers 表达接受范围, 不在任何 tier 内即拒):
  000 坤: 极深×[4-10] 拒, [31+] 列除中档拒
  001 艮: 全接受 (无强不买规律)
  010 坎: 全接受 (规律弱)
  011 巽: 极深×[4-10] 拒, 极深×[11-30] 拒
  100 震: 中档×[31+] 拒, [31+] 列偏弱
  101 离: [4-10] 列整列拒
  110 兑: 中档行整行拒
  111 乾: 全接受

优先买 (pool_priority_tiers, bonus 高的优先):
  000 坤: 浅×[0-3] (100, 主战场); 中档行(除[4-10]) (50)
  001 艮: 极深行(除[0-3]) (100, 兑现核心)
  010 坎: 无
  011 巽: 极深×[31+] (100); 浅×[0-3] (50)
  100 震: 深×[11-30] (100); 浅×[0-3] (50)
  101 离: 无
  110 兑: 深×[31+] (50)
  111 乾: 长池天列(中浅档+极深) (100)
"""

V3_PATCHES = {
    '000': {
        'pool_depth_tiers': [
            # 极深 ≤-400: 接受 [0-3] 和 [11-30]+, 排 [4-10]
            {'depth_max': -400, 'days_min': 0, 'days_max': None, 'days_exclude': [4, 10]},
            # 深 (-400,-350]: 接受全部 (无明确规律)
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中 (-350,-300]: 接受全部 (中档行 v3 优先买)
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            # 浅 (-300,-250]: 接受 [0-30], 排 [31+] 久磨
            {'depth_max': None, 'days_min': 0, 'days_max': 30},
        ],
        'pool_priority_tiers': [
            # 浅×[0-3] 主战场 (660 sig + 12 trd + 同向)
            {'depth_max': None, 'days_min': 0, 'days_max': 3, 'bonus': 100},
            # 中档行 [0-3]/[11-30]/[31+] (除 [4-10])
            {'depth_max': -300, 'days_min': 0, 'days_max': 3, 'bonus': 50},
            {'depth_max': -300, 'days_min': 11, 'days_max': None, 'bonus': 50},
        ],
    },
    '001': {
        # v3: 全接受
        'pool_priority_tiers': [
            # 极深行 × [4-10]/[11-30]/[31+] (兑现核心)
            {'depth_max': -400, 'days_min': 4, 'days_max': None, 'bonus': 100},
        ],
    },
    '010': {
        # v3: 规律太弱, 全接受 + 无优先
    },
    '011': {
        'pool_depth_tiers': [
            # 极深 ≤-400: 接受 [0-3] 和 [31+], 排 [4-30]
            {'depth_max': -400, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 30]},  # 排 [4-10] 和 [11-30] 合并
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 极深 × [31+] (双向大正)
            {'depth_max': -400, 'days_min': 31, 'days_max': None, 'bonus': 100},
            # 浅 × [0-3] (主战场)
            {'depth_max': None, 'days_min': 0, 'days_max': 3, 'bonus': 50},
        ],
    },
    '100': {
        'pool_depth_tiers': [
            # 极深 ≤-400: 接受 [0-3] 和 [31+] (sig+7.0)
            {'depth_max': -400, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 30]},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中 (-350,-300]: 排 [31+]
            {'depth_max': -300, 'days_min': 0, 'days_max': 30},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 深 × [11-30] (单格 trd +19.7)
            {'depth_max': -350, 'days_min': 11, 'days_max': 30, 'bonus': 100},
            # 浅 × [0-3]
            {'depth_max': None, 'days_min': 0, 'days_max': 3, 'bonus': 50},
        ],
    },
    '101': {
        'pool_depth_tiers': [
            # [4-10] 列整列拒
            {'depth_max': None, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
        ],
        # v3: 无优先买
    },
    '110': {
        'pool_depth_tiers': [
            # 中档 (-350,-300] 行整行拒, 其他全接受
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # (-350,-300] 行: days_min=99999 永不接受
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 深 × [31+] (8/2 双向大正)
            {'depth_max': -350, 'days_min': 31, 'days_max': None, 'bonus': 50},
        ],
    },
    '111': {
        # v3: 全接受
        'pool_priority_tiers': [
            # 长池天 × 中浅档
            {'depth_max': -300, 'days_min': 11, 'days_max': None, 'bonus': 100},
            {'depth_max': None, 'days_min': 11, 'days_max': None, 'bonus': 100},
            # 极深 × [31+]
            {'depth_max': -400, 'days_min': 31, 'days_max': None, 'bonus': 100},
        ],
    },
}
