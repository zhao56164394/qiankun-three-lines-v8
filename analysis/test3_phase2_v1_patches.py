# -*- coding: utf-8 -*-
"""Phase 2 v1 单 sig 视角 cfg patches (排得最多, 仅信任 sig 规律)"""

V1_PATCHES = {
    '000': {
        # 不买: 极深行(除[0-3]) + [11-30]列(除深档) + [31+]列
        'pool_depth_tiers': [
            # 极深 ≤-400: 仅接 [0-3]
            {'depth_max': -400, 'days_min': 0, 'days_max': 3},
            # 深 (-400,-350]: 接 [0-3]+[11-30]+[31+] (sig +1.0/+8.1/-3.4)
            #   实际 [31+] 偏负, v1 也排
            {'depth_max': -350, 'days_min': 0, 'days_max': 30,
             'days_exclude': [4, 10]},
            # 中 (-350,-300]: 接 [0-3]+[4-10]+[11-30] 排 [31+]
            {'depth_max': -300, 'days_min': 0, 'days_max': 30},
            # 浅 (-300,-250]: 接 [0-3]+[4-10] 排 [11-30]+[31+]
            {'depth_max': None, 'days_min': 0, 'days_max': 10},
        ],
        'pool_priority_tiers': [
            {'depth_max': None, 'days_min': 0, 'days_max': 3, 'bonus': 100},
            {'depth_max': -350, 'days_min': 11, 'days_max': 30, 'bonus': 50},
        ],
    },
    '001': {
        # 不买: (-400,-350] × ([4-10]/[31+]) 弱区 (强度低, v1 也仅排 [31+])
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': 30},  # 排 [31+]
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 极深行 (sig +28.2/+1.7/+15.3/+11.5)
            {'depth_max': -400, 'days_min': 0, 'days_max': None, 'bonus': 100},
            # [0-3] 极反列
            {'depth_max': None, 'days_min': 0, 'days_max': 3, 'bonus': 50},
        ],
    },
    '010': {
        # 不买: 极深行 + [31+]列(除浅档)
        'pool_depth_tiers': [
            # 极深 ≤-400 整行拒
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            # 深 (-400,-350]: 接 [0-3]/[4-10]/[11-30] 排 [31+]
            {'depth_max': -350, 'days_min': 0, 'days_max': 30},
            # 中: 同样排 [31+]
            {'depth_max': -300, 'days_min': 0, 'days_max': 30},
            # 浅: 全接受
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # [0-3] 极反列 (sig +13.6/+12.4/+13.8)
            {'depth_max': -350, 'days_min': 0, 'days_max': 3, 'bonus': 100},
            # 中档行 (除 [31+])
            {'depth_max': -300, 'days_min': 0, 'days_max': 30, 'bonus': 50},
        ],
    },
    '011': {
        # 不买: 极深行 (除 [31+])
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 31, 'days_max': None},  # 极深仅接 [31+]
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 浅档行 (sig 整行大正)
            {'depth_max': None, 'days_min': 0, 'days_max': None, 'bonus': 100},
            # 中档 [4-10]/[31+]
            {'depth_max': -300, 'days_min': 4, 'days_max': 10, 'bonus': 50},
            {'depth_max': -300, 'days_min': 31, 'days_max': None, 'bonus': 50},
        ],
    },
    '100': {
        # 不买: (-400,-350]行(除[11-30]) + 极深[4-10]/[11-30] + 中×[31+]
        'pool_depth_tiers': [
            # 极深: 仅接 [0-3] 和 [31+]
            {'depth_max': -400, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 30]},
            # 深 (-400,-350]: 仅接 [11-30]
            {'depth_max': -350, 'days_min': 11, 'days_max': 30},
            # 中: 排 [31+]
            {'depth_max': -300, 'days_min': 0, 'days_max': 30},
            # 浅: 全接受
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 浅档行 (主战场)
            {'depth_max': None, 'days_min': 0, 'days_max': None, 'bonus': 100},
            # 中档行 (除 [31+])
            {'depth_max': -300, 'days_min': 0, 'days_max': 30, 'bonus': 50},
        ],
    },
    '101': {
        # 不买: [4-10]列(除中×[11-30]) + (-400,-350]行 + 浅×[0-3]
        'pool_depth_tiers': [
            # 极深: 接 [11-30]/[31+]
            {'depth_max': -400, 'days_min': 11, 'days_max': None},
            # 深 (-400,-350]: 整行拒 (v1 全排)
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            # 中: 接 [11-30] (反例正格)
            {'depth_max': -300, 'days_min': 11, 'days_max': 30},
            # 浅: 接 [11-30]/[31+] (排 [0-3] [4-10])
            {'depth_max': None, 'days_min': 11, 'days_max': None},
        ],
        # 离卦 v1 无强优先
    },
    '110': {
        # 不买: (-350,-300]行 + 极深×[11-30] + 浅×[4-10]
        'pool_depth_tiers': [
            # 极深: 排 [11-30] (sig -13.5)
            {'depth_max': -400, 'days_min': 0, 'days_max': None,
             'days_exclude': [11, 30]},
            # 深 (-400,-350]: 全接受
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中 (-350,-300]: 整行拒
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            # 浅: 排 [4-10]
            {'depth_max': None, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
        ],
        'pool_priority_tiers': [
            # 深×[31+] (sig +17.7)
            {'depth_max': -350, 'days_min': 31, 'days_max': None, 'bonus': 50},
            # 浅×[11-30] (sig 17/+4.3 主战场)
            {'depth_max': None, 'days_min': 11, 'days_max': 30, 'bonus': 50},
        ],
    },
    '111': {
        # 不买: 几无 (整卦大正)
        'pool_priority_tiers': [
            # [31+] 列整列 (sig 大正)
            {'depth_max': None, 'days_min': 31, 'days_max': None, 'bonus': 100},
            # [11-30] 列 × 中浅档
            {'depth_max': -300, 'days_min': 11, 'days_max': 30, 'bonus': 50},
            {'depth_max': None, 'days_min': 11, 'days_max': 30, 'bonus': 50},
        ],
    },
}
