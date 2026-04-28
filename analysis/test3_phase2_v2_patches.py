# -*- coding: utf-8 -*-
"""Phase 2 v2 单 trd 视角 cfg patches (基于实买兑现规律)"""

V2_PATCHES = {
    '000': {
        # trd 大多正, 几无强不买. 仅排明显大亏: 浅×[31+]
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': 30},  # 浅排 [31+] (-2.9)
        ],
        'pool_priority_tiers': [
            # 物极列 (除 (-400,-350])
            {'depth_max': -400, 'days_min': 11, 'days_max': 30, 'bonus': 100},
            {'depth_max': -300, 'days_min': 11, 'days_max': 30, 'bonus': 100},
            {'depth_max': None, 'days_min': 11, 'days_max': 30, 'bonus': 100},
            # 中档行 (除 [4-10])
            {'depth_max': -300, 'days_min': 0, 'days_max': 3, 'bonus': 50},
            {'depth_max': -300, 'days_min': 31, 'days_max': None, 'bonus': 50},
        ],
    },
    '001': {
        # 不买: (-400,-350)行 + 浅档[0-3]/[4-10] + 中×[0-3]
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            # 深 (-400,-350]: 整行拒 (trd -4.8/-1.0)
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            # 中: 排 [0-3] (1/-7.2)
            {'depth_max': -300, 'days_min': 4, 'days_max': None},
            # 浅: 排 [0-3] [4-10] (3/-3.3, 2/-4.6)
            {'depth_max': None, 'days_min': 11, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 极深行 × [4-10]/[11-30]/[31+]
            {'depth_max': -400, 'days_min': 4, 'days_max': None, 'bonus': 100},
        ],
    },
    '010': {
        # 不买: (-300,-250]行(可见) + (-350,-300]行(可见)
        'pool_depth_tiers': [
            # 极深 ≤-400: trd 5/+15.8, 接受
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            # 深 (-400,-350]: trd 各 +10.8/+5.7, 接受
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中 (-350,-300]: 整行拒 (trd -0.8/-0.3)
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            # 浅: 整行拒 (trd -16.9/-3.6)
            {'depth_max': None, 'days_min': 99999, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 深档 [4-10] (单格最强 5/+15.8)
            {'depth_max': -400, 'days_min': 4, 'days_max': 10, 'bonus': 50},
            {'depth_max': -350, 'days_min': 4, 'days_max': 10, 'bonus': 50},
        ],
    },
    '011': {
        # 不买: [4-10] 列 (除 (-400,-350]) — trd -0.5/-6.2/-4.4
        'pool_depth_tiers': [
            # 极深: 排 [4-10]
            {'depth_max': -400, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
            # 深: 全接受
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中: 排 [4-10]
            {'depth_max': -300, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
            # 浅: 排 [4-10]
            {'depth_max': None, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
        ],
        'pool_priority_tiers': [
            # 极深×[31+]
            {'depth_max': -400, 'days_min': 31, 'days_max': None, 'bonus': 100},
        ],
    },
    '100': {
        # 不买: 中档行(除[11-30]) + [31+]列 + 浅×[11-30]
        'pool_depth_tiers': [
            # 极深: 排 [31+] (-4.7)
            {'depth_max': -400, 'days_min': 0, 'days_max': 30},
            # 深 (-400,-350]: 排 [31+] (-4.3)
            {'depth_max': -350, 'days_min': 0, 'days_max': 30},
            # 中: 仅接 [11-30] (其他 trd -8.5/-11.5/-8.1)
            {'depth_max': -300, 'days_min': 11, 'days_max': 30},
            # 浅: 排 [11-30] (-9.9) 和 [31+] (-5.4)
            {'depth_max': None, 'days_min': 0, 'days_max': 10},
        ],
        'pool_priority_tiers': [
            # 深 × [11-30] (单格 trd +19.7)
            {'depth_max': -350, 'days_min': 11, 'days_max': 30, 'bonus': 100},
        ],
    },
    '101': {
        # 不买: 极深×[4-10]/[11-30] + [4-10]列
        'pool_depth_tiers': [
            # 极深: 排 [4-10] [11-30] (trd -9.0 / -9.6)
            {'depth_max': -400, 'days_min': 31, 'days_max': None},
            # 深: 排 [4-10] (-2.5)
            {'depth_max': -350, 'days_min': 11, 'days_max': None,
             'days_exclude': [11, 30]},  # 除掉 [11-30] 也无 trd
            # 中: 排 [4-10] (-4.7)
            {'depth_max': -300, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
            # 浅: 排 [4-10] (-2.3)
            {'depth_max': None, 'days_min': 0, 'days_max': None,
             'days_exclude': [4, 10]},
        ],
        'pool_priority_tiers': [
            # 中×[11-30] (1/+16.6)
            {'depth_max': -300, 'days_min': 11, 'days_max': 30, 'bonus': 50},
        ],
    },
    '110': {
        # 不买: 中×[31+] + 浅×[0-3]/[31+]
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 中: 排 [31+] (-4.7)
            {'depth_max': -300, 'days_min': 0, 'days_max': 30},
            # 浅: 排 [0-3] (-3.8) [31+] (-5.2)
            {'depth_max': None, 'days_min': 4, 'days_max': 30},
        ],
        'pool_priority_tiers': [
            # 深×[31+] (2/+40.6)
            {'depth_max': -350, 'days_min': 31, 'days_max': None, 'bonus': 100},
        ],
    },
    '111': {
        # 不买: 浅×[0-3]/[4-10]
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            # 浅: 排 [0-3] (-9.0) [4-10] (-14.1)
            {'depth_max': None, 'days_min': 11, 'days_max': None},
        ],
        'pool_priority_tiers': [
            # 长池天列
            {'depth_max': None, 'days_min': 11, 'days_max': None, 'bonus': 100},
        ],
    },
}
