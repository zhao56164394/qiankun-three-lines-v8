# -*- coding: utf-8 -*-
"""Phase 2 单维池深 — 三版本 cfg patch

v1 单 sig 视角不买区 (sig 大样本 mean<-3 或 整卦低质):
  000 坤  极深 (sig 890/-3.0)
  010 坎  极深 (sig 33/-1.6, 弱样本)
  011 巽  极深 (sig 1168/-7.6) — 强
  101 离  整卦 4 档全负 — 强
  110 兑  极深 (sig 52/-5.9)

v2 单 trd 视角不买区 (trd 总利润 < -3 万):
  000 坤  无 (各档全正)
  001 艮  深 (trd -5.7) + 浅 (trd -2.6)
  010 坎  浅 (trd -9.3)
  100 震  中 (trd -11.8)
  101 离  极深 + 深 (trd -4.5/-3.4)
  110 兑  极深 (-5.2) + 浅 (-7.3)
  111 乾  中 (trd -11.1)

v3 综合视角 (双向同向负, sig_n>=20):
  101 离  4 档全负 — 强
  110 兑  极深 (sig 52/-5.9 + trd 7/-5.2) — 中等
"""

# v1 单 sig
V1_DEPTH_PATCHES = {
    '000': {
        'pool_depth_tiers': [
            # 极深 ≤-400 拒, 其他全接受
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
    '010': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
    '011': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
    '101': {
        # 整卦 4 档全负: cfg 直接 active=False
        'active': False,
    },
    '110': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
}

# v2 单 trd
V2_DEPTH_PATCHES = {
    '001': {
        'pool_depth_tiers': [
            # 接受极深 + 中, 拒 深 (-5.7) 和 浅 (-2.6)
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},  # 拒深
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 99999, 'days_max': None},  # 拒浅
        ],
    },
    '010': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            # 拒浅 (trd -9.3)
            {'depth_max': None, 'days_min': 99999, 'days_max': None},
        ],
    },
    '100': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 拒中 (trd -11.8)
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
        # 注: cfg 起点已有 pool_days_min=1, pool_days_max=7 — 保持
    },
    '101': {
        'pool_depth_tiers': [
            # 拒极深 (-4.5), 深 (-3.4)
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': -350, 'days_min': 99999, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
    '110': {
        'pool_depth_tiers': [
            # 拒极深 (-5.2), 浅 (-7.3)
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            {'depth_max': -300, 'days_min': 0, 'days_max': None},
            {'depth_max': None, 'days_min': 99999, 'days_max': None},
        ],
    },
    '111': {
        'pool_depth_tiers': [
            {'depth_max': -400, 'days_min': 0, 'days_max': None},
            {'depth_max': -350, 'days_min': 0, 'days_max': None},
            # 拒中 (trd -11.1)
            {'depth_max': -300, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
}

# v3 综合
V3_DEPTH_PATCHES = {
    '101': {
        # 整卦低质 (sig 4 档全负 + trd 极深+深双向负): 关闭
        'active': False,
    },
    '110': {
        'pool_depth_tiers': [
            # 拒极深 (双向同向负)
            {'depth_max': -400, 'days_min': 99999, 'days_max': None},
            {'depth_max': None, 'days_min': 0, 'days_max': None},
        ],
    },
}
