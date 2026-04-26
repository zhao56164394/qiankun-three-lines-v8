# -*- coding: utf-8 -*-
"""八卦分治策略 cfg 版本仓库

并存的策略版本, 通过环境变量 STRATEGY_VERSION 选择 (默认 test1).

  test1: Step 6 之后的最优配置 (2026-04-26 冻结)
         性能: 终值 6217万, 收益 +30986%, 回撤 43.39, 笔数 267, max_pos=3
         消融纪律走完 6 步:
           - Stage 2 5卦 y_gate, Stage 3b 110/111 ym_gate
           - Step 2b 敲 3 有害 gate (gen/li/dui)
           - Step 3+3b 删 7 有害 di/ren, 留 gen_allow_di_gua
           - Step 5 三个冗余卖法→bear, 仅 100 震保 bull
           - Step 4 离卦 double_rise→cross
           - Step 6 max_pos 5→3

  test2: 起点 = 纯 naked baseline. 等价于 derive_naked_cfg 在 test1 上的结果, 但抽出来当独立 cfg.
         性能预期: 终值 363万, 收益 +1715%, 回撤 55.10
         所有过滤 / gate / 专属卖法都关闭, 只保留:
           - 入池阈值 -250
           - 触发模式 (double_rise / cross 及阈值)
           - 分治架构本身 (8 个 tian_gua 分支)
           - 池深池天 tiers (视为分治结构性约束, 非过滤)

用法:
  from strategy_configs import get_strategy
  GUA_STRATEGY = get_strategy()  # 默认 test1, 或读 env STRATEGY_VERSION

  # 命令行切换:
  STRATEGY_VERSION=test2 python backtest_8gua_naked.py
"""
import copy
import os


UNIFIED_POOL_THRESHOLD = -250


# ============================================================
# test1: Step 6 后冻结的最优配置 (2026-04-26)
# 终值 6217万, 收益 +30986%, 回撤 43.39, 笔数 267 (max_pos=3)
# ============================================================
STRATEGY_TEST1 = {
    '000': {'sell': 'bear', 'active': True,
            'pool_threshold': -250,
            'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # 方案 α: 任何池深都接受, 但排除池天 4-10 (磨底死区)
            # 八卦: 0-3天=极而复反, 11+天=物极必反; 4-10天=阴中无阳磨底, 必亏
            'pool_depth_tiers': [
                {'depth_max': None, 'days_min': 0, 'days_max': None, 'days_exclude': [4, 10]},
            ],
            'kun_buy': True,
            'kun_buy_mode': 'double_rise',
            'kun_cross_threshold': 20,
            # Gate: y={101,110} ablation 验证独立有益 +171万
            'gate_disable_y_gua': {'101', '110'},
            },
    '001': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'gen_buy': True,
            'gen_allow_di_gua': {'000', '010'},
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            },
    '010': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'gate_disable_y_gua': {'101'},
            },
    '011': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'xun_buy': 'double_rise',
            'xun_buy_param': 11,
            'gate_disable_y_gua': {'101'},
            },
    '100': {'sell': 'bull', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': 1, 'pool_days_max': 7,
            'zhen_buy': True,
            'zhen_buy_mode': 'double_rise',
            'zhen_cross_threshold': 20,
            'zhen_allow_di_gua': None,
            },
    '101': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 99999, 'days_max': None},
                {'depth_max': -250, 'days_min': 0,     'days_max': 15},
            ],
            'li_buy': True,
            'li_buy_mode': 'cross',     # 消融验证: cross +1115万 vs double_rise
            'li_cross_threshold': 20,
            },
    '110': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'dui_buy': True,
            'dui_buy_mode': 'cross',
            'dui_cross_threshold': 20,
            },
    '111': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'qian_buy': True,
            'qian_cross_threshold': 60,
            'qian_exclude_ren_gua': set(),
            # ym gate ablation LOO +55万 独立有益, 保留
            'gate_disable_ym': {('011', '101'), ('000', '111')},
            },
}

# 默认仓位 (Step 6 验证 max_pos=3 比 5 多 +3635 万)
TEST1_MAX_POS = 3
TEST1_DAILY_LIMIT = 1


# ============================================================
# test2: 纯 naked baseline (起点)
# 等价于 test1 上跑 derive_naked_cfg 的结果
# 终值预期 363万 (回撤 55, 笔数 ~444)
# ============================================================
STRATEGY_TEST2 = {
    '000': {'sell': 'bear', 'active': True,
            'pool_threshold': -250,
            'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # 池深池天 tiers 视为分治结构性约束, 保留
            'pool_depth_tiers': [
                {'depth_max': None, 'days_min': 0, 'days_max': None, 'days_exclude': [4, 10]},
            ],
            'kun_buy': True,
            'kun_buy_mode': 'double_rise',
            'kun_cross_threshold': 20,
            # naked: 无 gate, 无 di/ren 黑白名单
            },
    '001': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'gen_buy': True,
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            # naked: 无 gen_allow_di_gua
            },
    '010': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            },
    '011': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'xun_buy': 'double_rise',
            'xun_buy_param': 11,
            },
    '100': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': 1, 'pool_days_max': 7,
            'zhen_buy': True,
            'zhen_buy_mode': 'double_rise',
            'zhen_cross_threshold': 20,
            },
    '101': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'pool_depth_tiers': [
                {'depth_max': -350, 'days_min': 99999, 'days_max': None},
                {'depth_max': -250, 'days_min': 0,     'days_max': 15},
            ],
            'li_buy': True,
            'li_buy_mode': 'double_rise',  # naked baseline 用 double_rise
            'li_cross_threshold': 20,
            },
    '110': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'dui_buy': True,
            'dui_buy_mode': 'cross',
            'dui_cross_threshold': 20,
            },
    '111': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'qian_buy': True,
            'qian_cross_threshold': 60,
            },
}

TEST2_MAX_POS = 5      # naked baseline 时代默认值
TEST2_DAILY_LIMIT = 1


# ============================================================
# 加载入口
# ============================================================
_REGISTRY = {
    'test1': {'strategy': STRATEGY_TEST1, 'max_pos': TEST1_MAX_POS, 'daily_limit': TEST1_DAILY_LIMIT},
    'test2': {'strategy': STRATEGY_TEST2, 'max_pos': TEST2_MAX_POS, 'daily_limit': TEST2_DAILY_LIMIT},
}


def get_version() -> str:
    """读取当前版本 (env STRATEGY_VERSION, 默认 test1)"""
    v = os.environ.get('STRATEGY_VERSION', 'test1')
    if v not in _REGISTRY:
        raise ValueError(f'未知 STRATEGY_VERSION={v}, 候选: {list(_REGISTRY)}')
    return v


def get_strategy() -> dict:
    """返回当前版本的 GUA_STRATEGY (深拷贝, 修改不会污染 cfg 仓库)"""
    return copy.deepcopy(_REGISTRY[get_version()]['strategy'])


def get_sim_params() -> dict:
    """返回当前版本的 max_pos / daily_limit"""
    cfg = _REGISTRY[get_version()]
    return {'max_pos': cfg['max_pos'], 'daily_limit': cfg['daily_limit']}
