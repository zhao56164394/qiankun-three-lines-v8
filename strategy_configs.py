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
# test3: 起点 = 裸跑, 但底座彻底换路径
# ----------------------------------------------------------
# 与 test2 的区别 (cfg 起点完全相同, 区别在优化方向):
#   - test3 明确放弃 di_gua / ren_gua 这条线 (cfg 永远不写 *_allow_di_gua / *_exclude_ren_gua)
#   - test3 的优化路径走 大盘 (y_gua,m_gua) + 个股 (stk_y,stk_m) 双环境开关
#
# 7 步 pipeline:
#   1. -250 入池
#   2. 池深 / 池天 (tiers)
#   3. 大盘 年/月 卦 gate
#   4. 个股 年/月 卦 stock_gate
#   5. 买点 (触发模式)
#   6. 卖点 (sell mode)
#   7. 仓位 (max_pos)
#
# Phase 2 决策 (2026-04-27):
#   - 池深规则 v1_y: 5 卦 (000/010/011/110) 排极深 ≤-400, 101 整卦不买 (sig 全负)
#   - 仅在 y_gua ∈ {010, 111} 时激活 (12 月窗口下: 010=反弹乏力, 111=主升)
#   - 7 段 walk-forward 全正/中性, 无反向, 均值 +8.5%
#   - 年卦窗口从 55 月改为 12 月 (命中率 45% → 72%, 滞后周期消除)
#   - 买卖点统一: 全部 double_rise + bear (110/111 cross 已改 double_rise)
# ============================================================
STRATEGY_TEST3 = {
    '000': {'sell': 'bear', 'active': True,
            'pool_threshold': -250,
            'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # Phase 2 v1_y (Walk-forward 7段全正): 仅在 y_gua ∈ {010,111} 时排极深 ≤-400
            # 12月窗口下 010=反弹乏力 / 111=主升, 此环境下大盘恐慌已过, 极深=陷阱
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},
                {'depth_max': None, 'days_min': 0, 'days_max': None},
            ],
            'pool_depth_tiers_only_y_gua': {'010', '111'},
            'kun_buy': True,
            'kun_buy_mode': 'double_rise',
            'kun_cross_threshold': 20,
            },
    '001': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'gen_buy': True,
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            },
    '010': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # Phase 2 v1_y: 同 000, 排极深 ≤-400, 仅在 y_gua ∈ {010,111} 时
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},
                {'depth_max': None, 'days_min': 0, 'days_max': None},
            ],
            'pool_depth_tiers_only_y_gua': {'010', '111'},
            },
    '011': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # Phase 2 v1_y: 排极深 ≤-400 (sig 1168/-7.6 最强证据)
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},
                {'depth_max': None, 'days_min': 0, 'days_max': None},
            ],
            'pool_depth_tiers_only_y_gua': {'010', '111'},
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
            # Phase 2 v1_y: 整卦低质 (sig 4 档 mean 全负). 用永不接 tier + only_y_gua 表达
            # "仅在 y_gua ∈ {010,111} 时全卦不买"
            'pool_depth_tiers': [
                {'depth_max': -10000, 'days_min': 99999, 'days_max': None},
            ],
            'pool_depth_tiers_only_y_gua': {'010', '111'},
            'li_buy': True,
            'li_buy_mode': 'double_rise',
            'li_cross_threshold': 20,
            },
    '110': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            # Phase 2 v1_y: 排极深 ≤-400
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999, 'days_max': None},
                {'depth_max': None, 'days_min': 0, 'days_max': None},
            ],
            'pool_depth_tiers_only_y_gua': {'010', '111'},
            'dui_buy': True,
            'dui_buy_mode': 'double_rise',
            'dui_cross_threshold': 20,
            },
    '111': {'sell': 'bear', 'active': True,
            'pool_threshold': -250, 'pool_depth': None,
            'pool_days_min': None, 'pool_days_max': None,
            'qian_buy': True,
            'qian_buy_mode': 'double_rise',
            'qian_cross_threshold': 20,
            },
}

TEST3_MAX_POS = 5
TEST3_DAILY_LIMIT = 1


# ============================================================
# test4: y_gua 主分治架构 — 8 d_gua 分支统一为 kun (000) 行为
# ============================================================
# 起点: 把 Phase 2 落地的 test3 中, 8 个 d_gua 分支的差异性参数都消除,
#       统一成 kun_strat 的行为模板. 这样 8 个 d_gua 分支不再有差异,
#       y_gua 通过 pool_depth_tiers_only_y_gua={'010','111'} 做条件激活.
# kun 行为模板:
#   - 触发: double_rise 双升 (主力散户共升 + trend>11)
#   - cross_threshold: 20 (cross 模式不启用, 但保留字段)
#   - 卖出: bear (kun_bear)
#   - 入池: -250
#   - 池深 tier: 排极深 ≤-400
#   - 池深 tier 仅在 y_gua ∈ {010, 111} 时激活
#   - 各类 *_exclude_ren_gua / *_allow_di_gua / *_exclude_di_gua: 全置空 (kun 无)
# 各分支保留各自字段名 (kun_buy_mode / gen_buy_mode / qian_buy_mode 等),
# 但参数值统一. 010 (坎) 走共享分支, 不需 *_buy 字段.
_KUN_BASE = {
    'sell': 'bear', 'active': True,
    'pool_threshold': -250,
    'pool_depth': None,
    'pool_days_min': None, 'pool_days_max': None,
    'pool_depth_tiers': [
        {'depth_max': -400, 'days_min': 99999, 'days_max': None},
        {'depth_max': None, 'days_min': 0, 'days_max': None},
    ],
    'pool_depth_tiers_only_y_gua': {'010', '111'},
}

STRATEGY_TEST4 = {
    '000': {**_KUN_BASE,
            'kun_buy': True,
            'kun_buy_mode': 'double_rise',
            'kun_cross_threshold': 20,
            'kun_exclude_ren_gua': set(),
            'kun_allow_di_gua': None,
            },
    '001': {**_KUN_BASE,
            'gen_buy': True,
            'gen_buy_mode': 'double_rise',
            'gen_cross_threshold': 20,
            'gen_exclude_ren_gua': set(),
            'gen_allow_di_gua': None,
            },
    '010': {**_KUN_BASE,
            # 010 走共享分支, 不需 *_buy 字段
            },
    '011': {**_KUN_BASE,
            'xun_buy': 'double_rise',
            'xun_buy_param': 11,
            'xun_allow_di_gua': None,
            },
    '100': {  # 震分支保留 test3 真规律 (池天 1-7), 不用 kun 模板的 pool_depth_tiers
            'sell': 'bear', 'active': True,
            'pool_threshold': -250,
            'pool_depth': None,
            'pool_days_min': 1, 'pool_days_max': 7,  # test3 真规律, 不能丢
            'zhen_buy': True,
            'zhen_buy_mode': 'double_rise',
            'zhen_cross_threshold': 20,
            'zhen_exclude_ren_gua': set(),
            'zhen_allow_di_gua': None,
            },
    '101': {**_KUN_BASE,
            'li_buy': True,
            'li_buy_mode': 'double_rise',
            'li_cross_threshold': 20,
            'li_exclude_ren_gua': set(),
            },
    '110': {**_KUN_BASE,
            'dui_buy': True,
            'dui_buy_mode': 'double_rise',
            'dui_cross_threshold': 20,
            'dui_exclude_ren_gua': set(),
            'dui_allow_di_gua': None,
            },
    '111': {**_KUN_BASE,
            'qian_buy': True,
            'qian_buy_mode': 'double_rise',
            'qian_cross_threshold': 20,
            'qian_exclude_ren_gua': set(),
            'qian_exclude_di_gua': set(),
            },
}

TEST4_MAX_POS = 5
TEST4_DAILY_LIMIT = 1


# ============================================================
# test5: y_gua 真分治 (按大盘当日 y_gua 分桶选 cfg)
# ============================================================
# 与 test4 的差别:
#   - test4: 8 个 d_gua 分支 cfg, 行为统一为 kun. 但路由仍按 d_gua.
#   - test5: 8 个 y_gua 桶 cfg, 行为统一为 kun. 路由按当日 y_gua.
# 字段简化为统一名 (不再 kun_/gen_/qian_ 前缀):
#   buy            : True (是否在该桶下买)
#   buy_mode       : 'double_rise' | 'cross'
#   cross_threshold: 20 (cross 模式下的阈值)
#   exclude_ren_gua: set() (排市场卦)
#   allow_di_gua   : None | set() (个股白名单)
#   sell           : 'bear'
#   pool_threshold / pool_depth_tiers / pool_depth_tiers_only_y_gua: 同 _KUN_BASE
# 起步: 8 桶全用同一套 kun 参数, 后续按桶逐个调.
_KUN_TEMPLATE_FOR_Y_GUA = {
    **_KUN_BASE,
    'buy': True,
    'buy_mode': 'double_rise',
    'cross_threshold': 20,
    'exclude_ren_gua': set(),
    'allow_di_gua': None,
}

STRATEGY_TEST5_BY_Y_GUA = {
    '000': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 坤 深熊
    '001': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 艮 吸筹
    '010': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 坎 乏力
    '011': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 巽 底爆
    '100': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 震 出货
    '101': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 离 护盘
    '110': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 兑 末减
    '111': {**_KUN_TEMPLATE_FOR_Y_GUA},  # 乾 疯牛
}

TEST5_MAX_POS = 5
TEST5_DAILY_LIMIT = 1


# ============================================================
# test6: y_gua 真分治 + 真裸基线 (无任何 Phase 2 痕迹)
# ============================================================
# 与 test5 的区别:
#   - test5 的 _KUN_BASE 还带 pool_depth_tiers (Phase 2 v1_y 痕迹)
#   - test6 完全清零: 仅 -250 入池 + 双升触发 + bear 卖出
# 用途: 作为 ablation 的真起点 (baseline). 加任何参数都能算到它的 alpha.
_NAKED_TEMPLATE = {
    'sell': 'bear', 'active': True,
    'pool_threshold': -250,
    'pool_depth': None,
    'pool_days_min': None, 'pool_days_max': None,
    # 关键: pool_depth_tiers / only_y_gua 一律不设, 走兼容旧逻辑分支 (无极深排除)
    'buy': True,
    'buy_mode': 'double_rise',
    'cross_threshold': 20,
    'exclude_ren_gua': set(),
    'allow_di_gua': None,
}

STRATEGY_TEST6_BY_Y_GUA = {
    '000': {**_NAKED_TEMPLATE},
    '001': {**_NAKED_TEMPLATE},
    '010': {**_NAKED_TEMPLATE},
    '011': {**_NAKED_TEMPLATE},
    '100': {**_NAKED_TEMPLATE},
    '101': {**_NAKED_TEMPLATE},
    '110': {**_NAKED_TEMPLATE},
    '111': {**_NAKED_TEMPLATE},
}

TEST6_MAX_POS = 5
TEST6_DAILY_LIMIT = 1


# ============================================================
# test7: y_gua 真分治 + 各桶池深×池天 ablation 落地
# ============================================================
# 起点 = test6 真裸基线, 各桶根据 IS sig×trd 矩阵分析得出 v1/v2/v3 三个版本.
# v1 = 单 sig 视角规律 (sig 大样本 + sig_mean 显著)
# v2 = 单 trd 视角规律 (排实战亏的 cell, 但风险: trd 凑路径 OOS 可能反向)
# v3 = sig + trd 双视角同向 (最保守, 双重证据)
#
# baseline 桶: 当样本不足 (<200 sig 或 <30 trd) 不下结论, 沿用 _NAKED_TEMPLATE.

# ----- v1: 单 sig 视角 -----
# 各桶规律 (来自 phase3_y_gua_bucket_analysis.py 输出):
#   000 坤: sig 全正, 不加 tier
#   001 艮: 样本不足, baseline
#   010 坎: 排 [11-30] 物极 列 (除中档行)
#   011 巽: 样本不足, baseline
#   100 震: 排浅档行 + 深档 (-400,-350] 行
#   101 离: 排极深行 + 深行 + 中行 (除浅档勉强中性偏正)
#   110 兑: 排浅档行 + 物极列上3行
#   111 乾: 排久磨 [31+] 列 + 排极深行 (除 [11-30])

STRATEGY_TEST7_V1_BY_Y_GUA = {
    '000': {**_NAKED_TEMPLATE},  # 不加 tier
    '001': {**_NAKED_TEMPLATE},
    '010': {**_NAKED_TEMPLATE,
            # 排 [11-30] 物极列 在极深/深/浅三档 (中档保留)
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_exclude': [11, 30]},
                {'depth_max': -350, 'days_exclude': [11, 30]},
                {'depth_max': -300, 'days_min': 0},  # 中档不排
                {'depth_max': None,  'days_exclude': [11, 30]},  # 浅档排
            ],
            },
    '011': {**_NAKED_TEMPLATE},
    '100': {**_NAKED_TEMPLATE,
            # 排浅档行 + 深档 (-400,-350] 行
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 0},  # 极深保留
                {'depth_max': -350, 'days_min': 99999},  # 深档整行排
                {'depth_max': -300, 'days_min': 0},  # 中档保留
                {'depth_max': None,  'days_min': 99999},  # 浅档整行排
            ],
            },
    '101': {**_NAKED_TEMPLATE,
            # 排极深 + 深 + 中, 仅留浅档
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999},  # 极深排
                {'depth_max': -350, 'days_min': 99999},  # 深排
                {'depth_max': -300, 'days_min': 99999},  # 中排
                {'depth_max': None,  'days_min': 0},  # 浅保留
            ],
            },
    '110': {**_NAKED_TEMPLATE,
            # 排浅档行 + 物极列上3行
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_exclude': [11, 30]},
                {'depth_max': -350, 'days_exclude': [11, 30]},
                {'depth_max': -300, 'days_exclude': [11, 30]},
                {'depth_max': None,  'days_min': 99999},  # 浅档整行排
            ],
            },
    '111': {**_NAKED_TEMPLATE,
            # 排久磨 [31+] 列 + 排极深行 (除 [11-30])
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 11, 'days_max': 30},  # 极深仅留 [11-30]
                {'depth_max': -350, 'days_max': 30},  # 深: 排久磨
                {'depth_max': -300, 'days_max': 30},  # 中: 排久磨
                {'depth_max': None,  'days_max': 30},  # 浅: 排久磨
            ],
            },
}

# ----- v2: 单 trd 视角 -----
#   000 坤: 排浅档行 (-300,-250)
#   001 艮: 样本不足, baseline
#   010 坎: 样本不足 (大多 cell n<5), baseline
#   011 巽: 样本不足, baseline
#   100 震: 排中档行
#   101 离: 排极深×物极 单格 + 浅×极反 单格 (单视角负但 sig 反例 trd 凑路径)
#   110 兑: 不下结论, baseline
#   111 乾: 排久磨列 + 排中档前 2 列 ([0-3], [4-10])

STRATEGY_TEST7_V2_BY_Y_GUA = {
    '000': {**_NAKED_TEMPLATE,
            # 排浅档行
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 0},
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 0},
                {'depth_max': None,  'days_min': 99999},  # 浅档整行排
            ],
            },
    '001': {**_NAKED_TEMPLATE},
    '010': {**_NAKED_TEMPLATE},
    '011': {**_NAKED_TEMPLATE},
    '100': {**_NAKED_TEMPLATE,
            # 排中档行 (-350,-300]
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 0},
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 99999},  # 中档整行排
                {'depth_max': None,  'days_min': 0},
            ],
            },
    '101': {**_NAKED_TEMPLATE,
            # 排极深×[11-30] + 浅×[0-3]
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_exclude': [11, 30]},  # 极深排物极
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 0},
                {'depth_max': None,  'days_exclude': [0, 3]},  # 浅档排极反
            ],
            },
    '110': {**_NAKED_TEMPLATE},
    '111': {**_NAKED_TEMPLATE,
            # 排久磨 [31+] 列 + 排中档 [0-3]/[4-10] 两列
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_max': 30},
                {'depth_max': -350, 'days_max': 30},
                {'depth_max': -300, 'days_min': 11, 'days_max': 30},  # 中档仅留 [11-30]
                {'depth_max': None,  'days_max': 30},
            ],
            },
}

# ----- v3: 综合 sig+trd -----
#   000 坤: 不加 tier (双视角无可信负区, 信号自由进, 仓位竞争自然过滤)
#   001 艮: baseline
#   010 坎: baseline (v1 sig 优势在 trd 没兑现)
#   011 巽: baseline
#   100 震: 排浅档×[11-30] 单格
#   101 离: 排极深行 (sig 大样本严重负 + trd 物极同向负)
#   110 兑: 排浅档×[4-10] 磨底 单格 (双视角同向负)
#   111 乾: 排久磨列 + 排极深×[4-10] 磨底 单格

STRATEGY_TEST7_V3_BY_Y_GUA = {
    '000': {**_NAKED_TEMPLATE},  # 不加 tier
    '001': {**_NAKED_TEMPLATE},
    '010': {**_NAKED_TEMPLATE},
    '011': {**_NAKED_TEMPLATE},
    '100': {**_NAKED_TEMPLATE,
            # 排浅档 × [11-30] 物极
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 0},
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 0},
                {'depth_max': None,  'days_exclude': [11, 30]},
            ],
            },
    '101': {**_NAKED_TEMPLATE,
            # 排极深行 (整行)
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 99999},
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 0},
                {'depth_max': None,  'days_min': 0},
            ],
            },
    '110': {**_NAKED_TEMPLATE,
            # 排浅档 × [4-10] 磨底 单格
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_min': 0},
                {'depth_max': -350, 'days_min': 0},
                {'depth_max': -300, 'days_min': 0},
                {'depth_max': None,  'days_exclude': [4, 10]},
            ],
            },
    '111': {**_NAKED_TEMPLATE,
            # 排久磨 [31+] 列 + 极深×[4-10] 磨底单格
            'pool_depth_tiers': [
                {'depth_max': -400, 'days_max': 30, 'days_exclude': [4, 10]},
                {'depth_max': -350, 'days_max': 30},
                {'depth_max': -300, 'days_max': 30},
                {'depth_max': None,  'days_max': 30},
            ],
            },
}

TEST7_MAX_POS = 5
TEST7_DAILY_LIMIT = 1


# ============================================================
# test10: 8 卦 → 4 季合并 (熊_探底 / 转牛 / 牛_主升 / 转熊)
# ============================================================
# 起点 = test6 真裸基线
# 来源: test9_season_ablation 的 LOO + add-one 双向消融, sig 视角
#
# 消融发现:
#   桶级双向 ★:
#     - 排 转熊 整季: LOO +3.53 [+3.10,+3.99] / add-one -8.16 ✗反 (n=4175)
#     - 留 熊_探底:  LOO -5.26 ✗反 / add-one +4.28 [+3.77,+4.77] (n=7623)
#     - 留 转牛:    add-one +2.51 [+0.31,+4.67] ★ 单向 (n=528, 小桶合并后才稳)
#   跨 4 季共识 (LOO ★ 或 add-one ✗反 方向一致):
#     - 排 [4-10] 磨底列: 4/4 季方向一致, 池天结构性陷阱
#     - 留 [0-3] 极反列: 4/4 季 add-one ★, V 反弹真规律
#
# v_a (单点最强 alpha): 仅 转熊整季 skip, 其他 3 季 naked
# v_b (跨 4 季共识):    v_a + 3 季排 [4-10] 磨底列
TEST10_MAX_POS = 5
TEST10_DAILY_LIMIT = 1

_SEASON_TO_YGUA = {
    '熊_探底': ['000'],
    '转牛': ['001', '010', '011'],
    '牛_主升': ['111'],
    '转熊': ['100', '101', '110'],
}


def _expand_season_cfg(season_cfg: dict) -> dict:
    """把 4 季 cfg 展开为 8 卦 cfg (同季 y_gua 共享 cfg 引用)"""
    out = {}
    for season, ygs in _SEASON_TO_YGUA.items():
        for yg in ygs:
            out[yg] = season_cfg[season]
    return out


# v_a: 仅转熊整季 skip
SEASON_CFG_V_A = {
    '熊_探底': {**_NAKED_TEMPLATE},
    '转牛': {**_NAKED_TEMPLATE},
    '牛_主升': {**_NAKED_TEMPLATE},
    '转熊': {**_NAKED_TEMPLATE, 'buy': False},
}

# v_b: v_a + 跨 4 季共识 (排 [4-10] 磨底列)
SEASON_CFG_V_B = {
    '熊_探底': {**_NAKED_TEMPLATE,
                'pool_depth_tiers': [
                    {'depth_max': None, 'days_exclude': [4, 10]},
                ]},
    '转牛': {**_NAKED_TEMPLATE,
             'pool_depth_tiers': [
                 {'depth_max': None, 'days_exclude': [4, 10]},
             ]},
    '牛_主升': {**_NAKED_TEMPLATE,
                'pool_depth_tiers': [
                    {'depth_max': None, 'days_exclude': [4, 10]},
                ]},
    '转熊': {**_NAKED_TEMPLATE, 'buy': False},
}

STRATEGY_TEST10_VA_BY_Y_GUA = _expand_season_cfg(SEASON_CFG_V_A)
STRATEGY_TEST10_VB_BY_Y_GUA = _expand_season_cfg(SEASON_CFG_V_B)


# ============================================================
# test11: 跳出 4 季合并冲突, 回到 8 卦 + 跨桶共识规律
# ============================================================
# 三爻 8 状态强行映射 4 季有数学冲突 (010/101 是"中乱"状态, 不属于顺序季节).
# 解法: 不做季合并, 8 卦各自独立, 仅应用真底层共识规律.
#
# 规律来源 (test8/test9 LOO+add-one 双向消融):
#   1. 跨 4 大样本桶共识: 排 [4-10] 磨底列 (池天结构性陷阱, 不依赖 regime)
#   2. 单桶最强 alpha: 101 离整桶 skip (LOO +2.84 ★ + add-one -9.47 ✗反)
#
# 100/110 (中样本, mean 接近 0) 不 skip — 它们承担"中性占位", skip 反而损失.
# 1+3+1+3 错位合并的 v_b 已证 [4-10] 排是真规律, 这版去掉错配只留真信号.
TEST11_MAX_POS = 5
TEST11_DAILY_LIMIT = 1

_NAKED_PLUS_NO_46 = {**_NAKED_TEMPLATE,
                      'pool_depth_tiers': [
                          {'depth_max': None, 'days_exclude': [4, 10]},
                      ]}

# v_clean: 8 卦各自排 [4-10] + 唯一 101 整桶 skip
STRATEGY_TEST11_CLEAN_BY_Y_GUA = {
    '000': {**_NAKED_PLUS_NO_46},
    '001': {**_NAKED_PLUS_NO_46},
    '010': {**_NAKED_PLUS_NO_46},
    '011': {**_NAKED_PLUS_NO_46},
    '100': {**_NAKED_PLUS_NO_46},
    '101': {**_NAKED_TEMPLATE, 'buy': False},  # 单点 skip
    '110': {**_NAKED_PLUS_NO_46},
    '111': {**_NAKED_PLUS_NO_46},
}

# v_only_46: 仅排 [4-10], 不 skip 101 (隔离测试 [4-10] 共识规律的纯净 alpha)
STRATEGY_TEST11_ONLY46_BY_Y_GUA = {
    yg: {**_NAKED_PLUS_NO_46} for yg in
    ['000', '001', '010', '011', '100', '101', '110', '111']
}


# ============================================================
# test12: 变爻 (from→to) 框架 - Phase 1 最小验证
# ============================================================
# 来源: test13_yao_change_ablation.py 的 LOO + add-one 双向消融
# 双向 ★ 真规律 (n>=1900):
#   111->101 LOO ★ +2.28 / add-one ✗反 -8.53 (主升中段反, 假反弹大坑)
#   001->000 LOO ★ +0.50 / add-one ✗反 -3.14 (短期反弹失败回熊)
#
# v_yao_min: 仅 skip 这 2 条双向 ★ (4827 sig, 占全量 35%)
# 关键: 变爻按"今日 y_gua + 上次切换前的 y_gua" 路由, 不依赖当前卦的静态分类
TEST12_MAX_POS = 5
TEST12_DAILY_LIMIT = 1

_YAO_MIN_SKIP = {'111->101', '001->000'}

STRATEGY_TEST12_YAO_MIN_BY_Y_GUA = {
    yg: {**_NAKED_TEMPLATE, 'change_type_skip': _YAO_MIN_SKIP}
    for yg in ['000', '001', '010', '011', '100', '101', '110', '111']
}


# ============================================================
# 加载入口
# ============================================================
_REGISTRY = {
    'test1': {'strategy': STRATEGY_TEST1, 'max_pos': TEST1_MAX_POS, 'daily_limit': TEST1_DAILY_LIMIT},
    'test2': {'strategy': STRATEGY_TEST2, 'max_pos': TEST2_MAX_POS, 'daily_limit': TEST2_DAILY_LIMIT},
    'test3': {'strategy': STRATEGY_TEST3, 'max_pos': TEST3_MAX_POS, 'daily_limit': TEST3_DAILY_LIMIT},
    'test4': {'strategy': STRATEGY_TEST4, 'max_pos': TEST4_MAX_POS, 'daily_limit': TEST4_DAILY_LIMIT},
    'test5': {'strategy': STRATEGY_TEST5_BY_Y_GUA, 'max_pos': TEST5_MAX_POS, 'daily_limit': TEST5_DAILY_LIMIT},
    'test6': {'strategy': STRATEGY_TEST6_BY_Y_GUA, 'max_pos': TEST6_MAX_POS, 'daily_limit': TEST6_DAILY_LIMIT},
    'test7v1': {'strategy': STRATEGY_TEST7_V1_BY_Y_GUA, 'max_pos': TEST7_MAX_POS, 'daily_limit': TEST7_DAILY_LIMIT},
    'test7v2': {'strategy': STRATEGY_TEST7_V2_BY_Y_GUA, 'max_pos': TEST7_MAX_POS, 'daily_limit': TEST7_DAILY_LIMIT},
    'test7v3': {'strategy': STRATEGY_TEST7_V3_BY_Y_GUA, 'max_pos': TEST7_MAX_POS, 'daily_limit': TEST7_DAILY_LIMIT},
    'test10va': {'strategy': STRATEGY_TEST10_VA_BY_Y_GUA, 'max_pos': TEST10_MAX_POS, 'daily_limit': TEST10_DAILY_LIMIT},
    'test10vb': {'strategy': STRATEGY_TEST10_VB_BY_Y_GUA, 'max_pos': TEST10_MAX_POS, 'daily_limit': TEST10_DAILY_LIMIT},
    'test11clean': {'strategy': STRATEGY_TEST11_CLEAN_BY_Y_GUA, 'max_pos': TEST11_MAX_POS, 'daily_limit': TEST11_DAILY_LIMIT},
    'test11only46': {'strategy': STRATEGY_TEST11_ONLY46_BY_Y_GUA, 'max_pos': TEST11_MAX_POS, 'daily_limit': TEST11_DAILY_LIMIT},
    'test12yaomin': {'strategy': STRATEGY_TEST12_YAO_MIN_BY_Y_GUA, 'max_pos': TEST12_MAX_POS, 'daily_limit': TEST12_DAILY_LIMIT},
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
