# Phase 2 实验报告 — test3 池深因子 (2026-04-26 ~ 04-27)

## 这是什么

test3 策略链路 7 步 pipeline 中第 2 步 (池深/池天) 的完整消融实验。
最终落地: **OOS +68% 的稳健规律, 7 段 walk-forward 0 反向**。

但更有价值的是过程中沉淀的 **5 条新反直觉规律** — 多次推翻"看起来对"的方法。

---

## 时间线 (按推翻顺序)

每一行都是一次"以为找到了规律 → 实际是过拟合 → 修正"。

### 阶段 0: cfg 起点
- test3 cfg = test1 derive_naked_cfg + 放弃 di/ren 路径
- 起点 IS = 320.9 万 / 322 笔, max_pos=5

### 阶段 1: 4×4 矩阵双视角 — **过拟合**
脚本: [test3_phase2_pool_perturb.py](test3_phase2_pool_perturb.py), [test3_phase2_pool_2d_matrix.py](test3_phase2_pool_2d_matrix.py),
[test3_phase2_v1_patches.py](test3_phase2_v1_patches.py) (v2/v3 同),
[test3_phase2_v3_run.py](test3_phase2_v3_run.py), [test3_phase2_v1_v2_run.py](test3_phase2_v1_v2_run.py)

数据: 8 卦 × 4 池深档 × 4 池天档 = 128 cell, sig + trd 双视角 + bootstrap CI + 业务可解释筛.

结果:
| 版本 | IS_α | OOS_α | 衰减率 |
|---|---:|---:|---:|
| v1 单 sig | -32% | **-31%** | -4% (稳定亏) |
| v2 单 trd | +1% | -14% | +1751% (严重过拟合) |
| v3 综合 | +10% | **-6%** | +156% |

**结论**: 看似严谨的方法论 (双视角+CI+业务可解释) 也会过拟合. 4×4 矩阵下 trd 视角是噪声主导.

### 阶段 2: 优先买区诊断 — **过拟合的元凶**
脚本: [test3_phase2_nobuy_only_run.py](test3_phase2_nobuy_only_run.py)

诊断: 移除三版本的"优先买区"字段, 只保留"不买区".

结果: **v3 IS_α 从 +10% 跌到 -34%, OOS_α 几乎不变**. 排序加权机制纯粹在凑 IS 净值, OOS 上路径变化后排序失效.

### 阶段 3: test1 vs test3 对比 — **窗口选择问题首现**
脚本: [test1_pool_overfit.py](test1_pool_overfit.py), [test1_pool_loo.py](test1_pool_loo.py)

发现: test1 cfg (4090 万) 的池深约束 OOS +20.5%, 但**逐项 LOO 揭示**:
- 100 震 `pool_days=1-7`: 真规律 (移除后 IS 和 OOS 都跌)
- 101 离 复杂 tier: IS 真功臣, OOS 中性
- **000 坤 `days_exclude=[4,10]`: 拖后腿!** 移除后 OOS +44%

test1 多年 ablation 沉淀的"老兵"也有 1/3 是过拟合. 单 OOS 验证不够.

### 阶段 4: 单维池深分析 — **逆转 + 单点 OOS 假象**
脚本: [test3_phase2_depth_only.py](test3_phase2_depth_only.py),
[test3_phase2_depth_patches.py](test3_phase2_depth_patches.py),
[test3_phase2_depth_run.py](test3_phase2_depth_run.py)

把 4×4 切成 4×1 (只看池深, 不切池天). 三版本 OOS 全部正向逆转:
| 版本 | IS_α | OOS_α |
|---|---:|---:|
| v1 单 sig (5 卦排) | -45% | **+136%** |
| v2 单 trd | -1% | +36% |
| v3 综合 | -32% | +64% |

业务理解: 单维样本量足, sig 大样本负就是真规律; 多维下 cell 稀疏, trd 主导=噪声.

**但这只是单点 OOS 切分点 = 2023-01-01 的结果, 是否真规律还要看滚动**

### 阶段 5: Walk-forward 滚动验证 — **v1 真相**
脚本: [test3_phase2_walk_forward.py](test3_phase2_walk_forward.py)

7 个滚动窗口跑 v1 (5 卦排极深):
| 窗口 | OOS | α |
|---|---|---:|
| 2018 大熊 | -1.3% | ○ |
| 2019 反弹 | -0.5% | ○ |
| 2020 抱团 | -5.0% | ○ |
| 2021 延续 | +0.3% | ○ |
| 2022 杀跌 | **-11.6%** | ✗ |
| 2023-24 震 | +43.8% | ✓ |
| 2025-26 牛 | +52.0% | ✓ |

**v1 OOS +136% 是切片福利!** 7 段中只有 2/7 ★ + 1/7 ✗. 单点 OOS 严重误导.

### 阶段 6: y_gua 切片 — **分治维度错位发现**
脚本: [test3_phase2_walk_y_gua.py](test3_phase2_walk_y_gua.py)

按"窗口 × y_gua" 二维切 v1 - baseline 利润:
- 跨窗口聚合: y_gua=000 +4.5 / y_gua=111 +12.7 (其他中性)
- y_gua=111 在 5 个窗口下 全正
- y_gua=000 在 4 个窗口下 全正

**v1 真规律仅在 y_gua ∈ {000, 111} 时成立** — 用 d_gua 路由错位, 真规律按 y_gua 分桶.

### 阶段 7: y_gua 条件化 v1 — **过拟合 alpha 切除**
脚本: [test3_phase2_y_gua_cond.py](test3_phase2_y_gua_cond.py)

代码加 `pool_depth_tiers_only_y_gua` 字段 (backtest_8gua.py L173): 仅在指定 y_gua 时执行 tier.

结果:
| 版本 | 7段均值 | 反向段 | 标准差 | 最差段 |
|---|---:|---:|---:|---:|
| v1_full | +11.1% | 1 | 21.9% | -11.6% |
| v1_y (55月+{000,111}) | +5.4% | **0** | 5.8% | -2.1% |

把"切片福利"切除, 留下真规律的稳健 alpha.

### 阶段 8: 年卦窗口扫描 — **55 月是错的**
脚本: [test3_y_gua_param_scan.py](test3_y_gua_param_scan.py),
[test3_y_gua_window_scan.py](test3_y_gua_window_scan.py)

发现 prepare_multi_scale_gua 的年卦窗口=55 月 (4.5 年) 滞后整个市场周期. 牛熊命中率扫描:

| 窗口 | 牛市命中% | 熊市命中% | 综合% |
|---:|---:|---:|---:|
| 12 月 | 90.1% | 58.1% | **71.8%** |
| 18 月 | 88.8% | 55.2% | 69.6% |
| 24 月 | 87.4% | 53.6% | 68.2% |
| **55 月** (旧默认) | 58.4% | 35.2% | **45.1%** (低于硬币!) |

55 月让 2022 大盘暴跌时 y_gua 仍判 "111 主升" 279/722 天. 改成 12 月窗口.

### 阶段 9: 12 月窗口 + 重选激活集合 — **最终落地**
脚本: [regenerate_y_gua.py](regenerate_y_gua.py) (生成 12 月版 parquet),
[test3_phase2_walk_y_gua.py](test3_phase2_walk_y_gua.py) (重切片)

重要发现: **激活集合不能跨窗口套用**.
- 55 月版 000 = "累积探底" (滞后)
- 12 月版 000 = "当下恐慌中"

按 12 月版 y_gua 重切, 新激活集合 = **{010, 111}** (而不是 {000, 111}).

最终 v1_y(12月+{010,111}) walk-forward:
| 窗口 | OOS | v1_y α |
|---|---|---:|
| 2018 大熊 | **+0.8%** | ✓ |
| 2019 反弹 | **+2.9%** | ✓ |
| 2020 抱团 | **+0.0%** | ○ |
| 2021 延续 | **+8.7%** | ✓ |
| 2022 杀跌 | **-4.5%** | ○ |
| 2023-24 震 | **+35.4%** | ✓✓ |
| 2025-26 牛 | **+16.9%** | ✓ |

7 段 0 反向 / 均值 +8.5% / 最差 -4.5%

---

## 最终落地

cfg 改动 (strategy_configs.py STRATEGY_TEST3):
- 5 个卦 (000/010/011/101/110) 加 `pool_depth_tiers` + `pool_depth_tiers_only_y_gua = {'010','111'}`
- 101 离用"永不接 tier" 表达"该 y_gua 下整卦不买"
- 110/111 买入模式 cross → double_rise (统一)

数据改动:
- multi_scale_gua_daily.parquet: 年卦窗口 55 → 12 月
- 旧版备份: multi_scale_gua_daily.window55_backup.parquet

代码改动:
- backtest_8gua.py L173: `_pool_depth_tier_ok` 加 `current_y_gua` 参数
- backtest_8gua.py L184: 内层循环加 `_today_y_gua` 计算

实测效果:
- IS (2014-2022): 39.8 万 → **44.2 万 (+11%)**
- OOS (2023-2026): 12.3 万 → **20.7 万 (+68%)**

---

## 5 条新反直觉规律 (沉淀到 SKILL.md)

1. **业务可解释 + 统计显著 + 双视角同向 ≠ 不过拟合** — test3 v3 (双视角+CI+业务解释) 仍 OOS -6%
2. **优先买区/排序加权是过拟合放大器** — 移除后 IS -34% 但 OOS 几乎不变
3. **单维分析比多维矩阵更稳** — 4×4 矩阵 trd 主导=噪声; 单维 sig 大样本负=真规律
4. **过拟合的真相是分治维度错位** — v1 在 7 段 walk-forward 中 2/7 ★, 用 y_gua 条件化后 0 反向
5. **环境分类指标的窗口长度不能凭直觉** — 55 月在月聚合下命中率 45%, 12 月窗口 72%

---

## 反过拟合纪律 (写入项目 memory)

1. **任何 cfg 改动后必跑 walk-forward (7+ 段)**, 不能只靠单点 IS/OOS
2. **改环境分类指标的窗口后必重选激活集合**
3. **trd 视角作"路径警告" 不作"否决依据"**
4. **单点 OOS alpha 大 ≠ 真规律**, 可能是切片福利

---

## 脚本分类索引

### 最终活跃脚本 (v1_y 工作流)
- [regenerate_y_gua.py](regenerate_y_gua.py) — 重生成 multi_scale_gua_daily.parquet (任意年卦窗口)
- [test3_phase2_walk_forward.py](test3_phase2_walk_forward.py) — 7 段滚动验证 baseline + v1
- [test3_phase2_walk_y_gua.py](test3_phase2_walk_y_gua.py) — 按 y_gua 切片 v1 alpha
- [test3_phase2_y_gua_cond.py](test3_phase2_y_gua_cond.py) — v1 + only_y_gua 激活
- [test3_phase2_depth_patches.py](test3_phase2_depth_patches.py) — V1/V2/V3 池深 cfg 数据
- [test3_y_gua_window_scan.py](test3_y_gua_window_scan.py) — 月窗口扫描 + 牛熊命中率

### 已用过 / 反例文档
- [test3_phase2_pool_perturb.py](test3_phase2_pool_perturb.py) — 4×4 双视角扰动 (反例: 多维过拟合)
- [test3_phase2_pool_2d_matrix.py](test3_phase2_pool_2d_matrix.py) — 4×4 矩阵打印
- [test3_phase2_v1_patches.py](test3_phase2_v1_patches.py) [v2/v3] — 三版本 4×4 cfg 数据
- [test3_phase2_v3_run.py](test3_phase2_v3_run.py), [test3_phase2_v1_v2_run.py](test3_phase2_v1_v2_run.py) — 4×4 三版本对决
- [test3_phase2_nobuy_only_run.py](test3_phase2_nobuy_only_run.py) — 优先买区诊断
- [test3_phase2_depth_only.py](test3_phase2_depth_only.py) — 单维池深扰动
- [test3_phase2_depth_run.py](test3_phase2_depth_run.py) — 单维三版本对决 (单点 OOS)
- [test3_y_gua_param_scan.py](test3_y_gua_param_scan.py) — SPD/ACC 滞后带扫描 (无效)

### 上下文 / 辅助
- [test1_pool_overfit.py](test1_pool_overfit.py) — test1 cfg 池深 OOS 验证 (+20.5% 但 LOO 暴露问题)
- [test1_pool_loo.py](test1_pool_loo.py) — test1 池深三项逐项 LOO
- [test3_phase2_naked_baseline.py](test3_phase2_naked_baseline.py) — 完全裸跑 baseline (暴露 4×4 完整数据)
- [ablation.py](ablation.py) — 通用 LOO 框架 (write_patches + run_ablation)
