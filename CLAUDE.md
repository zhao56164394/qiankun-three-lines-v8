# 乾坤三线 v8.0 — 项目协作规范

量化策略回测项目, Python 3.11, 数据层 Parquet (从 CSV 迁移), 数据规模 ~10M 行级.
机器: Windows 11, 32 GB RAM, 20 核.

---

## ⚡ 写任何处理 ≥10K 行数据的代码前必须问 4 个问题

**这是"动手前的强制 checklist", 不是事后审查**. 写代码前若没在心里过完这 4 问, 不准动键盘.

### 问题 1: 数据规模有多大?

预估代码主循环 / dictcomp / groupby / merge 的"调用次数"和"扫描行数".
- 主循环 < 1K 次: 怎么写都行, 跳过后续 3 问
- 主循环 1K - 100K 次: 后续 3 问全过
- 主循环 > 100K 次: 必须**先 profile 一份小数据**, 再决定写法

### 问题 2: 这是 O(N) 还是 O(N²)?

最常见的 O(N²) 陷阱:
```python
# 错: 对每个信号都扫一遍大 DataFrame, 37648 × 116万 = 400 亿次比较
for _, row in sigs.iterrows():
    sub = traj[(traj['code'] == row['code']) & (traj['date'] == row['date'])]
```
正确:
```python
# 对: 一次 groupby, 内部按 (code, date) 已分组
for (code, date), sub in traj.groupby(['code', 'date'], sort=False):
    ...
# 或: merge / set_index + .loc 一次性 join
```

### 问题 3: pandas 数据是 CSV 来的还是 Parquet 来的?

- Parquet 来: string 列 `.values` 是 ArrowStringArray, 单值访问慢 30×. **必须 `.to_numpy()`**
- 任何场景的 string 列要做 dictcomp / `arr[i]` 索引: **一律 `.to_numpy()`**

### 问题 4: 内循环里有没有"外层已知"的转换?

`str()` / `zfill` / `isinstance` / `pd.to_datetime` 这种操作进千万次内循环就是杀手. 看自己写的内循环, 任何依赖外层变量的转换, 全部外提到外层.

---

## 🚫 反 iterrows 铁律

`df.iterrows()` 在 ≥100K 行场景下永远是错的. 不分情况, 不允许例外. 替代方案:

| 场景 | 替代 |
|---|---|
| 简单读取多列 | `.to_numpy()` 后 `for i in range(len(df))` |
| 行级别业务逻辑 | `df.to_dict('records')` 走 numpy 路径 |
| 按某列分组处理 | `for key, sub in df.groupby('col', sort=False):` |
| 行间聚合统计 | `df.groupby(...).agg(...)` 向量化 |

如果你忍不住要写 `iterrows()`, **先停下来想一遍上面 4 种替代是不是真不行**.

---

## 沟通

- **用中文回应**.
- 用户明示前 **不主动 commit**, **不主动 push**.
- **实盘代码 `live/` 不动** — 那是独立模块, 与回测/分析主链路解耦.

---

## 主链路文件

| 用途 | 文件 |
|---|---|
| 主回测 (8 卦分治) | `backtest_8gua.py` |
| 资金回测 + loaders | `backtest_capital.py` |
| 单卦消融实验入口 | `experiment_gua.py` |
| 8 卦并行消融 | `run_ablation_parallel.py` |
| 裸跑(无过滤)对照 | `backtest_8gua_naked.py` |
| 策略参数版本注册 | `strategy_configs.py` (env `STRATEGY_VERSION` 切, 默认 test1) |
| 数据每日增量更新 | `data_layer/update_daily.py` |
| 底座层(横截面/卦)更新 | `data_layer/update_foundation.py` |
| 数据读取统一入口 | `data_layer/foundation_data._load_table` |
| dashboard | `dashboard/app.py` (Streamlit), 一键启动 `start.bat` |
| 系列消融范式 | `analysis/step{1..6}_*.py`, `analysis/test{1..3}_*.py` |
| 配置一致性校验 | `verify_config.py` (CI 用 `--strict`) |
| 数据完整性校验 | `data_layer/verify_foundation.py` |
| 性能基线 | `benchmark_parquet_migration.py` |
| 重建 dashboard 快照 | `rebuild_baseline_snapshot.py` |

---

## 性能 — 写回测/分析代码时强制遵守

数据规模决定了任何在主循环或 dict 构造里的"看似无害"小开销都会放大百万倍.

### Parquet 字符串列必须 `.to_numpy()` 不是 `.values`

**Parquet 列读出来是 pyarrow ExtensionArray, `.values` 仍是 Arrow 数组, 单值 `arr[i]` 比 numpy 慢 30 倍.** CSV 时代不会出问题, 迁 Parquet 后悄无声息变慢.

```python
# 错: pyarrow 字符串列 + .values, 单值访问慢 30x
_dates = df['date'].values
m = {(_dates[i], ...): ... for i in range(len(df))}

# 对: 强制转 numpy object 数组
_dates = df['date'].to_numpy()
m = {(_dates[i], ...): ... for i in range(len(df))}
```

任何要 dict comprehension / 索引取值的循环, **string 列必须先 `.to_numpy()`**. 7.7M 行场景下这一改差 70+ 秒.

### 主循环外提

外循环已知不变的转换, 一律外提:

```python
# 错: code 在外循环已知, 每次内循环重算
for i in range(len(df)):
    stock_ctx = stock_bagua_map.get((dt_str, str(code).zfill(6)))

# 对: 外循环算一次
code_str = str(code).zfill(6)
dates_str = df['date'].to_numpy().tolist()
for i in range(1, len(df)):
    dt_str = dates_str[i]
    stock_ctx = stock_bagua_map.get((dt_str, code_str))
```

`str()`, `zfill`, `isinstance`, 类型检查 — 不要进千万次内循环.

### profile 用法

发现回测变慢, 第一时间 profile, 不要猜:

```python
import cProfile, pstats
pr = cProfile.Profile(); pr.enable()
# ... 运行 ...
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(25)
```

看 `tottime` (self time) 找真热点, **不是** `cumulative` (会被外层函数误导).

### 验收必须对比关键指标

任何性能优化必须跑一次回测对比: **终值 / 笔数 / 最大回撤 / 胜率**. 数字不一致就是回归 bug, 立即排查. 不要"代码看着没问题就 ship".

---

## 数据访问规约

### 读

- **统一走 `data_layer/foundation_data._load_table`** (Parquet 优先 CSV 兜底)
- **不要** 在新代码里直接 `pd.read_csv(foundation_file('xxx.csv'))`
- 个股全市场: `backtest_capital.load_stocks()` (内部已 Parquet 化)

### 写

- 新输出文件**双写**: `df.to_csv(...)` + `df.to_parquet(..., engine='pyarrow', compression='snappy')`
- 模式参考 `data_layer/update_foundation._sync_parquet` 和 `update_daily._save_csv_and_parquet`
- 双轨期 (1-2 周) 同时维护 CSV+Parquet, 验证一致后再删 CSV

### 上游数据源 (Windows 路径)

```
E:/BaiduSyncdisk/A股数据_zip/指数/指数_日_kline.zip
E:/BaiduSyncdisk/A股数据_zip/daily_qfq.zip
E:/BaiduSyncdisk/A股数据_zip/股票列表.csv
E:/BaiduSyncdisk/指数数据/增量数据/指数日线行情/
E:/BaiduSyncdisk/A股数据_每日指标/增量数据/每日指标/
```

---

## 内存约束 — 多进程并行

- 每 worker 内存峰值 **~8.4 GB** (stocks + daily_bagua_map + stock_bagua_map + stk_mf_map + stock_gate_map)
- **最多 2 worker** (`--workers 2`). 实测 4 worker 黑屏.
- 并行入口: `run_ablation_parallel.py`, 不要重复造轮子.

```bash
python run_ablation_parallel.py --all-gua --layer naked,pool,buy,sell --workers 2
```

---

## 数据架构

- **字段命名**: `tian_gua` (天), `ren_gua` (人), `di_gua` (地). 详见 [feedback_gua_naming](C:\Users\asus\.claude\projects\e-------v8-0\memory\feedback_gua_naming.md).
- **策略参数唯一来源**: `strategy_configs.py`, **不要在 `backtest_8gua.py` 内硬编码参数**. 详见 [feedback_config_arch](C:\Users\asus\.claude\projects\e-------v8-0\memory\feedback_config_arch.md).
- **入池统一 `-250`**, 池底深度用 `pool_depth_tiers` (按池深分档). 详见 [feedback_pool_and_grade](C:\Users\asus\.claude\projects\e-------v8-0\memory\feedback_pool_and_grade.md).
- **三才(天/地/人)是主路径**, 月卦/年卦只作 gate 开关, **不进过滤白名单**. 详见 [feedback_gua_architecture](C:\Users\asus\.claude\projects\e-------v8-0\memory\feedback_gua_architecture.md).
- **多层缓存**: `build_payload_for_cfg` 内存 + 磁盘缓存 (`data_layer/data/payload_cache/`), 数据更新自动失效. 详见 [feedback_payload_cache](C:\Users\asus\.claude\projects\e-------v8-0\memory\feedback_payload_cache.md).

---

## Windows 命令约定

- 中文输出问题: 用 `PYTHONIOENCODING=utf-8 python xxx.py` 或在脚本头加 `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)`
- 路径用正斜杠或反斜杠都行, Python 内部 `os.path.join` 处理
- bash 工具下 wmic 不可用, 用 `powershell -Command "Get-CimInstance ..."`
- 跑 dashboard: 双击 `start.bat` 或手动 `streamlit run dashboard/app.py`

---

## 代码改动后的 sanity check

```bash
# 1. 语法检查
python -c "import py_compile; py_compile.compile('xxx.py', doraise=True); print('OK')"

# 2. 配置一致性 (改了 strategy_configs / GUA_STRATEGY 必跑)
python verify_config.py --strict

# 3. 数据完整性 (改了 prepare_*/update_* 必跑)
python data_layer/verify_foundation.py

# 4. 端到端跑一遍主回测对比指标
python backtest_8gua_naked.py
```

---

## Git

- 默认分支 `master`, 主仓库 `https://github.com/zhao56164394/qiankun-three-lines-v8`
- **不主动 commit / push**, 用户明示后做
- 提交信息用中文, 不写"Generated with Claude" trailer
