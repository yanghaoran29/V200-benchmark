# Flash CSA HT/LT 场景口径（A2/A3→A5）

范围仅 **DeepSeek V4 Flash CSA**。`seq_len` = `DECODE_START_POS` = **8192**。

## HT（高吞吐）

| 项 | 值 |
|----|-----|
| 定义 | 5 倍任务，粒度同主线 |
| 主线 | B=8 S=2，`NUM_QK_CORES=24` |
| 目标构图 | `N_BATCH_TILES=5`，`BATCH_TILE=8` → **B=40 S=2** |
| 占核 | 五刀可并发时 **5×24=120** AIC（AIV 240 按 1:2） |
| 本仓库代理叶 | `ht_batch100_mtp3`（5 tiles；精确 B=40 待上板） |
| 峰值主路径 | **120核仿真**（依赖满足即下发） |

## LT（低时延）

| 项 | 值 |
|----|-----|
| 定义 | 任务近等分 **切 5 份**；扫描 B∈{5,8,10,12,15} |
| 切分规则 | 见 [`task_split_5_feasibility.md`](task_split_5_feasibility.md)：&lt;5µs 不拆；原 bn≈1 **不切**；≥5µs 且 W>1 **主线保留 + 按 batch 再切**（`5×W`） |
| 切分 | `B_tile∈{⌊B/5⌋,⌈B/5⌉}`；S=2 时每刀 **T pad 到 8 倍数**；SPMD24 |
| 门闩 | 上板实测 **≥80% 逻辑任务时长 >5µs** 的最小 B |
| 禁止 | 用主线时长÷5 代替（预期 \(T_{split}>T_{base}/5\)） |
| 本仓库代理 | 旧 `lt_batch{12,16}` 仍为 tile=4；**对齐叶**：`lt_batch4`（tile=1×4）、`lt_batch5`（tile=1×5 推荐）、`lt_batch8`（tile=2×4） |
| 目标叶 | `lt_batch5_mtp7`（B=5，`BATCH_TILE=1`，`N_BATCH_TILES=5`；仅 C 类叠 batch） |

## 统计强制项

- \(N_{AIC},N_{AIV},N_{MIX}\) 分列 + 合计
- HT/LT 粒度直方图 + **前 80% 平均粒度**（升序最短 80%）
- kernel 级 **seq_len_dep**（相关/无关/弱相关）
- 精细峰值：**\(D_i\)**（下发）+ **\(C_i\)**（完成）及邻片

## 上板状态（本交付）

| 项 | 状态 |
|----|------|
| HT 目标 B=40 S=2 五刀 | **未新建叶**；用 `ht_batch100_mtp3` + **120核仿真** 交付峰值 |
| LT 目标 B∈{4,5,8} 同族切刀 | **`lt_batch4` tile=1×4、`lt_batch5` tile=1×5（推荐）、`lt_batch8` tile=2×4 均已重板**；span≈1500/1721/1762µs |
| 拆/不拆清单 | [`task_split_5_feasibility.md`](task_split_5_feasibility.md) |

精确目标叶上板后，重跑 `python artifacts/tools/sched_demand_estimate.py` 替换代理行即可。
