## 样例说明

qwen3_dynamic_manual_scope 样例：基于Qwen3 14B Decode设计，按照manual scope形式建立依赖
qwen3_dynamic_tensormap 样例：基于Qwen3 14B Decode设计，按照tensormap形式建立依赖
block_table样例：一个简单的block table实现，此样例必须使用TensorMap建立依赖，用于说明TensroMap的必要性

## qwen3样例的kernel执行时间参考数据

适用样例：qwen3_dynamic_manual_scope，qwen3_dynamic_tensormap

### a2a3上各kernel执行时间

数据来自 `merged_swimlane.json` 的 **AICore View**（test_qwen3_decode.py `-p a2a3 --enable-l2-swimlane` 一次实测）；"实例数"= AICore View 中的子任务事件数，SPMD 任务（`set_block_num(N)` > 1）按子任务数量统计。

| id | Kernel | core | 实例数 | 总耗时(us) | 均值 | min | max | median | P90 | P99 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | rmsnorm | aiv | 6 | 143.7 | 23.95 | 19.10 | 26.34 | 25.05 | 26.00 | 26.34 |
| 1 | q_proj | aic | 120 | 3127.8 | 26.06 | 22.00 | 46.80 | 22.63 | 41.54 | 46.74 |
| 2 | k_proj | aic | 48 | 872.2 | 18.17 | 12.94 | 39.98 | 14.14 | 38.68 | 39.98 |
| 3 | v_proj | aic | 48 | 858.9 | 17.89 | 13.12 | 38.78 | 14.05 | 37.14 | 38.78 |
| 4 | qk_norm | aiv | 6 | 79.2 | 13.19 | 12.58 | 14.18 | 13.03 | 13.62 | 14.18 |
| 5 | rope_kv_cache | aiv | 90 | 853.6 | 9.48 | 7.78 | 13.06 | 9.23 | 11.34 | 12.94 |
| 6 | qk_matmul | aic | 360 | 10567.0 | 29.35 | 2.58 | 65.96 | 28.04 | 50.40 | 63.62 |
| 7 | softmax | aiv | 360 | 6984.4 | 19.40 | 2.12 | 43.64 | 18.29 | 31.38 | 38.06 |
| 8 | sv_matmul | aic | 360 | 11395.2 | 31.65 | 2.78 | 61.82 | 30.59 | 55.34 | 59.74 |
| 9 | online_softmax | aiv | 360 | 7494.1 | 20.82 | 1.08 | 54.82 | 19.15 | 37.80 | 50.50 |
| 10/11 | out_proj_residual | mix | 240 | 9779.0 | 40.75 | 12.90 | 141.66 | 26.14 | 116.78 | 133.22 |
| 12 | post_rmsnorm | aiv | 6 | 146.3 | 24.39 | 19.98 | 31.54 | 23.26 | 27.06 | 31.54 |
| 13 | gate_proj | aic | 204 | 19522.8 | 95.70 | 66.84 | 139.36 | 95.29 | 101.42 | 138.32 |
| 14 | up_proj | aic | 204 | 19817.1 | 97.14 | 68.46 | 135.00 | 96.65 | 103.80 | 133.34 |
| 15 | silu | aiv | 204 | 575.9 | 2.82 | 2.12 | 4.06 | 2.76 | 3.38 | 3.72 |
| 16 | down_proj | aic | 240 | 17333.9 | 72.22 | 41.66 | 173.20 | 50.20 | 129.52 | 171.64 |
| 17 | down_proj_residual | aiv | 240 | 621.7 | 2.59 | 0.98 | 8.70 | 1.89 | 5.38 | 8.62 |
| | **TOTAL** | | **3096** | **110172.7** | 35.59 | 0.98 | 173.20 | 23.05 | 96.42 | 130.64 |

注：
- 实测子任务数验证：
  - id=6/7/8（`set_block_num(4)`）每次 launch 各 4 个子任务，90 launch × 4 = 360 ✓
  - id=10 `out_proj_residual_aic` 每次 launch 40 个子任务，6 × 40 = 240
  - id=11 `out_proj_residual_aiv` 每次 launch **80** 个子任务（非 40），6 × 80 = 480
- id=10/11 `out_proj_residual` 是同一个 mix 任务（aic+aiv 两个 lane 同步执行），合并为一行 type=mix：
  - 实例数 = aic 子任务数 = **240**（每次 launch 40 个 mix 实例，每个实例 = 1 个 aic 子任务 + 同一 AICore 上 2 个并行 aiv 子任务）
  - 单实例时间 = max(aic_dur, aiv_dur_1, aiv_dur_2)，按每个 launch 内 aic/aiv 时长降序 rank 配对计算（aiv 配对 2 个，取 max；它们在同一 AICore 的两个 vector unit 并行执行）
  - sum/mean/min/max/median/P90/P99 取自这 240 个 mix 实例时长
- 其它行：均值 = 总耗时 / 实例数；min/max/median/P90/P99 为子任务级分位数。

### AIC vs AIV 平均时长

| core | 子任务实例数 | 总耗时(us) | 平均时长(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AIC**（纯 cube） | 1584 | 83494.9 | **52.71** | 2.58 | 173.20 | 42.76 | 99.12 | 134.62 |
| **AIV**（纯 vector） | 1272 | 16898.9 | **13.29** | 0.98 | 54.82 | 10.06 | 30.40 | 43.38 |
| **MIX**（cube+vector） | 240 | 9779.0 | **40.75** | 12.90 | 141.66 | 26.14 | 116.78 | 133.22 |
| 全部 | 3096 | 110172.7 | 35.59 | 0.98 | 173.20 | 23.05 | 96.42 | 130.64 |

benchmark测试时，如果需要模拟AIC/AIV任务，可参照AIC/AIV平均时长。

## 修改历史
### 2026/5/27
增加simpler版本的样例：
· 基于Qwen3设计的dynamic_manual_scope样例
· 基于Qwen3设计的dynamic_tensormap样例
· 基于Qwen3设计的block_table样例

### 2026/5/28
补充qwen3样例的kernel执行时间参考数据

### 2026/5/30
kernel 执行时间参考数据按 `merged_swimlane.json` 的 **AICore View** 重新实测：
1. 数据源：test_qwen3_decode.py（`-p a2a3 --enable-l2-swimlane`）的 `simpler/outputs/.../merged_swimlane.json`，pid=1 AICore View 的全部 `ph='X'` 子任务事件，按 kernel 名聚合统计。
2. 口径：SPMD 任务（`set_block_num(N)`>1）按子任务数量统计，即每个 AICore View 事件为一个子任务实例。
3. id=10/11 `out_proj_residual` 是同一个 mix 任务（aic+aiv 同步执行），aic/aiv 两行合并为单行（type=mix）：实例数按 aic 子任务数 = 240（实测 aic 半每次 launch 40 个、aiv 半每次 launch 80 个，每个 mix 实例 = 1 个 aic + 同 AICore 上 2 个并行 aiv，单实例时间取三者 max，按每次 launch 内时长降序 rank 配对）。
4. 总耗时 110172.7 us 是统计样本（纯 aic + 纯 aiv 子任务 + 240 个 mix 实例时长）的累加，不是 wall-clock。

旧数据（5/28 task-launch 口径，aic/aiv 分行）：

| id | Kernel | core | 实例数 | 总耗时(us) | 均值(us) |
|---:|---|---|---:|---:|---:|
| 6 | qk_matmul | aic | 90 | 2655.0 | 29.50 |
| 7 | softmax | aiv | 90 | 1801.0 | 20.01 |
| 8 | sv_matmul | aic | 90 | 2833.0 | 31.48 |
| 10 | out_proj_residual_aic | aic | 6 | 261.5 | 43.59 |
| 11 | out_proj_residual_aiv | aiv | 6 | 547.4 | 91.23 |
| | **TOTAL** | | **2058** | **80769.0** | 39.25 |

| core | 实例数 | 总耗时(us) | 平均时长(us) |
|---|---:|---:|---:|
| AIC | 1050 | 68467.2 | 65.21 |
| AIV | 1008 | 12301.7 | 12.20 |
| 全部 | 2058 | 80769.0 | 39.25 |
