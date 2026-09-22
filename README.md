# 样例说明

本目录是 **单卡 decode 服务负载** 样例。按模型分目录，叶子按档位+形状命名。

- 模型：`qwen3_decode_layer/`（Qwen3-14B 单层 decode）与 `deepseek_v4_flash_csa/`（DeepSeek V4-Flash CSA）。
- **不含 pro**。
- 每个叶子：`pypto-lib-operator/` + `simpler-operator/` + 根上泳道 / `deps.json` / `concurrency_analysis.json`。

Qwen 公开 batch 沿 `batch_pad=16` 开窗口；**basic / ht** 的 Attention 为 `ATTN_SPMD_BLOCKS=24`，**lt** 为 **120**。Flash HT batch_tile=20、LT batch_tile=4；**cube tiling**（`QR_OK`/`KV_OK`/`QPROJ_MM_N`/`Q_OUT`/`PROJ_A`/`PROJ_B_D` 等）LT/HT/basic 均对齐 `pypto-lib/models/deepseek_v4_flash_mtp`。basic 无 `N_BATCH_TILES`。

工具链见 [VERSIONS.md](VERSIONS.md)。粒度明细在 [granularity/](granularity/)。

Golden：Qwen `out` 用 `ratio_allclose(atol=rtol=0.003, max_error_ratio=0.02)`；CSA `x_out` 用 `ratio_reldiff(0.004, 0.03, 2)`，`kv_cache` 用 `ratio_allclose(atol=1e-4, rtol=1/128)`。

- [交互 HTML](index.html)
- [总对比表](#comparison)
- [1. Qwen3 decode layer](#1-qwen3-decode-layer)
- [2. Flash CSA（DeepSeek）](#2-flash-csa)
- [3. 运行方式](#3-running)
- [修改历史](#changelog)

<a id="comparison"></a>

# 总对比表（AIC / AIV / MIX vs 主线）

口径：MIX = `kernel_ids[0]>=0` 且 `[1]>=0`；MIX 物理行不计入 AIC/AIV；MIX 实例 = 同 `task_id` 上 `max(AIC,AIV)`。脚本：[artifacts/tools/swimlane_timing_table.py](../artifacts/tools/swimlane_timing_table.py)。

- Qwen 主线：`batch=16` → AIC 33.45 / AIV 6.35 / MIX 180.68
- Flash 主线：`B=4 S=2`（mtp=1）→ AIC 9.93 / AIV 6.75 / MIX 16.40
- Qwen **同形状对照**用 [`basic_batch16`](qwen3_decode_layer/basic_batch16)（batch=16 attn分桶总数=24）；[`lt_batch16`](qwen3_decode_layer/lt_batch16) 同 batch、attn分桶总数=120。ht 与 basic 每窗 SPMD 同为 24，分桶总数随窗口放大。
- Flash **主线同形+同 tiling 对照**用 [`basic_batch4_mtp1`](deepseek_v4_flash_csa/basic_batch4_mtp1)（B=4 S=2，算子对齐 `deepseek_v4_flash_mtp`）；`lt_batch4_mtp7` 是 S=8 的 serving 切分。
- **推荐方案**：Qwen `lt_batch16`、`ht_batch80`；Flash `lt_batch12_mtp7`、`ht_batch60_mtp3`（样例名加粗）。相对主线任务执行时间浮动超过 15% 的均值加粗。

| 样例 | 形状 | AIC均值 | AIV均值 | MIX均值 |
|------|------|--------:|--------:|--------:|
| 主线 Qwen3 | batch=16 | 33.45 | 6.35 | 180.68 |
| 主线 Flash CSA | B=4 S=2 | 9.93 | 6.75 | 16.40 |
|[`qwen3_decode_layer/basic_batch16`](qwen3_decode_layer/basic_batch16)|batch=16 attn分桶总数=24|33.00|**8.33**|156.94|
|**[`qwen3_decode_layer/lt_batch16`](qwen3_decode_layer/lt_batch16)**|**batch=16 attn分桶总数=120**|33.17|**7.58**|**33.46**|
|[`qwen3_decode_layer/lt_batch32`](qwen3_decode_layer/lt_batch32)|batch=32 attn分桶总数=240|33.25|**7.97**|**35.06**|
|[`qwen3_decode_layer/ht_batch64`](qwen3_decode_layer/ht_batch64)|batch=64 attn分桶总数=96|33.02|**8.17**|164.78|
|**[`qwen3_decode_layer/ht_batch80`](qwen3_decode_layer/ht_batch80)**|**batch=80 attn分桶总数=120**|32.66|**8.25**|163.52|
|[`qwen3_decode_layer/ht_batch160`](qwen3_decode_layer/ht_batch160)|batch=160 attn分桶总数=240|32.49|**8.12**|159.84|
| [`deepseek_v4_flash_csa/basic_batch4_mtp1`](deepseek_v4_flash_csa/basic_batch4_mtp1) | B=4 S=2 mtp1 flash_mtp | 10.06 | 6.91 | **12.39** |
|[`deepseek_v4_flash_csa/lt_batch4_mtp7`](deepseek_v4_flash_csa/lt_batch4_mtp7)|B=4 S=8 mtp7 flash_mtp+batch_tile4|**15.15**|**15.26**|28.82|
|[`deepseek_v4_flash_csa/lt_batch8_mtp7`](deepseek_v4_flash_csa/lt_batch8_mtp7)|B=8 S=8 mtp7 flash_mtp+batch_tile4|**14.34**|**17.06**|**23.62**|
|**[`deepseek_v4_flash_csa/lt_batch12_mtp7`](deepseek_v4_flash_csa/lt_batch12_mtp7)**|**B=12 S=8 mtp7 flash_mtp+batch_tile4**|**14.41**|**17.91**|27.28|
|[`deepseek_v4_flash_csa/lt_batch16_mtp7`](deepseek_v4_flash_csa/lt_batch16_mtp7)|B=16 S=8 mtp7 flash_mtp+batch_tile4|**14.41**|**17.62**|**22.92**|
|**[`deepseek_v4_flash_csa/ht_batch60_mtp3`](deepseek_v4_flash_csa/ht_batch60_mtp3)**|**B=60 S=4 mtp3 flash_mtp+batch_tile20**|**31.17**|**40.55**|**48.51**|
|[`deepseek_v4_flash_csa/ht_batch100_mtp3`](deepseek_v4_flash_csa/ht_batch100_mtp3)|B=100 S=4 mtp3 flash_mtp+batch_tile20|**33.48**|**47.14**|**49.06**|
|[`deepseek_v4_flash_csa/ht_batch180_mtp3`](deepseek_v4_flash_csa/ht_batch180_mtp3)|B=180 S=4 mtp3 flash_mtp+batch_tile20|**33.56**|**45.60**|**50.90**|

注释（主线 Qwen3 vs [`basic_batch16`](qwen3_decode_layer/basic_batch16)）：同形 batch=16 / ATTN=24，AIC 已对齐（375 条，33.45 vs 33.00）。`basic_batch16` 去掉了核内 `syncall`，把 Phase-0 从 MIX 的 `attn_swpipe` 里拆成独立 `attn_phase0`（+32 条 AIV），故 AIV 均值上升（6.35→8.33），MIX 均值下降（180.68→156.94；实例数仍为 24）。

<a id="1-qwen3-decode-layer"></a>

# 1. Qwen3 decode layer

## 1.1 目录与形状

六档：`basic` / `lt` / `ht`。窗口宽 `batch_pad=16`。每窗口 `ATTN_SPMD_BLOCKS`：basic/ht=**24**，lt=**120**；表内 **attn分桶总数** = 窗口数 × `ATTN_SPMD_BLOCKS`（与 MIX/`attn_swpipe` 实例数一致）。

| 目录 | 档 | batch | 窗口 | ATTN_SPMD/窗 | attn分桶总数 | 逻辑任务 |
|------|----|------:|-----:|-------------:|-------------:|---------:|
|[`qwen3_decode_layer/basic_batch16`](qwen3_decode_layer/basic_batch16)|basic|16|1|24|24|166|
|[`qwen3_decode_layer/lt_batch16`](qwen3_decode_layer/lt_batch16)|lt|16|1|120|120|166|
|[`qwen3_decode_layer/lt_batch32`](qwen3_decode_layer/lt_batch32)|lt|32|2|120|240|332|
|[`qwen3_decode_layer/ht_batch64`](qwen3_decode_layer/ht_batch64)|ht|64|4|24|96|664|
|[`qwen3_decode_layer/ht_batch80`](qwen3_decode_layer/ht_batch80)|ht|80|5|24|120|830|
|[`qwen3_decode_layer/ht_batch160`](qwen3_decode_layer/ht_batch160)|ht|160|10|24|240|1660|

`lt_batch16` 与 `basic_batch16` 公开 batch 相同，只差 Attention SPMD（120 vs 24）。ht 与 basic 同 SPMD，仅窗口数不同。

### 1.1.1 叶子包

| 目录 | PyPTO | Simpler | 泳道 / 并发 |
|------|-------|---------|-------------|
|`qwen3_decode_layer/basic_batch16/`|[run_benchmark.py](qwen3_decode_layer/basic_batch16/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/basic_batch16/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/basic_batch16/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/basic_batch16/concurrency_analysis.json)|
|`qwen3_decode_layer/lt_batch16/`|[run_benchmark.py](qwen3_decode_layer/lt_batch16/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/lt_batch16/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/lt_batch16/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/lt_batch16/concurrency_analysis.json)|
|`qwen3_decode_layer/lt_batch32/`|[run_benchmark.py](qwen3_decode_layer/lt_batch32/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/lt_batch32/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/lt_batch32/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/lt_batch32/concurrency_analysis.json)|
|`qwen3_decode_layer/ht_batch64/`|[run_benchmark.py](qwen3_decode_layer/ht_batch64/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/ht_batch64/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/ht_batch64/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/ht_batch64/concurrency_analysis.json)|
|`qwen3_decode_layer/ht_batch80/`|[run_benchmark.py](qwen3_decode_layer/ht_batch80/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/ht_batch80/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/ht_batch80/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/ht_batch80/concurrency_analysis.json)|
|`qwen3_decode_layer/ht_batch160/`|[run_benchmark.py](qwen3_decode_layer/ht_batch160/pypto-lib-operator/run_benchmark.py)|[test_qwen3_decode_layer.py](qwen3_decode_layer/ht_batch160/simpler-operator/test_qwen3_decode_layer.py)|[merged_swimlane.json](qwen3_decode_layer/ht_batch160/merged_swimlane.json) / [concurrency_analysis.json](qwen3_decode_layer/ht_batch160/concurrency_analysis.json)|

## 1.2 样例设计

| 参数 | 值 |
|------|----|
| basic | batch=16，每窗 `ATTN_SPMD_BLOCKS=24` → attn分桶总数=24 |
| lt | batch=16/32，每窗 `ATTN_SPMD_BLOCKS=120` → attn分桶总数=120/240 |
| ht | batch=64/80/160，每窗 `ATTN_SPMD_BLOCKS=24` → attn分桶总数=96/120/240 |
| `batch_pad` | 16 |
| seq_len | seed=1234，[1,4096] |

| 阶段 | 每窗口逻辑任务 × BlockDim | 说明 |
|------|---------------------------|------|
| Q / K / V | 1×50 / 1×10 / 1×10 | 独立投影 |
| Phase 0 | 1×16 | AIV norm/RoPE |
| Attention | 1×**24**（basic/ht）或 1×**120**（lt） | **MIX**（`attn_swpipe`） |
| Out / Gate / Up / SiLU / Down | 同前 | 编号后缀同类合并统计 |

### 1.2.1 各 kernel 执行时间（同类合并）

#### `qwen3_decode_layer/basic_batch16`（batch=16 attn分桶总数=24）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_proj` | AIC | 85 | 3206.48 | 37.72 | 33.76 | 55.44 | 37.00 | 41.22 | 45.61 |
| `gate_proj` | AIC | 85 | 3371.90 | 39.67 | 35.86 | 48.20 | 38.66 | 44.02 | 46.89 |
| `up_proj` | AIC | 85 | 3325.56 | 39.12 | 34.94 | 47.62 | 38.46 | 43.13 | 45.72 |
| `out_proj` | AIC | 50 | 1054.74 | 21.09 | 15.76 | 25.18 | 21.39 | 22.83 | 24.68 |
| `q_proj` | AIC | 50 | 1067.98 | 21.36 | 17.28 | 26.72 | 21.00 | 24.46 | 26.69 |
| `attn_phase0` | AIV | 32 | 365.76 | 11.43 | 10.04 | 12.56 | 11.54 | 12.38 | 12.54 |
| `attn_swpipe` | MIX | 24 | 3766.52 | 156.94 | 97.14 | 190.92 | 181.85 | 190.15 | 190.86 |
| `silu` | AIV | 17 | 98.78 | 5.81 | 5.18 | 6.96 | 5.68 | 6.74 | 6.92 |
| `k_proj` | AIC | 10 | 172.24 | 17.22 | 16.16 | 17.94 | 17.33 | 17.90 | 17.94 |
| `v_proj` | AIC | 10 | 175.82 | 17.58 | 16.54 | 18.88 | 17.40 | 18.54 | 18.85 |
| `dcr_xgamma` | AIV | 5 | 20.96 | 4.19 | 3.88 | 4.46 | 4.26 | 4.39 | 4.45 |
| `residual_rms_cast` | AIV | 5 | 24.86 | 4.97 | 3.84 | 6.46 | 4.90 | 6.13 | 6.43 |
| `x_gamma0` | AIV | 5 | 16.32 | 3.26 | 3.14 | 3.46 | 3.22 | 3.40 | 3.45 |
| `attn_out_seed` | AIV | 1 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 |
| `copy_hidden` | AIV | 1 | 9.82 | 9.82 | 9.82 | 9.82 | 9.82 | 9.82 | 9.82 |
| `copy_out` | AIV | 1 | 8.52 | 8.52 | 8.52 | 8.52 | 8.52 | 8.52 | 8.52 |
| `kv_seed` | AIV | 1 | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 | 2.58 |
| `mlp_out_seed` | AIV | 1 | 24.60 | 24.60 | 24.60 | 24.60 | 24.60 | 24.60 | 24.60 |
| `post_rms_reduce` | AIV | 1 | 17.06 | 17.06 | 17.06 | 17.06 | 17.06 | 17.06 | 17.06 |
| `q_seed` | AIV | 1 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 |
| `rms_recip` | AIV | 1 | 5.76 | 5.76 | 5.76 | 5.76 | 5.76 | 5.76 | 5.76 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 375 | 33.00 | 24 | 83.1% |
| AIV | 72 | 8.33 | 48 | 18.9% |
| MIX | 24 | 156.94 | — | — |

逻辑任务 166；物理完整性 569/569；early_dispatch=true 0；采集跨度 876.88 μs。
AIC 逻辑包络峰值 110 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 0 | 0 | 52 | 68 | 254 | 1 | 375 |
| AIC 占比 | 0.0% | 0.0% | 0.0% | 13.9% | 18.1% | 67.7% | 0.3% | 100% |
| AIV count | 16 | 22 | 32 | 1 | 1 | 0 | 0 | 72 |
| AIV 占比 | 22.2% | 30.6% | 44.4% | 1.4% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 24 |
| MIX 占比 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | █ 4
 10-15 | ███████████████ 93
 15-20 | ████████ 49
 20-30 | ████ 24
 30-50 | ████████████████████████████████████████ 255
   50+ |  0

AIV
   0-5 | ██████████████████ 18
  5-10 | ████████████████████████████████████████ 41
 10-15 | ████████████ 12
 15-20 |  0
 20-30 | █ 1
 30-50 |  0
   50+ |  0

MIX
   0-5 |  0
  5-10 |  0
 10-15 |  0
 15-20 |  0
 20-30 |  0
 30-50 |  0
   50+ | ████████████████████████████████████████ 24

```

完整 md：[granularity/case_qwen_basic_batch16.md](granularity/case_qwen_basic_batch16.md)

#### `qwen3_decode_layer/lt_batch16`（batch=16 attn分桶总数=120）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_swpipe` | MIX | 120 | 4015.70 | 33.46 | 4.48 | 70.70 | 36.71 | 48.60 | 70.04 |
| `down_proj` | AIC | 85 | 3125.82 | 36.77 | 30.20 | 50.08 | 36.08 | 40.51 | 46.20 |
| `gate_proj` | AIC | 85 | 3534.72 | 41.58 | 34.42 | 52.10 | 41.44 | 46.66 | 49.80 |
| `up_proj` | AIC | 85 | 3387.50 | 39.85 | 33.28 | 45.72 | 39.90 | 43.71 | 44.43 |
| `out_proj` | AIC | 50 | 1002.42 | 20.05 | 16.42 | 22.38 | 20.05 | 21.63 | 22.26 |
| `q_proj` | AIC | 50 | 1041.26 | 20.83 | 17.74 | 26.44 | 19.86 | 24.16 | 26.31 |
| `attn_phase0` | AIV | 32 | 309.20 | 9.66 | 7.76 | 11.68 | 9.58 | 10.79 | 11.67 |
| `silu` | AIV | 17 | 97.50 | 5.74 | 5.00 | 6.66 | 5.64 | 6.38 | 6.63 |
| `k_proj` | AIC | 10 | 166.98 | 16.70 | 16.20 | 17.50 | 16.61 | 17.10 | 17.46 |
| `v_proj` | AIC | 10 | 179.16 | 17.92 | 17.22 | 18.80 | 17.84 | 18.66 | 18.79 |
| `dcr_xgamma` | AIV | 5 | 19.38 | 3.88 | 3.66 | 4.08 | 3.98 | 4.05 | 4.08 |
| `residual_rms_cast` | AIV | 5 | 26.50 | 5.30 | 3.58 | 7.14 | 4.68 | 7.11 | 7.14 |
| `x_gamma0` | AIV | 5 | 17.38 | 3.48 | 3.24 | 3.66 | 3.50 | 3.63 | 3.66 |
| `attn_out_seed` | AIV | 1 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 |
| `copy_hidden` | AIV | 1 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 | 10.00 |
| `copy_out` | AIV | 1 | 8.66 | 8.66 | 8.66 | 8.66 | 8.66 | 8.66 | 8.66 |
| `kv_seed` | AIV | 1 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 |
| `mlp_out_seed` | AIV | 1 | 24.68 | 24.68 | 24.68 | 24.68 | 24.68 | 24.68 | 24.68 |
| `post_rms_reduce` | AIV | 1 | 19.42 | 19.42 | 19.42 | 19.42 | 19.42 | 19.42 | 19.42 |
| `q_seed` | AIV | 1 | 4.40 | 4.40 | 4.40 | 4.40 | 4.40 | 4.40 | 4.40 |
| `rms_recip` | AIV | 1 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 375 | 33.17 | 24 | 83.6% |
| AIV | 72 | 7.58 | 48 | 20.6% |
| MIX | 120 | 33.46 | — | — |

逻辑任务 166；物理完整性 857/857；early_dispatch=true 0；采集跨度 904.00 μs。
AIC 逻辑包络峰值 120 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 0 | 0 | 71 | 49 | 253 | 2 | 375 |
| AIC 占比 | 0.0% | 0.0% | 0.0% | 18.9% | 13.1% | 67.5% | 0.5% | 100% |
| AIV count | 16 | 42 | 12 | 1 | 1 | 0 | 0 | 72 |
| AIV 占比 | 22.2% | 58.3% | 16.7% | 1.4% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 2 | 12 | 10 | 8 | 17 | 63 | 8 | 120 |
| MIX 占比 | 1.7% | 10.0% | 8.3% | 6.7% | 14.2% | 52.5% | 6.7% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | █ 1
 10-15 | ██████████████ 88
 15-20 | █████████ 58
 20-30 | ████ 23
 30-50 | ████████████████████████████████████████ 254
   50+ | █ 1

AIV
   0-5 | ██████████████████████ 18
  5-10 | ████████████████████████ 20
 10-15 | ████████████████████████████████████████ 33
 15-20 |  0
 20-30 | █ 1
 30-50 |  0
   50+ |  0

MIX
   0-5 |  0
  5-10 | ██████████ 14
 10-15 | ███████ 10
 15-20 | █ 2
 20-30 | ████████████████ 22
 30-50 | ████████████████████████████████████████ 56
   50+ | ███████████ 16

```

完整 md：[granularity/case_qwen_lt_batch16.md](granularity/case_qwen_lt_batch16.md)

#### `qwen3_decode_layer/lt_batch32`（batch=32 attn分桶总数=240）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `attn_swpipe` | MIX | 240 | 8413.60 | 35.06 | 4.24 | 86.92 | 33.03 | 64.71 | 86.08 |
| `down_proj` | AIC | 170 | 6238.20 | 36.70 | 29.32 | 55.28 | 36.45 | 39.75 | 47.86 |
| `gate_proj` | AIC | 170 | 7063.64 | 41.55 | 34.82 | 55.96 | 41.27 | 46.00 | 54.18 |
| `up_proj` | AIC | 170 | 6938.74 | 40.82 | 33.76 | 51.50 | 41.08 | 45.60 | 49.71 |
| `out_proj` | AIC | 100 | 2187.00 | 21.87 | 19.98 | 27.22 | 21.80 | 23.16 | 25.52 |
| `q_proj` | AIC | 100 | 1827.74 | 18.28 | 13.54 | 26.80 | 17.37 | 23.00 | 26.56 |
| `attn_phase0` | AIV | 64 | 651.44 | 10.18 | 7.70 | 13.10 | 10.14 | 11.41 | 12.97 |
| `silu` | AIV | 34 | 203.98 | 6.00 | 4.74 | 7.94 | 5.95 | 6.75 | 7.70 |
| `k_proj` | AIC | 20 | 331.04 | 16.55 | 10.56 | 21.20 | 16.96 | 18.53 | 20.74 |
| `v_proj` | AIC | 20 | 354.30 | 17.71 | 10.26 | 22.66 | 17.42 | 20.79 | 22.42 |
| `dcr_xgamma` | AIV | 10 | 44.42 | 4.44 | 3.96 | 4.82 | 4.48 | 4.80 | 4.82 |
| `residual_rms_cast` | AIV | 10 | 60.24 | 6.02 | 4.78 | 7.74 | 5.90 | 7.06 | 7.67 |
| `x_gamma0` | AIV | 10 | 28.56 | 2.86 | 2.38 | 3.60 | 2.74 | 3.42 | 3.58 |
| `attn_out_seed` | AIV | 2 | 1.02 | 0.51 | 0.26 | 0.76 | 0.51 | 0.71 | 0.76 |
| `copy_hidden` | AIV | 2 | 22.38 | 11.19 | 10.02 | 12.36 | 11.19 | 12.13 | 12.34 |
| `copy_out` | AIV | 2 | 27.88 | 13.94 | 8.34 | 19.54 | 13.94 | 18.42 | 19.43 |
| `kv_seed` | AIV | 2 | 5.64 | 2.82 | 2.62 | 3.02 | 2.82 | 2.98 | 3.02 |
| `mlp_out_seed` | AIV | 2 | 49.48 | 24.74 | 24.50 | 24.98 | 24.74 | 24.93 | 24.98 |
| `post_rms_reduce` | AIV | 2 | 33.06 | 16.53 | 14.22 | 18.84 | 16.53 | 18.38 | 18.79 |
| `q_seed` | AIV | 2 | 8.18 | 4.09 | 3.72 | 4.46 | 4.09 | 4.39 | 4.45 |
| `rms_recip` | AIV | 2 | 11.22 | 5.61 | 5.48 | 5.74 | 5.61 | 5.71 | 5.74 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 750 | 33.25 | 24 | 85.3% |
| AIV | 144 | 7.97 | 48 | 22.1% |
| MIX | 240 | 35.06 | — | — |

逻辑任务 332；物理完整性 1714/1714；early_dispatch=true 0；采集跨度 1693.58 μs。
AIC 逻辑包络峰值 190 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 0 | 20 | 89 | 132 | 501 | 8 | 750 |
| AIC 占比 | 0.0% | 0.0% | 2.7% | 11.9% | 17.6% | 66.8% | 1.1% | 100% |
| AIV count | 28 | 73 | 39 | 2 | 2 | 0 | 0 | 144 |
| AIV 占比 | 19.4% | 50.7% | 27.1% | 1.4% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 6 | 21 | 9 | 19 | 52 | 98 | 35 | 240 |
| MIX 占比 | 2.5% | 8.8% | 3.8% | 7.9% | 21.7% | 40.8% | 14.6% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | █ 17
 10-15 | █████████████████ 211
 15-20 | ███████ 86
 20-30 | ██ 30
 30-50 | ████████████████████████████████████████ 503
   50+ | █ 3

AIV
   0-5 | █████████████████████████████ 38
  5-10 | ███████████████████████████████████████ 51
 10-15 | ████████████████████████████████████████ 52
 15-20 | █ 1
 20-30 | ██ 2
 30-50 |  0
   50+ |  0

MIX
   0-5 | █ 1
  5-10 | ███████████ 24
 10-15 | ██████ 13
 15-20 | ███████ 15
 20-30 | █████████████████████████ 56
 30-50 | ████████████████████████████████████████ 88
   50+ | ████████████████████ 43

```

完整 md：[granularity/case_qwen_lt_batch32.md](granularity/case_qwen_lt_batch32.md)

#### `qwen3_decode_layer/ht_batch64`（batch=64 attn分桶总数=96）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_proj` | AIC | 340 | 13050.14 | 38.38 | 29.42 | 50.58 | 37.34 | 45.35 | 49.61 |
| `gate_proj` | AIC | 340 | 13675.70 | 40.22 | 33.92 | 50.44 | 39.44 | 45.13 | 48.79 |
| `up_proj` | AIC | 340 | 13319.34 | 39.17 | 31.42 | 52.12 | 38.34 | 45.49 | 49.44 |
| `out_proj` | AIC | 200 | 4036.52 | 20.18 | 15.90 | 24.58 | 19.96 | 23.11 | 24.38 |
| `q_proj` | AIC | 200 | 3979.06 | 19.90 | 9.74 | 26.60 | 19.67 | 24.10 | 25.34 |
| `attn_phase0` | AIV | 128 | 1353.84 | 10.58 | 5.86 | 17.80 | 10.07 | 15.56 | 17.26 |
| `attn_swpipe` | MIX | 96 | 15819.04 | 164.78 | 98.50 | 212.26 | 172.63 | 208.51 | 212.13 |
| `silu` | AIV | 68 | 400.20 | 5.89 | 4.78 | 7.44 | 5.87 | 6.47 | 7.28 |
| `k_proj` | AIC | 40 | 738.26 | 18.46 | 13.46 | 24.56 | 17.33 | 22.89 | 24.45 |
| `v_proj` | AIC | 40 | 732.64 | 18.32 | 13.82 | 27.60 | 17.02 | 23.55 | 27.43 |
| `dcr_xgamma` | AIV | 20 | 95.14 | 4.76 | 3.68 | 7.16 | 4.38 | 6.76 | 7.12 |
| `residual_rms_cast` | AIV | 20 | 103.80 | 5.19 | 3.26 | 7.36 | 5.59 | 6.59 | 7.26 |
| `x_gamma0` | AIV | 20 | 72.06 | 3.60 | 1.68 | 5.70 | 3.43 | 4.92 | 5.65 |
| `attn_out_seed` | AIV | 4 | 1.70 | 0.43 | 0.30 | 0.66 | 0.37 | 0.58 | 0.65 |
| `copy_hidden` | AIV | 4 | 51.32 | 12.83 | 9.64 | 18.22 | 11.73 | 16.56 | 18.05 |
| `copy_out` | AIV | 4 | 52.80 | 13.20 | 8.62 | 16.72 | 13.73 | 16.44 | 16.69 |
| `kv_seed` | AIV | 4 | 10.86 | 2.71 | 2.00 | 3.46 | 2.70 | 3.28 | 3.44 |
| `mlp_out_seed` | AIV | 4 | 103.64 | 25.91 | 23.96 | 28.66 | 25.51 | 27.89 | 28.58 |
| `post_rms_reduce` | AIV | 4 | 65.88 | 16.47 | 15.68 | 17.22 | 16.49 | 17.21 | 17.22 |
| `q_seed` | AIV | 4 | 15.76 | 3.94 | 3.44 | 4.54 | 3.89 | 4.38 | 4.52 |
| `rms_recip` | AIV | 4 | 24.84 | 6.21 | 5.38 | 6.92 | 6.27 | 6.92 | 6.92 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 1500 | 33.02 | 24 | 79.8% |
| AIV | 288 | 8.17 | 48 | 20.1% |
| MIX | 96 | 164.78 | — | — |

逻辑任务 664；物理完整性 2276/2276；early_dispatch=true 0；采集跨度 3465.76 μs。
AIC 逻辑包络峰值 230 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 2 | 10 | 256 | 213 | 1013 | 6 | 1500 |
| AIC 占比 | 0.0% | 0.1% | 0.7% | 17.1% | 14.2% | 67.5% | 0.4% | 100% |
| AIV count | 57 | 152 | 52 | 23 | 4 | 0 | 0 | 288 |
| AIV 占比 | 19.8% | 52.8% | 18.1% | 8.0% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 0 | 0 | 0 | 0 | 0 | 0 | 96 | 96 |
| MIX 占比 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | ██ 43
 10-15 | ████████████████ 410
 15-20 | ██████ 154
 20-30 | ███ 86
 30-50 | ████████████████████████████████████████ 1007
   50+ |  0

AIV
   0-5 | ██████████████████ 68
  5-10 | ████████████████████████████████████████ 148
 10-15 | █████████████████ 64
 15-20 | █ 4
 20-30 | █ 4
 30-50 |  0
   50+ |  0

MIX
   0-5 |  0
  5-10 |  0
 10-15 |  0
 15-20 |  0
 20-30 |  0
 30-50 |  0
   50+ | ████████████████████████████████████████ 96

```

完整 md：[granularity/case_qwen_ht_batch64.md](granularity/case_qwen_ht_batch64.md)

#### `qwen3_decode_layer/ht_batch80`（batch=80 attn分桶总数=120）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_proj` | AIC | 425 | 16252.48 | 38.24 | 29.46 | 53.84 | 37.36 | 44.20 | 49.61 |
| `gate_proj` | AIC | 425 | 17227.56 | 40.54 | 32.26 | 52.24 | 40.02 | 45.87 | 49.06 |
| `up_proj` | AIC | 425 | 16778.30 | 39.48 | 31.44 | 50.62 | 38.66 | 45.66 | 49.23 |
| `out_proj` | AIC | 250 | 5103.76 | 20.42 | 15.36 | 25.46 | 20.49 | 22.34 | 24.27 |
| `q_proj` | AIC | 250 | 4189.26 | 16.76 | 9.04 | 25.98 | 17.36 | 21.78 | 25.38 |
| `attn_phase0` | AIV | 160 | 1729.78 | 10.81 | 6.30 | 15.62 | 10.88 | 13.96 | 15.19 |
| `attn_swpipe` | MIX | 120 | 19622.50 | 163.52 | 60.50 | 214.16 | 172.12 | 210.60 | 213.93 |
| `silu` | AIV | 85 | 508.16 | 5.98 | 5.04 | 7.30 | 6.00 | 6.56 | 7.18 |
| `k_proj` | AIC | 50 | 868.76 | 17.38 | 9.90 | 26.10 | 17.25 | 20.63 | 24.48 |
| `v_proj` | AIC | 50 | 821.44 | 16.43 | 9.26 | 24.52 | 17.06 | 20.23 | 22.73 |
| `dcr_xgamma` | AIV | 25 | 110.64 | 4.43 | 3.46 | 6.66 | 4.18 | 5.54 | 6.62 |
| `residual_rms_cast` | AIV | 25 | 121.64 | 4.87 | 3.28 | 7.32 | 4.98 | 6.19 | 7.22 |
| `x_gamma0` | AIV | 25 | 83.92 | 3.36 | 1.74 | 5.18 | 3.32 | 4.43 | 5.09 |
| `attn_out_seed` | AIV | 5 | 1.98 | 0.40 | 0.20 | 0.66 | 0.32 | 0.64 | 0.66 |
| `copy_hidden` | AIV | 5 | 67.78 | 13.56 | 10.32 | 17.46 | 12.18 | 16.76 | 17.39 |
| `copy_out` | AIV | 5 | 72.38 | 14.48 | 8.98 | 19.56 | 15.18 | 18.50 | 19.45 |
| `kv_seed` | AIV | 5 | 16.20 | 3.24 | 2.96 | 3.62 | 3.04 | 3.60 | 3.62 |
| `mlp_out_seed` | AIV | 5 | 130.60 | 26.12 | 23.96 | 27.42 | 26.40 | 27.09 | 27.39 |
| `post_rms_reduce` | AIV | 5 | 70.76 | 14.15 | 10.28 | 16.10 | 14.50 | 16.06 | 16.10 |
| `q_seed` | AIV | 5 | 21.26 | 4.25 | 3.68 | 4.64 | 4.38 | 4.57 | 4.63 |
| `rms_recip` | AIV | 5 | 33.18 | 6.64 | 5.80 | 7.72 | 6.42 | 7.37 | 7.68 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 1875 | 32.66 | 24 | 80.2% |
| AIV | 360 | 8.25 | 48 | 20.0% |
| MIX | 120 | 163.52 | — | — |

逻辑任务 830；物理完整性 2845/2845；early_dispatch=true 0；采集跨度 4225.94 μs。
AIC 逻辑包络峰值 212 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 36 | 53 | 285 | 233 | 1260 | 8 | 1875 |
| AIC 占比 | 0.0% | 1.9% | 2.8% | 15.2% | 12.4% | 67.2% | 0.4% | 100% |
| AIV count | 74 | 156 | 115 | 10 | 5 | 0 | 0 | 360 |
| AIV 占比 | 20.6% | 43.3% | 31.9% | 2.8% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 0 | 0 | 0 | 0 | 0 | 0 | 120 | 120 |
| MIX 占比 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | ██ 77
 10-15 | █████████████████ 537
 15-20 | ██████ 191
 20-30 | ██ 48
 30-50 | ████████████████████████████████████████ 1268
   50+ | █ 4

AIV
   0-5 | ██████████████████ 84
  5-10 | ████████████████████████████████████████ 187
 10-15 | ████████████████ 76
 15-20 | █ 7
 20-30 | █ 6
 30-50 |  0
   50+ |  0

MIX
   0-5 |  0
  5-10 |  0
 10-15 |  0
 15-20 |  0
 20-30 |  0
 30-50 |  0
   50+ | ████████████████████████████████████████ 120

```

完整 md：[granularity/case_qwen_ht_batch80.md](granularity/case_qwen_ht_batch80.md)

#### `qwen3_decode_layer/ht_batch160`（batch=160 attn分桶总数=240）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `down_proj` | AIC | 850 | 31791.82 | 37.40 | 28.64 | 60.24 | 36.97 | 42.16 | 47.87 |
| `gate_proj` | AIC | 850 | 34060.10 | 40.07 | 32.76 | 53.70 | 39.41 | 45.67 | 50.35 |
| `up_proj` | AIC | 850 | 33341.76 | 39.23 | 30.26 | 53.90 | 38.60 | 44.54 | 49.59 |
| `out_proj` | AIC | 500 | 10262.98 | 20.53 | 15.30 | 27.84 | 20.19 | 23.38 | 27.04 |
| `q_proj` | AIC | 500 | 8690.94 | 17.38 | 8.76 | 31.32 | 17.93 | 22.91 | 28.35 |
| `attn_phase0` | AIV | 320 | 3405.74 | 10.64 | 5.72 | 18.00 | 10.51 | 13.86 | 17.28 |
| `attn_swpipe` | MIX | 240 | 38362.72 | 159.84 | 54.36 | 228.44 | 161.14 | 213.69 | 228.03 |
| `silu` | AIV | 170 | 1009.46 | 5.94 | 4.44 | 7.44 | 5.95 | 6.76 | 7.19 |
| `k_proj` | AIC | 100 | 1799.32 | 17.99 | 8.88 | 25.86 | 17.69 | 21.83 | 25.66 |
| `v_proj` | AIC | 100 | 1879.84 | 18.80 | 9.08 | 28.72 | 18.31 | 23.16 | 26.13 |
| `dcr_xgamma` | AIV | 50 | 206.00 | 4.12 | 3.20 | 6.36 | 3.98 | 5.04 | 6.24 |
| `residual_rms_cast` | AIV | 50 | 216.10 | 4.32 | 2.90 | 8.14 | 3.64 | 6.23 | 7.75 |
| `x_gamma0` | AIV | 50 | 168.46 | 3.37 | 1.88 | 4.94 | 3.27 | 4.73 | 4.93 |
| `attn_out_seed` | AIV | 10 | 4.18 | 0.42 | 0.22 | 0.70 | 0.35 | 0.68 | 0.70 |
| `copy_hidden` | AIV | 10 | 144.48 | 14.45 | 10.06 | 19.36 | 14.79 | 16.75 | 19.10 |
| `copy_out` | AIV | 10 | 141.12 | 14.11 | 8.68 | 19.32 | 14.69 | 18.74 | 19.26 |
| `kv_seed` | AIV | 10 | 29.94 | 2.99 | 1.84 | 3.80 | 3.07 | 3.49 | 3.77 |
| `mlp_out_seed` | AIV | 10 | 249.78 | 24.98 | 23.84 | 26.24 | 24.91 | 25.84 | 26.20 |
| `post_rms_reduce` | AIV | 10 | 162.24 | 16.22 | 13.64 | 19.68 | 16.03 | 17.41 | 19.45 |
| `q_seed` | AIV | 10 | 43.18 | 4.32 | 3.50 | 4.94 | 4.36 | 4.49 | 4.90 |
| `rms_recip` | AIV | 10 | 67.72 | 6.77 | 5.36 | 8.12 | 6.71 | 7.92 | 8.10 |

| 类型 | 记录/实例 | 平均(us) | 峰值并行 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 3750 | 32.49 | 24 | 81.5% |
| AIV | 720 | 8.12 | 48 | 20.8% |
| MIX | 240 | 159.84 | — | — |

逻辑任务 1660；物理完整性 5690/5690；early_dispatch=true 0；采集跨度 8280.86 μs。
AIC 逻辑包络峰值 194 blocks。

任务粒度分桶（μs，半开区间 `[lo,hi)`）：

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 0 | 64 | 90 | 575 | 474 | 2524 | 23 | 3750 |
| AIC 占比 | 0.0% | 1.7% | 2.4% | 15.3% | 12.6% | 67.3% | 0.6% | 100% |
| AIV count | 168 | 314 | 197 | 31 | 10 | 0 | 0 | 720 |
| AIV 占比 | 23.3% | 43.6% | 27.4% | 4.3% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 0 | 0 | 0 | 0 | 0 | 0 | 240 | 240 |
| MIX 占比 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 100% |

直方图：

```
AIC
   0-5 |  0
  5-10 | ████ 258
 10-15 | ██████████████████ 1112
 15-20 | ████ 248
 20-30 | █ 95
 30-50 | ████████████████████████████████████████ 2534
   50+ | █ 3

AIV
   0-5 | ███████████████████ 174
  5-10 | ████████████████████████████████████████ 367
 10-15 | ██████████████████ 161
 15-20 | █ 7
 20-30 | █ 11
 30-50 |  0
   50+ |  0

MIX
   0-5 |  0
  5-10 |  0
 10-15 |  0
 15-20 |  0
 20-30 |  0
 30-50 |  0
   50+ | ████████████████████████████████████████ 240

```

完整 md：[granularity/case_qwen_ht_batch160.md](granularity/case_qwen_ht_batch160.md)

## 1.3 六档对照

| 目录 | batch | attn分桶总数 | AIC均值 | AIV均值 | MIX均值 | 记录 AIC/AIV/MIX |
| ------ | ------: | -----: | --------: | --------: | --------: | ------------------: |
|[`qwen3_decode_layer/basic_batch16`](qwen3_decode_layer/basic_batch16)|16|24|33.00|8.33|156.94|375/72/24|
|[`qwen3_decode_layer/lt_batch16`](qwen3_decode_layer/lt_batch16)|16|120|33.17|7.58|33.46|375/72/120|
|[`qwen3_decode_layer/lt_batch32`](qwen3_decode_layer/lt_batch32)|32|240|33.25|7.97|35.06|750/144/240|
|[`qwen3_decode_layer/ht_batch64`](qwen3_decode_layer/ht_batch64)|64|96|33.02|8.17|164.78|1500/288/96|
|[`qwen3_decode_layer/ht_batch80`](qwen3_decode_layer/ht_batch80)|80|120|32.66|8.25|163.52|1875/360/120|
|[`qwen3_decode_layer/ht_batch160`](qwen3_decode_layer/ht_batch160)|160|240|32.49|8.12|159.84|3750/720/240|

<a id="2-flash-csa"></a>

# 2. Flash CSA（DeepSeek）

## 2.1 目录与形状

- **basic**：`mtp=1`（S=2），完整算子对齐 `deepseek_v4_flash_mtp`（无 batch-tile）
- **LT**：`mtp=7`（S=8），batch_tile=4 → `N_BATCH_TILES`；cube tiling 对齐 `flash_mtp`
- **HT**：`mtp=3`（S=4），batch_tile=20 → `N_BATCH_TILES`；cube tiling 对齐 `flash_mtp`

| 目录 | B | S | mtp | tile | 块数 | T/任务 | 逻辑任务 |
|------|--:|--:|----:|-----:|-----:|-------:|---------:|
| [`deepseek_v4_flash_csa/basic_batch4_mtp1`](deepseek_v4_flash_csa/basic_batch4_mtp1) | 4 | 2 | 1 | —（flash_mtp） | — | 8 | 72 |
|[`deepseek_v4_flash_csa/lt_batch4_mtp7`](deepseek_v4_flash_csa/lt_batch4_mtp7)|4|8|7|**1**|**4×1**|32|对齐重切（同 5）|
|[`deepseek_v4_flash_csa/lt_batch5_mtp7`](deepseek_v4_flash_csa/lt_batch5_mtp7)|5|8|7|**1**|**5×1**|40|已上板（推荐）|
|[`deepseek_v4_flash_csa/lt_batch8_mtp7`](deepseek_v4_flash_csa/lt_batch8_mtp7)|8|8|7|**2**|**4×2**|32|对齐重切（每 2 batch 一刀）|
|**[`deepseek_v4_flash_csa/lt_batch12_mtp7`](deepseek_v4_flash_csa/lt_batch12_mtp7)**|**12**|8|7|4|3×4|32|155|
|[`deepseek_v4_flash_csa/lt_batch16_mtp7`](deepseek_v4_flash_csa/lt_batch16_mtp7)|16|8|7|4|4×4|32|197|
|**[`deepseek_v4_flash_csa/ht_batch60_mtp3`](deepseek_v4_flash_csa/ht_batch60_mtp3)**|**60**|4|3|20|3×20|80|155|
|[`deepseek_v4_flash_csa/ht_batch100_mtp3`](deepseek_v4_flash_csa/ht_batch100_mtp3)|100|4|3|20|5×20|80|239|
|[`deepseek_v4_flash_csa/ht_batch180_mtp3`](deepseek_v4_flash_csa/ht_batch180_mtp3)|180|4|3|20|9×20|80|407|

### 2.1.1 叶子包

| 目录 | PyPTO | Simpler | 泳道 / 并发 |
|------|-------|---------|-------------|
| `deepseek_v4_flash_csa/basic_batch4_mtp1/` | [run_benchmark.py](deepseek_v4_flash_csa/basic_batch4_mtp1/pypto-lib-operator/run_benchmark.py) | [test_decode_csa.py](deepseek_v4_flash_csa/basic_batch4_mtp1/simpler-operator/test_decode_csa.py) | [merged_swimlane.json](deepseek_v4_flash_csa/basic_batch4_mtp1/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/basic_batch4_mtp1/concurrency_analysis.json) |
|`deepseek_v4_flash_csa/lt_batch4_mtp7/`|[run_benchmark.py](deepseek_v4_flash_csa/lt_batch4_mtp7/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/lt_batch4_mtp7/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/lt_batch4_mtp7/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/lt_batch4_mtp7/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/lt_batch5_mtp7/`|[run_benchmark.py](deepseek_v4_flash_csa/lt_batch5_mtp7/pypto-lib-operator/run_benchmark.py)|chip_swimlane + deps|[README](deepseek_v4_flash_csa/lt_batch5_mtp7/README.md) · [拆刀清单](granularity/task_split_5_feasibility.md)|
|`deepseek_v4_flash_csa/lt_batch8_mtp7/`|[run_benchmark.py](deepseek_v4_flash_csa/lt_batch8_mtp7/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/lt_batch8_mtp7/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/lt_batch8_mtp7/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/lt_batch8_mtp7/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/lt_batch12_mtp7/`|[run_benchmark.py](deepseek_v4_flash_csa/lt_batch12_mtp7/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/lt_batch12_mtp7/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/lt_batch12_mtp7/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/lt_batch12_mtp7/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/lt_batch16_mtp7/`|[run_benchmark.py](deepseek_v4_flash_csa/lt_batch16_mtp7/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/lt_batch16_mtp7/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/lt_batch16_mtp7/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/lt_batch16_mtp7/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/ht_batch60_mtp3/`|[run_benchmark.py](deepseek_v4_flash_csa/ht_batch60_mtp3/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/ht_batch60_mtp3/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/ht_batch60_mtp3/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/ht_batch60_mtp3/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/ht_batch100_mtp3/`|[run_benchmark.py](deepseek_v4_flash_csa/ht_batch100_mtp3/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/ht_batch100_mtp3/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/ht_batch100_mtp3/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/ht_batch100_mtp3/concurrency_analysis.json)|
|`deepseek_v4_flash_csa/ht_batch180_mtp3/`|[run_benchmark.py](deepseek_v4_flash_csa/ht_batch180_mtp3/pypto-lib-operator/run_benchmark.py)|[test_decode_csa.py](deepseek_v4_flash_csa/ht_batch180_mtp3/simpler-operator/test_decode_csa.py)|[merged_swimlane.json](deepseek_v4_flash_csa/ht_batch180_mtp3/merged_swimlane.json) / [concurrency_analysis.json](deepseek_v4_flash_csa/ht_batch180_mtp3/concurrency_analysis.json)|

## 2.2 样例设计

LT/HT 逻辑任务随 `N_BATCH_TILES` 放大（按 tile 拆 SPMD 后同名任务数 ≈ 原单任务 × tile）。`basic_batch4_mtp1` 无 batch-tile，与 `deepseek_v4_flash_mtp` 同 SPMD。MIX 主要为 `qk_pv` / `score`。HT/`basic` `ring_heap=2GiB`。

### 2.2.1 各 kernel 执行时间（同类合并）

#### `deepseek_v4_flash_csa/basic_batch4_mtp1`（B=4 S=2 mtp1；tiling=`flash_mtp`）


#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 273 | 2746.48 | 10.06 | 1.96 | 22.60 | 9.56 | 17.44 | 20.60 |
| AIV | 142 | 981.40 | 6.91 | 1.10 | 29.26 | 5.77 | 12.30 | 22.31 |
| MIX | 40 | 495.46 | 12.39 | 2.58 | 26.58 | 10.82 | 22.33 | 26.36 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `proj_a_mm` | AIC | 64 | 669.98 | 10.47 | 9.44 | 13.24 | 10.20 | 11.86 | 12.60 |
| `proj_b_mm` | AIC | 64 | 398.74 | 6.23 | 5.32 | 6.90 | 6.28 | 6.59 | 6.87 |
| `qproj_matmul` | AIC | 64 | 721.66 | 11.28 | 7.48 | 19.86 | 9.45 | 17.68 | 19.72 |
| `merge_norm` | AIV | 32 | 379.54 | 11.86 | 9.18 | 14.52 | 11.65 | 14.12 | 14.48 |
| `kv_score_proj` | AIC | 24 | 368.60 | 15.36 | 9.22 | 19.26 | 17.26 | 18.78 | 19.26 |
| `qk_pv` | MIX | 24 | 326.94 | 13.62 | 2.58 | 21.98 | 11.82 | 21.29 | 21.86 |
| `kv_proj_matmul` | AIC | 16 | 128.16 | 8.01 | 7.18 | 8.84 | 8.04 | 8.66 | 8.82 |
| `qproj_dequant_rms_nope_rope` | AIV | 16 | 108.78 | 6.80 | 5.76 | 8.12 | 6.75 | 7.80 | 8.12 |
| `qr_proj_matmul` | AIC | 16 | 212.06 | 13.25 | 12.06 | 14.08 | 13.35 | 13.98 | 14.08 |
| `qr_rope` | AIV | 16 | 77.02 | 4.81 | 3.92 | 5.78 | 4.67 | 5.70 | 5.77 |
| `score` | MIX | 16 | 168.52 | 10.53 | 3.54 | 26.58 | 5.95 | 25.88 | 26.50 |
| `hc_post` | AIV | 8 | 42.84 | 5.36 | 5.10 | 5.78 | 5.34 | 5.64 | 5.77 |
| `idx_qr_proj_dequant` | AIV | 8 | 18.38 | 2.30 | 2.06 | 2.84 | 2.25 | 2.53 | 2.81 |
| `idx_qr_proj_matmul` | AIC | 8 | 165.04 | 20.63 | 19.20 | 22.60 | 20.22 | 22.38 | 22.58 |
| `proj_b_act` | AIV | 8 | 37.12 | 4.64 | 4.04 | 5.46 | 4.50 | 5.17 | 5.43 |
| `qr_hadamard_matmul` | AIC | 8 | 19.98 | 2.50 | 1.96 | 2.94 | 2.50 | 2.84 | 2.93 |
| `qr_hadamard_quant` | AIV | 8 | 46.30 | 5.79 | 5.28 | 6.20 | 5.81 | 6.17 | 6.20 |
| `quant` | AIV | 8 | 19.14 | 2.39 | 2.22 | 2.60 | 2.36 | 2.56 | 2.60 |
| `topk` | AIV | 8 | 41.38 | 5.17 | 1.10 | 7.98 | 7.15 | 7.62 | 7.94 |
| `hc_pre_linear` | AIC | 4 | 37.10 | 9.28 | 9.06 | 9.44 | 9.30 | 9.40 | 9.44 |
| `mix_x` | AIV | 4 | 21.28 | 5.32 | 5.12 | 5.54 | 5.31 | 5.49 | 5.54 |
| `weights_proj` | AIC | 4 | 21.52 | 5.38 | 4.70 | 5.98 | 5.42 | 5.89 | 5.97 |
| `rope_interleave` | AIV | 2 | 4.30 | 2.15 | 1.78 | 2.52 | 2.15 | 2.45 | 2.51 |
| `scatter_softmax_pool` | AIV | 2 | 20.82 | 10.41 | 10.06 | 10.76 | 10.41 | 10.69 | 10.75 |
| `comb_sinkhorn` | AIV | 1 | 14.12 | 14.12 | 14.12 | 14.12 | 14.12 | 14.12 | 14.12 |
| `csa_cache_writeback` | AIV | 1 | 5.04 | 5.04 | 5.04 | 5.04 | 5.04 | 5.04 | 5.04 |
| `csa_cmp_rope` | AIV | 1 | 2.76 | 2.76 | 2.76 | 2.76 | 2.76 | 2.76 | 2.76 |
| `csa_rope_step` | AIV | 1 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 |
| `csa_slots_build_valid_qk_plan` | AIV | 1 | 5.66 | 5.66 | 5.66 | 5.66 | 5.66 | 5.66 | 5.66 |
| `hc_pre_linear_reduce` | AIV | 1 | 2.08 | 2.08 | 2.08 | 2.08 | 2.08 | 2.08 | 2.08 |
| `hc_pre_rms` | AIV | 1 | 9.68 | 9.68 | 9.68 | 9.68 | 9.68 | 9.68 | 9.68 |
| `kv_and_cache_write` | AIV | 1 | 8.72 | 8.72 | 8.72 | 8.72 | 8.72 | 8.72 | 8.72 |
| `kv_hadamard` | AIC | 1 | 3.64 | 3.64 | 3.64 | 3.64 | 3.64 | 3.64 | 3.64 |
| `kv_proj_seed` | AIV | 1 | 1.54 | 1.54 | 1.54 | 1.54 | 1.54 | 1.54 | 1.54 |
| `kv_rms_norm_rope` | AIV | 1 | 6.68 | 6.68 | 6.68 | 6.68 | 6.68 | 6.68 | 6.68 |
| `kv_touch` | AIV | 1 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| `prefetch_o_proj_w` | AIV | 1 | 29.26 | 29.26 | 29.26 | 29.26 | 29.26 | 29.26 | 29.26 |
| `q_rope_prepare` | AIV | 1 | 2.62 | 2.62 | 2.62 | 2.62 | 2.62 | 2.62 | 2.62 |
| `qr_proj_seed` | AIV | 1 | 2.02 | 2.02 | 2.02 | 2.02 | 2.02 | 2.02 | 2.02 |
| `qr_rms_norm_quant` | AIV | 1 | 4.88 | 4.88 | 4.88 | 4.88 | 4.88 | 4.88 | 4.88 |
| `qr_rope_swap_idx` | AIV | 1 | 2.04 | 2.04 | 2.04 | 2.04 | 2.04 | 2.04 | 2.04 |
| `rms_norm` | AIV | 1 | 10.84 | 10.84 | 10.84 | 10.84 | 10.84 | 10.84 | 10.84 |
| `rmsnorm_rope` | AIV | 1 | 27.72 | 27.72 | 27.72 | 27.72 | 27.72 | 27.72 | 27.72 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 12.42 | 12.42 | 12.42 | 12.42 | 12.42 | 12.42 | 12.42 |
| `rope_cs` | AIV | 1 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 | 3.34 |
| `split_pre_post` | AIV | 1 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 |
| `weights_proj_reduce` | AIV | 1 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 10 | 158 | 67 | 34 | 4 | 0 | 0 | 273 |
| AIC 占比 | 3.7% | 57.9% | 24.5% | 12.5% | 1.5% | 0.0% | 0.0% | 100% |
| AIV count | 48 | 58 | 34 | 0 | 2 | 0 | 0 | 142 |
| AIV 占比 | 33.8% | 40.8% | 23.9% | 0.0% | 1.4% | 0.0% | 0.0% | 100% |
| MIX count | 8 | 11 | 6 | 5 | 10 | 0 | 0 | 40 |
| MIX 占比 | 20.0% | 27.5% | 15.0% | 12.5% | 25.0% | 0.0% | 0.0% | 100% |

#### `deepseek_v4_flash_csa/lt_batch4_mtp7`（B=4 S=8 mtp7 batch_tile=4）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 345 | 5225.06 | 15.15 | 1.60 | 29.44 | 14.04 | 27.49 | 28.95 |
| AIV | 174 | 2654.44 | 15.26 | 1.30 | 53.48 | 16.82 | 20.81 | 34.09 |
| MIX | 80 | 2305.28 | 28.82 | 9.04 | 122.76 | 18.73 | 44.12 | 121.88 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kv_score_proj` | AIC | 72 | 689.28 | 9.57 | 8.04 | 15.86 | 8.70 | 14.14 | 15.79 |
| `merge_norm` | AIV | 64 | 1094.66 | 17.10 | 13.78 | 20.18 | 17.48 | 18.83 | 20.00 |
| `proj_a_mm` | AIC | 64 | 1759.20 | 27.49 | 25.02 | 29.44 | 27.75 | 28.66 | 29.28 |
| `proj_b_mm` | AIC | 64 | 924.08 | 14.44 | 12.80 | 16.98 | 14.29 | 15.89 | 16.80 |
| `qproj_matmul` | AIC | 64 | 1042.60 | 16.29 | 12.04 | 25.02 | 14.44 | 23.56 | 24.78 |
| `qk_pv` | MIX | 40 | 1732.02 | 43.30 | 17.14 | 122.76 | 29.85 | 104.12 | 122.32 |
| `score` | MIX | 40 | 573.26 | 14.33 | 9.04 | 21.32 | 14.78 | 19.43 | 20.99 |
| `qr_hadamard_matmul` | AIC | 32 | 90.40 | 2.82 | 1.60 | 5.10 | 2.75 | 3.42 | 4.91 |
| `kv_proj_matmul` | AIC | 16 | 152.36 | 9.52 | 7.00 | 10.62 | 9.76 | 10.45 | 10.61 |
| `qproj_dequant_rms_nope_rope` | AIV | 16 | 319.98 | 20.00 | 19.08 | 21.56 | 19.72 | 20.99 | 21.50 |
| `qr_proj_matmul` | AIC | 16 | 257.36 | 16.09 | 13.10 | 18.42 | 15.97 | 17.78 | 18.37 |
| `qr_rope` | AIV | 16 | 161.54 | 10.10 | 9.52 | 10.84 | 10.01 | 10.68 | 10.82 |
| `hc_post` | AIV | 8 | 150.18 | 18.77 | 18.50 | 19.08 | 18.78 | 19.02 | 19.07 |
| `idx_qr_proj_dequant` | AIV | 8 | 45.30 | 5.66 | 5.00 | 6.14 | 5.74 | 6.10 | 6.14 |
| `idx_qr_proj_matmul` | AIC | 8 | 211.60 | 26.45 | 25.66 | 27.64 | 26.23 | 27.54 | 27.63 |
| `proj_b_act` | AIV | 8 | 104.02 | 13.00 | 12.66 | 13.82 | 12.91 | 13.46 | 13.78 |
| `qr_hadamard_quant` | AIV | 8 | 168.04 | 21.00 | 20.66 | 21.46 | 20.93 | 21.33 | 21.45 |
| `quant` | AIV | 8 | 51.74 | 6.47 | 5.56 | 7.08 | 6.64 | 6.95 | 7.07 |
| `topk` | AIV | 8 | 192.00 | 24.00 | 19.80 | 26.68 | 26.04 | 26.64 | 26.68 |
| `hc_pre_linear` | AIC | 4 | 64.34 | 16.09 | 15.76 | 16.26 | 16.16 | 16.26 | 16.26 |
| `mix_x` | AIV | 4 | 50.28 | 12.57 | 11.98 | 13.16 | 12.57 | 13.05 | 13.15 |
| `weights_proj` | AIC | 4 | 31.18 | 7.80 | 6.62 | 8.48 | 8.04 | 8.35 | 8.47 |
| `rope_interleave` | AIV | 2 | 4.76 | 2.38 | 1.90 | 2.86 | 2.38 | 2.76 | 2.85 |
| `scatter_softmax_pool` | AIV | 2 | 49.38 | 24.69 | 23.00 | 26.38 | 24.69 | 26.04 | 26.35 |
| `comb_sinkhorn` | AIV | 1 | 53.48 | 53.48 | 53.48 | 53.48 | 53.48 | 53.48 | 53.48 |
| `csa_cache_writeback` | AIV | 1 | 9.04 | 9.04 | 9.04 | 9.04 | 9.04 | 9.04 | 9.04 |
| `csa_cmp_rope` | AIV | 1 | 5.12 | 5.12 | 5.12 | 5.12 | 5.12 | 5.12 | 5.12 |
| `csa_rope_step` | AIV | 1 | 14.24 | 14.24 | 14.24 | 14.24 | 14.24 | 14.24 | 14.24 |
| `csa_slots_build_valid_qk_plan` | AIV | 1 | 12.94 | 12.94 | 12.94 | 12.94 | 12.94 | 12.94 | 12.94 |
| `hc_pre_linear_reduce` | AIV | 1 | 3.60 | 3.60 | 3.60 | 3.60 | 3.60 | 3.60 | 3.60 |
| `hc_pre_rms` | AIV | 1 | 32.46 | 32.46 | 32.46 | 32.46 | 32.46 | 32.46 | 32.46 |
| `kv_and_cache_write` | AIV | 1 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 |
| `kv_hadamard` | AIC | 1 | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 | 2.66 |
| `kv_proj_seed` | AIV | 1 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 | 2.92 |
| `kv_rms_norm_rope` | AIV | 1 | 20.32 | 20.32 | 20.32 | 20.32 | 20.32 | 20.32 | 20.32 |
| `kv_touch` | AIV | 1 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 | 1.30 |
| `q_rope_prepare` | AIV | 1 | 8.04 | 8.04 | 8.04 | 8.04 | 8.04 | 8.04 | 8.04 |
| `qr_proj_seed` | AIV | 1 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 | 5.40 |
| `qr_rms_norm_quant` | AIV | 1 | 16.68 | 16.68 | 16.68 | 16.68 | 16.68 | 16.68 | 16.68 |
| `qr_rope_swap_idx` | AIV | 1 | 2.54 | 2.54 | 2.54 | 2.54 | 2.54 | 2.54 | 2.54 |
| `rms_norm` | AIV | 1 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 |
| `rmsnorm_rope` | AIV | 1 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 10.06 | 10.06 | 10.06 | 10.06 | 10.06 | 10.06 | 10.06 |
| `rope_cs` | AIV | 1 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 |
| `rope_cs_swap` | AIV | 1 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 | 1.76 |
| `split_pre_post` | AIV | 1 | 7.50 | 7.50 | 7.50 | 7.50 | 7.50 | 7.50 | 7.50 |
| `weights_proj_reduce` | AIV | 1 | 2.68 | 2.68 | 2.68 | 2.68 | 2.68 | 2.68 | 2.68 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 32 | 76 | 105 | 45 | 87 | 0 | 0 | 345 |
| AIC 占比 | 9.3% | 22.0% | 30.4% | 13.0% | 25.2% | 0.0% | 0.0% | 100% |
| AIV count | 10 | 28 | 37 | 71 | 25 | 2 | 1 | 174 |
| AIV 占比 | 5.7% | 16.1% | 21.3% | 40.8% | 14.4% | 1.1% | 0.6% | 100% |
| MIX count | 0 | 9 | 12 | 25 | 14 | 12 | 8 | 80 |
| MIX 占比 | 0.0% | 11.2% | 15.0% | 31.2% | 17.5% | 15.0% | 10.0% | 100% |

#### `deepseek_v4_flash_csa/lt_batch8_mtp7`（B=8 S=8 mtp7 batch_tile=4）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 617 | 8850.82 | 14.34 | 1.54 | 38.76 | 13.76 | 26.84 | 37.11 |
| AIV | 335 | 5715.74 | 17.06 | 1.40 | 53.58 | 18.84 | 24.02 | 45.29 |
| MIX | 160 | 3778.66 | 23.62 | 1.48 | 109.26 | 16.88 | 39.85 | 107.01 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 128 | 2608.56 | 20.38 | 13.46 | 26.92 | 21.66 | 23.78 | 26.14 |
| `proj_a_mm` | AIC | 128 | 3185.84 | 24.89 | 18.14 | 38.76 | 23.07 | 35.13 | 38.10 |
| `proj_b_mm` | AIC | 128 | 1609.46 | 12.57 | 9.36 | 21.44 | 12.42 | 15.17 | 21.00 |
| `qproj_matmul` | AIC | 128 | 1621.04 | 12.66 | 9.10 | 23.62 | 11.79 | 19.77 | 23.08 |
| `qk_pv` | MIX | 80 | 2850.14 | 35.63 | 1.48 | 109.26 | 27.15 | 90.99 | 108.22 |
| `score` | MIX | 80 | 928.52 | 11.61 | 2.26 | 25.18 | 10.13 | 22.24 | 24.26 |
| `kv_score_proj` | AIC | 72 | 1116.96 | 15.51 | 13.50 | 23.70 | 14.13 | 21.62 | 23.44 |
| `qr_hadamard_matmul` | AIC | 64 | 149.76 | 2.34 | 1.54 | 4.06 | 2.41 | 3.21 | 3.69 |
| `kv_proj_matmul` | AIC | 32 | 239.74 | 7.49 | 5.54 | 9.94 | 7.19 | 9.71 | 9.92 |
| `qproj_dequant_rms_nope_rope` | AIV | 32 | 641.62 | 20.05 | 18.82 | 21.82 | 19.90 | 21.12 | 21.72 |
| `qr_proj_matmul` | AIC | 32 | 418.50 | 13.08 | 9.64 | 17.32 | 11.75 | 16.85 | 17.31 |
| `qr_rope` | AIV | 32 | 326.30 | 10.20 | 9.16 | 11.74 | 10.18 | 10.95 | 11.62 |
| `hc_post` | AIV | 16 | 293.56 | 18.35 | 18.10 | 18.84 | 18.28 | 18.64 | 18.81 |
| `idx_qr_proj_dequant` | AIV | 16 | 88.34 | 5.52 | 4.98 | 6.10 | 5.49 | 5.83 | 6.06 |
| `idx_qr_proj_matmul` | AIC | 16 | 356.56 | 22.29 | 18.24 | 26.66 | 22.23 | 26.29 | 26.62 |
| `proj_b_act` | AIV | 16 | 214.68 | 13.42 | 12.42 | 14.20 | 13.51 | 13.89 | 14.17 |
| `qr_hadamard_quant` | AIV | 16 | 337.70 | 21.11 | 20.26 | 21.84 | 21.09 | 21.74 | 21.82 |
| `quant` | AIV | 16 | 102.90 | 6.43 | 5.48 | 8.48 | 6.13 | 7.62 | 8.45 |
| `topk` | AIV | 16 | 402.24 | 25.14 | 19.72 | 26.86 | 26.08 | 26.65 | 26.85 |
| `hc_pre_linear` | AIC | 8 | 107.72 | 13.47 | 9.34 | 21.08 | 13.18 | 17.61 | 20.73 |
| `mix_x` | AIV | 8 | 104.32 | 13.04 | 12.48 | 13.86 | 12.86 | 13.78 | 13.85 |
| `weights_proj` | AIC | 8 | 42.80 | 5.35 | 4.48 | 6.54 | 5.23 | 6.34 | 6.52 |
| `comb_sinkhorn` | AIV | 2 | 107.10 | 53.55 | 53.52 | 53.58 | 53.55 | 53.57 | 53.58 |
| `csa_cache_writeback` | AIV | 2 | 17.90 | 8.95 | 8.76 | 9.14 | 8.95 | 9.10 | 9.14 |
| `csa_slots_build_valid_qk_plan` | AIV | 2 | 25.72 | 12.86 | 12.52 | 13.20 | 12.86 | 13.13 | 13.19 |
| `hc_pre_linear_reduce` | AIV | 2 | 5.14 | 2.57 | 2.56 | 2.58 | 2.57 | 2.58 | 2.58 |
| `hc_pre_rms` | AIV | 2 | 68.92 | 34.46 | 32.12 | 36.80 | 34.46 | 36.33 | 36.75 |
| `kv_rms_norm_rope` | AIV | 2 | 32.44 | 16.22 | 15.68 | 16.76 | 16.22 | 16.65 | 16.75 |
| `kv_touch` | AIV | 2 | 3.04 | 1.52 | 1.44 | 1.60 | 1.52 | 1.58 | 1.60 |
| `q_rope_prepare` | AIV | 2 | 15.00 | 7.50 | 7.46 | 7.54 | 7.50 | 7.53 | 7.54 |
| `qr_rms_norm_quant` | AIV | 2 | 32.46 | 16.23 | 16.22 | 16.24 | 16.23 | 16.24 | 16.24 |
| `rms_norm` | AIV | 2 | 79.06 | 39.53 | 39.20 | 39.86 | 39.53 | 39.79 | 39.85 |
| `rope_cs` | AIV | 2 | 10.00 | 5.00 | 4.88 | 5.12 | 5.00 | 5.10 | 5.12 |
| `rope_interleave` | AIV | 2 | 7.74 | 3.87 | 3.80 | 3.94 | 3.87 | 3.93 | 3.94 |
| `scatter_softmax_pool` | AIV | 2 | 99.80 | 49.90 | 48.08 | 51.72 | 49.90 | 51.36 | 51.68 |
| `split_pre_post` | AIV | 2 | 15.38 | 7.69 | 7.44 | 7.94 | 7.69 | 7.89 | 7.94 |
| `weights_proj_reduce` | AIV | 2 | 6.10 | 3.05 | 2.66 | 3.44 | 3.05 | 3.36 | 3.43 |
| `csa_cmp_rope` | AIV | 1 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 | 5.64 |
| `csa_rope_step` | AIV | 1 | 24.78 | 24.78 | 24.78 | 24.78 | 24.78 | 24.78 | 24.78 |
| `kv_and_cache_write` | AIV | 1 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 | 5.00 |
| `kv_hadamard` | AIC | 1 | 2.44 | 2.44 | 2.44 | 2.44 | 2.44 | 2.44 | 2.44 |
| `kv_proj_seed` | AIV | 1 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 | 4.44 |
| `qr_proj_seed` | AIV | 1 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 |
| `qr_rope_swap_idx` | AIV | 1 | 1.78 | 1.78 | 1.78 | 1.78 | 1.78 | 1.78 | 1.78 |
| `rmsnorm_rope` | AIV | 1 | 4.52 | 4.52 | 4.52 | 4.52 | 4.52 | 4.52 | 4.52 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 15.14 | 15.14 | 15.14 | 15.14 | 15.14 | 15.14 | 15.14 |
| `rope_cs_swap` | AIV | 1 | 1.40 | 1.40 | 1.40 | 1.40 | 1.40 | 1.40 | 1.40 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 69 | 92 | 232 | 128 | 72 | 24 | 0 | 617 |
| AIC 占比 | 11.2% | 14.9% | 37.6% | 20.7% | 11.7% | 3.9% | 0.0% | 100% |
| AIV count | 14 | 53 | 67 | 57 | 136 | 5 | 3 | 335 |
| AIV 占比 | 4.2% | 15.8% | 20.0% | 17.0% | 40.6% | 1.5% | 0.9% | 100% |
| MIX count | 28 | 26 | 22 | 15 | 39 | 14 | 16 | 160 |
| MIX 占比 | 17.5% | 16.2% | 13.8% | 9.4% | 24.4% | 8.8% | 10.0% | 100% |

#### `deepseek_v4_flash_csa/lt_batch12_mtp7`（B=12 S=8 mtp7 batch_tile=4）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 889 | 12811.70 | 14.41 | 1.32 | 42.70 | 12.50 | 22.63 | 40.90 |
| AIV | 496 | 8881.10 | 17.91 | 1.52 | 80.46 | 19.42 | 25.69 | 41.43 |
| MIX | 240 | 6546.64 | 27.28 | 1.30 | 132.00 | 17.87 | 57.30 | 128.81 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 192 | 4198.50 | 21.87 | 13.80 | 27.34 | 23.20 | 25.76 | 27.17 |
| `proj_a_mm` | AIC | 192 | 4758.04 | 24.78 | 18.04 | 42.70 | 18.82 | 39.57 | 42.44 |
| `proj_b_mm` | AIC | 192 | 2284.58 | 11.90 | 9.20 | 22.62 | 10.33 | 15.49 | 21.42 |
| `qproj_matmul` | AIC | 192 | 2360.54 | 12.29 | 8.78 | 23.40 | 11.41 | 16.20 | 22.86 |
| `qk_pv` | MIX | 120 | 4992.02 | 41.60 | 1.30 | 132.00 | 27.29 | 106.77 | 130.60 |
| `score` | MIX | 120 | 1554.62 | 12.96 | 2.92 | 24.38 | 11.70 | 20.78 | 22.66 |
| `qr_hadamard_matmul` | AIC | 96 | 234.70 | 2.44 | 1.32 | 4.36 | 2.61 | 3.28 | 3.92 |
| `kv_score_proj` | AIC | 72 | 1526.02 | 21.19 | 19.02 | 31.58 | 19.87 | 29.45 | 31.48 |
| `kv_proj_matmul` | AIC | 48 | 324.86 | 6.77 | 5.40 | 10.24 | 6.10 | 9.48 | 10.16 |
| `qproj_dequant_rms_nope_rope` | AIV | 48 | 977.42 | 20.36 | 18.82 | 22.26 | 20.22 | 21.21 | 22.18 |
| `qr_proj_matmul` | AIC | 48 | 613.94 | 12.79 | 9.20 | 19.80 | 10.78 | 19.15 | 19.79 |
| `qr_rope` | AIV | 48 | 496.86 | 10.35 | 8.96 | 12.50 | 10.23 | 11.62 | 12.27 |
| `hc_post` | AIV | 24 | 453.78 | 18.91 | 18.34 | 19.46 | 18.86 | 19.40 | 19.45 |
| `idx_qr_proj_dequant` | AIV | 24 | 141.66 | 5.90 | 4.72 | 7.32 | 6.04 | 6.52 | 7.21 |
| `idx_qr_proj_matmul` | AIC | 24 | 505.96 | 21.08 | 17.96 | 27.00 | 19.01 | 26.17 | 26.94 |
| `proj_b_act` | AIV | 24 | 313.68 | 13.07 | 12.00 | 13.88 | 13.13 | 13.55 | 13.88 |
| `qr_hadamard_quant` | AIV | 24 | 513.22 | 21.38 | 20.72 | 22.20 | 21.35 | 21.97 | 22.18 |
| `quant` | AIV | 24 | 147.34 | 6.14 | 5.52 | 7.86 | 5.90 | 7.06 | 7.77 |
| `topk` | AIV | 24 | 591.34 | 24.64 | 19.74 | 26.94 | 26.01 | 26.49 | 26.87 |
| `hc_pre_linear` | AIC | 12 | 140.60 | 11.72 | 9.18 | 16.50 | 9.69 | 16.05 | 16.45 |
| `mix_x` | AIV | 12 | 155.58 | 12.97 | 12.58 | 13.52 | 12.82 | 13.49 | 13.52 |
| `weights_proj` | AIC | 12 | 60.10 | 5.01 | 3.92 | 6.92 | 4.46 | 6.46 | 6.87 |
| `comb_sinkhorn` | AIV | 3 | 161.28 | 53.76 | 53.74 | 53.80 | 53.74 | 53.79 | 53.80 |
| `csa_cache_writeback` | AIV | 3 | 33.84 | 11.28 | 10.80 | 12.16 | 10.88 | 11.90 | 12.13 |
| `csa_slots_build_valid_qk_plan` | AIV | 3 | 39.10 | 13.03 | 12.98 | 13.10 | 13.02 | 13.08 | 13.10 |
| `hc_pre_linear_reduce` | AIV | 3 | 8.14 | 2.71 | 2.64 | 2.78 | 2.72 | 2.77 | 2.78 |
| `hc_pre_rms` | AIV | 3 | 100.32 | 33.44 | 33.18 | 33.60 | 33.54 | 33.59 | 33.60 |
| `kv_rms_norm_rope` | AIV | 3 | 51.22 | 17.07 | 17.02 | 17.18 | 17.02 | 17.15 | 17.18 |
| `kv_touch` | AIV | 3 | 5.80 | 1.93 | 1.70 | 2.14 | 1.96 | 2.10 | 2.14 |
| `q_rope_prepare` | AIV | 3 | 25.06 | 8.35 | 8.22 | 8.56 | 8.28 | 8.50 | 8.55 |
| `qr_rms_norm_quant` | AIV | 3 | 48.50 | 16.17 | 15.92 | 16.44 | 16.14 | 16.38 | 16.43 |
| `rms_norm` | AIV | 3 | 119.72 | 39.91 | 38.74 | 40.78 | 40.20 | 40.66 | 40.77 |
| `rope_cs` | AIV | 3 | 17.36 | 5.79 | 5.54 | 6.12 | 5.70 | 6.04 | 6.11 |
| `split_pre_post` | AIV | 3 | 22.38 | 7.46 | 7.42 | 7.52 | 7.44 | 7.50 | 7.52 |
| `weights_proj_reduce` | AIV | 3 | 8.26 | 2.75 | 2.58 | 2.84 | 2.84 | 2.84 | 2.84 |
| `rope_interleave` | AIV | 2 | 8.30 | 4.15 | 3.80 | 4.50 | 4.15 | 4.43 | 4.49 |
| `scatter_softmax_pool` | AIV | 2 | 156.08 | 78.04 | 75.62 | 80.46 | 78.04 | 79.98 | 80.41 |
| `csa_cmp_rope` | AIV | 1 | 6.14 | 6.14 | 6.14 | 6.14 | 6.14 | 6.14 | 6.14 |
| `csa_rope_step` | AIV | 1 | 34.46 | 34.46 | 34.46 | 34.46 | 34.46 | 34.46 | 34.46 |
| `kv_and_cache_write` | AIV | 1 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 | 6.98 |
| `kv_hadamard` | AIC | 1 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 |
| `kv_proj_seed` | AIV | 1 | 5.16 | 5.16 | 5.16 | 5.16 | 5.16 | 5.16 | 5.16 |
| `qr_proj_seed` | AIV | 1 | 10.52 | 10.52 | 10.52 | 10.52 | 10.52 | 10.52 | 10.52 |
| `qr_rope_swap_idx` | AIV | 1 | 1.52 | 1.52 | 1.52 | 1.52 | 1.52 | 1.52 | 1.52 |
| `rmsnorm_rope` | AIV | 1 | 5.18 | 5.18 | 5.18 | 5.18 | 5.18 | 5.18 | 5.18 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 14.28 | 14.28 | 14.28 | 14.28 | 14.28 | 14.28 | 14.28 |
| `rope_cs_swap` | AIV | 1 | 2.12 | 2.12 | 2.12 | 2.12 | 2.12 | 2.12 | 2.12 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 105 | 173 | 252 | 233 | 58 | 68 | 0 | 889 |
| AIC 占比 | 11.8% | 19.5% | 28.3% | 26.2% | 6.5% | 7.6% | 0.0% | 100% |
| AIV count | 15 | 79 | 97 | 72 | 221 | 7 | 5 | 496 |
| AIV 占比 | 3.0% | 15.9% | 19.6% | 14.5% | 44.6% | 1.4% | 1.0% | 100% |
| MIX count | 12 | 35 | 55 | 35 | 54 | 22 | 27 | 240 |
| MIX 占比 | 5.0% | 14.6% | 22.9% | 14.6% | 22.5% | 9.2% | 11.2% | 100% |

#### `deepseek_v4_flash_csa/lt_batch16_mtp7`（B=16 S=8 mtp7 batch_tile=4）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 1161 | 16727.48 | 14.41 | 1.36 | 48.50 | 11.32 | 25.36 | 42.61 |
| AIV | 657 | 11576.50 | 17.62 | 1.18 | 107.24 | 19.08 | 24.92 | 42.27 |
| MIX | 320 | 7335.92 | 22.92 | 1.48 | 116.44 | 14.81 | 39.74 | 113.51 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 256 | 5330.62 | 20.82 | 13.70 | 26.66 | 21.46 | 24.89 | 26.07 |
| `proj_a_mm` | AIC | 256 | 6312.30 | 24.66 | 18.04 | 48.50 | 18.97 | 41.32 | 47.04 |
| `proj_b_mm` | AIC | 256 | 2961.40 | 11.57 | 9.12 | 28.24 | 10.30 | 15.12 | 25.24 |
| `qproj_matmul` | AIC | 256 | 3110.30 | 12.15 | 8.94 | 24.08 | 11.32 | 13.93 | 23.66 |
| `qk_pv` | MIX | 160 | 5626.18 | 35.16 | 1.48 | 116.44 | 25.91 | 97.31 | 114.87 |
| `score` | MIX | 160 | 1709.74 | 10.69 | 1.92 | 21.44 | 10.75 | 18.21 | 20.02 |
| `qr_hadamard_matmul` | AIC | 128 | 287.90 | 2.25 | 1.36 | 4.26 | 2.08 | 3.11 | 3.94 |
| `kv_score_proj` | AIC | 72 | 1963.54 | 27.27 | 24.48 | 39.90 | 25.30 | 37.76 | 39.89 |
| `kv_proj_matmul` | AIC | 64 | 423.06 | 6.61 | 5.08 | 10.00 | 6.05 | 9.55 | 9.95 |
| `qproj_dequant_rms_nope_rope` | AIV | 64 | 1299.70 | 20.31 | 18.80 | 22.08 | 20.26 | 21.41 | 21.90 |
| `qr_proj_matmul` | AIC | 64 | 751.52 | 11.74 | 9.52 | 17.84 | 10.85 | 16.34 | 17.26 |
| `qr_rope` | AIV | 64 | 697.56 | 10.90 | 8.98 | 12.48 | 10.92 | 12.05 | 12.45 |
| `hc_post` | AIV | 32 | 604.02 | 18.88 | 17.88 | 19.54 | 18.90 | 19.33 | 19.50 |
| `idx_qr_proj_dequant` | AIV | 32 | 177.46 | 5.55 | 4.64 | 6.86 | 5.51 | 6.29 | 6.83 |
| `idx_qr_proj_matmul` | AIC | 32 | 653.24 | 20.41 | 18.12 | 26.30 | 18.80 | 25.61 | 26.15 |
| `proj_b_act` | AIV | 32 | 413.66 | 12.93 | 12.02 | 14.16 | 13.08 | 13.48 | 14.06 |
| `qr_hadamard_quant` | AIV | 32 | 671.96 | 21.00 | 20.08 | 22.30 | 20.93 | 21.68 | 22.29 |
| `quant` | AIV | 32 | 196.26 | 6.13 | 5.48 | 8.46 | 5.81 | 7.47 | 8.32 |
| `topk` | AIV | 32 | 800.36 | 25.01 | 19.70 | 26.56 | 25.98 | 26.48 | 26.55 |
| `hc_pre_linear` | AIC | 16 | 186.44 | 11.65 | 9.34 | 16.68 | 10.42 | 16.33 | 16.65 |
| `mix_x` | AIV | 16 | 207.38 | 12.96 | 11.98 | 13.62 | 12.98 | 13.41 | 13.59 |
| `weights_proj` | AIC | 16 | 75.52 | 4.72 | 3.96 | 5.90 | 4.38 | 5.73 | 5.88 |
| `comb_sinkhorn` | AIV | 4 | 217.44 | 54.36 | 53.60 | 54.90 | 54.47 | 54.80 | 54.89 |
| `csa_cache_writeback` | AIV | 4 | 36.80 | 9.20 | 9.08 | 9.30 | 9.21 | 9.28 | 9.30 |
| `csa_slots_build_valid_qk_plan` | AIV | 4 | 56.22 | 14.05 | 13.74 | 14.58 | 13.95 | 14.41 | 14.56 |
| `hc_pre_linear_reduce` | AIV | 4 | 11.32 | 2.83 | 2.60 | 3.14 | 2.79 | 3.05 | 3.13 |
| `hc_pre_rms` | AIV | 4 | 140.44 | 35.11 | 33.30 | 36.46 | 35.34 | 36.33 | 36.45 |
| `kv_rms_norm_rope` | AIV | 4 | 67.72 | 16.93 | 16.76 | 17.08 | 16.94 | 17.05 | 17.08 |
| `kv_touch` | AIV | 4 | 7.52 | 1.88 | 1.56 | 2.12 | 1.92 | 2.11 | 2.12 |
| `q_rope_prepare` | AIV | 4 | 34.28 | 8.57 | 8.12 | 8.92 | 8.62 | 8.88 | 8.92 |
| `qr_rms_norm_quant` | AIV | 4 | 65.86 | 16.46 | 16.34 | 16.58 | 16.47 | 16.57 | 16.58 |
| `rms_norm` | AIV | 4 | 159.88 | 39.97 | 39.30 | 40.84 | 39.87 | 40.68 | 40.82 |
| `rope_cs` | AIV | 4 | 22.18 | 5.54 | 5.34 | 5.72 | 5.56 | 5.68 | 5.72 |
| `split_pre_post` | AIV | 4 | 27.20 | 6.80 | 6.70 | 6.92 | 6.79 | 6.90 | 6.92 |
| `weights_proj_reduce` | AIV | 4 | 9.46 | 2.36 | 2.26 | 2.42 | 2.39 | 2.42 | 2.42 |
| `rope_interleave` | AIV | 2 | 6.82 | 3.41 | 3.08 | 3.74 | 3.41 | 3.67 | 3.73 |
| `scatter_softmax_pool` | AIV | 2 | 208.62 | 104.31 | 101.38 | 107.24 | 104.31 | 106.65 | 107.18 |
| `csa_cmp_rope` | AIV | 1 | 8.08 | 8.08 | 8.08 | 8.08 | 8.08 | 8.08 | 8.08 |
| `csa_rope_step` | AIV | 1 | 44.08 | 44.08 | 44.08 | 44.08 | 44.08 | 44.08 | 44.08 |
| `kv_and_cache_write` | AIV | 1 | 8.62 | 8.62 | 8.62 | 8.62 | 8.62 | 8.62 | 8.62 |
| `kv_hadamard` | AIC | 1 | 2.26 | 2.26 | 2.26 | 2.26 | 2.26 | 2.26 | 2.26 |
| `kv_proj_seed` | AIV | 1 | 6.84 | 6.84 | 6.84 | 6.84 | 6.84 | 6.84 | 6.84 |
| `qr_proj_seed` | AIV | 1 | 12.90 | 12.90 | 12.90 | 12.90 | 12.90 | 12.90 | 12.90 |
| `qr_rope_swap_idx` | AIV | 1 | 2.14 | 2.14 | 2.14 | 2.14 | 2.14 | 2.14 | 2.14 |
| `rmsnorm_rope` | AIV | 1 | 4.82 | 4.82 | 4.82 | 4.82 | 4.82 | 4.82 | 4.82 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 17.10 | 17.10 | 17.10 | 17.10 | 17.10 | 17.10 | 17.10 |
| `rope_cs_swap` | AIV | 1 | 1.18 | 1.18 | 1.18 | 1.18 | 1.18 | 1.18 | 1.18 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 141 | 202 | 392 | 241 | 107 | 78 | 0 | 1161 |
| AIC 占比 | 12.1% | 17.4% | 33.8% | 20.8% | 9.2% | 6.7% | 0.0% | 100% |
| AIV count | 21 | 91 | 124 | 138 | 268 | 9 | 6 | 657 |
| AIV 占比 | 3.2% | 13.9% | 18.9% | 21.0% | 40.8% | 1.4% | 0.9% | 100% |
| MIX count | 51 | 47 | 63 | 55 | 52 | 20 | 32 | 320 |
| MIX 占比 | 15.9% | 14.7% | 19.7% | 17.2% | 16.2% | 6.2% | 10.0% | 100% |

#### `deepseek_v4_flash_csa/ht_batch60_mtp3`（B=60 S=4 mtp3 batch_tile=20）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 889 | 27706.28 | 31.17 | 1.90 | 89.98 | 25.56 | 47.12 | 85.73 |
| AIV | 496 | 20115.02 | 40.55 | 1.08 | 319.08 | 44.94 | 53.69 | 97.63 |
| MIX | 240 | 11643.26 | 48.51 | 1.76 | 248.14 | 32.73 | 85.67 | 243.45 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 192 | 8994.18 | 46.84 | 33.90 | 56.96 | 47.53 | 53.85 | 56.75 |
| `proj_a_mm` | AIC | 192 | 10690.90 | 55.68 | 43.06 | 89.98 | 45.29 | 83.34 | 88.82 |
| `proj_b_mm` | AIC | 192 | 4779.12 | 24.89 | 21.40 | 31.88 | 23.87 | 28.25 | 30.71 |
| `qproj_matmul` | AIC | 192 | 4976.40 | 25.92 | 20.02 | 42.64 | 24.89 | 29.38 | 41.41 |
| `qk_pv` | MIX | 120 | 9257.40 | 77.14 | 1.76 | 248.14 | 67.17 | 201.18 | 246.33 |
| `score` | MIX | 120 | 2385.86 | 19.88 | 2.26 | 41.78 | 20.18 | 38.50 | 41.14 |
| `qr_hadamard_matmul` | AIC | 96 | 352.92 | 3.68 | 1.90 | 7.16 | 3.69 | 5.41 | 6.88 |
| `kv_score_proj` | AIC | 72 | 3432.18 | 47.67 | 43.84 | 69.54 | 44.60 | 65.99 | 69.36 |
| `kv_proj_matmul` | AIC | 48 | 655.34 | 13.65 | 11.50 | 17.70 | 12.94 | 16.34 | 17.51 |
| `qproj_dequant_rms_nope_rope` | AIV | 48 | 2222.76 | 46.31 | 44.62 | 48.46 | 46.14 | 47.46 | 48.29 |
| `qr_proj_matmul` | AIC | 48 | 1252.46 | 26.09 | 22.44 | 33.28 | 24.53 | 31.28 | 32.72 |
| `qr_rope` | AIV | 48 | 1080.76 | 22.52 | 21.16 | 24.52 | 22.46 | 23.38 | 24.34 |
| `hc_post` | AIV | 24 | 1097.82 | 45.74 | 45.02 | 46.42 | 45.87 | 46.31 | 46.40 |
| `idx_qr_proj_dequant` | AIV | 24 | 284.38 | 11.85 | 10.22 | 16.44 | 10.67 | 14.87 | 16.15 |
| `idx_qr_proj_matmul` | AIC | 24 | 1114.58 | 46.44 | 42.26 | 53.00 | 43.94 | 52.59 | 52.98 |
| `proj_b_act` | AIV | 24 | 718.12 | 29.92 | 29.40 | 30.60 | 29.99 | 30.29 | 30.56 |
| `qr_hadamard_quant` | AIV | 24 | 1222.54 | 50.94 | 49.86 | 52.72 | 50.85 | 51.64 | 52.65 |
| `quant` | AIV | 24 | 363.12 | 15.13 | 12.84 | 27.72 | 13.61 | 21.05 | 27.03 |
| `topk` | AIV | 24 | 1383.10 | 57.63 | 44.96 | 64.34 | 63.09 | 64.14 | 64.34 |
| `hc_pre_linear` | AIC | 12 | 332.88 | 27.74 | 22.76 | 35.64 | 24.79 | 35.44 | 35.62 |
| `mix_x` | AIV | 12 | 361.78 | 30.15 | 29.28 | 30.66 | 30.24 | 30.48 | 30.64 |
| `weights_proj` | AIC | 12 | 112.04 | 9.34 | 8.46 | 10.86 | 9.00 | 10.28 | 10.80 |
| `comb_sinkhorn` | AIV | 3 | 396.86 | 132.29 | 131.72 | 133.20 | 131.94 | 132.95 | 133.17 |
| `csa_cache_writeback` | AIV | 3 | 79.54 | 26.51 | 25.44 | 27.84 | 26.26 | 27.52 | 27.81 |
| `csa_slots_build_valid_qk_plan` | AIV | 3 | 89.28 | 29.76 | 29.52 | 30.10 | 29.66 | 30.01 | 30.09 |
| `hc_pre_linear_reduce` | AIV | 3 | 17.74 | 5.91 | 5.76 | 6.02 | 5.96 | 6.01 | 6.02 |
| `hc_pre_rms` | AIV | 3 | 239.70 | 79.90 | 78.68 | 81.18 | 79.84 | 80.91 | 81.15 |
| `kv_rms_norm_rope` | AIV | 3 | 126.46 | 42.15 | 41.94 | 42.38 | 42.14 | 42.33 | 42.38 |
| `kv_touch` | AIV | 3 | 9.22 | 3.07 | 3.06 | 3.10 | 3.06 | 3.09 | 3.10 |
| `q_rope_prepare` | AIV | 3 | 46.68 | 15.56 | 15.50 | 15.68 | 15.50 | 15.64 | 15.68 |
| `qr_rms_norm_quant` | AIV | 3 | 116.62 | 38.87 | 38.22 | 39.30 | 39.10 | 39.26 | 39.30 |
| `rms_norm` | AIV | 3 | 283.44 | 94.48 | 93.42 | 95.84 | 94.18 | 95.51 | 95.81 |
| `rope_cs` | AIV | 3 | 29.76 | 9.92 | 9.66 | 10.06 | 10.04 | 10.06 | 10.06 |
| `split_pre_post` | AIV | 3 | 45.72 | 15.24 | 15.10 | 15.36 | 15.26 | 15.34 | 15.36 |
| `weights_proj_reduce` | AIV | 3 | 16.62 | 5.54 | 5.18 | 5.86 | 5.58 | 5.80 | 5.85 |
| `rope_interleave` | AIV | 2 | 16.72 | 8.36 | 7.92 | 8.80 | 8.36 | 8.71 | 8.79 |
| `scatter_softmax_pool` | AIV | 2 | 618.40 | 309.20 | 299.32 | 319.08 | 309.20 | 317.10 | 318.88 |
| `csa_cmp_rope` | AIV | 1 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 | 21.36 |
| `csa_rope_step` | AIV | 1 | 85.94 | 85.94 | 85.94 | 85.94 | 85.94 | 85.94 | 85.94 |
| `kv_and_cache_write` | AIV | 1 | 30.36 | 30.36 | 30.36 | 30.36 | 30.36 | 30.36 | 30.36 |
| `kv_hadamard` | AIC | 1 | 7.46 | 7.46 | 7.46 | 7.46 | 7.46 | 7.46 | 7.46 |
| `kv_proj_seed` | AIV | 1 | 11.86 | 11.86 | 11.86 | 11.86 | 11.86 | 11.86 | 11.86 |
| `qr_proj_seed` | AIV | 1 | 23.40 | 23.40 | 23.40 | 23.40 | 23.40 | 23.40 | 23.40 |
| `qr_rope_swap_idx` | AIV | 1 | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 | 1.96 |
| `rmsnorm_rope` | AIV | 1 | 13.08 | 13.08 | 13.08 | 13.08 | 13.08 | 13.08 | 13.08 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 64.66 | 64.66 | 64.66 | 64.66 | 64.66 | 64.66 | 64.66 |
| `rope_cs_swap` | AIV | 1 | 1.08 | 1.08 | 1.08 | 1.08 | 1.08 | 1.08 | 1.08 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 84 | 22 | 39 | 12 | 406 | 246 | 80 | 889 |
| AIC 占比 | 9.4% | 2.5% | 4.4% | 1.3% | 45.7% | 27.7% | 9.0% | 100% |
| AIV count | 5 | 9 | 46 | 9 | 73 | 234 | 120 | 496 |
| AIV 占比 | 1.0% | 1.8% | 9.3% | 1.8% | 14.7% | 47.2% | 24.2% | 100% |
| MIX count | 50 | 22 | 6 | 16 | 14 | 50 | 82 | 240 |
| MIX 占比 | 20.8% | 9.2% | 2.5% | 6.7% | 5.8% | 20.8% | 34.2% | 100% |

#### `deepseek_v4_flash_csa/ht_batch100_mtp3`（B=100 S=4 mtp3 batch_tile=20）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 1433 | 47972.22 | 33.48 | 1.88 | 143.46 | 24.70 | 70.82 | 131.17 |
| AIV | 818 | 38564.06 | 47.14 | 1.44 | 515.32 | 46.11 | 72.57 | 109.30 |
| MIX | 400 | 19623.74 | 49.06 | 1.68 | 249.46 | 31.16 | 88.06 | 247.84 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 320 | 20091.58 | 62.79 | 38.40 | 109.46 | 56.09 | 99.11 | 106.54 |
| `proj_a_mm` | AIC | 320 | 20257.76 | 63.31 | 43.74 | 143.46 | 50.50 | 113.32 | 138.62 |
| `proj_b_mm` | AIC | 320 | 8156.60 | 25.49 | 20.88 | 50.82 | 23.84 | 28.98 | 48.91 |
| `qproj_matmul` | AIC | 320 | 7973.00 | 24.92 | 20.56 | 38.26 | 24.36 | 27.64 | 36.37 |
| `qk_pv` | MIX | 200 | 15721.00 | 78.61 | 1.68 | 249.46 | 69.15 | 206.77 | 247.98 |
| `score` | MIX | 200 | 3902.74 | 19.51 | 2.34 | 46.52 | 18.45 | 39.37 | 44.20 |
| `qr_hadamard_matmul` | AIC | 160 | 517.56 | 3.23 | 1.88 | 7.06 | 3.06 | 4.72 | 6.00 |
| `kv_proj_matmul` | AIC | 80 | 1060.52 | 13.26 | 11.78 | 17.84 | 12.64 | 16.71 | 17.79 |
| `qproj_dequant_rms_nope_rope` | AIV | 80 | 3698.72 | 46.23 | 45.10 | 47.84 | 46.12 | 47.20 | 47.81 |
| `qr_proj_matmul` | AIC | 80 | 1977.66 | 24.72 | 22.50 | 31.08 | 23.99 | 29.42 | 31.03 |
| `qr_rope` | AIV | 80 | 1808.56 | 22.61 | 21.16 | 24.54 | 22.56 | 23.74 | 24.29 |
| `kv_score_proj` | AIC | 72 | 5536.86 | 76.90 | 71.30 | 112.44 | 72.59 | 106.23 | 111.70 |
| `hc_post` | AIV | 40 | 1839.82 | 46.00 | 45.06 | 46.98 | 46.10 | 46.54 | 46.86 |
| `idx_qr_proj_dequant` | AIV | 40 | 485.48 | 12.14 | 10.32 | 15.58 | 11.49 | 14.21 | 15.45 |
| `idx_qr_proj_matmul` | AIC | 40 | 1781.72 | 44.54 | 42.14 | 51.98 | 43.04 | 50.57 | 51.88 |
| `proj_b_act` | AIV | 40 | 1198.16 | 29.95 | 28.94 | 31.36 | 29.87 | 30.63 | 31.17 |
| `qr_hadamard_quant` | AIV | 40 | 2031.74 | 50.79 | 49.58 | 52.00 | 50.84 | 51.68 | 51.98 |
| `quant` | AIV | 40 | 584.14 | 14.60 | 12.88 | 31.36 | 13.48 | 15.38 | 30.91 |
| `topk` | AIV | 40 | 2293.52 | 57.34 | 44.72 | 63.88 | 62.58 | 63.52 | 63.84 |
| `hc_pre_linear` | AIC | 20 | 513.34 | 25.67 | 21.52 | 36.36 | 23.05 | 36.11 | 36.33 |
| `mix_x` | AIV | 20 | 604.00 | 30.20 | 29.48 | 31.32 | 30.18 | 30.77 | 31.23 |
| `weights_proj` | AIC | 20 | 184.14 | 9.21 | 8.46 | 10.94 | 8.92 | 10.42 | 10.87 |
| `comb_sinkhorn` | AIV | 5 | 654.78 | 130.96 | 130.12 | 132.12 | 130.72 | 131.78 | 132.09 |
| `csa_cache_writeback` | AIV | 5 | 109.32 | 21.86 | 21.14 | 22.52 | 21.86 | 22.41 | 22.51 |
| `csa_slots_build_valid_qk_plan` | AIV | 5 | 153.18 | 30.64 | 30.14 | 31.26 | 30.38 | 31.24 | 31.26 |
| `hc_pre_linear_reduce` | AIV | 5 | 28.34 | 5.67 | 5.46 | 6.00 | 5.64 | 5.90 | 5.99 |
| `hc_pre_rms` | AIV | 5 | 416.96 | 83.39 | 78.70 | 88.52 | 81.72 | 88.14 | 88.48 |
| `kv_rms_norm_rope` | AIV | 5 | 199.62 | 39.92 | 39.54 | 40.32 | 39.94 | 40.30 | 40.32 |
| `kv_touch` | AIV | 5 | 22.50 | 4.50 | 3.52 | 5.18 | 4.56 | 4.98 | 5.16 |
| `q_rope_prepare` | AIV | 5 | 76.76 | 15.35 | 15.04 | 15.52 | 15.46 | 15.51 | 15.52 |
| `qr_rms_norm_quant` | AIV | 5 | 194.98 | 39.00 | 38.68 | 39.18 | 39.12 | 39.16 | 39.18 |
| `rms_norm` | AIV | 5 | 472.94 | 94.59 | 92.64 | 95.80 | 94.56 | 95.78 | 95.80 |
| `rope_cs` | AIV | 5 | 50.10 | 10.02 | 9.70 | 10.28 | 10.02 | 10.23 | 10.28 |
| `split_pre_post` | AIV | 5 | 77.56 | 15.51 | 15.04 | 16.08 | 15.44 | 15.89 | 16.06 |
| `weights_proj_reduce` | AIV | 5 | 24.50 | 4.90 | 4.56 | 5.36 | 4.84 | 5.24 | 5.35 |
| `rope_interleave` | AIV | 2 | 23.60 | 11.80 | 11.50 | 12.10 | 11.80 | 12.04 | 12.09 |
| `scatter_softmax_pool` | AIV | 2 | 1009.76 | 504.88 | 494.44 | 515.32 | 504.88 | 513.23 | 515.11 |
| `csa_cmp_rope` | AIV | 1 | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| `csa_rope_step` | AIV | 1 | 141.96 | 141.96 | 141.96 | 141.96 | 141.96 | 141.96 | 141.96 |
| `kv_and_cache_write` | AIV | 1 | 50.22 | 50.22 | 50.22 | 50.22 | 50.22 | 50.22 | 50.22 |
| `kv_hadamard` | AIC | 1 | 13.06 | 13.06 | 13.06 | 13.06 | 13.06 | 13.06 | 13.06 |
| `kv_proj_seed` | AIV | 1 | 20.08 | 20.08 | 20.08 | 20.08 | 20.08 | 20.08 | 20.08 |
| `qr_proj_seed` | AIV | 1 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 | 38.50 |
| `qr_rope_swap_idx` | AIV | 1 | 1.92 | 1.92 | 1.92 | 1.92 | 1.92 | 1.92 | 1.92 |
| `rmsnorm_rope` | AIV | 1 | 21.24 | 21.24 | 21.24 | 21.24 | 21.24 | 21.24 | 21.24 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 106.08 | 106.08 | 106.08 | 106.08 | 106.08 | 106.08 | 106.08 |
| `rope_cs_swap` | AIV | 1 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 | 1.44 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 150 | 26 | 73 | 12 | 681 | 240 | 251 | 1433 |
| AIC 占比 | 10.5% | 1.8% | 5.1% | 0.8% | 47.5% | 16.7% | 17.5% | 100% |
| AIV count | 9 | 10 | 77 | 16 | 118 | 280 | 308 | 818 |
| AIV 占比 | 1.1% | 1.2% | 9.4% | 2.0% | 14.4% | 34.2% | 37.7% | 100% |
| MIX count | 71 | 51 | 7 | 22 | 48 | 62 | 139 | 400 |
| MIX 占比 | 17.8% | 12.8% | 1.8% | 5.5% | 12.0% | 15.5% | 34.8% | 100% |

#### `deepseek_v4_flash_csa/ht_batch180_mtp3`（B=180 S=4 mtp3 batch_tile=20）



#### AIC / AIV / MIX 平均时间（MIX 不计入 AIC/AIV）

| 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC | 2521 | 84602.32 | 33.56 | 1.84 | 194.34 | 24.22 | 63.06 | 132.09 |
| AIV | 1462 | 66673.32 | 45.60 | 1.46 | 955.82 | 46.36 | 68.35 | 109.53 |
| MIX | 720 | 36646.98 | 50.90 | 1.70 | 264.60 | 30.25 | 92.30 | 261.30 |

#### 每种任务的执行平均时间（同类编号合并）

| 算子 | 类型 | 记录/实例数 | 总时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `merge_norm` | AIV | 576 | 33155.78 | 57.56 | 38.26 | 111.44 | 53.14 | 72.99 | 108.76 |
| `proj_a_mm` | AIC | 576 | 35222.72 | 61.15 | 43.48 | 147.66 | 51.79 | 87.37 | 139.26 |
| `proj_b_mm` | AIC | 576 | 14676.42 | 25.48 | 21.14 | 55.18 | 23.76 | 29.57 | 46.94 |
| `qproj_matmul` | AIC | 576 | 14029.16 | 24.36 | 20.54 | 38.24 | 23.86 | 27.17 | 37.18 |
| `qk_pv` | MIX | 360 | 29686.02 | 82.46 | 1.70 | 264.60 | 70.11 | 227.79 | 263.28 |
| `score` | MIX | 360 | 6960.96 | 19.34 | 2.08 | 46.66 | 19.53 | 39.17 | 43.45 |
| `qr_hadamard_matmul` | AIC | 288 | 1067.16 | 3.71 | 1.84 | 8.34 | 3.56 | 5.63 | 7.04 |
| `kv_proj_matmul` | AIC | 144 | 1901.34 | 13.20 | 11.28 | 19.34 | 12.83 | 14.57 | 18.58 |
| `qproj_dequant_rms_nope_rope` | AIV | 144 | 6748.54 | 46.86 | 45.34 | 48.60 | 46.89 | 47.83 | 48.53 |
| `qr_proj_matmul` | AIC | 144 | 3547.86 | 24.64 | 21.96 | 33.04 | 23.98 | 27.32 | 31.94 |
| `qr_rope` | AIV | 144 | 3255.56 | 22.61 | 20.30 | 25.60 | 22.56 | 24.37 | 25.24 |
| `hc_post` | AIV | 72 | 3324.06 | 46.17 | 45.40 | 47.50 | 46.10 | 46.64 | 47.41 |
| `idx_qr_proj_dequant` | AIV | 72 | 849.38 | 11.80 | 10.16 | 15.04 | 11.11 | 13.82 | 15.00 |
| `idx_qr_proj_matmul` | AIC | 72 | 3166.30 | 43.98 | 42.14 | 51.36 | 43.09 | 49.53 | 51.29 |
| `kv_score_proj` | AIC | 72 | 9776.04 | 135.78 | 127.04 | 194.34 | 128.32 | 187.23 | 194.31 |
| `proj_b_act` | AIV | 72 | 2190.56 | 30.42 | 28.44 | 36.46 | 29.59 | 34.86 | 36.26 |
| `qr_hadamard_quant` | AIV | 72 | 3682.76 | 51.15 | 49.78 | 52.72 | 51.17 | 51.84 | 52.56 |
| `quant` | AIV | 72 | 1041.94 | 14.47 | 12.78 | 30.24 | 13.85 | 15.35 | 27.57 |
| `topk` | AIV | 72 | 4138.66 | 57.48 | 44.94 | 64.00 | 62.62 | 63.70 | 63.99 |
| `hc_pre_linear` | AIC | 36 | 869.30 | 24.15 | 21.36 | 36.58 | 21.97 | 33.11 | 36.51 |
| `mix_x` | AIV | 36 | 1083.46 | 30.10 | 28.92 | 31.18 | 30.17 | 30.83 | 31.16 |
| `weights_proj` | AIC | 36 | 331.60 | 9.21 | 8.50 | 12.34 | 9.01 | 10.23 | 12.16 |
| `comb_sinkhorn` | AIV | 9 | 1187.64 | 131.96 | 130.96 | 133.10 | 132.12 | 132.84 | 133.07 |
| `csa_cache_writeback` | AIV | 9 | 211.50 | 23.50 | 21.88 | 26.54 | 23.00 | 25.60 | 26.45 |
| `csa_slots_build_valid_qk_plan` | AIV | 9 | 284.28 | 31.59 | 30.94 | 32.52 | 31.38 | 32.30 | 32.50 |
| `hc_pre_linear_reduce` | AIV | 9 | 55.14 | 6.13 | 5.86 | 6.36 | 6.12 | 6.28 | 6.35 |
| `hc_pre_rms` | AIV | 9 | 772.92 | 85.88 | 77.00 | 89.22 | 87.82 | 88.88 | 89.19 |
| `kv_rms_norm_rope` | AIV | 9 | 383.76 | 42.64 | 41.36 | 44.06 | 42.60 | 43.40 | 43.99 |
| `kv_touch` | AIV | 9 | 27.84 | 3.09 | 2.44 | 3.70 | 3.08 | 3.56 | 3.69 |
| `q_rope_prepare` | AIV | 9 | 143.04 | 15.89 | 15.40 | 16.56 | 15.78 | 16.35 | 16.54 |
| `qr_rms_norm_quant` | AIV | 9 | 354.82 | 39.42 | 38.98 | 39.88 | 39.36 | 39.85 | 39.88 |
| `rms_norm` | AIV | 9 | 869.30 | 96.59 | 95.14 | 99.36 | 95.88 | 99.30 | 99.35 |
| `rope_cs` | AIV | 9 | 95.04 | 10.56 | 10.28 | 11.10 | 10.44 | 10.89 | 11.08 |
| `split_pre_post` | AIV | 9 | 142.94 | 15.88 | 15.00 | 17.02 | 15.84 | 16.75 | 16.99 |
| `weights_proj_reduce` | AIV | 9 | 47.50 | 5.28 | 4.88 | 5.70 | 5.16 | 5.64 | 5.69 |
| `rope_interleave` | AIV | 2 | 39.10 | 19.55 | 19.38 | 19.72 | 19.55 | 19.69 | 19.72 |
| `scatter_softmax_pool` | AIV | 2 | 1860.48 | 930.24 | 904.66 | 955.82 | 930.24 | 950.70 | 955.31 |
| `csa_cmp_rope` | AIV | 1 | 57.34 | 57.34 | 57.34 | 57.34 | 57.34 | 57.34 | 57.34 |
| `csa_rope_step` | AIV | 1 | 253.10 | 253.10 | 253.10 | 253.10 | 253.10 | 253.10 | 253.10 |
| `kv_and_cache_write` | AIV | 1 | 87.66 | 87.66 | 87.66 | 87.66 | 87.66 | 87.66 | 87.66 |
| `kv_hadamard` | AIC | 1 | 14.42 | 14.42 | 14.42 | 14.42 | 14.42 | 14.42 | 14.42 |
| `kv_proj_seed` | AIV | 1 | 35.38 | 35.38 | 35.38 | 35.38 | 35.38 | 35.38 | 35.38 |
| `qr_proj_seed` | AIV | 1 | 67.04 | 67.04 | 67.04 | 67.04 | 67.04 | 67.04 | 67.04 |
| `qr_rope_swap_idx` | AIV | 1 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 |
| `rmsnorm_rope` | AIV | 1 | 37.02 | 37.02 | 37.02 | 37.02 | 37.02 | 37.02 | 37.02 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 186.22 | 186.22 | 186.22 | 186.22 | 186.22 | 186.22 | 186.22 |
| `rope_cs_swap` | AIV | 1 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 |

#### 任务粒度分布（μs，半开区间 [lo,hi)）

| 类型 | 0-5 | 5-10 | 10-15 | 15-20 | 20-30 | 30-50 | 50+ | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIC count | 241 | 79 | 135 | 14 | 1243 | 394 | 415 | 2521 |
| AIC 占比 | 9.6% | 3.1% | 5.4% | 0.6% | 49.3% | 15.6% | 16.5% | 100% |
| AIV count | 13 | 16 | 143 | 28 | 218 | 512 | 532 | 1462 |
| AIV 占比 | 0.9% | 1.1% | 9.8% | 1.9% | 14.9% | 35.0% | 36.4% | 100% |
| MIX count | 143 | 72 | 14 | 30 | 99 | 110 | 252 | 720 |
| MIX 占比 | 19.9% | 10.0% | 1.9% | 4.2% | 13.8% | 15.3% | 35.0% | 100% |


## 2.3 八档对照

| 目录 | B/S | 块数 | AIC均值 | AIV均值 | MIX均值 | 记录 AIC/AIV/MIX |
| ------ | ----- | ------ | --------: | --------: | --------: | ------------------: |
| [`deepseek_v4_flash_csa/basic_batch4_mtp1`](deepseek_v4_flash_csa/basic_batch4_mtp1) | 4/2 | flash_mtp | 10.06 | 6.91 | **12.39** | 273/142/40 |
|[`deepseek_v4_flash_csa/lt_batch4_mtp7`](deepseek_v4_flash_csa/lt_batch4_mtp7)|4/8|batch_tile=4|**15.15**|**15.26**|28.82|345/174/80|
|[`deepseek_v4_flash_csa/lt_batch8_mtp7`](deepseek_v4_flash_csa/lt_batch8_mtp7)|8/8|batch_tile=4|**14.34**|**17.06**|**23.62**|617/335/160|
|**[`deepseek_v4_flash_csa/lt_batch12_mtp7`](deepseek_v4_flash_csa/lt_batch12_mtp7)**|**12/8**|batch_tile=4|**14.41**|**17.91**|27.28|889/496/240|
|[`deepseek_v4_flash_csa/lt_batch16_mtp7`](deepseek_v4_flash_csa/lt_batch16_mtp7)|16/8|batch_tile=4|**14.41**|**17.62**|**22.92**|1161/657/320|
|**[`deepseek_v4_flash_csa/ht_batch60_mtp3`](deepseek_v4_flash_csa/ht_batch60_mtp3)**|**60/4**|batch_tile=20|**31.17**|**40.55**|**48.51**|889/496/240|
|[`deepseek_v4_flash_csa/ht_batch100_mtp3`](deepseek_v4_flash_csa/ht_batch100_mtp3)|100/4|batch_tile=20|**33.48**|**47.14**|**49.06**|1433/818/400|
|[`deepseek_v4_flash_csa/ht_batch180_mtp3`](deepseek_v4_flash_csa/ht_batch180_mtp3)|180/4|batch_tile=20|**33.56**|**45.60**|**50.90**|2521/1462/720|


<a id="3-running"></a>

# 3. 运行方式

```bash
./run_all.sh pypto -p a2a3 -d "$TASK_DEVICE"
./run_all.sh simpler -p a2a3 -d "$TASK_DEVICE"

task-submit --timeout 600 --max-time 900 --device auto --device-num 1 \
  --run './run_leaf_onboard.sh qwen3_decode_layer/basic_batch16 basic_batch16 $TASK_DEVICE'
```

<a id="changelog"></a>

# 修改历史

### 2026/9/17（Flash 推荐 HT + SPMD 按 tile 拆）

- Flash HT 推荐改为 `ht_batch60_mtp3`（取消 `ht_batch180_mtp3`）。
- Flash 样例：跨 batch_tile（LT=4 / HT=20）的工作不再打进同一个 `pl.spmd`，改为 `pl.range(N_BATCH_TILES)` × 单 tile `pl.spmd(n_base)`。

### 2026/9/17（Flash LT batch12）

- 新增 `deepseek_v4_flash_csa/lt_batch12_mtp7`（B=12 S=8 batch_tile=4 → `N_BATCH_TILES=3`，`idx_qr`/`kv_proj`/`qr_hadamard` SPMD=96，避开 120 核上 128=+8 余轮）；保留 `lt_batch16_mtp7`。


### 2026/9/16（HT ATTN=24）

- Qwen ht：`ATTN_SPMD_BLOCKS=24`；lt 仍为 120；basic 仍为 24。

### 2026/9/16（Qwen ht/lt/basic）

- Qwen 目录改为 `basic_batch16` / `lt_batch{16,32}` / `ht_batch{64,80,160}`。
