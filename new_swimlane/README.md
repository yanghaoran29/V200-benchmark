# new_swimlane

本轮产物副本（原路径仍保留在各叶目录下）。

## onboard/ — 上板实测（本轮重采）

| 目录 | 说明 |
|------|------|
| `csa_lt_batch5_mtp7/` | CSA LT：补切 slots/split_pre/reduce 等后重采 |
| `csa_ht_batch20_mtp3/` | CSA HT：B=20、mtp3、`batch_tile=4`（`N_BATCH_TILES=5`） |
| `csa_ht_batch100_mtp3/` | CSA HT：scatter/rope/rmsnorm/kv_write 等按 `N_BATCH_TILES=5` 切后重采 |

Qwen：去掉 `manual_scope` / `scratch_ready` 跨窗串行，每窗独立 PA scratch，依赖改由 tensormap 构建。

## sim120/ — 离线 120 AIC / 240 AIV 模拟泳道

| 目录 | 叶 | assigned_span_us | 5µs D/C |
|------|-----|------------------|---------|
| `csa_lt_batch5_mtp7/` | `lt_batch5_mtp7` | 277.0 | 200 / 160 |
| `csa_ht_batch20_mtp3/` | `ht_batch20_mtp3` | 336.0 | 240 / 240 |
| `csa_ht_batch100_mtp3/` | `ht_batch100_mtp3` | 1340.0 | 240 / 200 |
| `qwen_lt_batch16/` | `lt_batch16` | 262.0 | 240 / 240 |
| `qwen_ht_batch80/` | `ht_batch80` | 615.0 | 132 / 73 |

每目录含 `merged_swimlane.json`、`chip_swimlane_records.json`、`summary.json`、`deps.json`、`name_map.json`、`README.md`。

### 串行 AIV 对照（详见各目录 README）

| 叶 | 现象 | 主因 |
|----|------|------|
| **qwen_lt_batch16** | span **262µs**。`gate_proj_1` 不再挡住 `gate_proj_6`（同 k 的 SPMD 与 deferred 列不交） | 已删 `gate_proj*` / `up_proj*` 之间的 tensormap 边，以及先前的 fold、out_proj、residual_rms_cast 假边 |
| qwen_ht_batch80 | 局部 cast/copy 链 | 窗内 `residual_rms_cast→_0…`、窗间 `copy_out→_0…`；**无** fold；DAG≈12 |
| csa_lt_batch5_mtp7 | deps 有同名 dequant 链，泳道反而是高峰 | 同名边导出时已拆；忙窗 peak≈120 |
| csa_ht_batch20_mtp3 | span 336µs。同名链已删；act 只等本刀；`scatter_softmax_pool*` 已按 5 刀切 | `batch_tile=4` 五刀。`proj_b_act` 不再等齐 40 个 `proj_b_mm`。`scatter_softmax_pool`/`_0`：`spmd(5)×每刀 4 batch` |
| csa_ht_batch100_mtp3 | 尾段 quant 略疏 | 同 LT 结构；主跨度在 AIC/MIX |

各叶细节见对应 `sim120_swimlane/README.md`；页面摘要见 `V200-benchmark/index.html#v200-sim120`。
