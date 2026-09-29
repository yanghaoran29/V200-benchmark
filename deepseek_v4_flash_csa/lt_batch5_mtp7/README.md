# lt_batch5_mtp7 — Flash CSA LT 五刀（B=5，仅 C 类主线×batch）

| 项 | 值 |
|----|-----|
| case | `LT_BATCH5_MTP7` |
| B / S / mtp | 5 / 8 / 7 |
| `BATCH_TILE` | **1**（每刀 1 request） |
| `N_BATCH_TILES` | **5** |
| `T_TILE_BATCH` | 8 |
| 拆刀规则 | [`task_split_5_feasibility.md`](../../granularity/task_split_5_feasibility.md) |
| SPMD 回并 | 桶规则 `k=⌈N/120⌉`、`N'≥60`：只降 bn。`qproj`/`merge` 64→22；`proj_a/b` 8→3；`score`/`qk_pv` 40→20；`qr_hadamard` 32→16 |

## 构图

- **C 类**（≥5µs 且 W>1）：`pl.range(5)×pl.spmd(W)`，主线宽度不变。
- **A/B**：尽量单任务（`spmd(1)` + 内层串 tile）；勿裸 `spmd(5)` 替换主线。例外：`hc_pre_rms`、`comb_sinkhorn`、`csa_rope_step`、`rms_norm`、`qr_rms_norm_quant`、`kv_rms_norm_rope`、`scatter_softmax_pool`、`scatter_softmax_pool_0` 按 batch 切 `spmd(5)`。

## 上板

```bash
task-submit --timeout 600 --max-time 600 --device auto --device-num 1 --run '
cd .../lt_batch5_mtp7/pypto-lib-operator &&
python3 run_benchmark.py --platform a2a3 --device $TASK_DEVICE \
  --serving-case LT_BATCH5_MTP7 --enable-chip-swimlane 4 --enable-dep-gen --skip-golden \
  --dep-output-dir ../dfx_outputs
'
# pack
bash ../../pack_leaf_capture.sh .. ../dfx_outputs
# C 类粒度表
python3 ../../../../artifacts/tools/analyze_lt_batch5_c_grain.py
```
