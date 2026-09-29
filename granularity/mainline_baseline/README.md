# 改参前基线 = pypto-lib 主线

**唯一有效的改参前对照**。来源：独立 worktree `pypto-lib-mainline-baseline` @ `3ac8fd8`（`main`），
默认 B/S / batch，**未**套用 `V200_SERVING_CASE`。

统计按 AIC / AIV / MIX 三分列。Flash 同形状服务负载包装见
[`../../deepseek_v4_flash_csa/basic_batch4_mtp1`](../../deepseek_v4_flash_csa/basic_batch4_mtp1)
（B=4 S=2，算子/tiling 对齐仓库 `deepseek_v4_flash_mtp`）；`lt_batch4_mtp7`（S=8）勿与主线混读。Qwen 同形状对照见
[`../../qwen3_decode_layer/basic_batch16`](../../qwen3_decode_layer/basic_batch16)
（batch=16 ATTN=24）；`lt_batch16` 同 batch、ATTN=120。

| 样例 | 默认形状 | 结果 | AIC 均值 | AIV 均值 | MIX 均值 | 记录 AIC/AIV/MIX | 明细 |
|------|----------|------|--------:|--------:|--------:|------------------:|------|
| flash CSA | B=4 S=2 | PASS | 9.93 | 6.75 | 16.40 | 273 / 142 / 40 | [baseline_mainline_flash_csa.md](baseline_mainline_flash_csa.md) |
| qwen3 | batch=16，`batch_pad=16`，ATTN=24 | PASS | 33.45 | 6.35 | 180.68 | 375 / 40 / 24 | [baseline_mainline_qwen3.md](baseline_mainline_qwen3.md) |

### 原始泳道

[live_captures/](live_captures/)：`flash_csa/`、`qwen3/`。
