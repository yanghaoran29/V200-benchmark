# Serving-load / 任务粒度

样例在仓库根下按模型分目录：

- `qwen3_decode_layer/{basic_batch16,lt_batch16,lt_batch32,ht_batch64,ht_batch80,ht_batch160}/`
  - basic/ht：`ATTN_SPMD_BLOCKS=24`；lt：`120`
- `deepseek_v4_flash_csa/{basic_batch4_mtp1,ht_batch*_mtp3,lt_batch*_mtp7}/`

每个叶子自带该档 Python 源、Simpler C++ 和泳道。本目录的 `case_*.md` 是同一批上板的 AIC/AIV/MIX 三表。

**改参前基线（有效）**： [mainline_baseline/](mainline_baseline/) — pypto-lib `main` @ `3ac8fd8`。
同形状对照：`../qwen3_decode_layer/basic_batch16/`；Flash 主线同形+同 tiling：`../deepseek_v4_flash_csa/basic_batch4_mtp1/`（对齐 `pypto-lib/models/deepseek_v4_flash_mtp`）。

Flash 沿 batch 轴切分（ht tile=20 / lt·basic tile=4）。Qwen 按 `batch_pad=16` 分窗口。不含 pro。

| file | 含义 |
|------|------|
| mainline_baseline/baseline_mainline_*.md | 主线改参前三表（AIC/AIV/MIX） |
| case_qwen_*.md / case_flash_*.md | 服务负载 Case 明细 |
| summary_mix_v2.json | 14 叶子 + 主线均值摘要 |
| live_captures_v2/ | 上板原始编译目录（改构图后需清缓存再采） |

见 [../README.md](../README.md)。
