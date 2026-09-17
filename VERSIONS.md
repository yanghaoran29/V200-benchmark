# Toolchain and source versions

| Component | Revision |
|---|---|
| pypto-lib | `092efde` on `yanghaoran29/pypto-lib-aicore-5x-tiling`, branch `aicore-5x-tiling` |
| PyPTO | `6fdd989fdf45b10011c4bd6b0291a7e6c18fd7e4` |
| Simpler | `39ce891dbb3f665e72e99b4a1387b47012fb18bd` (PyPTO runtime gitlink) |
| PTO ISA | `5a4f74cbf627d4aac2e0ce10d5e0d8b118343265` |
| PTOAS | `v0.61` |
| Archived benchmark | `82f5a5c` (2026-07-14) |

Use the selected PyPTO checkout's installation procedure and pinned dependencies,
with a matching CANN installation. Capture environment: a2a3, 24 AIC / 48 AIV,
CANN 9.0.0. The 120-AIC target has not been measured.

Serving-load sample tree (14 leaves):

```
V200-benchmark/
  qwen3_decode_layer/{basic_batch16,lt_batch16,lt_batch32,
                      ht_batch64,ht_batch80,ht_batch160}/
  deepseek_v4_flash_csa/{basic_batch4_mtp1,
                         ht_batch180_mtp3,ht_batch100_mtp3,ht_batch60_mtp3,
                         lt_batch16_mtp7,lt_batch12_mtp7,lt_batch8_mtp7,lt_batch4_mtp7}/
    pypto-lib-operator/
    simpler-operator/
    chip_swimlane_records.json
    merged_swimlane.json
    deps.json
    name_map.json
    concurrency_analysis.json
```

Qwen basic/ht use `ATTN_SPMD_BLOCKS=24`; lt uses 120. Flash HT tile=20,
LT tile=4 (batch axis). Cube tiles (`QR_OK`/`KV_OK`/`QPROJ_MM_N`/`Q_OUT`/
`PROJ_A`/`PROJ_B_D`/…) on LT/HT/basic match `pypto-lib/models/deepseek_v4_flash_mtp`.
`basic_batch4_mtp1` vendors that operator stack without `N_BATCH_TILES`.
`lt_batch12_mtp7` and `ht_batch60_mtp3` are the recommended Flash serving
shapes. `lt_batch12_mtp7` keeps per-tile `idx_qr`/`kv_proj` SPMD so a
120-core wave has no 128→+8 remainder. Cross-tile work is no longer packed
into one `pl.spmd`. `basic_batch16` is same-shape vs Qwen mainline. No A/B/C
scheme dirs and no pro sample.

Each leaf's `run_benchmark.py` defaults to that leaf's serving case. Results and
the vs-mainline table are in the root README. The 改参前 baseline is worktree
`pypto-lib-mainline-baseline` @ `3ac8fd8`, re-scored with the same AIC/AIV/MIX
split.

Each `pypto-lib-operator/golden` directory vendors the pypto-lib golden harness.
Both execution paths require PyPTO and its pinned Simpler runtime; the Simpler
entry uses runtime-directory replay, not PyPTO code generation.

Original source copyright headers are preserved. PyPTO source and generated
artifacts are distributed subject to their upstream CANN Open Software License
Agreement Version 2.0.
