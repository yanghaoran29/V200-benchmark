# AscendC CSA 与 pypto Flash CSA 任务对照

## 口径

| 侧 | 来源 | 说明 |
|----|------|------|
| AscendC | 外部给出的各 fused kernel 墙钟（µs） | **seq / B / S 未标明**；LI 对长度极敏感 |
| pypto | [`deepseek_v4_flash_csa/basic_batch4_mtp1`](../deepseek_v4_flash_csa/basic_batch4_mtp1)（B=4 S=2），[`case_flash_basic_batch4.md`](case_flash_basic_batch4.md) | 泳道 **mean / max**；整段 `captured_test_span_us` ≈ **445** |

**读数注意：**

- AscendC = 融合大 kernel **墙钟**；pypto = **SPMD 切分**后的单任务 mean/max。
- 多对一时用「阶段代理墙钟」（串行链上的 max/mean 之和），**不要**用 case 表「总时间」列直接减。
- 绝对值仅在同形、同 `seq_len`（同 \(L_{\text{idx}}\approx\text{seq}/4\)）下才可比。本调度计划固定 **seq_len=8192**。

**缩写：**

- **LI** = Lightning Indexer（打分 + topk，选出压缩序列上的稀疏位置）
- **C4A** = ratio-4 Compressed Sparse Attention（`sparse_attn_csa`）

pypto 主干（`decode_csa`）：`hc_pre` → `qkv` → **main compressor** → **indexer（含 inner compressor）** → **sparse_attn** → `hc_post`。AscendC 下列 12 步从 compressor 起，不含 hc_pre / 完整 qkv / hc_post。

---

## 1. 逐步任务对应（含时间与 seq_len_dep）

| # | AscendC | AscendC (µs) | pypto 对应（name_hint） | pypto mean (µs) | pypto max (µs) | n | seq_len_dep |
|---|---------|-------------:|-------------------------|----------------:|---------------:|--:|-------------|
| 1 | Matmul | **7.1** | `kv_score_proj`（main） | 15.36 | 19.26 | 16† | 无关 |
| 2 | InplacePartialRotaryMul | **8.7** | `rmsnorm_rope_cache_write`（内 RoPE） | 12.42 | 12.42 | 1 | 弱相关 |
| 3 | KvCompressEpilog | **3.06** | `scatter_softmax_pool`（main） | 10.41 | 10.76 | 1† | 无关 |
| 4 | Compressor | **61.2** | main `compressor_ratio4` 主体（rotate=False） | （见阶段合计） | — | — | 无关 |
| 5 | KvCompressEpilog | **2.68** | main 写 `cmp_kv` 尾 | （含在 2/3） | — | — | 弱相关 |
| 6 | Matmul | **4.0** | `idx_qr_proj_matmul`（可含 `qr_hadamard_matmul`） | 20.63 | 22.60 | 8 | 无关 |
| 7 | Muls | **3.0** | `idx_qr_proj_dequant` / `weights_proj` 等 | 2.30 / 5.38 | 2.84 / 5.98 | 8 / 4 | 无关 |
| 8 | Cast | **3.0** | `qr_hadamard_quant` 等 | 5.79 | 6.20 | 8 | 无关 |
| 9 | Compressor | **59.2** | inner `kv_score_proj_0` + `scatter_softmax_pool_0` | （见阶段合计） | — | 8+1† | 无关 |
| 10 | IndexerCompressEpilog | **2.5** | `rmsnorm_rope` + `kv_hadamard` + `kv_and_cache_write` | 27.72 + 3.64 + 8.72 | 同 | 1+1+1 | 弱相关（写 cache） |
| 11 | **LI** | **366.37** | **`score` + `topk`** | 10.53 + 5.17 | **26.58 + 7.98** | 16+8 | **相关** |
| 12 | **C4A** | **102.7** | **`qk_pv` + `merge_norm` + o_proj 链** | 见下表 | 见下表 | — | 无关（稀疏 TOPK 块数固定） |

† n 不再把 main 与 inner 加在一起。第 1 步只计 `kv_score_proj`（n=16），第 9 步计 `kv_score_proj_0`（n=8）；第 3 步只计 `scatter_softmax_pool`（n=1），第 9 步计 `scatter_softmax_pool_0`（n=1）。表中 mean/max 仍是合并后的泳道读数。

### C4A 细项（pypto）

| name_hint | mean (µs) | max (µs) | n | seq_len_dep |
|-----------|----------:|---------:|--:|-------------|
| `qk_pv` | 13.62 | 21.98 | 24 | 无关 |
| `merge_norm` | 11.86 | 14.52 | 32 | 无关 |
| `proj_a_mm` | 10.47 | 13.24 | 64 | 无关 |
| `quant` | 2.39 | 2.60 | 8 | 无关 |
| `proj_b_mm` | 6.23 | 6.90 | 64 | 无关 |
| `proj_b_act` | 4.64 | 5.46 | 8 | 无关 |

### 不在 AscendC 12 步内（pypto 另有）

`hc_pre_*`、`rms_norm`、完整 `qkv_proj_rope`（`qr_proj` / `qproj` / `kv_proj` …）、`csa_cache_writeback`、`hc_post` 等（多为 **无关**；`*_cache_write` / `kv_touch` 为 **弱相关**）。

---

## 2. 阶段合计与快慢

| 阶段 | AscendC (µs) | pypto 代理墙钟 (µs) | 代理算法 | 相对 | seq_len |
|------|-------------:|-------------------:|----------|------|---------|
| Main compress（1–5） | **82.7** | **~42** | `kv_score(max)+scatter+rmsnorm_rope_cache_write` | pypto 快 | 弱/无关 |
| Indexer Q prep（6–8） | **10.0** | **~36** | `idx_qr+dequant+qr_rope+hadamard_mm+hadamard_quant` 串加 | pypto 慢 | 无关 |
| Inner compress（9–10） | **61.7** | **~70** | `kv_score(max)+scatter+rmsnorm_rope+kv_hadamard+cache_write` | 接近 / 略慢 | 弱相关 |
| LI（11） | **366.4** | **~35** | `score(max)+topk(max)` | **pypto 快很多**（形态差 + 可能 seq 不同） | **相关** |
| C4A（12） | **102.7** | **~60** | `qk_pv(max)+merge(max)+o_proj 链 mean` | pypto 快 | 无关 |
| **上表合计** | **~624** | **~243** | — | — | — |
| 参考：pypto 整段 span | — | **~445** | `captured_test_span_us`（含 hc/qkv 等） | — | — |

---

## 3. LI 说明

- **作用：** 对压缩 KV 位置打分并 topk，供 C4A 做稀疏注意力；不是 attention 本体。
- **AscendC「LI」≠ 整个 pypto `indexer()`：** AscendC 已把 Q 准备（6–8）与 inner compressor（9–10）拆开；**LI 仅对应 `score` + `topk`**。
- **与 seq 强相关：** \(L_{\text{idx}}\approx\min(\text{cache}/4,\ (\text{pos}+1)/4,\ \text{MAX}/4)\)；比 LI 时间必须对齐同一 \(L_{\text{idx}}\)（本计划 **8192**）。

---

## 4. 模块归并（AscendC 细算子 → pypto 粒度融合）

调度对标用途：AscendC **更细、小算子多** → 调度次数上界更大。将细碎步 **融合** 成与 pypto 当前任务对齐的等效任务后，再比时间量级（校验仿真输入）。

| 融合后等效任务（pypto 粒度） | AscendC 细步 | AscendC 合计 (µs) | pypto 代理 (µs) | 量级 | seq_len_dep |
|------------------------------|--------------|------------------:|----------------:|------|-------------|
| main `compressor_ratio4` 阶段 | 1–5 | 82.7 | ~42 | 同量级（pypto 更快） | 弱/无关 |
| indexer Q 准备 | 6–8 | 10.0 | ~36 | pypto 更慢（切分开销） | 无关 |
| `indexer_compressor` 阶段 | 9–10 | 61.7 | ~70 | 接近 | 弱相关 |
| LI = `score`+`topk` | 11 | 366.4 | ~35 | **差大**：优先查 seq/形态 | **相关** |
| C4A = `qk_pv`+merge+o_proj | 12 | 102.7 | ~60 | 同量级 | 无关 |

**调度次数定性：** AscendC 若按 12+ 细算子 × SPMD 下发，次数 ≫ pypto 融合后逻辑任务数；融合对齐后应用 pypto 的 \(N\) / `sum(block_num)` 做 SNC 对照（见 [`sched_demand_a2a3_to_a5.md`](sched_demand_a2a3_to_a5.md) §7）。

---

## 5. 模块归并索引

| 阶段 | AscendC 步号 | pypto 模块 |
|------|-------------|------------|
| 主压缩 | 1–5 | `compressor_ratio4` |
| Indexer Q 准备 | 6–8 | `indexer` 前半 |
| Indexer 内压缩 | 9–10 | `indexer_compressor` |
| Lightning Indexer | 11 | `score` + `topk` |
| 压缩稀疏注意力 | 12 | `sparse_attn_csa` |
