# DeepSeek V4 CSA — 方案 B

方案 A 保留在 `../deepseek-v4-csa/`。方案 B 面向 120 AIC，扩大每次调度承载的计算，不做 A/B 性能比较。

## 配置与任务划分

| 项目 | 方案 B |
|---|---|
| Batch / 每请求 token / 总 token | 20 / 2 / 40 |
| 输入种子 | 1234；用于生成张量数据，长度不由随机种子决定 |
| 默认 start_pos / KV seq_len | 全部20个请求：8192 / 8194 |
| Indexer score | 1 次调用，100 blocks；`SCORE_BLOCKS=100`、`REDUCE_NSPLIT=5` |
| QK/PV | 1 次调用，100 blocks；200 个工作项，每 block 顺序处理 2 项 |
| 其他算子 | 保留方案 A 调用数和 BlockDim，新增工作在任务内部循环 |
| 编排逻辑任务 / AIC记录 / AIV记录 | 70 / 825 / 597 |
| 总展开记录 | 1422；mixed scope 的 AIC 和两路 AIV 分别计数 |

score保留40×5=200个 `(token, split)` 工作项；第i个block顺序处理i和i+100两个工作项，对应token t与t+20的同一split。新增的任务内循环不会增加编排调用数。每个token的第 split 路负责 `split, split+5, split+10, ...` 缓存页。每页输出独立写入，页内 matmul、ReLU、加权 head 归约不变，不涉及跨路浮点求和。
QK/PV 第 i 个 block 执行工作项 i 和 i+100，保持 5 个 sparse blocks、各项独立 mi/li/oi、无效块中性值及后续 merge 顺序。因此两项压缩均可保留语义，验证后未启用回退。

主要固定宽度：qproj=128、qr_proj=64、kv_proj=32、idx_qr_proj matmul/dequant各32、qr_hadamard=32、两处kv_score_proj=64/8、topk=8、merge_norm=64；输出投影保持8组、每组16 blocks。
Topk 仍排序固定4096宽度，每 block 遍历5个 token；固定ratio=4的压缩、归一化、RoPE和缓存写回不扩大调度宽度。

Matmul 保持16行微块，通过内部循环覆盖40行（padding到48行）；尾部只有8行有效。Indexer weights projection和输出投影已去除单M tile假设，压缩器按16请求分块处理20个请求。
稀疏索引准备按8个token分组，避免40行整块运算超过UB，仍只有一个任务。有效数据量扩大5倍不代表padding后的物理计算量或执行时间严格扩大5倍。

保留 manual_scope、真实 deps、score 的 `split(NONE, slot_num=2)`、Split-K原子操作和核内计算同步。无early dispatch、dummy、syncall、预取或非空块优先调度。`kv_touch`保留缓存依赖作用。

## 长度与缓存

默认由 `torch.full((B,), DECODE_START_POS)` 生成，其中 `B=20`、`DECODE_START_POS=8192`，不再循环原来的长短序列边界集合。

```text
start_pos = [8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192,
             8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192]
```

每个请求本次处理位置 `[8192,8193]`，新增token数 `S=2`，KV seq_len为 `8192+2=8194`；总处理token数仍为40。8192是统一指定的起始位置，不是batch、任务数或随机长度。可用 `--start-pos N` 将全部请求覆盖为N，对应KV seq_len=N+2。

这次修改只统一长度，score与QK/PV仍各100个SPMD blocks。按当前公式，每个token可见 `min(8194//4, (position+1)//4, 4096)=2048` 个压缩位置，score每页处理32个位置，共64页。5路分片分别处理 `[13,13,13,13,12]` 页；每block处理两个token的同一分片，所以100个score blocks中80个处理26页、20个处理24页。

QK/PV保留每token的1个滑窗块和4个压缩块，每block处理两个工作项。默认长度下具备填满这些块所需的有效历史，但滑窗与压缩块的访存方式不同。因此统一seq_len不意味着所有blocks耗时完全一致。

当前起始位置8192满足 `8192%4=0`，本次两个token尚未到达下一个四token压缩边界；压缩器的边界更新分支不触发，已有压缩历史仍参与score与attention。冷启动和边界行为通过独立的 `--start-pos 0/127` 测试覆盖。

测试页表按实际长度分配互不重叠的请求页，未分配逻辑页为-1。默认主/内压缩状态各40980页，ori/cmp/indexer cache各1300页；这些是物理池容量，不是seq_len或任务数。相比旧混合长度输入，统一长历史需要更大的缓存池，源码入口会据长度重新推导容量。

修改默认输入后已重新生成配套Simpler产物与采集，不沿用旧混合长度的耗时。运行其他静态配置必须使用匹配的编译产物。

## 并发分析

[deps viewer](deps_viewer.html)、[deps.json](deps.json)、[原始泳道](Chip_swimlane_records.json)、[合并泳道](merged_swimlane.json)、[完整分析](concurrency_analysis.json) 来自同一次采集命令。运行时以一次deps采集和一次干净计时组成配对采集。

按deps所有wait边的传递闭包计算最大互无依赖集合（最大加权反链）：

- 最大逻辑任务并发度为 **12**，具体集合见分析JSON。
- 最大AIC声明工作量为 **324 blocks**：qproj128 + kv_proj32 + 主kv_score_proj64 + score100。
- 最大AIV声明工作量为 **221 records**。这与AIC最大集合是分别求得的界，不能相加解释为同一时刻占用。

以上是忽略时长和资源限制的静态可就绪集合，不是实测同时执行核数。120 AIC将同时执行的AIC数量限制在120；score和QK/PV各自单独只有100 blocks，剩余20核能否利用取决于其他任务是否就绪。

本次实际机器为 **24 AIC / 48 AIV，a2a3 device 3**，120 AIC实测尚未完成：

| 类型 | 完整记录 | 峰值执行并发 | 核执行区间平均占用 |
|---|---:|---:|---:|
| AIC | 825 | 24 | 58.19% |
| AIV | 597 | 48 | 31.32% |

首个dispatch到最后finish的采集跨度为1109.96 us，含四级采集影响，不作为多轮benchmark median。实际AIC逻辑任务包络峰值8个（合计128个声明blocks），最大声明需求包络为两处kv_score_proj（8+64）+ kv_proj32 + qr_proj64，共168 blocks；包络内的所有blocks并非同时执行。

### 逐算子执行时间

以下均为本次采集的物理核执行时间，total是核时间之和，不是端到端时长；同名输出投影分组合并统计。

| 算子 | 类型 | 调用数 | 记录数 | total(us) | mean | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hc_pre_rms` | AIV | 1 | 1 | 39.80 | 39.80 | 39.80 | 39.80 | 39.80 | 39.80 | 39.80 |
| `hc_pre_linear` | AIC | 1 | 4 | 87.10 | 21.78 | 21.12 | 22.22 | 21.88 | 22.21 | 22.22 |
| `hc_pre_linear_reduce` | AIV | 1 | 1 | 4.56 | 4.56 | 4.56 | 4.56 | 4.56 | 4.56 | 4.56 |
| `split_pre_post` | AIV | 1 | 1 | 8.56 | 8.56 | 8.56 | 8.56 | 8.56 | 8.56 | 8.56 |
| `comb_sinkhorn` | AIV | 1 | 1 | 65.98 | 65.98 | 65.98 | 65.98 | 65.98 | 65.98 | 65.98 |
| `mix_x` | AIV | 1 | 4 | 65.52 | 16.38 | 15.64 | 16.78 | 16.55 | 16.72 | 16.77 |
| `csa_rope_step` | AIV | 1 | 1 | 20.28 | 20.28 | 20.28 | 20.28 | 20.28 | 20.28 | 20.28 |
| `rope_interleave` | AIV | 1 | 1 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 |
| `csa_cmp_rope` | AIV | 1 | 1 | 8.70 | 8.70 | 8.70 | 8.70 | 8.70 | 8.70 | 8.70 |
| `rope_interleave_0` | AIV | 1 | 1 | 4.16 | 4.16 | 4.16 | 4.16 | 4.16 | 4.16 | 4.16 |
| `rms_norm` | AIV | 1 | 1 | 48.04 | 48.04 | 48.04 | 48.04 | 48.04 | 48.04 | 48.04 |
| `q_rope_prepare` | AIV | 1 | 1 | 8.22 | 8.22 | 8.22 | 8.22 | 8.22 | 8.22 | 8.22 |
| `qr_proj_seed` | AIV | 1 | 1 | 5.58 | 5.58 | 5.58 | 5.58 | 5.58 | 5.58 | 5.58 |
| `qr_proj_matmul` | AIC | 1 | 64 | 364.70 | 5.70 | 4.30 | 7.40 | 5.74 | 6.85 | 7.37 |
| `qr_rms_norm_quant` | AIV | 1 | 1 | 19.80 | 19.80 | 19.80 | 19.80 | 19.80 | 19.80 | 19.80 |
| `qproj_matmul` | AIC | 1 | 128 | 1768.88 | 13.82 | 8.52 | 23.58 | 12.74 | 19.39 | 23.17 |
| `qproj_dequant_rms_nope_rope` | AIV | 1 | 16 | 410.08 | 25.63 | 24.22 | 27.34 | 25.72 | 26.85 | 27.27 |
| `kv_proj_seed` | AIV | 1 | 1 | 3.46 | 3.46 | 3.46 | 3.46 | 3.46 | 3.46 | 3.46 |
| `kv_proj_matmul` | AIC | 1 | 32 | 222.60 | 6.96 | 4.50 | 11.46 | 6.43 | 11.20 | 11.44 |
| `kv_rms_norm_rope` | AIV | 1 | 1 | 20.62 | 20.62 | 20.62 | 20.62 | 20.62 | 20.62 | 20.62 |
| `csa_cache_writeback` | AIV | 1 | 1 | 14.54 | 14.54 | 14.54 | 14.54 | 14.54 | 14.54 | 14.54 |
| `kv_score_proj` | AIC | 1 | 64 | 740.66 | 11.57 | 10.40 | 14.96 | 11.23 | 12.59 | 14.27 |
| `scatter_softmax_pool` | AIV | 1 | 1 | 21.32 | 21.32 | 21.32 | 21.32 | 21.32 | 21.32 | 21.32 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 1 | 19.54 | 19.54 | 19.54 | 19.54 | 19.54 | 19.54 | 19.54 |
| `idx_qr_proj_matmul` | AIC | 1 | 32 | 450.18 | 14.07 | 11.20 | 16.54 | 14.26 | 16.03 | 16.49 |
| `idx_qr_proj_dequant` | AIV | 1 | 32 | 129.74 | 4.05 | 3.22 | 5.52 | 3.95 | 4.75 | 5.39 |
| `qr_rope_swap_idx` | AIV | 1 | 1 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 | 2.10 |
| `qr_rope` | AIV | 1 | 16 | 222.72 | 13.92 | 12.70 | 15.16 | 13.82 | 14.77 | 15.11 |
| `qr_hadamard_matmul` | AIC | 1 | 32 | 133.16 | 4.16 | 3.10 | 6.66 | 3.97 | 5.10 | 6.34 |
| `qr_hadamard_quant` | AIV | 1 | 8 | 205.00 | 25.63 | 24.58 | 26.26 | 25.69 | 26.13 | 26.25 |
| `weights_proj` | AIC | 1 | 4 | 28.42 | 7.10 | 6.86 | 7.30 | 7.13 | 7.25 | 7.30 |
| `weights_proj_reduce` | AIV | 1 | 1 | 3.54 | 3.54 | 3.54 | 3.54 | 3.54 | 3.54 | 3.54 |
| `kv_score_proj_0` | AIC | 1 | 8 | 151.96 | 19.00 | 17.82 | 20.42 | 19.20 | 19.87 | 20.37 |
| `scatter_softmax_pool_0` | AIV | 1 | 1 | 20.16 | 20.16 | 20.16 | 20.16 | 20.16 | 20.16 | 20.16 |
| `rmsnorm_rope` | AIV | 1 | 1 | 7.36 | 7.36 | 7.36 | 7.36 | 7.36 | 7.36 | 7.36 |
| `kv_hadamard` | AIC | 1 | 1 | 3.62 | 3.62 | 3.62 | 3.62 | 3.62 | 3.62 | 3.62 |
| `kv_and_cache_write` | AIV | 1 | 1 | 3.94 | 3.94 | 3.94 | 3.94 | 3.94 | 3.94 | 3.94 |
| `score_aic` | AIC | 1 | 100 | 2420.22 | 24.20 | 14.36 | 35.98 | 23.37 | 30.25 | 35.45 |
| `score_aiv` | AIV | 1 | 200 | 4808.74 | 24.04 | 14.34 | 36.60 | 23.27 | 30.05 | 34.27 |
| `topk` | AIV | 1 | 8 | 258.84 | 32.35 | 31.82 | 33.12 | 32.25 | 32.91 | 33.10 |
| `kv_touch` | AIV | 1 | 1 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 | 1.68 |
| `csa_slots_build_valid_qk_plan` | AIV | 1 | 1 | 15.30 | 15.30 | 15.30 | 15.30 | 15.30 | 15.30 | 15.30 |
| `qk_pv_aic` | AIC | 1 | 100 | 4266.14 | 42.66 | 14.08 | 59.80 | 46.21 | 56.36 | 58.20 |
| `qk_pv_aiv` | AIV | 1 | 200 | 8444.94 | 42.22 | 13.84 | 56.08 | 46.65 | 51.82 | 54.59 |
| `rope_cs` | AIV | 1 | 1 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 | 7.02 |
| `merge_norm` | AIV | 1 | 64 | 1315.14 | 20.55 | 16.68 | 25.00 | 20.89 | 23.62 | 24.81 |
| `proj_b_act` | AIV | 1 | 8 | 130.60 | 16.32 | 15.58 | 16.78 | 16.56 | 16.75 | 16.78 |
| `hc_post` | AIV | 1 | 8 | 184.68 | 23.09 | 22.90 | 23.30 | 23.07 | 23.26 | 23.30 |
| `proj_a_mm` | AIC | 8 | 128 | 2827.72 | 22.09 | 19.54 | 24.16 | 22.17 | 23.29 | 24.01 |
| `quant` | AIV | 8 | 8 | 70.10 | 8.76 | 8.26 | 9.42 | 8.68 | 9.31 | 9.41 |
| `proj_b_mm` | AIC | 8 | 128 | 1288.02 | 10.06 | 9.10 | 11.58 | 10.00 | 10.76 | 11.49 |

## 验证与复现

已通过原有Golden标准：默认全部start_pos=8192的PyPTO编译运行、默认Simpler C++ replay、全部start_pos=0、全部start_pos=127。
校验全部40个token的x_out和KV cache，未放宽阈值：x_out为`ratio_reldiff(0.004, 0.03, 2)`，KV为`ratio_allclose(atol=1e-4, rtol=1/128)`。
另外断言默认20个请求的位置均为[8192,8193]，检查score的200个逻辑工作项恰好分配到100个blocks、每个block两项，并穷举有效score长度0..4096，检查页覆盖一次且不遗漏；检查200个QK工作项恰好分配一次、实际页表/写入slot跨请求隔离，以及全部70项的BlockDim与方案A的差异仅为score和QK/PV。见[结构验证](structural_validation.json)。这不是对4097种长度逐一做NPU数值测试。

```bash
# 在已分配的设备上运行；在benchmark根目录执行。
python deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py -p a2a3 -d DEVICE
python deepseek-v4-csa-b/simpler-operator/test_decode_csa.py -p a2a3 -d DEVICE
python deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py -p a2a3 -d DEVICE \
  --enable-dep-gen --enable-chip-swimlane 4 --dep-output-dir outputs/csa-b
python deepseek-v4-csa-b/verify_scheme_b.py
python deepseek-v4-csa-b/analyze_capture.py
```

结构检查需要PyPTO环境；分析还使用NetworkX。源码、生成代码和归档采集哈希见[PROVENANCE.json](PROVENANCE.json)。PyPTO入口支持`a2a3/a2a3sim/a5/a5sim`；随包Simpler C++来自a2a3，不能直接作为a5产物使用。120 AIC目标机需使用对应平台和匹配工具链重新编译/验证，不将本次24核数据外推为120核性能。
