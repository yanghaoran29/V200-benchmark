# 样例说明

本版面向 **120 AIC 目标机**，提供 DeepSeek V4-Flash CSA decode 和 Qwen3-14B 单层 decode 两个模型的 benchmark，其中 CSA 提供方案 A 和方案 B。
每个 benchmark 同时包含调整后的 PyPTO 算子与其生成的 Simpler C++，使用相同默认输入和 golden。

历史内容见 [old/README.md](old/README.md)，归档版本为 `82f5a5c`（2026-07-14）。当前旧工作区快照已作废。
新 benchmark 不设置根级 support 目录，各自携带必要 Python 支持代码。工具链见 [VERSIONS.md](VERSIONS.md)。三个样例的逐文件用途和删除建议见 [FILE_GUIDE.md](FILE_GUIDE.md)，完整目录清单见 [FILE_INVENTORY.csv](FILE_INVENTORY.csv)。

| benchmark | PyPTO 入口 | Simpler C++ 入口 | 依赖图 | 泳道 |
|---|---|---|---|---|
| DeepSeek V4 CSA 方案 A | [run_benchmark.py](deepseek-v4-csa/pypto-lib-operator/run_benchmark.py) | [test_decode_csa.py](deepseek-v4-csa/simpler-operator/test_decode_csa.py) | [HTML](deepseek-v4-csa/deps_viewer.html) | [原始记录](deepseek-v4-csa/chip_swimlane_records.json) / [合并泳道](deepseek-v4-csa/merged_swimlane.json) |
| DeepSeek V4 CSA 方案 B | [run_benchmark.py](deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py) | [test_decode_csa.py](deepseek-v4-csa-b/simpler-operator/test_decode_csa.py) | [HTML](deepseek-v4-csa-b/deps_viewer.html) | [原始记录](deepseek-v4-csa-b/chip_swimlane_records.json) / [合并泳道](deepseek-v4-csa-b/merged_swimlane.json) |
| Qwen3 decode layer | [run_benchmark.py](qwen3-decode-layer/pypto-lib-operator/run_benchmark.py) | [test_qwen3_decode_layer.py](qwen3-decode-layer/simpler-operator/test_qwen3_decode_layer.py) | [HTML](qwen3-decode-layer/deps_viewer.html) | [原始记录](qwen3-decode-layer/chip_swimlane_records.json) / [合并泳道](qwen3-decode-layer/merged_swimlane.json) |

## 算子依赖关系图（SVG）

- [Qwen3 decode layer](qwen3-decode-layer/dependency_graph.svg)：617个物理任务（原始泳道857条记录）。
- [DeepSeek V4 CSA 方案A](deepseek-v4-csa/dependency_graph.svg)：942个物理任务（原始泳道1182条记录）。
- [DeepSeek V4 CSA 方案B](deepseek-v4-csa-b/dependency_graph.svg)：1022个物理任务（原始泳道1422条记录），全部请求start_pos=8192。

图中AIC为红色、AIV为蓝色、MIX为紫色；每个节点右下角的`x N`按物理任务计数：纯AIC/AIV沿用执行记录数，MIX的一个混合SPMD block计为一个物理任务，不重复累计其AIC/AIV记录。Qwen Attention为`x 120`；CSA A的score/QK-PV分别为`x 80`/`x 40`；CSA B两者均为`x 100`。原始泳道记录及其他章节的采集统计保持原有口径，JSON同时保存`physical_tasks`与`physical_records`供核对。
同类分组调用合并展示，主compressor与Indexer compressor保持独立。连线从`deps.json`的wait依赖生成，折叠无执行记录的张量创建节点，并省略可通过其他路径到达的传递边：存在`A→B→C`时不再画`A→C`，更长路径同样处理；生成时检查可达关系不变且没有冗余边。合并节点之间的箭头表示成员间存在依赖，不表示整个算子全部blocks完成后才允许下游启动；完整分组依赖仍以deps viewer为准。融合Attention内部阶段不重复拆分计数。

布局采用固定主轴与两侧分支：Qwen的Q/K/V同层排列、Gate/Up左右展开；CSA将Indexer主链居中，Q/KV与compressor分布两侧，QK/PV后的输出链保持居中。CSA A/B使用相同节点坐标，只改变标题和记录数，便于对照。模型分支本身不完全对称，布局不添加虚假节点或依赖来凑对称。

每张图旁的`dependency_graph_counts.json`列出节点物理计数、逻辑TaskId和依赖边，方便核对。安装NetworkX和Graphviz后，在仓库根目录运行`python generate_dependency_svgs.py`可重新生成三张图。

# 1. Qwen3-14B Decode Layer Benchmark

## 1.1 样例设计

| 参数 | 默认值 |
|---|---|
| decode 层数 / batch | 1 / 16 |
| hidden / intermediate | 5120 / 17408 |
| Q heads / KV heads / head dim | 40 / 8 / 128 |
| page size / 最大上下文 | 128 / 4096 |
| 输入与长度种子 | 1234 |
| seq_len | 每个请求独立生成，范围 [1,4096]；可用 `--max-seq` 全部设置为 4096 |

每个工作项对应一个请求和 KV head，共 `16×8=128` 个 Attention 工作项。
Attention 的 BlockDim 为 120：block 0–7 各执行两个工作项，其余各执行一个。
长度以张量输入传入；相同种子重复运行会生成相同长度，但每项 KV 扫描量不同，不能把 120 blocks 理解成等长任务。

## 1.2 SPMD 与依赖设计

| 阶段 | 逻辑任务 × BlockDim | 工作分配与依赖 |
|---|---|---|
| Q/K/V | 1×50 / 1×10 / 1×10 | 独立投影分支，合计 70 个 AIC blocks |
| Phase 0 | 1×16 | AIV 完成 norm/RoPE，Attention 等待真实 TaskId |
| Attention | 1×120 | mixed AIC/AIV；无同步启动、无全核屏障 |
| Out | 10×10 | 合计 100 blocks；每两组完成后可释放对应 residual cast |
| Gate / Up | 各17×5 | 互相独立；合计170 blocks，120 AIC 上超过一波容量 |
| SiLU | 17×1 | 每项等待对应 Gate、Up 和共享 RMS |
| Down | 85×1 | 每5项依赖对应 SiLU，可与其他 Gate/Up 分组重叠 |

保留 manual scope 和显式真实依赖；无 early dispatch、dummy 任务或 syncall。
固定采用以上分组。

## 1.3 a2a3 上各 kernel 执行时间

数据来自随包的 2026-09-14 09:39:42 四级泳道（24 AIC、48 AIV，device 1）。
下表是**物理核执行记录**，同名编号后缀合并统计；mixed 算子的 AIC/AIV 分列，不按时长排序拼成伪造的 block 配对。
各行总执行时间是核时间之和，不是端到端耗时。mixed 的逻辑任务数会同时出现在 AIC/AIV 行，不能直接求和。

| 算子 | 类型 | 逻辑任务数 | 物理记录数 | 总核执行时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dcr_xgamma` | AIV | 1 | 5 | 23.18 | 4.64 | 4.16 | 5.10 | 4.68 | 4.96 | 5.09 |
| `copy_hidden` | AIV | 1 | 1 | 9.78 | 9.78 | 9.78 | 9.78 | 9.78 | 9.78 | 9.78 |
| `x_gamma0` | AIV | 1 | 5 | 16.64 | 3.33 | 3.06 | 3.56 | 3.34 | 3.48 | 3.55 |
| `attn_out_seed` | AIV | 1 | 1 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 |
| `rms_recip` | AIV | 1 | 1 | 9.10 | 9.10 | 9.10 | 9.10 | 9.10 | 9.10 | 9.10 |
| `q_seed` | AIV | 1 | 1 | 4.60 | 4.60 | 4.60 | 4.60 | 4.60 | 4.60 | 4.60 |
| `q_proj` | AIC | 1 | 50 | 1007.16 | 20.14 | 17.98 | 22.94 | 20.19 | 21.40 | 22.92 |
| `kv_seed` | AIV | 1 | 1 | 4.14 | 4.14 | 4.14 | 4.14 | 4.14 | 4.14 | 4.14 |
| `mlp_out_seed` | AIV | 1 | 1 | 28.10 | 28.10 | 28.10 | 28.10 | 28.10 | 28.10 | 28.10 |
| `k_proj` | AIC | 1 | 10 | 168.36 | 16.84 | 14.54 | 18.18 | 16.85 | 18.00 | 18.16 |
| `v_proj` | AIC | 1 | 10 | 183.34 | 18.33 | 17.62 | 19.84 | 18.09 | 18.94 | 19.75 |
| `attn_phase0` | AIV | 1 | 32 | 386.86 | 12.09 | 10.56 | 14.12 | 12.02 | 13.42 | 13.98 |
| `attn_swpipe_aic` | AIC | 1 | 120 | 3867.84 | 32.23 | 4.46 | 66.54 | 32.95 | 48.80 | 66.33 |
| `attn_swpipe_aiv` | AIV | 1 | 240 | 7752.44 | 32.30 | 4.16 | 67.74 | 33.17 | 48.81 | 67.05 |
| `out_proj` | AIC | 10 | 100 | 1244.18 | 12.44 | 9.18 | 16.78 | 12.01 | 14.73 | 16.66 |
| `residual_rms_cast` | AIV | 5 | 5 | 19.26 | 3.85 | 3.06 | 5.94 | 3.44 | 4.95 | 5.84 |
| `post_rms_reduce` | AIV | 1 | 1 | 10.36 | 10.36 | 10.36 | 10.36 | 10.36 | 10.36 | 10.36 |
| `gate_proj` | AIC | 17 | 85 | 3521.16 | 41.43 | 35.74 | 49.08 | 41.22 | 45.45 | 48.64 |
| `up_proj` | AIC | 17 | 85 | 3570.88 | 42.01 | 35.06 | 48.78 | 41.50 | 46.37 | 48.49 |
| `silu` | AIV | 17 | 17 | 106.62 | 6.27 | 5.24 | 7.42 | 6.30 | 7.02 | 7.37 |
| `down_proj` | AIC | 85 | 85 | 3199.86 | 37.65 | 31.18 | 43.00 | 37.96 | 40.61 | 42.87 |
| `copy_out` | AIV | 1 | 1 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 |

| 核类型 | 物理记录 | 平均执行时间(us) | 峰值并行数 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 545 | 30.76 | 24 | 82.5% |
| AIV | 312 | 26.86 | 48 | 19.3% |

# 2. DeepSeek V4 CSA Benchmark（方案 A）

## 2.1 样例设计

| 参数 | 默认值 |
|---|---|
| batch / 每请求 token / 总 token | 4 / 2 / 8 |
| hidden / Q LoRA | 4096 / 1024 |
| attention heads / head dim | 64 / 512 |
| compressor ratio / sparse blocks | 4 / 5 |
| start_pos | `[8192, 0, 2, 3]`；KV长度为`[8194, 2, 4, 5]`，覆盖8k和压缩边界；非随机长度 |
| 长度覆盖 | `--start-pos N` 可统一覆盖起始位置；必须满足源码的缓存容量约束 |

位置、KV 长度影响压缩器更新、Indexer 有效归约元素和稀疏块有效性。
QK/PV 默认只有 `8×5=40` 个工作项，增大 BlockDim 本身不会产生更多有效工作。
短序列时 score 的 80 blocks 中有效工作更少；这些任务不保证均等时长。

## 2.2 Tiling 与并发设计

| 算子 | BlockDim | 120 AIC 上的含义 |
|---|---:|---|
| qproj_matmul | 128 | 120+8 尾波，与其他就绪分支竞争资源 |
| qr_proj_matmul | 64 | Split-K=8；原子扇入需目标机实测 |
| kv_proj_matmul | 32 | Split-K=8；保留计算所需原子累加 |
| idx_qr_proj_matmul/dequant | 各32 | MM_N_TILE 与 Q_OUT_TILE 均为256 |
| qr_hadamard_matmul | 32 | M tile=16，增加权重重复读取 |
| score | 80 | 有效并行量受长度影响；保留显式 split/slot 配置 |
| qk_pv | 40 | 自然 token/block 顺序，无非空块优先排序 |
| 主 kv_score_proj | 64 | N tile=16，存在窄输出传输效率风险 |
| proj_a_mm | 8组×16 | 静态可提供128 blocks；保留 manual scope 和真实依赖 |

移除 RMS 延迟 dummy、early dispatch 和输出权重预取。
`kv_touch` 自拷贝承担 KV-cache 正确性依赖，继续保留；核内流水、AIC/AIV 同步属于计算实现，不作为高级调度优化删除。

## 2.3 a2a3 上各 kernel 执行时间

数据来自随包的 2026-09-14 18:58:37 四级泳道（24 AIC、48 AIV，device 0），统计口径与 Qwen 相同。

| 算子 | 类型 | 逻辑任务数 | 物理记录数 | 总核执行时间(us) | 平均(us) | min | max | median | P90 | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hc_pre_rms` | AIV | 1 | 1 | 9.94 | 9.94 | 9.94 | 9.94 | 9.94 | 9.94 | 9.94 |
| `hc_pre_linear` | AIC | 1 | 4 | 38.68 | 9.67 | 9.24 | 10.46 | 9.49 | 10.23 | 10.44 |
| `hc_pre_linear_reduce` | AIV | 1 | 1 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 | 1.46 |
| `split_pre_post` | AIV | 1 | 1 | 3.88 | 3.88 | 3.88 | 3.88 | 3.88 | 3.88 | 3.88 |
| `comb_sinkhorn` | AIV | 1 | 1 | 14.86 | 14.86 | 14.86 | 14.86 | 14.86 | 14.86 | 14.86 |
| `mix_x` | AIV | 1 | 4 | 19.38 | 4.84 | 4.70 | 4.98 | 4.85 | 4.96 | 4.98 |
| `csa_rope_step` | AIV | 1 | 1 | 6.94 | 6.94 | 6.94 | 6.94 | 6.94 | 6.94 | 6.94 |
| `rope_interleave` | AIV | 2 | 2 | 4.92 | 2.46 | 2.42 | 2.50 | 2.46 | 2.49 | 2.50 |
| `csa_cmp_rope` | AIV | 1 | 1 | 3.08 | 3.08 | 3.08 | 3.08 | 3.08 | 3.08 | 3.08 |
| `rms_norm` | AIV | 1 | 1 | 11.20 | 11.20 | 11.20 | 11.20 | 11.20 | 11.20 | 11.20 |
| `q_rope_prepare` | AIV | 1 | 1 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 |
| `qr_proj_seed` | AIV | 1 | 1 | 2.90 | 2.90 | 2.90 | 2.90 | 2.90 | 2.90 | 2.90 |
| `qr_proj_matmul` | AIC | 1 | 64 | 245.30 | 3.83 | 2.32 | 7.44 | 3.92 | 4.98 | 7.19 |
| `qr_rms_norm_quant` | AIV | 1 | 1 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 | 4.84 |
| `qproj_matmul` | AIC | 1 | 128 | 1093.74 | 8.54 | 3.70 | 18.12 | 7.44 | 14.98 | 18.10 |
| `qproj_dequant_rms_nope_rope` | AIV | 1 | 16 | 99.08 | 6.19 | 5.10 | 7.82 | 5.89 | 7.15 | 7.73 |
| `kv_proj_seed` | AIV | 1 | 1 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 | 2.36 |
| `kv_proj_matmul` | AIC | 1 | 32 | 135.28 | 4.23 | 2.60 | 5.90 | 4.36 | 5.30 | 5.89 |
| `kv_rms_norm_rope` | AIV | 1 | 1 | 5.52 | 5.52 | 5.52 | 5.52 | 5.52 | 5.52 | 5.52 |
| `csa_cache_writeback` | AIV | 1 | 1 | 3.32 | 3.32 | 3.32 | 3.32 | 3.32 | 3.32 | 3.32 |
| `kv_score_proj` | AIC | 2 | 72 | 506.40 | 7.03 | 5.04 | 11.90 | 6.61 | 10.13 | 11.83 |
| `scatter_softmax_pool` | AIV | 2 | 2 | 22.02 | 11.01 | 10.58 | 11.44 | 11.01 | 11.35 | 11.43 |
| `rmsnorm_rope_cache_write` | AIV | 1 | 1 | 14.98 | 14.98 | 14.98 | 14.98 | 14.98 | 14.98 | 14.98 |
| `idx_qr_proj_matmul` | AIC | 1 | 32 | 242.98 | 7.59 | 4.32 | 9.04 | 7.55 | 8.79 | 9.03 |
| `idx_qr_proj_dequant` | AIV | 1 | 32 | 58.86 | 1.84 | 0.98 | 3.66 | 1.68 | 3.26 | 3.59 |
| `qr_rope_swap_idx` | AIV | 1 | 1 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 | 1.50 |
| `qr_rope` | AIV | 1 | 16 | 90.98 | 5.69 | 4.38 | 6.90 | 5.71 | 6.67 | 6.87 |
| `qr_hadamard_matmul` | AIC | 1 | 32 | 79.08 | 2.47 | 1.34 | 4.58 | 2.32 | 3.83 | 4.47 |
| `qr_hadamard_quant` | AIV | 1 | 8 | 48.18 | 6.02 | 5.62 | 6.56 | 6.01 | 6.36 | 6.54 |
| `weights_proj` | AIC | 1 | 4 | 21.88 | 5.47 | 5.32 | 5.62 | 5.47 | 5.59 | 5.62 |
| `weights_proj_reduce` | AIV | 1 | 1 | 1.74 | 1.74 | 1.74 | 1.74 | 1.74 | 1.74 | 1.74 |
| `rmsnorm_rope` | AIV | 1 | 1 | 4.70 | 4.70 | 4.70 | 4.70 | 4.70 | 4.70 | 4.70 |
| `kv_hadamard` | AIC | 1 | 1 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 | 2.64 |
| `kv_and_cache_write` | AIV | 1 | 1 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 | 3.02 |
| `score_aic` | AIC | 1 | 80 | 512.62 | 6.41 | 1.36 | 14.60 | 5.03 | 12.67 | 14.13 |
| `score_aiv` | AIV | 1 | 160 | 899.70 | 5.62 | 0.88 | 15.54 | 4.42 | 12.70 | 14.54 |
| `topk` | AIV | 1 | 8 | 41.36 | 5.17 | 1.28 | 7.76 | 7.01 | 7.62 | 7.75 |
| `kv_touch` | AIV | 1 | 1 | 1.24 | 1.24 | 1.24 | 1.24 | 1.24 | 1.24 | 1.24 |
| `csa_slots_build_valid_qk_plan` | AIV | 1 | 1 | 4.38 | 4.38 | 4.38 | 4.38 | 4.38 | 4.38 | 4.38 |
| `qk_pv_aic` | AIC | 1 | 40 | 352.18 | 8.80 | 0.58 | 22.94 | 4.43 | 22.50 | 22.91 |
| `qk_pv_aiv` | AIV | 1 | 80 | 696.12 | 8.70 | 0.82 | 21.52 | 5.24 | 20.89 | 21.35 |
| `rope_cs` | AIV | 1 | 1 | 3.26 | 3.26 | 3.26 | 3.26 | 3.26 | 3.26 | 3.26 |
| `merge_norm` | AIV | 1 | 64 | 630.02 | 9.84 | 4.12 | 15.98 | 10.83 | 13.27 | 15.87 |
| `proj_b_act` | AIV | 1 | 8 | 38.74 | 4.84 | 4.50 | 5.24 | 4.82 | 5.24 | 5.24 |
| `hc_post` | AIV | 1 | 8 | 43.32 | 5.42 | 5.06 | 5.88 | 5.32 | 5.84 | 5.88 |
| `proj_a_mm` | AIC | 8 | 128 | 1382.50 | 10.80 | 9.42 | 14.40 | 10.62 | 11.95 | 13.12 |
| `quant` | AIV | 8 | 8 | 23.58 | 2.95 | 2.34 | 3.58 | 2.87 | 3.43 | 3.56 |
| `proj_b_mm` | AIC | 8 | 128 | 751.66 | 5.87 | 5.06 | 7.22 | 5.82 | 6.48 | 6.90 |

| 核类型 | 物理记录 | 平均执行时间(us) | 峰值并行数 | 核区间平均占用 |
|---|---:|---:|---:|---:|
| AIC | 745 | 7.20 | 24 | 40.4% |
| AIV | 437 | 6.46 | 48 | 10.2% |

## 2.4 方案 B：batch扩大5倍，统一8192起始位置

方案 A 保留在 `deepseek-v4-csa/`。方案 B 面向 120 AIC，扩大每次调度承载的计算，不做 A/B 性能比较。

### 2.4.1 配置与任务划分

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

### 2.4.2 长度与缓存

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

### 2.4.3 并发分析

[deps viewer](deepseek-v4-csa-b/deps_viewer.html)、[deps.json](deepseek-v4-csa-b/deps.json)、[原始泳道](deepseek-v4-csa-b/chip_swimlane_records.json)、[合并泳道](deepseek-v4-csa-b/merged_swimlane.json)、[完整分析](deepseek-v4-csa-b/concurrency_analysis.json) 来自同一次采集命令。运行时以一次deps采集和一次干净计时组成配对采集。

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

#### 逐算子执行时间

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

### 2.4.4 验证与复现

已通过原有Golden标准：默认全部start_pos=8192的PyPTO编译运行、默认Simpler C++ replay、全部start_pos=0、全部start_pos=127。
校验全部40个token的x_out和KV cache，未放宽阈值：x_out为`ratio_reldiff(0.004, 0.03, 2)`，KV为`ratio_allclose(atol=1e-4, rtol=1/128)`。
另外断言默认20个请求的位置均为[8192,8193]，检查score的200个逻辑工作项恰好分配到100个blocks、每个block两项，并穷举有效score长度0..4096，检查页覆盖一次且不遗漏；检查200个QK工作项恰好分配一次、实际页表/写入slot跨请求隔离，以及全部70项的BlockDim与方案A的差异仅为score和QK/PV。见[结构验证](deepseek-v4-csa-b/structural_validation.json)。这不是对4097种长度逐一做NPU数值测试。

```bash
# 在已分配的设备上运行；在benchmark根目录执行。
python deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py -p a2a3 -d DEVICE
python deepseek-v4-csa-b/simpler-operator/test_decode_csa.py -p a2a3 -d DEVICE
python deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py -p a2a3 -d DEVICE \
  --enable-dep-gen --enable-chip-swimlane 4 --dep-output-dir outputs/csa-b
python deepseek-v4-csa-b/verify_scheme_b.py
python deepseek-v4-csa-b/analyze_capture.py
```

结构检查需要PyPTO环境；分析还使用NetworkX。源码、生成代码和归档采集哈希见[PROVENANCE.json](deepseek-v4-csa-b/PROVENANCE.json)。PyPTO入口支持`a2a3/a2a3sim/a5/a5sim`；随包Simpler C++来自a2a3，不能直接作为a5产物使用。120 AIC目标机需使用对应平台和匹配工具链重新编译/验证，不将本次24核数据外推为120核性能。

## 2.5 CSA A/B 任务数量与平均粒度对比

任务数量沿用SVG口径：纯AIC/AIV按物理执行记录计数，MIX每个混合SPMD block只计一个任务。同类分组调用合并统计，主compressor与Indexer compressor分别列出。

平均任务粒度定义为该算子的**核执行时间总和÷任务数量**，单位为核微秒/任务；MIX累加其AIC与AIV执行时间，再除以混合block数，表示平均每任务消耗的核执行工作量，**不是任务墙钟时长**。合计行按任务数量加权。数据来自随包24 AIC/48 AIV采集；A使用混合长度，B统一start_pos=8192，因此该表不表示同输入下的性能优劣，也不外推120 AIC性能。

| 算子 | A任务数量 | B任务数量 | A平均粒度（核μs/任务） | B平均粒度（核μs/任务） |
|---|---:|---:|---:|---:|
| `hc_pre_rms` | 1 | 1 | 9.94 | 39.80 |
| `hc_pre_linear` | 4 | 4 | 9.67 | 21.77 |
| `hc_pre_linear_reduce` | 1 | 1 | 1.46 | 4.56 |
| `split_pre_post` | 1 | 1 | 3.88 | 8.56 |
| `comb_sinkhorn` | 1 | 1 | 14.86 | 65.98 |
| `mix_x` | 4 | 4 | 4.85 | 16.38 |
| `csa_rope_step` | 1 | 1 | 6.94 | 20.28 |
| `Q: rope_interleave` | 1 | 1 | 2.42 | 3.50 |
| `csa_cmp_rope` | 1 | 1 | 3.08 | 8.70 |
| `compressed: rope_interleave` | 1 | 1 | 2.50 | 4.16 |
| `rms_norm` | 1 | 1 | 11.20 | 48.04 |
| `q_rope_prepare` | 1 | 1 | 2.64 | 8.22 |
| `qr_proj_seed` | 1 | 1 | 2.90 | 5.58 |
| `qr_proj_matmul` | 64 | 64 | 3.83 | 5.70 |
| `qr_rms_norm_quant` | 1 | 1 | 4.84 | 19.80 |
| `qproj_matmul` | 128 | 128 | 8.54 | 13.82 |
| `qproj_dequant_rms_nope_rope` | 16 | 16 | 6.19 | 25.63 |
| `kv_proj_seed` | 1 | 1 | 2.36 | 3.46 |
| `kv_proj_matmul` | 32 | 32 | 4.23 | 6.96 |
| `kv_rms_norm_rope` | 1 | 1 | 5.52 | 20.62 |
| `csa_cache_writeback` | 1 | 1 | 3.32 | 14.54 |
| `main: kv_score_proj` | 64 | 64 | 6.55 | 11.57 |
| `main: scatter_softmax_pool` | 1 | 1 | 10.58 | 21.32 |
| `rmsnorm_rope_cache_write` | 1 | 1 | 14.98 | 19.54 |
| `idx_qr_proj_matmul` | 32 | 32 | 7.59 | 14.07 |
| `idx_qr_proj_dequant` | 32 | 32 | 1.84 | 4.05 |
| `qr_rope_swap_idx` | 1 | 1 | 1.50 | 2.10 |
| `qr_rope` | 16 | 16 | 5.69 | 13.92 |
| `qr_hadamard_matmul` | 32 | 32 | 2.47 | 4.16 |
| `qr_hadamard_quant` | 8 | 8 | 6.02 | 25.62 |
| `weights_proj` | 4 | 4 | 5.47 | 7.10 |
| `weights_proj_reduce` | 1 | 1 | 1.74 | 3.54 |
| `indexer: kv_score_proj` | 8 | 8 | 10.94 | 19.00 |
| `indexer: scatter_softmax_pool` | 1 | 1 | 11.44 | 20.16 |
| `rmsnorm_rope` | 1 | 1 | 4.70 | 7.36 |
| `kv_hadamard` | 1 | 1 | 2.64 | 3.62 |
| `kv_and_cache_write` | 1 | 1 | 3.02 | 3.94 |
| `Indexer score` | 80 | 100 | 17.65 | 72.29 |
| `topk` | 8 | 8 | 5.17 | 32.35 |
| `kv_touch` | 1 | 1 | 1.24 | 1.68 |
| `csa_slots_build_valid_qk_plan` | 1 | 1 | 4.38 | 15.30 |
| `QK / PV` | 40 | 100 | 26.21 | 127.11 |
| `rope_cs` | 1 | 1 | 3.26 | 7.02 |
| `merge_norm` | 64 | 64 | 9.84 | 20.55 |
| `proj_a_mm` | 128 | 128 | 10.80 | 22.09 |
| `quant` | 8 | 8 | 2.95 | 8.76 |
| `proj_b_mm` | 128 | 128 | 5.87 | 10.06 |
| `proj_b_act` | 8 | 8 | 4.84 | 16.32 |
| `hc_post` | 8 | 8 | 5.42 | 23.09 |
| **合计／加权平均** | **942** | **1022** | **8.69** | **30.70** |

# 3. 并发与验证结果

| 样例 | 逻辑任务 | AIC记录 | AIV记录 | 物理记录完整性 | early_dispatch=true | 采集跨度(us) | 120 AIC实测 |
|---|---:|---:|---:|---|---:|---:|---|
| Qwen | 166 | 545 | 312 | 857/857 | 0 | 906.76 | 待测 |
| CSA A | 70 | 745 | 437 | 1182/1182 | 0 | 578.24 | 待测 |
| CSA B | 70 | 825 | 597 | 1422/1422 | 0 | 1109.96 | 待测 |

采集跨度定义为最早 dispatch 到最晚 finish，包含四级采集影响，不能视为 benchmark 多轮 median。
AIC/AIV 区间平均占用各自以首个 kernel 开始到最后一个结束为分母。

- Qwen：24-AIC 实测最大逻辑 AIC 包络120 blocks（Attention）；MLP 最大同时活跃逻辑任务24个，合计68个声明 blocks。
- CSA A：QR(64)+主compressor(64)+KV(32)的逻辑包络为160 blocks；Q(128)+Indexer(32)也有重叠。输出阶段峰值6个逻辑任务、96 blocks。
- 逻辑包络按每个任务首个物理 block 开始至末个 block 结束统计，其声明 blocks 不是同时执行的核数。
- 24-AIC 上两样例 AIC/AIV 物理峰值均为24/48；不能据此声称120-AIC上的实际并发或加速比。

打包后在 a2a3 device 0 验证：两个 Simpler C++ replay 入口和两个 PyPTO 编译入口均通过 golden。Qwen 重新编译后的 deps 与归档版本具有相同的166个任务、kernel ID、BlockDim和441条依赖边。

Golden 标准：Qwen out 使用 `ratio_allclose(atol=rtol=0.003, max_error_ratio=0.02)`；CSA x_out 使用 `ratio_reldiff(0.004, 0.03, 2)`，kv_cache 使用 `ratio_allclose(atol=1e-4, rtol=1/128)`。

# 4. 运行方式

安装 [VERSIONS.md](VERSIONS.md) 的 PyPTO、Simpler、PTOAS、PTO ISA 及匹配 CANN，激活对应环境。
无需另外 clone pypto-lib。Simpler 入口使用 PyPTO replay loader 绑定 ABI 和 bundled golden，执行的是随包 C++，不进行 PyPTO 图编译。

```bash
# 自行分配/锁定设备后运行；脚本不会占用未经分配的设备。
./run_all.sh simpler -p a2a3 -d 0
./run_all.sh pypto -p a2a3 -d 0

# 使用 task-submit 的环境可由队列分配设备。
task-submit --device auto --run './run_all.sh simpler -p a2a3 -d "$TASK_DEVICE"'

# 单个样例的配套依赖和泳道采集：运行时会分别采集图和干净计时。
python deepseek-v4-csa/pypto-lib-operator/run_benchmark.py -p a2a3 -d 0 \
  --enable-dep-gen --enable-chip-swimlane 4 --dep-output-dir outputs/csa
python qwen3-decode-layer/pypto-lib-operator/run_benchmark.py -p a2a3 -d 0 \
  --enable-dep-gen --enable-chip-swimlane 4 --dep-output-dir outputs/qwen
```

运行日志与临时编译产物不进入版本库。源码入口的 `--dep-output-dir` 指定导出编译产物目录；Simpler replay 的新采集写在自身 `dfx_outputs/`，不会覆盖 benchmark 根下的归档采集。
更换 batch、静态维度或 tiling 后必须重新生成 Simpler C++；不能只修改输入绕过 ABI 校验。

# 5. benchmark 模型相对于 pypto-lib 主线模型的修改

对照基线为本次改造起点的 pypto-lib 主线提交 `c3f0dea274f55d9648920f17c968e8564ae9fcdc`，对应 `models/qwen3_14b` 和 `models/deepseek_v4_flash_mtp`。这里比较模型实现、tiling 和输入配置；该固定基线不代表主线后续提交。随包代码及采集的精确版本见各样例的 `PROVENANCE.json`。

## 5.1 Qwen3-14B 单层 decode

模型维度与默认 batch=16 保持主线配置，调整任务划分和跨任务调度。

| 项目 | pypto-lib 主线基线 | 当前 benchmark |
|---|---|---|
| Q/K/V 投影 | 分别50/10/10个SPMD blocks | 保持50/10/10，保留Split-K原子累加 |
| Attention | BlockDim=24 | `ATTN_SPMD_BLOCKS=120`；128个请求/KV-head工作项按120步长分配，前8个block各处理2项，其余各1项 |
| Out projection | N分片10、Split-K=5，共50个工作项；26个普通任务加一次24-block SPMD | N分片20、输出N tile从512降到256，共100个工作项；按每10项组成一次SPMD，共10次调用 |
| Gate / Up | 各85个工作项；前6个N分片使用SPMD，其余通过dummy延迟调度 | 每个N分片的5个Split-K工作项组成一次SPMD，各17次调用×5 blocks；计算工作项总数不变 |
| SiLU / Down | 17个SiLU任务、85个Down工作项，带优先波次调度 | 数量保持；每个SiLU等待对应Gate/Up分组及共享RMS，Down等待对应SiLU |
| 调度 | early dispatch、dummy及同步启动/全核屏障等控制 | 移除这些控制；保留manual scope、表达真实数据依赖的deps及计算所需核内同步 |
| 输入长度 | 固定种子生成每请求seq_len | 保持seed=1234及长度范围[1,4096]；长度仍影响各Attention工作项的扫描量 |

Out投影每组覆盖两个相邻N tile及其全部5路Split-K；每两组完成即可释放对应的residual cast。增加Attention的BlockDim不增加128个数学工作项，也不保证任务等长。

## 5.2 DeepSeek V4 CSA 方案 A

保持主线的B=4、S=2、T=8和模型维度，主要通过缩小矩阵tile、增加Split-K或重划归约分片提高可调度任务数。

| 算子 | 主线BlockDim | 方案A BlockDim | 实现调整 |
|---|---:|---:|---|
| qproj_matmul | 64 | 128 | `QPROJ_MM_N_TILE`：512→256 |
| qr_proj_matmul | 16 | 64 | `QR_OK`：2→8，增加Split-K原子累加扇入 |
| kv_proj_matmul | 16 | 32 | `KV_OK`：4→8 |
| idx_qr_proj_matmul / dequant | 各8 | 各32 | `Q_OUT_TILE`：1024→256，`MM_N_TILE`：512→256，保证输出分片匹配 |
| qr_hadamard_matmul | 8 | 32 | `QH_MM_TILE`：64→16 |
| score | 16 | 80 | `REDUCE_NSPLIT`：2→10，8个token各10路归约 |
| QK/PV | 24 | 40 | `NUM_QK_CORES`：24→40，对应8×5个token/稀疏块工作项 |
| proj_a_mm | 每次8，共8次 | 每次16，共8次 | `PROJ_A_MM_N_TILE`：128→64 |
| proj_b_mm | 每次8，共8次 | 每次16，共8次 | `PROJ_B_D_TILE`：512→256 |
| merge_norm | 32 | 64 | merge阶段Head tile：16→8；QK/PV仍保留16-Head生产分组，重写merge的组内索引 |
| 主compressor kv_score_proj | 16 | 64 | 输出N tile：64→16；不将此配置应用到内层Indexer compressor |

调度方面，移除early dispatch、RMS延迟dummy、输出投影权重预取及QK/PV非空块优先排序，按自然token/block顺序分配工作。保留manual scope、真实deps、score显式split/slot配置，以及承担KV-cache依赖的`kv_touch`。Split-K原子累加和核内流水同步继续用于计算正确性。

默认起始位置为`[8192,0,2,3]`，KV长度为`[8194,2,4,5]`。这些长度不是均匀负载：score有效页数、QK/PV有效稀疏块及压缩边界分支都会影响任务时长。

## 5.3 DeepSeek V4 CSA 方案 B

方案B继承方案A的算子调整，在主线B=4、S=2的基础上将batch改为20，S仍为2，总token数由8增为40。非长度相关算子保持方案A的调用数和BlockDim，新增token在任务内部处理。

| 项目 | pypto-lib主线基线 | 方案B |
|---|---|---|
| batch / 本次token总数 | 4 / 8 | 20 / 40 |
| 默认start_pos / KV长度 | `[8192,0,2,3]` / `[8194,2,4,5]` | 全部20个请求为8192 / 8194 |
| score | 16 blocks，8个token×2路归约 | 100 blocks，40个token×5路归约共200工作项，每block处理2项 |
| QK/PV | 24 blocks处理40工作项 | 100 blocks处理200工作项，每block处理2项 |
| 缓存测试输入 | 按原batch和长度组织 | 按20个请求实际长度重新分配互不重叠的物理页，保留跨请求隔离 |

score每个token仍覆盖全部有效压缩位置；QK/PV仍覆盖每个token的5个稀疏块，合并工作项不改变归约或attention语义。统一8192起始位置后，score的80个blocks各处理26页、20个各处理24页，仍不能将100个blocks视为完全等长。完整参数、逐算子统计与验证见本文第2.4节。

上述修改均以120 AIC为目标；当前Golden和配套泳道在24 AIC / 48 AIV机器上验证，不据此给出120 AIC加速比或A/B性能结论。

# 修改历史

### 2026/9/15

- 方案B默认20个请求的start_pos统一为8192（KV seq_len=8194），重新验证并采集；根README完整列出方案B设计、长度含义、逐算子统计和复现方式。

### 2026/9/14

- 保留CSA方案A，新增独立方案B：batch20、score100、QK/PV100；配套生成代码、deps、泳道、静态并发及Golden验证。

- 将7月版本归档到old，作废迁移前当前快照。
- 新增两个配套PyPTO/Simpler benchmark，使用已验证的120-AIC目标tiling。
- 保留manual scope和CSA split/slot，关闭early dispatch并整理配套采集。
- 重新计算逐算子统计，区分24-AIC实测与120-AIC目标分析。
