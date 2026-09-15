# 样例说明

本版面向 **120 AIC 目标机**，提供 DeepSeek V4-Flash CSA decode 和 Qwen3-14B 单层 decode 两个 benchmark。
每个 benchmark 同时包含调整后的 PyPTO 算子与其生成的 Simpler C++，使用相同默认输入和 golden。

历史内容见 [old/README.md](old/README.md)，归档版本为 `82f5a5c`（2026-07-14）。当前旧工作区快照已作废。
新 benchmark 不设置根级 support 目录，各自携带必要 Python 支持代码。工具链见 [VERSIONS.md](VERSIONS.md)。

| benchmark | PyPTO 入口 | Simpler C++ 入口 | 依赖图 | 泳道 |
|---|---|---|---|---|
| DeepSeek V4 CSA | [run_benchmark.py](deepseek-v4-csa/pypto-lib-operator/run_benchmark.py) | [test_decode_csa.py](deepseek-v4-csa/simpler-operator/test_decode_csa.py) | [HTML](deepseek-v4-csa/deps_viewer.html) | [原始记录](deepseek-v4-csa/Chip_swimlane_records.json) / [合并泳道](deepseek-v4-csa/merged_swimlane.json) |
| Qwen3 decode layer | [run_benchmark.py](qwen3-decode-layer/pypto-lib-operator/run_benchmark.py) | [test_qwen3_decode_layer.py](qwen3-decode-layer/simpler-operator/test_qwen3_decode_layer.py) | [HTML](qwen3-decode-layer/deps_viewer.html) | [原始记录](qwen3-decode-layer/Chip_swimlane_records.json) / [合并泳道](qwen3-decode-layer/merged_swimlane.json) |

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
固定采用以上分组，不再提供旧版统一 SPMD 宽度的五档切换。

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

# 2. DeepSeek V4 CSA Benchmark

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

# 3. 并发与验证结果

| 样例 | 逻辑任务 | AIC记录 | AIV记录 | 物理记录完整性 | early_dispatch=true | 采集跨度(us) | 120 AIC实测 |
|---|---:|---:|---:|---|---:|---:|---|
| Qwen | 166 | 545 | 312 | 857/857 | 0 | 906.76 | 待测 |
| CSA | 70 | 745 | 437 | 1182/1182 | 0 | 578.24 | 待测 |

采集跨度定义为最早 dispatch 到最晚 finish，包含四级采集影响，不能视为 benchmark 多轮 median。
AIC/AIV 区间平均占用各自以首个 kernel 开始到最后一个结束为分母。

- Qwen：24-AIC 实测最大逻辑 AIC 包络120 blocks（Attention）；MLP 最大同时活跃逻辑任务24个，合计68个声明 blocks。
- CSA：QR(64)+主compressor(64)+KV(32)的逻辑包络为160 blocks；Q(128)+Indexer(32)也有重叠。输出阶段峰值6个逻辑任务、96 blocks。
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

# 5. 相对于原 benchmark 的修改

| 项目 | 7月原版（old/） | 当前 benchmark |
|---|---|---|
| 内容 | Qwen整层、Scope2 attention与独立Paged Attention | DeepSeek V4 CSA和Qwen单层decode |
| 源码 | 手写/参数化的benchmark编排及kernel | 调整后的pypto-lib算子和对应生成Simpler代码配套提供 |
| Qwen默认batch | 旧整层90（padding96）；Scope2样例30 | 单层16，不沿用旧负载的耗时结论 |
| Qwen Attention | 原独立PA和Scope2分组方式 | Phase0独立16，Attention120；128工作项按120步长分配 |
| SPMD | 全局统一宽度五档 | 按算子固定分组，Out每10项，Gate/Up每5项 |
| 依赖 | manual/tensormap方案及局部静态构图设计 | 保留当前manual scope与真实deps；SiLU按Gate/Up分组依赖 |
| 调度技巧 | 各历史方案见原文档 | 关闭early dispatch，移除dummy、同步启动、全核屏障；CSA移除预取与非空优先排序 |
| 显式优化配置 | 历史实现 | 按要求保留CSA score的split/slot配置 |
| 长度 | 旧负载配置 | Qwen固定种子可变seq_len；CSA canonical起始位置集合；报告长度敏感性 |
| 产物 | 历史目录及泳道 | 每例均带deps、原始泳道、HTML、名称映射、合并泳道和并发分析 |
| 验证 | 历史测试标准 | 两种入口使用同一输入生成和数值golden，不跳过校验 |

CSA相对于此次改造前的pypto-lib基线：qproj 64→128、qr_proj 16→64、kv_proj 16→32；其余细节以随包配置及任务表为准。
移除高级调度不等于移除计算所需的原子累加、显式依赖或核内同步。

# 修改历史

### 2026/9/14

- 将7月版本归档到old，作废迁移前当前快照。
- 新增两个配套PyPTO/Simpler benchmark，使用已验证的120-AIC目标tiling。
- 保留manual scope和CSA split/slot，关闭early dispatch并整理配套采集。
- 重新计算逐算子统计，区分24-AIC实测与120-AIC目标分析。
