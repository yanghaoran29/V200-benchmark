# Benchmark 文件用途与删除建议

范围：三个当前样例目录。仅实施泳道文件小写重命名及引用更新；以下删除项为建议，本次没有删除算子、采集或缓存。

## 结论

- 直接可清理：停跑后的本地`__pycache__/`、`.pyc`、编译缓存`cache/`、`.o`、`.so`、`.bin`。这些是Git忽略的构建产物；清理后首次运行需要重建。
- 优先考虑精简的已提交文件：`merged_swimlane.json`（三份合计约8.8 MiB），其次`dependency_graph_counts.json`；都不影响模型执行，但会失去对应展示/审计入口，必须更新README及哈希清单。
- 条件可删：分析JSON、分析/结构校验脚本及B的独立README；只在明确放弃相应复现/说明能力时删除。`deps_viewer.html`与SVG虽可重建，属于当前明确要求的交付物，建议保留。
- 保留：PyPTO源码闭包、Golden、Simpler注册/ABI元数据、所有被注册的C++与编排、LICENSE、deps、原始泳道、名称映射和版本清单。
- 不要因为A/B文件相似而删除一份：两个目录按独立运行组织，跨目录共享需要先重构入口和导入，并重新验证。

## 运行依赖

`pypto-lib-operator/run_benchmark.py → 模型及输入构造 → golden.run → PyPTO编译/执行/比较`。

`simpler-operator/test_*.py → 兄弟目录run_benchmark.py → 同一输入及Golden → runtime_dir replay → kernel_config.py + compiled_meta.json + orchestration/*.cpp + kernels/**/*.cpp`。

因此Simpler路径也不能直接删除Python源码或Golden。Qwen的`rms_lm_head.py`在默认单层中不执行，但被`decode_fwd.py`顶层导入；要精简必须先拆分该模块的多层/LM-head定义和导入，而不是直接删文件。本次没有实施这种代码裁剪。

展示数据不参与模型数值执行：`deps.json + name_map.json → deps_viewer.html`；再加`chip_swimlane_records.json → SVG/计数/并发分析`。原始采集是实测证据，重新跑只能得到新的采集，不能保证恢复旧耗时。

## 逐文件说明（受版本控制的交付文件）

以下逐一列出文件。文件名链接可直接打开；相同名称在A/B中仍分别列出。所有生成核均应与所在目录的kernel_config及编排配套。

### deepseek-v4-csa

| 文件 | 作用 | 删除建议 |
|---|---|---|
| [PROVENANCE.json](deepseek-v4-csa/PROVENANCE.json) | 源版本、采集身份、工具/验证信息及随包文件SHA-256清单。 | 保留；不是运行必需，但用于确认代码与采集配套。文件删改后必须同步清单。 |
| [chip_swimlane_records.json](deepseek-v4-csa/chip_swimlane_records.json) | 原始芯片采集：核类型、任务ID、开始/结束时间及调度记录；物理记录统计的依据。 | 保留；删除将失去该次实测证据，重新运行不保证相同耗时。 |
| [concurrency_analysis.json](deepseek-v4-csa/concurrency_analysis.json) | 该次采集的静态/实测并发分析；B还包含逐kernel时间统计。 | 可选删除；丢失机器可读分析且README引用需调整，建议保留。 |
| [dependency_graph.svg](deepseek-v4-csa/dependency_graph.svg) | 合并算子后的近似对称依赖图，省略传递边，MIX每block计一个任务。 | 可重建，但用户指定交付，建议保留。 |
| [dependency_graph_counts.json](deepseek-v4-csa/dependency_graph_counts.json) | SVG节点计数、成员TaskId、原始记录数、合并前后依赖边；便于审计图示。 | 可选删除；SVG生成时可重建，不影响运行或打开SVG。 |
| [deps.json](deepseek-v4-csa/deps.json) | 逻辑任务、kernel ID、BlockDim、张量和wait依赖；依赖图及静态并发分析的依据。 | 保留；是约定交付物和SVG生成输入。 |
| [deps_viewer.html](deepseek-v4-csa/deps_viewer.html) | 可直接打开的逻辑依赖图网页。 | 技术上可从deps和名称映射重建；按当前交付要求保留。 |
| [merged_swimlane.json](deepseek-v4-csa/merged_swimlane.json) | 用于trace查看器的合并泳道展示数据，将采集整理为时间线事件。 | 可选删除；不影响算子执行，但失去现成时间线；需同步README/PROVENANCE并使用匹配工具重建。 |
| [name_map.json](deepseek-v4-csa/name_map.json) | callable/kernel ID到可读算子名称的映射。 | 保留；SVG及分析脚本直接读取，不能只删此文件。 |
| [pypto-lib-operator/LICENSE](deepseek-v4-csa/pypto-lib-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [pypto-lib-operator/config.py](deepseek-v4-csa/pypto-lib-operator/config.py) | 模型维度、batch及tiling常量；A/B配置不同，不可跨方案随意替换。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_compressor_ratio4.py](deepseek-v4-csa/pypto-lib-operator/decode_compressor_ratio4.py) | 主ratio-4压缩器：投影、状态池化、归一化/RoPE及压缩cache更新。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_csa.py](deepseek-v4-csa/pypto-lib-operator/decode_csa.py) | CSA顶层计算图、输入/缓存测试数据构造与Golden参考；A/B均由此组装执行。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_indexer.py](deepseek-v4-csa/pypto-lib-operator/decode_indexer.py) | Indexer的查询投影、Hadamard/量化、score归约与Top-K选取。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_indexer_compressor.py](deepseek-v4-csa/pypto-lib-operator/decode_indexer_compressor.py) | Indexer内部压缩器及其状态/cache处理，区别于主attention压缩器。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_sparse_attn_csa.py](deepseek-v4-csa/pypto-lib-operator/decode_sparse_attn_csa.py) | 根据Top-K和滑窗构造稀疏访问计划，执行QK/PV、merge及输出投影。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/golden/__init__.py](deepseek-v4-csa/pypto-lib-operator/golden/__init__.py) | 导出run、规格和比较器等Golden接口。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/runner.py](deepseek-v4-csa/pypto-lib-operator/golden/runner.py) | 编译/replay、设备执行、输入保存/读取及Golden比较流程。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/spec.py](deepseek-v4-csa/pypto-lib-operator/golden/spec.py) | TensorSpec、输入初始化和测试数据描述。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/validation.py](deepseek-v4-csa/pypto-lib-operator/golden/validation.py) | 数值误差比较器和校验结果定义。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/hc_post.py](deepseek-v4-csa/pypto-lib-operator/hc_post.py) | Hyper-connection后处理，结合残差流与输出混合系数形成层输出。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/hc_pre.py](deepseek-v4-csa/pypto-lib-operator/hc_pre.py) | Hyper-connection前处理：RMS/线性、系数拆分、Sinkhorn和输入混合。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/qkv_proj_rope.py](deepseek-v4-csa/pypto-lib-operator/qkv_proj_rope.py) | Q/QR/KV投影、Split-K累加、量化/归一化与RoPE辅助算子。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/rmsnorm.py](deepseek-v4-csa/pypto-lib-operator/rmsnorm.py) | CSA使用的RMS归一化及相关辅助实现。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/rope_interleave.py](deepseek-v4-csa/pypto-lib-operator/rope_interleave.py) | RoPE交错排布的辅助算子。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/run_benchmark.py](deepseek-v4-csa/pypto-lib-operator/run_benchmark.py) | 命令行入口，构建输入、选择PyPTO编译或runtime_dir replay并执行Golden校验。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/utils.py](deepseek-v4-csa/pypto-lib-operator/utils.py) | 位置、seq_len、slot/page映射、缓存fixture及参考运算等共享辅助。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [simpler-operator/LICENSE](deepseek-v4-csa/simpler-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [simpler-operator/compiled_meta.json](deepseek-v4-csa/simpler-operator/compiled_meta.json) | PyPTO replay所需编译ABI元数据：参数顺序、shape、dtype和方向等。 | 保留；当前replay loader绑定ABI需要。 |
| [simpler-operator/kernel_config.py](deepseek-v4-csa/simpler-operator/kernel_config.py) | 运行时kernel ID、源文件、核类型、参数方向和orchestration入口注册表。 | 保留；所有注册源文件与此表配套。 |
| [simpler-operator/kernels/aic/hc_pre_linear.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/hc_pre_linear.cpp) | Hyper-connection系数线性投影。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/idx_qr_proj_matmul.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/idx_qr_proj_matmul.cpp) | Indexer查询投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_hadamard.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/kv_hadamard.cpp) | Indexer KV Hadamard变换。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_proj_matmul.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/kv_proj_matmul.cpp) | KV投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_score_proj.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/kv_score_proj.cpp) | 压缩器KV/score联合投影；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_score_proj_0.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/kv_score_proj_0.cpp) | 压缩器KV/score联合投影；独立注册的专用分组/调用版本，后缀不代表可删除副本；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/proj_a_mm.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/proj_a_mm.cpp) | 低秩输出投影第一段矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/proj_b_mm.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/proj_b_mm.cpp) | 低秩输出投影第二段矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qk_pv_aic.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/qk_pv_aic.cpp) | mixed稀疏attention的QK/PV矩阵计算。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qproj_matmul.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/qproj_matmul.cpp) | Q展开投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qr_hadamard_matmul.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/qr_hadamard_matmul.cpp) | 查询Hadamard矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qr_proj_matmul.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/qr_proj_matmul.cpp) | QR低秩投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/score_aic.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/score_aic.cpp) | mixed score的AIC矩阵计算部分。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/weights_proj.cpp](deepseek-v4-csa/simpler-operator/kernels/aic/weights_proj.cpp) | Indexer头权重投影。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/comb_sinkhorn.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/comb_sinkhorn.cpp) | 组合系数的Sinkhorn归一化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_cache_writeback.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/csa_cache_writeback.cpp) | KV缓存写回。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_cmp_rope.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/csa_cmp_rope.cpp) | 准备压缩位置RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_rope_step.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/csa_rope_step.cpp) | 准备当前decode步的RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_slots_build_valid_qk_plan.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/csa_slots_build_valid_qk_plan.cpp) | 构造滑窗/压缩稀疏slot及有效QK计划。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_post.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/hc_post.cpp) | Hyper-connection输出与残差流组合。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_pre_linear_reduce.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/hc_pre_linear_reduce.cpp) | 线性投影分片结果归约。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_pre_rms.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/hc_pre_rms.cpp) | 输入RMS统计。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/idx_qr_proj_dequant.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/idx_qr_proj_dequant.cpp) | Indexer投影反量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_and_cache_write.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/kv_and_cache_write.cpp) | Indexer KV数据处理与cache写入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_proj_seed.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/kv_proj_seed.cpp) | 初始化KV投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_rms_norm_rope.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/kv_rms_norm_rope.cpp) | KV归一化和RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_touch.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/kv_touch.cpp) | 通过cache自拷贝表达真实缓存依赖；不是可删除dummy。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/merge_norm.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/merge_norm.cpp) | 合并稀疏块的attention输出和归一化统计。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/mix_x.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/mix_x.cpp) | 按系数混合残差流作为attention输入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/proj_b_act.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/proj_b_act.cpp) | 第二段输出投影结果处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/q_rope_prepare.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/q_rope_prepare.cpp) | 准备Q侧RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qk_pv_aiv.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qk_pv_aiv.cpp) | mixed稀疏attention的数据准备、softmax及向量协作。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qproj_dequant_rms_nope_rope.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qproj_dequant_rms_nope_rope.cpp) | Q反量化、归一化及NoPE/RoPE处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_hadamard_quant.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qr_hadamard_quant.cpp) | Hadamard结果量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_proj_seed.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qr_proj_seed.cpp) | 初始化QR投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rms_norm_quant.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qr_rms_norm_quant.cpp) | QR结果RMS归一化及量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rope.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qr_rope.cpp) | Indexer查询RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rope_swap_idx.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/qr_rope_swap_idx.cpp) | 构造RoPE交换索引。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/quant.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/quant.cpp) | 输出投影中间结果量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rms_norm.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rms_norm.cpp) | RMS归一化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rmsnorm_rope.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rmsnorm_rope.cpp) | Indexer压缩结果归一化及RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rmsnorm_rope_cache_write.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rmsnorm_rope_cache_write.cpp) | 主压缩结果归一化、RoPE与cache写入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_cs.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rope_cs.cpp) | 输出阶段使用的RoPE cos/sin数据准备。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_interleave.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rope_interleave.cpp) | RoPE数据交错排布。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_interleave_0.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/rope_interleave_0.cpp) | RoPE数据交错排布；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/scatter_softmax_pool.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/scatter_softmax_pool.cpp) | 压缩状态scatter、softmax池化；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/scatter_softmax_pool_0.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/scatter_softmax_pool_0.cpp) | 压缩状态scatter、softmax池化；独立注册的专用分组/调用版本，后缀不代表可删除副本；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/score_aiv.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/score_aiv.cpp) | mixed score的AIV搬运、反量化及归约部分。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/split_pre_post.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/split_pre_post.cpp) | 拆分pre/post混合系数。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/topk.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/topk.cpp) | 从score选择Top-K位置。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/weights_proj_reduce.cpp](deepseek-v4-csa/simpler-operator/kernels/aiv/weights_proj_reduce.cpp) | 头权重分片归约。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/orchestration/attention_csa_test.cpp](deepseek-v4-csa/simpler-operator/orchestration/attention_csa_test.cpp) | 生成的AICPU编排：创建/提交任务、传递参数、连接依赖和中间张量。 | 保留；删除会使随包Simpler无法重建运行。 |
| [simpler-operator/test_decode_csa.py](deepseek-v4-csa/simpler-operator/test_decode_csa.py) | Simpler replay薄入口，导入兄弟Python目录并传入runtime_dir执行生成C++。 | 保留；当前Simpler命令入口。 |

### deepseek-v4-csa-b

| 文件 | 作用 | 删除建议 |
|---|---|---|
| [PROVENANCE.json](deepseek-v4-csa-b/PROVENANCE.json) | 源版本、采集身份、工具/验证信息及随包文件SHA-256清单。 | 保留；不是运行必需，但用于确认代码与采集配套。文件删改后必须同步清单。 |
| [README.md](deepseek-v4-csa-b/README.md) | 方案B的参数、任务划分、长度敏感性、统计、验证和复现说明。 | 可合并到根README后删除；目前有链接，建议保留独立入口。 |
| [analyze_capture.py](deepseek-v4-csa-b/analyze_capture.py) | 读取配套deps、泳道、名称映射，输出资源占用、逻辑包络、DAG并发及时间分布。 | 仅运行模型时可删；保留可复现分析能力时应保留。 |
| [chip_swimlane_records.json](deepseek-v4-csa-b/chip_swimlane_records.json) | 原始芯片采集：核类型、任务ID、开始/结束时间及调度记录；物理记录统计的依据。 | 保留；删除将失去该次实测证据，重新运行不保证相同耗时。 |
| [concurrency_analysis.json](deepseek-v4-csa-b/concurrency_analysis.json) | 该次采集的静态/实测并发分析；B还包含逐kernel时间统计。 | 可选删除；丢失机器可读分析且README引用需调整，建议保留。 |
| [dependency_graph.svg](deepseek-v4-csa-b/dependency_graph.svg) | 合并算子后的近似对称依赖图，省略传递边，MIX每block计一个任务。 | 可重建，但用户指定交付，建议保留。 |
| [dependency_graph_counts.json](deepseek-v4-csa-b/dependency_graph_counts.json) | SVG节点计数、成员TaskId、原始记录数、合并前后依赖边；便于审计图示。 | 可选删除；SVG生成时可重建，不影响运行或打开SVG。 |
| [deps.json](deepseek-v4-csa-b/deps.json) | 逻辑任务、kernel ID、BlockDim、张量和wait依赖；依赖图及静态并发分析的依据。 | 保留；是约定交付物和SVG生成输入。 |
| [deps_viewer.html](deepseek-v4-csa-b/deps_viewer.html) | 可直接打开的逻辑依赖图网页。 | 技术上可从deps和名称映射重建；按当前交付要求保留。 |
| [merged_swimlane.json](deepseek-v4-csa-b/merged_swimlane.json) | 用于trace查看器的合并泳道展示数据，将采集整理为时间线事件。 | 可选删除；不影响算子执行，但失去现成时间线；需同步README/PROVENANCE并使用匹配工具重建。 |
| [name_map.json](deepseek-v4-csa-b/name_map.json) | callable/kernel ID到可读算子名称的映射。 | 保留；SVG及分析脚本直接读取，不能只删此文件。 |
| [pypto-lib-operator/LICENSE](deepseek-v4-csa-b/pypto-lib-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [pypto-lib-operator/config.py](deepseek-v4-csa-b/pypto-lib-operator/config.py) | 模型维度、batch及tiling常量；A/B配置不同，不可跨方案随意替换。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_compressor_ratio4.py](deepseek-v4-csa-b/pypto-lib-operator/decode_compressor_ratio4.py) | 主ratio-4压缩器：投影、状态池化、归一化/RoPE及压缩cache更新。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_csa.py](deepseek-v4-csa-b/pypto-lib-operator/decode_csa.py) | CSA顶层计算图、输入/缓存测试数据构造与Golden参考；A/B均由此组装执行。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_indexer.py](deepseek-v4-csa-b/pypto-lib-operator/decode_indexer.py) | Indexer的查询投影、Hadamard/量化、score归约与Top-K选取。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_indexer_compressor.py](deepseek-v4-csa-b/pypto-lib-operator/decode_indexer_compressor.py) | Indexer内部压缩器及其状态/cache处理，区别于主attention压缩器。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_sparse_attn_csa.py](deepseek-v4-csa-b/pypto-lib-operator/decode_sparse_attn_csa.py) | 根据Top-K和滑窗构造稀疏访问计划，执行QK/PV、merge及输出投影。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/golden/__init__.py](deepseek-v4-csa-b/pypto-lib-operator/golden/__init__.py) | 导出run、规格和比较器等Golden接口。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/runner.py](deepseek-v4-csa-b/pypto-lib-operator/golden/runner.py) | 编译/replay、设备执行、输入保存/读取及Golden比较流程。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/spec.py](deepseek-v4-csa-b/pypto-lib-operator/golden/spec.py) | TensorSpec、输入初始化和测试数据描述。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/validation.py](deepseek-v4-csa-b/pypto-lib-operator/golden/validation.py) | 数值误差比较器和校验结果定义。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/hc_post.py](deepseek-v4-csa-b/pypto-lib-operator/hc_post.py) | Hyper-connection后处理，结合残差流与输出混合系数形成层输出。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/hc_pre.py](deepseek-v4-csa-b/pypto-lib-operator/hc_pre.py) | Hyper-connection前处理：RMS/线性、系数拆分、Sinkhorn和输入混合。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/qkv_proj_rope.py](deepseek-v4-csa-b/pypto-lib-operator/qkv_proj_rope.py) | Q/QR/KV投影、Split-K累加、量化/归一化与RoPE辅助算子。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/rmsnorm.py](deepseek-v4-csa-b/pypto-lib-operator/rmsnorm.py) | CSA使用的RMS归一化及相关辅助实现。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/rope_interleave.py](deepseek-v4-csa-b/pypto-lib-operator/rope_interleave.py) | RoPE交错排布的辅助算子。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/run_benchmark.py](deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py) | 命令行入口，构建输入、选择PyPTO编译或runtime_dir replay并执行Golden校验。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/utils.py](deepseek-v4-csa-b/pypto-lib-operator/utils.py) | 位置、seq_len、slot/page映射、缓存fixture及参考运算等共享辅助。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [simpler-operator/LICENSE](deepseek-v4-csa-b/simpler-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [simpler-operator/compiled_meta.json](deepseek-v4-csa-b/simpler-operator/compiled_meta.json) | PyPTO replay所需编译ABI元数据：参数顺序、shape、dtype和方向等。 | 保留；当前replay loader绑定ABI需要。 |
| [simpler-operator/kernel_config.py](deepseek-v4-csa-b/simpler-operator/kernel_config.py) | 运行时kernel ID、源文件、核类型、参数方向和orchestration入口注册表。 | 保留；所有注册源文件与此表配套。 |
| [simpler-operator/kernels/aic/hc_pre_linear.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/hc_pre_linear.cpp) | Hyper-connection系数线性投影。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/idx_qr_proj_matmul.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/idx_qr_proj_matmul.cpp) | Indexer查询投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_hadamard.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/kv_hadamard.cpp) | Indexer KV Hadamard变换。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_proj_matmul.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/kv_proj_matmul.cpp) | KV投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_score_proj.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/kv_score_proj.cpp) | 压缩器KV/score联合投影；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/kv_score_proj_0.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/kv_score_proj_0.cpp) | 压缩器KV/score联合投影；独立注册的专用分组/调用版本，后缀不代表可删除副本；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/proj_a_mm.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/proj_a_mm.cpp) | 低秩输出投影第一段矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/proj_b_mm.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/proj_b_mm.cpp) | 低秩输出投影第二段矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qk_pv_aic.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/qk_pv_aic.cpp) | mixed稀疏attention的QK/PV矩阵计算。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qproj_matmul.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/qproj_matmul.cpp) | Q展开投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qr_hadamard_matmul.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/qr_hadamard_matmul.cpp) | 查询Hadamard矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/qr_proj_matmul.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/qr_proj_matmul.cpp) | QR低秩投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/score_aic.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/score_aic.cpp) | mixed score的AIC矩阵计算部分。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/weights_proj.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aic/weights_proj.cpp) | Indexer头权重投影。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/comb_sinkhorn.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/comb_sinkhorn.cpp) | 组合系数的Sinkhorn归一化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_cache_writeback.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/csa_cache_writeback.cpp) | KV缓存写回。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_cmp_rope.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/csa_cmp_rope.cpp) | 准备压缩位置RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_rope_step.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/csa_rope_step.cpp) | 准备当前decode步的RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/csa_slots_build_valid_qk_plan.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/csa_slots_build_valid_qk_plan.cpp) | 构造滑窗/压缩稀疏slot及有效QK计划。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_post.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/hc_post.cpp) | Hyper-connection输出与残差流组合。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_pre_linear_reduce.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/hc_pre_linear_reduce.cpp) | 线性投影分片结果归约。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/hc_pre_rms.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/hc_pre_rms.cpp) | 输入RMS统计。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/idx_qr_proj_dequant.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/idx_qr_proj_dequant.cpp) | Indexer投影反量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_and_cache_write.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/kv_and_cache_write.cpp) | Indexer KV数据处理与cache写入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_proj_seed.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/kv_proj_seed.cpp) | 初始化KV投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_rms_norm_rope.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/kv_rms_norm_rope.cpp) | KV归一化和RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_touch.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/kv_touch.cpp) | 通过cache自拷贝表达真实缓存依赖；不是可删除dummy。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/merge_norm.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/merge_norm.cpp) | 合并稀疏块的attention输出和归一化统计。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/mix_x.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/mix_x.cpp) | 按系数混合残差流作为attention输入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/proj_b_act.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/proj_b_act.cpp) | 第二段输出投影结果处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/q_rope_prepare.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/q_rope_prepare.cpp) | 准备Q侧RoPE数据。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qk_pv_aiv.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qk_pv_aiv.cpp) | mixed稀疏attention的数据准备、softmax及向量协作。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qproj_dequant_rms_nope_rope.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qproj_dequant_rms_nope_rope.cpp) | Q反量化、归一化及NoPE/RoPE处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_hadamard_quant.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qr_hadamard_quant.cpp) | Hadamard结果量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_proj_seed.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qr_proj_seed.cpp) | 初始化QR投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rms_norm_quant.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qr_rms_norm_quant.cpp) | QR结果RMS归一化及量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rope.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qr_rope.cpp) | Indexer查询RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/qr_rope_swap_idx.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/qr_rope_swap_idx.cpp) | 构造RoPE交换索引。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/quant.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/quant.cpp) | 输出投影中间结果量化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rms_norm.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rms_norm.cpp) | RMS归一化。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rmsnorm_rope.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rmsnorm_rope.cpp) | Indexer压缩结果归一化及RoPE。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rmsnorm_rope_cache_write.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rmsnorm_rope_cache_write.cpp) | 主压缩结果归一化、RoPE与cache写入。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_cs.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rope_cs.cpp) | 输出阶段使用的RoPE cos/sin数据准备。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_interleave.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rope_interleave.cpp) | RoPE数据交错排布。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rope_interleave_0.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/rope_interleave_0.cpp) | RoPE数据交错排布；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/scatter_softmax_pool.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/scatter_softmax_pool.cpp) | 压缩状态scatter、softmax池化；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/scatter_softmax_pool_0.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/scatter_softmax_pool_0.cpp) | 压缩状态scatter、softmax池化；独立注册的专用分组/调用版本，后缀不代表可删除副本；_0为Indexer内压缩分支，无后缀为主压缩分支。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/score_aiv.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/score_aiv.cpp) | mixed score的AIV搬运、反量化及归约部分。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/split_pre_post.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/split_pre_post.cpp) | 拆分pre/post混合系数。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/topk.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/topk.cpp) | 从score选择Top-K位置。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/weights_proj_reduce.cpp](deepseek-v4-csa-b/simpler-operator/kernels/aiv/weights_proj_reduce.cpp) | 头权重分片归约。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/orchestration/attention_csa_test.cpp](deepseek-v4-csa-b/simpler-operator/orchestration/attention_csa_test.cpp) | 生成的AICPU编排：创建/提交任务、传递参数、连接依赖和中间张量。 | 保留；删除会使随包Simpler无法重建运行。 |
| [simpler-operator/test_decode_csa.py](deepseek-v4-csa-b/simpler-operator/test_decode_csa.py) | Simpler replay薄入口，导入兄弟Python目录并传入runtime_dir执行生成C++。 | 保留；当前Simpler命令入口。 |
| [structural_validation.json](deepseek-v4-csa-b/structural_validation.json) | 上述结构校验最近一次结果，不是Golden数值验证的替代品。 | 可选删除；可运行verify_scheme_b.py重建，但会失去该次结果记录。 |
| [verify_scheme_b.py](deepseek-v4-csa-b/verify_scheme_b.py) | 检查B默认位置、工作项覆盖、页表隔离及相对A的BlockDim变化。 | 仅运行模型时可删；建议保留结构回归能力。 |

### qwen3-decode-layer

| 文件 | 作用 | 删除建议 |
|---|---|---|
| [PROVENANCE.json](qwen3-decode-layer/PROVENANCE.json) | 源版本、采集身份、工具/验证信息及随包文件SHA-256清单。 | 保留；不是运行必需，但用于确认代码与采集配套。文件删改后必须同步清单。 |
| [chip_swimlane_records.json](qwen3-decode-layer/chip_swimlane_records.json) | 原始芯片采集：核类型、任务ID、开始/结束时间及调度记录；物理记录统计的依据。 | 保留；删除将失去该次实测证据，重新运行不保证相同耗时。 |
| [concurrency_analysis.json](qwen3-decode-layer/concurrency_analysis.json) | 该次采集的静态/实测并发分析；B还包含逐kernel时间统计。 | 可选删除；丢失机器可读分析且README引用需调整，建议保留。 |
| [dependency_graph.svg](qwen3-decode-layer/dependency_graph.svg) | 合并算子后的近似对称依赖图，省略传递边，MIX每block计一个任务。 | 可重建，但用户指定交付，建议保留。 |
| [dependency_graph_counts.json](qwen3-decode-layer/dependency_graph_counts.json) | SVG节点计数、成员TaskId、原始记录数、合并前后依赖边；便于审计图示。 | 可选删除；SVG生成时可重建，不影响运行或打开SVG。 |
| [deps.json](qwen3-decode-layer/deps.json) | 逻辑任务、kernel ID、BlockDim、张量和wait依赖；依赖图及静态并发分析的依据。 | 保留；是约定交付物和SVG生成输入。 |
| [deps_viewer.html](qwen3-decode-layer/deps_viewer.html) | 可直接打开的逻辑依赖图网页。 | 技术上可从deps和名称映射重建；按当前交付要求保留。 |
| [merged_swimlane.json](qwen3-decode-layer/merged_swimlane.json) | 用于trace查看器的合并泳道展示数据，将采集整理为时间线事件。 | 可选删除；不影响算子执行，但失去现成时间线；需同步README/PROVENANCE并使用匹配工具重建。 |
| [name_map.json](qwen3-decode-layer/name_map.json) | callable/kernel ID到可读算子名称的映射。 | 保留；SVG及分析脚本直接读取，不能只删此文件。 |
| [pypto-lib-operator/LICENSE](qwen3-decode-layer/pypto-lib-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [pypto-lib-operator/config.py](qwen3-decode-layer/pypto-lib-operator/config.py) | 模型维度、batch及tiling常量；A/B配置不同，不可跨方案随意替换。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/constants.py](qwen3-decode-layer/pypto-lib-operator/constants.py) | Qwen模型规格与默认tiling数据结构，供config和attention模块导入。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/decode_fwd.py](qwen3-decode-layer/pypto-lib-operator/decode_fwd.py) | Qwen decode计算图、输入构造和Golden；同时保留上游多层/LM-head相关定义。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/golden/__init__.py](qwen3-decode-layer/pypto-lib-operator/golden/__init__.py) | 导出run、规格和比较器等Golden接口。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/runner.py](qwen3-decode-layer/pypto-lib-operator/golden/runner.py) | 编译/replay、设备执行、输入保存/读取及Golden比较流程。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/spec.py](qwen3-decode-layer/pypto-lib-operator/golden/spec.py) | TensorSpec、输入初始化和测试数据描述。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/golden/validation.py](qwen3-decode-layer/pypto-lib-operator/golden/validation.py) | 数值误差比较器和校验结果定义。 | 保留；两个运行入口共同依赖。 |
| [pypto-lib-operator/paged_attention_pypto.py](qwen3-decode-layer/pypto-lib-operator/paged_attention_pypto.py) | 原生PyPTO分页attention：Phase0预处理和mixed QK/softmax/PV计算。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/rms_lm_head.py](qwen3-decode-layer/pypto-lib-operator/rms_lm_head.py) | 最终RMSNorm/LM-head实现；默认单层不执行，但decode_fwd.py顶层直接导入，不能直接删除。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [pypto-lib-operator/run_benchmark.py](qwen3-decode-layer/pypto-lib-operator/run_benchmark.py) | 命令行入口，构建输入、选择PyPTO编译或runtime_dir replay并执行Golden校验。 | 保留；当前Python导入/运行闭包组成部分，不能按默认图未执行直接删。 |
| [simpler-operator/LICENSE](qwen3-decode-layer/simpler-operator/LICENSE) | 该份源码/生成代码的上游许可证文本。 | 保留；随代码分发，不作为冗余文件清理。 |
| [simpler-operator/compiled_meta.json](qwen3-decode-layer/simpler-operator/compiled_meta.json) | PyPTO replay所需编译ABI元数据：参数顺序、shape、dtype和方向等。 | 保留；当前replay loader绑定ABI需要。 |
| [simpler-operator/kernel_config.py](qwen3-decode-layer/simpler-operator/kernel_config.py) | 运行时kernel ID、源文件、核类型、参数方向和orchestration入口注册表。 | 保留；所有注册源文件与此表配套。 |
| [simpler-operator/kernels/aic/attn_swpipe_aic.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/attn_swpipe_aic.cpp) | mixed分页attention的QK/PV矩阵计算。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/down_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/down_proj.cpp) | MLP Down投影。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj.cpp) | MLP Gate投影分组。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_0.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_0.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_1.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_1.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_10.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_10.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_11.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_11.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_12.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_12.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_13.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_13.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_14.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_14.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_15.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_15.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_2.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_2.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_3.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_3.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_4.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_4.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_5.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_5.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_6.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_6.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_7.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_7.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_8.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_8.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/gate_proj_9.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/gate_proj_9.cpp) | MLP Gate投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/k_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/k_proj.cpp) | K投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj.cpp) | attention输出投影分组。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_0.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_0.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_1.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_1.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_2.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_2.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_3.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_3.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_4.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_4.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_5.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_5.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_6.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_6.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_7.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_7.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/out_proj_8.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/out_proj_8.cpp) | attention输出投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/q_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/q_proj.cpp) | Q投影Split-K矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj.cpp) | MLP Up投影分组。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_0.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_0.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_1.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_1.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_10.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_10.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_11.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_11.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_12.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_12.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_13.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_13.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_14.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_14.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_15.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_15.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_2.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_2.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_3.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_3.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_4.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_4.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_5.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_5.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_6.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_6.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_7.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_7.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_8.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_8.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/up_proj_9.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/up_proj_9.cpp) | MLP Up投影分组；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aic/v_proj.cpp](qwen3-decode-layer/simpler-operator/kernels/aic/v_proj.cpp) | V投影矩阵乘。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/attn_out_seed.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/attn_out_seed.cpp) | 初始化attention输出缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/attn_phase0.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/attn_phase0.cpp) | QK归一化、RoPE及KV cache等attention前处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/attn_swpipe_aiv.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/attn_swpipe_aiv.cpp) | mixed分页attention的数据搬运、softmax和向量协作。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/copy_hidden.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/copy_hidden.cpp) | 输入hidden复制。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/copy_out.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/copy_out.cpp) | 最终输出拷贝。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/dcr_xgamma.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/dcr_xgamma.cpp) | Down投影后残差与归一化相关处理。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/kv_seed.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/kv_seed.cpp) | 初始化K/V投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/mlp_out_seed.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/mlp_out_seed.cpp) | 初始化MLP相关输出/累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/post_rms_reduce.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/post_rms_reduce.cpp) | 后RMS统计归约。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/q_seed.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/q_seed.cpp) | 初始化Q投影累加缓冲。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/residual_rms_cast.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/residual_rms_cast.cpp) | 残差相加、类型转换及后RMS局部统计。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/residual_rms_cast_0.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/residual_rms_cast_0.cpp) | 残差相加、类型转换及后RMS局部统计；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/residual_rms_cast_1.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/residual_rms_cast_1.cpp) | 残差相加、类型转换及后RMS局部统计；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/residual_rms_cast_2.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/residual_rms_cast_2.cpp) | 残差相加、类型转换及后RMS局部统计；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/residual_rms_cast_3.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/residual_rms_cast_3.cpp) | 残差相加、类型转换及后RMS局部统计；独立注册的专用分组/调用版本，后缀不代表可删除副本。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/rms_recip.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/rms_recip.cpp) | 计算RMS归一化倒数。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/silu.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/silu.cpp) | 对应Gate/Up的SiLU激活与逐元素乘积。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/kernels/aiv/x_gamma0.cpp](qwen3-decode-layer/simpler-operator/kernels/aiv/x_gamma0.cpp) | 输入RMS相关的gamma加权准备。 | 保留；kernel_config.py引用的生成核，变更分组应重新生成而非手动删文件。 |
| [simpler-operator/orchestration/decode_fwd_layers.cpp](qwen3-decode-layer/simpler-operator/orchestration/decode_fwd_layers.cpp) | 生成的AICPU编排：创建/提交任务、传递参数、连接依赖和中间张量。 | 保留；删除会使随包Simpler无法重建运行。 |
| [simpler-operator/test_qwen3_decode_layer.py](qwen3-decode-layer/simpler-operator/test_qwen3_decode_layer.py) | Simpler replay薄入口，导入兄弟Python目录并传入runtime_dir执行生成C++。 | 保留；当前Simpler命令入口。 |

## 本地运行产物

本次扫描另有394个本地/忽略文件，合计31.57 MiB。这些文件没有加入Git。完整逐文件路径、大小、用途与删除建议见[FILE_INVENTORY.csv](FILE_INVENTORY.csv)。CSV是本次目录快照，未来运行产生的新文件不会自动加入。

## 重命名及验证

三个根级原始泳道统一为`chip_swimlane_records.json`。同步更新根README、B README、SVG生成器、B分析器和PROVENANCE键名/受影响文件哈希；原始采集字节保持不变。其他仓库的历史采集快照不在本次重命名范围内。

检验包括：三份原始采集哈希与重命名前一致；全部PROVENANCE文件哈希匹配；SVG可按新路径生成；B采集分析能读取新路径；文档链接可解析；旧大写名称无残留（历史old/不改）。
