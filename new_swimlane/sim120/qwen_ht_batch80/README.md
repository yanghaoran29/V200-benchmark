# sim120 模拟泳道 — `ht_batch80`

离线 ASAP 仿真（**非**上板）：AIC=120，AIV=240（a2a3 24→×5）。
调度语义同 `artifacts/tools/sched_demand_estimate.py::simulate_120`。
绑核：同名兄弟调用去掉 tensormap 串行链，放到不同核上同时开始；跨 kernel 依赖保留，且消费方等待整组兄弟结束。同核同一时刻只跑一个任务。eligible 核按累计忙碌最少分散；写出前 per-core cycle 打包保证同核无重叠。

| 项 | 值 |
|----|-----|
| 逻辑任务 | 1345 |
| 仿真 span | 615.91 µs |
| 绑核/打包后 span | 615.0 µs |
| 页口径 N (MIX计AIC) | 2275 = 1875+280+120 |
| 泳道行数 | 2395（MIX 每实例 1+1 轨） |
| D_peak /5µs | 132 |
| C_peak /5µs | 73 |
| cycle_bumps / same_core_overlaps | 372 / 0 |
| 对照实测 N / span | 2355 / 7049.24 µs |

## 串行 AIV 段（对照 Qwen LT）

本叶 **尚未**上 per-k / `gate_fold` 改造，AIV 最长 DAG ≈ **12**，远短于 LT 的 fold 链。

- **窗内**：`residual_rms_cast → _0 → _1 → _2 → _3`（tensormap，异名 tile 写共享残差/rms 缓冲）——同名拆边无效。
- **窗间**：`copy_out → copy_out_0 → …` 类似整张量 WAW。
- 最「串」的 100µs 窗约 **[220,320)**：~30 AIV / peak≈6，主体是各窗 `residual_rms_cast_*` / `post_rms_reduce_*`，密度和细长程度都不如 LT 的 250–350µs fold 蛇。

主时间线仍由 AIC/MIX（q_proj、attn_swpipe）撑满；AIV 串行是局部尾巴，不是整图主矛盾。

## 打开方式

- **pypto3 toolkit**：打开本目录（需同目录 `deps.json` + `name_map.json`，任务名经 `kernel_ids` → `callable_id_to_name` 解析）
- Perfetto: https://ui.perfetto.dev/ → Open `merged_swimlane.json`
- 或 Chrome `chrome://tracing`

文件：

- [`merged_swimlane.json`](merged_swimlane.json) — Chrome Trace（120 AIC + 240 AIV 轨）
- [`chip_swimlane_records.json`](chip_swimlane_records.json) — 合成 records（可 `read_perf_data`）
- [`deps.json`](deps.json) / [`name_map.json`](name_map.json) — toolkit 侧车（自叶子目录拷贝）
- [`summary.json`](summary.json)
