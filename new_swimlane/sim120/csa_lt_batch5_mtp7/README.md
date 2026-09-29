# sim120 模拟泳道 — `lt_batch5_mtp7`

离线 ASAP 仿真（**非**上板）：AIC=120，AIV=240（a2a3 24→×5）。
调度语义同 `artifacts/tools/sched_demand_estimate.py::simulate_120`。
绑核：同名兄弟调用去掉 tensormap 串行链，放到不同核上同时开始；跨 kernel 依赖保留，且消费方等待整组兄弟结束。同核同一时刻只跑一个任务。eligible 核按累计忙碌最少分散；写出前 per-core cycle 打包保证同核无重叠。

| 项 | 值 |
|----|-----|
| 逻辑任务 | 239 |
| 仿真 span | 276.95 µs |
| 绑核/打包后 span | 277.0 µs |
| 页口径 N (MIX计AIC) | 1563 = 747+616+200 |
| 泳道行数 | 1763（MIX 每实例 1+1 轨） |
| D_peak /5µs | 200 |
| C_peak /5µs | 160 |
| cycle_bumps / same_core_overlaps | 11 / 0 |
| 对照实测 N / span | 1563 / 1745.6 µs |

## 串行 AIV？deps vs 泳道

`deps.json` 里能看到同名链，例如 `idx_qr_proj_dequant → idx_qr_proj_dequant`（×4）、`qr_rope → qr_rope`——典型整张量 tensormap WAW。  
**但导出时同名兄弟边已拆掉**，所以泳道上 AIV DAG 最长链只有约 **11**，且最忙 100µs 窗 **[70,170)** 是 **400 AIV / peak≈120**（`merge_norm` / `qr_rope` / `qproj_dequant_rms_nope_rope` 等）——这是高并发段，不是 Qwen LT 那种 peak≈3 的细蛇。

结论：CSA LT 的「deps 串行」大多被 sibling 并行化吃掉；目视泳道时不要把 deps 链直接当成仿真时间线上的串行。

## 打开方式

- **pypto3 toolkit**：打开本目录（需同目录 `deps.json` + `name_map.json`，任务名经 `kernel_ids` → `callable_id_to_name` 解析）
- Perfetto: https://ui.perfetto.dev/ → Open `merged_swimlane.json`
- 或 Chrome `chrome://tracing`

文件：

- [`merged_swimlane.json`](merged_swimlane.json) — Chrome Trace（120 AIC + 240 AIV 轨）
- [`chip_swimlane_records.json`](chip_swimlane_records.json) — 合成 records（可 `read_perf_data`）
- [`deps.json`](deps.json) / [`name_map.json`](name_map.json) — toolkit 侧车（自叶子目录拷贝）
- [`summary.json`](summary.json)
