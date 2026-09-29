# sim120 模拟泳道 — `ht_batch100_mtp3`

离线 ASAP 仿真（**非**上板）：AIC=120，AIV=240（a2a3 24→×5）。
调度语义同 `artifacts/tools/sched_demand_estimate.py::simulate_120`。
绑核：同名兄弟调用去掉 tensormap 串行链，放到不同核上同时开始；跨 kernel 依赖保留，且消费方等待整组兄弟结束。同核同一时刻只跑一个任务。eligible 核按累计忙碌最少分散；写出前 per-core cycle 打包保证同核无重叠。

| 项 | 值 |
|----|-----|
| 逻辑任务 | 239 |
| 仿真 span | 1339.47 µs |
| 绑核/打包后 span | 1340.0 µs |
| 页口径 N (MIX计AIC) | 2679 = 1437+842+400 |
| 泳道行数 | 3079（MIX 每实例 1+1 轨） |
| D_peak /5µs | 240 |
| C_peak /5µs | 200 |
| cycle_bumps / same_core_overlaps | 120 / 0 |
| 对照实测 N / span | 2679 / 6051.36 µs |

## 串行 AIV 段（对照）

与 CSA LT 同构图：deps 上仍有同名 `*dequant*` / `*quant*` tensormap 链，导出后同名边拆除，AIV 最长 DAG ≈ **11**。  
相对「最串」的 100µs 窗约 **[1020,1120)**：~19 个 `quant` 类 AIV、peak≈6——尾段收尾，并发不高，但远达不到 Qwen LT fold 段「76 事件 / peak 3」那种假 WAW 拉长效应。  
主跨度仍由 AIC/MIX 批次数×tile 决定（span ~1340µs）。

## 打开方式

- **pypto3 toolkit**：打开本目录（需同目录 `deps.json` + `name_map.json`，任务名经 `kernel_ids` → `callable_id_to_name` 解析）
- Perfetto: https://ui.perfetto.dev/ → Open `merged_swimlane.json`
- 或 Chrome `chrome://tracing`

文件：

- [`merged_swimlane.json`](merged_swimlane.json) — Chrome Trace（120 AIC + 240 AIV 轨）
- [`chip_swimlane_records.json`](chip_swimlane_records.json) — 合成 records（可 `read_perf_data`）
- [`deps.json`](deps.json) / [`name_map.json`](name_map.json) — toolkit 侧车（自叶子目录拷贝）
- [`summary.json`](summary.json)
