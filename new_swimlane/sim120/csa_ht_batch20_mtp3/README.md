# sim120 模拟泳道 — `ht_batch20_mtp3`

离线 ASAP 仿真（**非**上板）：AIC=120，AIV=240（a2a3 24→×5）。
调度语义同 `artifacts/tools/sched_demand_estimate.py::simulate_120`。
绑核：同名兄弟调用去掉 tensormap 串行链，放到不同核上同时开始；跨 kernel 依赖保留，且消费方等待整组兄弟结束。同核同一时刻只跑一个任务。eligible 核按累计忙碌最少分散；写出前 per-core cycle 打包保证同核无重叠。

| 项 | 值 |
|----|-----|
| 逻辑任务 | 239 |
| 仿真 span | 335.84 µs |
| 绑核/打包后 span | 336.0 µs |
| 页口径 N (MIX计AIC) | 2659 = 1433+826+400 |
| 泳道行数 | 3059（MIX 每实例 1+1 轨） |
| D_peak /5µs | 240 |
| C_peak /5µs | 240 |
| cycle_bumps / same_core_overlaps | 216 / 0 |
| 对照实测 N / span | 2659 / 2004.84 µs |



## 去掉的跨 batch-tile 依赖

调度时去掉跨 tile 前驱 20 条。同名 tensormap 边把整张量记成 overlap=covered，5 刀写的是互不重叠的行。`proj_b_act` 的显式 deps 原先包含全部 `proj_b_mm`；跨组（同一刀的各列）保留，跨刀删除。`hc_post` 只等自己那一刀的 `proj_b_act`。

## 打开方式

- **pypto3 toolkit**：打开本目录（需同目录 `deps.json` + `name_map.json`，任务名经 `kernel_ids` → `callable_id_to_name` 解析）
- Perfetto: https://ui.perfetto.dev/ → Open `merged_swimlane.json`
- 或 Chrome `chrome://tracing`

文件：

- [`merged_swimlane.json`](merged_swimlane.json) — Chrome Trace（120 AIC + 240 AIV 轨）
- [`chip_swimlane_records.json`](chip_swimlane_records.json) — 合成 records（可 `read_perf_data`）
- [`deps.json`](deps.json) / [`name_map.json`](name_map.json) — toolkit 侧车（自叶子目录拷贝）
- [`summary.json`](summary.json)
