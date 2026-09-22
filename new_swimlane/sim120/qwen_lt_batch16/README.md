# sim120 模拟泳道 — `lt_batch16`

离线 ASAP 仿真（**非**上板）：AIC=120，AIV=240（a2a3 24→×5）。
调度语义同 `artifacts/tools/sched_demand_estimate.py::simulate_120`。
绑核：同名兄弟调用去掉 tensormap 串行链，放到不同核上同时开始；跨 kernel 依赖保留，且消费方等待整组兄弟结束。同核同一时刻只跑一个任务。eligible 核按累计忙碌最少分散；写出前 per-core cycle 打包保证同核无重叠。

| 项 | 值 |
|----|-----|
| 逻辑任务 | 405 |
| 仿真 span | 261.62 µs |
| 绑核/打包后 span | 262.0 µs |
| 页口径 N (MIX计AIC) | 687 = 375+192+120 |
| 泳道行数 | 807（MIX 每实例 1+1 轨） |
| D_peak /5µs | 240 |
| C_peak /5µs | 240 |
| cycle_bumps / same_core_overlaps | 4 / 0 |
| 对照实测 N / span | 703 / 4235.22 µs |

## 去掉的 fold 依赖

删掉 fold→fold 边 166 条（跨 n 的 `gate_fold_2 → gate_fold` 以及 tile 内四段 RMW 链）。各 n 的列区间不交，`S[n]=Σ_k A_k[n]` 不要求这些边。同一 tile 的四段 fold 与 17 路 n 同时开始；`silu` 仍等齐本 tile 的全部分段。

## 去掉的 out_proj / residual_rms_cast 依赖

删掉 tensormap 边 260 条：`out_proj → out_proj` / `out_proj → out_proj_0`（列区间不交，或同一列上 atomic-add 可交换），`residual_rms_cast → _0 → _1 → _2 → _3`（各写 1024 列，互不覆盖），`gate_proj_k → gate_proj_{k+4}` / `up_proj` 同理（同一 per-k buffer 上 SPMD 与 deferred 的列不交），以及由此把整张量当成 overlap=covered 的跨消费者边。显式边保留：每个 cast 仍等覆盖自己 K 段的 out_proj。

## 打开方式

- **pypto3 toolkit**：打开本目录（需同目录 `deps.json` + `name_map.json`，任务名经 `kernel_ids` → `callable_id_to_name` 解析）
- Perfetto: https://ui.perfetto.dev/ → Open `merged_swimlane.json`
- 或 Chrome `chrome://tracing`

文件：

- [`merged_swimlane.json`](merged_swimlane.json) — Chrome Trace（120 AIC + 240 AIV 轨）
- [`chip_swimlane_records.json`](chip_swimlane_records.json) — 合成 records（可 `read_perf_data`）
- [`deps.json`](deps.json) / [`name_map.json`](name_map.json) — toolkit 侧车（自叶子目录拷贝）
- [`summary.json`](summary.json)
