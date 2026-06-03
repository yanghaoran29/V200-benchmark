# qwen3 SPMD 泳道图运行时间汇总

| 测试样例+条件+参数（文件夹名） | 运行时间 total_span(us) | AICore_span(us) | 事件数 | AICore事件数 |
|---|---:|---:|---:|---:|
| qwen3_dynamic_manual_scope_all_spmd__1c1vsim__Case2_thread7_block120 | 2951370.66 | 2951309.42 | 9015 | 3336 |
| qwen3_dynamic_manual_scope_all_spmd__a2a3__Case1_thread4_block24 | 3330.64 | 3283.28 | 8970 | 3576 |
| qwen3_dynamic_manual_scope_no_spmd__1c1vsim__Case2_thread7_block120 | 3006828.12 | 3006747.26 | 13998 | 3336 |
| qwen3_dynamic_manual_scope_no_spmd__a2a3__Case1_thread4_block24 | 5074.82 | 5030.64 | 11766 | 3576 |
| qwen3_dynamic_manual_scope_partial_spmd__1c1vsim__Case2_thread7_block120 | 3112718.76 | 3112627.56 | 10815 | 3336 |
| qwen3_dynamic_manual_scope_partial_spmd__a2a3__Case1_thread4_block24 | 4088.64 | 4025.7 | 9881 | 3576 |
| qwen3_dynamic_tensormap_all_spmd__1c1vsim__Case2_thread7_block120 | 2882226.1 | 2882186.18 | 8716 | 3336 |
| qwen3_dynamic_tensormap_all_spmd__a2a3__Case1_thread4_block24 | 3734.98 | 3692.26 | 8944 | 3576 |
| qwen3_dynamic_tensormap_no_spmd__1c1vsim__Case2_thread7_block120 | 5437404.66 | 5437317.74 | 14657 | 3336 |
| qwen3_dynamic_tensormap_no_spmd__a2a3__Case1_thread4_block24 | 10917.62 | 10843.92 | 13340 | 3576 |
| qwen3_dynamic_tensormap_partial_spmd__1c1vsim__Case2_thread7_block120 | 4458995.78 | 4458916.86 | 10605 | 3336 |
| qwen3_dynamic_tensormap_partial_spmd__a2a3__Case1_thread4_block24 | 7326.78 | 7269.76 | 10742 | 3576 |
