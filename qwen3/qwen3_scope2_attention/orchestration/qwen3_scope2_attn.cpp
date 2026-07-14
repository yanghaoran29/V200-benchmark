// Scope2-only Qwen3 attention (rope → qk → softmax → sv → online_softmax).
// Extracted from qwen3_dynamic_tensormap orchestration; user_batch from
// orch_args.tensor(0).shapes[0] (Case1 uses 30).
//
// External args (expected_arg_count=11):
//   0 q_proj_norm [B,5120] fp32
//   1 k_proj_norm [B,1024] fp32
//   2 v_proj      [B,1024] fp32
//   3 seq_lens    [B]      int32
//   4 block_table [B*32]   int32
//   5 slot_mapping[B]      int32
//   6 rope_cos    [4096,128] fp32
//   7 rope_sin    [4096,128] fp32
//   8 k_cache     [cache_rows,128] bf16 INOUT
//   9 v_cache     [cache_rows,128] bf16 INOUT
//  10 attn_out    [B,5120] bf16 OUT
//
// Kernel func_ids: 0 rope, 1 qk_matmul, 2 softmax, 3 sv_matmul, 4 online_softmax.
// Default QWEN3_SPMD_TIER=4 (all-spmd); attention ops have 4 chunks.

#include "runtime.h"
#include <stdint.h>
#include <algorithm>

#include "pto_orchestration_api.h"

#ifndef QWEN3_SPMD_TIER
#define QWEN3_SPMD_TIER 4
#endif

namespace {
constexpr int kSpmdTier = QWEN3_SPMD_TIER;
static_assert(kSpmdTier >= 0 && kSpmdTier <= 4, "QWEN3_SPMD_TIER must be 0..4");
constexpr int kSpmdAllWidth = 1 << 30;
constexpr int kSpmdTargets[5] = {1, 2, 4, 8, kSpmdAllWidth};
constexpr int kSpmdTarget = kSpmdTargets[kSpmdTier];

constexpr int blocks_per_task(int total_chunks) {
    return total_chunks < kSpmdTarget ? total_chunks : kSpmdTarget;
}

inline Tensor alloc_tensor(const uint32_t shapes[], uint32_t ndims, DataType dtype) {
    TensorCreateInfo ci(shapes, ndims, dtype);
    TaskOutputTensors out = alloc_tensors(ci);
    return out.get_ref(0);
}
}  // namespace

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{.expected_arg_count = 11};
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const L2TaskArgs& orch_args) {
    Tensor ext_q_proj_norm = orch_args.tensor(0).ref();   // [B,5120] fp32
    Tensor ext_k_proj_norm = orch_args.tensor(1).ref();   // [B,1024] fp32
    Tensor ext_v_proj      = orch_args.tensor(2).ref();   // [B,1024] fp32
    Tensor ext_seq_lens    = orch_args.tensor(3).ref();   // [B] int32
    Tensor ext_block_table = orch_args.tensor(4).ref();   // [B*32] int32
    Tensor ext_slot_mapping = orch_args.tensor(5).ref();  // [B] int32
    Tensor ext_rope_cos    = orch_args.tensor(6).ref();   // [4096,128] fp32
    Tensor ext_rope_sin    = orch_args.tensor(7).ref();   // [4096,128] fp32
    Tensor ext_k_cache     = orch_args.tensor(8).ref();   // [cache_rows,128] bf16
    Tensor ext_v_cache     = orch_args.tensor(9).ref();   // [cache_rows,128] bf16
    Tensor ext_attn_out    = orch_args.tensor(10).ref();  // [B,5120] bf16

    // RO / host-filled inputs and whole-cache buffer: skip OverlapMap (same as full
    // decode). attn_out stays tracked so online_softmax col-slice views get proper
    // tensormap edges.
    for (Tensor* t : {&ext_q_proj_norm, &ext_k_proj_norm, &ext_v_proj, &ext_seq_lens,
                      &ext_block_table, &ext_slot_mapping, &ext_rope_cos, &ext_rope_sin,
                      &ext_k_cache, &ext_v_cache}) {
        t->manual_dep = true;
    }

    int64_t user_batch = static_cast<int64_t>(ext_q_proj_norm.shapes[0]);

    PTO2_SCOPE() {
        // all_q_padded rows = user_batch * num_q_heads(=40)? No — full decode uses
        // BATCH*128 = BATCH * HEAD_DIM. Keep same convention: [B*128, 128].
        uint32_t aqp_shp[2] = {static_cast<uint32_t>(user_batch * 128), 128};
        Tensor all_q_padded = alloc_tensor(aqp_shp, 2, DataType::BFLOAT16);

        const int qk_total_chunks = 4, sm_total_chunks = 4, sv_total_chunks = 4, os_total_chunks = 4;
        const int qk_blocks_per_task = blocks_per_task(qk_total_chunks);
        const int sm_blocks_per_task = blocks_per_task(sm_total_chunks);
        const int sv_blocks_per_task = blocks_per_task(sv_total_chunks);
        const int os_blocks_per_task = blocks_per_task(os_total_chunks);

        for (int64_t b = 0; b < user_batch; b += 1) {
            PTO2_SCOPE() {
                uint32_t all_raw_scores_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_raw_scores_ci(all_raw_scores_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_exp_padded_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_exp_padded_ci(all_exp_padded_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t all_cur_mi_ci_shapes[2] = {4096, 1};
                TensorCreateInfo all_cur_mi_ci(all_cur_mi_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_cur_li_ci_shapes[2] = {4096, 1};
                TensorCreateInfo all_cur_li_ci(all_cur_li_ci_shapes, 2, DataType::FLOAT32);
                uint32_t all_oi_tmp_ci_shapes[2] = {4096, 128};
                TensorCreateInfo all_oi_tmp_ci(all_oi_tmp_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_ws = alloc_tensors(
                    all_raw_scores_ci, all_exp_padded_ci, all_cur_mi_ci, all_cur_li_ci, all_oi_tmp_ci);
                const Tensor& all_raw_scores = alloc_ws.get_ref(0);
                const Tensor& all_exp_padded = alloc_ws.get_ref(1);
                const Tensor& all_cur_mi = alloc_ws.get_ref(2);
                const Tensor& all_cur_li = alloc_ws.get_ref(3);
                const Tensor& all_oi_tmp = alloc_ws.get_ref(4);

                uint32_t indices_ctx_len[1] = {static_cast<uint32_t>(b)};
                int32_t ctx_len = get_tensor_data<int32_t>(ext_seq_lens, 1, indices_ctx_len);
                int64_t pos = (static_cast<int64_t>(ctx_len) - 1);
                int64_t ctx_blocks = ((static_cast<int64_t>(ctx_len) + 127) / 128);
                int64_t block_table_base = (b * 32);
                uint32_t indices_slot[1] = {static_cast<uint32_t>(b)};
                int32_t slot = get_tensor_data<int32_t>(ext_slot_mapping, 1, indices_slot);
                int64_t slot_block = (static_cast<int64_t>(slot) / 128);
                int64_t slot_offset = (static_cast<int64_t>(slot) - (slot_block * 128));

                uint32_t cos_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                uint32_t cos_row_shapes[2] = {
                    (cos_row_offsets[0] >= ext_rope_cos.shapes[0]
                         ? 0u
                         : std::min<uint32_t>(1, ext_rope_cos.shapes[0] - cos_row_offsets[0])),
                    (cos_row_offsets[1] >= ext_rope_cos.shapes[1]
                         ? 0u
                         : std::min<uint32_t>(128, ext_rope_cos.shapes[1] - cos_row_offsets[1]))};
                Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
                uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                uint32_t sin_row_shapes[2] = {
                    (sin_row_offsets[0] >= ext_rope_sin.shapes[0]
                         ? 0u
                         : std::min<uint32_t>(1, ext_rope_sin.shapes[0] - sin_row_offsets[0])),
                    (sin_row_offsets[1] >= ext_rope_sin.shapes[1]
                         ? 0u
                         : std::min<uint32_t>(128, ext_rope_sin.shapes[1] - sin_row_offsets[1]))};
                Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);

                uint32_t cos_lo_offsets[2] = {0, 0};
                uint32_t cos_lo_shapes[2] = {1, 64};
                Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets, /*in_manual_dep=*/true);
                uint32_t cos_hi_offsets[2] = {0, 64};
                uint32_t cos_hi_shapes[2] = {1, 64};
                Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets, /*in_manual_dep=*/true);
                uint32_t sin_lo_offsets[2] = {0, 0};
                uint32_t sin_lo_shapes[2] = {1, 64};
                Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets, /*in_manual_dep=*/true);
                uint32_t sin_hi_offsets[2] = {0, 64};
                uint32_t sin_hi_shapes[2] = {1, 64};
                Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets, /*in_manual_dep=*/true);

                uint32_t row_off[2] = {static_cast<uint32_t>(b), 0u};
                uint32_t row1024[2] = {1u, 1024u};
                uint32_t row5120[2] = {1u, 5120u};
                Tensor k_proj_norm_row = ext_k_proj_norm.view(row1024, row_off, /*in_manual_dep=*/true);
                Tensor v_proj_row = ext_v_proj.view(row1024, row_off, /*in_manual_dep=*/true);
                Tensor q_proj_norm_row = ext_q_proj_norm.view(row5120, row_off, /*in_manual_dep=*/true);

                // Task 0: rope_kv_cache
                L0TaskArgs params_t0;
                params_t0.add_output(all_q_padded);
                params_t0.add_output(ext_k_cache);
                params_t0.add_output(ext_v_cache);
                params_t0.add_input(k_proj_norm_row);
                params_t0.add_input(cos_lo);
                params_t0.add_input(sin_lo);
                params_t0.add_input(cos_hi);
                params_t0.add_input(sin_hi);
                params_t0.add_input(v_proj_row);
                params_t0.add_input(q_proj_norm_row);
                params_t0.add_scalar(slot_block);
                params_t0.add_scalar(slot_offset);
                params_t0.add_scalar(b);
                rt_submit_aiv_task(0, params_t0);

                Tensor attn_row = ext_attn_out.view(row5120, row_off);

                // Task 1: qk_matmul
                for (int base = 0; base < qk_total_chunks; base += qk_blocks_per_task) {
                    int cur_blocks = std::min(qk_blocks_per_task, qk_total_chunks - base);
                    L0TaskArgs params_t1;
                    params_t1.add_input(all_q_padded);
                    params_t1.add_output(all_raw_scores);
                    params_t1.add_input(ext_block_table);
                    params_t1.add_input(ext_k_cache);
                    params_t1.add_scalar(b);
                    params_t1.add_scalar(ctx_blocks);
                    params_t1.add_scalar(block_table_base);
                    params_t1.add_scalar(static_cast<int64_t>(base));
                    params_t1.launch_spec.set_block_num(cur_blocks);
                    rt_submit_aic_task(1, params_t1);
                }

                // Task 2: softmax
                for (int base = 0; base < sm_total_chunks; base += sm_blocks_per_task) {
                    int cur_blocks = std::min(sm_blocks_per_task, sm_total_chunks - base);
                    L0TaskArgs params_t2;
                    params_t2.add_output(all_cur_li);
                    params_t2.add_output(all_cur_mi);
                    params_t2.add_output(all_exp_padded);
                    params_t2.add_input(all_raw_scores);
                    params_t2.add_scalar(ctx_blocks);
                    params_t2.add_scalar(ctx_len);
                    params_t2.add_scalar(static_cast<int64_t>(base));
                    params_t2.launch_spec.set_block_num(cur_blocks);
                    rt_submit_aiv_task(2, params_t2);
                }

                // Task 3: sv_matmul
                for (int base = 0; base < sv_total_chunks; base += sv_blocks_per_task) {
                    int cur_blocks = std::min(sv_blocks_per_task, sv_total_chunks - base);
                    L0TaskArgs params_t3;
                    params_t3.add_output(all_oi_tmp);
                    params_t3.add_input(ext_block_table);
                    params_t3.add_input(all_exp_padded);
                    params_t3.add_input(ext_v_cache);
                    params_t3.add_scalar(ctx_blocks);
                    params_t3.add_scalar(block_table_base);
                    params_t3.add_scalar(static_cast<int64_t>(base));
                    params_t3.launch_spec.set_block_num(cur_blocks);
                    rt_submit_aic_task(3, params_t3);
                }

                // Task 4: online_softmax
                for (int base = 0; base < os_total_chunks; base += os_blocks_per_task) {
                    int cur_blocks = std::min(os_blocks_per_task, os_total_chunks - base);
                    uint32_t oi_off[2] = {static_cast<uint32_t>(base * 1024), 0};
                    uint32_t oi_shp[2] = {static_cast<uint32_t>(cur_blocks * 1024), 128};
                    Tensor all_oi_tmp_v = all_oi_tmp.view(oi_shp, oi_off);
                    uint32_t ml_off[2] = {static_cast<uint32_t>(base * 1024), 0};
                    uint32_t ml_shp[2] = {static_cast<uint32_t>(cur_blocks * 1024), 1};
                    Tensor all_cur_mi_v = all_cur_mi.view(ml_shp, ml_off);
                    Tensor all_cur_li_v = all_cur_li.view(ml_shp, ml_off);
                    uint32_t ar_off[2] = {0, static_cast<uint32_t>(base * 1280)};
                    uint32_t ar_shp[2] = {1, static_cast<uint32_t>(cur_blocks * 1280)};
                    Tensor attn_row_v = attn_row.view(ar_shp, ar_off);
                    L0TaskArgs params_t4;
                    params_t4.add_input(all_oi_tmp_v);
                    params_t4.add_input(all_cur_mi_v);
                    params_t4.add_input(all_cur_li_v);
                    params_t4.add_inout(attn_row_v);
                    params_t4.add_scalar(ctx_blocks);
                    params_t4.add_scalar(static_cast<int64_t>(base));
                    params_t4.launch_spec.set_block_num(cur_blocks);
                    rt_submit_aiv_task(4, params_t4);
                }
            }
        }
    }
}

}  // extern "C"
