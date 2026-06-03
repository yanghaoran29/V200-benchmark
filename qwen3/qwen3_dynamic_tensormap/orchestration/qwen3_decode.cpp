// Orchestration Function: qwen3_decode (dynamic tensormap, configurable-SPMD variant).
//
// SPMD tiering (compile-time, selected by QWEN3_SPMD_TIER):
//   Each chunkable operator runs n total sub-tasks. A tier picks, per operator,
//   the SPMD width m: m == n -> one SPMD launch (block_num = n); m == 1 -> n
//   single-task launches (block_num = 1). Both modes feed the SAME kernels:
//   kernel_entry derives its chunk index as (chunk_base + logical block_idx).
//
//   Tier 0 (no SPMD)  : every chunkable op runs m=1 (max launch count).
//   Tier 1 (partial)  : attention (qk/softmax/sv/online) and gate/up/silu use
//                       SPMD; projections, out_proj, down_proj(_residual) run m=1.
//   Tier 2 (all SPMD) : every chunkable op runs a single SPMD launch (default).
//
// Cross-task ordering is data-driven: this orchestration uses the auto-emit
// PTO2_SCOPE() (no manual deps). Multiple per-chunk launches write the same
// shared output tile, so the runtime serializes those writes (WAW) and orders
// each downstream reader after them automatically.

#include "runtime.h"
#include <iostream>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pto_orchestration_api.h"

#include <type_traits>

// QWEN3_SPMD_TIER: 0 = no SPMD, 1 = partial SPMD, 2 = all SPMD (default).
#ifndef QWEN3_SPMD_TIER
#define QWEN3_SPMD_TIER 2
#endif

namespace {
constexpr int kSpmdTier = QWEN3_SPMD_TIER;
static_assert(kSpmdTier >= 0 && kSpmdTier <= 2, "QWEN3_SPMD_TIER must be 0, 1 or 2");

// Per-operator-group SPMD enablement by tier.
constexpr bool kSpmdProj = (kSpmdTier >= 2);  // q/k/v_proj
constexpr bool kSpmdAttn = (kSpmdTier >= 1);  // qk_matmul/softmax/sv_matmul/online_softmax
constexpr bool kSpmdOut  = (kSpmdTier >= 2);  // out_proj (mixed)
constexpr bool kSpmdMlp1 = (kSpmdTier >= 1);  // gate_proj/up_proj/silu
constexpr bool kSpmdDown = (kSpmdTier >= 2);  // down_proj/down_proj_residual

// SPMD width for an operator with n total sub-tasks: n when enabled, else 1.
constexpr int spmd_width(bool enabled, int n) { return enabled ? n : 1; }

// MixedKernels is platform-shaped: a2a3/a5 expose two vector slots
// (aiv0_kernel_id, aiv1_kernel_id), 1c1v only one. Set the second vector slot
// via if-constexpr member detection so the same orchestration source compiles
// on both — a compile-time conditional that needs no platform macro.
template <typename, typename = void>
struct mk_has_aiv1 : std::false_type {};
template <typename T>
struct mk_has_aiv1<T, std::void_t<decltype(std::declval<T&>().aiv1_kernel_id)>> : std::true_type {};

template <typename MK>
inline MK make_mixed(int aic, int aiv0, int aiv1) {
    MK mk{};
    mk.aic_kernel_id = aic;
    mk.aiv0_kernel_id = aiv0;
    if constexpr (mk_has_aiv1<MK>::value) {
        mk.aiv1_kernel_id = aiv1;
    }
    return mk;
}
}  // namespace

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 20,
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(const ChipStorageTaskArgs& orch_args) {
    // External tensors
    Tensor ext_hidden_states = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_input_rms_weight = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_wq = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_wk = from_tensor_arg(orch_args.tensor(3));
    Tensor ext_wv = from_tensor_arg(orch_args.tensor(4));
    Tensor ext_q_norm_weight = from_tensor_arg(orch_args.tensor(5));
    Tensor ext_k_norm_weight = from_tensor_arg(orch_args.tensor(6));
    Tensor ext_seq_lens = from_tensor_arg(orch_args.tensor(7));
    Tensor ext_block_table = from_tensor_arg(orch_args.tensor(8));
    Tensor ext_slot_mapping = from_tensor_arg(orch_args.tensor(9));
    Tensor ext_rope_cos = from_tensor_arg(orch_args.tensor(10));
    Tensor ext_rope_sin = from_tensor_arg(orch_args.tensor(11));
    Tensor ext_k_cache = from_tensor_arg(orch_args.tensor(12));
    Tensor ext_v_cache = from_tensor_arg(orch_args.tensor(13));
    Tensor ext_wo = from_tensor_arg(orch_args.tensor(14));
    Tensor ext_post_rms_weight = from_tensor_arg(orch_args.tensor(15));
    Tensor ext_w_gate = from_tensor_arg(orch_args.tensor(16));
    Tensor ext_w_up = from_tensor_arg(orch_args.tensor(17));
    Tensor ext_w_down = from_tensor_arg(orch_args.tensor(18));
    Tensor ext_out = from_tensor_arg(orch_args.tensor(19));

    PTO2_SCOPE() {
        uint32_t all_q_padded_ci_shapes[2] = {11520, 128};
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_0 = alloc_tensors(all_q_padded_ci);
        const Tensor& all_q_padded = alloc_0.get_ref(0);
        int64_t user_batch = (int64_t)orch_args.tensor(0).shapes[0];
        int64_t batch_padded = (((user_batch + 15) / 16) * 16);
        uint32_t q_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_ci(q_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_1 = alloc_tensors(q_proj_ci);
        const Tensor& q_proj = alloc_1.get_ref(0);
        uint32_t k_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_ci(k_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_2 = alloc_tensors(k_proj_ci);
        const Tensor& k_proj = alloc_2.get_ref(0);
        uint32_t v_proj_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo v_proj_ci(v_proj_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_3 = alloc_tensors(v_proj_ci);
        const Tensor& v_proj = alloc_3.get_ref(0);
        uint32_t q_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo q_proj_norm_ci(q_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_4 = alloc_tensors(q_proj_norm_ci);
        const Tensor& q_proj_norm = alloc_4.get_ref(0);
        uint32_t k_proj_norm_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 1024};
        TensorCreateInfo k_proj_norm_ci(k_proj_norm_ci_shapes, 2, DataType::FLOAT32);
        TaskOutputTensors alloc_5 = alloc_tensors(k_proj_norm_ci);
        const Tensor& k_proj_norm = alloc_5.get_ref(0);

        // q/k/v projection chunk counts and SPMD widths.
        const int q_n = 20, k_n = 8, v_n = 8;
        const int q_m = spmd_width(kSpmdProj, q_n);
        const int k_m = spmd_width(kSpmdProj, k_n);
        const int v_m = spmd_width(kSpmdProj, v_n);

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            PTO2_SCOPE() {
                uint32_t normed_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo normed_tile_ci(normed_tile_ci_shapes, 2, DataType::BFLOAT16);
                TaskOutputTensors alloc_6 = alloc_tensors(normed_tile_ci);
                const Tensor& normed_tile = alloc_6.get_ref(0);
                int64_t cur_valid = std::min<int64_t>((user_batch - b0), 16);

                // Task 0: rmsnorm
                Arg params_t0;
                params_t0.add_input(ext_hidden_states);
                params_t0.add_output(normed_tile);
                params_t0.add_input(ext_input_rms_weight);
                params_t0.add_scalar(b0);
                params_t0.add_scalar(cur_valid);
                rt_submit_aiv_task(0, params_t0);

                // Task 1: q_proj
                for (int qi = 0; qi < q_n / q_m; ++qi) {
                    Arg params_t1;
                    params_t1.add_input(normed_tile);
                    params_t1.add_input(ext_wq);
                    params_t1.add_output(q_proj);
                    params_t1.add_scalar(b0);
                    params_t1.add_scalar(static_cast<int64_t>(qi) * q_m);
                    params_t1.launch_spec.set_block_num(q_m);
                    rt_submit_aic_task(1, params_t1);
                }
                int64_t q0 = 0;

                // Task 2: k_proj
                for (int ki = 0; ki < k_n / k_m; ++ki) {
                    Arg params_t2;
                    params_t2.add_input(normed_tile);
                    params_t2.add_input(ext_wk);
                    params_t2.add_output(k_proj);
                    params_t2.add_scalar(b0);
                    params_t2.add_scalar(static_cast<int64_t>(ki) * k_m);
                    params_t2.launch_spec.set_block_num(k_m);
                    rt_submit_aic_task(2, params_t2);
                }

                // Task 3: v_proj
                for (int vi = 0; vi < v_n / v_m; ++vi) {
                    Arg params_t3;
                    params_t3.add_input(normed_tile);
                    params_t3.add_input(ext_wv);
                    params_t3.add_output(v_proj);
                    params_t3.add_scalar(b0);
                    params_t3.add_scalar(static_cast<int64_t>(vi) * v_m);
                    params_t3.launch_spec.set_block_num(v_m);
                    rt_submit_aic_task(3, params_t3);
                }

                // Task 4: qk_norm
                Arg params_t4;
                params_t4.add_output(k_proj_norm);
                params_t4.add_output(q_proj_norm);
                params_t4.add_input(q_proj);
                params_t4.add_input(ext_q_norm_weight);
                params_t4.add_input(k_proj);
                params_t4.add_input(ext_k_norm_weight);
                params_t4.add_scalar(q0);
                params_t4.add_scalar(b0);
                rt_submit_aiv_task(4, params_t4);
            }
        }
        uint32_t attn_out_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_7 = alloc_tensors(attn_out_ci);
        const Tensor& attn_out = alloc_7.get_ref(0);

        // Attention sub-task counts (all block_num 4 in SPMD mode).
        const int qk_n = 4, sm_n = 4, sv_n = 4, os_n = 4;
        const int qk_m = spmd_width(kSpmdAttn, qk_n);
        const int sm_m = spmd_width(kSpmdAttn, sm_n);
        const int sv_m = spmd_width(kSpmdAttn, sv_n);
        const int os_m = spmd_width(kSpmdAttn, os_n);

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
                TaskOutputTensors alloc_8 = alloc_tensors(all_raw_scores_ci, all_exp_padded_ci, all_cur_mi_ci, all_cur_li_ci, all_oi_tmp_ci);
                const Tensor& all_raw_scores = alloc_8.get_ref(0);
                const Tensor& all_exp_padded = alloc_8.get_ref(1);
                const Tensor& all_cur_mi = alloc_8.get_ref(2);
                const Tensor& all_cur_li = alloc_8.get_ref(3);
                const Tensor& all_oi_tmp = alloc_8.get_ref(4);
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
                uint32_t cos_row_shapes[2] = {(cos_row_offsets[0] >= ext_rope_cos.shapes[0] ? 0u : std::min<uint32_t>(1, ext_rope_cos.shapes[0] - cos_row_offsets[0])), (cos_row_offsets[1] >= ext_rope_cos.shapes[1] ? 0u : std::min<uint32_t>(128, ext_rope_cos.shapes[1] - cos_row_offsets[1]))};
                Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
                uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
                uint32_t sin_row_shapes[2] = {(sin_row_offsets[0] >= ext_rope_sin.shapes[0] ? 0u : std::min<uint32_t>(1, ext_rope_sin.shapes[0] - sin_row_offsets[0])), (sin_row_offsets[1] >= ext_rope_sin.shapes[1] ? 0u : std::min<uint32_t>(128, ext_rope_sin.shapes[1] - sin_row_offsets[1]))};
                Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
                uint32_t cos_lo_offsets[2] = {0, 0};
                uint32_t cos_lo_shapes[2] = {(cos_lo_offsets[0] >= cos_row.shapes[0] ? 0u : std::min<uint32_t>(1, cos_row.shapes[0] - cos_lo_offsets[0])), (cos_lo_offsets[1] >= cos_row.shapes[1] ? 0u : std::min<uint32_t>(64, cos_row.shapes[1] - cos_lo_offsets[1]))};
                Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets);
                uint32_t cos_hi_offsets[2] = {0, 64};
                uint32_t cos_hi_shapes[2] = {(cos_hi_offsets[0] >= cos_row.shapes[0] ? 0u : std::min<uint32_t>(1, cos_row.shapes[0] - cos_hi_offsets[0])), (cos_hi_offsets[1] >= cos_row.shapes[1] ? 0u : std::min<uint32_t>(64, cos_row.shapes[1] - cos_hi_offsets[1]))};
                Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets);
                uint32_t sin_lo_offsets[2] = {0, 0};
                uint32_t sin_lo_shapes[2] = {(sin_lo_offsets[0] >= sin_row.shapes[0] ? 0u : std::min<uint32_t>(1, sin_row.shapes[0] - sin_lo_offsets[0])), (sin_lo_offsets[1] >= sin_row.shapes[1] ? 0u : std::min<uint32_t>(64, sin_row.shapes[1] - sin_lo_offsets[1]))};
                Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets);
                uint32_t sin_hi_offsets[2] = {0, 64};
                uint32_t sin_hi_shapes[2] = {(sin_hi_offsets[0] >= sin_row.shapes[0] ? 0u : std::min<uint32_t>(1, sin_row.shapes[0] - sin_hi_offsets[0])), (sin_hi_offsets[1] >= sin_row.shapes[1] ? 0u : std::min<uint32_t>(64, sin_row.shapes[1] - sin_hi_offsets[1]))};
                Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets);

                // Task 5: rope_kv_cache
                Arg params_t5;
                params_t5.add_output(all_q_padded);
                params_t5.add_output(ext_k_cache);
                params_t5.add_output(ext_v_cache);
                params_t5.add_input(k_proj_norm);
                params_t5.add_input(cos_lo);
                params_t5.add_input(sin_lo);
                params_t5.add_input(cos_hi);
                params_t5.add_input(sin_hi);
                params_t5.add_input(v_proj);
                params_t5.add_input(q_proj_norm);
                params_t5.add_scalar(slot_block);
                params_t5.add_scalar(slot_offset);
                params_t5.add_scalar(b);
                rt_submit_aiv_task(5, params_t5);
                uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
                uint32_t attn_row_shapes[2] = {(attn_row_offsets[0] >= attn_out.shapes[0] ? 0u : std::min<uint32_t>(1, attn_out.shapes[0] - attn_row_offsets[0])), (attn_row_offsets[1] >= attn_out.shapes[1] ? 0u : std::min<uint32_t>(5120, attn_out.shapes[1] - attn_row_offsets[1]))};
                Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

                // Task 6: qk_matmul
                for (int ci = 0; ci < qk_n / qk_m; ++ci) {
                    Arg params_t6;
                    params_t6.add_input(all_q_padded);
                    params_t6.add_output(all_raw_scores);
                    params_t6.add_input(ext_block_table);
                    params_t6.add_input(ext_k_cache);
                    params_t6.add_scalar(b);
                    params_t6.add_scalar(ctx_blocks);
                    params_t6.add_scalar(block_table_base);
                    params_t6.add_scalar(static_cast<int64_t>(ci) * qk_m);
                    params_t6.launch_spec.set_block_num(qk_m);
                    rt_submit_aic_task(6, params_t6);
                }

                // Task 7: softmax
                for (int ci = 0; ci < sm_n / sm_m; ++ci) {
                    Arg params_t7;
                    params_t7.add_output(all_cur_li);
                    params_t7.add_output(all_cur_mi);
                    params_t7.add_output(all_exp_padded);
                    params_t7.add_input(all_raw_scores);
                    params_t7.add_scalar(ctx_blocks);
                    params_t7.add_scalar(ctx_len);
                    params_t7.add_scalar(static_cast<int64_t>(ci) * sm_m);
                    params_t7.launch_spec.set_block_num(sm_m);
                    rt_submit_aiv_task(7, params_t7);
                }

                // Task 8: sv_matmul
                for (int ci = 0; ci < sv_n / sv_m; ++ci) {
                    Arg params_t8;
                    params_t8.add_output(all_oi_tmp);
                    params_t8.add_input(ext_block_table);
                    params_t8.add_input(all_exp_padded);
                    params_t8.add_input(ext_v_cache);
                    params_t8.add_scalar(ctx_blocks);
                    params_t8.add_scalar(block_table_base);
                    params_t8.add_scalar(static_cast<int64_t>(ci) * sv_m);
                    params_t8.launch_spec.set_block_num(sv_m);
                    rt_submit_aic_task(8, params_t8);
                }

                // Task 9: online_softmax
                for (int ci = 0; ci < os_n / os_m; ++ci) {
                    Arg params_t9;
                    params_t9.add_input(all_oi_tmp);
                    params_t9.add_input(all_cur_mi);
                    params_t9.add_input(all_cur_li);
                    params_t9.add_inout(attn_row);
                    params_t9.add_scalar(ctx_blocks);
                    params_t9.add_scalar(static_cast<int64_t>(ci) * os_m);
                    params_t9.launch_spec.set_block_num(os_m);
                    rt_submit_aiv_task(9, params_t9);
                }
            }
        }

        // out_proj / down_proj(_residual) chunk counts; gate/up/silu = 34.
        const int op_n = 40, gate_n = 34, up_n = 34, silu_n = 34, down_n = 40, dres_n = 40;
        const int op_m = spmd_width(kSpmdOut, op_n);
        const int gate_m = spmd_width(kSpmdMlp1, gate_n);
        const int up_m = spmd_width(kSpmdMlp1, up_n);
        const int silu_m = spmd_width(kSpmdMlp1, silu_n);
        const int down_m = spmd_width(kSpmdDown, down_n);
        const int dres_m = spmd_width(kSpmdDown, dres_n);

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            PTO2_SCOPE() {
                uint32_t resid1_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo resid1_tile_ci(resid1_tile_ci_shapes, 2, DataType::FLOAT32);
                uint32_t gm_pipe_buffer_0_ci_shapes[1] = {static_cast<uint32_t>((16384) * (40))};
                TensorCreateInfo gm_pipe_buffer_0_ci(gm_pipe_buffer_0_ci_shapes, 1, DataType::FLOAT32, /*manual_dep=*/true);
                uint32_t post_norm_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo post_norm_tile_ci(post_norm_tile_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t mlp_tile_ci_shapes[2] = {16, 17408};
                TensorCreateInfo mlp_tile_ci(mlp_tile_ci_shapes, 2, DataType::BFLOAT16);
                uint32_t gate_tile_ci_shapes[2] = {16, 17408};
                TensorCreateInfo gate_tile_ci(gate_tile_ci_shapes, 2, DataType::FLOAT32);
                uint32_t up_tile_ci_shapes[2] = {16, 17408};
                TensorCreateInfo up_tile_ci(up_tile_ci_shapes, 2, DataType::FLOAT32);
                uint32_t down_tile_ci_shapes[2] = {16, 5120};
                TensorCreateInfo down_tile_ci(down_tile_ci_shapes, 2, DataType::FLOAT32);
                TaskOutputTensors alloc_9 = alloc_tensors(resid1_tile_ci, gm_pipe_buffer_0_ci, post_norm_tile_ci, mlp_tile_ci, gate_tile_ci, up_tile_ci, down_tile_ci);
                const Tensor& resid1_tile = alloc_9.get_ref(0);
                const Tensor& gm_pipe_buffer_0 = alloc_9.get_ref(1);
                const Tensor& post_norm_tile = alloc_9.get_ref(2);
                const Tensor& mlp_tile = alloc_9.get_ref(3);
                const Tensor& gate_tile = alloc_9.get_ref(4);
                const Tensor& up_tile = alloc_9.get_ref(5);
                const Tensor& down_tile = alloc_9.get_ref(6);
                int64_t cur_valid__ssa_v1 = std::min<int64_t>((user_batch - b0), 16);

                // Task 10/11: out_proj_residual MixedKernels (AIC + AIV lanes), per output column chunk.
                for (int oi = 0; oi < op_n / op_m; ++oi) {
                    Arg params_t10;
                    params_t10.add_input(ext_hidden_states);
                    params_t10.add_input(attn_out);
                    params_t10.add_input(ext_wo);
                    params_t10.add_inout(resid1_tile);
                    params_t10.add_output(gm_pipe_buffer_0);
                    params_t10.add_scalar(b0);
                    params_t10.add_scalar(cur_valid__ssa_v1);
                    params_t10.add_scalar(static_cast<int64_t>(oi) * op_m);
                    MixedKernels mixed_10 = make_mixed<MixedKernels>(10, 11, 11);
                    params_t10.launch_spec.set_block_num(op_m);
                    rt_submit_task(mixed_10, params_t10);
                }

                // Task 12: post_rmsnorm
                Arg params_t11;
                params_t11.add_input(resid1_tile);
                params_t11.add_output(post_norm_tile);
                params_t11.add_input(ext_post_rms_weight);
                rt_submit_aiv_task(12, params_t11);

                // Task 13: gate_proj
                for (int gi = 0; gi < gate_n / gate_m; ++gi) {
                    Arg params_t12;
                    params_t12.add_input(post_norm_tile);
                    params_t12.add_input(ext_w_gate);
                    params_t12.add_inout(gate_tile);
                    params_t12.add_scalar(static_cast<int64_t>(gi) * gate_m);
                    params_t12.launch_spec.set_block_num(gate_m);
                    rt_submit_aic_task(13, params_t12);
                }

                // Task 14: up_proj
                for (int ui = 0; ui < up_n / up_m; ++ui) {
                    Arg params_t13;
                    params_t13.add_input(post_norm_tile);
                    params_t13.add_input(ext_w_up);
                    params_t13.add_inout(up_tile);
                    params_t13.add_scalar(static_cast<int64_t>(ui) * up_m);
                    params_t13.launch_spec.set_block_num(up_m);
                    rt_submit_aic_task(14, params_t13);
                }

                // Task 15: silu
                for (int si = 0; si < silu_n / silu_m; ++si) {
                    Arg params_t14;
                    params_t14.add_input(gate_tile);
                    params_t14.add_input(up_tile);
                    params_t14.add_inout(mlp_tile);
                    params_t14.add_scalar(static_cast<int64_t>(si) * silu_m);
                    params_t14.launch_spec.set_block_num(silu_m);
                    rt_submit_aiv_task(15, params_t14);
                }

                // Task 16: down_proj
                for (int di = 0; di < down_n / down_m; ++di) {
                    Arg params_t15;
                    params_t15.add_input(mlp_tile);
                    params_t15.add_input(ext_w_down);
                    params_t15.add_inout(down_tile);
                    params_t15.add_scalar(static_cast<int64_t>(di) * down_m);
                    params_t15.launch_spec.set_block_num(down_m);
                    rt_submit_aic_task(16, params_t15);
                }

                // Task 17: down_proj_residual
                for (int ri = 0; ri < dres_n / dres_m; ++ri) {
                    Arg params_t16;
                    params_t16.add_input(down_tile);
                    params_t16.add_input(resid1_tile);
                    params_t16.add_output(ext_out);
                    params_t16.add_scalar(cur_valid__ssa_v1);
                    params_t16.add_scalar(b0);
                    params_t16.add_scalar(static_cast<int64_t>(ri) * dres_m);
                    params_t16.launch_spec.set_block_num(dres_m);
                    rt_submit_aiv_task(17, params_t16);
                }
            }
        }
    }
}

}  // extern "C"
