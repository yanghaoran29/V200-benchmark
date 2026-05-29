// Orchestration Function: qwen3_decode (dynamic-tensormap, manual-scope all-SPMD variant).
// Derived from examples/qwen3/dynamic_tensormap_allspmd/orchestration/qwen3_decode.cpp; the
// dependency pattern follows simpler_testcase/qwen3_dynamic_manual_scope.
//
// Differences from the auto-emit dynamic_tensormap_allspmd variant:
//   * Outer scope uses PTO2_SCOPE(PTO2ScopeMode::MANUAL); inner PTO2_SCOPE() wrappers
//     are removed (AUTO nested in MANUAL is unsupported).
//   * All cross-task ordering is expressed explicitly with ArgWithDeps<N> + add_dep(...)
//     using PTO2TaskId values returned by rt_submit_*_task.
//
// Differences from qwen3_dynamic_manual_scope:
//   * Every chunk loop (q/k/v_proj, gate/up/silu, down_proj/down_proj_residual) and the
//     per-chunk online_softmax loop is collapsed into a single SPMD launch using
//     launch_spec.set_block_num(...), matching the dynamic_tensormap_allspmd kernels and
//     their task-argument lists (no per-chunk q0/kv0/gi0/d0 scalars; gate/up/down write
//     full INOUT tiles allocated up-front).
//
// Dependency summary (per CALLABLE func id):
//   Tile loop b0 in [0, batch_padded), step 16:
//     Func0 (rmsnorm)         : no deps
//     Func1/2/3 (q/k/v_proj)  : dep Func0 for this tile (SPMD: block_num 20/8/8)
//     Func4 (qk_norm)         : dep Func1+Func2+Func3 for this tile
//   Per-batch loop b in [0, user_batch):
//     Func5 (rope_kv_cache)   : dep Func4 for tile b/16 and all_q_padded alloc
//     Func6 (qk_matmul)       : dep Func5 and per-batch attn scratch alloc (SPMD: block_num 4)
//     Func7 (softmax)         : dep Func6 (SPMD: block_num 4)
//     Func8 (sv_matmul)       : dep Func5 and Func7 (SPMD: block_num 4)
//     Func9 (online_softmax)  : dep Func8 (SPMD: block_num 4, single launch per batch)
//   Tile loop b0 in [0, batch_padded), step 16:
//     Func10/11 (out_proj mixed) : dep every Func9 for batches in [b0, b0+cur_valid)
//     Func12 (post_rmsnorm)      : dep Func10/11
//     Func13/14 (gate/up_proj)   : dep Func12 (SPMD: block_num 34)
//     Func15 (silu)              : dep Func13+Func14 (SPMD: block_num 34)
//     Func16 (down_proj)         : dep Func15 (SPMD: block_num 40, full mlp_tile read)
//     Func17 (down_proj_residual): dep Func10/11 and Func16 (SPMD: block_num 40)

#include "runtime.h"
#include <algorithm>
#include <vector>

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

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

    PTO2_SCOPE(PTO2ScopeMode::MANUAL) {
        const int64_t user_batch = static_cast<int64_t>(orch_args.tensor(0).shapes[0]);
        const int64_t batch_padded = (((user_batch + 15) / 16) * 16);
        const int64_t num_tiles = batch_padded / 16;

        uint32_t all_q_padded_ci_shapes[2] = {11520, 128};
        TensorCreateInfo all_q_padded_ci(all_q_padded_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_0 = alloc_tensors(all_q_padded_ci);
        const Tensor& all_q_padded = alloc_0.get_ref(0);
        const PTO2TaskId all_q_padded_alloc_task = alloc_0.task_id();

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

        std::vector<PTO2TaskId> q_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> k_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> v_proj_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> qk_norm_task_per_tile(static_cast<size_t>(num_tiles), PTO2TaskId::invalid());
        std::vector<PTO2TaskId> online_softmax_task_by_b(static_cast<size_t>(user_batch), PTO2TaskId::invalid());

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
            const size_t tix = static_cast<size_t>(b0 / 16);
            uint32_t normed_tile_ci_shapes[2] = {16, 5120};
            TensorCreateInfo normed_tile_ci(normed_tile_ci_shapes, 2, DataType::BFLOAT16);
            TaskOutputTensors alloc_normed_tile = alloc_tensors(normed_tile_ci);
            const Tensor& normed_tile = alloc_normed_tile.get_ref(0);
            const int64_t cur_valid = std::min<int64_t>(user_batch - b0, 16);

            // Task 0: rmsnorm
            ArgWithDeps<256> params_t0;
            params_t0.add_input(ext_hidden_states);
            params_t0.add_output(normed_tile);
            params_t0.add_input(ext_input_rms_weight);
            params_t0.add_scalar(b0);
            params_t0.add_scalar(cur_valid);
            TaskOutputTensors __rt_rms = rt_submit_aiv_task(0, params_t0);
            const PTO2TaskId rmsnorm_id = __rt_rms.task_id();

            // Spmd q_proj_spmd: q_proj (HIDDEN / Q_OUT_CHUNK = 5120/256 = 20 blocks)
            ArgWithDeps<256> params_t1;
            params_t1.add_input(normed_tile);
            params_t1.add_input(ext_wq);
            params_t1.add_output(q_proj);
            params_t1.add_scalar(b0);
            params_t1.launch_spec.set_block_num(20);
            params_t1.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_q = rt_submit_aic_task(1, params_t1);
            q_proj_task_per_tile[tix] = __rt_q.task_id();
            const int64_t q0 = 0;

            // Spmd k_proj_spmd: k_proj (KV_HIDDEN / KV_OUT_CHUNK = 1024/128 = 8 blocks)
            ArgWithDeps<256> params_t2;
            params_t2.add_input(normed_tile);
            params_t2.add_input(ext_wk);
            params_t2.add_output(k_proj);
            params_t2.add_scalar(b0);
            params_t2.launch_spec.set_block_num(8);
            params_t2.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_k = rt_submit_aic_task(2, params_t2);
            k_proj_task_per_tile[tix] = __rt_k.task_id();

            // Spmd v_proj_spmd: v_proj (KV_HIDDEN / KV_OUT_CHUNK = 1024/128 = 8 blocks)
            ArgWithDeps<256> params_t3;
            params_t3.add_input(normed_tile);
            params_t3.add_input(ext_wv);
            params_t3.add_output(v_proj);
            params_t3.add_scalar(b0);
            params_t3.launch_spec.set_block_num(8);
            params_t3.add_dep(rmsnorm_id);
            TaskOutputTensors __rt_v = rt_submit_aic_task(3, params_t3);
            v_proj_task_per_tile[tix] = __rt_v.task_id();

            // Task 4: qk_norm — fans in q/k/v_proj tasks for this tile.
            ArgWithDeps<256> params_t4;
            params_t4.add_output(k_proj_norm);
            params_t4.add_output(q_proj_norm);
            params_t4.add_input(q_proj);
            params_t4.add_input(ext_q_norm_weight);
            params_t4.add_input(k_proj);
            params_t4.add_input(ext_k_norm_weight);
            params_t4.add_scalar(q0);
            params_t4.add_scalar(b0);
            params_t4.add_dep(q_proj_task_per_tile[tix]);
            params_t4.add_dep(k_proj_task_per_tile[tix]);
            params_t4.add_dep(v_proj_task_per_tile[tix]);
            TaskOutputTensors __rt_qk = rt_submit_aiv_task(4, params_t4);
            qk_norm_task_per_tile[tix] = __rt_qk.task_id();
        }

        uint32_t attn_out_ci_shapes[2] = {static_cast<uint32_t>(batch_padded), 5120};
        TensorCreateInfo attn_out_ci(attn_out_ci_shapes, 2, DataType::BFLOAT16);
        TaskOutputTensors alloc_7 = alloc_tensors(attn_out_ci);
        const Tensor& attn_out = alloc_7.get_ref(0);

        for (int64_t b = 0; b < user_batch; b += 1) {
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
            TaskOutputTensors alloc_8 = alloc_tensors(
                all_raw_scores_ci, all_exp_padded_ci, all_cur_mi_ci, all_cur_li_ci, all_oi_tmp_ci);
            const PTO2TaskId batch_attn_scratch_alloc_task = alloc_8.task_id();
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
            uint32_t cos_row_shapes[2] = {
                (cos_row_offsets[0] >= ext_rope_cos.shapes[0] ? 0u : std::min<uint32_t>(1, ext_rope_cos.shapes[0] - cos_row_offsets[0])),
                (cos_row_offsets[1] >= ext_rope_cos.shapes[1] ? 0u : std::min<uint32_t>(128, ext_rope_cos.shapes[1] - cos_row_offsets[1])),
            };
            Tensor cos_row = ext_rope_cos.view(cos_row_shapes, cos_row_offsets);
            uint32_t sin_row_offsets[2] = {static_cast<uint32_t>(pos), 0};
            uint32_t sin_row_shapes[2] = {
                (sin_row_offsets[0] >= ext_rope_sin.shapes[0] ? 0u : std::min<uint32_t>(1, ext_rope_sin.shapes[0] - sin_row_offsets[0])),
                (sin_row_offsets[1] >= ext_rope_sin.shapes[1] ? 0u : std::min<uint32_t>(128, ext_rope_sin.shapes[1] - sin_row_offsets[1])),
            };
            Tensor sin_row = ext_rope_sin.view(sin_row_shapes, sin_row_offsets);
            uint32_t cos_lo_offsets[2] = {0, 0};
            uint32_t cos_lo_shapes[2] = {
                (cos_lo_offsets[0] >= cos_row.shapes[0] ? 0u : std::min<uint32_t>(1, cos_row.shapes[0] - cos_lo_offsets[0])),
                (cos_lo_offsets[1] >= cos_row.shapes[1] ? 0u : std::min<uint32_t>(64, cos_row.shapes[1] - cos_lo_offsets[1])),
            };
            Tensor cos_lo = cos_row.view(cos_lo_shapes, cos_lo_offsets);
            uint32_t cos_hi_offsets[2] = {0, 64};
            uint32_t cos_hi_shapes[2] = {
                (cos_hi_offsets[0] >= cos_row.shapes[0] ? 0u : std::min<uint32_t>(1, cos_row.shapes[0] - cos_hi_offsets[0])),
                (cos_hi_offsets[1] >= cos_row.shapes[1] ? 0u : std::min<uint32_t>(64, cos_row.shapes[1] - cos_hi_offsets[1])),
            };
            Tensor cos_hi = cos_row.view(cos_hi_shapes, cos_hi_offsets);
            uint32_t sin_lo_offsets[2] = {0, 0};
            uint32_t sin_lo_shapes[2] = {
                (sin_lo_offsets[0] >= sin_row.shapes[0] ? 0u : std::min<uint32_t>(1, sin_row.shapes[0] - sin_lo_offsets[0])),
                (sin_lo_offsets[1] >= sin_row.shapes[1] ? 0u : std::min<uint32_t>(64, sin_row.shapes[1] - sin_lo_offsets[1])),
            };
            Tensor sin_lo = sin_row.view(sin_lo_shapes, sin_lo_offsets);
            uint32_t sin_hi_offsets[2] = {0, 64};
            uint32_t sin_hi_shapes[2] = {
                (sin_hi_offsets[0] >= sin_row.shapes[0] ? 0u : std::min<uint32_t>(1, sin_row.shapes[0] - sin_hi_offsets[0])),
                (sin_hi_offsets[1] >= sin_row.shapes[1] ? 0u : std::min<uint32_t>(64, sin_row.shapes[1] - sin_hi_offsets[1])),
            };
            Tensor sin_hi = sin_row.view(sin_hi_shapes, sin_hi_offsets);

            // Task 5: rope_kv_cache
            ArgWithDeps<256> params_t5;
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
            params_t5.add_dep(all_q_padded_alloc_task);
            params_t5.add_dep(qk_norm_task_per_tile[static_cast<size_t>(b / 16)]);
            TaskOutputTensors __rt_rope = rt_submit_aiv_task(5, params_t5);
            const PTO2TaskId rope_kv_id = __rt_rope.task_id();

            uint32_t attn_row_offsets[2] = {static_cast<uint32_t>(b), 0};
            uint32_t attn_row_shapes[2] = {
                (attn_row_offsets[0] >= attn_out.shapes[0] ? 0u : std::min<uint32_t>(1, attn_out.shapes[0] - attn_row_offsets[0])),
                (attn_row_offsets[1] >= attn_out.shapes[1] ? 0u : std::min<uint32_t>(5120, attn_out.shapes[1] - attn_row_offsets[1])),
            };
            Tensor attn_row = attn_out.view(attn_row_shapes, attn_row_offsets);

            // Spmd qk_matmul_spmd: qk_matmul
            ArgWithDeps<256> params_t6;
            params_t6.add_input(all_q_padded);
            params_t6.add_output(all_raw_scores);
            params_t6.add_input(ext_block_table);
            params_t6.add_input(ext_k_cache);
            params_t6.add_scalar(b);
            params_t6.add_scalar(ctx_blocks);
            params_t6.add_scalar(block_table_base);
            params_t6.launch_spec.set_block_num(4);
            params_t6.add_dep(rope_kv_id);
            params_t6.add_dep(batch_attn_scratch_alloc_task);
            TaskOutputTensors __rt_qkmm = rt_submit_aic_task(6, params_t6);
            const PTO2TaskId qk_matmul_id = __rt_qkmm.task_id();

            // Spmd softmax_spmd: softmax
            ArgWithDeps<256> params_t7;
            params_t7.add_output(all_cur_li);
            params_t7.add_output(all_cur_mi);
            params_t7.add_output(all_exp_padded);
            params_t7.add_input(all_raw_scores);
            params_t7.add_scalar(ctx_blocks);
            params_t7.add_scalar(ctx_len);
            params_t7.launch_spec.set_block_num(4);
            params_t7.add_dep(qk_matmul_id);
            TaskOutputTensors __rt_sm = rt_submit_aiv_task(7, params_t7);
            const PTO2TaskId softmax_id = __rt_sm.task_id();

            // Spmd sv_matmul_spmd: sv_matmul (needs rope's KV-cache writes + softmax outputs)
            ArgWithDeps<256> params_t8;
            params_t8.add_output(all_oi_tmp);
            params_t8.add_input(ext_block_table);
            params_t8.add_input(all_exp_padded);
            params_t8.add_input(ext_v_cache);
            params_t8.add_scalar(ctx_blocks);
            params_t8.add_scalar(block_table_base);
            params_t8.launch_spec.set_block_num(4);
            params_t8.add_dep(rope_kv_id);
            params_t8.add_dep(softmax_id);
            TaskOutputTensors __rt_sv = rt_submit_aic_task(8, params_t8);
            const PTO2TaskId sv_matmul_id = __rt_sv.task_id();

            // Spmd online_softmax_spmd: online_softmax (single SPMD launch, block_num 4)
            ArgWithDeps<256> params_t9;
            params_t9.add_input(all_oi_tmp);
            params_t9.add_input(all_cur_mi);
            params_t9.add_input(all_cur_li);
            params_t9.add_inout(attn_row);
            params_t9.add_scalar(ctx_blocks);
            params_t9.launch_spec.set_block_num(4);
            params_t9.add_dep(sv_matmul_id);
            TaskOutputTensors __rt_os = rt_submit_aiv_task(9, params_t9);
            online_softmax_task_by_b[static_cast<size_t>(b)] = __rt_os.task_id();
        }

        for (int64_t b0 = 0; b0 < batch_padded; b0 += 16) {
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
            TaskOutputTensors alloc_9 = alloc_tensors(
                resid1_tile_ci, gm_pipe_buffer_0_ci, post_norm_tile_ci, mlp_tile_ci,
                gate_tile_ci, up_tile_ci, down_tile_ci);
            const Tensor& resid1_tile = alloc_9.get_ref(0);
            const Tensor& gm_pipe_buffer_0 = alloc_9.get_ref(1);
            const Tensor& post_norm_tile = alloc_9.get_ref(2);
            const Tensor& mlp_tile = alloc_9.get_ref(3);
            const Tensor& gate_tile = alloc_9.get_ref(4);
            const Tensor& up_tile = alloc_9.get_ref(5);
            const Tensor& down_tile = alloc_9.get_ref(6);
            const int64_t cur_valid = std::min<int64_t>(user_batch - b0, 16);

            // Group out_proj_residual: MixedKernels (AIC + AIV lanes) — Func10/11.
            ArgWithDeps<256> params_t10;
            params_t10.add_input(ext_hidden_states);
            params_t10.add_input(attn_out);
            params_t10.add_input(ext_wo);
            params_t10.add_inout(resid1_tile);
            params_t10.add_output(gm_pipe_buffer_0);
            params_t10.add_scalar(b0);
            params_t10.add_scalar(cur_valid);
            MixedKernels mixed_10 = {10, 11, 11};
            params_t10.launch_spec.set_block_num(40);
            for (int64_t __row = 0; __row < cur_valid; ++__row) {
                const int64_t bb = b0 + __row;
                const PTO2TaskId& __os_tid = online_softmax_task_by_b[static_cast<size_t>(bb)];
                if (__os_tid.is_valid()) {
                    params_t10.add_dep(__os_tid);
                }
            }
            TaskOutputTensors __rt_op = rt_submit_task(mixed_10, params_t10);
            const PTO2TaskId out_proj_mixed_id = __rt_op.task_id();

            // Task 12: post_rmsnorm
            ArgWithDeps<256> params_t11;
            params_t11.add_input(resid1_tile);
            params_t11.add_output(post_norm_tile);
            params_t11.add_input(ext_post_rms_weight);
            params_t11.add_dep(out_proj_mixed_id);
            TaskOutputTensors __rt_pr = rt_submit_aiv_task(12, params_t11);
            const PTO2TaskId post_rmsnorm_id = __rt_pr.task_id();

            // Spmd gate_proj_spmd: gate_proj (INTERMEDIATE / MLP_OUT_CHUNK = 17408/512 = 34 blocks)
            ArgWithDeps<256> params_t12;
            params_t12.add_input(post_norm_tile);
            params_t12.add_input(ext_w_gate);
            params_t12.add_inout(gate_tile);
            params_t12.launch_spec.set_block_num(34);
            params_t12.add_dep(post_rmsnorm_id);
            TaskOutputTensors __rt_gate = rt_submit_aic_task(13, params_t12);
            const PTO2TaskId gate_id = __rt_gate.task_id();

            // Spmd up_proj_spmd: up_proj
            ArgWithDeps<256> params_t13;
            params_t13.add_input(post_norm_tile);
            params_t13.add_input(ext_w_up);
            params_t13.add_inout(up_tile);
            params_t13.launch_spec.set_block_num(34);
            params_t13.add_dep(post_rmsnorm_id);
            TaskOutputTensors __rt_up = rt_submit_aic_task(14, params_t13);
            const PTO2TaskId up_id = __rt_up.task_id();

            // Spmd silu_spmd: silu
            ArgWithDeps<256> params_t14;
            params_t14.add_input(gate_tile);
            params_t14.add_input(up_tile);
            params_t14.add_inout(mlp_tile);
            params_t14.launch_spec.set_block_num(34);
            params_t14.add_dep(gate_id);
            params_t14.add_dep(up_id);
            TaskOutputTensors __rt_silu = rt_submit_aiv_task(15, params_t14);
            const PTO2TaskId silu_id = __rt_silu.task_id();

            // Spmd down_proj_spmd: down_proj (HIDDEN / DOWN_OUT_CHUNK = 5120/128 = 40 blocks).
            // Reads full mlp_tile, must wait for silu.
            ArgWithDeps<256> params_t15;
            params_t15.add_input(mlp_tile);
            params_t15.add_input(ext_w_down);
            params_t15.add_inout(down_tile);
            params_t15.launch_spec.set_block_num(40);
            params_t15.add_dep(silu_id);
            TaskOutputTensors __rt_down = rt_submit_aic_task(16, params_t15);
            const PTO2TaskId down_proj_id = __rt_down.task_id();

            // Spmd down_proj_residual_spmd: down_proj_residual — needs down_proj output and resid1_tile.
            ArgWithDeps<256> params_t16;
            params_t16.add_input(down_tile);
            params_t16.add_input(resid1_tile);
            params_t16.add_output(ext_out);
            params_t16.add_scalar(cur_valid);
            params_t16.add_scalar(b0);
            params_t16.launch_spec.set_block_num(40);
            params_t16.add_dep(down_proj_id);
            params_t16.add_dep(out_proj_mixed_id);
            (void)rt_submit_aiv_task(17, params_t16);
        }
    }
}

}  // extern "C"
