# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "attention_csa_test.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "hc_pre_rms", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_rms.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 1, "name": "hc_pre_linear", "source": str(_ROOT_DIR / "kernels" / "aic" / "hc_pre_linear.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 2, "name": "hc_pre_linear_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_linear_reduce.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 3, "name": "split_pre_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "split_pre_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 4, "name": "comb_sinkhorn", "source": str(_ROOT_DIR / "kernels" / "aiv" / "comb_sinkhorn.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 5, "name": "mix_x", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mix_x.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 6, "name": "csa_rope_step", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_rope_step.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 7, "name": "rope_interleave", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_interleave.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.OUT]},
	{"func_id": 8, "name": "csa_cmp_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_cmp_rope.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 9, "name": "rope_interleave_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_interleave_0.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.OUT]},
	{"func_id": 10, "name": "rms_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rms_norm.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 11, "name": "q_rope_prepare", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_rope_prepare.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 12, "name": "qr_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 13, "name": "qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 14, "name": "qr_rms_norm_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rms_norm_quant.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 15, "name": "qproj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qproj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 16, "name": "qproj_dequant_rms_nope_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qproj_dequant_rms_nope_rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 17, "name": "kv_proj_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_proj_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 18, "name": "kv_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 19, "name": "kv_rms_norm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_rms_norm_rope.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 20, "name": "csa_cache_writeback", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_cache_writeback.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 21, "name": "kv_score_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 22, "name": "scatter_softmax_pool", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 23, "name": "rmsnorm_rope_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope_cache_write.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 24, "name": "idx_qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "idx_qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 25, "name": "idx_qr_proj_dequant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "idx_qr_proj_dequant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN, _D.IN]},
	{"func_id": 26, "name": "qr_rope_swap_idx", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rope_swap_idx.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 27, "name": "qr_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 28, "name": "qr_hadamard_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_hadamard_matmul.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 29, "name": "qr_hadamard_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_hadamard_quant.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN]},
	{"func_id": 30, "name": "weights_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "weights_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 31, "name": "weights_proj_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "weights_proj_reduce.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 32, "name": "kv_score_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_score_proj_0.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 33, "name": "scatter_softmax_pool_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "scatter_softmax_pool_0.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 34, "name": "rmsnorm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rmsnorm_rope.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 35, "name": "kv_hadamard", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_hadamard.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 36, "name": "kv_and_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_and_cache_write.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 37, "name": "score_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "score_aic.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 38, "name": "score_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "score_aiv.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 39, "name": "topk", "source": str(_ROOT_DIR / "kernels" / "aiv" / "topk.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 40, "name": "kv_touch", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_touch.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 41, "name": "csa_slots_build_valid_qk_plan", "source": str(_ROOT_DIR / "kernels" / "aiv" / "csa_slots_build_valid_qk_plan.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.OUT, _D.IN]},
	{"func_id": 42, "name": "qk_pv_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "qk_pv_aic.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 43, "name": "qk_pv_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_pv_aiv.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 44, "name": "rope_cs_swap", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_cs_swap.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 45, "name": "rope_cs", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_cs.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 46, "name": "merge_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "merge_norm.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 47, "name": "proj_a_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "proj_a_mm.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 48, "name": "quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "quant.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN]},
	{"func_id": 49, "name": "proj_b_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "proj_b_mm.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 50, "name": "proj_b_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "proj_b_act.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 51, "name": "hc_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN]},
]
