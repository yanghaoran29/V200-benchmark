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
	"source": str(_ROOT_DIR / "orchestration" / "decode_fwd_layers.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "copy_hidden", "source": str(_ROOT_DIR / "kernels" / "aiv" / "copy_hidden.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 1, "name": "x_gamma0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "x_gamma0.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 2, "name": "attn_out_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "attn_out_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 3, "name": "rms_recip", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rms_recip.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 4, "name": "q_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 5, "name": "q_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "q_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 6, "name": "kv_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT]},
	{"func_id": 7, "name": "mlp_out_seed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mlp_out_seed.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 8, "name": "k_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "k_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 9, "name": "v_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "v_proj.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 10, "name": "attn_swpipe_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "attn_swpipe_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.OUT]},
	{"func_id": 11, "name": "attn_swpipe_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "attn_swpipe_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.OUT]},
	{"func_id": 12, "name": "out_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "out_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 13, "name": "out_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "out_proj_0.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 14, "name": "residual_rms_cast", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 15, "name": "residual_rms_cast_0", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_0.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 16, "name": "residual_rms_cast_1", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_1.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 17, "name": "residual_rms_cast_2", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_2.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 18, "name": "residual_rms_cast_3", "source": str(_ROOT_DIR / "kernels" / "aiv" / "residual_rms_cast_3.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 19, "name": "post_rms_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "post_rms_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 20, "name": "gate_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 21, "name": "up_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 22, "name": "gate_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj_0.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 23, "name": "up_proj_0", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj_0.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 24, "name": "gate_proj_1", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj_1.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 25, "name": "up_proj_1", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj_1.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 26, "name": "gate_proj_2", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj_2.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 27, "name": "up_proj_2", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj_2.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 28, "name": "gate_proj_3", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj_3.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 29, "name": "up_proj_3", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj_3.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 30, "name": "gate_proj_4", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_proj_4.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 31, "name": "up_proj_4", "source": str(_ROOT_DIR / "kernels" / "aic" / "up_proj_4.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 32, "name": "silu", "source": str(_ROOT_DIR / "kernels" / "aiv" / "silu.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 33, "name": "down_proj", "source": str(_ROOT_DIR / "kernels" / "aic" / "down_proj.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 34, "name": "dcr_xgamma", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dcr_xgamma.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.IN, _D.OUT]},
	{"func_id": 35, "name": "copy_out", "source": str(_ROOT_DIR / "kernels" / "aiv" / "copy_out.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
]
