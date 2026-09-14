#!/usr/bin/env python3
"""Harvested pypto-lib DeepSeek V4-Flash CSA decode attention (decode_csa.py).

Provenance: see ../../VERSIONS.md. C++ under orchestration/ + kernels/ is
verbatim pypto codegen for models/deepseek_v4_flash_mtp/decode_csa.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

# Prefer sibling pypto-lib fixture builder when present (same workspace layout).
_WS = Path(__file__).resolve().parents[2].parent
_LIB = _WS / "pypto-lib"
if _LIB.is_dir():
    sys.path.insert(0, str(_LIB))


def _synth():
    """Materialize CSA TensorSpecs via pypto-lib build_tensor_specs when available."""
    import sys
    from pathlib import Path

    model_dir = Path(__file__).resolve().parents[2].parent / "pypto-lib" / "models" / "deepseek_v4_flash_mtp"
    lib_root = model_dir.parent.parent
    sys.path.insert(0, str(model_dir))
    sys.path.insert(0, str(lib_root))
    from decode_csa import build_tensor_specs  # noqa: PLC0415

    tensors = {}
    for spec in build_tensor_specs(None):
        if spec.init_value is None:
            t = torch.zeros(list(spec.shape), dtype=spec.dtype)
        else:
            v = spec.init_value() if callable(spec.init_value) else spec.init_value
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=spec.dtype)
            if tuple(t.shape) != tuple(spec.shape):
                t = t.reshape(list(spec.shape))
            t = t.to(spec.dtype)
        tensors[spec.name] = t.contiguous()
    return tensors


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestDeepseekDecodeCsa(SceneTestCase):
    """CSA attention orchestration harvested from pypto-lib decode_csa."""

    RTOL = 1e-2
    ATOL = 1e-2
    SKIP_GOLDEN = True

    CALLABLE = {
        "orchestration": {
            "source": 'orchestration/attention_csa_test.cpp',
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": 'hc_pre_rms',
                "source": 'kernels/aiv/hc_pre_rms.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "name": 'hc_pre_linear',
                "source": 'kernels/aic/hc_pre_linear.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": 'hc_pre_linear_reduce',
                "source": 'kernels/aiv/hc_pre_linear_reduce.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 3,
                "name": 'split_pre_post',
                "source": 'kernels/aiv/split_pre_post.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 4,
                "name": 'comb_sinkhorn',
                "source": 'kernels/aiv/comb_sinkhorn.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 5,
                "name": 'mix_x',
                "source": 'kernels/aiv/mix_x.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 6,
                "name": 'csa_rope_step',
                "source": 'kernels/aiv/csa_rope_step.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": 'rope_interleave',
                "source": 'kernels/aiv/rope_interleave.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN, D.OUT],
            },
            {
                "func_id": 8,
                "name": 'csa_cmp_rope',
                "source": 'kernels/aiv/csa_cmp_rope.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 9,
                "name": 'rope_interleave_0',
                "source": 'kernels/aiv/rope_interleave_0.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN, D.OUT],
            },
            {
                "func_id": 10,
                "name": 'rms_norm',
                "source": 'kernels/aiv/rms_norm.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 11,
                "name": 'q_rope_prepare',
                "source": 'kernels/aiv/q_rope_prepare.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 12,
                "name": 'qr_proj_seed',
                "source": 'kernels/aiv/qr_proj_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT],
            },
            {
                "func_id": 13,
                "name": 'qr_proj_matmul',
                "source": 'kernels/aic/qr_proj_matmul.cpp',
                "core_type": 'aic',
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 14,
                "name": 'qr_rms_norm_quant',
                "source": 'kernels/aiv/qr_rms_norm_quant.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 15,
                "name": 'qproj_matmul',
                "source": 'kernels/aic/qproj_matmul.cpp',
                "core_type": 'aic',
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 16,
                "name": 'qproj_dequant_rms_nope_rope',
                "source": 'kernels/aiv/qproj_dequant_rms_nope_rope.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 17,
                "name": 'kv_proj_seed',
                "source": 'kernels/aiv/kv_proj_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT],
            },
            {
                "func_id": 18,
                "name": 'kv_proj_matmul',
                "source": 'kernels/aic/kv_proj_matmul.cpp',
                "core_type": 'aic',
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 19,
                "name": 'kv_rms_norm_rope',
                "source": 'kernels/aiv/kv_rms_norm_rope.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 20,
                "name": 'prefetch_o_proj_w',
                "source": 'kernels/aiv/prefetch_o_proj_w.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN],
            },
            {
                "func_id": 21,
                "name": 'csa_cache_writeback',
                "source": 'kernels/aiv/csa_cache_writeback.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 22,
                "name": 'kv_score_proj',
                "source": 'kernels/aic/kv_score_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 23,
                "name": 'scatter_softmax_pool',
                "source": 'kernels/aiv/scatter_softmax_pool.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 24,
                "name": 'rmsnorm_rope_cache_write',
                "source": 'kernels/aiv/rmsnorm_rope_cache_write.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.INOUT, D.IN, D.OUT, D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 25,
                "name": 'idx_qr_proj_matmul',
                "source": 'kernels/aic/idx_qr_proj_matmul.cpp',
                "core_type": 'aic',
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 26,
                "name": 'idx_qr_proj_dequant',
                "source": 'kernels/aiv/idx_qr_proj_dequant.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 27,
                "name": 'qr_rope_swap_idx',
                "source": 'kernels/aiv/qr_rope_swap_idx.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT],
            },
            {
                "func_id": 28,
                "name": 'qr_rope',
                "source": 'kernels/aiv/qr_rope.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 29,
                "name": 'qr_hadamard_matmul',
                "source": 'kernels/aic/qr_hadamard_matmul.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 30,
                "name": 'qr_hadamard_quant',
                "source": 'kernels/aiv/qr_hadamard_quant.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 31,
                "name": 'weights_proj',
                "source": 'kernels/aic/weights_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 32,
                "name": 'weights_proj_reduce',
                "source": 'kernels/aiv/weights_proj_reduce.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 33,
                "name": 'kv_score_proj_0',
                "source": 'kernels/aic/kv_score_proj_0.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 34,
                "name": 'scatter_softmax_pool_0',
                "source": 'kernels/aiv/scatter_softmax_pool_0.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 35,
                "name": 'rmsnorm_rope',
                "source": 'kernels/aiv/rmsnorm_rope.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 36,
                "name": 'kv_hadamard',
                "source": 'kernels/aic/kv_hadamard.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 37,
                "name": 'kv_and_cache_write',
                "source": 'kernels/aiv/kv_and_cache_write.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.OUT, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 38,
                "name": 'score_aic',
                "source": 'kernels/aic/score_aic.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 39,
                "name": 'score_aiv',
                "source": 'kernels/aiv/score_aiv.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 40,
                "name": 'topk',
                "source": 'kernels/aiv/topk.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 41,
                "name": 'kv_touch',
                "source": 'kernels/aiv/kv_touch.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT],
            },
            {
                "func_id": 42,
                "name": 'csa_slots_build_valid_qk_plan',
                "source": 'kernels/aiv/csa_slots_build_valid_qk_plan.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.OUT, D.INOUT, D.IN, D.OUT, D.INOUT, D.OUT],
            },
            {
                "func_id": 43,
                "name": 'qk_pv_aic',
                "source": 'kernels/aic/qk_pv_aic.cpp',
                "core_type": 'aic',
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 44,
                "name": 'qk_pv_aiv',
                "source": 'kernels/aiv/qk_pv_aiv.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 45,
                "name": 'rope_cs',
                "source": 'kernels/aiv/rope_cs.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 46,
                "name": 'merge_norm',
                "source": 'kernels/aiv/merge_norm.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 47,
                "name": 'proj_a_mm',
                "source": 'kernels/aic/proj_a_mm.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 48,
                "name": 'quant',
                "source": 'kernels/aiv/quant.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 49,
                "name": 'proj_b_mm',
                "source": 'kernels/aic/proj_b_mm.cpp',
                "core_type": 'aic',
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 50,
                "name": 'proj_b_act',
                "source": 'kernels/aiv/proj_b_act.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 51,
                "name": 'hc_post',
                "source": 'kernels/aiv/hc_post.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4},
            "params": {},
            "skip_golden": True,
        },
    ]

    def generate_args(self, params):
        tensors = _synth()
        order = [
            'x_hc',
            'hc_attn_fn',
            'hc_attn_scale',
            'hc_attn_base',
            'attn_norm_w',
            'wq_a',
            'wq_b',
            'wq_b_scale',
            'wkv',
            'gamma_cq',
            'gamma_ckv',
            'freqs_cos',
            'freqs_sin',
            'cmp_wkv',
            'cmp_wgate',
            'cmp_ape',
            'cmp_norm_w',
            'compress_state',
            'compress_state_block_table',
            'idx_wq_b',
            'idx_wq_b_scale',
            'weights_proj',
            'hadamard_idx',
            'inner_wkv',
            'inner_wgate',
            'inner_ape',
            'inner_norm_w',
            'inner_compress_state',
            'inner_compress_state_block_table',
            'kv_cache',
            'cmp_kv',
            'cmp_block_table',
            'idx_kv_cache',
            'idx_kv_scale',
            'idx_block_table',
            'ori_slot_mapping',
            'window_swa_indices',
            'window_swa_lens',
            'cmp_slot_mapping',
            'idx_slot_mapping',
            'state_slot_mapping',
            'inner_state_slot_mapping',
            'position_ids',
            'kv_seq_lens',
            'attn_sink',
            'wo_a',
            'wo_b',
            'wo_b_scale',
            'x_out'
        ]
        return TaskArgsBuilder(*[TensorArg(n, tensors[n]) for n in order])

    def compute_golden(self, args, params):
        return


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
