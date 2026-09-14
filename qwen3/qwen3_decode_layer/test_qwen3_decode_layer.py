#!/usr/bin/env python3
"""Harvested pypto-lib Qwen3-14B single-layer decode_fwd_layers (_decode_layer).

Provenance: see ../../VERSIONS.md. C++ under orchestration/ + kernels/ is
verbatim pypto codegen for models/qwen3_14b/decode_fwd.py (default single-layer).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, TensorArg, scene_test

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 17408
BLOCK_SIZE = 128
HALF_DIM = HEAD_DIM // 2


def _max_blocks_per_seq() -> int:
    return (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE


def _paged_block_table_slot_mapping(seq_lens: torch.Tensor):
    batch = int(seq_lens.numel())
    max_blocks = _max_blocks_per_seq()
    block_table = torch.arange(batch * max_blocks, dtype=torch.int32)
    slot_mapping = (seq_lens - 1).to(torch.int32) + (
        torch.arange(batch, dtype=torch.int32) * max_blocks * BLOCK_SIZE
    )
    return block_table, slot_mapping


def _synth(seed: int = 1234) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def rn(shape, std=1.0, bias=0.0):
        return torch.empty(shape).normal_(0.0, std, generator=g) + bias

    seq_lens = torch.randint(1, MAX_SEQ + 1, (BATCH,), generator=g, dtype=torch.int32)
    cache_rows = BATCH * _max_blocks_per_seq() * NUM_KV_HEADS * BLOCK_SIZE
    block_table, slot_mapping = _paged_block_table_slot_mapping(seq_lens)
    posv = torch.arange(MAX_SEQ).float().unsqueeze(1)
    inv_freq = 1.0 / (1.0e4 ** (torch.arange(0, HALF_DIM).float() / HALF_DIM))
    ang = posv * inv_freq.unsqueeze(0)
    rope_cos = torch.cat([ang.cos(), ang.cos()], dim=1).float()
    rope_sin = torch.cat([ang.sin(), ang.sin()], dim=1).float()
    return {
        "hidden_states": rn([BATCH, HIDDEN], 1.0).to(torch.bfloat16),
        "input_rms_weight": rn([1, HIDDEN], 0.1, 1.0).float(),
        "wq": rn([HIDDEN, HIDDEN], 0.02).to(torch.bfloat16),
        "wk": rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16),
        "wv": rn([HIDDEN, KV_HIDDEN], 0.02).to(torch.bfloat16),
        "q_norm_weight": rn([1, HEAD_DIM], 0.1, 1.0).float(),
        "k_norm_weight": rn([1, HEAD_DIM], 0.1, 1.0).float(),
        "seq_lens": seq_lens,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "k_cache": rn([cache_rows, HEAD_DIM], 0.01).to(torch.bfloat16),
        "v_cache": rn([cache_rows, HEAD_DIM], 0.02, 0.3).to(torch.bfloat16),
        "wo": rn([HIDDEN, HIDDEN], 0.0006).to(torch.bfloat16),
        "w_gate": rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16),
        "w_up": rn([HIDDEN, INTERMEDIATE], 0.02).to(torch.bfloat16),
        "w_down": rn([INTERMEDIATE, HIDDEN], 0.0004).to(torch.bfloat16),
        "post_rms_weight": rn([1, HIDDEN], 0.1, 1.0).float(),
        "out": torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16),
    }


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen3DecodeLayer(SceneTestCase):
    """Single-layer Qwen3 decode harvested from pypto-lib decode_fwd_layers."""

    RTOL = 3e-3
    ATOL = 3e-3
    # Full torch golden lives in pypto-lib; onboard smoke verifies compile+run+ABI.
    SKIP_GOLDEN = True

    CALLABLE = {
        "orchestration": {
            "source": 'orchestration/decode_fwd_layers.cpp',
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "name": 'copy_hidden',
                "source": 'kernels/aiv/copy_hidden.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "name": 'x_gamma0',
                "source": 'kernels/aiv/x_gamma0.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 2,
                "name": 'attn_out_seed',
                "source": 'kernels/aiv/attn_out_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT],
            },
            {
                "func_id": 3,
                "name": 'rms_recip',
                "source": 'kernels/aiv/rms_recip.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 4,
                "name": 'q_seed',
                "source": 'kernels/aiv/q_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT],
            },
            {
                "func_id": 5,
                "name": 'q_proj',
                "source": 'kernels/aic/q_proj.cpp',
                "core_type": 'aic',
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 6,
                "name": 'kv_seed',
                "source": 'kernels/aiv/kv_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT],
            },
            {
                "func_id": 7,
                "name": 'mlp_out_seed',
                "source": 'kernels/aiv/mlp_out_seed.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.OUT, D.OUT],
            },
            {
                "func_id": 8,
                "name": 'k_proj',
                "source": 'kernels/aic/k_proj.cpp',
                "core_type": 'aic',
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 9,
                "name": 'v_proj',
                "source": 'kernels/aic/v_proj.cpp',
                "core_type": 'aic',
                "signature": [D.INOUT, D.IN, D.IN],
            },
            {
                "func_id": 10,
                "name": 'attn_swpipe_aic',
                "source": 'kernels/aic/attn_swpipe_aic.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.OUT],
            },
            {
                "func_id": 11,
                "name": 'attn_swpipe_aiv',
                "source": 'kernels/aiv/attn_swpipe_aiv.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.INOUT, D.INOUT, D.INOUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.OUT],
            },
            {
                "func_id": 12,
                "name": 'out_proj',
                "source": 'kernels/aic/out_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 13,
                "name": 'out_proj_0',
                "source": 'kernels/aic/out_proj_0.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 14,
                "name": 'residual_rms_cast',
                "source": 'kernels/aiv/residual_rms_cast.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 15,
                "name": 'residual_rms_cast_0',
                "source": 'kernels/aiv/residual_rms_cast_0.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 16,
                "name": 'residual_rms_cast_1',
                "source": 'kernels/aiv/residual_rms_cast_1.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 17,
                "name": 'residual_rms_cast_2',
                "source": 'kernels/aiv/residual_rms_cast_2.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 18,
                "name": 'residual_rms_cast_3',
                "source": 'kernels/aiv/residual_rms_cast_3.cpp',
                "core_type": 'aiv',
                "signature": [D.INOUT, D.INOUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 19,
                "name": 'post_rms_reduce',
                "source": 'kernels/aiv/post_rms_reduce.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 20,
                "name": 'gate_proj',
                "source": 'kernels/aic/gate_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 21,
                "name": 'up_proj',
                "source": 'kernels/aic/up_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 22,
                "name": 'gate_proj_0',
                "source": 'kernels/aic/gate_proj_0.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 23,
                "name": 'up_proj_0',
                "source": 'kernels/aic/up_proj_0.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 24,
                "name": 'gate_proj_1',
                "source": 'kernels/aic/gate_proj_1.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 25,
                "name": 'up_proj_1',
                "source": 'kernels/aic/up_proj_1.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 26,
                "name": 'gate_proj_2',
                "source": 'kernels/aic/gate_proj_2.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 27,
                "name": 'up_proj_2',
                "source": 'kernels/aic/up_proj_2.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 28,
                "name": 'gate_proj_3',
                "source": 'kernels/aic/gate_proj_3.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 29,
                "name": 'up_proj_3',
                "source": 'kernels/aic/up_proj_3.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 30,
                "name": 'gate_proj_4',
                "source": 'kernels/aic/gate_proj_4.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 31,
                "name": 'up_proj_4',
                "source": 'kernels/aic/up_proj_4.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 32,
                "name": 'silu',
                "source": 'kernels/aiv/silu.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 33,
                "name": 'down_proj',
                "source": 'kernels/aic/down_proj.cpp',
                "core_type": 'aic',
                "signature": [D.IN, D.IN, D.INOUT],
            },
            {
                "func_id": 34,
                "name": 'dcr_xgamma',
                "source": 'kernels/aiv/dcr_xgamma.cpp',
                "core_type": 'aiv',
                "signature": [D.IN, D.IN, D.OUT, D.IN, D.OUT],
            },
            {
                "func_id": 35,
                "name": 'copy_out',
                "source": 'kernels/aiv/copy_out.cpp',
                "core_type": 'aiv',
                "signature": [D.OUT, D.IN],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4},
            "params": {"seed": 1234},
            "skip_golden": True,
        },
    ]

    def generate_args(self, params):
        tensors = _synth(seed=int(params.get("seed", 1234)))
        order = [
            "hidden_states", "input_rms_weight", "wq", "wk", "wv",
            "q_norm_weight", "k_norm_weight", "seq_lens", "block_table",
            "slot_mapping", "rope_cos", "rope_sin", "k_cache", "v_cache",
            "wo", "w_gate", "w_up", "w_down", "post_rms_weight", "out",
        ]
        return TaskArgsBuilder(*[TensorArg(n, tensors[n]) for n in order])

    def compute_golden(self, args, params):
        return


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
