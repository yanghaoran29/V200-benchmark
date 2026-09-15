#!/usr/bin/env python3
"""Qwen3 Scope2 attention-only sample (user_batch=30).

Extracts the per-batch attention loop from qwen3_dynamic_tensormap:
rope_kv_cache → qk_matmul → softmax → sv_matmul → online_softmax.
"""
from __future__ import annotations

import math

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

BATCH = 30
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
BLOCK_SIZE = 128
Q_HEAD_BATCH = 5
EPS = 1e-6
SYNTHETIC_PROJ_SCALE = 0.5


def _max_blocks_per_seq() -> int:
    return (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE


def _cache_rows(batch: int) -> int:
    return batch * _max_blocks_per_seq() * NUM_KV_HEADS * BLOCK_SIZE


def _compute_golden(tensors: dict) -> None:
    """Attention-only golden starting from pre-normed q/k and raw v."""
    q_proj_norm = tensors["q_proj_norm"]
    k_proj_norm = tensors["k_proj_norm"]
    v_proj = tensors["v_proj"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]

    batch = q_proj_norm.shape[0]
    head_dim = HEAD_DIM
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    max_ctx_blocks = _max_blocks_per_seq()
    q_per_kv = NUM_HEADS // NUM_KV_HEADS
    q_groups = q_per_kv // Q_HEAD_BATCH

    attn_out = torch.zeros(batch, HIDDEN, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj_norm[b].view(NUM_KV_HEADS, head_dim)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        for ki in range(NUM_KV_HEADS):
            cache_row = (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj_norm[b].view(NUM_HEADS, head_dim)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        attn_row = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
        for kvh in range(NUM_KV_HEADS):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)
                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi, li, mi = oi_tmp, cur_li, cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                attn_row[:, q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim] = ctx.reshape(
                    1, -1
                ).to(torch.bfloat16)

        attn_out[b : b + 1, :] = attn_row

    tensors["attn_out"][:] = attn_out


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen3Scope2Attention(SceneTestCase):
    RTOL = 3e-3
    ATOL = 3e-3

    CALLABLE = {
        "orchestration": {
            "source": "orchestration/qwen3_scope2_attn.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.OUT,
            ],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "rope_kv_cache",
                "source": "kernels/aiv/rope_kv_cache.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 1,
                "name": "qk_matmul",
                "source": "kernels/aic/qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 2,
                "name": "softmax",
                "source": "kernels/aiv/softmax.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.IN],
            },
            {
                "func_id": 3,
                "name": "sv_matmul",
                "source": "kernels/aic/sv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 4,
                "name": "online_softmax",
                "source": "kernels/aiv/online_softmax.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3", "a2a3sim"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"dtype": "bfloat16"},
        },
    ]

    def generate_args(self, params):
        batch = BATCH
        max_blocks = _max_blocks_per_seq()
        num_blocks = batch * max_blocks
        cache_rows = _cache_rows(batch)

        # Already "normed" Q/K and raw V — Scope2 starts after qk_norm.
        q_proj_norm = (torch.rand(batch, HIDDEN, dtype=torch.float32) - 0.5) * SYNTHETIC_PROJ_SCALE
        k_proj_norm = (torch.rand(batch, KV_HIDDEN, dtype=torch.float32) - 0.5) * SYNTHETIC_PROJ_SCALE
        v_proj = (torch.rand(batch, KV_HIDDEN, dtype=torch.float32) - 0.5) * SYNTHETIC_PROJ_SCALE

        seq_lens = torch.randint(1, MAX_SEQ + 1, (batch,), dtype=torch.int32)
        block_table = torch.arange(num_blocks, dtype=torch.int32)
        slot_mapping = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * max_blocks + logical_block
            slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset

        rope_cos = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5
        rope_sin = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5
        k_cache = ((torch.rand(cache_rows, HEAD_DIM, dtype=torch.float32) - 0.5)).to(torch.bfloat16)
        v_cache = (
            SYNTHETIC_PROJ_SCALE * (torch.rand(cache_rows, HEAD_DIM, dtype=torch.float32) - 0.5)
        ).to(torch.bfloat16)
        attn_out = torch.zeros(batch, HIDDEN, dtype=torch.bfloat16)

        return TaskArgsBuilder(
            Tensor("q_proj_norm", q_proj_norm),
            Tensor("k_proj_norm", k_proj_norm),
            Tensor("v_proj", v_proj),
            Tensor("seq_lens", seq_lens),
            Tensor("block_table", block_table),
            Tensor("slot_mapping", slot_mapping),
            Tensor("rope_cos", rope_cos),
            Tensor("rope_sin", rope_sin),
            Tensor("k_cache", k_cache),
            Tensor("v_cache", v_cache),
            Tensor("attn_out", attn_out),
        )

    def compute_golden(self, args, params):
        tensors = {s.name: s.value for s in args.specs if isinstance(s, Tensor)}
        _compute_golden(tensors)
        for s in args.specs:
            if isinstance(s, Tensor) and s.name == "attn_out":
                getattr(args, s.name)[:] = tensors[s.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
