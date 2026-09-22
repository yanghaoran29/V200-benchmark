"""Single-card serving-load case table for V200-benchmark.

All batch / mtp values are per card. DeepSeek: DECODE_SEQ = mtp + 1.
Qwen ignores mtp and only changes public batch (not batch_pad / SPMD width).

Larger public batch is split along the batch axis so per-task work stays
at a fixed tile: flash HT=20, flash LT=4, qwen=16.

Select a case with env ``V200_SERVING_CASE`` or ``--serving-case`` on
run_benchmark.py. Each sample leaf defaults to its directory case name.
"""

from __future__ import annotations

import os
from typing import Optional

# case -> (decode_batch, mtp, decode_seq, T)
DS_CASES: dict[str, tuple[int, int, int, int]] = {
    "HT_BATCH180_MTP3": (180, 3, 4, 720),
    "HT_BATCH100_MTP3": (100, 3, 4, 400),
    "HT_BATCH60_MTP3": (60, 3, 4, 240),
    "LT_BATCH16_MTP7": (16, 7, 8, 128),
    "LT_BATCH12_MTP7": (12, 7, 8, 96),
    "LT_BATCH8_MTP7": (8, 7, 8, 64),
    "LT_BATCH4_MTP7": (4, 7, 8, 32),
    # Mainline-shaped control: B=4 S=2 (mtp=1). Leaf vendors flash_mtp tiling (no N_BATCH_TILES).
    "BASIC_BATCH4_MTP1": (4, 1, 2, 8),
}

QWEN_CASES: dict[str, int] = {
    "BASIC_BATCH16": 16,
    "LT_BATCH16": 16,
    "LT_BATCH32": 32,
    "HT_BATCH64": 64,
    "HT_BATCH80": 80,
    "HT_BATCH160": 160,
}

QWEN_ATTN_SPMD_BASIC = 24
QWEN_ATTN_SPMD_SERVING = 120

# Per-task batch tiles. Serving batch must be an integer multiple.
FLASH_BATCH_TILE_HT = 20
FLASH_BATCH_TILE_LT = 4
QWEN_BATCH_TILE = 16

ENV_NAME = "V200_SERVING_CASE"

# Folder leaf name -> case key
LEAF_TO_CASE: dict[str, str] = {
    "ht_batch180_mtp3": "HT_BATCH180_MTP3",
    "ht_batch100_mtp3": "HT_BATCH100_MTP3",
    "ht_batch60_mtp3": "HT_BATCH60_MTP3",
    "lt_batch16_mtp7": "LT_BATCH16_MTP7",
    "lt_batch12_mtp7": "LT_BATCH12_MTP7",
    "lt_batch8_mtp7": "LT_BATCH8_MTP7",
    "lt_batch4_mtp7": "LT_BATCH4_MTP7",
    "basic_batch4_mtp1": "BASIC_BATCH4_MTP1",
    "basic_batch16": "BASIC_BATCH16",
    "lt_batch16": "LT_BATCH16",
    "lt_batch32": "LT_BATCH32",
    "ht_batch64": "HT_BATCH64",
    "ht_batch80": "HT_BATCH80",
    "ht_batch160": "HT_BATCH160",
}


def normalize_case(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    low = s.lower().replace("-", "_")
    if low in LEAF_TO_CASE:
        return LEAF_TO_CASE[low]
    return s.upper().replace("-", "_")


def serving_case_name(explicit: Optional[str] = None) -> Optional[str]:
    raw = explicit if explicit is not None else os.environ.get(ENV_NAME, "")
    name = normalize_case(raw or "")
    if not name or name in {"BASELINE", "ORIG", "ORIGINAL", "NONE"}:
        return None
    if name not in DS_CASES and name not in QWEN_CASES:
        raise ValueError(
            f"Unknown {ENV_NAME}={name!r}; expected one of "
            f"{sorted(DS_CASES) + sorted(QWEN_CASES)} or unset"
        )
    return name


def apply_cli_case(name: Optional[str]) -> Optional[str]:
    """Set V200_SERVING_CASE before importing config-dependent modules."""
    resolved = serving_case_name(name)
    if resolved is None:
        os.environ.pop(ENV_NAME, None)
        return None
    os.environ[ENV_NAME] = resolved
    return resolved


def is_ht_case(case: Optional[str] = None) -> bool:
    name = serving_case_name(case)
    return bool(name) and name.startswith("HT_")


def flash_batch_tile(case: Optional[str] = None, decode_batch: Optional[int] = None) -> int:
    """Per-task request count for flash: 20 (HT) or 4 (LT/BASIC). Unset keeps one tile."""
    name = serving_case_name(case)
    if name is None:
        return decode_batch if decode_batch is not None else FLASH_BATCH_TILE_HT
    if name not in DS_CASES:
        return decode_batch if decode_batch is not None else FLASH_BATCH_TILE_HT
    return FLASH_BATCH_TILE_HT if name.startswith("HT_") else FLASH_BATCH_TILE_LT


def ds_batch_seq(default_batch: int, default_seq: int, case: Optional[str] = None) -> tuple[int, int]:
    name = serving_case_name(case)
    if name is None or name not in DS_CASES:
        return default_batch, default_seq
    b, _mtp, s, _t = DS_CASES[name]
    tile = flash_batch_tile(name)
    if b % tile != 0:
        raise ValueError(
            f"{name}: DECODE_BATCH={b} is not a multiple of flash batch tile {tile}"
        )
    return b, s


def qwen_batch(default_batch: int, case: Optional[str] = None) -> int:
    name = serving_case_name(case)
    if name is None or name not in QWEN_CASES:
        return default_batch
    b = QWEN_CASES[name]
    if b % QWEN_BATCH_TILE != 0:
        raise ValueError(
            f"{name}: qwen batch={b} is not a multiple of {QWEN_BATCH_TILE}"
        )
    return b


def qwen_attn_spmd_blocks(case: Optional[str] = None) -> int:
    """Attention SPMD width: basic/ht=24; lt=120."""
    name = serving_case_name(case)
    if name is None or name not in QWEN_CASES:
        return QWEN_ATTN_SPMD_BASIC
    if name.startswith("LT_"):
        return QWEN_ATTN_SPMD_SERVING
    return QWEN_ATTN_SPMD_BASIC
