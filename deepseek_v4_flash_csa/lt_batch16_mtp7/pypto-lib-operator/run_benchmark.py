"""Run the packaged operator with its original fixture and golden validation."""

import argparse
import os
import sys
from pathlib import Path

_LEAF_SERVING_CASE = "LT_BATCH16_MTP7"
_V200_ROOT = Path(__file__).resolve().parents[3]
if str(_V200_ROOT) not in sys.path:
    sys.path.insert(0, str(_V200_ROOT))
from serving_load import apply_cli_case


def main(runtime_dir=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim", "a5", "a5sim"], default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--enable-chip-swimlane",
        type=int,
        nargs="?",
        const=4,
        default=0,
        choices=range(5),
    )
    parser.add_argument("--enable-dep-gen", action="store_true")
    parser.add_argument("--dep-output-dir", type=Path)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str)
    parser.add_argument("--start-pos", type=int, default=None)
    parser.add_argument(
        "--serving-case",
        default=os.environ.get("V200_SERVING_CASE") or _LEAF_SERVING_CASE,
        help="serving case; default for this leaf is LT_BATCH16_MTP7",
    )
    parser.add_argument("--skip-golden", action="store_true")
    args = parser.parse_args()
    apply_cli_case(args.serving_case)
    from golden import run, ratio_allclose, ratio_reldiff
    from decode_csa import attention_csa_test, build_tensor_specs, golden_attention_csa

    fn = attention_csa_test
    specs = build_tensor_specs(args.start_pos)
    golden_fn = None if args.skip_golden else golden_attention_csa
    compare = {
        "x_out": ratio_reldiff(diff_thd=4e-3, pct_thd=0.03, max_diff_hd=2),
        "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
    }
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        runtime_dir=str(runtime_dir) if runtime_dir else None,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
            save_kernels=args.dep_output_dir is not None,
            save_kernels_dir=str(args.dep_output_dir.resolve())
            if args.dep_output_dir
            else None,
            # Batch-tile SPMD (HT tile T=80 × N tiles) overflows the 256 MiB
            # default ring heap (C_HT HEAP_RING_DEADLOCK). Match prefill-layer 1 GiB.
            ring_heap=2 * 1024 * 1024 * 1024,
        ),
        compare_fn=compare,
    )
    if not result.passed:
        raise SystemExit(str(result.error or "Golden validation failed"))


if __name__ == "__main__":
    main()
