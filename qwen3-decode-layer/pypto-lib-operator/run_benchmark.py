"""Run the packaged operator with its original fixture and golden validation."""

import argparse
from pathlib import Path


def main(runtime_dir=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim"], default="a2a3")
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-seq", action="store_true")
    args = parser.parse_args()
    from golden import run, ratio_allclose
    import decode_fwd

    # Match the original CLI single-layer trace configuration.
    decode_fwd._CHUNK_NLAYERS = 1
    decode_fwd._FWD_NLAYERS = 1
    from decode_fwd import (
        decode_fwd_layers,
        random_inputs,
        _build_specs,
        golden_decode_layer,
        _backend_type,
    )
    from pypto.backend import set_backend_type

    set_backend_type(_backend_type(args.platform))
    fn = decode_fwd_layers
    specs = _build_specs(random_inputs(seed=args.seed, full_seq=args.max_seq))
    golden_fn = golden_decode_layer
    compare = {"out": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)}
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
        ),
        compare_fn=compare,
    )
    if not result.passed:
        raise SystemExit(str(result.error or "Golden validation failed"))


if __name__ == "__main__":
    main()
