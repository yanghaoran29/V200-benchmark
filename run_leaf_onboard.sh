#!/usr/bin/env bash
# Usage: run_leaf_onboard.sh <leaf_rel> <capture_tag> [device]
set -euo pipefail
REPO=/data/y00955915/Desktop/pypto-lib-09141
V200=$REPO/V200-benchmark
LEAF_REL="${1:?leaf}"
TAG="${2:?tag}"
DEVICE="${3:?device}"
OUT=$V200/granularity/live_captures_v2/$TAG
# Fresh capture dir avoids stale deps/swimlane when SPMD/shape changes.
rm -rf "$OUT"
mkdir -p "$OUT"
cd "$REPO"
# shellcheck disable=SC1091
. .venv/bin/activate
python "$V200/$LEAF_REL/pypto-lib-operator/run_benchmark.py" \
  -p a2a3 -d "$DEVICE" --skip-golden --enable-chip-swimlane 4 --enable-dep-gen \
  --dep-output-dir "$OUT"
bash "$V200/pack_leaf_capture.sh" "$V200/$LEAF_REL" "$OUT"
