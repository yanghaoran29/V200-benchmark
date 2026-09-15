#!/usr/bin/env bash
# Each run is a separate Python process: model-local config modules must not mix.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${1:-simpler}"
if [ "$#" -gt 0 ]; then shift; fi
case "$ENGINE" in
  pypto) ENTRIES=(deepseek-v4-csa/pypto-lib-operator/run_benchmark.py deepseek-v4-csa-b/pypto-lib-operator/run_benchmark.py qwen3-decode-layer/pypto-lib-operator/run_benchmark.py) ;;
  simpler) ENTRIES=(deepseek-v4-csa/simpler-operator/test_decode_csa.py deepseek-v4-csa-b/simpler-operator/test_decode_csa.py qwen3-decode-layer/simpler-operator/test_qwen3_decode_layer.py) ;;
  *) echo "Usage: $0 [pypto|simpler] [-p a2a3|a2a3sim] [-d DEVICE] [capture flags]" >&2; exit 2 ;;
esac
for entry in "${ENTRIES[@]}"; do
  python "$ROOT/$entry" "$@"
done
