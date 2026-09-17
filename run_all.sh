#!/usr/bin/env bash
# Each run is a separate Python process: model-local config modules must not mix.
# Leaves: qwen3_decode_layer/{basic,lt,ht}_* and deepseek_v4_flash_csa/{basic_*,ht_*,lt_*}.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${1:-simpler}"
if [ "$#" -gt 0 ]; then shift; fi
case "$ENGINE" in
  pypto|simpler) ;;
  *) echo "Usage: $0 [pypto|simpler] [-p a2a3|a2a3sim] [-d DEVICE] [capture flags]" >&2; exit 2 ;;
esac
ENTRIES=()
for leaf in \
  qwen3_decode_layer/basic_batch16 \
  qwen3_decode_layer/lt_batch16 \
  qwen3_decode_layer/lt_batch32 \
  qwen3_decode_layer/ht_batch64 \
  qwen3_decode_layer/ht_batch80 \
  qwen3_decode_layer/ht_batch160 \
  deepseek_v4_flash_csa/basic_batch4_mtp1 \
  deepseek_v4_flash_csa/lt_batch4_mtp7 \
  deepseek_v4_flash_csa/lt_batch8_mtp7 \
  deepseek_v4_flash_csa/lt_batch12_mtp7 \
  deepseek_v4_flash_csa/lt_batch16_mtp7 \
  deepseek_v4_flash_csa/ht_batch60_mtp3 \
  deepseek_v4_flash_csa/ht_batch100_mtp3 \
  deepseek_v4_flash_csa/ht_batch180_mtp3
do
  if [ "$ENGINE" = pypto ]; then
    ENTRIES+=("$leaf/pypto-lib-operator/run_benchmark.py")
  elif [[ "$leaf" == qwen3_decode_layer/* ]]; then
    ENTRIES+=("$leaf/simpler-operator/test_qwen3_decode_layer.py")
  else
    ENTRIES+=("$leaf/simpler-operator/test_decode_csa.py")
  fi
done
for entry in "${ENTRIES[@]}"; do
  python "$ROOT/$entry" "$@"
done
