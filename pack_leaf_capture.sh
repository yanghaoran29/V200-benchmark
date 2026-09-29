#!/usr/bin/env bash
# Package one leaf onboard capture into simpler-operator + root swimlane files.
set -euo pipefail
LEAF="${1:?leaf dir}"
CAP="${2:?capture dir}"
ROOT="$(cd "$(dirname "$LEAF")" && pwd)/$(basename "$LEAF")"
# allow relative
LEAF="$(cd "$LEAF" && pwd)"
CAP="$(cd "$CAP" && pwd)"

SIMPLER="$LEAF/simpler-operator"
mkdir -p "$SIMPLER"
rm -rf "$SIMPLER/kernels" "$SIMPLER/orchestration"
if [ -d "$CAP/kernels" ]; then cp -a "$CAP/kernels" "$SIMPLER/kernels"; fi
if [ -d "$CAP/orchestration" ]; then cp -a "$CAP/orchestration" "$SIMPLER/orchestration"; fi
for f in kernel_config.py compiled_meta.json; do
  [ -f "$CAP/$f" ] && cp -f "$CAP/$f" "$SIMPLER/$f"
done

DFX="$CAP/dfx_outputs"
[ -d "$DFX" ] || DFX="$CAP"
for f in chip_swimlane_records.json deps.json; do
  src="$DFX/$f"
  [ -f "$src" ] || src="$CAP/$f"
  [ -f "$src" ] && cp -f "$src" "$LEAF/$f"
done
# merged_swimlane may be timestamped
if [ -f "$DFX/merged_swimlane.json" ]; then
  cp -f "$DFX/merged_swimlane.json" "$LEAF/merged_swimlane.json"
elif ls "$DFX"/merged_swimlane_*.json >/dev/null 2>&1; then
  cp -f "$(ls "$DFX"/merged_swimlane_*.json | sort | tail -1)" "$LEAF/merged_swimlane.json"
elif [ -f "$CAP/merged_swimlane.json" ]; then
  cp -f "$CAP/merged_swimlane.json" "$LEAF/merged_swimlane.json"
elif ls "$CAP"/merged_swimlane_*.json >/dev/null 2>&1; then
  cp -f "$(ls "$CAP"/merged_swimlane_*.json | sort | tail -1)" "$LEAF/merged_swimlane.json"
fi
# name_map
if [ -f "$DFX/name_map.json" ]; then
  cp -f "$DFX/name_map.json" "$LEAF/name_map.json"
elif ls "$DFX"/name_map*.json >/dev/null 2>&1; then
  cp -f "$(ls "$DFX"/name_map*.json | head -1)" "$LEAF/name_map.json"
elif ls "$CAP"/name_map*.json >/dev/null 2>&1; then
  cp -f "$(ls "$CAP"/name_map*.json | head -1)" "$LEAF/name_map.json"
fi
echo "packaged $LEAF"
