#!/usr/bin/env bash
# Run the four V200 benchmark samples on a2a3 (real device) and/or 1c1vsim (simulator).
#
#   Case1  ->  -p a2a3      (onboard,  block_dim=24,  aicpu_thread_num=4)
#   Case2  ->  -p 1c1vsim   (simulator, block_dim=120, aicpu_thread_num=7)
#
# Real-device (a2a3) runs go through `task-submit` for exclusive NPU locking
# (see simpler/.claude/rules/task-submit-isolation.md); the simulator does not.
#
# Usage:
#   ./run_all.sh                 # both platforms (a2a3 then 1c1vsim)
#   ./run_all.sh a2a3            # real device only
#   ./run_all.sh 1c1vsim         # simulator only
#   SWIMLANE=1 ./run_all.sh a2a3 # add --enable-l2-swimlane (kernel timing)
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMPLER="$(cd "$ROOT/../simpler" 2>/dev/null && pwd || true)"

PLATFORMS=("$@")
if [ "${#PLATFORMS[@]}" -eq 0 ]; then
    PLATFORMS=(a2a3 1c1vsim)
fi

# sample dir : test entry
SAMPLES=(
    "qwen3/qwen3_dynamic_manual_scope:test_qwen3_decode.py"
    "qwen3/qwen3_dynamic_tensormap:test_qwen3_decode.py"
    "paged_attention/paged_attention_unroll:test_paged_attention_unroll.py"
    "paged_attention/paged_attention_unroll_manual_scope:test_paged_attention_unroll.py"
)

EXTRA=""
[ "${SWIMLANE:-0}" = "1" ] && EXTRA="--enable-l2-swimlane"

fail=0

run_onboard() {  # $1=dir  $2=entry  $3=platform
    local dir="$1" entry="$2" plat="$3"
    if command -v task-submit >/dev/null 2>&1; then
        task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
            --run "cd '$ROOT/$dir' && python3 '$entry' -p $plat -d \$TASK_DEVICE $EXTRA"
    else
        echo "[WARN] task-submit not found; running unlocked — results may be noisy if another process shares this NPU"
        ( cd "$ROOT/$dir" && python3 "$entry" -p "$plat" $EXTRA )
    fi
}

run_sim() {  # $1=dir  $2=entry  $3=platform
    ( cd "$ROOT/$1" && python3 "$2" -p "$3" $EXTRA )
}

for plat in "${PLATFORMS[@]}"; do
    # Onboard arch precheck (only for real silicon; sim variants are silicon-agnostic).
    # check.sh exits: 0=match, 1=can't determine (e.g. npu-smi can't init from a
    # bare login shell on this box), 2=confirmed arch mismatch.
    case "$plat" in
        *sim)  : ;;  # a2a3sim / 1c1vsim — skip precheck
        a2a3|a5)
            if [ -n "$SIMPLER" ] && [ -x "$SIMPLER/.claude/skills/onboard-arch-precheck/check.sh" ]; then
                "$SIMPLER/.claude/skills/onboard-arch-precheck/check.sh" "$plat"
                rc=$?
                if [ "$rc" -eq 2 ]; then
                    echo "[SKIP] arch mismatch for -p $plat — refusing to run"; fail=1; continue
                elif [ "$rc" -eq 1 ]; then
                    echo "[WARN] arch precheck could not determine silicon (npu-smi may only work inside the task-submit allocation); proceeding — task-submit owns device access"
                fi
            fi
            ;;
    esac

    for s in "${SAMPLES[@]}"; do
        dir="${s%%:*}"; entry="${s##*:}"
        echo "================= [-p $plat] $dir ================="
        case "$plat" in
            *sim) run_sim "$dir" "$entry" "$plat" ;;
            *)    run_onboard "$dir" "$entry" "$plat" ;;
        esac
        if [ $? -ne 0 ]; then
            echo "FAILED: $dir -p $plat"; fail=1
        fi
    done
done

[ $fail -eq 0 ] && echo "All runs completed." || echo "Some runs failed (see FAILED lines above)."
exit $fail
