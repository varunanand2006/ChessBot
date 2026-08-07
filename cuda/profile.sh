#!/usr/bin/env bash
# Phase 4 profiling — run ON THE GPU BOX after building cuda_sweep_check.
#
# Produces the headline number (achieved DRAM bandwidth) three ways, cheapest
# first: the harness's own estimate, targeted ncu metrics to stdout, and a full
# Nsight Compute report file to open in the UI. See cuda/PROFILING.md for what
# each metric means and the optimization decision loop.
#
# Usage:  cuda/profile.sh [MATERIAL]      (default KQKR)
#   e.g.  cuda/profile.sh KQKR
#
# Requires root/CAP_SYS_ADMIN for ncu hardware counters (ERR_NVGPUCTRPERM
# otherwise — see the GPU rental notes). Confirm with:  ncu --version
set -euo pipefail

BIN=./build/cuda/cuda_sweep_check
MAT="${1:-KQKR}"

if [[ ! -x "$BIN" ]]; then
    echo "build first:  cmake -B build && cmake --build build --target cuda_sweep_check"
    exit 1
fi

echo "=== 1/3  harness self-report (correctness + value-buffer bandwidth) ==="
"$BIN" "$MAT"

echo
echo "=== 2/3  Nsight targeted metrics (one steady-state pass) ==="
# --launch-skip 5 --launch-count 1: skip pass 0 (atypical: classification-heavy
# frontier) and profile a single representative mid-sweep pass, not all ~65.
# The metrics, in priority order for THIS kernel:
#   dram throughput %      — headline: are we bandwidth-bound and how close to peak
#   sm throughput %        — compute intensity (if this >> dram, we're compute-bound)
#   warps_active %         — achieved occupancy
#   thread_inst .ratio     — warp execution efficiency (1.0 = no divergence)
#   registers_per_thread   — register pressure (drives the occupancy ceiling)
ncu --launch-skip 5 --launch-count 1 \
    --metrics \
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
launch__registers_per_thread \
    "$BIN" "$MAT"

echo
echo "=== 3/3  full Nsight report -> sweep_${MAT}_full.ncu-rep ==="
ncu --set full --launch-skip 5 --launch-count 1 -f -o "sweep_${MAT}_full" "$BIN" "$MAT"
echo "Open sweep_${MAT}_full.ncu-rep in the Nsight Compute UI for the full analysis."
