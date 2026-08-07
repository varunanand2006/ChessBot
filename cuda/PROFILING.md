# Phase 4 — Profiling & optimization (GPU-only)

Phase 4 is measurement-driven, so unlike Phases 0–3 it **cannot be done off the
GPU**. This file is the plan: the metric that matters, how to read it, and the
optimization candidates in priority order — each applied **only after** Nsight
confirms it's the bottleneck, then re-measured. The kernel stays bit-exact vs
`solve_sweep_comb` after every change (`cuda_sweep_check` is the gate that rides
along).

## The headline metric

The sweep is a **bandwidth-bound dense sweep**, so the number is **achieved DRAM
bandwidth (% of peak)**. Three sources, cheapest first (all via `cuda/profile.sh`):

1. **Harness self-report** — `cuda_sweep_check` prints a value-buffer bandwidth
   *lower bound* (5 B/pos/pass: v_old read + state read + v_new write) and % of
   the card's theoretical peak. Free, no counters.
2. **`ncu` targeted metrics** — the true `dram__throughput` % of peak, plus
   occupancy, warp efficiency, registers/thread.
3. **Full `ncu --set full` report** — the whole analysis, opened in the UI.

**Phase 4 gate (from ROADMAP):** a documented achieved bandwidth + a speedup vs
the CPU baseline (**KQKR ~10.5 min single-thread memoryless**; `cuda_sweep_check
KQKR` prints the GPU wall time to compare).

## How to read the profile → what to do

| Nsight signal | Meaning | Action |
|---|---|---|
| `dram__throughput` high (>70% peak) | already bandwidth-bound, near peak | **done** — report it; little left to win |
| `dram__throughput` low + `sm__throughput` low | latency/occupancy bound | raise occupancy (candidates 1–3) |
| `sm__throughput` high, `dram` low | compute-bound (unexpected here) | cut per-thread work (candidates 1, 4) |
| `warps_active` low + high `registers_per_thread` | register pressure caps occupancy | shrink per-thread footprint (candidates 1, 2) |
| `thread_inst..ratio` << 1.0 | warp divergence | live-node work-list (candidate 4) |

## Optimization candidates — priority order (static analysis; confirm with Nsight)

The per-thread footprint of `sweep_update` is the prime suspect for low
occupancy. Rough stack/local sizes per thread:

- `generate_legal` builds **two** `MoveList`s (pseudo + out) = **~1 KB** ← biggest
- `int empty[64]` in `comb_decode` **and** every `comb_encode` call = 256 B each
- `Position` (8 bitboards = 64 B), `MoveList ml` in the update (512 B)

So the checklist's headline item (`empty[64]`) is real but **not** the largest —
measure before assuming. Candidates, each oracle-gated after applying:

1. **Fuse the legality filter into the sweep (drop the second MoveList).** The
   sweep never needs a materialized legal list — it only iterates legal moves
   once to take `max age(-child)`. Generate *pseudo-legal* into one list and, per
   move, inline make→king-safety→(child eval)→unmake. Saves ~512 B/thread and a
   second pass. Host-gateable (result unchanged). *Likely the biggest occupancy
   win.*
2. **`int empty[64]` → `uint64_t occ` bitmask + popcount** (checklist item 1).
   encode compaction: `coord(s) = s - popcount(occ & ((1<<s)-1))`, then
   `occ |= 1<<s`. decode select: the `coord`-th zero bit of `occ` (no PDEP on
   CUDA — popcount binary-search or a bounded 64-bit loop). Removes 256 B/thread
   from decode + each encode. Host-gateable via `test_tb_comb_index_device`.
3. **Read-only tables placement.** Small, uniformly-read: D4 (512 B), shrunk
   binom `[64][8]` (~4 KB), king `id`/`transform` (8 KB) → `__constant__`. Large,
   divergently-read: magic attack tables (~860 KB) exceed the 64 KB constant
   budget → keep in global as `const __restrict__` and load via `__ldg` (read-only
   data cache), which suits our scattered per-square access better than
   `__constant__` broadcast anyway. Measure both.
   - **APPLIED (the value/sub buffers):** `v_old`, `v_new`, `state`, `subs` on
     the kernel and `v_old` + the hoisted `s.value` local in `sweep_update` are
     now `const __restrict__`, so the scattered child gathers
     `v_old[encode(child)]` / `s.value[encode(child)]` route through the
     read-only cache (LDG). This is the zero-risk member of candidate #3: a pure
     aliasing hint (the ping-pong buffers provably never alias within a pass),
     correctness unchanged — the host sweep gate still diffs the fixpoint
     bit-for-bit. Confirm the win on the box via `l2_tex__t_sector_hit_rate`.
     The `__constant__` placement of the small tables is the part still pending
     Nsight.
4. **Live-node work-list (divergence).** Only `SW_SOLVE` nodes do work; the rest
   idle in-warp. If `thread_inst..ratio` is low, compact SOLVE indices into a
   dense work-list (built once on the host, or a stream-compaction each pass) so
   warps do uniform work. Also lets the grid shrink to live nodes.
5. **Coalescing.** `v_new[i]` writes and `v_old[i]`/`state[i]` reads are already
   coalesced (thread i ↔ index i). The **child** reads `v_old[encode(child)]` are
   an irreducible scatter/gather (retrograde analysis) — rely on L2; measure
   `l2_tex__t_sector_hit_rate`. Don't fight the inherent gather.
6. **Buffer minimalism.** `int16` value buffers are already minimal. `state[]`
   (1 B/pos) could be folded into a sentinel value in the buffer after pass 0 —
   minor, only if bandwidth-bound and every byte counts.

## Considered and rejected

- **Zero-copy (mapped) memory for the `changed` flag.** Tempting — it removes the
  per-pass `cudaMemset` + 4-byte `cudaMemcpy`. Rejected because it conflicts with
  the `atomicOr(changed, 1)` convergence write: mapping the flag into host memory
  turns every changing thread's atomic into an atomic *over PCIe to system
  memory*, so on early/busy passes we'd pay thousands-to-millions of PCIe atomic
  transactions per pass — an unbounded cost — to save one tiny bounded copy. The
  synchronous wait is unchanged either way (`cudaDeviceSynchronize()` still
  required before reading the flag), and the flag is entirely off the headline
  `dram__throughput` metric (that's per-kernel device bandwidth). If per-pass host
  overhead ever *did* matter, the right fix is the opposite direction — keep the
  flag in fast device memory and check convergence only every K passes (amortize
  the sync), not push the hot atomic across the bus.

## Workflow on the box

```
cmake -B build && cmake --build build --target cuda_sweep_check
cuda/profile.sh KQKR                 # self-report + ncu metrics + full report
# read dram/occupancy/divergence -> pick the top matching candidate
# apply it, then re-gate + re-measure:
./build/cuda/cuda_sweep_check KQKR   # must still PASS (bit-exact) + note new ms/bandwidth
```

Record before/after **absolute** numbers (GB/s, ms, Mpos/s) — never % speedups —
in the commit message, per the project rule.
