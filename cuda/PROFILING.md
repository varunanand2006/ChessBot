# Phase 4 — Profiling & optimization (GPU-only)

Phase 4 is measurement-driven, so unlike Phases 0–3 it **cannot be done off the
GPU**. This file is the plan: the metric that matters, how to read it, and the
optimization candidates in priority order — each applied **only after** Nsight
confirms it's the bottleneck, then re-measured. The kernel stays bit-exact vs
`solve_sweep_comb` after every change (`cuda_sweep_check` is the gate that rides
along).

## Results — measured on an RTX 4090 (sm_89, CUDA 12.8), 2026-08-07

Solving **KQKR** (all 3,494,568 comb positions, to mate-in-35), 65 Jacobi passes,
each optimization re-gated bit-exact vs `solve_sweep_comb`:

| Kernel (cumulative) | Time | ms/pass | Mpos/s | vs naive |
|---|---:|---:|---:|---:|
| naive port | 9,872 ms | 151.9 | 23.0 | 1.0× |
| + fuse legality filter into the sweep (#1) | 9,416 ms | 144.9 | 24.1 | 1.05× |
| + O(1) small-k device binom (#3, arithmetic) | 3,057 ms | 47.0 | 74.3 | 3.23× |
| + free-square bitmask, no `empty[64]` (#2) | **575 ms** | **8.85** | **394.8** | **17.2×** |

- vs CPU: **≈1,096×** faster than the same memoryless sweep on CPU (~630 s single-
  thread), and **~9.4×** faster than the CPU's materialized solver (~5.4 s) at
  near-zero memory. (CPU is single-thread on a different host — impl-to-impl, not
  same-silicon.) KRK end-to-end: 53.9 → 7.1 ms.
- **Surprise:** candidate #1 (the "biggest occupancy win" by static analysis) gave
  only 1.05×; candidate #2 (`empty[64]`→bitmask, ranked *minor*) gave **5.3×**. The
  256 B/thread stack array's occupancy cost + the O(ne) coord scans dominated.
  Measurement beat the priority order — as intended.

### Phase 5 — the optimized sweep at full 5-man scale — ALL 28 distinct materials (2026-08-07/08)

The same kernel run on the real target: **every distinct-piece pawnless 5-man
material — 28 in all, 209,674,080 comb positions each, on the RTX 4090.** Each was
solved then checked against the Lichess (Gaviota DTM) API. **28/28 materials pass;
896/896 sample positions match; zero mismatches.** Depth mate-in-5 → mate-in-107.

| material | mate-in | passes | solve (s) | Lichess |
|---|---:|---:|---:|:---:|
| KBN vs KN | **107** | 154 | 68.6 | 32/32 |
| KRB vs KQ | 70 | 91 | 60.8 | 32/32 |
| KRN vs KQ | 69 | 97 | 63.9 | 32/32 |
| KQR vs KQ | 67 | 123 | 84.8 | 32/32 |
| KRB vs KR | 65 | 118 | 64.7 | 32/32 |
| KBN vs KQ | 53 | 91 | 59.7 | 32/32 |
| KQN vs KQ | 41 | 71 | 46.1 | 32/32 |
| KQN vs KR | 41 | 55 | 33.1 | 32/32 |
| KRN vs KR | 41 | 66 | 34.5 | 32/32 |
| KBN vs KR | 41 | 26 | 13.4 | 32/32 |
| KQR vs KN | 40 | 19 | 11.7 | 32/32 |
| KQB vs KR | 40 | 41 | 25.2 | 32/32 |
| KRB vs KN | 40 | 53 | 28.9 | 32/32 |
| KBN vs KB | 39 | 36 | 16.9 | 32/32 |
| KRN vs KN | 37 | 61 | 29.9 | 32/32 |
| KQR vs KR | 34 | 37 | 22.6 | 32/32 |
| KQB vs KQ | 33 | 61 | 40.6 | 32/32 |
| KRN vs KB | 31 | 55 | 28.7 | 32/32 |
| KRB vs KB | 30 | 55 | 29.0 | 32/32 |
| KRBN vs K | 29 | 23 | 12.7 | 32/32 |
| KQR vs KB | 29 | 21 | 12.9 | 32/32 |
| KQB vs KN | 21 | 23 | 13.7 | 32/32 |
| KQN vs KN | 21 | 27 | 15.0 | 32/32 |
| KQB vs KB | 17 | 25 | 14.6 | 32/32 |
| KQN vs KB | 17 | 29 | 16.9 | 32/32 |
| KQBN vs K | 7 | 15 | 9.8 | 32/32 |
| KQRB vs K | 5 | 11 | 7.3 | 32/32 |
| KQRN vs K | 5 | 11 | 7.2 | 32/32 |

- **Total GPU solve time 873 s** (~14.5 min) for all 28. Solve time tracks mate
  depth (Jacobi pass count), not table size — every material is the same
  209,674,080 positions. Deepest **KBN vs KN mate-in-107 (213 plies)** is among the
  deepest pawnless 5-man endings known; matched to Gaviota exactly.
- **Validated vs the Lichess (Gaviota DTM) API** — category + signed DTM on a
  30-position spread + each table's two deepest mates. The full memoryless CPU
  oracle (~15–20 h *per material*) is not run; correctness rests on the bit-exact
  ≤4-man device gates (same kernels, larger N) + this external sample oracle.
- Per-material throughput ≈ 315–395 Mpos/s (holds at 60× the KQKR positions). Host
  setup ~96 s/material (classify 209M + three ≤4-man capture sub-tables via the
  dense-solve-then-remap `solve_sub_comb`, not the ~10-min-each memoryless solver).
  Device footprint ≈ 1.07 GB (two 419 MB int16 buffers + 209 MB state).

### nsys breakdown (no HW counters needed — this part works on RunPod)

- `k_sweep_pass` = **100% of GPU time** (577.9 ms / 65 launches, 8.89 ms/pass,
  ±2%). `cudaLaunchKernel` 2.7 ms total, `cudaMemset` 0.11 ms, GPU memcpy 1.8 ms —
  **per-pass host/launch/copy/sync overhead is ~0.5%.** So "check convergence every
  K passes" would save ~nothing; the kernel *is* the cost.
- Uniform 8.89 ms/pass even late in convergence ⇒ every pass recomputes all
  `SW_SOLVE` nodes regardless of whether they've settled. The remaining
  algorithmic lever is a **frontier work-list** (re-process only nodes whose
  predecessors changed) — a redesign, not a tweak.
- Still **compute-bound on movegen**: achieved value-buffer bandwidth ≈ 2 GB/s of
  the 1,008 GB/s peak (0.2%). Big bandwidth headroom remains.

### The one metric we could NOT capture

The true `dram__throughput` % — the headline this phase targeted — needs Nsight
**Compute** counters. RunPod runs pods as non-privileged containers, so `ncu`
returns **`ERR_NVGPUCTRPERM`** (counter access needs host-level CAP_SYS_ADMIN /
the `NVreg_RestrictProfilingToAdminUsers=0` driver flag, unfixable from inside).
nsys tracing works; the counter-based per-warp/occupancy/bandwidth numbers need a
counter-enabled host (bare-metal or a provider that grants it). Everything below
is still the plan for that run.

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
