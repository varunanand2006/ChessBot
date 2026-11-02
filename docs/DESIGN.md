# Design & architecture

This document records the architecture of the engine and the reasoning behind
the performance-relevant decisions. The guiding principle throughout: every
choice that affects speed (memory layout, table sizing, branch structure, kernel
configuration) is made deliberately and explained, with the rationale and the
cost of the alternative stated alongside it. Measurable numbers are treated as
first-class deliverables — see [`../FINDINGS.md`](../FINDINGS.md) — and are
always reported as absolute figures (NPS, positions/sec, milliseconds), never as
percentage speedups.

## Goals

Rewrite an existing Python chess engine (~1600 ELO, alpha-beta search,
Texel-tuned evaluation) in C++ for speed, then add a CUDA component that
generates endgame tablebases via retrograde analysis. The CPU engine and the GPU
tablebase generator share a single C++ dialect so that the verified host code
ports to the device largely verbatim.

## Architecture

- **CPU side** — a C++ chess engine: bitboard move generation, alpha-beta
  search, handcrafted (Texel-tuned) evaluation.
- **GPU side** — CUDA-generated endgame tablebases using retrograde analysis: a
  dense sweep over an integer-indexed position space, iterated to convergence. A
  bandwidth-bound frontier BFS expressed as a Jacobi iteration.

Two alternatives were considered and deliberately rejected:

- **NNUE forward pass over PCIe** — dropped for scope.
- **MCTS / AlphaZero-style search on the GPU** — a poor GPU workload: an
  irregular tree with heavy warp divergence and atomic contention on
  backpropagation. Retrograde tablebase generation is the regular,
  data-parallel workload the GPU is good at.

## Language and style

C++20, written in a C-style, data-oriented way. CUDA is C++ natively, so a single
dialect spans host and device code — but device code and hot loops need
predictable, flat memory access, which drives the style rules below.

**Do:**
- Structs of bitboards (`uint64_t`); flat data, no indirection.
- `constexpr` for compile-time table generation.
- `enum class` for piece types, colors, squares.
- Templates only where they eliminate a runtime branch.
- `std::array` with fixed capacity; `-O3 -march=native` in release.

**Avoid:**
- Class hierarchies for pieces or moves — a move is a `uint16_t`.
- Virtual functions anywhere in search or move generation.
- Exceptions in hot paths.
- `std::vector` or heap allocation below the root of search (offline tablebase
  generation is the one place heap containers are allowed).
- Smart pointers in performance-critical structures.

## Toolchain

- **Compiler:** GCC 16.1.0 (MinGW-w64 UCRT, WinLibs).
- **Build:** CMake (≥ 3.20) + Ninja (CMake auto-selects Ninja). Release is the
  default build type.
- **Sanitizers:** this MinGW ships no `libasan`/`libubsan`, so the Debug config
  uses UBSan **trap mode** (`-fsanitize=undefined -fsanitize-trap=undefined`)
  plus `-D_GLIBCXX_ASSERTIONS` and `-fstack-protector-strong`. On a toolchain
  that ships the runtimes this can be switched to full
  `-fsanitize=address,undefined`.

See [`COMMANDS.md`](COMMANDS.md) for the full build / test / benchmark command
reference.

---

## The engine (Phase 1)

A bitboard engine verified by perft to exact node counts on the standard
position set (startpos→6 = 119,060,324; Kiwipete→5 = 193,690,690; positions
3/4/5/6 all exact against the Chess Programming Wiki reference values).
Single-threaded move-generation baseline: **~21 M nodes/sec** (Release,
`-O3 -march=native`, correctness-first make/unmake legality filter).

Key decisions, each with a rationale comment in the code:

- **LERF square mapping** (a1=0, h1=7, a8=56) — the standard bitboard convention,
  so shift-based move generation reads like the textbook.
- **`Position` = struct of 8 bitboards** (`by_type[6]` + `by_color[2]`) + scalars,
  with **no redundant mailbox** — less state to keep in sync across make/unmake
  and the later CUDA kernels. A mailbox would be added only if profiling proved
  piece lookup hot.
- **Magic bitboards, not BMI2 PEXT** — even though the development CPU has PEXT.
  CUDA has no PEXT and the retrograde phase needs sliders in device code;
  multiply-shift magics port to the GPU verbatim. Magics are generated at init by
  seeded search (self-verifying, so there is no copied-constant transcription
  risk).
- **Make/unmake with a caller-owned `StateInfo` stack, not copy-board** — no
  per-node `Position` copy, and the CUDA retrograde phase is built on *unmove*
  generation, so an explicit host `unmake` is the primitive those kernels mirror.
- **Zobrist hashing** (constexpr tables, fixed seed) as a cheap correctness
  invariant: after every make, the incrementally-updated key must equal a
  from-scratch recompute.
- **Legality filter** (make → is-king-attacked → unmake) handles pins and
  en-passant discovered check for free; castling through/into check is the one
  case handled explicitly, via a transit-square attack test.
- **Perft bulk counting** at the last ply (return the move count without
  make/unmake).

## Search and evaluation

A faithful port of the Python engine's search and evaluation, same shape and
heuristics, made fast with data-oriented structures. Throughput: **~3.5 M
nodes/sec** aggregate in full alpha-beta search (single-threaded, TB off).

- **Evaluation** (White-relative): material + piece-square tables + in-check
  penalty + an endgame king-distance term. The blended Texel values in
  `src/eval_tables.inc` are **generated** from the Python constants by
  `python/gen_eval_tables.py` (byte-for-byte, no hand transcription).
- **Search:** iterative-deepening alpha-beta as explicit White(max)/Black(min)
  branches over the White-relative eval; a transposition table with
  exact/lower/upper bounds; quiescence search over captures with side-aware
  stand-pat and delta pruning; MVV-LVA plus TT/prev-best move ordering;
  threefold-repetition detection via a game + search position history.
- **Data-oriented hot path:** the TT is a **fixed-size flat array** (2²² slots,
  direct-mapped, always-replace, with full-key verification) and the repetition
  history is a ply-indexed path array. Both replaced an earlier
  `std::unordered_map` port, cutting per-node hashing and allocation and raising
  throughput from ~1.9 to ~3.5 M nodes/sec. Scores are kept White-relative (not
  negamax-relative) to match the Python original exactly.

---

## Endgame tablebases

### CPU retrograde analysis (Phase 2) — the correctness oracle

Single-threaded CPU retrograde analysis serves as both the correctness oracle
and the performance baseline for the GPU phase.

- **Indexing** (`tb::Index`): a bijective dense `index <-> position` map for
  pawnless material (two kings + a list of extra pieces), using 8-fold D4 board
  symmetry. The canonical representative is the minimum raw code over the 8
  transforms. Gated by a partition test (orbit sizes sum to the enumerated
  legal-position count) plus symmetry invariance, for 3-man and 4-man.
- **Two independent solvers that must agree:** `solve_sweep` is an iterated
  negamax fixpoint (the shape the GPU dense sweep takes); `solve_bfs` is a
  min-priority-queue retrograde over the reverse graph. Both use an int16
  mate-score encoding (+win / −loss / 0 draw). They agree position-by-position:
  KQK is mate-in-10, KRK mate-in-16; the first 4-man table, KQKR, is mate-in-35
  over all 2.47 M positions — matching the known Nalimov maximum.
- **Material-DAG solver** (the 4-man step): a 4-man capture removes exactly one
  piece and lands in a 3-man sub-table. Every sub-material is solved first
  (recursively), and the graph builder probes those solved tables as **fixed
  boundary values** on capture edges instead of treating captures as draws. This
  yields true distance-to-mate across material boundaries, not
  distance-to-conversion.

### Combinatorial indexer (Phase 2.5) — making 5-man feasible

The dense `64ⁿ` index is ~134 MB at 4 men and explodes at 5. An **arithmetic
combinatorial** index replaces it so that 5-man (and identical pieces) become
feasible. Every part is CUDA-portable by design — the device kernels reuse this
exact arithmetic.

- **Combinatorial number system** (`include/combinatorial.hpp`, header-only
  `constexpr`, no chess types): a compile-time Pascal triangle plus
  `rank_combination` / `unrank_combination`.
- **King-anchored king table** (`tb_king_table`): maps a legal (wk, bk) pair to a
  dense canonical id and the D4 transform that reaches it, anchoring symmetry on
  the kings so pieces can be placed combinatorially without enumerating `64ⁿ`.
  462 canonical configs over 3612 legal pairs. A deliberate, documented
  stabilizer caveat causes a mild (~2%) over-count on on-axis king pairs, but two
  distinct positions never collide.
- **CombIndex** (`tb_comb_index`): a mixed-radix index — `king_id`, then per group
  of identical pieces `index = index * C(R, m) + rank(...)`, then `* 2 + stm`.
  Self-bijection and a regression against the dense index are both gated,
  including duplicate-piece materials (e.g. KRRK).
- **Memoryless combinatorial sweep** (`tb::solve_sweep_comb`): the GPU-faithful
  solver shape — store only `value[N]` as int16, regenerate every move each pass
  in place, iterate to a fixpoint. This avoids materializing the whole forward
  graph (billions of edges, tens of GB at 5-man). Validated bit-exact against the
  dense solver (KQKR: all 2,467,122 positions, mate-in-35). The single-threaded
  cost (~10.5 min for KQKR) is exactly why the full 5-man solve runs on the GPU.

### GPU retrograde sweep (Phases 3–5) — validated on an RTX 4090

The GPU sweep is correct, optimized on real hardware, and run at full 5-man
scale. Full numbers are in [`../cuda/PROFILING.md`](../cuda/PROFILING.md); the
phase-by-phase gates are in [`../cuda/ROADMAP.md`](../cuda/ROADMAP.md).

- **Correctness:** the device sweep is bit-exact against the CPU oracle on all
  3,494,568 KQKR comb positions (mate-in-35). Every device stage — index,
  sliders, movegen, comb-index including duplicate pieces — is independently
  gated on the host, so the port is trusted before it ever runs on a GPU.
- **Optimization, measured:** the KQKR sweep kernel was tuned from a naive port
  at 9,872 ms down to 575 ms — a **17.2× speedup**, each step re-gated bit-exact.
  The biggest single win (5.3×) came from replacing a 256-byte per-thread scratch
  array with a single 64-bit bitmask: an occupancy effect that only measurement
  caught, which static reasoning had ranked minor. This is the project's clearest
  illustration of why performance claims are measured, not argued.
- **Full 5-man:** all 28 distinct-piece pawnless 5-man materials (209,674,080
  comb positions each) solved on the GPU and verified 28/28 against the Lichess
  (Gaviota DTM) API — 896/896 sampled positions, zero mismatches. Depth range
  mate-in-5 to mate-in-107 (KBN vs KN, one of the deepest known pawnless 5-man
  mates), matched to Gaviota exactly.

### Tablebase probing in search

At a search node whose material is a generated table, the engine returns the
exact WDL + DTM result as a cutoff instead of the heuristic. Node reductions at
fixed depth: KRK ~10,900×, KQK ~19,900×, KQKR ~18,500×. Probing is off by default
so ordinary play and tests never pay the lazy build cost.

---

## Status

| Area | State |
|---|---|
| Phase 1 — C++ engine core | Complete; perft exact on the full standard set; ~21 M nodes/s |
| Search + evaluation | Complete; ~3.5 M nodes/s; faithful Python port, data-oriented hot path |
| Phase 2 — CPU tablebase (3- & 4-man) | Complete; two independent solvers agree; KQKR mate-in-35 |
| Phase 2.5 — combinatorial 5-man indexer | Complete; validated vs the dense oracle |
| Phases 3–5 — GPU sweep + full 5-man | Complete on an RTX 4090; 28/28 materials verified vs Lichess |
| Tablebase probing in search | Complete (off by default) |

## Roadmap / future work

- **Persist + probe a GPU-generated 5-man table** — the tables are currently
  generated, verified, then freed. A disk format plus a probe loader would wire a
  GPU-generated 5-man table into the engine's search.
- **Identical-piece 5-man** (KRR vs K, KBB vs KN, …) — the CombIndex already
  indexes duplicate pieces; the solver's "which bitboard emptied" capture
  detection needs a count-based variant for duplicate groups.
- **Frontier work-list** — the current sweep recomputes all live nodes every pass
  even after they settle. Re-processing only nodes whose predecessors changed
  (a true retrograde frontier BFS) is the next GPU optimization.
- **True DRAM bandwidth** — re-run the profiler on a counter-enabled host to
  capture the `dram__throughput` the rented pod's locked counters couldn't.
- **Pawn coverage** — pawns reduce symmetry to 2-fold and need promotion/capture
  DAG edges, i.e. a separate indexer variant from the current pawnless 8-fold D4.
- **UCI protocol** — so the engine can run in any chess GUI or play online; the
  current `chess play` is a simple built-in text driver.
