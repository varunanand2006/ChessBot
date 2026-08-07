# Chess Engine — C++/CUDA Rewrite

## Project goal

Rewrite an existing Python chess engine (~1600 ELO, alpha-beta search,
Texel-tuned evaluation) in C++, then add a CUDA component that generates
endgame tablebases via retrograde analysis.

This is a portfolio project targeting GPU/systems roles. Two consequences shape
every decision:

1. **Measurable performance numbers matter more than features.** Instrumentation
   and benchmarking are first-class deliverables, not afterthoughts.
2. **Every performance-relevant choice must be explainable.** When a decision
   affects speed (memory layout, table sizing, branch structure, kernel config),
   leave a comment stating the rationale and what the alternative would cost. Do
   not silently apply an optimization.

## Architecture (settled — do not redesign)

- **CPU side:** C++ chess engine — bitboard move generation, alpha-beta search,
  handcrafted evaluation.
- **GPU side:** CUDA-generated endgame tablebases using retrograde analysis.
  Dense sweep over an integer-indexed position space, iterated to convergence.
  A bandwidth-bound frontier BFS.
- **Explicitly rejected:** NNUE forward pass over PCIe (dropped for scope);
  MCTS / AlphaZero-style search (bad GPU workload — irregular tree, warp
  divergence, atomic contention on backpropagation).

## Language and style

C++20, written in a C-style, data-oriented way.

**Do:**
- Structs of bitboards (`uint64_t`). Flat data, no indirection.
- `constexpr` for compile-time table generation.
- `enum class` for piece types, colors, squares.
- Templates only where they eliminate runtime branches.
- `std::array` with fixed capacity. `-O3 -march=native` in release.

**Do NOT:**
- Class hierarchies for pieces or moves. A move is a `uint16_t`.
- Virtual functions anywhere in search or move generation.
- Exceptions in hot paths.
- `std::vector` or any heap allocation below the root of search.
- Smart pointers in performance-critical structures.

Rationale: CUDA is C++ natively, so a single dialect across host and device
code — but device code and hot loops need predictable, flat memory access.

## Repo layout

```
/            CMakeLists.txt, CLAUDE.md, README, .gitignore
/include     C++ engine headers
/src         C++ engine sources
/tests       perft and unit tests (CTest)
/benchmarks  timing harnesses (NPS)
/python      legacy Python engine (preserved, NOT modified — the project's origin)
/cuda        CUDA kernels (later phase — does not exist yet)
```

The Python engine stays as the origin point. Do not modify it. The C++ side
mirrors its architecture (per-piece pseudo-legal generation + legality filter,
Zobrist hashing, integer move encoding) but bitboard-idiomatic and flat.

## Toolchain

- **Compiler:** GCC 16.1.0 (MinGW-w64 UCRT, WinLibs). No Clang/MSVC/`make`.
- **Build:** CMake (≥3.20) + Ninja (CMake auto-selects Ninja).
- **Sanitizers:** this MinGW ships no `libasan`/`libubsan`, so the Debug config
  uses UBSan **trap mode** (`-fsanitize=undefined -fsanitize-trap=undefined`) +
  `-D_GLIBCXX_ASSERTIONS` + `-fstack-protector-strong` instead of full ASan/UBSan.
  Switch to real `-fsanitize=address,undefined` only on a toolchain that ships
  those runtimes.

## Build / test / benchmark

```bash
# Configure + build (Release is the default build type)
cmake -B build
cmake --build build

# Run the fast test suite (excludes the ~28s deep perft)
ctest --test-dir build -E perft_deep

# Run everything including deep perft
ctest --test-dir build

# Debug build (UBSan trap + assertions)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

# Perft CLI / divide debugging tool
./build/chess.exe perft  6
./build/chess.exe divide 5 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# NPS benchmark (best-of-N; N defaults to 1)
./build/benchmarks/bench.exe 3

# Tablebase: inspect a 3-man endgame (WDL + deepest-mate FEN)
./build/chess.exe tb R          # KRK; also Q, B, N

# Tablebase generation baseline (positions/sec, WDL/DTM stats)
./build/benchmarks/tb_bench.exe

# Search throughput benchmark (aggregate NPS across a position set)
./build/benchmarks/search_bench.exe

# Search: best move for a position (prints per-depth line + nodes/NPS)
./build/chess.exe search 6                     # startpos, depth 6
./build/chess.exe search 5 "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"

# Play a game vs the engine (you are White). Optional depth + FEN.
./build/chess.exe play 4

# Tablebase probing: node-count reduction, heuristic vs TB-probing search
./build/benchmarks/tb_probe_bench.exe

# Dump sample TB positions as CSV (fen,category,signed_dtm) for verification
./build/chess.exe tbdump KQKR 25

# Verify generated tables vs the Lichess tablebase API (needs network + curl)
python python/verify_tablebase.py 25 KQK KRK KQKR

# 5-man combinatorial indexer tests (fast tiers auto-run in ctest)
./build/tests/test_combinatorial.exe        # rank/unrank number system
./build/tests/test_tb_king_table.exe        # 462 canonical king-pair table
./build/tests/test_tb_comb_index.exe        # comb encode/decode (fast 3-man)
./build/tests/test_tb_comb_index.exe slow   # +4-man, duplicate pieces, 5-man
./build/tests/test_tb_comb_solve.exe        # memoryless sweep == dense (3-man)
./build/tests/test_tb_comb_solve.exe slow   # MANUAL: 4-man KQKR, ~10 min

# Regenerate the eval tables from the Python Texel constants (one-off)
python python/gen_eval_tables.py > src/eval_tables.inc
```

---

## Current status

**Phase 1 (C++ engine core) is COMPLETE.** All nine steps passed their gates.

| Step | Deliverable | Gate | Status |
|---|---|---|---|
| 1 | Build scaffolding | cmake builds + runs | ✅ |
| 2 | Board rep + FEN | byte-exact FEN round-trips | ✅ |
| 3 | Attack tables (constexpr) | knight/king counts | ✅ |
| 4 | Magic bitboards | magic == ray-walk reference, 1.28M sets | ✅ |
| 5 | Move encoding (`uint16_t`) | all move types round-trip | ✅ |
| 6 | Make/unmake + Zobrist | byte-identical + identical key | ✅ |
| 7 | Legal move generation | produces moves | ✅ |
| 8 | **Perft (phase gate)** | exact vs. wiki: startpos→6, Kiwipete→5, pos 3/4/5 | ✅ zero mismatches |
| 9 | NPS benchmark | absolute NPS reported | ✅ |

- **Perft:** exact match on the full standard set (startpos→6 = 119,060,324;
  Kiwipete→5 = 193,690,690; positions 3/4/5/6 all exact). Reference values from
  the Chess Programming Wiki (https://www.chessprogramming.org/Perft_Results),
  transcribed with citation — not recalled.
- **NPS baseline: ~21 Mnps aggregate** (best-of-3, Release `-O3 -march=native`,
  single-threaded, correctness-first make/unmake legality filter — no pin-aware
  or staged movegen yet). Reported as absolute NPS, never % speedup. This is the
  number future optimizations are measured against.

### Key design decisions (all with rationale comments in code)

- **LERF square mapping** (a1=0, h1=7, a8=56) — standard bitboard convention so
  shift-based movegen reads textbook.
- **`Position` = struct of 8 bitboards** (`by_type[6]` + `by_color[2]`) + scalars,
  **no redundant mailbox** — less state to keep in sync across make/unmake and
  the later CUDA kernels. Add a mailbox only if profiling proves piece lookup hot.
- **Magic bitboards, not BMI2 PEXT** — even though this CPU has PEXT. CUDA has no
  PEXT and the retrograde phase needs sliders in device code; multiply-shift
  magics port to the GPU verbatim. Magics are generated at init by seeded search
  (self-verifying; no copied-constant transcription risk).
- **Make/unmake with a caller-owned `StateInfo` stack, not copy-board** — no
  per-node `Position` copy, and the CUDA retrograde phase is built on *unmove*
  generation, so an explicit host `unmake` is the primitive those kernels mirror.
- **Zobrist introduced now** (constexpr tables, fixed seed) as a cheap correctness
  invariant: after every make, incremental key == from-scratch recompute.
- **Legality filter (make → is-king-attacked → unmake)** handles pins and
  en-passant discovered check for free; castling through/into check is the one
  case handled explicitly (transit-square attack test).
- **Perft bulk counting** at the last ply (return move count without make/unmake).

### Phase 2 (CPU tablebase) status — 3-man and 4-man COMPLETE

Single-threaded CPU retrograde analysis: the correctness oracle and performance
baseline for the CUDA phase. Files: `tb_index.{hpp,cpp}`, `tb_solve.{hpp,cpp}`,
`benchmarks/tb_bench.cpp`, tests `test_tb_index`/`test_tb_solve`.

- **Indexing** (`tb::Index`): bijective dense `index <-> position` for pawnless
  material (two kings + a list of `Piece{color,type}` extras), 8-fold D4
  symmetry. Canonical = min raw code over the 8 transforms (the unimpeachable
  reference). Gate = partition test (orbit sizes sum to the enumerated
  legal-position count) + symmetry invariance — passes for 3-man and 4-man
  (KQKR = 2.47M indices, KRKN, KRBK). `test_tb_index slow` runs the 4-man tier.
- **Solver** (`solve_sweep`, `solve_bfs`): two independent DTM solvers that must
  agree. `solve_sweep` = iterated negamax fixpoint (the GPU dense-sweep shape);
  `solve_bfs` = min-priority-queue retrograde over the reverse graph. mate-score
  int16 encoding (+win / −loss / 0 draw). Handles **3-man AND 4-man** via a
  material-DAG (below). 3-man theory: KQK mate-in-10, KRK mate-in-16; KBK/KNK all
  draws. First 4-man: **KQKR = mate-in-35** (69 plies, both solvers agree over
  all 2.47M positions) — matches the known Nalimov KQvKR maximum; KRKN mostly
  drawn (2.09M/2.92M) as theory predicts.
- **Material-DAG solver** (the 4-man step): a 4-man capture removes exactly one
  extra (kings are never captured) and lands in a *3-man sub-table* — KQKR:
  `QxR`→KQK, `RxQ`→a KRK that keeps Black's rook (the indexer is color-general,
  so the sub-material is built with its real colors; no color-flip bookkeeping).
  `solve_material` solves every sub-material first (recursively, same solver) and
  the graph builder probes them as **fixed boundary values** on capture edges
  instead of calling every capture a draw. 3-man is the base case (its only
  capture exits to bare KK = draw), so old 3-man numbers are reproduced exactly.
- **Baseline:** 3-man ~0.6M positions/sec; KQKR (4-man, incl. its 3-man
  sub-tables) ~0.46M positions/sec, ~15M position-updates/sec, 33 passes,
  single-threaded Release. `chess tb <material>` (e.g. `KQKR`, `KRKN`) prints
  WDL + a deepest-mate FEN; `test_tb_solve slow` / ctest `tb_solve_4man` is the
  4-man gate.

Key Phase 2 decisions: iterated forward sweep (not explicit unmove generation)
to match the GPU "dense sweep to convergence" shape and reuse Phase 1 movegen;
table-based canonicalization (not closed-form king-triangle) for correctness,
reusable by the GPU as device index tables; DAG recursion probes solved
sub-tables as boundary values (true DTM across material boundaries, not DTC);
BFS uses a min-priority-queue (not the old 3-man FIFO) because capture-exits
inject win/loss results at arbitrary DTM, out of natural discovery order;
`std::vector` allowed here (offline gen, not the search hot path).

### Phase 2.5 (5-man combinatorial indexer) — INDEX COMPLETE, full solve deferred to GPU

The dense `64^men` `tb::Index` is ~134 MB at 4 men and explodes at 5. Phase 2.5
replaces it with an *arithmetic combinatorial* index so 5-man (and identical
pieces) become feasible. Built and gated one step at a time; the parts are
CUDA-portable by design (the device kernels will reuse this exact arithmetic).

- **Combinatorial number system** (`include/combinatorial.hpp`, header-only
  constexpr, NO chess types): `combo::binom` (compile-time Pascal triangle, n≤64;
  largest C(64,32)=1.83e18 fits uint64) + `rank_combination`/`unrank_combination`
  (pointer-based). Gate `test_combinatorial`: Pascal/symmetry/anchors, round-trip
  both ways, full bijection by independent subset enumeration.
- **King-anchored king table** (`tb_king_table.{hpp,cpp}`): legal (wk,bk)→dense
  canonical id + the D4 transform reaching canonical. Anchors symmetry on the
  kings (rotate whole position into that frame) instead of the whole-tuple
  min-code, so pieces can be placed combinatorially without enumerating 64^men.
  **462 canonical configs / 3612 legal pairs.** Stabilizer caveat (deliberate,
  commented): on-axis king pairs pick one transform → a piece placement and its
  residual-symmetry mirror get DIFFERENT indices (mild over-count, ~2%), but two
  distinct positions NEVER collide. Gate `test_tb_king_table`: its symmetry
  partition of the 3612 pairs is identical (bijective) to `tb::Index` on bare
  kings — king-anchored folding == the already-verified whole-tuple folding.
- **CombIndex** (`tb_comb_index.{hpp,cpp}`): mixed-radix `index = king_id`, then
  per group `index = index*C(R_G,|G|) + rank(group squares among empties)`, then
  `*2 + stm`. Groups = identical (color,type) pieces (unordered subset → radix
  C(R,m)); radices are per-material constants so decode peels with div/mod. Gate
  `test_tb_comb_index` (fast 3-man / `slow` 4-man+dup+5-man): **Gate A** full
  self-bijection `encode(decode(i))==i` (KRK/KQK/KQKR/**KRRK** duplicate; sampled
  for 5-man **KQRKR** size 209,674,080); **Gate B** regression vs dense
  `tb::Index` — every legal dense class encodes to a comb index whose decode is
  symmetry-equivalent, and no two distinct legal orbits collide. comb is looser
  than dense (KQKR comb 3.49M vs dense 2.47M): includes illegal-for-other-reasons
  placements + the on-axis over-count (deliberate "overcount, filter at solve").
- **Memoryless combinatorial sweep** (`tb_solve_comb.cpp`,
  `tb::solve_sweep_comb`): the GPU-faithful solver shape — stores only value[N]
  int16, **regenerates every move each pass** in place, iterates to fixpoint
  (vs. the dense solver materializing the whole forward graph = billions of edges
  / tens of GB at 5-man). Distinct pieces only (capture-exit detection = which
  (color,type) bitboard emptied); capture boundaries from dense sub-tables. Gate
  `test_tb_comb_solve` (fast 3-man auto; **4-man KQKR is MANUAL** —
  `./build/tests/test_tb_comb_solve.exe slow`, ~10 min): the comb sweep value ==
  the dense `solve_sweep` value for EVERY legal position — KRK (50,015), KQK
  (46,137), KQKR (all 2,467,122, maxWinDTM=69 = mate-in-35, 43 passes).
- **Baseline (single-thread memoryless): KQKR ~10.5 min wall** (3.49M positions,
  43 passes) — ~100× the materialized dense solver because moves are regenerated
  every pass. This is the CPU baseline the CUDA sweep must beat. A full 5-man is
  ~15–20 h single-threaded → **DECISION: the actual full 5-man table is generated
  on the GPU in Phase 3** (its stated purpose), not ground out on CPU here. The
  index + solver path is validated and ready to feed the GPU port.

### CUDA Phase 3 + 4 — DONE, validated on a real RTX 4090 (2026-08-07)

The GPU retrograde sweep is built, correct, and optimized on real hardware.
Full numbers: `cuda/PROFILING.md`; status matrix: `cuda/ROADMAP.md`.

- **Correctness:** `cuda_sweep_check KQKR` is **bit-exact vs `solve_sweep_comb` on
  all 3,494,568 comb positions, mate-in-35.** Every device stage independently
  gated (index, sliders, movegen, comb-index incl. KRRK duplicates).
- **Performance:** KQKR solved in **575 ms** (RTX 4090, sm_89), a **17.2× kernel
  speedup** over the naive port (9,872 ms) and **≈1,096× vs the CPU memoryless
  baseline (~630 s)**. Optimizations, each re-gated bit-exact: fuse legality filter
  (1.05×), O(1) small-k device binom (→3.23×), `int empty[64]`→`uint64_t` bitmask
  (→17.2×; the 5.3× surprise — occupancy, opposite of the static-analysis order).
- **Profiling:** nsys shows the kernel is 100% of GPU time (overhead ~0.5%) and
  still **compute-bound on movegen** (value BW ~2 GB/s of 1,008 peak). The true
  `dram__throughput` % was NOT captured — RunPod's non-privileged containers lock
  Nsight Compute counters (`ERR_NVGPUCTRPERM`); needs a counter-enabled host.
- **On-box bugs fixed** (invisible to host gates, only real nvcc found them):
  4× `CH_HD` functions ODR-using host tables (D4, binom, attack tables,
  CASTLE_MASK) + a slider-init-ordering segfault in `sweep_check.cu`. All on master.

### >>> PICK UP HERE <<<

Phase 3+4 are done on the GPU. Remaining candidate steps:
- **Phase 5 — scale to a real 5-man** (KQRKR etc.) on the GPU: needs count-based
  capture detection for duplicate-piece groups in the solver, a host-RAM check
  (int16 table ~210–420 MB; scale the movegen_check sampling stride), then verify
  one 5-man vs the Lichess API.
- **Frontier work-list** (the next real GPU optimization): every pass currently
  recomputes ALL live `SW_SOLVE` nodes even after they settle (uniform 8.89 ms/pass
  to the end). Re-process only nodes whose predecessors changed last pass — a
  retrograde frontier BFS (like the CPU `solve_bfs`), a redesign not a tweak.
- **True DRAM bandwidth:** re-run `cuda/profile.sh` on a counter-enabled host
  (bare-metal / a provider that grants GPU perf counters) to capture the
  `dram__throughput` % the RunPod pod couldn't.
- **(Optional, if a CPU 5-man table is wanted before the GPU)** parallelize the
  sweep (double-buffered Jacobi across cores) — a real CPU baseline artifact —
  then verify one 5-man vs the Lichess API. Also needs: skip the unused
  compute_zobrist in the sweep decode; count-based capture detection for
  duplicate-piece 5-man (KRRKN etc.); RAM check (int16 table ~210–420 MB).
- **Identical-piece / pawn coverage:** CombIndex already indexes duplicates
  (KRRK gate passes); the SOLVER still needs count-based capture detection for
  them. Pawns reduce symmetry to 2-fold (left-right mirror only) + need
  promotion/capture DAG edges — a separate indexer variant.

---

## Later phases (context — not started)

- **Phase 3+4: DONE on a real RTX 4090** (see the CUDA status section above).
  Device-side movegen + per-pass DTM sweep + convergence reduction, bit-exact on
  KQKR (mate-in-35), 17.2× kernel speedup, ≈1,096× vs the CPU baseline. `/cuda`
  exists with the kernels + gates. Profiled with nsys (compute-bound); the true
  Nsight-Compute DRAM-bandwidth % awaits a counter-enabled host.
- **Phase 4: DONE.** Tablebase probing in the search (below) + external DTM
  verification: our tables match the Lichess tablebase API (Gaviota DTM source)
  on 133/133 sampled positions across KQK/KRK/KBK/KQKR/KRKN — exact category and
  signed distance-to-mate, including the deepest mates. See `chess tbdump` +
  `python/verify_tablebase.py`.
- **Phase 5 (stretch):** Use the tablebase as a perfect-play oracle to measure how
  often alpha-beta selects an optimal move at fixed node budgets. Also needs the
  search.

### Engine (search + evaluation) — PORTED from Python (2026-08-06)

A faithful port of `python/search.py` + `python/constants.py`, same shape and
heuristics (`eval.{hpp,cpp}`, `search.{hpp,cpp}`, tests `test_eval`/`test_search`,
CLI `chess search` / `chess play`):

- **Evaluation** (`eval::evaluate`, White-relative): material + piece-square
  tables + in-check penalty + endgame king-distance term. The blended Texel
  values live in `src/eval_tables.inc`, **generated** from the Python constants by
  `python/gen_eval_tables.py` (byte-for-byte, no hand transcription; re-run to
  regenerate). One deliberate deviation from the port: total material is summed
  in full before the middlegame/endgame king-table pick, fixing the Python
  single-pass order dependence. Gate = start position == 0 + color-swap/mirror
  antisymmetry + material dominance.
- **Search** (`search::find_best_move`): iterative-deepening alpha-beta as
  explicit White(max)/Black(min) branches over the White-relative eval;
  transposition table (exact/lower/upper bounds); quiescence over captures with
  side-aware stand-pat + delta pruning; MVV-LVA + TT/prev-best move ordering;
  threefold-repetition via a game+search position history. TT + history are
  game-global, reset by `search::new_game()`. Gate = legal best move + mate-in-1
  both colors + wins a hanging queen.
- **Throughput: ~3.5 Mnps aggregate** (best-of-1, Release, single-threaded, TB
  off), up from ~1.9 Mnps after moving the two hot-path structures off
  `std::unordered_map` (below). `benchmarks/search_bench.exe` is the harness.
  Reported as absolute NPS, never % speedup.
- **Design notes:** the TT is a **fixed-size flat array** (2^22 slots, direct-
  mapped, always-replace, full-key verification) and repetition history is the
  **game line + a ply-indexed path array** — both replaced the original
  `std::unordered_map` port to cut per-node hashing/allocation (the data-oriented
  style the rest of the engine follows). Scores stay White-relative (not
  negamax-relative) to match the Python original exactly.

### Tablebase probing in search — DONE (Phase 4 hookup, 2026-08-06)

`tb_probe.{hpp,cpp}` + `search::set_use_tablebase(bool)` (default OFF).
`benchmarks/tb_probe_bench.cpp`, test `test_tb_probe`.

- At a search node whose material is a generated table (pawnless, ≤4 men,
  distinct pieces), `tb::probe` returns exact WDL + DTM (side-to-move relative);
  the search converts it to its White-relative mate scale and returns it as an
  exact cutoff. Unsupported materials (pawns, 5+ men, identical pieces, bare KK→
  draw) fall back to the heuristic. Tables are built lazily and cached by
  material on first probe (a 4-man build is ~seconds; every probe after is O(1)).
- **Headline result — node reduction (fixed depth, heuristic vs probing):**
  KRK d12 1.83M→168 (~10,900×); KQK d12 5.0M→252 (~19,900×); KQKR d10 554k→30
  (~18,500×). The probing search also returns exact mate scores instead of fuzzy
  material scores. Probing is OFF by default so ordinary play/tests never pay the
  build cost.
- **Robustness:** `chess search`/`play` now reject illegal input positions (side
  to move able to capture the enemy king) — otherwise the mover legally "captures"
  the exposed king and eval hits a missing-king out-of-bounds. `position_legal`
  in `main.cpp`.

### External DTM verification — DONE (2026-08-06)

`chess tbdump <material> [N]` emits sample positions as CSV (fen, category,
signed DTM in plies: + if side-to-move mates, − if being mated, 0 draw — the
Lichess/Gaviota convention). `python/verify_tablebase.py` runs the dump, queries
the Lichess tablebase API (Gaviota DTM for ≤7 men) per FEN via curl, and compares
category + signed DTM. Result: **133/133 match** across KQK, KRK, KBK, KQKR,
KRKN (incl. deepest mates). Manual/network — NOT in the offline ctest suite. The
script shells out to curl because Python's SSL trust store is misconfigured in
this environment.

### Not-yet-built engine work

- **UCI protocol** — so the engine can run in any chess GUI / play on Lichess.
  Not started; `chess play` is a simple built-in text driver for now.
- **Search optimizations** — flat-array TT + array repetition history are DONE
  (~1.9→~3.5 Mnps). Remaining if NPS stays a focus: staged move generation,
  killer/history heuristics, aspiration windows, depth-preferred TT replacement.
  A latent accuracy item: quiescence stand-pats even when in check (faithful to
  the Python original) — searching evasions there would improve accuracy.

## Working agreement

- One step at a time. Report the verification gate result before moving on.
- If a gate fails, fix it before proceeding — do not work around it.
- Ask before deviating from the architecture or adding dependencies.
- Prefer clarity in non-hot code; prefer flat and fast in move generation and search.
- Report NPS, not percentage speedups.
