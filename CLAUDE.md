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

---

## Later phases (context — not started)

- **Phase 2:** Bijective position indexing with symmetry reduction (8-fold king
  triangle for pawnless configs, 4-fold with pawns). Single-threaded CPU
  retrograde analysis as correctness oracle and performance baseline.
- **Phase 3:** CUDA retrograde sweep kernels. Device-side move/unmove generation,
  per-pass DTM update, convergence reduction. Profile with Nsight Compute; target
  achieved memory bandwidth as the headline metric.
- **Phase 4:** Verify generated tables against Syzygy or Gaviota. Integrate
  tablebase probing into leaf evaluation; measure node-count reduction.
- **Phase 5 (stretch):** Use the tablebase as a perfect-play oracle to measure how
  often alpha-beta selects an optimal move at fixed node budgets.

## Working agreement

- One step at a time. Report the verification gate result before moving on.
- If a gate fails, fix it before proceeding — do not work around it.
- Ask before deviating from the architecture or adding dependencies.
- Prefer clarity in non-hot code; prefer flat and fast in move generation and search.
- Report NPS, not percentage speedups.
