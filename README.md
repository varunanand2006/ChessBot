# Chess Engine — C++ & CUDA

A chess engine that plays at roughly **2000 ELO**, written in data-oriented
C++20, plus a **CUDA** component that generates endgame tablebases on the GPU by
retrograde analysis. It started life as a Python engine (still in `python/`, and
walked through in [`python/README.md`](python/README.md)); this is the C++
rewrite, made much faster and extended onto the GPU.

The headline GPU result: every distinct-piece, pawnless **5-piece endgame** —
all 28 material combinations, 5.9 billion positions — solved to perfect play on
rented GPUs, streamed to S3, and verified against an external tablebase with zero
mismatches. The deepest forced mate found is **mate-in-107**, one of the longest
known in this class of endgame.

---

## Play against it

```bash
cmake -B build && cmake --build build     # one-time build (GCC/MinGW + Ninja)
python web/server.py                       # serves the web UI at :8000
```

Open **http://localhost:8000** and play (you are White). The browser UI has a
click-to-move board, a live evaluation bar you can hide, undo/redo, a move list,
and a strength slider that sets the engine's search depth — all served by one C++
process behind a ~120-line standard-library Python bridge.

<!-- Screenshot: drop an image at docs/web-ui.png and uncomment the line below.
<p align="center"><img src="docs/web-ui.png" alt="Web UI" width="720"></p> -->

To make the engine play **perfect endgames**, point it at a directory of
generated `.tb` tablebase files — it then probes them instead of guessing:

```bash
CHESS_TB_DIR=/path/to/tables python web/server.py
```

Prefer the terminal? `./build/chess.exe play 4` plays a game in the console at
depth 4.

---

## Highlights

**The engine**
- ~2000 ELO: material + Texel-tuned piece-square evaluation, iterative-deepening
  alpha-beta search with a transposition table and quiescence.
- Move generator verified by **perft** to exact node counts on the standard
  position set (startpos to depth 6 = 119,060,324 nodes; Kiwipete to depth 5 =
  193,690,690), matching the published reference values.
- ~21 million nodes/sec in move generation, ~3.5 million nodes/sec in full
  alpha-beta search (single thread, `-O3 -march=native`). Reported as absolute
  numbers throughout, never as percentage speedups.

**The GPU tablebase pipeline**
- All **28 distinct-piece pawnless 5-piece endgames** solved to perfect play,
  209,674,080 positions each, on rented RTX 4090s.
- **Multi-GPU**: a fan-out script gives each GPU a disjoint slice of the work, so
  N GPUs run at ~N× throughput. The 28-table batch finished in under an hour on a
  4-GPU pod.
- **Cloud-native**: infrastructure (S3 bucket, scoped IAM, GPU pod) is defined in
  **Terraform**; each solved table streams straight to **S3**, versioned and
  encrypted.
- **Verified**: every table checked against the Lichess (Gaviota) tablebase API —
  896/896 sampled positions match on exact category and distance-to-mate.
- The generated tables load back into the engine, so the bot plays these endgames
  perfectly. See [`FINDINGS.md`](FINDINGS.md) for the endgame results that fell
  out of the tables (fortress draws, the mate-in-107, and more).

---

## How the engine plays

The board is represented with bitboards (one 64-bit word per piece type and
color) for fast move generation; the interesting decisions are in *search* and
*evaluation*, so those are what this section covers.

**Search — how it picks a move.** The engine models the game with **minimax**:
White maximizes a score, Black minimizes it, exploring move sequences to a fixed
depth and evaluating the leaves. On top of that:

- **Alpha-beta pruning** discards branches that provably can't affect the result,
  which (with good move ordering) roughly doubles the depth reachable in a given
  time budget without changing the answer.
- **Move ordering** (most-valuable-victim / least-valuable-attacker for captures,
  plus the best move from the transposition table) tries strong moves first, so
  alpha-beta prunes as early as possible.
- **Quiescence search** extends past the depth limit through capture sequences
  until the position is quiet, which removes the "horizon effect" where the engine
  stops mid-trade and mis-scores the position.
- A **transposition table** keyed by an incrementally-updated **Zobrist hash**
  remembers positions already searched, so a position reached by a different move
  order isn't re-searched from scratch. The same hashes detect threefold
  repetition.
- **Iterative deepening** searches depth 1, then 2, and so on. Each pass seeds the
  next one's move ordering, so searching to depth *N* this way is faster than
  jumping straight to *N*.

**Evaluation — how it scores a position.** Material count plus **piece-square
tables** that reward good squares (knights in the center, rooks on open files, a
castled king in the middlegame but a central king in the endgame). The table
values were **Texel-tuned**: fit to hundreds of thousands of real positions by
gradient descent rather than set by hand. The full tuning pipeline lives on the
Python side ([`python/README.md`](python/README.md)).

**Endgame tablebases — perfect play, no search.** When the position is one of the
generated tablebases, the engine skips the heuristic entirely and looks up the
exact result — win/draw/loss and distance-to-mate. This both plays perfectly and
prunes hard: in a K+R-vs-K endgame it settles the position in ~170 nodes where the
heuristic search visits ~1.8 million.

---

## Endgame tablebases on the GPU

A tablebase is a solved database of every position in a given material (say, rook
and bishop vs. rook): for each one it stores the game-theoretic result and how
many moves the mate takes under perfect play. They're built by **retrograde
analysis** — start from checkmates and work backwards, repeatedly, until every
position's value stops changing.

That backwards sweep is a great GPU workload: the same simple update applied to
hundreds of millions of positions in parallel, iterated to convergence. This
project builds it up in verified stages.

**Indexing.** Every position maps to a dense integer and back. A naive scheme
would need `64⁵` slots and explode at 5 pieces, so the index uses the
**combinatorial number system** plus an 8-fold board-symmetry table — squeezing
each 5-piece material down to ~210 million indices.

**Solving.** One GPU thread per position runs the retrograde update; the whole
table is swept over and over (a Jacobi iteration) until it converges. A CPU
solver computes the same tables independently and the two are diffed
position-by-position, so the GPU results are trusted before they're ever run at
full scale.

**Optimization, measured on real hardware.** On an RTX 4090 the sweep kernel was
tuned from a naive port at 9,872 ms down to 575 ms on the 4-piece benchmark — a
**17.2× speedup**, each step re-checked for bit-exact correctness. The biggest
single win (5.3×) came from replacing a 256-byte per-thread scratch array with a
single 64-bit bitmask: an occupancy effect that only measurement caught, not
static reasoning. Full numbers and the profiling notes are in
[`cuda/PROFILING.md`](cuda/PROFILING.md).

**Scaling out.** The multi-GPU fan-out and the Terraform-provisioned S3 pipeline
(above) take it from one benchmark material to all 28, generated in the cloud and
stored durably. The infrastructure and its design trade-offs are documented in
[`cloud/README.md`](cloud/README.md).

| | |
|---|---|
| Positions indexed / solved | 5.87 billion indexed, 4.33 billion legal |
| GPU throughput | ~343 million positions/sec at 5-piece scale |
| GPU vs. CPU | ~1,096× faster than the same sweep single-threaded on CPU |
| Deepest mate | mate-in-107 (KBN vs. KN), matched to Gaviota exactly |
| Verification | 28/28 materials, 896/896 sampled positions match Lichess |

---

## Build, test, benchmark

Toolchain: GCC (MinGW-w64 UCRT) + CMake ≥ 3.20 + Ninja. Release is the default
(`-O3 -march=native`). The CUDA targets build only when a CUDA compiler is
present, so the engine builds and all host-side tests run without a GPU.

```bash
cmake -B build
cmake --build build

# Fast test suite (drop -E for everything, including the deep perft)
ctest --test-dir build -E perft_deep

# Benchmarks: move-gen NPS, search NPS, tablebase generation
./build/benchmarks/bench.exe 3
./build/benchmarks/search_bench.exe
./build/benchmarks/tb_bench.exe

# Try the engine directly
./build/chess.exe search 6                 # best move for the start position
./build/chess.exe tb KQKR                  # inspect a generated tablebase
./build/chess.exe play 4                   # play a game in the terminal
```

The correctness story is that the GPU code is a `__host__ __device__` port of the
verified CPU code: every device stage is diffed against the CPU oracle on the host
(no GPU needed) before it's trusted on hardware. `CLAUDE.md` has the exhaustive
command list; [`cuda/ROADMAP.md`](cuda/ROADMAP.md) has the phase-by-phase gates.

## Repo layout

```
include/     C++ engine headers (board, movegen, search, eval, tablebase, indexing)
src/         engine sources + the CLI (chess.exe)
tests/       perft + unit + host-side GPU-port gates (CTest, dependency-free)
benchmarks/  timing harnesses (move-gen NPS, search NPS, tablebase generation)
cuda/        CUDA kernels and gates + ROADMAP / PROFILING notes
cloud/       Terraform + Docker + fan-out for multi-GPU generation to S3
web/         browser UI (index.html) + a stdlib Python bridge to the engine
python/      the original Python engine — preserved as the project's origin
```

## Further reading

- [`FINDINGS.md`](FINDINGS.md) — performance numbers and endgame results from the tablebases
- [`cuda/PROFILING.md`](cuda/PROFILING.md) — GPU optimization log and profiling
- [`cloud/README.md`](cloud/README.md) — the multi-GPU + S3 generation pipeline
- [`python/README.md`](python/README.md) — the original Python engine, feature by feature
- `CLAUDE.md` — full design rationale and the complete command reference
