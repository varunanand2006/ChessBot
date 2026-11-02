# Command reference

The full set of build, test, benchmark, and CLI commands. The quick-start subset
is in the top-level [`../README.md`](../README.md); this is the exhaustive list.

## Build

```bash
# Configure + build (Release is the default build type: -O3 -march=native)
cmake -B build
cmake --build build

# Debug build (UBSan trap mode + assertions — see docs/DESIGN.md, Toolchain)
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

The CUDA targets build only when a CUDA compiler is present, so the engine and
all host-side tests build and run without a GPU.

## Test

```bash
# Fast test suite (excludes the ~28s deep perft)
ctest --test-dir build -E perft_deep

# Everything, including deep perft
ctest --test-dir build
```

## Benchmarks

```bash
./build/benchmarks/bench.exe 3          # move-generation NPS (best-of-N; N defaults to 1)
./build/benchmarks/search_bench.exe     # search throughput (aggregate NPS over a position set)
./build/benchmarks/tb_bench.exe         # tablebase generation baseline (positions/sec, WDL/DTM stats)
./build/benchmarks/tb_probe_bench.exe   # TB-probing node-count reduction (heuristic vs TB-probing search)
```

## Engine CLI (`chess.exe`)

```bash
# Perft / divide debugging
./build/chess.exe perft  6
./build/chess.exe divide 5 "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Search: best move for a position (prints per-depth line + nodes/NPS)
./build/chess.exe search 6                              # startpos, depth 6
./build/chess.exe search 5 "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1"

# Play a game vs the engine (you are White). Optional depth + FEN.
./build/chess.exe play 4

# Tablebase: inspect a 3-man endgame (WDL + deepest-mate FEN)
./build/chess.exe tb R          # KRK; also Q, B, N
./build/chess.exe tb KQKR       # 4-man; also KRKN, etc.

# Dump sample TB positions as CSV (fen,category,signed_dtm) for verification
./build/chess.exe tbdump KQKR 25
```

## Tablebase tests

```bash
./build/tests/test_combinatorial.exe        # rank/unrank number system
./build/tests/test_tb_king_table.exe        # 462 canonical king-pair table
./build/tests/test_tb_comb_index.exe        # comb encode/decode (fast 3-man)
./build/tests/test_tb_comb_index.exe slow   # + 4-man, duplicate pieces, 5-man
./build/tests/test_tb_comb_solve.exe        # memoryless sweep == dense (3-man)
./build/tests/test_tb_comb_solve.exe slow   # MANUAL: 4-man KQKR, ~10 min
```

## CUDA

```bash
# Build the CUDA targets (requires nvcc) and run the on-GPU gates
cmake --build build
# Profiling harness (needs a GPU; DRAM counters need a counter-enabled host)
./cuda/profile.sh
```

See [`../cuda/ROADMAP.md`](../cuda/ROADMAP.md) for the phase-by-phase gates and
[`../cuda/PROFILING.md`](../cuda/PROFILING.md) for the optimization log.

## External verification (needs network + curl)

```bash
# Verify generated tables against the Lichess (Gaviota DTM) tablebase API
python python/verify_tablebase.py 25 KQK KRK KQKR
```

The script shells out to `curl` because Python's SSL trust store is misconfigured
in the development environment.

## Regenerating the evaluation tables

```bash
# One-off: regenerate src/eval_tables.inc from the Python Texel constants
python python/gen_eval_tables.py > src/eval_tables.inc
```

## Cloud (multi-GPU generation)

See [`../cloud/README.md`](../cloud/README.md) for the Terraform + Docker +
fan-out pipeline that generates all 28 tables on rented GPUs and streams them to
S3.
