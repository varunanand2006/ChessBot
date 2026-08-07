# CUDA Port Roadmap — GPU retrograde tablebase sweep

**Goal:** run the endgame-tablebase retrograde sweep on the GPU. The kernel is the
CPU `solve_sweep_comb` inner loop, parallelized one thread per position, iterated
to convergence. Headline metric = **achieved memory bandwidth** (Nsight Compute),
since it's a bandwidth-bound dense sweep.

**Guiding rule:** the CPU `CombIndex` + `solve_sweep_comb` is a **bit-exact
oracle**. Every device step is diffed against it before moving on — never trust a
device result that doesn't match the host.

---

## Phases (each has a gate that must pass before the next)

**Phase 0 — Scaffold** *(local, no GPU)* — **BUILT (local half done; on-box gate pending a GPU)**
Create `/cuda`, add guarded CUDA to CMake (CPU-only build still works with no
toolkit), a `CH_HD` macro (`__host__ __device__` under nvcc, empty otherwise) so
shared headers compile for both, a hello-kernel + a device↔host equality harness.
**Gate:** builds and runs on the rented box.

- Files: `include/cuda_compat.hpp` (CH_HD macro), `cuda/hello.cu` (hello kernel +
  device↔host bit-exact equality harness over a CH_HD `mix64`), `cuda/CMakeLists.txt`,
  and the guarded `check_language(CUDA)` block at the bottom of the top-level
  `CMakeLists.txt`.
- **Local half verified:** with no CUDA toolkit, cmake prints "No CUDA compiler
  found; skipping cuda/" and the CPU build + all 19 fast tests pass unchanged.
  The `cuda/` targets only exist when nvcc is present, so the laptop build is
  byte-for-byte what it was.
- **On the rented box (the actual Phase 0 gate):**
  ```bash
  cmake -B build                       # now prints "CUDA compiler found ... arch: native"
  cmake --build build --target cuda_hello
  ./build/cuda/cuda_hello              # prints device + peak GB/s, then PASS
  ctest --test-dir build -R cuda_hello # same, as the gate
  ncu --set full ./build/cuda/cuda_hello   # confirms profiling counters (ERR_NVGPUCTRPERM => bad host)
  ```
  Exit 0 / "PASS: host == device on all 1048576 values (CH_HD verified)" is the
  gate. On CMake <3.24 pass `-DCMAKE_CUDA_ARCHITECTURES=89` (RTX 4090 = sm_89).

**Phase 1 — Index primitives**
Port `combinatorial` (binom → shrunk `__constant__`, rank/unrank as `CH_HD`), D4
transforms, king table → device arrays.
**Gate:** device decode/encode == host `CombIndex` on a batch of indices.

**Phase 2 — Move generation + make/unmake**
Magic sliders + attack tables → `__constant__`; `Position`/`MoveList` device-side.
This is the largest, most divergent piece.
**Gate:** device legal-move sets == host (device perft-style check).

**Phase 3 — The sweep kernel**
Classify kernel (pass 0) + update kernel (one pass) + host driver loop with a
global convergence flag; ping-pong `int16[N]` value buffers.
**Gate:** final device table is **bit-exact** to `solve_sweep_comb` — KQKR, all
2,467,122 positions, mate-in-35.

**Phase 4 — Profile + optimize** *(the portfolio meat)*
Nsight Compute → achieved bandwidth / occupancy / divergence. Then, measuring
each: `int empty[64]` → `uint64_t` bitmask + popcount; live-node work-list to cut
warp divergence; coalesced buffer layout.
**Gate:** documented bandwidth + speedup vs the CPU baseline.

**Phase 5 — Scale**
Solve a real 5-man to completion; report positions/sec + bandwidth vs the
~10.5-min KQKR single-thread CPU baseline.

---

## Device design (fixed decisions)

- **One thread per position**, ping-pong buffers (read old → write new, swap each
  pass) — clean Jacobi iteration, no races.
- **Convergence:** global `changed` atomic flag, read by host each pass.
- **Read-only device data** (`__constant__`/global): D4 (512 B), shrunk binom
  (~4 KB), king table (4096 + 4096 + 462), magic + attack tables, per-material
  group layout, and **host-solved capture sub-table values** (the dense solver
  stays on the host; the kernel only probes uploaded results).
- **Main risk:** `generate_legal`'s make→king-safety→unmake per move is heavy and
  divergent (variable move counts, `Position` register pressure). Nsight decides
  whether to compact live nodes / simplify the legality test.

## Workflow (laptop is Intel = no local CUDA run)

`edit .cu locally → git push → ssh rented box → git pull → cmake+nvcc build →
run → ncu --set full → commit Nsight report → terminate instance.`
First thing on the box: `ncu` a hello-kernel to confirm profiling-counter access.
Budget **~5–12 GPU-hours ≈ $3–10**.
