// Host/device compilation compatibility — the one macro that lets a single set
// of headers compile for BOTH the CPU engine and the CUDA device.
//
// WHY: the project is deliberately a single C++ dialect across host and device
// (see CLAUDE.md) so the tablebase index arithmetic, D4 symmetry, and magic
// sliders port to the GPU *verbatim* rather than being re-implemented. The only
// thing device code needs that host code does not is the `__host__ __device__`
// annotation on shared functions. This macro supplies it under nvcc and expands
// to nothing under a plain host compiler — so the same header is valid CPU-only
// (no toolkit installed) and device-callable when built with nvcc.
//
// Usage: annotate a shared free function with CH_HD:
//     CH_HD uint64_t rank_combination(const int* elems, int k) { ... }
// Phase 1 retrofits combinatorial.hpp / tb_symmetry.hpp with this. Phase 0 only
// needs the macro to exist and be proven by the hello harness.

#pragma once

// __CUDACC__ is defined by nvcc for every translation unit it compiles (both the
// host and device passes), and by nothing else — so it is the correct switch for
// "am I being compiled by the CUDA compiler", independent of whether this
// particular function ends up on the device.
#if defined(__CUDACC__)
  #define CH_HD __host__ __device__
#else
  #define CH_HD
#endif
