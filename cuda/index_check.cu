// Phase 1 gate (primitives) — the combinatorial number system + D4 symmetry
// run on the DEVICE and produce results bit-identical to the host.
//
// combinatorial.hpp and tb_symmetry.hpp are now CH_HD-annotated, so the SAME
// functions the CPU indexer uses compile for the device. This harness proves
// they compute the same thing on the GPU — the prerequisite for the full
// device CombIndex port (step 1c), whose bit-exact-vs-host diffing is only
// meaningful if the primitives underneath already agree.
//
// Three checks, mirroring tests/test_combinatorial.cpp but executed on-device:
//   A. binom(n,k) device == host for the whole 0..64 triangle.
//   B. rank/unrank round-trip on device (rank(unrank(r))==r) AND the device
//      unrank output equals the host unrank output, for a grid of (n,k).
//   C. transform_square(g,s) device == host for all 8x64 entries.
//
// The host reference is computed by calling the very same CH_HD functions on
// the host in this file — so a divergence means the toolchain miscompiled one
// side (e.g. a fast-math reassociation), which is exactly what we want caught
// before trusting any device table.
//
// Exit 0 == all agree. Built only when a CUDA compiler is present.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "combinatorial.hpp"
#include "tb_symmetry.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

// --- A. binom over the whole triangle --------------------------------------
__global__ void k_binom(uint64_t* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 65 * 65) return;
    int n = idx / 65, k = idx % 65;
    out[idx] = combo::binom(n, k);
}

// --- B. rank/unrank for one (n,k) ------------------------------------------
// One thread per rank r in [0,total): unrank on device into elems[r*k..], and
// record rr[r] = rank(unrank(r)) for the round-trip check.
__global__ void k_rank_unrank(int n, int k, uint64_t total, int* elems,
                              uint64_t* rr) {
    uint64_t r = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= total) return;
    int buf[16];  // k <= 10 in the cases below; 16 is comfortable headroom.
    combo::unrank_combination(r, k, buf);
    for (int i = 0; i < k; ++i) elems[r * k + i] = buf[i];
    rr[r] = combo::rank_combination(buf, k);
    (void)n;  // n only bounds the range; unrank doesn't read it.
}

// --- C. D4 transform table -------------------------------------------------
__global__ void k_transform(int* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 8 * 64) return;
    int g = idx / 64, s = idx % 64;
    out[idx] = tb::transform_square(g, s);
}

static int g_fail = 0;
static void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_fail; }

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }

    // --- A ---
    {
        const int N = 65 * 65;
        uint64_t* d; CUDA_CHECK(cudaMalloc(&d, N * sizeof(uint64_t)));
        k_binom<<<(N + 255) / 256, 256>>>(d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t* h = (uint64_t*)std::malloc(N * sizeof(uint64_t));
        CUDA_CHECK(cudaMemcpy(h, d, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        for (int n = 0; n <= 64; ++n)
            for (int k = 0; k <= 64; ++k)
                if (h[n * 65 + k] != combo::binom(n, k))
                    { fail("binom device != host"); n = 100; break; }
        std::free(h); CUDA_CHECK(cudaFree(d));
    }

    // --- B ---
    {
        struct { int n, k; } cases[] = {
            {5, 2}, {8, 3}, {12, 4}, {16, 5}, {24, 6}, {20, 10}, {32, 3}, {64, 2},
        };
        for (auto c : cases) {
            const uint64_t total = combo::binom(c.n, c.k);
            int* d_elems; CUDA_CHECK(cudaMalloc(&d_elems, total * c.k * sizeof(int)));
            uint64_t* d_rr; CUDA_CHECK(cudaMalloc(&d_rr, total * sizeof(uint64_t)));
            const int block = 256;
            const uint64_t grid = (total + block - 1) / block;
            k_rank_unrank<<<static_cast<unsigned>(grid), block>>>(
                c.n, c.k, total, d_elems, d_rr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            int* h_elems = (int*)std::malloc(total * c.k * sizeof(int));
            uint64_t* h_rr = (uint64_t*)std::malloc(total * sizeof(uint64_t));
            CUDA_CHECK(cudaMemcpy(h_elems, d_elems, total * c.k * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_rr, d_rr, total * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));

            int ref[16];
            for (uint64_t r = 0; r < total; ++r) {
                if (h_rr[r] != r) { fail("rank(unrank(r)) != r on device"); break; }
                combo::unrank_combination(r, c.k, ref);  // host reference
                for (int i = 0; i < c.k; ++i)
                    if (h_elems[r * c.k + i] != ref[i])
                        { fail("device unrank != host unrank"); r = total; break; }
            }
            std::free(h_elems); std::free(h_rr);
            CUDA_CHECK(cudaFree(d_elems)); CUDA_CHECK(cudaFree(d_rr));
        }
    }

    // --- C ---
    {
        const int N = 8 * 64;
        int* d; CUDA_CHECK(cudaMalloc(&d, N * sizeof(int)));
        k_transform<<<(N + 255) / 256, 256>>>(d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int* h = (int*)std::malloc(N * sizeof(int));
        CUDA_CHECK(cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost));
        for (int g = 0; g < 8; ++g)
            for (int s = 0; s < 64; ++s)
                if (h[g * 64 + s] != tb::transform_square(g, s))
                    { fail("transform_square device != host"); g = 100; break; }
        std::free(h); CUDA_CHECK(cudaFree(d));
    }

    if (g_fail == 0) {
        std::printf("PASS: binom / rank-unrank / D4 all match host on device.\n");
        return 0;
    }
    std::printf("FAIL: %d check(s) mismatched.\n", g_fail);
    return 1;
}
