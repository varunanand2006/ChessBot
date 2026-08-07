// Phase 0 gate — CUDA scaffold smoke test.
//
// Proves three things before any real port work starts:
//   1. The toolchain builds a .cu (nvcc + our C++20 host dialect) and launches a
//      kernel on the rented box.
//   2. The CH_HD macro works: the SAME free function compiles for host and
//      device, and produces BIT-IDENTICAL results on both. This is the whole
//      premise of the port (index math written once, run on both sides), so it
//      is the thing the gate actually checks — not just "a kernel ran".
//   3. Basic device introspection + the theoretical memory bandwidth, since a
//      bandwidth-bound dense sweep is the point of the GPU phase and this is the
//      ceiling every later Nsight number is measured against.
//
// Exit code 0 == host and device agree on all N values. Non-zero == mismatch or
// CUDA error, so ctest (and a human on the box) catch a broken toolchain/port.
//
// Build: added to the CMake build ONLY when a CUDA compiler is found (see the
// top-level CMakeLists guard), so it never affects the CPU-only laptop build.

#include <cstdint>
#include <cstdio>

#include "cuda_compat.hpp"

// --- Error checking --------------------------------------------------------
// Hot device code never uses exceptions; this is host-side setup/teardown, so a
// simple abort-on-error macro is the right tool — fail loud and immediately at
// the offending call site rather than limping on with a corrupt state.
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(err_), __FILE__, __LINE__,          \
                         cudaGetErrorString(err_));                           \
            return 2;                                                          \
        }                                                                      \
    } while (0)

// --- The shared function under test ----------------------------------------
// A deterministic integer mix (SplitMix64's finalizer). Nothing chess-specific;
// it just has to be non-trivial arithmetic that must come out bit-identical on
// host and device — a stand-in for the index/rank arithmetic Phase 1 ports with
// this exact CH_HD annotation. If host and device ever disagreed here (e.g. a
// stray fast-math flag reassociating integer ops), the port's bit-exact oracle
// diffing would be meaningless, so we pin it down at Phase 0.
CH_HD uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// --- The kernel ------------------------------------------------------------
// One thread per element — the exact launch shape the sweep kernel will use
// (one thread per position). Grid-stride so a too-small grid still covers N.
__global__ void fill_mix(uint64_t* out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = mix64(static_cast<uint64_t>(i));
    }
}

static void print_device_info() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return;

    // Theoretical peak DRAM bandwidth = 2 (DDR) * busWidth/8 bytes * memClk.
    // memoryClockRate is in kHz, memoryBusWidth in bits. This is the ceiling the
    // achieved-bandwidth headline metric will be reported against.
    const double gb_per_s =
        2.0 * (p.memoryBusWidth / 8.0) * (p.memoryClockRate * 1e3) / 1e9;

    std::printf("Device: %s (compute capability %d.%d)\n", p.name, p.major, p.minor);
    std::printf("  SMs: %d   global mem: %.1f GB   bus: %d-bit\n",
                p.multiProcessorCount,
                p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                p.memoryBusWidth);
    std::printf("  theoretical peak DRAM bandwidth: %.0f GB/s\n", gb_per_s);
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        std::fprintf(stderr, "No CUDA device found.\n");
        return 2;
    }
    print_device_info();

    // A size that spans many blocks so the launch config is exercised, and that
    // is small enough to verify exhaustively on the host.
    const int N = 1 << 20;  // 1,048,576
    const size_t bytes = static_cast<size_t>(N) * sizeof(uint64_t);

    uint64_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const int block = 256;
    const int grid = (N + block - 1) / block;
    fill_mix<<<grid, block>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError());       // launch error (bad config)
    CUDA_CHECK(cudaDeviceSynchronize());  // execution error (in-kernel fault)

    // Pull the device results back and compare against the host computation of
    // the SAME CH_HD function. Bit-exact equality is the gate.
    uint64_t* h_out = static_cast<uint64_t*>(std::malloc(bytes));
    if (!h_out) { std::fprintf(stderr, "host malloc failed\n"); return 2; }
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != mix64(static_cast<uint64_t>(i))) {
            if (mismatches < 5)
                std::fprintf(stderr, "mismatch at %d: device=%llu host=%llu\n",
                             i, (unsigned long long)h_out[i],
                             (unsigned long long)mix64((uint64_t)i));
            ++mismatches;
        }
    }

    std::free(h_out);
    CUDA_CHECK(cudaFree(d_out));

    if (mismatches == 0) {
        std::printf("PASS: host == device on all %d values (CH_HD verified).\n", N);
        return 0;
    }
    std::printf("FAIL: %d / %d mismatches.\n", mismatches, N);
    return 1;
}
