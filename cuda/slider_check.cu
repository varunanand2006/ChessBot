// Phase 2a gate (device half) — the magic slider attack query runs on the GPU
// and produces the same rook/bishop/queen attack sets as the host slider.cpp.
//
// The magic tables are built once on the host (slider::init) and uploaded; this
// kernel does the O(1) multiply-shift query on the device (slider_device.hpp)
// for many random occupancies per square, and the host diffs each against
// slider::rook_attacks/bishop_attacks/queen_attacks. Same idea as
// tests/test_slider.cpp, but device-side. Built only where nvcc exists.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "slider.hpp"          // host build + reference query + get_device_sliders
#include "slider_device.hpp"   // CH_HD device query
#include "types.hpp"

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

// Deterministic per-test occupancy so host and device generate the SAME input
// from the index alone (no occupancy array to upload/keep in sync).
CH_HD inline uint64_t occ_of(uint64_t i) {
    uint64_t x = i + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__global__ void k_sliders(uint64_t n, slider::DeviceSliders s,
                          uint64_t* rook, uint64_t* bishop, uint64_t* queen) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int sq = static_cast<int>(i & 63);
    const uint64_t occ = occ_of(i);
    rook[i]   = slider::rook_attacks_dev(s, sq, occ);
    bishop[i] = slider::bishop_attacks_dev(s, sq, occ);
    queen[i]  = slider::queen_attacks_dev(s, sq, occ);
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }

    slider::init();
    slider::DeviceSliders host_s;
    slider::get_device_sliders(host_s);

    // Upload the four tables and rebind the pointers to device memory.
    slider::DeviceSliders dev_s = host_s;
    slider::DeviceMagic* d_rook_m;
    slider::DeviceMagic* d_bishop_m;
    uint64_t* d_rook_t;
    uint64_t* d_bishop_t;
    CUDA_CHECK(cudaMalloc(&d_rook_m, 64 * sizeof(slider::DeviceMagic)));
    CUDA_CHECK(cudaMalloc(&d_bishop_m, 64 * sizeof(slider::DeviceMagic)));
    CUDA_CHECK(cudaMalloc(&d_rook_t, host_s.rook_table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bishop_t, host_s.bishop_table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_rook_m, host_s.rook, 64 * sizeof(slider::DeviceMagic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bishop_m, host_s.bishop, 64 * sizeof(slider::DeviceMagic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rook_t, host_s.rook_table, host_s.rook_table_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bishop_t, host_s.bishop_table, host_s.bishop_table_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    dev_s.rook = d_rook_m; dev_s.bishop = d_bishop_m;
    dev_s.rook_table = d_rook_t; dev_s.bishop_table = d_bishop_t;

    const uint64_t N = 64ull * 8192;  // 8192 random occupancies per square
    uint64_t *d_r, *d_b, *d_q;
    CUDA_CHECK(cudaMalloc(&d_r, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_q, N * sizeof(uint64_t)));

    const int block = 256;
    k_sliders<<<static_cast<unsigned>((N + block - 1) / block), block>>>(N, dev_s, d_r, d_b, d_q);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> hr(N), hb(N), hq(N);
    CUDA_CHECK(cudaMemcpy(hr.data(), d_r, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hb.data(), d_b, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hq.data(), d_q, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint64_t bad = 0;
    for (uint64_t i = 0; i < N; ++i) {
        const Square sq = static_cast<Square>(i & 63);
        const uint64_t occ = occ_of(i);
        const uint64_t r = slider::rook_attacks(sq, occ);
        const uint64_t b = slider::bishop_attacks(sq, occ);
        const uint64_t q = slider::queen_attacks(sq, occ);
        if (hr[i] != r || hb[i] != b || hq[i] != q) {
            if (bad < 5)
                std::fprintf(stderr, "  mismatch i=%llu sq=%d\n",
                             (unsigned long long)i, (int)(i & 63));
            ++bad;
        }
    }

    if (bad == 0) {
        std::printf("PASS: device sliders == host on %llu (square,occ) cases.\n",
                    (unsigned long long)N);
        return 0;
    }
    std::printf("FAIL: %llu / %llu mismatches.\n",
                (unsigned long long)bad, (unsigned long long)N);
    return 1;
}
