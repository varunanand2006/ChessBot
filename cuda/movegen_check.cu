// Phase 2 gate (device half) — movegen_dev::generate_legal run in a CUDA kernel
// produces the same legal move set as the host movegen::generate_legal, over
// real tablebase positions (decoded from CombIndex).
//
// The host half (tests/test_movegen_device.cpp) already proves the device-shaped
// generator == the reference exhaustively over perft trees on the CPU. This
// harness runs the SAME code on the GPU: each thread decodes one position's
// squares (passed in), runs generate_legal, and emits an order-independent
// signature (count + XOR of a per-move hash) of its legal set. The host computes
// the same signature from movegen::generate_legal and diffs. Built only where a
// CUDA compiler exists.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "movegen.hpp"            // host reference
#include "movegen_device.hpp"    // CH_HD device generator
#include "position.hpp"
#include "slider.hpp"
#include "tb_comb_index.hpp"     // decode positions to sweep
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

// Order-independent per-move hash; XOR-combined over a move list so the
// signature does not depend on generation order (belt-and-suspenders: host and
// device generate in the same order anyway).
CH_HD inline uint64_t move_hash(Move m) {
    uint64_t x = static_cast<uint64_t>(m) + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__global__ void k_movegen(int n, const Position* pos, slider::DeviceSliders S,
                          int* out_count, uint64_t* out_sig) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Position p = pos[i];  // thread-local copy; generate_legal make/unmakes it
    MoveList list;
    movegen_dev::generate_legal(p, list, S);
    uint64_t sig = 0;
    for (int j = 0; j < list.size(); ++j) sig ^= move_hash(list[j]);
    out_count[i] = list.size();
    out_sig[i] = sig;
}

static int g_fail = 0;

static void run_material(std::vector<tb::Piece> extras, const char* name,
                         const slider::DeviceSliders& dev_s, std::size_t step) {
    tb::CombIndex idx(std::move(extras));
    const std::size_t N = idx.size();

    // Decode the sampled positions on the host.
    std::vector<Position> host_pos;
    host_pos.reserve(N / step + 1);
    for (std::size_t i = 0; i < N; i += step) host_pos.push_back(idx.decode_pos(i));
    const int n = static_cast<int>(host_pos.size());

    Position* d_pos;
    CUDA_CHECK(cudaMalloc(&d_pos, n * sizeof(Position)));
    CUDA_CHECK(cudaMemcpy(d_pos, host_pos.data(), n * sizeof(Position), cudaMemcpyHostToDevice));
    int* d_count; CUDA_CHECK(cudaMalloc(&d_count, n * sizeof(int)));
    uint64_t* d_sig; CUDA_CHECK(cudaMalloc(&d_sig, n * sizeof(uint64_t)));

    const int block = 128;  // Position is register-heavy; a smaller block helps occupancy
    k_movegen<<<(n + block - 1) / block, block>>>(n, d_pos, dev_s, d_count, d_sig);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_count(n);
    std::vector<uint64_t> h_sig(n);
    CUDA_CHECK(cudaMemcpy(h_count.data(), d_count, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sig.data(), d_sig, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; ++i) {
        Position p = host_pos[i];
        MoveList ref;
        movegen::generate_legal(p, ref);
        uint64_t sig = 0;
        for (int j = 0; j < ref.size(); ++j) sig ^= move_hash(ref[j]);
        if (h_count[i] != ref.size() || h_sig[i] != sig) {
            if (bad < 5)
                std::fprintf(stderr, "  mismatch idx-sample %d: dev(count=%d) ref(count=%d)\n",
                             i, h_count[i], ref.size());
            ++bad;
        }
    }

    CUDA_CHECK(cudaFree(d_pos)); CUDA_CHECK(cudaFree(d_count)); CUDA_CHECK(cudaFree(d_sig));

    if (bad == 0)
        std::printf("[ OK ] %-7s  positions=%d  device == host movegen\n", name, n);
    else {
        std::printf("[FAIL] %-7s  %d / %d mismatches\n", name, bad, n);
        ++g_fail;
    }
}

int main() {
    static_assert(std::is_trivially_copyable_v<Position>,
                  "Position must be memcpy-able to the device");

    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }

    slider::init();
    slider::DeviceSliders host_s;
    slider::get_device_sliders(host_s);

    slider::DeviceSliders dev_s = host_s;
    slider::DeviceMagic* d_rm; slider::DeviceMagic* d_bm; uint64_t* d_rt; uint64_t* d_bt;
    CUDA_CHECK(cudaMalloc(&d_rm, 64 * sizeof(slider::DeviceMagic)));
    CUDA_CHECK(cudaMalloc(&d_bm, 64 * sizeof(slider::DeviceMagic)));
    CUDA_CHECK(cudaMalloc(&d_rt, host_s.rook_table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, host_s.bishop_table_size * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_rm, host_s.rook, 64 * sizeof(slider::DeviceMagic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bm, host_s.bishop, 64 * sizeof(slider::DeviceMagic), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rt, host_s.rook_table, host_s.rook_table_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt, host_s.bishop_table, host_s.bishop_table_size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    dev_s.rook = d_rm; dev_s.bishop = d_bm; dev_s.rook_table = d_rt; dev_s.bishop_table = d_bt;

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    // `step` is the sampling stride: host_pos holds N/step decoded Positions
    // (~80 B each). Keep N/step bounded — a 5-man material (KQRKR ~209M indices)
    // at step=1 would need ~16 GB of host RAM for host_pos ALONE, choking a small
    // rented instance before the GPU runs. Scale step up with material size.
    run_material({{W, PT::Rook}},                 "KRK",  dev_s, 1);   // full 57k
    run_material({{W, PT::Queen}},                "KQK",  dev_s, 1);   // full 57k
    run_material({{W, PT::Queen}, {B, PT::Rook}}, "KQKR", dev_s, 3);   // ~1.16M sample

    // Free the uploaded magic tables (the per-material scratch is freed inside
    // run_material). Harmless at process exit, but leak-clean for reuse.
    cudaFree(d_rm); cudaFree(d_bm); cudaFree(d_rt); cudaFree(d_bt);

    if (g_fail == 0) {
        std::printf("PASS: device movegen == host over all sampled TB positions.\n");
        return 0;
    }
    return 1;
}
