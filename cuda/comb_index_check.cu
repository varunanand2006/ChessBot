// Phase 1c gate (device half) — comb_encode / comb_decode run in a CUDA kernel
// produce exactly what the host tb::CombIndex does.
//
// The host half (tests/test_tb_comb_index_device.cpp) already proves the
// device-shaped functions == CombIndex exhaustively on the CPU. This harness
// runs the SAME functions on the GPU over the whole index space and diffs the
// device output against the host CombIndex — closing the loop: host-mirror ==
// CombIndex (CPU test) and device == host-mirror (here), so device == CombIndex.
//
// Per index i: the kernel computes comb_decode(i) -> squares, then
// comb_encode(squares) -> reenc. Host checks squares == CombIndex::decode(i)
// and reenc == i. Built only where a CUDA compiler exists.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tb_comb_index.hpp"          // host CombIndex (reference + fill_device)
#include "tb_comb_index_device.hpp"   // POD structs + CH_HD comb_encode/decode
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

// One thread per index: decode then re-encode entirely on the device.
__global__ void k_roundtrip(uint64_t n, tb::DeviceKingTable kt,
                            tb::DeviceMaterial mat, int* out_sq,
                            uint64_t* out_reenc) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int sq[tb::kDevMaxMen];
    int bit = 0;
    tb::comb_decode(i, sq, &bit, kt, mat);
    for (int m = 0; m < mat.men; ++m) out_sq[i * mat.men + m] = sq[m];
    out_reenc[i] = tb::comb_encode(sq, bit, kt, mat);
}

// Copy the three KingTable arrays to the device and return a DeviceKingTable of
// device pointers. Pushes the allocations onto `frees` for later cudaFree.
static tb::DeviceKingTable upload_king_table(const tb::DeviceKingTable& host,
                                             std::vector<void*>& frees) {
    tb::DeviceKingTable dev = host;
    const size_t id_bytes = 4096 * sizeof(int16_t);
    const size_t tf_bytes = 4096 * sizeof(int8_t);
    const size_t cp_bytes = static_cast<size_t>(host.num_canonical) * sizeof(uint16_t);

    int16_t* d_id; CUDA_CHECK(cudaMalloc(&d_id, id_bytes));
    int8_t*  d_tf; CUDA_CHECK(cudaMalloc(&d_tf, tf_bytes));
    uint16_t* d_cp; CUDA_CHECK(cudaMalloc(&d_cp, cp_bytes));
    CUDA_CHECK(cudaMemcpy(d_id, host.id, id_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tf, host.transform, tf_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cp, host.canon_pair, cp_bytes, cudaMemcpyHostToDevice));

    dev.id = d_id; dev.transform = d_tf; dev.canon_pair = d_cp;
    frees.push_back(d_id); frees.push_back(d_tf); frees.push_back(d_cp);
    return dev;
}

static int g_fail = 0;

static void run_material(std::vector<tb::Piece> extras, const char* name) {
    tb::CombIndex idx(std::move(extras));
    const int men = idx.men();
    const uint64_t N = idx.size();

    tb::DeviceKingTable host_kt{};
    tb::DeviceMaterial  mat{};
    idx.fill_device(host_kt, mat);

    std::vector<void*> frees;
    tb::DeviceKingTable dev_kt = upload_king_table(host_kt, frees);

    int* d_sq; CUDA_CHECK(cudaMalloc(&d_sq, N * men * sizeof(int)));
    uint64_t* d_reenc; CUDA_CHECK(cudaMalloc(&d_reenc, N * sizeof(uint64_t)));

    const int block = 256;
    const uint64_t grid = (N + block - 1) / block;
    k_roundtrip<<<static_cast<unsigned>(grid), block>>>(N, dev_kt, mat, d_sq, d_reenc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_sq(N * men);
    std::vector<uint64_t> h_reenc(N);
    CUDA_CHECK(cudaMemcpy(h_sq.data(), d_sq, N * men * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_reenc.data(), d_reenc, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    uint64_t bad = 0;
    for (uint64_t i = 0; i < N; ++i) {
        int refsq[tb::kDevMaxMen];
        Color refstm;
        idx.decode(i, refsq, &refstm);
        bool ok = (h_reenc[i] == i);
        for (int m = 0; m < men && ok; ++m)
            if (h_sq[i * men + m] != refsq[m]) ok = false;
        if (!ok && bad < 5)
            std::fprintf(stderr, "  mismatch at i=%llu (reenc=%llu)\n",
                         (unsigned long long)i, (unsigned long long)h_reenc[i]);
        if (!ok) ++bad;
    }

    CUDA_CHECK(cudaFree(d_sq));
    CUDA_CHECK(cudaFree(d_reenc));
    for (void* p : frees) CUDA_CHECK(cudaFree(p));

    if (bad == 0)
        std::printf("[ OK ] %-7s  men=%d  size=%llu  device == host CombIndex\n",
                    name, men, (unsigned long long)N);
    else {
        std::printf("[FAIL] %-7s  %llu / %llu mismatches\n", name,
                    (unsigned long long)bad, (unsigned long long)N);
        ++g_fail;
    }
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    run_material({{W, PT::Rook}},                 "KRK");
    run_material({{W, PT::Queen}},                "KQK");
    run_material({{W, PT::Queen}, {B, PT::Rook}}, "KQKR");   // 4-man distinct
    run_material({{W, PT::Rook}, {W, PT::Rook}},  "KRRK");   // 4-man duplicate

    if (g_fail == 0) {
        std::printf("PASS: device comb index == host CombIndex on all materials.\n");
        return 0;
    }
    return 1;
}
