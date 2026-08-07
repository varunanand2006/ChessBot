// Phase 3 gate (device half) + the retrograde sweep kernel itself.
//
// This is the portfolio spine: the CPU solve_sweep_comb's per-node update, run
// as a CUDA kernel one thread per position, iterated Jacobi (ping-pong buffers)
// to a fixpoint. The final device table must be BIT-EXACT to solve_sweep_comb —
// KQKR, all 2,467,122 legal positions, mate-in-35.
//
// The host builds the one-time data (classification + capture sub-tables, via
// tb_sweep_setup) and uploads it; the device does every pass. Convergence is a
// single global flag the host reads each pass. This is a bandwidth-bound dense
// sweep — Phase 4 profiles achieved bandwidth with Nsight; Phase 3 is
// correctness first.
//
// Built only where a CUDA compiler exists.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_solve.hpp"
#include "tb_sweep_device.hpp"
#include "tb_sweep_setup.hpp"
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

// One pass of the sweep: one thread per position. SOLVE nodes recompute their
// value = max over moves of age(-child); everything else copies through. A
// device `changed` flag (set with a plain write — any thread setting it is
// enough, no atomic needed for a boolean "did anything change") drives host
// convergence.
__global__ void k_sweep_pass(uint64_t n, const int16_t* v_old, int16_t* v_new,
                             const uint8_t* state, tb::DeviceKingTable kt,
                             tb::DeviceMaterial mat, slider::DeviceSliders S,
                             const tb::DeviceSub* subs, int* changed) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != tb::SW_SOLVE) { v_new[i] = v_old[i]; return; }
    const int16_t nv = tb::sweep_update(i, v_old, kt, mat, S, subs);
    v_new[i] = nv;
    if (nv != v_old[i]) *changed = 1;
}

// --- small upload helpers --------------------------------------------------
template <class T>
static T* upload(const T* host, size_t count) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, host, count * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

static slider::DeviceSliders upload_sliders(std::vector<void*>& frees) {
    slider::init();
    slider::DeviceSliders h;
    slider::get_device_sliders(h);
    slider::DeviceSliders d = h;
    d.rook         = upload(h.rook, 64);
    d.bishop       = upload(h.bishop, 64);
    d.rook_table   = upload(h.rook_table, h.rook_table_size);
    d.bishop_table = upload(h.bishop_table, h.bishop_table_size);
    frees.push_back((void*)d.rook); frees.push_back((void*)d.bishop);
    frees.push_back((void*)d.rook_table); frees.push_back((void*)d.bishop_table);
    return d;
}

static tb::DeviceKingTable upload_king_table(const tb::DeviceKingTable& h,
                                             std::vector<void*>& frees) {
    tb::DeviceKingTable d = h;
    d.id         = upload(h.id, 4096);
    d.transform  = upload(h.transform, 4096);
    d.canon_pair = upload(h.canon_pair, static_cast<size_t>(h.num_canonical));
    frees.push_back((void*)d.id); frees.push_back((void*)d.transform);
    frees.push_back((void*)d.canon_pair);
    return d;
}

int main(int argc, char** argv) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "No CUDA device.\n"); return 2; }

    // Material: KQKR by default; `KRK`/`KQK` for a quick smoke run.
    const std::string which = (argc > 1) ? argv[1] : "KQKR";
    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;
    std::vector<tb::Piece> extras;
    if (which == "KRK")       extras = {{W, PT::Rook}};
    else if (which == "KQK")  extras = {{W, PT::Queen}};
    else if (which == "KQKR") extras = {{W, PT::Queen}, {B, PT::Rook}};
    else { std::fprintf(stderr, "unknown material %s\n", which.c_str()); return 2; }

    tb::CombIndex idx(extras);
    const uint64_t N = idx.size();
    std::printf("Material %s: %llu comb indices\n", which.c_str(), (unsigned long long)N);

    // --- Host one-time setup (classification + capture sub-tables) ----------
    tb::SweepSetup setup = tb::build_sweep_setup(idx);

    std::vector<void*> frees;
    slider::DeviceSliders d_sliders = upload_sliders(frees);
    tb::DeviceKingTable   d_kt      = upload_king_table(setup.kt, frees);

    // Sub-tables: upload each non-draw value array and build a device DeviceSub[].
    std::vector<tb::DeviceSub> h_subs(static_cast<size_t>(setup.mat.num_groups));
    setup.device_subs(h_subs.data());  // sets draw/mat + host value pointers
    for (int g = 0; g < setup.mat.num_groups; ++g) {
        if (!setup.subs[static_cast<size_t>(g)].draw) {
            const size_t len = setup.subs[static_cast<size_t>(g)].value.size();
            h_subs[static_cast<size_t>(g)].value =
                upload(setup.subs[static_cast<size_t>(g)].value.data(), len);
            frees.push_back((void*)h_subs[static_cast<size_t>(g)].value);
        }
    }
    tb::DeviceSub* d_subs = upload(h_subs.data(), h_subs.size());
    frees.push_back(d_subs);

    uint8_t*  d_state = upload(setup.state.data(), setup.state.size());
    int16_t*  d_a     = upload(setup.v_init.data(), setup.v_init.size());
    int16_t*  d_b     = upload(setup.v_init.data(), setup.v_init.size());
    int*      d_changed;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    frees.push_back(d_state); frees.push_back(d_a); frees.push_back(d_b); frees.push_back(d_changed);

    // --- Iterate to a fixpoint ---------------------------------------------
    const int block = 128;
    const uint64_t grid = (N + block - 1) / block;
    int16_t* v_old = d_a;
    int16_t* v_new = d_b;
    int passes = 0;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (;;) {
        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
        k_sweep_pass<<<static_cast<unsigned>(grid), block>>>(
            N, v_old, v_new, d_state, d_kt, setup.mat, d_sliders, d_subs, d_changed);
        CUDA_CHECK(cudaGetLastError());
        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        int16_t* tmp = v_old; v_old = v_new; v_new = tmp;  // swap
        ++passes;
        if (!changed) break;
        if (passes > tb::MATE) { std::fprintf(stderr, "did not converge\n"); return 2; }
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    // v_old now holds the converged table.
    std::vector<int16_t> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), v_old, N * sizeof(int16_t), cudaMemcpyDeviceToHost));

    std::printf("Converged in %d passes, %.1f ms (%.1f Mpos/s over passes)\n",
                passes, ms, (double)N * passes / (ms * 1e3));

    // --- Gate: bit-exact vs solve_sweep_comb -------------------------------
    tb::Table oracle = tb::solve_sweep_comb(idx);
    uint64_t diffs = 0, first = 0;
    int max_win = 0;
    for (uint64_t i = 0; i < N; ++i) {
        if (got[i] != oracle.value[i]) { if (!diffs) first = i; ++diffs; }
        if (got[i] > 0) { int d = tb::win_dtm(got[i]); if (d > max_win) max_win = d; }
    }

    for (void* p : frees) cudaFree(p);

    if (diffs == 0) {
        std::printf("PASS: device sweep == solve_sweep_comb on all %llu positions "
                    "(max win DTM=%d plies = mate-in-%d).\n",
                    (unsigned long long)N, max_win, (max_win + 1) / 2);
        return 0;
    }
    std::printf("FAIL: %llu / %llu differ (first at %llu: got=%d oracle=%d)\n",
                (unsigned long long)diffs, (unsigned long long)N,
                (unsigned long long)first, got[first], oracle.value[first]);
    return 1;
}
