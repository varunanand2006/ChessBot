// Phase 3 gate (device half) + the retrograde sweep kernel itself.
//
// This is the portfolio spine: the CPU solve_sweep_comb's per-node update, run
// as a CUDA kernel one thread per position, iterated Jacobi (ping-pong buffers)
// to a fixpoint. For <=4-man the final device table must be BIT-EXACT to
// solve_sweep_comb (KQKR, all 2,467,122 legal positions, mate-in-35). For 5-man
// the memoryless CPU oracle is ~15-20 h, so instead we report WDL/DTM stats and
// dump sample positions for external (Lichess/Gaviota) verification — the same
// device kernels, gated bit-exact at <=4-man, run at 5-man scale.
//
// The host builds the one-time data (classification + capture sub-tables, via
// tb_sweep_setup) and uploads it; the device does every pass. Convergence is a
// single global flag the host reads each pass. This is a bandwidth-bound dense
// sweep — Phase 4 profiles achieved bandwidth with Nsight; Phase 3 is
// correctness first.
//
// Built only where a CUDA compiler exists.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "position.hpp"        // to_fen for the 5-man sample dump
#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_material.hpp"      // shared tb::parse_material (comb cap)
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
// value = max over moves of age(-child); everything else copies through. The
// `changed` flag drives host convergence. Correctness never depended on an
// atomic here — every writer stores the identical value 1 to one aligned word,
// so any interleaving yields 1 — but a plain write is a formal data race that
// compute-sanitizer flags. atomicOr is the sanitizer-clean equivalent at
// negligible cost: it fires at most once per thread and only on passes that
// actually change something (the last, converging pass sets it zero times).
// v_old/v_new/state/subs are __restrict__: they are four distinct device
// allocations that never alias (v_old and v_new are the ping-pong buffers,
// swapped between passes but always distinct within a pass). This lets nvcc load
// the scattered child gathers in sweep_update through the read-only data cache
// (LDG). Correctness-neutral (a pure aliasing hint); the win is GPU-only.
__global__ void k_sweep_pass(uint64_t n, const int16_t* __restrict__ v_old,
                             int16_t* __restrict__ v_new,
                             const uint8_t* __restrict__ state,
                             tb::DeviceKingTable kt, tb::DeviceMaterial mat,
                             slider::DeviceSliders S,
                             const tb::DeviceSub* __restrict__ subs, int* changed) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (state[i] != tb::SW_SOLVE) { v_new[i] = v_old[i]; return; }
    const int16_t nv = tb::sweep_update(i, v_old, kt, mat, S, subs);
    v_new[i] = nv;
    if (nv != v_old[i]) atomicOr(changed, 1);
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

    // Build the host magic slider tables up front. build_sweep_setup() below runs
    // the host legality/classification pass, which calls slider::rook_attacks on
    // the CPU to test king safety; those tables are lazily built by slider::init()
    // (idempotent). upload_sliders() also calls init(), but only AFTER
    // build_sweep_setup — so without this the first host rook_attacks dereferences
    // an unbuilt table and segfaults. Every other CPU-side tool inits at startup;
    // this harness must too.
    slider::init();

    // Material: KQKR by default. Any distinct-piece pawnless material the
    // combinatorial indexer handles — KRK/KQK (smoke), KQKR (the 4-man gate),
    // KQRKR/KQRKN/... (5-man). Parsed by the shared tb::parse_material with the
    // 6-extra comb cap. argv[2] optionally sets the sample count for the 5-man
    // dump (default 40; unused by the <=4-man bit-exact gate).
    const std::string which = (argc > 1) ? argv[1] : "KQKR";
    const int n_samples = (argc > 2) ? std::max(1, std::atoi(argv[2])) : 40;
    std::vector<tb::Piece> extras;
    std::string name;
    if (!tb::parse_material(which.c_str(), extras, name, /*max_extras=*/6)) {
        std::fprintf(stderr, "unknown/unsupported material %s\n", which.c_str());
        return 2;
    }

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
    // Block size 128 is a STARTING default, not a tuned value: sweep_update is
    // register-heavy (a full Position + MoveLists live per thread), so a smaller
    // block often gives better occupancy than the usual 256 when registers, not
    // block count, are the limiter. The right value is whatever Nsight's
    // occupancy analysis says on the actual card (Phase 4) — profile.sh reports
    // registers/thread and achieved occupancy to drive that choice.
    const int block = 128;
    const uint64_t grid = (N + block - 1) / block;  // <= ~1.6M for 5-man; fits grid.x
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
        if (passes > tb::MATE) {  // safety net; a correct sweep converges long before
            std::fprintf(stderr, "did not converge\n");
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            for (void* p : frees) cudaFree(p);
            return 2;
        }
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    // v_old now holds the converged table.
    std::vector<int16_t> got(N);
    CUDA_CHECK(cudaMemcpy(got.data(), v_old, N * sizeof(int16_t), cudaMemcpyDeviceToHost));

    // --- Headline metrics ---------------------------------------------------
    // The sweep is bandwidth-bound, so the number that matters is achieved DRAM
    // bandwidth. ncu (dram__bytes.sum) gives the true figure counting every
    // table/movegen load; here we report a LOWER BOUND from the value-buffer
    // traffic alone — each pass streams v_old (read, 2 B/pos), state (read,
    // 1 B/pos) and v_new (write, 2 B/pos) = 5 B/pos — plus positions/sec and
    // per-pass time. The true achieved bandwidth is higher (child/table reads);
    // profile.sh + Nsight report it exactly.
    const double value_bytes = 5.0 * (double)N * (double)passes;
    const double gb_per_s    = value_bytes / (ms * 1e6);  // (bytes)/(ms*1e-3)/1e9

    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    const double peak_gb_s =
        2.0 * (prop.memoryBusWidth / 8.0) * (prop.memoryClockRate * 1e3) / 1e9;

    std::printf("Converged in %d passes, %.1f ms  (%.2f ms/pass)\n",
                passes, ms, ms / passes);
    std::printf("  throughput: %.1f Mpos/s (over all passes)\n",
                (double)N * passes / (ms * 1e3));
    std::printf("  value-buffer bandwidth: %.0f GB/s  (lower bound; %.1f%% of %.0f GB/s peak)\n",
                gb_per_s, peak_gb_s > 0 ? 100.0 * gb_per_s / peak_gb_s : 0.0, peak_gb_s);
    std::printf("  (run cuda/profile.sh for the true achieved DRAM bandwidth via Nsight)\n");

    // --- Verify ------------------------------------------------------------
    // <=4-man: the CPU oracle (solve_sweep_comb) is affordable (~seconds to
    // ~minutes), so gate the device table BIT-EXACT against it — the Phase 3
    // correctness contract. 5-man: that same memoryless oracle over ~210M
    // positions is ~15-20 h, so we do NOT run it. Every device stage underneath
    // (index, sliders, movegen, comb-index, and the sweep kernel itself) is
    // already gated bit-exact on <=4-man, so a 5-man run exercises the SAME
    // kernels at larger N. The fitting check there is an INDEPENDENT external
    // oracle on a sample: report WDL/DTM stats + dump sample positions
    // (fen,category,signed_dtm) for python/verify_tablebase.py --csv (Lichess).
    const bool have_oracle = (idx.men() <= 4);

    if (have_oracle) {
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

    // --- 5-man: stats + sample dump (no full CPU oracle) -------------------
    // WDL over LEGAL positions only (the comb space is intentionally loose;
    // illegal indices are pinned and must not be counted). Track the deepest
    // forced win/loss — the hardest positions, where bugs would hide — to both
    // report and include in the dump.
    uint64_t wins = 0, losses = 0, draws = 0;
    int      max_win = 0, max_loss = 0;
    uint64_t deepest_win = 0, deepest_loss = 0;
    bool     have_win = false, have_loss = false;
    for (uint64_t i = 0; i < N; ++i) {
        if (setup.state[i] == tb::SW_ILLEGAL) continue;
        const int16_t v = got[i];
        if (v > 0) {
            ++wins;
            const int d = tb::win_dtm(v);
            if (d > max_win) { max_win = d; deepest_win = i; have_win = true; }
        } else if (v < 0) {
            ++losses;
            const int d = tb::loss_dtm(v);
            if (d > max_loss) { max_loss = d; deepest_loss = i; have_loss = true; }
        } else {
            ++draws;
        }
    }

    std::printf("\n%s (5-man): W=%llu L=%llu D=%llu (legal positions)\n",
                name.c_str(), (unsigned long long)wins, (unsigned long long)losses,
                (unsigned long long)draws);
    std::printf("  deepest win: %d plies = mate-in-%d\n", max_win, (max_win + 1) / 2);

    // Emit fen,category,signed_dtm for a spread of legal positions + the two
    // deepest mates. signed_dtm = plies to mate, + if side-to-move wins, - if it
    // is being mated, 0 draw (the Lichess/Gaviota convention verify reads).
    const std::string csv_path = name + "_dump.csv";
    std::ofstream csv(csv_path);
    auto emit = [&](uint64_t i) {
        if (setup.state[i] == tb::SW_ILLEGAL) return;
        const int16_t v = got[i];
        const char* cat = (v > 0) ? "win" : (v < 0) ? "loss" : "draw";
        const int sdtm = (v > 0) ? tb::win_dtm(v) : (v < 0) ? -tb::loss_dtm(v) : 0;
        csv << to_fen(idx.decode_pos(i)) << ',' << cat << ',' << sdtm << '\n';
    };
    // Spread picks skip illegal indices by scanning forward to the next legal one.
    for (int k = 0; k < n_samples; ++k) {
        uint64_t i = (uint64_t)k * N / (uint64_t)n_samples;
        for (uint64_t j = 0; j < N && setup.state[i] == tb::SW_ILLEGAL; ++j)
            i = (i + 1) % N;
        emit(i);
    }
    if (have_win)  emit(deepest_win);
    if (have_loss) emit(deepest_loss);
    csv.close();

    for (void* p : frees) cudaFree(p);
    std::printf("  wrote %d+2 sample positions to %s\n", n_samples, csv_path.c_str());
    std::printf("  verify: python python/verify_tablebase.py --csv %s\n", csv_path.c_str());
    return 0;
}
