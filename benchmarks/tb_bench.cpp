// Phase 2 / Step 2.3 — tablebase generation baseline + WDL/DTM statistics.
//
// Times single-threaded retrograde generation (the dense negamax sweep) for
// each 3-man endgame and reports:
//   * positions/sec         — indices solved per second (the headline baseline
//                             Phase 3's CUDA kernels must beat)
//   * position-updates/sec  — N x passes per second: the sweep re-touches every
//                             position each pass, so this is the value-update
//                             rate the GPU version (a bandwidth-bound sweep)
//                             most directly competes with
//   * WDL counts and a distance-to-mate histogram
//
// Absolute numbers only, never percentage speedups (project rule).
// Meaningful only in Release (-O3 -march=native).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void run(PieceType wp, const char* name) {
    // Build the index (timed separately from solving).
    const auto ti0 = std::chrono::steady_clock::now();
    tb::Index idx(wp);
    const double index_secs = secs_since(ti0);
    const std::size_t N = idx.size();

    // Best-of-3 solve time (deterministic result; timing varies with the OS).
    double   best_solve = 1e300;
    tb::Table tbl;
    for (int r = 0; r < 3; ++r) {
        const auto ts0 = std::chrono::steady_clock::now();
        tbl = tb::solve_sweep(idx);
        const double dt = secs_since(ts0);
        if (dt < best_solve) best_solve = dt;
    }

    const double pos_per_s = static_cast<double>(N) / best_solve;
    const double upd = static_cast<double>(N) * static_cast<double>(tbl.passes);
    const double upd_per_s = upd / best_solve;

    std::printf("== %s ==\n", name);
    std::printf("  indices              %zu\n", N);
    std::printf("  index build          %.3f s\n", index_secs);
    std::printf("  solve (best of 3)    %.3f s   (%d passes)\n", best_solve, tbl.passes);
    std::printf("  positions/sec        %.2f M\n", pos_per_s / 1e6);
    std::printf("  position-updates/sec %.2f M   (%.0f M updates total)\n", upd_per_s / 1e6, upd / 1e6);
    std::printf("  W / L / D            %zu / %zu / %zu\n", tbl.wins, tbl.losses, tbl.draws);
    std::printf("  max DTM              win %d plies (%d moves), loss %d plies\n",
                tbl.max_win_dtm, (tbl.max_win_dtm + 1) / 2, tbl.max_loss_dtm);

    // Win-DTM histogram (by move number), only non-empty buckets.
    if (tbl.wins > 0) {
        std::vector<std::size_t> hist(static_cast<std::size_t>(tbl.max_win_dtm) + 1, 0);
        for (const int16_t v : tbl.value)
            if (v > 0) ++hist[static_cast<std::size_t>(tb::win_dtm(v))];
        std::printf("  win DTM histogram (plies: count):\n   ");
        int shown = 0;
        for (std::size_t d = 1; d < hist.size(); ++d) {
            if (hist[d] == 0) continue;
            std::printf(" %zu:%zu", d, hist[d]);
            if (++shown % 8 == 0) std::printf("\n   ");
        }
        std::printf("\n");
    }
    std::printf("\n");
}

}  // namespace

int main() {
    slider::init();
    std::printf("Tablebase generation baseline (single-threaded, Release)\n\n");
    run(PieceType::Queen,  "KQK");
    run(PieceType::Rook,   "KRK");
    run(PieceType::Bishop, "KBK");
    run(PieceType::Knight, "KNK");
    return 0;
}
