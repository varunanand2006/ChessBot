// Phase 4 measurement — what tablebase probing buys the search.
//
// For each endgame, run a fixed-depth search twice: once with the heuristic
// evaluation only, once with tablebase probing on. A probe returns the exact
// win/draw/loss at the node and cuts the search off there, so the tree collapses
// to (root moves) x (one probe each). We report the absolute node counts and the
// reduction factor — the headline number for this step.
//
// The TB tables are built lazily on first probe; a short untimed warmup search
// builds them so the measured run reflects steady-state probing, not one-time
// generation. Node counts are deterministic and unaffected by that warmup.

#include <chrono>
#include <cstdint>
#include <cstdio>

#include "position.hpp"
#include "search.hpp"
#include "slider.hpp"
#include "tb_probe.hpp"

namespace {

double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

search::SearchResult timed(Position& pos, int depth, double& out_secs) {
    search::new_game();
    search::history_add(pos.zobrist);
    const auto t0 = std::chrono::steady_clock::now();
    const search::SearchResult r = search::find_best_move(pos, depth, /*verbose=*/false);
    out_secs = secs_since(t0);
    return r;
}

void run(const char* fen, const char* name, int depth) {
    Position pos;
    if (!set_fen(pos, fen)) { std::printf("bad FEN: %s\n", fen); return; }

    // Baseline: heuristic eval only.
    search::set_use_tablebase(false);
    double s0;
    const search::SearchResult r0 = timed(pos, depth, s0);

    // Warm the table (untimed; node count of the measured run is unaffected).
    search::set_use_tablebase(true);
    double warm;
    timed(pos, 2, warm);

    // Measured: with probing.
    double s1;
    const search::SearchResult r1 = timed(pos, depth, s1);

    const double factor = r1.nodes ? static_cast<double>(r0.nodes) / static_cast<double>(r1.nodes) : 0.0;
    std::printf("== %-5s  depth %d ==\n", name, depth);
    std::printf("  heuristic : %10llu nodes   %6.3fs   best %s  score %+d\n",
                static_cast<unsigned long long>(r0.nodes), s0,
                move_to_uci(r0.best_move).c_str(), r0.score);
    std::printf("  tablebase : %10llu nodes   %6.3fs   best %s  score %+d\n",
                static_cast<unsigned long long>(r1.nodes), s1,
                move_to_uci(r1.best_move).c_str(), r1.score);
    std::printf("  node reduction: %.1fx\n\n", factor);
}

}  // namespace

int main() {
    slider::init();
    std::printf("Tablebase-probing node reduction (single-threaded, Release)\n\n");
    // 3-man (tables build in a blink). Legal positions: the side to move is not
    // giving check to the enemy king (a1 sliders would).
    run("8/8/8/4k3/8/8/8/3RK3 w - - 0 1",  "KRK",  12);
    run("8/8/8/4k3/8/8/8/3QK3 w - - 0 1",  "KQK",  12);
    // 4-man: builds KQKR (~2.5M) + its 3-man sub-tables once, then probes.
    run("4k3/8/8/8/8/8/4r3/3QK3 w - - 0 1", "KQKR", 10);
    return 0;
}
