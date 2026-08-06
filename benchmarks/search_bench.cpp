// Search throughput benchmark — nodes/sec of the alpha-beta search.
//
// Absolute NPS only, never % speedups (project rule). Meaningful in Release
// (-O3 -march=native). A fixed set of positions searched to fixed depth; the
// headline is aggregate NPS (search + quiescence nodes / wall time). Tablebase
// probing is off so this measures the search itself.

#include <chrono>
#include <cstdint>
#include <cstdio>

#include "position.hpp"
#include "search.hpp"
#include "slider.hpp"

namespace {

struct Case { const char* fen; const char* name; int depth; };

// A spread: opening, a tactical middlegame (Kiwipete), a quieter middlegame, and
// an endgame. Depths chosen so each runs in a fraction of a second.
const Case CASES[] = {
    {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",              "startpos",   7},
    {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  "kiwipete",   6},
    {"r1bq1rk1/pp2bppp/2n2n2/2pp4/3P4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 1",     "middlegame", 7},
    {"8/2k5/8/8/8/4K3/8/R7 w - - 0 1",                                        "endgame",    12},
};

}  // namespace

int main() {
    slider::init();
    std::printf("Search throughput (single-threaded, Release, TB off)\n\n");

    uint64_t total_nodes = 0;
    double   total_secs  = 0.0;

    for (const Case& c : CASES) {
        Position pos;
        if (!set_fen(pos, c.fen)) { std::printf("bad FEN: %s\n", c.fen); continue; }

        search::new_game();
        search::history_add(pos.zobrist);

        const auto t0 = std::chrono::steady_clock::now();
        const search::SearchResult r = search::find_best_move(pos, c.depth);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        total_nodes += r.nodes;
        total_secs  += secs;
        const double mnps = secs > 0 ? r.nodes / secs / 1e6 : 0.0;
        std::printf("  %-11s depth %2d   %10llu nodes   %7.3fs   %6.2f Mnps   best %s\n",
                    c.name, c.depth, static_cast<unsigned long long>(r.nodes), secs, mnps,
                    move_to_uci(r.best_move).c_str());
    }

    const double agg = total_secs > 0 ? total_nodes / total_secs / 1e6 : 0.0;
    std::printf("\n  aggregate: %llu nodes / %.3fs = %.2f Mnps\n",
                static_cast<unsigned long long>(total_nodes), total_secs, agg);
    return 0;
}
