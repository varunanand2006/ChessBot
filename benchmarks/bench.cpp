// NPS benchmark harness.
//
// Runs perft at a fixed depth across the standard perft position set and
// reports nodes-per-second — per position and in aggregate. Per the project's
// benchmarking rule, this reports absolute NPS, never percentage speedups: a
// speedup is only meaningful relative to a stated baseline machine + build, so
// the raw number is what gets recorded and compared over time.
//
// Each row also re-verifies the node count against the Chess Programming Wiki
// reference (https://www.chessprogramming.org/Perft_Results). A benchmark of
// an incorrect engine is worthless, so a wrong count is flagged loudly and the
// row is excluded from the NPS aggregate.
//
// Methodology: best-of-N wall-clock time per position (min time => max NPS),
// which discards OS-scheduling jitter and reports the machine's achievable
// throughput rather than an unlucky sample. N defaults to 1; pass a count as
// argv[1] for a steadier number (e.g. `bench 5`).
//
// Build note: only meaningful in the Release configuration (-O3 -march=native).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "perft.hpp"
#include "position.hpp"
#include "slider.hpp"

namespace {

struct Case {
    const char* name;
    const char* fen;
    int         depth;
    uint64_t    expected;  // reference node count (wiki), for self-verification
};

// Fixed set, chosen so each position does non-trivial work while the whole
// suite stays ~10-15s single-run on the reference machine.
constexpr Case CASES[] = {
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",              6, 119060324},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
    {"position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                            6, 11030083},
    {"position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",     5, 15833292},
    {"position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",            5, 89941194},
    {"position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4, 3894594},
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();

    int runs = 1;
    if (argc >= 2) {
        runs = std::atoi(argv[1]);
        if (runs < 1) runs = 1;
    }

    std::printf("NPS benchmark  (best of %d run%s per position, Release -O3 -march=native)\n\n",
                runs, runs == 1 ? "" : "s");
    std::printf("%-10s %5s %14s %10s %9s   %s\n",
                "position", "depth", "nodes", "time(s)", "Mnps", "ok");
    std::printf("--------------------------------------------------------------------\n");

    uint64_t total_nodes = 0;
    double   total_time  = 0.0;
    bool     all_ok      = true;

    for (const Case& c : CASES) {
        Position pos;
        if (!set_fen(pos, c.fen)) { std::printf("%-10s  BAD FEN\n", c.name); all_ok = false; continue; }

        double   best_time = 1e300;
        uint64_t nodes     = 0;
        for (int r = 0; r < runs; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            nodes = perft::perft(pos, c.depth);
            const double dt = seconds_since(t0);
            if (dt < best_time) best_time = dt;
        }

        const bool ok = (nodes == c.expected);
        all_ok = all_ok && ok;
        const double mnps = best_time > 0 ? static_cast<double>(nodes) / best_time / 1e6 : 0.0;

        std::printf("%-10s %5d %14llu %10.3f %9.2f   %s\n",
                    c.name, c.depth, static_cast<unsigned long long>(nodes),
                    best_time, mnps, ok ? "yes" : "NO <-- WRONG");

        if (ok) {  // only aggregate verified rows
            total_nodes += nodes;
            total_time  += best_time;
        }
    }

    std::printf("--------------------------------------------------------------------\n");
    const double agg_mnps = total_time > 0 ? static_cast<double>(total_nodes) / total_time / 1e6 : 0.0;
    std::printf("%-10s %5s %14llu %10.3f %9.2f\n",
                "TOTAL", "", static_cast<unsigned long long>(total_nodes), total_time, agg_mnps);

    if (!all_ok) {
        std::printf("\nWARNING: at least one node count did not match the reference.\n");
        return 1;
    }
    std::printf("\nAggregate: %.2f Mnps over %llu verified nodes.\n",
                agg_mnps, static_cast<unsigned long long>(total_nodes));
    return 0;
}
