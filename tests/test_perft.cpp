// Step 8 gate: perft must exactly match the published reference counts.
//
// Reference values are from the Chess Programming Wiki "Perft Results" page:
//   https://www.chessprogramming.org/Perft_Results
// (fetched and transcribed 2026-08-06). These are NOT recalled from memory.
//
// Two tiers:
//   * default    — shallow depths for all six positions; fast (<2s), runs in
//                  the normal test suite.
//   * "deep" arg — the plan-mandated heavy depths (startpos->6, Kiwipete->5,
//                  and other large counts); ~25s, registered with a long
//                  timeout as a separate CTest entry.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "perft.hpp"
#include "position.hpp"
#include "slider.hpp"

namespace {

int g_failures = 0;

// FENs (Chess Programming Wiki).
constexpr const char* START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr const char* KIWI  = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
constexpr const char* POS3  = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
constexpr const char* POS4  = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
constexpr const char* POS5  = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
constexpr const char* POS6  = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

void check(const char* name, const char* fen, int depth, uint64_t expected) {
    Position pos;
    if (!set_fen(pos, fen)) { std::printf("[FAIL] %-14s bad FEN\n", name); ++g_failures; return; }
    const uint64_t got = perft::perft(pos, depth);
    if (got != expected) {
        std::printf("[FAIL] %-14s depth %d: got %llu, expected %llu\n",
                    name, depth, (unsigned long long)got, (unsigned long long)expected);
        ++g_failures;
    } else {
        std::printf("[ OK ] %-14s depth %d = %llu\n",
                    name, depth, (unsigned long long)got);
    }
}

void run_fast() {
    // Startpos 1..5.
    check("startpos", START, 1, 20);
    check("startpos", START, 2, 400);
    check("startpos", START, 3, 8902);
    check("startpos", START, 4, 197281);
    check("startpos", START, 5, 4865609);
    // Kiwipete 1..4.
    check("kiwipete", KIWI, 1, 48);
    check("kiwipete", KIWI, 2, 2039);
    check("kiwipete", KIWI, 3, 97862);
    check("kiwipete", KIWI, 4, 4085603);
    // Position 3, 1..5.
    check("position3", POS3, 1, 14);
    check("position3", POS3, 2, 191);
    check("position3", POS3, 3, 2812);
    check("position3", POS3, 4, 43238);
    check("position3", POS3, 5, 674624);
    // Position 4, 1..4.
    check("position4", POS4, 1, 6);
    check("position4", POS4, 2, 264);
    check("position4", POS4, 3, 9467);
    check("position4", POS4, 4, 422333);
    // Position 5, 1..4.
    check("position5", POS5, 1, 44);
    check("position5", POS5, 2, 1486);
    check("position5", POS5, 3, 62379);
    check("position5", POS5, 4, 2103487);
    // Position 6, 1..3.
    check("position6", POS6, 1, 46);
    check("position6", POS6, 2, 2079);
    check("position6", POS6, 3, 89890);
}

void run_deep() {
    // The plan-mandated heavy depths plus a few more large counts.
    check("startpos",  START, 6, 119060324);
    check("kiwipete",  KIWI,  5, 193690690);
    check("position3", POS3,  6, 11030083);
    check("position4", POS4,  5, 15833292);
    check("position5", POS5,  5, 89941194);
    check("position6", POS6,  4, 3894594);
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();

    const bool deep = (argc >= 2 && std::strcmp(argv[1], "deep") == 0);
    run_fast();
    if (deep) {
        std::printf("--- deep ---\n");
        run_deep();
    }

    if (g_failures == 0) {
        std::printf("\nAll perft checks passed%s.\n", deep ? " (incl. deep)" : "");
        return 0;
    }
    std::printf("\n%d perft check(s) FAILED.\n", g_failures);
    return 1;
}
