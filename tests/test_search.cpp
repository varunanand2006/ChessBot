// Search gates. The search is deterministic, so we assert on concrete outcomes
// rather than exact scores:
//   A. Best move is always legal.
//   B. Forced mate-in-1 is found (both colors), with a mate-magnitude score.
//   C. A free hanging piece is captured (both colors).
// Low depths keep this in the fast test tier.

#include <cstdio>
#include <cstring>

#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"
#include "slider.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

constexpr int MATE = 99'999;

bool is_legal(Position& pos, Move m) {
    MoveList ml;
    movegen::generate_legal(pos, ml);
    for (const Move x : ml) if (x == m) return true;
    return false;
}

// Search from `fen` to `depth`, return the result; also assert legality.
search::SearchResult run(const char* fen, int depth) {
    Position pos;
    set_fen(pos, fen);
    search::new_game();
    search::history_add(pos.zobrist);
    const search::SearchResult r = search::find_best_move(pos, depth, /*verbose=*/false);
    if (!is_legal(pos, r.best_move)) fail("best move is not legal");
    return r;
}

void expect_move(const char* fen, int depth, const char* uci, const char* label) {
    const search::SearchResult r = run(fen, depth);
    const std::string got = move_to_uci(r.best_move);
    if (got != uci) {
        std::printf("[FAIL] %s: expected %s, got %s (score %+d)\n", label, uci, got.c_str(), r.score);
        ++g_failures;
    } else {
        std::printf("[ OK ] %-22s %s  score %+d\n", label, uci, r.score);
    }
}

}  // namespace

int main() {
    slider::init();

    // B. Mate in 1, both colors. Score magnitude must be mate-level.
    {
        const search::SearchResult w = run("k7/8/1K6/8/8/8/8/7R w - - 0 1", 3);
        if (move_to_uci(w.best_move) != "h1h8") fail("white mate-in-1: wrong move");
        if (w.score < MATE) fail("white mate-in-1: score not mate-level");
        else std::printf("[ OK ] white mate-in-1      h1h8  score %+d\n", w.score);

        const search::SearchResult b = run("7r/8/8/8/8/1k6/8/K7 b - - 0 1", 3);
        if (move_to_uci(b.best_move) != "h8h1") fail("black mate-in-1: wrong move");
        if (b.score > -MATE) fail("black mate-in-1: score not mate-level");
        else std::printf("[ OK ] black mate-in-1      h8h1  score %+d\n", b.score);
    }

    // C. Win a hanging queen, both colors.
    expect_move("4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1", 4, "d1d4", "white grabs queen");
    expect_move("3rk3/8/8/8/3Q4/8/8/4K3 b - - 0 1", 4, "d8d4", "black grabs queen");

    if (g_failures == 0) { std::printf("\nAll search tests passed.\n"); return 0; }
    std::printf("\n%d search test(s) failed.\n", g_failures);
    return 1;
}
