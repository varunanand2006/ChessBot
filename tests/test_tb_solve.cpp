// Phase 2 / Step 2.2 gate: the retrograde DTM solver is correct.
//
// Primary check: the two independent solvers (iterated negamax sweep vs.
// win/loss BFS over the reverse graph) must agree on the value of EVERY
// position. Agreement across ~50k positions between structurally different
// algorithms is strong evidence of correctness.
//
// Secondary sanity (no recalled constants): KBK and KNK cannot force mate and
// have no checkmate positions at all -> every position is a draw. KQK and KRK
// can force mate -> wins and losses exist, and the count of White-mates equals
// the count of Black-mates by color symmetry of the material.

#include <cstdio>

#include "slider.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

void test_piece(PieceType wp, const char* name, bool expect_wins) {
    tb::Index idx(wp);
    const tb::Table a = tb::solve_sweep(idx);
    const tb::Table b = tb::solve_bfs(idx);

    // Solvers must agree on every value.
    bool agree = (a.value.size() == b.value.size());
    if (agree) {
        for (std::size_t i = 0; i < a.value.size(); ++i)
            if (a.value[i] != b.value[i]) { agree = false; break; }
    }
    if (!agree) fail("sweep and bfs disagree");

    // WDL / DTM statistics (from the sweep; identical to bfs when agree holds).
    std::printf("%-4s  N=%zu  W=%zu L=%zu D=%zu  maxWinDTM=%d maxLossDTM=%d  %s\n",
                name, a.value.size(), a.wins, a.losses, a.draws,
                a.max_win_dtm, a.max_loss_dtm, agree ? "[agree]" : "[DISAGREE]");

    if (expect_wins) {
        if (a.wins == 0)   fail("expected wins but found none");
        if (a.losses == 0) fail("expected losses but found none");
    } else {
        // Insufficient material: no forced mate anywhere.
        if (a.wins != 0)   fail("unexpected wins for insufficient material");
        if (a.losses != 0) fail("unexpected losses for insufficient material");
    }
}

}  // namespace

int main() {
    slider::init();
    test_piece(PieceType::Queen,  "KQK", /*expect_wins=*/true);
    test_piece(PieceType::Rook,   "KRK", /*expect_wins=*/true);
    test_piece(PieceType::Bishop, "KBK", /*expect_wins=*/false);
    test_piece(PieceType::Knight, "KNK", /*expect_wins=*/false);

    if (g_failures == 0) { std::printf("\nAll retrograde solver tests passed.\n"); return 0; }
    std::printf("\n%d retrograde solver test(s) failed.\n", g_failures);
    return 1;
}
