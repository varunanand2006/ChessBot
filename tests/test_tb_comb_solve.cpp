// Step 5.4 gate (correctness half): the memoryless combinatorial sweep
// (solve_sweep_comb over tb::CombIndex) produces the SAME game-theoretic result
// as the already-verified dense solver (solve_sweep over tb::Index) — position
// by position, exact WDL and exact DTM.
//
// The dense solver is the oracle (it agreed with an independent BFS solver AND
// matched the Lichess/Gaviota API on 133/133 sampled positions). So for every
// legal position, comb_value[encode_comb(pos)] must equal dense_value[encode_dense(pos)].
// That validates the whole combinatorial path (index + Position glue + sweep)
// on materials small enough to also solve densely, before we trust it on 5-man.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "slider.hpp"
#include "tb_comb_index.hpp"
#include "tb_index.hpp"
#include "tb_solve.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

void check(std::vector<tb::Piece> extras, const char* name) {
    tb::Index     dense(extras);
    tb::CombIndex comb(extras);

    tb::Table dt = tb::solve_sweep(dense);
    tb::Table ct = tb::solve_sweep_comb(comb);

    // Walk every dense (legal, canonical) position; its comb value must match.
    long long compared = 0;
    for (std::size_t d = 0; d < dense.size(); ++d) {
        Position pos = dense.decode(d);
        const std::size_t c = comb.encode(pos);
        if (c >= comb.size()) { fail("comb index out of range"); return; }
        if (dt.value[d] != ct.value[c]) { fail("comb value != dense value"); return; }
        ++compared;
    }

    // Deepest mates must match exactly (a max is unaffected by duplication).
    if (dt.max_win_dtm != ct.max_win_dtm)  fail("max win DTM differs");
    if (dt.max_loss_dtm != ct.max_loss_dtm) fail("max loss DTM differs");
    // The comb table is intentionally LOOSER: on-axis king configs get a few
    // redundant indices (the KingTable stabilizer over-count), so comb has >=
    // the dense win/loss counts. Equality is NOT expected; the >= direction is
    // the meaningful invariant (never fewer — that would mean a lost position).
    if (ct.wins < dt.wins || ct.losses < dt.losses)
        fail("comb has FEWER wins/losses than dense (positions lost)");

    std::printf("[ OK ] %-6s  positions=%lld  maxWinDTM=%d  W(dense/comb)=%zu/%zu  L=%zu/%zu  passes=%d\n",
                name, compared, ct.max_win_dtm, dt.wins, ct.wins, dt.losses, ct.losses, ct.passes);
}

}  // namespace

int main(int argc, char** argv) {
    slider::init();
    const bool slow = (argc >= 2 && std::strcmp(argv[1], "slow") == 0);

    using PT = PieceType;
    const Color W = Color::White, B = Color::Black;

    // Fast: 3-man — both solvers run in well under a second.
    check({{W, PT::Rook}},  "KRK");
    check({{W, PT::Queen}}, "KQK");

    if (slow) {
        std::printf("--- slow (4-man) ---\n");
        check({{W, PT::Queen}, {B, PT::Rook}}, "KQKR");  // the mate-in-35 table
    }

    if (g_failures == 0) {
        std::printf("\nAll comb-solve tests passed (match dense solver)%s.\n", slow ? " incl. 4-man" : "");
        return 0;
    }
    std::printf("\n%d comb-solve test(s) failed.\n", g_failures);
    return 1;
}
