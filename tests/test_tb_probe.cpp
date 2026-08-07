// Tablebase-probing gates.
//   A. probe() returns the right WDL for known materials (both sides to move),
//      and correctly declines unsupported materials (pawns).
//   B. With probing on, the search returns an exact result and PRUNES: an
//      endgame search visits far fewer nodes than the heuristic-only search,
//      while still choosing a winning move.
// Only 3-man tables are used so the tables build near-instantly (fast tier).

#include <cstdio>

#include "position.hpp"
#include "search.hpp"
#include "slider.hpp"
#include "tb_probe.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

void expect_wdl(const char* fen, int wdl, const char* label) {
    Position pos;
    set_fen(pos, fen);
    const tb::ProbeResult r = tb::probe(pos);
    if (!r.found || r.wdl != wdl) {
        std::printf("[FAIL] %s: found=%d wdl=%d (want %d)\n", label, r.found, r.wdl, wdl);
        ++g_failures;
    } else {
        std::printf("[ OK ] %-28s wdl %+d  dtm %d\n", label, r.wdl, r.dtm);
    }
}

}  // namespace

int main() {
    slider::init();

    // A. WDL for known materials.
    expect_wdl("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", +1, "KQK, strong side to move");
    expect_wdl("4k3/8/8/8/8/8/8/3QK3 b - - 0 1", -1, "KQK, lone king to move");
    expect_wdl("4k3/8/8/8/8/8/8/3RK3 w - - 0 1", +1, "KRK, strong side to move");
    expect_wdl("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", 0, "KBK, draw (insufficient)");
    expect_wdl("4k3/8/8/8/8/8/8/4K3 w - - 0 1",   0, "KK, draw");

    // Unsupported material (a pawn) must decline so the search uses the heuristic.
    {
        Position pos;
        set_fen(pos, "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
        if (tb::probe(pos).found) fail("pawn material should be unsupported");
        else std::printf("[ OK ] %-28s declined (heuristic fallback)\n", "KPK (has a pawn)");
    }

    // B. Probing prunes the search and still wins. KRK, White to move, winning.
    {
        const char* fen = "8/8/8/4k3/8/8/8/R3K3 w - - 0 1";
        Position pos;

        set_fen(pos, fen);
        search::set_use_tablebase(false);
        search::new_game(); search::history_add(pos.zobrist);
        const search::SearchResult heur = search::find_best_move(pos, 8);

        set_fen(pos, fen);
        search::set_use_tablebase(true);
        search::new_game(); search::history_add(pos.zobrist);
        const search::SearchResult tbon = search::find_best_move(pos, 8);
        search::set_use_tablebase(false);  // leave global state clean

        std::printf("[info] KRK depth 8: heuristic %llu nodes, tablebase %llu nodes\n",
                    static_cast<unsigned long long>(heur.nodes),
                    static_cast<unsigned long long>(tbon.nodes));
        if (tbon.nodes >= heur.nodes) fail("tablebase probing did not reduce nodes");
        if (heur.score <= 0)          fail("heuristic search: KRK should be winning for White");
        if (tbon.score <= 0)          fail("tablebase search: KRK should be winning for White");
    }

    if (g_failures == 0) { std::printf("\nAll tablebase-probe tests passed.\n"); return 0; }
    std::printf("\n%d tablebase-probe test(s) failed.\n", g_failures);
    return 1;
}
