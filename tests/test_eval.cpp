// Evaluation gates. The eval has no external reference table to match, so we
// test its structural invariants instead:
//   A. The start position is exactly 0 (perfectly symmetric material + PSTs).
//   B. Color-swap + vertical-mirror antisymmetry: mirroring a position and
//      swapping colors must negate the score. This exercises material, every
//      piece-square table, the check term, and the king-distance term at once —
//      a bug in any of them breaks the identity.
//   C. Material dominance: being up a rook is clearly positive for that side.

#include <cstdint>
#include <cstdio>

#include "eval.hpp"
#include "position.hpp"
#include "slider.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }

// Vertical mirror (rank flip = byteswap on a LERF board) plus color swap. Eval
// must be antisymmetric under this: evaluate(mirror(p)) == -evaluate(p).
Position mirror(const Position& p) {
    Position m;
    m.clear();
    for (int t = 0; t < 6; ++t) m.by_type[t] = __builtin_bswap64(p.by_type[t]);
    m.by_color[color_index(Color::White)] = __builtin_bswap64(p.by_color[color_index(Color::Black)]);
    m.by_color[color_index(Color::Black)] = __builtin_bswap64(p.by_color[color_index(Color::White)]);
    m.side_to_move = ~p.side_to_move;
    return m;
}

void check_antisymmetric(const char* fen) {
    Position p;
    if (!set_fen(p, fen)) { fail("bad FEN in test"); return; }
    const int a = eval::evaluate(p);
    const int b = eval::evaluate(mirror(p));
    if (a != -b) {
        std::printf("[FAIL] antisymmetry: eval=%d  mirror=%d  (%s)\n", a, b, fen);
        ++g_failures;
    }
}

}  // namespace

int main() {
    slider::init();  // is_attacked() (check term) needs the magic tables

    // A. Start position is dead level.
    Position start;
    set_fen(start, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    if (eval::evaluate(start) != 0) fail("start position eval != 0");

    // B. Antisymmetry across a spread of positions (opening, middlegame,
    // tactical, and endgames that trigger the king-distance term).
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/8/8/4k3/8/8/2K5/6Q1 w - - 0 1",   // endgame: king-distance term active
        "8/2k5/8/8/8/5K2/8/7R b - - 0 1",     // endgame, Black to move
        "4k3/8/8/8/3q4/8/8/3RK3 w - - 0 1",   // tactical, in-check terms in play
    };
    for (const char* f : fens) check_antisymmetric(f);

    // C. Material dominance: White up a rook is clearly better for White; the
    // mirrored (Black up a rook) is the negation.
    Position up_rook;
    set_fen(up_rook, "4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    if (eval::evaluate(up_rook) <= 300) fail("up a rook should be clearly positive");

    if (g_failures == 0) { std::printf("All eval tests passed.\n"); return 0; }
    std::printf("\n%d eval test(s) failed.\n", g_failures);
    return 1;
}
