// Step 6 gate: for every move from a set of test positions, make() followed by
// unmake() restores byte-identical board state AND an identical Zobrist key.
//
// Legal-move generation doesn't exist until Step 7, so instead of "every legal
// move" this uses crafted positions where each special move type (double push,
// capture, en passant, both castles, all promotions incl. promo-capture) is
// available and its flag is unambiguous. Reversibility does not depend on a
// move being legal, only structurally valid, so this fully exercises make/
// unmake. Step 8 (perft) then re-verifies make/unmake across millions of real
// moves as a side effect of correct node counts.

#include <cstdio>
#include <string_view>

#include "move.hpp"
#include "position.hpp"

namespace {

int g_failures = 0;

// Field-by-field equality (avoids memcmp reading struct padding).
bool same(const Position& a, const Position& b) {
    for (int i = 0; i < NUM_PIECE_TYPES; ++i)
        if (a.by_type[i] != b.by_type[i]) return false;
    if (a.by_color[0] != b.by_color[0] || a.by_color[1] != b.by_color[1]) return false;
    return a.side_to_move == b.side_to_move && a.castling == b.castling &&
           a.ep_square == b.ep_square && a.halfmove_clock == b.halfmove_clock &&
           a.fullmove_number == b.fullmove_number && a.zobrist == b.zobrist;
}

void fail(std::string_view what) { std::printf("[FAIL] %s\n", what.data()); ++g_failures; }

// make() then check the incremental key matches a recompute; then unmake() and
// check we are byte-identical to where we started.
void check_move(std::string_view name, std::string_view fen, Move m) {
    Position pos;
    if (!set_fen(pos, fen)) { fail(name); std::printf("       set_fen failed\n"); return; }

    const Position before = pos;
    StateInfo st;
    make_move(pos, m, st);

    bool ok = true;
    if (pos.zobrist != compute_zobrist(pos)) {
        std::printf("[FAIL] %-16s incremental key != recomputed key after make (%s)\n",
                    name.data(), move_to_uci(m).c_str());
        ++g_failures;
        ok = false;
    }

    unmake_move(pos, m, st);
    if (!same(pos, before)) {
        std::printf("[FAIL] %-16s state not restored after unmake (%s)\n",
                    name.data(), move_to_uci(m).c_str());
        std::printf("       fen in : %s\n       fen out: %s\n",
                    to_fen(before).c_str(), to_fen(pos).c_str());
        ++g_failures;
        ok = false;
    }
    if (ok) std::printf("[ OK ] %-16s %s\n", name.data(), move_to_uci(m).c_str());
}

}  // namespace

int main() {
    // --- Single-move reversibility across every move type ---

    // Quiet + double push.
    constexpr std::string_view start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    check_move("quiet",       start, make_move(Square::G1, Square::F3, MoveFlag::Quiet));
    check_move("double push", start, make_move(Square::E2, Square::E4, MoveFlag::DoublePush));

    // Castling — clean board with only kings and rooks, rights available.
    constexpr std::string_view castle_w = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    constexpr std::string_view castle_b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";
    check_move("O-O white",   castle_w, make_move(Square::E1, Square::G1, MoveFlag::KingCastle));
    check_move("O-O-O white", castle_w, make_move(Square::E1, Square::C1, MoveFlag::QueenCastle));
    check_move("O-O black",   castle_b, make_move(Square::E8, Square::G8, MoveFlag::KingCastle));
    check_move("O-O-O black", castle_b, make_move(Square::E8, Square::C8, MoveFlag::QueenCastle));

    // Capture (rook captures rook on a8, which must also clear black's O-O-O).
    constexpr std::string_view rxr = "r3k3/8/8/8/8/8/8/R3K3 w Qq - 0 1";
    check_move("capture Rxr", "r3k3/R7/8/8/8/8/8/4K3 w q - 0 1",
               make_move(Square::A7, Square::A8, MoveFlag::Capture));

    // En passant (white e5 x d6, black pawn on d5, ep target d6).
    constexpr std::string_view ep = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
    check_move("en passant",  ep, make_move(Square::E5, Square::D6, MoveFlag::EnPassant));

    // All four quiet promotions (white pawn a7 -> a8, empty a8).
    constexpr std::string_view promo = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1";
    check_move("promo N", promo, make_move(Square::A7, Square::A8, MoveFlag::PromoKnight));
    check_move("promo B", promo, make_move(Square::A7, Square::A8, MoveFlag::PromoBishop));
    check_move("promo R", promo, make_move(Square::A7, Square::A8, MoveFlag::PromoRook));
    check_move("promo Q", promo, make_move(Square::A7, Square::A8, MoveFlag::PromoQueen));

    // All four promotion-captures (white pawn a7 x b8, black knight on b8).
    constexpr std::string_view promo_cap = "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1";
    check_move("promo N cap", promo_cap, make_move(Square::A7, Square::B8, MoveFlag::PromoKnightCapture));
    check_move("promo B cap", promo_cap, make_move(Square::A7, Square::B8, MoveFlag::PromoBishopCapture));
    check_move("promo R cap", promo_cap, make_move(Square::A7, Square::B8, MoveFlag::PromoRookCapture));
    check_move("promo Q cap", promo_cap, make_move(Square::A7, Square::B8, MoveFlag::PromoQueenCapture));
    (void)rxr;

    // --- Multi-move sequence: exercise the StateInfo stack ---
    // Push a run of moves, then unmake them in reverse; end state must equal
    // the start byte-for-byte. Real opening moves (all quiet/double push).
    {
        Position pos;
        set_fen(pos, start);
        const Position origin = pos;

        const Move seq[] = {
            make_move(Square::E2, Square::E4, MoveFlag::DoublePush),
            make_move(Square::E7, Square::E5, MoveFlag::DoublePush),
            make_move(Square::G1, Square::F3, MoveFlag::Quiet),
            make_move(Square::B8, Square::C6, MoveFlag::Quiet),
            make_move(Square::F1, Square::B5, MoveFlag::Quiet),  // Ruy Lopez
        };
        constexpr int N = 5;
        StateInfo stack[N];

        bool key_ok = true;
        for (int i = 0; i < N; ++i) {
            make_move(pos, seq[i], stack[i]);
            if (pos.zobrist != compute_zobrist(pos)) key_ok = false;
        }
        for (int i = N - 1; i >= 0; --i) unmake_move(pos, seq[i], stack[i]);

        if (!key_ok) fail("sequence keys consistent");
        else         std::printf("[ OK ] sequence keys consistent\n");
        if (!same(pos, origin)) fail("sequence restores start");
        else                    std::printf("[ OK ] sequence restores start\n");
    }

    if (g_failures == 0) { std::printf("\nAll make/unmake tests passed.\n"); return 0; }
    std::printf("\n%d make/unmake test(s) failed.\n", g_failures);
    return 1;
}
