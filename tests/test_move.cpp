// Step 5 gate: encode/decode round-trips for all move types, including all
// four promotions (and promotion-captures).

#include <cstdio>
#include <string_view>

#include "move.hpp"
#include "types.hpp"

namespace {

int g_failures = 0;

void expect(std::string_view name, bool cond) {
    if (!cond) { std::printf("[FAIL] %s\n", name.data()); ++g_failures; }
    else       { std::printf("[ OK ] %s\n", name.data()); }
}

// Encode a move then decode every field back and compare.
void roundtrip(std::string_view name, Square from, Square to, MoveFlag flag,
               bool cap, bool promo, PieceType promo_pt) {
    const Move m = make_move(from, to, flag);
    bool ok = move_from(m) == from &&
              move_to(m) == to &&
              move_flag(m) == flag &&
              is_capture(m) == cap &&
              is_promotion(m) == promo;
    if (promo) ok = ok && promotion_type(m) == promo_pt;
    if (!ok) {
        std::printf("[FAIL] %-22s got from=%d to=%d flag=%d cap=%d promo=%d\n",
                    name.data(), sq_index(move_from(m)), sq_index(move_to(m)),
                    static_cast<int>(move_flag(m)), is_capture(m), is_promotion(m));
        ++g_failures;
    } else {
        std::printf("[ OK ] %-22s %s\n", name.data(), move_to_uci(m).c_str());
    }
}

}  // namespace

// Compile-time proof the codec is constexpr and correct on a promotion capture.
static_assert(move_from(make_move(Square::E7, Square::F8, MoveFlag::PromoQueenCapture)) == Square::E7);
static_assert(move_to(make_move(Square::E7, Square::F8, MoveFlag::PromoQueenCapture)) == Square::F8);
static_assert(promotion_type(make_move(Square::E7, Square::F8, MoveFlag::PromoQueenCapture)) == PieceType::Queen);
static_assert(is_capture(make_move(Square::E7, Square::F8, MoveFlag::PromoQueenCapture)));

int main() {
    // Non-promotion move types.
    roundtrip("quiet",        Square::E2, Square::E3, MoveFlag::Quiet,       false, false, PieceType::None);
    roundtrip("double push",  Square::E2, Square::E4, MoveFlag::DoublePush,  false, false, PieceType::None);
    roundtrip("capture",      Square::D4, Square::E5, MoveFlag::Capture,     true,  false, PieceType::None);
    // En passant sets the capture bit (flag 5 = 0b0101) — it IS a capture.
    roundtrip("en passant",   Square::D5, Square::E6, MoveFlag::EnPassant,   true,  false, PieceType::None);
    roundtrip("king castle",  Square::E1, Square::G1, MoveFlag::KingCastle,  false, false, PieceType::None);
    roundtrip("queen castle", Square::E1, Square::C1, MoveFlag::QueenCastle, false, false, PieceType::None);

    // All four quiet promotions.
    roundtrip("promo N",      Square::A7, Square::A8, MoveFlag::PromoKnight, false, true, PieceType::Knight);
    roundtrip("promo B",      Square::A7, Square::A8, MoveFlag::PromoBishop, false, true, PieceType::Bishop);
    roundtrip("promo R",      Square::A7, Square::A8, MoveFlag::PromoRook,   false, true, PieceType::Rook);
    roundtrip("promo Q",      Square::A7, Square::A8, MoveFlag::PromoQueen,  false, true, PieceType::Queen);

    // All four promotion-captures.
    roundtrip("promo N cap",  Square::B7, Square::A8, MoveFlag::PromoKnightCapture, true, true, PieceType::Knight);
    roundtrip("promo B cap",  Square::B7, Square::A8, MoveFlag::PromoBishopCapture, true, true, PieceType::Bishop);
    roundtrip("promo R cap",  Square::B7, Square::A8, MoveFlag::PromoRookCapture,   true, true, PieceType::Rook);
    roundtrip("promo Q cap",  Square::B7, Square::A8, MoveFlag::PromoQueenCapture,  true, true, PieceType::Queen);

    // Extreme squares to exercise the full 6-bit range.
    roundtrip("corner a1->h8", Square::A1, Square::H8, MoveFlag::Quiet, false, false, PieceType::None);
    roundtrip("corner h8->a1", Square::H8, Square::A1, MoveFlag::Quiet, false, false, PieceType::None);

    // UCI formatting spot checks.
    expect("uci e2e4",  move_to_uci(make_move(Square::E2, Square::E4, MoveFlag::DoublePush)) == "e2e4");
    expect("uci e7e8q", move_to_uci(make_move(Square::E7, Square::E8, MoveFlag::PromoQueen)) == "e7e8q");
    expect("uci a7a8n", move_to_uci(make_move(Square::A7, Square::A8, MoveFlag::PromoKnight)) == "a7a8n");

    // MoveList: add / count / iterate.
    MoveList list;
    list.add(make_move(Square::E2, Square::E4, MoveFlag::DoublePush));
    list.add(make_move(Square::G1, Square::F3, MoveFlag::Quiet));
    expect("movelist count", list.size() == 2);
    int seen = 0;
    for (Move mv : list) { (void)mv; ++seen; }
    expect("movelist iterate", seen == 2);
    expect("movelist index", list[0] == make_move(Square::E2, Square::E4, MoveFlag::DoublePush));

    if (g_failures == 0) { std::printf("\nAll move-encoding tests passed.\n"); return 0; }
    std::printf("\n%d move-encoding test(s) failed.\n", g_failures);
    return 1;
}
