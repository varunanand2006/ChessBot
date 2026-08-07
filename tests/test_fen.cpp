// Step 2 gate: FEN must round-trip byte-exactly on startpos, Kiwipete, and a
// position with an en-passant target square.
//
// Round-trip == set_fen(parse) followed by to_fen(serialize) reproduces the
// original string exactly. This only holds for FENs already in canonical form
// (all six fields present, castling in KQkq order) — all test strings are.

#include <cstdio>
#include <string>
#include <string_view>

#include "position.hpp"

namespace {

int g_failures = 0;

void check_roundtrip(std::string_view name, std::string_view fen) {
    Position pos;
    if (!set_fen(pos, fen)) {
        std::printf("[FAIL] %-12s set_fen rejected: %.*s\n",
                    name.data(), static_cast<int>(fen.size()), fen.data());
        ++g_failures;
        return;
    }
    const std::string out = to_fen(pos);
    if (out != fen) {
        std::printf("[FAIL] %-12s round-trip mismatch\n  in : %.*s\n  out: %s\n",
                    name.data(), static_cast<int>(fen.size()), fen.data(), out.c_str());
        ++g_failures;
        return;
    }
    std::printf("[ OK ] %-12s %s\n", name.data(), out.c_str());
}

// Spot-check a specific parsed field, so a round-trip that is self-consistent
// but wrong (e.g. mirrored board) can't silently pass.
void check_true(std::string_view name, bool cond) {
    if (!cond) {
        std::printf("[FAIL] %s\n", name.data());
        ++g_failures;
    } else {
        std::printf("[ OK ] %s\n", name.data());
    }
}

}  // namespace

int main() {
    // The three gate positions.
    constexpr std::string_view startpos =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    constexpr std::string_view kiwipete =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    constexpr std::string_view en_passant =
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";

    check_roundtrip("startpos", startpos);
    check_roundtrip("kiwipete", kiwipete);
    check_roundtrip("en_passant", en_passant);

    // Targeted invariants on the en-passant position: side = Black, ep = e3.
    Position ep;
    set_fen(ep, en_passant);
    check_true("ep side to move is Black", ep.side_to_move == Color::Black);
    check_true("ep target square is e3",   ep.ep_square == Square::E3);

    // Startpos sanity: known piece placements from the LERF mapping.
    Position sp;
    set_fen(sp, startpos);
    check_true("a1 is a white rook",
               sp.piece_type_on(Square::A1) == PieceType::Rook &&
               sp.color_on(Square::A1) == Color::White);
    check_true("e8 is a black king",
               sp.piece_type_on(Square::E8) == PieceType::King &&
               sp.color_on(Square::E8) == Color::Black);
    check_true("e4 is empty", sp.piece_type_on(Square::E4) == PieceType::None);

    // Reject malformed input rather than crashing.
    Position junk;
    check_true("rejects garbage FEN", !set_fen(junk, "not a fen"));
    check_true("rejects short rank",   !set_fen(junk, "8/8/8/8/8/8/8 w - - 0 1"));

    if (g_failures == 0) {
        std::printf("\nAll FEN tests passed.\n");
        return 0;
    }
    std::printf("\n%d FEN test(s) failed.\n", g_failures);
    return 1;
}
