// Move encoding: a move is a single uint16_t. No class, no object — the whole
// engine passes moves by value in a register.
//
// Bit layout (16 bits):
//   bits  0- 5 : from square (0..63)
//   bits  6-11 : to   square (0..63)
//   bits 12-15 : flag (4-bit code, see MoveFlag)
//
// The 4-bit flag is not arbitrary — it is the standard structured encoding:
//   bit 3 (0x8) : promotion
//   bit 2 (0x4) : capture
//   bits 1-0    : sub-type. For promotions these two bits select the piece
//                 (0=N,1=B,2=R,3=Q), matching PieceType's Knight..Queen order
//                 so promotion_type() is a single add, not a lookup.
// This lets is_capture()/is_promotion() be one masked test each with no board
// access — the property is baked into the move at generation time, which is
// exactly what move ordering (MVV-LVA) will want later without a piece lookup.

#pragma once

#include <cstdint>
#include <string>

#include "types.hpp"

using Move = uint16_t;

// The all-zero move (from==to==a1, quiet) can never be produced by real
// generation, so it doubles as a "no move" sentinel.
constexpr Move MOVE_NONE = 0;

enum class MoveFlag : uint16_t {
    Quiet              = 0,
    DoublePush         = 1,   // pawn two-square advance (sets en-passant target)
    KingCastle         = 2,
    QueenCastle        = 3,
    Capture            = 4,
    EnPassant          = 5,   // 0b0101: capture bit set (en passant is a capture)
    // 6,7 unused
    PromoKnight        = 8,
    PromoBishop        = 9,
    PromoRook          = 10,
    PromoQueen         = 11,
    PromoKnightCapture = 12,
    PromoBishopCapture = 13,
    PromoRookCapture   = 14,
    PromoQueenCapture  = 15,
};

namespace detail {
constexpr uint16_t FROM_SHIFT = 0;
constexpr uint16_t TO_SHIFT   = 6;
constexpr uint16_t FLAG_SHIFT = 12;
constexpr uint16_t SQ_MASK    = 0x3F;  // 6 bits
constexpr uint16_t FLAG_MASK  = 0x0F;  // 4 bits
constexpr uint16_t PROMO_BIT  = 0x8;
constexpr uint16_t CAPTURE_BIT = 0x4;
}  // namespace detail

CH_HD constexpr Move make_move(Square from, Square to, MoveFlag flag = MoveFlag::Quiet) {
    return static_cast<Move>(
        (static_cast<uint16_t>(sq_index(from)) << detail::FROM_SHIFT) |
        (static_cast<uint16_t>(sq_index(to))   << detail::TO_SHIFT) |
        (static_cast<uint16_t>(flag)           << detail::FLAG_SHIFT));
}

CH_HD constexpr Square move_from(Move m) {
    return static_cast<Square>((m >> detail::FROM_SHIFT) & detail::SQ_MASK);
}
CH_HD constexpr Square move_to(Move m) {
    return static_cast<Square>((m >> detail::TO_SHIFT) & detail::SQ_MASK);
}
CH_HD constexpr MoveFlag move_flag(Move m) {
    return static_cast<MoveFlag>((m >> detail::FLAG_SHIFT) & detail::FLAG_MASK);
}

CH_HD constexpr bool is_capture(Move m) {
    return ((m >> detail::FLAG_SHIFT) & detail::CAPTURE_BIT) != 0;
}
CH_HD constexpr bool is_promotion(Move m) {
    return ((m >> detail::FLAG_SHIFT) & detail::PROMO_BIT) != 0;
}
CH_HD constexpr bool is_en_passant(Move m) {
    return move_flag(m) == MoveFlag::EnPassant;
}

// Valid only when is_promotion(m). bits1-0 of the flag select the piece:
// 0->Knight, 1->Bishop, 2->Rook, 3->Queen. PieceType orders these 1..4, so the
// promoted type is just Knight + those two bits.
CH_HD constexpr PieceType promotion_type(Move m) {
    const uint16_t sub = (m >> detail::FLAG_SHIFT) & 0x3;
    return static_cast<PieceType>(type_index(PieceType::Knight) + sub);
}

// UCI text form, e.g. "e2e4", "e7e8q". Debug/IO helper — never on a hot path,
// so std::string is fine here.
inline std::string move_to_uci(Move m) {
    if (m == MOVE_NONE) return "0000";
    const Square from = move_from(m);
    const Square to   = move_to(m);
    std::string s;
    s += static_cast<char>('a' + file_of(from));
    s += static_cast<char>('1' + rank_of(from));
    s += static_cast<char>('a' + file_of(to));
    s += static_cast<char>('1' + rank_of(to));
    if (is_promotion(m)) {
        const char pc[] = {'n', 'b', 'r', 'q'};
        s += pc[(m >> detail::FLAG_SHIFT) & 0x3];
    }
    return s;
}

// Fixed-capacity move list: a flat array plus a count. No heap — this is
// stack-allocated at every search node. 256 slots: the maximum number of legal
// moves in any legal chess position is 218, so 256 is a safe power-of-two cap
// with headroom for pseudo-legal over-generation before the legality filter.
struct MoveList {
    Move moves[256];
    int  count = 0;

    CH_HD void add(Move m) { moves[count++] = m; }
    CH_HD void clear() { count = 0; }
    CH_HD int  size() const { return count; }
    CH_HD Move operator[](int i) const { return moves[i]; }

    // Range-for support.
    CH_HD const Move* begin() const { return moves; }
    CH_HD const Move* end() const { return moves + count; }
};
