#include "eval.hpp"

#include <cstdlib>  // std::abs

#include "bitboard.hpp"
#include "movegen.hpp"

namespace eval {

namespace {

// Blended piece values and piece-square tables (LERF, White's perspective).
// Generated from the Python constants — see the file header. Kept in an
// anonymous namespace so they are internal to this translation unit.
#include "eval_tables.inc"

// Black reuses a White table by mirroring the square vertically (rank flip),
// which on a LERF board is XOR 56 — a1<->a8. This is why only one orientation
// of each table is stored.
constexpr int flip_sq(int sq) { return sq ^ 56; }

}  // namespace

int piece_value(PieceType pt) {
    const int t = type_index(pt);
    return (t >= 0 && t < 6) ? PIECE_VALUE[t] : 0;
}

int evaluate(const Position& pos) {
    // Total non-king material decides the king table (middlegame vs endgame)
    // and gates the endgame-only king-distance term. Computed in full up front:
    // the Python original folded this into its single square-scan and so read a
    // PARTIAL sum when it reached each king (an iteration-order artifact). Doing
    // it correctly here is the one deliberate deviation from the port — it makes
    // the endgame/middlegame boundary well-defined instead of king-placement
    // dependent.
    int total_material = 0;
    for (int t = type_index(PieceType::Pawn); t <= type_index(PieceType::Queen); ++t)
        total_material += popcount(pos.by_type[t]) * PIECE_VALUE[t];

    const int16_t* king_pst = (total_material < 1500) ? PST_KING_END : PST_KING;
    const int16_t* pst[6] = {PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, king_pst};

    int material_score   = 0;
    int positional_score = 0;
    for (int t = 0; t < 6; ++t) {
        const int value = PIECE_VALUE[t];
        Bitboard w = pos.by_type[t] & pos.by_color[color_index(Color::White)];
        while (w) {
            const Square s = pop_lsb(w);
            material_score   += value;
            positional_score += pst[t][sq_index(s)];
        }
        Bitboard b = pos.by_type[t] & pos.by_color[color_index(Color::Black)];
        while (b) {
            const Square s = pop_lsb(b);
            material_score   -= value;
            positional_score -= pst[t][flip_sq(sq_index(s))];
        }
    }

    // In-check penalty: bad for the checked side. From White's view, White in
    // check subtracts, Black in check adds.
    const Square wk = movegen::king_square(pos, Color::White);
    const Square bk = movegen::king_square(pos, Color::Black);
    int check_penalty = 0;
    if (movegen::is_attacked(pos, wk, Color::Black)) check_penalty -= 50;
    if (movegen::is_attacked(pos, bk, Color::White)) check_penalty += 50;

    // Endgame king-distance: the side that is ahead wants the kings close (to
    // corner the lone king). Manhattan distance, 5 cp per step.
    int king_distance_score = 0;
    if (total_material < 2000 && material_score != 0) {
        const int dist = std::abs(rank_of(wk) - rank_of(bk)) +
                         std::abs(file_of(wk) - file_of(bk));
        king_distance_score = (material_score > 0) ? -dist * 5 : dist * 5;
    }

    return material_score + positional_score + check_penalty + king_distance_score;
}

}  // namespace eval
