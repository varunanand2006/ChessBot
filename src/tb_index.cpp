#include "tb_index.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "slider.hpp"

namespace tb {

namespace {

// The 8 elements of D4 as square permutations, indexed [transform][square].
// Built once. Each transform maps (file,rank) as follows:
//   0 identity        (f, r)
//   1 mirror file     (7-f, r)
//   2 mirror rank     (f, 7-r)
//   3 rotate 180      (7-f, 7-r)
//   4 main diagonal   (r, f)
//   5 anti-diagonal   (7-r, 7-f)
//   6 rotate 90       (r, 7-f)
//   7 rotate 270      (7-r, f)
// These are precisely the symmetries that leave a pawnless board equivalent.
int SYM[8][64];
bool g_sym_ready = false;

int sq_of(int file, int rank) { return rank * 8 + file; }

void init_sym() {
    if (g_sym_ready) return;
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        SYM[0][s] = sq_of(f, r);
        SYM[1][s] = sq_of(7 - f, r);
        SYM[2][s] = sq_of(f, 7 - r);
        SYM[3][s] = sq_of(7 - f, 7 - r);
        SYM[4][s] = sq_of(r, f);
        SYM[5][s] = sq_of(7 - r, 7 - f);
        SYM[6][s] = sq_of(r, 7 - f);
        SYM[7][s] = sq_of(7 - r, f);
    }
    g_sym_ready = true;
}

// Attack set of the single white piece from square pc under occupancy occ.
Bitboard white_piece_attacks(PieceType wp, int pc, Bitboard occ) {
    const Square s = static_cast<Square>(pc);
    switch (wp) {
        case PieceType::Rook:   return slider::rook_attacks(s, occ);
        case PieceType::Bishop: return slider::bishop_attacks(s, occ);
        case PieceType::Queen:  return slider::queen_attacks(s, occ);
        case PieceType::Knight: return attacks::knight(s);
        default:                return 0;
    }
}

}  // namespace

bool Index::legal(PieceType wp, int wk, int pc, int bk, Color stm) {
    // Distinct squares.
    if (wk == pc || wk == bk || pc == bk) return false;

    // Kings never adjacent.
    if (attacks::king(static_cast<Square>(wk)) & square_bb(static_cast<Square>(bk))) return false;

    // The side NOT to move must not be in check. If White is to move, Black just
    // moved, so the Black king must not be attacked by White. (If Black is to
    // move, the only White attacker of the White king would be adjacency, which
    // is already excluded — so no extra test is needed.)
    if (stm == Color::White) {
        const Bitboard occ = square_bb(static_cast<Square>(wk)) |
                             square_bb(static_cast<Square>(pc)) |
                             square_bb(static_cast<Square>(bk));
        if (white_piece_attacks(wp, pc, occ) & square_bb(static_cast<Square>(bk))) return false;
    }
    return true;
}

uint32_t Index::canonical_spatial(int wk, int pc, int bk) {
    uint32_t best = 0xFFFFFFFFu;
    for (int g = 0; g < 8; ++g) {
        const uint32_t code =
            (static_cast<uint32_t>(SYM[g][wk]) * 64u + static_cast<uint32_t>(SYM[g][pc])) * 64u +
            static_cast<uint32_t>(SYM[g][bk]);
        if (code < best) best = code;
    }
    return best;
}

Index::Index(PieceType white_piece) : white_piece_(white_piece) {
    init_sym();
    slider::init();  // legality checks for sliders need the magic tables

    // Combined code space: spatial (64^3) x side-to-move (2).
    constexpr int CODE_SPACE = 64 * 64 * 64 * 2;
    code_to_dense_.assign(CODE_SPACE, -1);

    // Enumerate canonical, legal representatives and assign dense indices.
    // Because legality is symmetry-invariant, testing the canonical rep decides
    // the whole orbit, so we only enroll orbit minima.
    for (int stm_bit = 0; stm_bit < 2; ++stm_bit) {
        const Color stm = (stm_bit == 0) ? Color::White : Color::Black;
        for (int wk = 0; wk < 64; ++wk) {
            for (int pc = 0; pc < 64; ++pc) {
                for (int bk = 0; bk < 64; ++bk) {
                    if (!legal(white_piece_, wk, pc, bk, stm)) continue;
                    const uint32_t raw = (static_cast<uint32_t>(wk) * 64u +
                                          static_cast<uint32_t>(pc)) * 64u +
                                         static_cast<uint32_t>(bk);
                    if (raw != canonical_spatial(wk, pc, bk)) continue;  // non-minimal
                    const uint32_t combined = raw * 2u + static_cast<uint32_t>(stm_bit);
                    code_to_dense_[combined] = static_cast<int32_t>(dense_to_code_.size());
                    dense_to_code_.push_back(combined);
                }
            }
        }
    }
}

Position Index::decode(std::size_t index) const {
    const uint32_t combined = dense_to_code_[index];
    const int  stm_bit = static_cast<int>(combined & 1u);
    uint32_t   spatial = combined >> 1;
    const int  bk = static_cast<int>(spatial % 64); spatial /= 64;
    const int  pc = static_cast<int>(spatial % 64); spatial /= 64;
    const int  wk = static_cast<int>(spatial);

    Position pos;
    pos.clear();
    const Bitboard wk_bb = square_bb(static_cast<Square>(wk));
    const Bitboard bk_bb = square_bb(static_cast<Square>(bk));
    const Bitboard pc_bb = square_bb(static_cast<Square>(pc));

    pos.by_type[type_index(PieceType::King)]  |= wk_bb | bk_bb;
    pos.by_type[type_index(white_piece_)]     |= pc_bb;
    pos.by_color[color_index(Color::White)]   |= wk_bb | pc_bb;
    pos.by_color[color_index(Color::Black)]   |= bk_bb;
    pos.side_to_move = (stm_bit == 0) ? Color::White : Color::Black;
    pos.zobrist = compute_zobrist(pos);
    return pos;
}

std::size_t Index::encode_raw(int wk, int pc, int bk, Color stm) const {
    const uint32_t canonical = canonical_spatial(wk, pc, bk);
    const uint32_t combined = canonical * 2u + (stm == Color::Black ? 1u : 0u);
    return static_cast<std::size_t>(code_to_dense_[combined]);
}

std::size_t Index::encode(const Position& pos) const {
    const int wk = sq_index(lsb(pos.by_type[type_index(PieceType::King)] &
                                pos.by_color[color_index(Color::White)]));
    const int bk = sq_index(lsb(pos.by_type[type_index(PieceType::King)] &
                                pos.by_color[color_index(Color::Black)]));
    const int pc = sq_index(lsb(pos.by_type[type_index(white_piece_)] &
                                pos.by_color[color_index(Color::White)]));
    return encode_raw(wk, pc, bk, pos.side_to_move);
}

}  // namespace tb
