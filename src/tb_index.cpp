#include "tb_index.hpp"

#include <cstdio>
#include <cstdlib>

#include "attacks.hpp"
#include "bitboard.hpp"
#include "slider.hpp"
#include "tb_symmetry.hpp"  // shared D4 transforms (was a local runtime copy)

namespace tb {

namespace {

Bitboard piece_attacks(PieceType pt, int sq, Bitboard occ) {
    const Square s = static_cast<Square>(sq);
    switch (pt) {
        case PieceType::Rook:   return slider::rook_attacks(s, occ);
        case PieceType::Bishop: return slider::bishop_attacks(s, occ);
        case PieceType::Queen:  return slider::queen_attacks(s, occ);
        case PieceType::Knight: return attacks::knight(s);
        case PieceType::King:   return attacks::king(s);
        default:                return 0;  // pawns unsupported here
    }
}

}  // namespace

bool Index::legal(const std::vector<Piece>& extras, const int* squares, Color stm) {
    const int men = 2 + static_cast<int>(extras.size());

    // All squares distinct.
    for (int i = 0; i < men; ++i)
        for (int j = i + 1; j < men; ++j)
            if (squares[i] == squares[j]) return false;

    const int wk = squares[0], bk = squares[1];

    // Kings never adjacent.
    if (attacks::king(static_cast<Square>(wk)) & square_bb(static_cast<Square>(bk))) return false;

    // The side NOT to move must not be in check by any of the mover's pieces.
    // The mover's king can't be the checker — adjacent kings were excluded above
    // — so only the mover's extra pieces need testing.
    const int dksq = (~stm == Color::White) ? wk : bk;

    Bitboard occ = 0;
    for (int i = 0; i < men; ++i) occ |= square_bb(static_cast<Square>(squares[i]));

    for (std::size_t e = 0; e < extras.size(); ++e) {
        if (extras[e].color != stm) continue;
        if (piece_attacks(extras[e].type, squares[2 + e], occ) &
            square_bb(static_cast<Square>(dksq)))
            return false;
    }
    return true;
}

uint32_t Index::canonical_spatial(const int* squares) const {
    uint32_t best = 0xFFFFFFFFu;
    for (int g = 0; g < 8; ++g) {
        uint32_t code = 0;
        for (int i = 0; i < men_; ++i)
            code = code * 64u + static_cast<uint32_t>(transform_square(g, squares[i]));
        if (code < best) best = code;
    }
    return best;
}

Index::Index(std::vector<Piece> extras) : extras_(std::move(extras)) {
    men_ = 2 + static_cast<int>(extras_.size());
    if (men_ > 4) {
        // Direct 64^men table is too large beyond 4 men; needs arithmetic
        // combinatorial indexing (a later step).
        std::fprintf(stderr, "tb::Index: %d men unsupported (direct table too large)\n", men_);
        std::abort();
    }

    slider::init();  // legality checks for sliders need the magic tables

    uint64_t spatial_space = 1;
    for (int i = 0; i < men_; ++i) spatial_space *= 64;
    code_to_dense_.assign(static_cast<std::size_t>(spatial_space) * 2, -1);

    int sq[4];
    for (int stm_bit = 0; stm_bit < 2; ++stm_bit) {
        const Color stm = (stm_bit == 0) ? Color::White : Color::Black;
        for (uint64_t spatial = 0; spatial < spatial_space; ++spatial) {
            // Unpack base-64 digits: sq[0] is most significant.
            uint64_t t = spatial;
            for (int i = men_ - 1; i >= 0; --i) { sq[i] = static_cast<int>(t % 64); t /= 64; }

            if (!legal(extras_, sq, stm)) continue;
            if (static_cast<uint32_t>(spatial) != canonical_spatial(sq)) continue;  // non-minimal

            const uint32_t combined = static_cast<uint32_t>(spatial) * 2u +
                                      static_cast<uint32_t>(stm_bit);
            code_to_dense_[combined] = static_cast<int32_t>(dense_to_code_.size());
            dense_to_code_.push_back(combined);
        }
    }
}

Index::Index(PieceType white_piece) : Index(std::vector<Piece>{{Color::White, white_piece}}) {}

void Index::raw_squares(std::size_t index, int* out) const {
    uint32_t spatial = dense_to_code_[index] >> 1;
    for (int i = men_ - 1; i >= 0; --i) { out[i] = static_cast<int>(spatial % 64); spatial /= 64; }
}

Color Index::side_to_move(std::size_t index) const {
    return (dense_to_code_[index] & 1u) ? Color::Black : Color::White;
}

Position Index::make_position(const int* squares, Color stm) const {
    Position pos;
    pos.clear();
    const Bitboard wk_bb = square_bb(static_cast<Square>(squares[0]));
    const Bitboard bk_bb = square_bb(static_cast<Square>(squares[1]));
    pos.by_type[type_index(PieceType::King)] |= wk_bb | bk_bb;
    pos.by_color[color_index(Color::White)]  |= wk_bb;
    pos.by_color[color_index(Color::Black)]  |= bk_bb;

    for (std::size_t e = 0; e < extras_.size(); ++e) {
        const Bitboard b = square_bb(static_cast<Square>(squares[2 + e]));
        pos.by_type[type_index(extras_[e].type)]   |= b;
        pos.by_color[color_index(extras_[e].color)] |= b;
    }
    pos.side_to_move = stm;
    pos.zobrist = compute_zobrist(pos);
    return pos;
}

Position Index::decode(std::size_t index) const {
    int sq[4];
    raw_squares(index, sq);
    return make_position(sq, side_to_move(index));
}

std::size_t Index::encode_squares(const int* squares, Color stm) const {
    const uint32_t canonical = canonical_spatial(squares);
    const uint32_t combined = canonical * 2u + (stm == Color::Black ? 1u : 0u);
    return static_cast<std::size_t>(code_to_dense_[combined]);
}

std::size_t Index::encode(const Position& pos) const {
    int sq[4];
    sq[0] = sq_index(lsb(pos.by_type[type_index(PieceType::King)] & pos.by_color[color_index(Color::White)]));
    sq[1] = sq_index(lsb(pos.by_type[type_index(PieceType::King)] & pos.by_color[color_index(Color::Black)]));
    for (std::size_t e = 0; e < extras_.size(); ++e) {
        sq[2 + e] = sq_index(lsb(pos.by_type[type_index(extras_[e].type)] &
                                 pos.by_color[color_index(extras_[e].color)]));
    }
    return encode_squares(sq, pos.side_to_move);
}

}  // namespace tb
