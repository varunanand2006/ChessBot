// Phase 2 — bijective position indexing for pawnless endgames (generalized).
//
// Material = two kings + a list of extra pieces, each tagged with a color and
// type (e.g. KQKR = extras {White Queen, Black Rook}; KRRK would be two White
// Rooks). The board's 8-element symmetry group (D4) folds the index space ~8x.
// Both sides-to-move are distinct positions.
//
// Correctness is by construction (as in the 3-man version): the canonical
// representative of a position is the minimum raw code over the 8 symmetry
// transforms, which is trivially a bijection between dense indices and orbits.
// The test verifies it with a partition check needing no recalled constants.
//
// Scope of THIS implementation:
//   * Pawnless only (full 8-fold symmetry). Pawns reduce symmetry and add
//     promotion/capture cross-table dependencies — a later step.
//   * Direct lookup tables sized 64^men. Practical through 4 men (~134 MB);
//     5+ men needs arithmetic (combinatorial) indexing instead — a later step.
//   * Extra pieces must have DISTINCT (color,type). Identical pieces (e.g. two
//     white rooks) need unordered/combinatorial handling — a later step.
//
// std::vector is fine here: offline tablebase setup, not the search hot path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "position.hpp"
#include "types.hpp"

namespace tb {

struct Piece {
    Color     color;
    PieceType type;
};

class Index {
public:
    // General pawnless material. `extras` are the non-king pieces; they must
    // have distinct (color,type) pairs, and 2 + extras.size() must be <= 4.
    explicit Index(std::vector<Piece> extras);

    // Convenience: 3-man KXK (a single White piece).
    explicit Index(PieceType white_piece);

    std::size_t size() const { return dense_to_code_.size(); }
    int         men() const { return men_; }
    const std::vector<Piece>& extras() const { return extras_; }
    PieceType   white_piece() const { return extras_[0].type; }  // valid for KXK

    // Dense index <-> position.
    Position    decode(std::size_t index) const;
    std::size_t encode(const Position& pos) const;

    // Raw-square interface (slot order: [wk, bk, extra0, extra1, ...]). Used by
    // tools and the bijection test. `squares` points to men() ints.
    static bool legal(const std::vector<Piece>& extras, const int* squares, Color stm);
    Position    make_position(const int* squares, Color stm) const;
    std::size_t encode_squares(const int* squares, Color stm) const;
    void        raw_squares(std::size_t index, int* out) const;  // fills men() ints
    Color       side_to_move(std::size_t index) const;

private:
    uint32_t canonical_spatial(const int* squares) const;

    int                   men_;
    std::vector<Piece>    extras_;
    std::vector<uint32_t> dense_to_code_;  // index -> canonical combined code
    std::vector<int32_t>  code_to_dense_;  // combined code -> index, or -1
};

}  // namespace tb
