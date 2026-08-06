// Phase 2 — bijective position indexing for a 3-man pawnless endgame.
//
// Material: White King, one White piece (Rook/Queen/Bishop/Knight), Black King.
// The board's 8-element symmetry group (D4: the reflections/rotations of a
// square) is used to fold the index space ~8x. Both sides-to-move are distinct
// positions (distance-to-mate depends on who moves), so the index carries stm.
//
// Correctness approach (correctness-first, mirroring the magic-vs-reference
// discipline from Phase 1): the canonical representative of a position is
// DEFINED as the minimum raw code over all 8 symmetry transforms. That
// definition is obviously a bijection between dense indices and orbits, so the
// index is correct by construction; the test then verifies the whole mapping
// with a partition check that relies on no externally-recalled constants.
//
// A closed-form "king triangle" arithmetic index would use less memory than the
// lookup tables here, but the table-based canonicalization is the unimpeachable
// baseline and is exactly what the GPU can also use (index tables in device
// memory). Optimize to closed-form later only if the tables prove too large.
//
// std::vector is used for the (runtime-sized) tables: this is offline tablebase
// setup, not the search hot path, so heap here does not violate the engine's
// no-heap-below-search rule.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "position.hpp"
#include "types.hpp"

namespace tb {

class Index {
public:
    // white_piece must be Rook, Queen, Bishop, or Knight (not Pawn/King).
    explicit Index(PieceType white_piece);

    std::size_t size() const { return dense_to_code_.size(); }
    PieceType   white_piece() const { return white_piece_; }

    // Is this raw square assignment a legal position of the material class for
    // the given side to move? Legality is symmetry-invariant.
    static bool legal(PieceType wp, int wk, int pc, int bk, Color stm);

    // Dense index -> a concrete canonical Position (castling/ep cleared, key set).
    Position decode(std::size_t index) const;

    // Any legal position of this class -> its dense index.
    std::size_t encode(const Position& pos) const;

    // Raw-square form of encode (used internally and by tests).
    std::size_t encode_raw(int wk, int pc, int bk, Color stm) const;

private:
    // Minimum spatial code (wk,pc,bk packed) over the 8 symmetry transforms.
    static uint32_t canonical_spatial(int wk, int pc, int bk);

    PieceType             white_piece_;
    std::vector<uint32_t> dense_to_code_;  // index -> canonical combined code
    std::vector<int32_t>  code_to_dense_;  // combined code -> index, or -1
};

}  // namespace tb
