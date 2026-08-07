// Combinatorial position index — step 3 of the 5-man indexer. Composes the two
// primitives already built:
//   * combo::rank/unrank  (the combinatorial number system)   — combinatorial.hpp
//   * tb::KingTable       (king-anchored D4 canonicalization)  — tb_king_table.hpp
// into a full bijective encode/decode for a fixed pawnless material, WITHOUT
// ever materializing the 64^men space. This is what makes 5-man feasible and
// what lets identical pieces (two rooks) be indexed as an unordered subset.
//
// THE COMPOSITION (mixed-radix, most-significant digit first):
//
//   index = king_id
//   for each piece group G (in a fixed order):        // radix C(R_G, |G|)
//       index = index * C(R_G, |G|) + rank(G's squares among the empty squares)
//   index = index * 2 + side_to_move_bit              // radix 2, least significant
//
// where R_G = 62 - (pieces placed before G): after the two kings claim their
// squares, each group is placed on the shrinking set of still-empty squares.
// Because every group has a fixed size, every R_G (and thus every radix) is a
// COMPILE-TIME-CONSTANT per material — decode peels the digits with plain
// integer div/mod, no search.
//
// A group is a set of identical (color,type) pieces: distinct pieces are groups
// of size 1 (radix = remaining-square count); m identical pieces form one group
// of size m (radix = C(remaining, m)), which is exactly how duplicates stop
// being double-counted.
//
// SIZE vs. the dense tb::Index: this space is LOOSER — it folds kings 8-fold but
// includes placements that are illegal for other reasons (side-not-to-move in
// check) and, for on-axis king configs, a few symmetry-redundant duplicates (the
// KingTable stabilizer over-count). That is the deliberate "overcount, filter at
// solve time" trade the rest of the tablebase already makes; the index never
// MERGES two distinct positions, which is the only thing that would be a bug.
//
// std::vector is fine here (offline tablebase setup, not the search hot path).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "position.hpp"
#include "tb_comb_index_device.hpp"  // DeviceKingTable / DeviceMaterial (POD)
#include "tb_index.hpp"       // reuse tb::Piece
#include "tb_king_table.hpp"
#include "types.hpp"

namespace tb {

class CombIndex {
public:
    // General pawnless material: 2 kings + `extras` (each a colored piece type).
    // Identical (color,type) extras are allowed and handled as one group.
    explicit CombIndex(std::vector<Piece> extras);

    // Upper bound on men (2 kings + up to 6 extras). Fixes the size of the
    // fixed square buffers in encode/decode; the constructor rejects anything
    // larger so those buffers can never overflow.
    static constexpr int kMaxMen = 8;

    std::size_t size() const { return size_; }
    int         men() const { return men_; }
    const std::vector<Piece>& extras() const { return extras_; }

    // Raw-square interface, slot order [wk, bk, extra0, extra1, ...] matching
    // `extras`. encode requires a legal king pair (caller guarantees). For
    // identical pieces the slots within a group are interchangeable: encode is
    // invariant to swapping them, decode fills them in ascending-square order.
    std::size_t encode(const int* squares, Color stm) const;
    void        decode(std::size_t index, int* squares, Color* stm) const;

    // Position glue for the solver. make_position builds a Position from raw
    // squares; decode_pos = decode + make_position; encode(Position) extracts
    // the men squares (handling identical pieces) and encodes. Symmetric with
    // tb::Index's Position interface so the solver code reads the same.
    //
    // with_zobrist=false skips the compute_zobrist() over all pieces: the
    // memoryless sweep rebuilds a position per node per pass but only ever reads
    // its bitboards (encode works off bitboards, not the key), so hashing there
    // is pure waste. Callers that need a valid pos.zobrist keep the default.
    Position    make_position(const int* squares, Color stm, bool with_zobrist = true) const;
    Position    decode_pos(std::size_t index) const;
    std::size_t encode(const Position& pos) const;

    // Fill the device-side POD descriptors (see tb_comb_index_device.hpp) from
    // this material: `kt` points at the KingTable's raw arrays, `mat` copies the
    // per-group placement constants. The GPU port uploads these; the host
    // comb_encode/comb_decode over them are diffed against this class's
    // encode/decode as the correctness gate. kt aliases KingTable storage owned
    // by this CombIndex, so it stays valid only as long as this object does.
    void fill_device(DeviceKingTable& kt, DeviceMaterial& mat) const;

private:
    static constexpr int kMaxGroup = 6;  // max identical pieces in one group

    struct Group {
        Color     color;
        PieceType type;
        int       count;                    // |G|
        int       slots[kMaxGroup];         // extras indices in this group (asc)
        int       R;                        // empty squares available at placement
        uint64_t  radix;                    // C(R, count)
    };

    int                 men_;
    std::vector<Piece>  extras_;
    KingTable           kings_;
    std::vector<Group>  groups_;            // in fixed (color,type) order
    std::size_t         size_ = 0;
};

}  // namespace tb
