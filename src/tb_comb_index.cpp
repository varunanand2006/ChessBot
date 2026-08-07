#include "tb_comb_index.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "bitboard.hpp"
#include "combinatorial.hpp"
#include "tb_symmetry.hpp"

namespace tb {

namespace {

// Fixed group order = ascending (color, type). encode and decode both iterate
// groups in this order (decode peels radices in reverse), so the two stay in
// lockstep regardless of the order the caller listed `extras`.
bool group_less(Color ac, PieceType at, Color bc, PieceType bt) {
    if (color_index(ac) != color_index(bc)) return color_index(ac) < color_index(bc);
    return type_index(at) < type_index(bt);
}

// Ascending insertion sort for a tiny fixed array (a piece group is <=6). Beats
// std::sort here on both counts: no introsort machinery for 2-6 elements, and it
// sidesteps a GCC -Warray-bounds false positive from inlining sort over a small
// stack buffer whose length it can't prove <= the buffer capacity.
void sort_small(int* a, int n) {
    for (int i = 1; i < n; ++i) {
        const int v = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; --j; }
        a[j + 1] = v;
    }
}

}  // namespace

CombIndex::CombIndex(std::vector<Piece> extras) : extras_(std::move(extras)) {
    men_ = 2 + static_cast<int>(extras_.size());
    // The fixed square buffers in encode/decode are sized kMaxMen; reject any
    // material that would overflow them (the dense Index guards this the same
    // way). Supported materials are <= 5-man, well under the limit.
    if (men_ > kMaxMen) {
        std::fprintf(stderr, "CombIndex: %d men exceeds kMaxMen=%d\n", men_, kMaxMen);
        std::abort();
    }

    // Bucket extras into groups of identical (color,type). Iterate a stable
    // (color,type) order and collect matching extras indices so the group order
    // is deterministic and independent of the caller's listing order.
    std::vector<bool> used(extras_.size(), false);
    for (std::size_t i = 0; i < extras_.size(); ++i) {
        if (used[i]) continue;
        Group g{};
        g.color = extras_[i].color;
        g.type  = extras_[i].type;
        g.count = 0;
        for (std::size_t j = i; j < extras_.size(); ++j) {
            if (used[j]) continue;
            if (extras_[j].color == g.color && extras_[j].type == g.type) {
                if (g.count >= kMaxGroup) { std::fprintf(stderr, "CombIndex: group too large\n"); std::abort(); }
                g.slots[g.count++] = static_cast<int>(j);
                used[j] = true;
            }
        }
        groups_.push_back(g);
    }
    // Sort groups into the canonical (color,type) order.
    std::sort(groups_.begin(), groups_.end(), [](const Group& a, const Group& b) {
        return group_less(a.color, a.type, b.color, b.type);
    });

    // Assign each group its R (empty squares available) and radix C(R,count),
    // in placement order. Kings claim 2 squares, so the first group sees 62.
    int remaining = 62;
    uint64_t product = 1;
    for (Group& g : groups_) {
        g.R = remaining;
        g.radix = combo::binom(g.R, g.count);
        product *= g.radix;
        remaining -= g.count;
    }

    // size = (#canonical king configs) * prod(radix) * 2 (side to move).
    size_ = static_cast<std::size_t>(kings_.count()) * static_cast<std::size_t>(product) * 2u;
}

std::size_t CombIndex::encode(const int* squares, Color stm) const {
    const int wk = squares[0], bk = squares[1];
    const int id = kings_.id_of(wk, bk);
    const int g  = kings_.transform_of(wk, bk);

    // Rotate the whole position into the kings' canonical frame.
    int t[kMaxMen];
    for (int i = 0; i < men_; ++i) t[i] = transform_square(g, squares[i]);
    const int cwk = t[0], cbk = t[1];

    // Empty squares (ascending) = all but the two canonical king squares.
    int empty[64];
    int ne = 0;
    for (int s = 0; s < 64; ++s) if (s != cwk && s != cbk) empty[ne++] = s;

    uint64_t index = static_cast<uint64_t>(id);
    for (const Group& grp : groups_) {
        // Gather this group's transformed squares, ascending (identical pieces
        // are an unordered set, so the order the caller passed is irrelevant).
        int gs[kMaxGroup];
        for (int j = 0; j < grp.count; ++j) gs[j] = t[2 + grp.slots[j]];
        sort_small(gs, grp.count);

        // Compact each square to its position among the still-empty squares.
        // Bounded by `ne`: for a legal position gs[j] is always an empty (non-
        // king) square and is found before the bound, so this never triggers in
        // practice — but the bound turns a malformed input (a piece coincident
        // with a king, say) into a clamped miss instead of an out-of-bounds read.
        int coords[kMaxGroup];
        for (int j = 0; j < grp.count; ++j) {
            int pos = 0;
            while (pos < ne && empty[pos] != gs[j]) ++pos;   // ascending list; linear is fine (<=62)
            coords[j] = pos;
        }
        const uint64_t r = combo::rank_combination(coords, grp.count);
        index = index * grp.radix + r;

        // Remove the placed squares from the empty list (compact in place).
        int w = 0;
        for (int rd = 0; rd < ne; ++rd) {
            bool placed = false;
            for (int j = 0; j < grp.count; ++j) if (empty[rd] == gs[j]) { placed = true; break; }
            if (!placed) empty[w++] = empty[rd];
        }
        ne = w;
    }

    return static_cast<std::size_t>(index * 2u + (stm == Color::Black ? 1u : 0u));
}

void CombIndex::decode(std::size_t index, int* squares, Color* stm) const {
    uint64_t x = static_cast<uint64_t>(index);
    const int stm_bit = static_cast<int>(x & 1u);
    x >>= 1;

    // Peel group ranks from the LAST group to the first (reverse of the encode
    // multiply). Radices are fixed constants, so this is plain div/mod. A stack
    // buffer (not std::vector): decode runs once per node per pass in the sweep,
    // so this must not allocate. #groups <= #extras <= kMaxMen-2.
    uint64_t r[kMaxMen];
    for (std::size_t gi = groups_.size(); gi-- > 0;) {
        r[gi] = x % groups_[gi].radix;
        x /= groups_[gi].radix;
    }
    const int king_id = static_cast<int>(x);

    int cwk, cbk;
    kings_.kings_of(king_id, cwk, cbk);
    squares[0] = cwk;
    squares[1] = cbk;

    // Forward reconstruction, mirroring encode's shrinking empty set.
    int empty[64];
    int ne = 0;
    for (int s = 0; s < 64; ++s) if (s != cwk && s != cbk) empty[ne++] = s;

    for (std::size_t gi = 0; gi < groups_.size(); ++gi) {
        const Group& grp = groups_[gi];
        int coords[kMaxGroup];
        combo::unrank_combination(r[gi], grp.count, coords);

        int gs[kMaxGroup];
        for (int j = 0; j < grp.count; ++j) gs[j] = empty[coords[j]];  // ascending

        // Fill the group's slots in ascending-square order (slots are stored
        // ascending; identical pieces are interchangeable).
        for (int j = 0; j < grp.count; ++j) squares[2 + grp.slots[j]] = gs[j];

        // Remove placed squares from empty.
        int w = 0;
        for (int rd = 0; rd < ne; ++rd) {
            bool placed = false;
            for (int j = 0; j < grp.count; ++j) if (empty[rd] == gs[j]) { placed = true; break; }
            if (!placed) empty[w++] = empty[rd];
        }
        ne = w;
    }

    *stm = stm_bit ? Color::Black : Color::White;
}

Position CombIndex::make_position(const int* squares, Color stm, bool with_zobrist) const {
    Position pos;
    pos.clear();
    const Bitboard wk_bb = square_bb(static_cast<Square>(squares[0]));
    const Bitboard bk_bb = square_bb(static_cast<Square>(squares[1]));
    pos.by_type[type_index(PieceType::King)] |= wk_bb | bk_bb;
    pos.by_color[color_index(Color::White)]  |= wk_bb;
    pos.by_color[color_index(Color::Black)]  |= bk_bb;

    for (std::size_t e = 0; e < extras_.size(); ++e) {
        const Bitboard b = square_bb(static_cast<Square>(squares[2 + e]));
        pos.by_type[type_index(extras_[e].type)]    |= b;
        pos.by_color[color_index(extras_[e].color)] |= b;
    }
    pos.side_to_move = stm;
    if (with_zobrist) pos.zobrist = compute_zobrist(pos);  // else stays 0 (unused by the sweep)
    return pos;
}

Position CombIndex::decode_pos(std::size_t index) const {
    int sq[kMaxMen];
    Color stm;
    decode(index, sq, &stm);
    // Skip the zobrist: the sweep reads only bitboards from this position.
    return make_position(sq, stm, /*with_zobrist=*/false);
}

std::size_t CombIndex::encode(const Position& pos) const {
    int sq[kMaxMen];
    sq[0] = sq_index(lsb(pos.by_type[type_index(PieceType::King)] & pos.by_color[color_index(Color::White)]));
    sq[1] = sq_index(lsb(pos.by_type[type_index(PieceType::King)] & pos.by_color[color_index(Color::Black)]));

    // Extract each extra's square. Identical pieces share a bitboard, so peel
    // off squares already assigned to earlier extras of the same (color,type).
    for (std::size_t e = 0; e < extras_.size(); ++e) {
        Bitboard bb = pos.by_type[type_index(extras_[e].type)] &
                      pos.by_color[color_index(extras_[e].color)];
        for (std::size_t k = 0; k < e; ++k)
            if (extras_[k].color == extras_[e].color && extras_[k].type == extras_[e].type)
                bb &= ~square_bb(static_cast<Square>(sq[2 + k]));
        sq[2 + e] = sq_index(lsb(bb));
    }
    return encode(sq, pos.side_to_move);
}

}  // namespace tb
