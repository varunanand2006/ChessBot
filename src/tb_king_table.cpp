#include "tb_king_table.hpp"

#include "attacks.hpp"
#include "bitboard.hpp"
#include "tb_symmetry.hpp"  // shared D4 transforms (was a local runtime copy)
#include "types.hpp"

namespace tb {

KingTable::KingTable() {
    id_.fill(-1);
    transform_.fill(0);

    // First pass: identify canonical pairs (a legal pair is canonical iff it is
    // the min-code representative of its own orbit) and assign dense ids.
    for (int wk = 0; wk < 64; ++wk) {
        for (int bk = 0; bk < 64; ++bk) {
            if (wk == bk) continue;
            // Kings never adjacent.
            if (attacks::king(static_cast<Square>(wk)) & square_bb(static_cast<Square>(bk)))
                continue;

            // Canonical code + which transform achieves it (argmin, first-wins).
            uint32_t best = 0xFFFFFFFFu;
            for (int g = 0; g < 8; ++g) {
                const uint32_t code = static_cast<uint32_t>(transform_square(g, wk)) * 64u +
                                      static_cast<uint32_t>(transform_square(g, bk));
                if (code < best) best = code;
            }
            const int self = wk * 64 + bk;
            if (static_cast<uint32_t>(self) == best) {
                // This pair IS its own canonical rep -> a new canonical id.
                canon_pair_.push_back(static_cast<uint16_t>(self));
                id_[static_cast<std::size_t>(self)] = static_cast<int16_t>(num_canonical_);
                ++num_canonical_;
            }
        }
    }

    // Second pass: map every legal pair to its canonical id and record the
    // transform that carries it there. Done after ids exist so we can point
    // straight at the canonical rep's id.
    for (int wk = 0; wk < 64; ++wk) {
        for (int bk = 0; bk < 64; ++bk) {
            if (wk == bk) continue;
            if (attacks::king(static_cast<Square>(wk)) & square_bb(static_cast<Square>(bk)))
                continue;

            uint32_t best = 0xFFFFFFFFu;
            int best_g = 0;
            for (int g = 0; g < 8; ++g) {
                const uint32_t code = static_cast<uint32_t>(transform_square(g, wk)) * 64u +
                                      static_cast<uint32_t>(transform_square(g, bk));
                if (code < best) { best = code; best_g = g; }
            }
            const int self = wk * 64 + bk;
            id_[static_cast<std::size_t>(self)]        = id_[static_cast<std::size_t>(best)];
            transform_[static_cast<std::size_t>(self)] = static_cast<int8_t>(best_g);
        }
    }
}

}  // namespace tb
