// King-anchored canonical king-pair table — step 2 of the 5-man combinatorial
// indexer.
//
// WHY a separate king table (vs. the whole-tuple min-code in tb::Index):
// tb::Index canonicalizes by taking the minimum raw code over all 8 D4
// transforms applied to the ENTIRE man-tuple. That is unimpeachably correct but
// requires materializing the full 64^men space to build — the very thing that
// makes 5-man impossible with a direct table. The combinatorial index instead
// *anchors on the kings*: it chooses the symmetry transform from the king pair
// alone, rotates the whole position into that frame, and then places the
// remaining pieces combinatorially on the empty squares. This table is that
// anchor: legal (wk,bk) -> dense canonical id + the transform that gets there.
//
// STABILIZER CAVEAT (documented, deliberate): when a king pair lies on a D4
// symmetry axis its stabilizer has size > 1, so several transforms map it to the
// same canonical pair. We pick ONE deterministically (the argmin transform).
// Consequence: for those on-axis king configs a piece placement and its mirror
// under the residual symmetry receive DIFFERENT combinatorial indices — a mild
// over-count (a few extra, redundant indices), never a COLLISION (two distinct
// positions sharing an index, which would corrupt the solver). Over-count is the
// safe direction; tightening the on-axis case is a known later refinement.
//
// The gate (test_tb_king_table) cross-checks this table against tb::Index built
// for bare kings: same canonical set, same count, same folding — so we know the
// king-anchored symmetry matches the already-verified whole-tuple version.

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace tb {

struct KingTable {
    KingTable();

    // Number of canonical king configurations (dense ids are [0, count())).
    int count() const { return num_canonical_; }

    // Legal raw pair -> dense canonical id; -1 if illegal (equal or adjacent).
    int id_of(int wk, int bk) const { return id_[static_cast<std::size_t>(wk) * 64 + bk]; }

    // The D4 transform g in [0,8) carrying the raw pair to its canonical rep
    // (argmin transform). Meaningful only for legal pairs.
    int transform_of(int wk, int bk) const {
        return transform_[static_cast<std::size_t>(wk) * 64 + bk];
    }

    // Canonical id -> its canonical (wk,bk).
    void kings_of(int id, int& wk, int& bk) const {
        const uint16_t p = canon_pair_[static_cast<std::size_t>(id)];
        wk = p >> 6;
        bk = p & 63;
    }

    // Raw contiguous arrays — exactly what the CUDA port uploads to the device
    // (id/transform become __constant__, canon_pair global). Exposed read-only
    // so the device-descriptor builder can point at them without copying.
    const int16_t*  id_data() const { return id_.data(); }
    const int8_t*   transform_data() const { return transform_.data(); }
    const uint16_t* canon_pair_data() const { return canon_pair_.data(); }

private:
    int                        num_canonical_ = 0;
    std::array<int16_t, 4096>  id_{};         // wk*64+bk -> canonical id (or -1)
    std::array<int8_t, 4096>   transform_{};  // wk*64+bk -> chosen transform
    std::vector<uint16_t>      canon_pair_;    // id -> canonical (wk*64+bk)
};

}  // namespace tb
