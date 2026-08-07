// Device-side combinatorial index — the port artifact for the GPU sweep.
//
// This is CombIndex::encode/decode (see src/tb_comb_index.cpp) rewritten over
// PLAIN POD structs instead of std::vector<Group> + the KingTable class, so the
// exact same index math runs inside a CUDA kernel. Deliberately LEAN: it pulls
// in only combinatorial.hpp + tb_symmetry.hpp (both CH_HD) and NO chess/position
// types, so a .cu including it compiles a small, device-clean translation unit.
//
// WHY it can be trusted before we ever touch a GPU: the functions are CH_HD, so
// they also compile for the host, where test_tb_comb_index_device diffs them
// bit-for-bit against the authoritative CombIndex over whole materials. The GPU
// then only has to reproduce the host result (cuda/comb_index_check.cu) — and
// the primitives underneath (binom/rank/unrank/D4) are already device-verified
// by cuda/index_check.cu. So "device == host" is checked in two independent
// places, and "host-mirror == real CombIndex" is checked exhaustively on CPU.
//
// The host fills these structs from a CombIndex via CombIndex::fill_device();
// the raw arrays are exactly what the kernel uploads to __constant__/global
// memory later (a Phase 4 change — for now they are ordinary device pointers).

#pragma once

#include <cstdint>

#include "combinatorial.hpp"
#include "cuda_compat.hpp"
#include "tb_symmetry.hpp"

namespace tb {

// Sizing mirrors CombIndex (2 kings + up to 6 extras; a group is up to 6
// identical pieces). Fixed so every buffer below is a small stack array.
inline constexpr int kDevMaxMen   = 8;
inline constexpr int kDevMaxGroup = 6;

// KingTable's three arrays as raw pointers (id_[4096], transform_[4096],
// canon_pair_[num_canonical]) — the device upload of the king-anchored fold.
struct DeviceKingTable {
    const int16_t*  id;             // wk*64+bk -> canonical id (or -1)
    const int8_t*   transform;      // wk*64+bk -> D4 transform g
    const uint16_t* canon_pair;     // id -> canonical (wk*64+bk)
    int             num_canonical;
};

// One piece group's placement constants (all compile-time per material).
struct DeviceGroup {
    int      count;                 // |G|
    int      R;                     // empties available at placement (unused by
                                    // encode/decode; kept for parity/debug)
    uint64_t radix;                 // C(R, count)
    int      slots[kDevMaxGroup];   // extras indices in this group (ascending)
    int      color;                 // Color of this group's pieces (0/1)
    int      type;                  // PieceType of this group's pieces (0..5)
                                    // color/type are unused by encode/decode; the
                                    // retrograde sweep uses them to rebuild a
                                    // Position and detect which piece a capture
                                    // removed (see tb_sweep_device.hpp).
};

// A full material's descriptor: men count + the ordered groups.
struct DeviceMaterial {
    int         men;
    int         num_groups;
    DeviceGroup groups[kDevMaxGroup];  // #groups <= #extras <= men-2 <= 6
};

// --- bit helpers for the free-square set (candidate #2: bitmask, no empty[64]) --
CH_HD inline int dev_popcount(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    int c = 0; while (x) { x &= x - 1; ++c; } return c;
#endif
}
// Index of the n-th set bit (0-based) == the n-th smallest free square.
CH_HD inline int dev_nth_set(uint64_t bb, int n) {
    for (int s = 0; s < 64; ++s)
        if ((bb >> s) & 1ull) { if (n == 0) return s; --n; }
    return 63;  // unreachable when bb has > n set bits (caller guarantees)
}

// index -> squares[0..men) (slot order [wk,bk,extra0,...]) + side-to-move bit.
// Mirror of CombIndex::decode (the empty-square list is a bitmask; see helpers).
CH_HD inline void comb_decode(uint64_t index, int* squares, int* stm_bit,
                              const DeviceKingTable& kt,
                              const DeviceMaterial& mat) {
    uint64_t x = index;
    const int sb = static_cast<int>(x & 1u);
    x >>= 1;

    // Peel group ranks last-group-first (reverse of encode's multiply).
    uint64_t r[kDevMaxMen];
    for (int gi = mat.num_groups; gi-- > 0;) {
        r[gi] = x % mat.groups[gi].radix;
        x /= mat.groups[gi].radix;
    }
    const int king_id = static_cast<int>(x);

    const uint16_t p = kt.canon_pair[king_id];
    const int cwk = p >> 6, cbk = p & 63;
    squares[0] = cwk;
    squares[1] = cbk;

    // Free-square set as a bitmask (bit s set => square s still available),
    // replacing the int empty[64] scratch list (candidate #2): -256 B/thread of
    // stack and no O(ne) compaction — a placed square is cleared with one AND.
    uint64_t free_sq = (~0ull) & ~(1ull << cwk) & ~(1ull << cbk);

    for (int gi = 0; gi < mat.num_groups; ++gi) {
        const DeviceGroup& grp = mat.groups[gi];
        int coords[kDevMaxGroup];
        combo::unrank_combination(r[gi], grp.count, coords);

        // Resolve every coord against the SAME pre-placement free set, THEN clear
        // the chosen squares (coords index the empty list before this group is
        // removed — matters for duplicate-piece groups).
        for (int j = 0; j < grp.count; ++j)
            squares[2 + grp.slots[j]] = dev_nth_set(free_sq, coords[j]);
        for (int j = 0; j < grp.count; ++j)
            free_sq &= ~(1ull << squares[2 + grp.slots[j]]);
    }
    *stm_bit = sb;
}

// squares[0..men) + side-to-move bit -> index. Mirror of CombIndex::encode.
// Requires a legal king pair (kt.id[...] >= 0); caller guarantees, exactly as
// the host encode does.
CH_HD inline uint64_t comb_encode(const int* squares, int stm_bit,
                                  const DeviceKingTable& kt,
                                  const DeviceMaterial& mat) {
    const int wk = squares[0], bk = squares[1];
    const int id = kt.id[wk * 64 + bk];
    const int g  = kt.transform[wk * 64 + bk];

    int t[kDevMaxMen];
    for (int i = 0; i < mat.men; ++i) t[i] = tb::transform_square(g, squares[i]);
    const int cwk = t[0], cbk = t[1];

    // Free-square bitmask instead of int empty[64] (candidate #2). A square's
    // coordinate among the free set is now a single popcount of the free bits
    // below it — O(1) vs the old O(ne) linear scan — and removal is one AND. This
    // is the hotter path: called per legal child via sweep_encode_position.
    uint64_t free_sq = (~0ull) & ~(1ull << cwk) & ~(1ull << cbk);

    uint64_t index = static_cast<uint64_t>(id);
    for (int gi = 0; gi < mat.num_groups; ++gi) {
        const DeviceGroup& grp = mat.groups[gi];

        int gs[kDevMaxGroup];
        for (int j = 0; j < grp.count; ++j) gs[j] = t[2 + grp.slots[j]];
        // Ascending insertion sort (tiny fixed group; identical pieces unordered).
        for (int i = 1; i < grp.count; ++i) {
            const int v = gs[i];
            int j = i - 1;
            while (j >= 0 && gs[j] > v) { gs[j + 1] = gs[j]; --j; }
            gs[j + 1] = v;
        }

        int coords[kDevMaxGroup];
        for (int j = 0; j < grp.count; ++j)               // rank among free squares
            coords[j] = dev_popcount(free_sq & ((1ull << gs[j]) - 1u));
        const uint64_t r = combo::rank_combination(coords, grp.count);
        index = index * grp.radix + r;

        for (int j = 0; j < grp.count; ++j) free_sq &= ~(1ull << gs[j]);
    }
    return index * 2u + static_cast<uint64_t>(stm_bit ? 1u : 0u);
}

}  // namespace tb
