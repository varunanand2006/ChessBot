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
};

// A full material's descriptor: men count + the ordered groups.
struct DeviceMaterial {
    int         men;
    int         num_groups;
    DeviceGroup groups[kDevMaxGroup];  // #groups <= #extras <= men-2 <= 6
};

// index -> squares[0..men) (slot order [wk,bk,extra0,...]) + side-to-move bit.
// Line-for-line the mirror of CombIndex::decode.
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

    int empty[64];
    int ne = 0;
    for (int s = 0; s < 64; ++s)
        if (s != cwk && s != cbk) empty[ne++] = s;

    for (int gi = 0; gi < mat.num_groups; ++gi) {
        const DeviceGroup& grp = mat.groups[gi];
        int coords[kDevMaxGroup];
        combo::unrank_combination(r[gi], grp.count, coords);

        int gs[kDevMaxGroup];
        for (int j = 0; j < grp.count; ++j) gs[j] = empty[coords[j]];  // ascending

        for (int j = 0; j < grp.count; ++j) squares[2 + grp.slots[j]] = gs[j];

        int w = 0;
        for (int rd = 0; rd < ne; ++rd) {
            bool placed = false;
            for (int j = 0; j < grp.count; ++j)
                if (empty[rd] == gs[j]) { placed = true; break; }
            if (!placed) empty[w++] = empty[rd];
        }
        ne = w;
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

    int empty[64];
    int ne = 0;
    for (int s = 0; s < 64; ++s)
        if (s != cwk && s != cbk) empty[ne++] = s;

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
        for (int j = 0; j < grp.count; ++j) {
            int pos = 0;
            while (pos < ne && empty[pos] != gs[j]) ++pos;
            coords[j] = pos;
        }
        const uint64_t r = combo::rank_combination(coords, grp.count);
        index = index * grp.radix + r;

        int w = 0;
        for (int rd = 0; rd < ne; ++rd) {
            bool placed = false;
            for (int j = 0; j < grp.count; ++j)
                if (empty[rd] == gs[j]) { placed = true; break; }
            if (!placed) empty[w++] = empty[rd];
        }
        ne = w;
    }
    return index * 2u + static_cast<uint64_t>(stm_bit ? 1u : 0u);
}

}  // namespace tb
