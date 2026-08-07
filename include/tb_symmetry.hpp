// The 8 elements of D4 — the symmetry group of a pawnless board — as square
// permutations, built once at compile time.
//
// SINGLE SOURCE OF TRUTH. tb::Index folds whole man-tuples by these (min-code
// canonicalization) and tb::KingTable folds king pairs by them (king-anchored);
// previously each kept its own runtime-initialized copy. Sharing one constexpr
// table removes the duplication and the lazy-init flag, and it is the exact
// table the CUDA retrograde port will upload as a device constant.
//
// The tablebase TESTS deliberately keep their OWN independent copy of these
// formulas, so a bug in this table cannot hide behind itself in a self-check.

#pragma once

#include <array>
#include <cstdint>

#include "cuda_compat.hpp"  // CH_HD — transform_square is called from device code.

namespace tb {

// Square s -> its image under each transform. Stored uint8_t (squares are 0..63)
// so the whole table is 512 bytes — cache-friendly now, a tidy device constant
// later. Formulas: with sq(file,rank)=rank*8+file, the group is {identity, file
// mirror, rank mirror, 180° rotation, both diagonal reflections, both 90° rots}.
constexpr std::array<std::array<uint8_t, 64>, 8> make_d4_table() {
    std::array<std::array<uint8_t, 64>, 8> t{};
    for (int s = 0; s < 64; ++s) {
        const int f = s & 7, r = s >> 3;
        t[0][s] = static_cast<uint8_t>(r * 8 + f);
        t[1][s] = static_cast<uint8_t>(r * 8 + (7 - f));
        t[2][s] = static_cast<uint8_t>((7 - r) * 8 + f);
        t[3][s] = static_cast<uint8_t>((7 - r) * 8 + (7 - f));
        t[4][s] = static_cast<uint8_t>(f * 8 + r);
        t[5][s] = static_cast<uint8_t>((7 - f) * 8 + (7 - r));
        t[6][s] = static_cast<uint8_t>((7 - f) * 8 + r);
        t[7][s] = static_cast<uint8_t>(f * 8 + (7 - r));
    }
    return t;
}

inline constexpr auto D4 = make_d4_table();

// Apply D4 transform g in [0,8) to square s in [0,64).
CH_HD constexpr int transform_square(int g, int s) { return D4[g][s]; }

}  // namespace tb
