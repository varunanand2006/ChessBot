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
// Closed-form image of square s under D4 transform g in [0,8). SINGLE SOURCE for
// both the constexpr table below and transform_square, so the host lookup table
// and the device inline path can never drift apart.
//
// Why inline arithmetic and not a shared array lookup: transform_square is CH_HD
// (called from device code). A runtime index into a namespace-scope host array
// (`D4[g][s]`) is an ODR-use nvcc cannot satisfy in device code — it fails with
// "identifier tb::D4 is undefined in device code". Recomputing the permutation is
// ~4 ALU ops per call; the alternative (making D4 a __constant__ and binding it on
// every device TU) costs a 512 B constant-memory table plus the upload/bind
// plumbing, and on the GPU a few ALU ops beat even a constant-cache load. So the
// device path computes; the host keeps the table for its tight canonicalization
// loops (tb::Index / KingTable) where the precomputed array is marginally cheaper.
CH_HD constexpr int d4_image(int g, int s) {
    const int f = s & 7, r = s >> 3;
    switch (g) {
        case 0:  return r * 8 + f;              // identity
        case 1:  return r * 8 + (7 - f);        // file mirror
        case 2:  return (7 - r) * 8 + f;        // rank mirror
        case 3:  return (7 - r) * 8 + (7 - f);  // 180° rotation
        case 4:  return f * 8 + r;              // main diagonal
        case 5:  return (7 - f) * 8 + (7 - r);  // anti-diagonal
        case 6:  return (7 - f) * 8 + r;        // 90°
        default: return f * 8 + (7 - r);        // 270° (g == 7)
    }
}

constexpr std::array<std::array<uint8_t, 64>, 8> make_d4_table() {
    std::array<std::array<uint8_t, 64>, 8> t{};
    for (int g = 0; g < 8; ++g)
        for (int s = 0; s < 64; ++s)
            t[g][s] = static_cast<uint8_t>(d4_image(g, s));
    return t;
}

inline constexpr auto D4 = make_d4_table();

// Apply D4 transform g in [0,8) to square s in [0,64). Device-safe: computes from
// d4_image rather than indexing the host-only D4 array (see the note above).
CH_HD constexpr int transform_square(int g, int s) { return d4_image(g, s); }

}  // namespace tb
