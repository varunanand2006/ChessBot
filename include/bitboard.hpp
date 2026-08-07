// Bitboard iteration helpers. Thin wrappers over the C++20 <bit> intrinsics so
// movegen/search read clearly; all compile to single BMI/POPCNT instructions
// under -march=native.

#pragma once

#include <bit>

#include "types.hpp"

inline int popcount(Bitboard b) { return std::popcount(b); }

// Index of the least-significant set bit as a Square. UB if b == 0, so callers
// must guard (the while(bb) pop loops below never call it on empty).
inline Square lsb(Bitboard b) { return static_cast<Square>(std::countr_zero(b)); }

// Return the LSB square and clear it from b. Standard set-iteration primitive:
//   while (bb) { Square s = pop_lsb(bb); ... }
inline Square pop_lsb(Bitboard& b) {
    const Square s = lsb(b);
    b &= b - 1;  // clear lowest set bit
    return s;
}
