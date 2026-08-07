// Device-callable bitboard iteration helpers.
//
// bitboard.hpp uses std::popcount / std::countr_zero — great on the host, but
// those are not available in CUDA device code. These CH_HD wrappers select the
// GPU intrinsics (__popcll / __ffsll) when compiling for the device and the
// std:: versions otherwise, so the same movegen code iterates bitboards on both
// sides. Behaviour is identical to bitboard.hpp on the host.

#pragma once

#include <cstdint>

#include "cuda_compat.hpp"
#include "types.hpp"

#if !defined(__CUDA_ARCH__)
#include <bit>
#endif

CH_HD inline int popcount_dev(Bitboard b) {
#if defined(__CUDA_ARCH__)
    return __popcll(static_cast<unsigned long long>(b));
#else
    return std::popcount(b);
#endif
}

// Index of the least-significant set bit (0..63). UB if b == 0 — callers guard
// via the while(bb) pop loops, exactly as bitboard.hpp requires.
CH_HD inline int ctz_dev(Bitboard b) {
#if defined(__CUDA_ARCH__)
    // __ffsll returns a 1-based position (0 if b==0); subtract 1 for the index.
    return __ffsll(static_cast<long long>(b)) - 1;
#else
    return std::countr_zero(b);
#endif
}

CH_HD inline Square lsb_dev(Bitboard b) { return static_cast<Square>(ctz_dev(b)); }

CH_HD inline Square pop_lsb_dev(Bitboard& b) {
    const Square s = lsb_dev(b);
    b &= b - 1;  // clear lowest set bit
    return s;
}
