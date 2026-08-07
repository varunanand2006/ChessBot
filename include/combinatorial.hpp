// Combinatorial number system — the arithmetic primitive under 5-man (and
// larger) tablebase indexing.
//
// WHY this exists: the current tb::Index reserves one table slot per raw
// square-tuple (a dense 64^men array). That is ~134 MB at 4 men and explodes at
// 5+ (64^5 = 1.07e9 slots). It also can't represent identical pieces (two white
// rooks) without double-counting the ordered pair. Both problems dissolve if we
// stop numbering ordered tuples and start numbering *combinations*: an unordered
// k-subset of the board maps bijectively onto [0, C(n,k)), with no wasted slots
// and no order to double-count.
//
// This header is the one primitive that buys both: rank/unrank of a k-subset via
// the combinatorial number system. Everything about placing groups of pieces on
// a shrinking set of empty squares composes out of it.
//
// STYLE: header-only + constexpr + pointer-based (no allocation, no std::vector).
// The binomial table is built at compile time. Deliberately kept free of any
// chess types so it (a) unit-tests as pure math and (b) ports verbatim to CUDA
// device code later — the retrograde kernels need this exact index arithmetic.

#pragma once

#include <array>
#include <cstdint>

#include "cuda_compat.hpp"  // CH_HD — makes the runtime-callable functions below
                            // compile for the CUDA device as well as the host.

namespace combo {

// Board has 64 squares, so no combination we index draws from more than 64
// elements. The full triangle up to n=64 fits in uint64_t: the largest entry is
// C(64,32) = 1.83e18 < 2^63, so signed/unsigned both hold it without overflow.
inline constexpr int kMaxN = 64;

// Pascal's triangle, filled at compile time (constexpr loop, C++20). Size is
// 65*65*8 = ~34 KB — trivial, and being a table it is trivially auditable
// against Pascal's rule in the test rather than trusted as a transcribed formula.
constexpr std::array<std::array<uint64_t, kMaxN + 1>, kMaxN + 1> make_binom() {
    std::array<std::array<uint64_t, kMaxN + 1>, kMaxN + 1> c{};
    for (int n = 0; n <= kMaxN; ++n) {
        c[n][0] = 1;
        for (int k = 1; k <= n; ++k)
            c[n][k] = c[n - 1][k - 1] + c[n - 1][k];  // c[n-1][k]==0 when k>n-1
    }
    return c;
}

inline constexpr auto kBinom = make_binom();

// C(n, k), with out-of-range arguments defined to 0 (k<0, k>n) — matches the
// convention the rank/unrank recurrences rely on at their boundaries.
CH_HD constexpr uint64_t binom(int n, int k) {
    if (n < 0 || k < 0 || k > n || n > kMaxN) return 0;
#ifdef __CUDA_ARCH__
    // Device: recompute instead of loading the host kBinom table. Indexing a
    // namespace-scope host array at runtime is an ODR-use nvcc rejects in device
    // code ("identifier combo::kBinom is undefined in device code"). Fold to the
    // smaller tail (C(n,k)=C(n,n-k), so k<=32), then run a rolling Pascal row of
    // pure ADDITIONS — exact with no intermediate overflow (the multiplicative
    // form would overflow: C(64,32)=1.83e18 times (n-i) blows past 2^64, whereas
    // the true value itself fits < 2^63). Cost is O(n*k) adds (<=~2K) vs the
    // host's O(1) load. The Phase 4 plan promotes kBinom to __constant__ so the
    // device also gets an O(1) cached load; correctness-first, this computed form
    // needs zero upload/bind plumbing and can't drift from the host recurrence
    // (same Pascal rule).
    if (k > n - k) k = n - k;
    uint64_t row[kMaxN / 2 + 2] = {};    // k <= 32 after the fold
    row[0] = 1;
    for (int i = 1; i <= n; ++i)
        for (int j = (i < k ? i : k); j > 0; --j)
            row[j] += row[j - 1];
    return row[k];
#else
    return kBinom[n][k];
#endif
}

// Rank a k-combination into [0, C(n, k)).
//
// `elems` must be a STRICTLY INCREASING array of k distinct values (the chosen
// squares, in ascending order). The map is the combinatorial number system:
//
//     rank = sum over i of C(elems[i], i + 1)
//
// i.e. the (i+1)-th smallest chosen element contributes C(value, i+1). This is a
// bijection between k-subsets of {0,1,2,...} and the non-negative integers; for
// subsets of {0..n-1} the image is exactly [0, C(n,k)). It does not depend on n
// itself — n only bounds the range — so the same code indexes any board size.
CH_HD constexpr uint64_t rank_combination(const int* elems, int k) {
    uint64_t r = 0;
    for (int i = 0; i < k; ++i) r += binom(elems[i], i + 1);
    return r;
}

// Inverse of rank_combination: given a rank r and the subset size k, reconstruct
// the strictly-increasing subset into out[0..k).
//
// Greedy from the largest element down: the (i+1)-th smallest element is the
// largest c with C(c, i+1) <= remaining rank (its minimum possible value is i,
// since i smaller elements sit below it). Subtract its contribution and recurse.
CH_HD constexpr void unrank_combination(uint64_t r, int k, int* out) {
    for (int i = k - 1; i >= 0; --i) {
        int c = i;
        // Grow to the largest fitting element. The `c + 1 < kMaxN` bound is a
        // hard safety cap: elements of a subset over a <=64-square universe are
        // always < kMaxN, so it never affects a valid unrank, but it guarantees
        // termination if `r` is ever out of range for `k`. Without it, binom()
        // returns 0 past n=kMaxN, so `0 <= r` would spin forever — on the GPU
        // that hangs the whole kernel (no watchdog to lean on), so the primitive
        // must be self-limiting rather than trusting every caller's arithmetic.
        while (c + 1 < kMaxN && binom(c + 1, i + 1) <= r) ++c;
        out[i] = c;
        r -= binom(c, i + 1);
    }
}

}  // namespace combo
