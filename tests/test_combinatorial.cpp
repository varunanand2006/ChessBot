// Step 5.1 gate: the combinatorial number system primitive (combo::binom +
// rank/unrank of k-subsets) is a correct bijection. Pure math — zero chess.
//
// Checks (no recalled constants except a few textbook binomials as an anchor):
//   A. binom table obeys Pascal's rule, symmetry C(n,k)=C(n,n-k), and the
//      boundary conventions rank/unrank depend on.
//   B. round trip: unrank(rank(S)) == S for every k-subset, and
//      rank(unrank(r)) == r for every r in [0, C(n,k)).
//   C. bijection: the ranks of all C(n,k) subsets are exactly {0 .. C(n,k)-1}
//      (a permutation), verified by independent enumeration of every subset.
//   D. strictly-increasing + in-range: unrank output is a valid subset of
//      {0..n-1}.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "combinatorial.hpp"

namespace {

int g_failures = 0;
void fail(const char* m) { std::printf("[FAIL] %s\n", m); ++g_failures; }
void expect(bool ok, const char* m) { if (!ok) fail(m); }

// Enumerate every strictly-increasing k-subset of {0..n-1} and invoke f(subset).
template <class F>
void for_each_subset(int n, int k, F&& f) {
    std::vector<int> s(k);
    for (int i = 0; i < k; ++i) s[i] = i;  // first subset {0,1,...,k-1}
    if (k == 0) { f(s); return; }
    while (true) {
        f(s);
        // Advance to the next subset in lexicographic order (odometer from the
        // right): find the rightmost element that can still be incremented.
        int i = k - 1;
        while (i >= 0 && s[i] == n - k + i) --i;
        if (i < 0) break;
        ++s[i];
        for (int j = i + 1; j < k; ++j) s[j] = s[j - 1] + 1;
    }
}

// A. binom table structural checks.
void test_binom_table() {
    using combo::binom;
    // Boundary conventions the recurrences lean on.
    expect(binom(0, 0) == 1, "C(0,0)==1");
    expect(binom(5, 0) == 1, "C(n,0)==1");
    expect(binom(5, 5) == 1, "C(n,n)==1");
    expect(binom(5, 6) == 0, "C(n,k)==0 for k>n");
    expect(binom(5, -1) == 0, "C(n,k)==0 for k<0");

    // Pascal's rule + symmetry across the whole table.
    for (int n = 1; n <= combo::kMaxN; ++n)
        for (int k = 0; k <= n; ++k) {
            if (binom(n, k) != binom(n - 1, k - 1) + binom(n - 1, k))
                { fail("Pascal's rule violated"); return; }
            if (binom(n, k) != binom(n, n - k))
                { fail("C(n,k)!=C(n,n-k)"); return; }
        }

    // A few textbook anchors (transcribed, so an independent sanity check that
    // the recurrence produced the right magnitudes, not just self-consistency).
    expect(binom(64, 2) == 2016,      "C(64,2)==2016");
    expect(binom(64, 3) == 41664,     "C(64,3)==41664");
    expect(binom(52, 5) == 2598960,   "C(52,5)==2598960 (poker hands)");
    expect(binom(10, 5) == 252,       "C(10,5)==252");
}

// B/C/D. round trip + bijection + validity for one (n,k).
void test_pair(int n, int k) {
    const uint64_t total = combo::binom(n, k);
    int buf[64];  // reused across iterations; k here is small (<=10), <= board's 64.

    // C: collect the rank of every subset; must be a permutation of [0,total).
    std::vector<char> seen(static_cast<std::size_t>(total), 0);
    for_each_subset(n, k, [&](const std::vector<int>& s) {
        const uint64_t r = combo::rank_combination(s.data(), k);
        if (r >= total) { fail("rank out of range"); return; }
        if (seen[r]) { fail("two subsets share a rank (not injective)"); return; }
        seen[r] = 1;

        // B: unrank must recover the exact subset.
        combo::unrank_combination(r, k, buf);
        for (int i = 0; i < k; ++i)
            if (buf[i] != s[i]) { fail("unrank(rank(S)) != S"); return; }
    });
    for (uint64_t r = 0; r < total; ++r)
        if (!seen[r]) { fail("some rank in [0,C(n,k)) unhit (not surjective)"); return; }

    // B(other direction) + D: rank(unrank(r))==r and output is a valid subset.
    for (uint64_t r = 0; r < total; ++r) {
        combo::unrank_combination(r, k, buf);
        for (int i = 0; i < k; ++i) {
            if (buf[i] < 0 || buf[i] >= n) { fail("unrank element out of [0,n)"); return; }
            if (i > 0 && buf[i] <= buf[i - 1]) { fail("unrank not strictly increasing"); return; }
        }
        if (combo::rank_combination(buf, k) != r) { fail("rank(unrank(r)) != r"); return; }
    }
}

}  // namespace

int main() {
    test_binom_table();

    // Exhaustive over a grid of (n,k) small enough to enumerate every subset.
    // C(24,6)=134596 and C(20,10)=184756 keep the total subset count modest
    // while covering k from tiny to n/2 (the widest, worst-case row).
    struct { int n, k; } cases[] = {
        {1, 0}, {1, 1}, {5, 0}, {5, 1}, {5, 2}, {5, 3}, {5, 5},
        {8, 3}, {12, 4}, {16, 2}, {20, 10}, {24, 6}, {32, 3}, {64, 2},
    };
    for (auto c : cases) test_pair(c.n, c.k);

    // Safety cap: unrank of an OUT-OF-RANGE rank must TERMINATE (not spin on the
    // GPU) and stay within [0, kMaxN). If this test hangs, the bound in
    // unrank_combination regressed. We only assert boundedness; the output for a
    // bad rank is meaningfully garbage, but it must be safe garbage.
    for (int k = 1; k <= 6; ++k) {
        int out[8];
        combo::unrank_combination(~0ull, k, out);  // rank far past any C(n,k)
        for (int i = 0; i < k; ++i)
            if (out[i] < 0 || out[i] >= combo::kMaxN) { fail("unrank exceeded kMaxN on bad rank"); break; }
    }

    if (g_failures == 0) {
        std::printf("All combinatorial (rank/unrank) tests passed.\n");
        return 0;
    }
    std::printf("\n%d combinatorial test(s) failed.\n", g_failures);
    return 1;
}
